//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Support/FVInt.h"
#include <algorithm>
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTMOORETOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

using comb::ICmpPredicate;
using llvm::SmallDenseSet;

namespace {

static std::optional<FVInt>
tryEvaluateMooreIntValue(Value value,
                         DenseMap<Value, std::optional<FVInt>> &cache,
                         SmallDenseSet<Value, 8> &visiting);

static std::string formatFourValuedConstant(const FVInt &value,
                                            moore::IntFormat format) {
  auto bitWidth = value.getBitWidth();
  if (bitWidth == 0) {
    if (format == moore::IntFormat::Decimal)
      return "0";
    return "";
  }

  auto padLeft = [](StringRef text, unsigned targetWidth, char padChar) {
    if (text.size() >= targetWidth)
      return text.str();
    std::string out;
    out.reserve(targetWidth);
    out.append(targetWidth - text.size(), padChar);
    out.append(text.begin(), text.end());
    return out;
  };

  switch (format) {
  case moore::IntFormat::Decimal: {
    unsigned padWidth = sim::FormatDecOp::getDecimalWidth(bitWidth, false);
    if (value.hasUnknown()) {
      char digit = value.isAllZ() ? 'z' : 'x';
      return padLeft(StringRef(&digit, 1), padWidth, ' ');
    }

    auto apint = value.toAPInt(false);
    SmallString<16> strBuf;
    apint.toString(strBuf, /*Radix=*/10, /*Signed=*/false);
    return padLeft(strBuf, padWidth, ' ');
  }

  case moore::IntFormat::Binary: {
    std::string out;
    out.reserve(bitWidth);
    for (unsigned i = 0; i < bitWidth; ++i) {
      auto bit = value.getBit(bitWidth - 1 - i);
      switch (bit) {
      case FVInt::V0:
        out.push_back('0');
        break;
      case FVInt::V1:
        out.push_back('1');
        break;
      case FVInt::X:
        out.push_back('x');
        break;
      case FVInt::Z:
        out.push_back('z');
        break;
      }
    }
    return out;
  }

  case moore::IntFormat::Octal: {
    std::string out;
    unsigned digits = (bitWidth + 2) / 3;
    out.reserve(digits);
    for (unsigned d = 0; d < digits; ++d) {
      bool hasX = false;
      bool hasZ = false;
      uint8_t val = 0;
      for (unsigned b = 0; b < 3; ++b) {
        unsigned idx = digits * 3 - 1 - (d * 3 + b);
        if (idx >= bitWidth)
          continue;
        auto bit = value.getBit(idx);
        switch (bit) {
        case FVInt::V0:
          break;
        case FVInt::V1:
          val |= (1u << (2 - b));
          break;
        case FVInt::X:
          hasX = true;
          break;
        case FVInt::Z:
          hasZ = true;
          break;
        }
      }
      if (hasX)
        out.push_back('x');
      else if (hasZ)
        out.push_back('z');
      else
        out.push_back(static_cast<char>('0' + val));
    }
    return out;
  }

  case moore::IntFormat::HexLower:
  case moore::IntFormat::HexUpper: {
    std::string out;
    unsigned digits = (bitWidth + 3) / 4;
    out.reserve(digits);
    for (unsigned d = 0; d < digits; ++d) {
      bool hasX = false;
      bool hasZ = false;
      uint8_t val = 0;
      for (unsigned b = 0; b < 4; ++b) {
        unsigned idx = digits * 4 - 1 - (d * 4 + b);
        if (idx >= bitWidth)
          continue;
        auto bit = value.getBit(idx);
        switch (bit) {
        case FVInt::V0:
          break;
        case FVInt::V1:
          val |= (1u << (3 - b));
          break;
        case FVInt::X:
          hasX = true;
          break;
        case FVInt::Z:
          hasZ = true;
          break;
        }
      }
      if (hasX) {
        out.push_back(format == moore::IntFormat::HexUpper ? 'X' : 'x');
        continue;
      }
      if (hasZ) {
        out.push_back(format == moore::IntFormat::HexUpper ? 'Z' : 'z');
        continue;
      }

      if (val < 10) {
        out.push_back(static_cast<char>('0' + val));
        continue;
      }
      out.push_back(static_cast<char>(
          (format == moore::IntFormat::HexUpper ? 'A' : 'a') + (val - 10)));
    }
    return out;
  }
  }
  llvm_unreachable("unknown integer format");
}

static std::optional<FVInt>
tryEvaluateMooreRefValue(Value ref,
                         DenseMap<Value, std::optional<FVInt>> &cache,
                         SmallDenseSet<Value, 8> &visiting) {
  if (!ref)
    return std::nullopt;

  // Support constant nets driven solely by constant continuous assignments.
  if (auto net = ref.getDefiningOp<NetOp>()) {
    SmallVector<FVInt, 2> candidates;

    auto addCandidate = [&](Value v) -> LogicalResult {
      if (!v)
        return failure();
      auto fv = tryEvaluateMooreIntValue(v, cache, visiting);
      if (!fv)
        return failure();
      candidates.push_back(*fv);
      return success();
    };

    if (auto assignment = net.getAssignment())
      if (failed(addCandidate(assignment)))
        return std::nullopt;

    for (auto &use : ref.getUses()) {
      auto *owner = use.getOwner();
      if (auto assign = dyn_cast<ContinuousAssignOp>(owner)) {
        if (assign.getDst() != ref)
          return std::nullopt;
        if (failed(addCandidate(assign.getSrc())))
          return std::nullopt;
        continue;
      }
      if (isa<ReadOp, OutputOp>(owner))
        continue;
      return std::nullopt;
    }

    if (candidates.empty())
      return std::nullopt;

    for (auto &candidate : candidates)
      if (!(candidate == candidates.front()))
        return std::nullopt;
    return candidates.front();
  }

  return std::nullopt;
}

static std::optional<FVInt>
tryEvaluateMooreIntValue(Value value,
                         DenseMap<Value, std::optional<FVInt>> &cache,
                         SmallDenseSet<Value, 8> &visiting) {
  if (!value)
    return std::nullopt;

  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;

  if (visiting.contains(value))
    return std::nullopt;
  visiting.insert(value);
  auto guard = llvm::make_scope_exit([&] { visiting.erase(value); });

  std::optional<FVInt> result;

  if (auto constant = value.getDefiningOp<ConstantOp>()) {
    result = constant.getValue();
  } else if (auto read = value.getDefiningOp<ReadOp>()) {
    result = tryEvaluateMooreRefValue(read.getInput(), cache, visiting);
  } else if (auto eq = value.getDefiningOp<EqOp>()) {
    auto lhs = tryEvaluateMooreIntValue(eq.getLhs(), cache, visiting);
    auto rhs = tryEvaluateMooreIntValue(eq.getRhs(), cache, visiting);
    if (lhs && rhs) {
      if (lhs->getBitWidth() != rhs->getBitWidth())
        return std::nullopt;

      auto unknownBits = lhs->getUnknownBits() | rhs->getUnknownBits();
      auto knownMask = ~(lhs->getUnknownBits() | rhs->getUnknownBits());
      auto mismatchBits = (lhs->getRawValue() ^ rhs->getRawValue()) & knownMask;
      if (!mismatchBits.isZero())
        result = FVInt(/*numBits=*/1, 0);
      else if (!unknownBits.isZero())
        result = FVInt::getAllX(/*numBits=*/1);
      else
        result = FVInt(/*numBits=*/1, 1);
    }
  } else if (auto ne = value.getDefiningOp<NeOp>()) {
    auto lhs = tryEvaluateMooreIntValue(ne.getLhs(), cache, visiting);
    auto rhs = tryEvaluateMooreIntValue(ne.getRhs(), cache, visiting);
    if (lhs && rhs) {
      if (lhs->getBitWidth() != rhs->getBitWidth())
        return std::nullopt;

      auto unknownBits = lhs->getUnknownBits() | rhs->getUnknownBits();
      auto knownMask = ~(lhs->getUnknownBits() | rhs->getUnknownBits());
      auto mismatchBits = (lhs->getRawValue() ^ rhs->getRawValue()) & knownMask;
      if (!mismatchBits.isZero())
        result = FVInt(/*numBits=*/1, 1);
      else if (!unknownBits.isZero())
        result = FVInt::getAllX(/*numBits=*/1);
      else
        result = FVInt(/*numBits=*/1, 0);
    }
  } else if (auto caseEq = value.getDefiningOp<CaseEqOp>()) {
    auto lhs = tryEvaluateMooreIntValue(caseEq.getLhs(), cache, visiting);
    auto rhs = tryEvaluateMooreIntValue(caseEq.getRhs(), cache, visiting);
    if (lhs && rhs)
      result = FVInt(/*numBits=*/1, (*lhs == *rhs) ? 1 : 0);
  } else if (auto caseNe = value.getDefiningOp<CaseNeOp>()) {
    auto lhs = tryEvaluateMooreIntValue(caseNe.getLhs(), cache, visiting);
    auto rhs = tryEvaluateMooreIntValue(caseNe.getRhs(), cache, visiting);
    if (lhs && rhs)
      result = FVInt(/*numBits=*/1, (*lhs != *rhs) ? 1 : 0);
  } else if (auto intToLogic = value.getDefiningOp<IntToLogicOp>()) {
    auto input = tryEvaluateMooreIntValue(intToLogic.getInput(), cache, visiting);
    if (input)
      result = FVInt(input->toAPInt(false));
  } else if (auto logicToInt = value.getDefiningOp<LogicToIntOp>()) {
    auto input = tryEvaluateMooreIntValue(logicToInt.getInput(), cache, visiting);
    if (input)
      result = FVInt(input->toAPInt(false));
  }

  cache.try_emplace(value, result);
  return result;
}

/// Returns the passed value if the integer width is already correct.
/// Zero-extends if it is too narrow.
/// Truncates if the integer is too wide and the truncated part is zero, if it
/// is not zero it returns the max value integer of target-width.
static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                uint32_t targetWidth, Location loc) {
  uint32_t intWidth = value.getType().getIntOrFloatBitWidth();
  if (intWidth == targetWidth)
    return value;

  if (intWidth < targetWidth) {
    Value zeroExt = hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return comb::ConcatOp::create(builder, loc, ValueRange{zeroExt, value});
  }

  Value hi = comb::ExtractOp::create(builder, loc, value, targetWidth,
                                     intWidth - targetWidth);
  Value zero = hw::ConstantOp::create(
      builder, loc, builder.getIntegerType(intWidth - targetWidth), 0);
  Value isZero = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, hi,
                                      zero, false);
  Value lo = comb::ExtractOp::create(builder, loc, value, 0, targetWidth);
  Value max = hw::ConstantOp::create(builder, loc,
                                     builder.getIntegerType(targetWidth), -1);
  return comb::MuxOp::create(builder, loc, isZero, lo, max, false);
}

/// Populate `portInfos` with the HW-oriented port description of `op`.
static void collectModulePortInfo(const TypeConverter &typeConverter,
                                  SVModuleOp op,
                                  SmallVectorImpl<hw::PortInfo> &portInfos) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  portInfos.clear();
  portInfos.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);
    if (!portTy)
      portTy = port.type;
    if (auto ioTy = dyn_cast_or_null<hw::InOutType>(portTy)) {
      portInfos.push_back(hw::PortInfo(
          {{port.name, ioTy.getElementType(), hw::ModulePort::InOut},
           inputNum++,
           {}}));
      continue;
    }

    if (port.dir == hw::ModulePort::Direction::Output) {
      portInfos.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      // FIXME: Once we support net<...>, ref<...> type to represent type of
      // special port like inout or ref port which is not a input or output
      // port. It can change to generate corresponding types for direction of
      // port or do specified operation to it. Now inout and ref port is treated
      // as input port.
      portInfos.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }
}

static hw::HWModuleOp createHWModuleFromPorts(OpBuilder &builder, Location loc,
                                              StringAttr name,
                                              ArrayRef<hw::PortInfo> ports) {
  OperationState state(loc, hw::HWModuleOp::getOperationName());
  state.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<hw::ModulePort> modulePorts;
  modulePorts.reserve(ports.size());
  SmallVector<Attribute> perPortAttrs;
  perPortAttrs.reserve(ports.size());
  bool anyPortAttrs = false;
  for (const auto &port : ports) {
    modulePorts.push_back({port.name, port.type, port.dir});
    DictionaryAttr dict = port.attrs;
    if (!dict)
      dict = builder.getDictionaryAttr({});
    else if (!dict.empty())
      anyPortAttrs = true;
    perPortAttrs.push_back(dict);
  }

  auto moduleType =
      hw::ModuleType::get(builder.getContext(), modulePorts);
  state.addAttribute(
      hw::HWModuleOp::getModuleTypeAttrName(state.name),
      TypeAttr::get(moduleType));

  auto perPortAttrName =
      hw::HWModuleOp::getPerPortAttrsAttrName(state.name);
  Attribute perPortAttrValue =
      anyPortAttrs ? builder.getArrayAttr(perPortAttrs)
                   : builder.getArrayAttr({});
  state.addAttribute(perPortAttrName, perPortAttrValue);

  state.addAttribute("parameters", builder.getArrayAttr({}));
  state.addAttribute("comment", builder.getStringAttr(""));

  auto unknownLocAttr = cast<LocationAttr>(builder.getUnknownLoc());
  SmallVector<Attribute> resultLocs;
  resultLocs.reserve(moduleType.getNumOutputs());
  for (const auto &port : ports) {
    if (!port.isOutput())
      continue;
    resultLocs.push_back(port.loc ? Attribute(port.loc) : unknownLocAttr);
  }
  state.addAttribute(hw::HWModuleOp::getResultLocsAttrName(state.name),
                     builder.getArrayAttr(resultLocs));

  state.addRegion();
  auto *created = builder.create(state);
  return cast<hw::HWModuleOp>(created);
}

//===----------------------------------------------------------------------===//
// Structural Conversion
//===----------------------------------------------------------------------===//

struct SVModuleOpConversion : public OpConversionPattern<SVModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SVModuleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);

    // Create the hw.module to replace moore.module
    SmallVector<hw::PortInfo> portInfos;
    collectModulePortInfo(*typeConverter, op, portInfos);
    auto hwModuleOp = createHWModuleFromPorts(
        rewriter, op.getLoc(), op.getSymNameAttr(), portInfos);
    // Make hw.module have the same visibility as the moore.module.
    // The entry/top level module is public, otherwise is private.
    SymbolTable::setSymbolVisibility(hwModuleOp,
                                     SymbolTable::getSymbolVisibility(op));
    if (failed(
            rewriter.convertRegionTypes(&op.getBodyRegion(), *typeConverter)))
      return failure();
    rewriter.inlineRegionBefore(op.getBodyRegion(), hwModuleOp.getBodyRegion(),
                                hwModuleOp.getBodyRegion().end());

    // Erase the original op
    rewriter.eraseOp(op);
    return success();
  }
};

struct OutputOpConversion : public OpConversionPattern<OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

struct InstanceOpConversion : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = op.getInstanceNameAttr();
    auto moduleName = op.getModuleNameAttr();

    // Create the new hw instanceOp to replace the original one.
    rewriter.setInsertionPoint(op);
    auto instOp = hw::InstanceOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), instName, moduleName,
        op.getInputs(), op.getInputNamesAttr(), op.getOutputNamesAttr(),
        /*Parameter*/ rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr,
        /*doNotPrint*/ nullptr);

    // Replace uses chain and erase the original op.
    op.replaceAllUsesWith(instOp.getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

static void getValuesToObserve(Region *region,
                               function_ref<void(Value)> setInsertionPoint,
                               const TypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value> &observeValues) {
  SmallDenseSet<Value> alreadyObserved;
  Location loc = region->getLoc();

  auto probeIfSignal = [&](Value value) -> Value {
    if (!isa<hw::InOutType>(value.getType()))
      return value;
    return llhd::PrbOp::create(rewriter, loc, value);
  };

  region->getParentOp()->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](Operation *operation) {
        for (auto value : operation->getOperands()) {
          if (isa<BlockArgument>(value))
            value = rewriter.getRemappedValue(value);

          if (region->isAncestor(value.getParentRegion()))
            continue;
          if (auto *defOp = value.getDefiningOp();
              defOp && defOp->hasTrait<OpTrait::ConstantLike>())
            continue;
          if (!alreadyObserved.insert(value).second)
            continue;

          OpBuilder::InsertionGuard g(rewriter);
          if (auto remapped = rewriter.getRemappedValue(value)) {
            if (!hw::isHWValueType(remapped.getType()))
              continue;
            setInsertionPoint(remapped);
            observeValues.push_back(probeIfSignal(remapped));
            continue;
          }

          auto type = typeConverter->convertType(value.getType());
          if (!type || !hw::isHWValueType(type))
            continue;

          setInsertionPoint(value);
          auto converted = typeConverter->materializeTargetConversion(
              rewriter, loc, type, value);
          observeValues.push_back(probeIfSignal(converted));
        }
      });
}

struct ProcedureOpConversion : public OpConversionPattern<ProcedureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProcedureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Collect values to observe before we do any modifications to the region.
    SmallVector<Value> observedValues;
    if (op.getKind() == ProcedureKind::AlwaysComb ||
        op.getKind() == ProcedureKind::AlwaysLatch) {
      auto setInsertionPoint = [&](Value value) {
        rewriter.setInsertionPoint(op);
      };
      getValuesToObserve(&op.getBody(), setInsertionPoint, typeConverter,
                         rewriter, observedValues);
    }

    auto loc = op.getLoc();
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter)))
      return failure();

    // Handle initial and final procedures. These lower to a corresponding
    // `llhd.process` or `llhd.final` op that executes the body and then halts.
    if (op.getKind() == ProcedureKind::Initial ||
        op.getKind() == ProcedureKind::Final) {
      Operation *newOp;
      if (op.getKind() == ProcedureKind::Initial)
        newOp = llhd::ProcessOp::create(rewriter, loc, TypeRange{});
      else
        newOp = llhd::FinalOp::create(rewriter, loc);
      auto &body = newOp->getRegion(0);
      rewriter.inlineRegionBefore(op.getBody(), body, body.end());
      for (auto returnOp :
           llvm::make_early_inc_range(body.getOps<ReturnOp>())) {
        rewriter.setInsertionPoint(returnOp);
        rewriter.replaceOpWithNewOp<llhd::HaltOp>(returnOp, ValueRange{});
      }
      rewriter.eraseOp(op);
      return success();
    }

    // All other procedures lower to a an `llhd.process`.
    auto newOp = llhd::ProcessOp::create(rewriter, loc, TypeRange{});

    // We need to add an empty entry block because it is not allowed in MLIR to
    // branch back to the entry block. Instead we put the logic in the second
    // block and branch to that.
    rewriter.createBlock(&newOp.getBody());
    auto *block = &op.getBody().front();
    cf::BranchOp::create(rewriter, loc, block);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());

    // Add special handling for `always_comb` and `always_latch` procedures.
    // These run once at simulation startup and then implicitly wait for any of
    // the values they access to change before running again. To implement this,
    // we create another basic block that contains the implicit wait, and make
    // all `moore.return` ops branch to that wait block instead of immediately
    // jumping back up to the body.
    if (op.getKind() == ProcedureKind::AlwaysComb ||
        op.getKind() == ProcedureKind::AlwaysLatch) {
      Block *waitBlock = rewriter.createBlock(&newOp.getBody());
      llhd::WaitOp::create(rewriter, loc, ValueRange{}, Value(), observedValues,
                           ValueRange{}, block);
      block = waitBlock;
    }

    // Make all `moore.return` ops branch back up to the beginning of the
    // process, or the wait block created above for `always_comb` and
    // `always_latch` procedures.
    for (auto returnOp : llvm::make_early_inc_range(newOp.getOps<ReturnOp>())) {
      rewriter.setInsertionPoint(returnOp);
      cf::BranchOp::create(rewriter, loc, block);
      rewriter.eraseOp(returnOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct WaitEventOpConversion : public OpConversionPattern<WaitEventOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitEventOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In order to convert the `wait_event` op we need to create three separate
    // blocks at the location of the op:
    //
    // - A "wait" block that reads the current state of any values used to
    //   detect events and then waits until any of those values change. When a
    //   change occurs, control transfers to the "check" block.
    // - A "check" block which is executed after any interesting signal has
    //   changed. This is where any `detect_event` ops read the current state of
    //   interesting values and compare them against their state before the wait
    //   in order to detect an event. If any events were detected, control
    //   transfers to the "resume" block; otherwise control goes back to the
    //   "wait" block.
    // - A "resume" block which holds any ops after the `wait_event` op. This is
    //   where control is expected to resume after an event has happened.
    //
    // Block structure before:
    //     opA
    //     moore.wait_event { ... }
    //     opB
    //
    // Block structure after:
    //     opA
    //     cf.br ^wait
    // ^wait:
    //     <read "before" values>
    //     llhd.wait ^check, ...
    // ^check:
    //     <read "after" values>
    //     <detect edges>
    //     cf.cond_br %event, ^resume, ^wait
    // ^resume:
    //     opB
    auto *resumeBlock =
        rewriter.splitBlock(op->getBlock(), ++Block::iterator(op));

    // If the 'wait_event' op is empty, we can lower it to a 'llhd.wait' op
    // without any observed values, but since the process will never wake up
    // from suspension anyway, we can also just terminate it using the
    // 'llhd.halt' op.
    if (op.getBody().front().empty()) {
      // Let the cleanup iteration after the dialect conversion clean up all
      // remaining unreachable blocks.
      rewriter.replaceOpWithNewOp<llhd::HaltOp>(op, ValueRange{});
      return success();
    }

    auto *waitBlock = rewriter.createBlock(resumeBlock);
    auto *checkBlock = rewriter.createBlock(resumeBlock);

    auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, loc, waitBlock);

    // We need to inline two copies of the `wait_event`'s body region: one is
    // used to determine the values going into `detect_event` ops before the
    // `llhd.wait`, and one will do the actual event detection after the
    // `llhd.wait`.
    //
    // Create a copy of the entire `wait_event` op in the wait block, which also
    // creates a copy of its region. Take note of all inputs to `detect_event`
    // ops and delete the `detect_event` ops in this copy.
    SmallVector<Value> valuesBefore;
    rewriter.setInsertionPointToEnd(waitBlock);
    auto clonedOp = cast<WaitEventOp>(rewriter.clone(*op));
    bool allDetectsAreAnyChange = true;
    for (auto detectOp :
         llvm::make_early_inc_range(clonedOp.getOps<DetectEventOp>())) {
      if (detectOp.getEdge() != Edge::AnyChange || detectOp.getCondition())
        allDetectsAreAnyChange = false;
      valuesBefore.push_back(detectOp.getInput());
      rewriter.eraseOp(detectOp);
    }

    // Determine the values used during event detection that are defined outside
    // the `wait_event`'s body region. We want to wait for a change on these
    // signals before we check if any interesting event happened.
    SmallVector<Value> observeValues;
    auto setInsertionPointAfterDef = [&](Value value) {
      if (auto *op = value.getDefiningOp())
        rewriter.setInsertionPointAfter(op);
      if (auto arg = dyn_cast<BlockArgument>(value))
        rewriter.setInsertionPointToStart(value.getParentBlock());
    };

    getValuesToObserve(&clonedOp.getBody(), setInsertionPointAfterDef,
                       typeConverter, rewriter, observeValues);

    // Create the `llhd.wait` op that suspends the current process and waits for
    // a change in the interesting values listed in `observeValues`. When a
    // change is detected, execution resumes in the "check" block.
    auto waitOp = llhd::WaitOp::create(rewriter, loc, ValueRange{}, Value(),
                                       observeValues, ValueRange{}, checkBlock);
    rewriter.inlineBlockBefore(&clonedOp.getBody().front(), waitOp);
    rewriter.eraseOp(clonedOp);

    // Also observe the values that the wait event itself is tracking. These
    // have now been materialized in the wait block as `valuesBefore`, so append
    // them to the wait op's observed operand segment.
    SmallVector<Value> trackedValues;
    for (auto value : valuesBefore) {
      auto type = typeConverter->convertType(value.getType());
      if (!type || !hw::isHWValueType(type))
        continue;
      Value converted = value;
      if (converted.getType() != type) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(waitOp);
        converted = typeConverter->materializeTargetConversion(
            rewriter, loc, type, converted);
      }
      trackedValues.push_back(converted);
    }
    if (!trackedValues.empty())
      waitOp.getObservedMutable().append(trackedValues);

    // Collect a list of all detect ops and inline the `wait_event` body into
    // the check block.
    SmallVector<DetectEventOp> detectOps(op.getBody().getOps<DetectEventOp>());
    rewriter.inlineBlockBefore(&op.getBody().front(), checkBlock,
                               checkBlock->end());
    rewriter.eraseOp(op);

    // Helper function to detect if a certain change occurred between a value
    // before the `llhd.wait` and after.
    auto computeTrigger = [&](Value before, Value after, Edge edge) -> Value {
      assert(before.getType() == after.getType() &&
             "mismatched types after clone op");
      auto beforeType = cast<IntType>(before.getType());

      // 9.4.2 IEEE 1800-2017: An edge event shall be detected only on the LSB
      // of the expression
      if (beforeType.getWidth() != 1 && edge != Edge::AnyChange) {
        constexpr int LSB = 0;
        beforeType =
            IntType::get(rewriter.getContext(), 1, beforeType.getDomain());
        before =
            moore::ExtractOp::create(rewriter, loc, beforeType, before, LSB);
        after = moore::ExtractOp::create(rewriter, loc, beforeType, after, LSB);
      }

      auto intType = rewriter.getIntegerType(beforeType.getWidth());
      before = typeConverter->materializeTargetConversion(rewriter, loc,
                                                          intType, before);
      after = typeConverter->materializeTargetConversion(rewriter, loc, intType,
                                                         after);

      if (edge == Edge::AnyChange)
        return comb::ICmpOp::create(rewriter, loc, ICmpPredicate::ne, before,
                                    after, true);

      SmallVector<Value> disjuncts;
      Value trueVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 1));

      if (edge == Edge::PosEdge || edge == Edge::BothEdges) {
        Value notOldVal =
            comb::XorOp::create(rewriter, loc, before, trueVal, true);
        Value posedge =
            comb::AndOp::create(rewriter, loc, notOldVal, after, true);
        disjuncts.push_back(posedge);
      }

      if (edge == Edge::NegEdge || edge == Edge::BothEdges) {
        Value notCurrVal =
            comb::XorOp::create(rewriter, loc, after, trueVal, true);
        Value posedge =
            comb::AndOp::create(rewriter, loc, before, notCurrVal, true);
        disjuncts.push_back(posedge);
      }

      return rewriter.createOrFold<comb::OrOp>(loc, disjuncts, true);
    };

    // Convert all `detect_event` ops into a check for the corresponding event
    // between the value before and after the `llhd.wait`. The "before" value
    // has been collected into `valuesBefore` in the "wait" block; the "after"
    // value corresponds to the detect op's input.
    SmallVector<Value> triggers;
    for (auto [detectOp, before] : llvm::zip(detectOps, valuesBefore)) {
      if (!allDetectsAreAnyChange) {
        if (!isa<IntType>(before.getType()))
          return detectOp->emitError() << "requires int operand";

        rewriter.setInsertionPoint(detectOp);
        auto trigger =
            computeTrigger(before, detectOp.getInput(), detectOp.getEdge());
        if (detectOp.getCondition()) {
          auto condition = typeConverter->materializeTargetConversion(
              rewriter, loc, rewriter.getI1Type(), detectOp.getCondition());
          trigger =
              comb::AndOp::create(rewriter, loc, trigger, condition, true);
        }
        triggers.push_back(trigger);
      }

      rewriter.eraseOp(detectOp);
    }

    rewriter.setInsertionPointToEnd(checkBlock);
    if (triggers.empty()) {
      // If there are no triggers to check, we always branch to the resume
      // block. If there are no detect_event operations in the wait event, the
      // 'llhd.wait' operation will not have any observed values and thus the
      // process will hang there forever.
      cf::BranchOp::create(rewriter, loc, resumeBlock);
    } else {
      // If any `detect_event` op detected an event, branch to the "resume"
      // block which contains any code after the `wait_event` op. If no events
      // were detected, branch back to the "wait" block to wait for the next
      // change on the interesting signals.
      auto triggered = rewriter.createOrFold<comb::OrOp>(loc, triggers, true);
      cf::CondBranchOp::create(rewriter, loc, triggered, resumeBlock,
                               waitBlock);
    }

    return success();
  }
};

// moore.wait_delay -> llhd.wait
static LogicalResult convert(WaitDelayOp op, WaitDelayOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto *resumeBlock =
      rewriter.splitBlock(op->getBlock(), ++Block::iterator(op));
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<llhd::WaitOp>(op, ValueRange{},
                                            adaptor.getDelay(), ValueRange{},
                                            ValueRange{}, resumeBlock);
  rewriter.setInsertionPointToStart(resumeBlock);
  return success();
}

// moore.unreachable -> llhd.halt
static LogicalResult convert(UnreachableOp op, UnreachableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::HaltOp>(op, ValueRange{});
  return success();
}

//===----------------------------------------------------------------------===//
// Declaration Conversion
//===----------------------------------------------------------------------===//

struct VariableOpConversion : public OpConversionPattern<VariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op.getLoc(), "invalid variable type");

    // Determine the initial value of the signal.
    Value init = adaptor.getInitial();
    if (!init) {
      Type elementType;
      if (auto inoutTy = dyn_cast<hw::InOutType>(resultType))
        elementType = inoutTy.getElementType();
      else if (auto ptrTy = dyn_cast<llhd::PtrType>(resultType))
        elementType = ptrTy.getElementType();
      else
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "unexpected converted variable type");
      if (isa<hw::StringType>(elementType)) {
        rewriter.getContext()->getOrLoadDialect<sv::SVDialect>();
        init = rewriter.create<sv::ConstantStrOp>(loc, cast<hw::StringType>(elementType),
                                                  rewriter.getStringAttr(""));
      } else {
      int64_t width = hw::getBitWidth(elementType);
      if (width == -1)
        return failure();

      // TODO: Once the core dialects support four-valued integers, this code
      // will additionally need to generate an all-X value for four-valued
      // variables.
      Value constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
      init = rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);
      }
    }

    if (isa<llhd::PtrType>(resultType)) {
      rewriter.replaceOpWithNewOp<llhd::VarOp>(op, resultType, init);
      return success();
    }

    rewriter.replaceOpWithNewOp<llhd::SignalOp>(op, resultType, op.getNameAttr(),
                                                init);
    return success();
  }
};

static func::FuncOp getOrCreateStringCmpFn(ModuleOp module,
                                           ConversionPatternRewriter &rewriter) {
  if (auto existing = module.lookupSymbol<func::FuncOp>("circt_sv_strcmp"))
    return existing;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto strTy = hw::StringType::get(module.getContext());
  auto i32Ty = rewriter.getI32Type();
  auto fnType = rewriter.getFunctionType({strTy, strTy}, {i32Ty});
  auto fn =
      rewriter.create<func::FuncOp>(module.getLoc(), "circt_sv_strcmp", fnType);
  fn.setPrivate();
  return fn;
}

struct StringCmpOpConversion : public OpConversionPattern<StringCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto strcmpFn = getOrCreateStringCmpFn(module, rewriter);
    auto call = rewriter.create<func::CallOp>(op.getLoc(), strcmpFn,
                                              ValueRange{adaptor.getLhs(),
                                                         adaptor.getRhs()});
    Value cmp = call.getResult(0);
    Value zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                        rewriter.getI32Type(), 0);

    comb::ICmpPredicate pred;
    switch (op.getPredicate()) {
    case moore::StringCmpPredicate::eq:
      pred = comb::ICmpPredicate::eq;
      break;
    case moore::StringCmpPredicate::ne:
      pred = comb::ICmpPredicate::ne;
      break;
    case moore::StringCmpPredicate::lt:
      pred = comb::ICmpPredicate::slt;
      break;
    case moore::StringCmpPredicate::le:
      pred = comb::ICmpPredicate::sle;
      break;
    case moore::StringCmpPredicate::gt:
      pred = comb::ICmpPredicate::sgt;
      break;
    case moore::StringCmpPredicate::ge:
      pred = comb::ICmpPredicate::sge;
      break;
    }

    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, cmp, zero);
    return success();
  }
};

struct NetOpConversion : public OpConversionPattern<NetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getKind() != NetKind::Wire)
      return rewriter.notifyMatchFailure(loc, "only wire nets supported");

    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(loc, "invalid net type");

    // TODO: Once the core dialects support four-valued integers, this code
    // will additionally need to generate an all-X value for four-valued nets.
    auto elementType = cast<hw::InOutType>(resultType).getElementType();
    int64_t width = hw::getBitWidth(elementType);
    if (width == -1)
      return failure();

    // Many sv-tests simulation cases declare constant-driven wires via a net
    // initializer (e.g. `wire [N:0] a = <const>;`) and then immediately
    // compute expressions from those nets inside a `final` block. Later in the
    // pipeline we intentionally drop LLHD inout storage semantics for
    // `llhd.signal`/`llhd.drv`/`llhd.prb` in graph regions; if we always
    // initialize nets to 0 and model the initializer as an epsilon-timed drive
    // only, the driven value can get lost and these tests become vacuous or
    // wrong. As a best-effort, if the net initializer is trivially constant,
    // use it as the signal's initializer directly.
    auto isTriviallyConstant = [](Value v) -> bool {
      while (Operation *defOp = v.getDefiningOp()) {
        if (defOp->hasTrait<OpTrait::ConstantLike>())
          return true;
        if (auto bitcast = dyn_cast<hw::BitcastOp>(defOp)) {
          v = bitcast.getInput();
          continue;
        }
        if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
          if (cast.getInputs().size() != 1)
            break;
          v = cast.getInputs().front();
          continue;
        }
        break;
      }
      return false;
    };

    Value init;
    if (auto assignedValue = adaptor.getAssignment())
      if (assignedValue.getType() == elementType &&
          isTriviallyConstant(assignedValue))
        init = assignedValue;

    if (!init) {
      auto constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
      init = rewriter.createOrFold<hw::BitcastOp>(loc, elementType, constZero);
    }

    auto signal = rewriter.replaceOpWithNewOp<llhd::SignalOp>(
        op, resultType, op.getNameAttr(), init);

    if (auto assignedValue = adaptor.getAssignment()) {
      // If we used the assignment as an initializer already, there's no need
      // to add an extra drive. This also avoids introducing epsilon scheduling
      // requirements for constant-driven nets.
      if (assignedValue == init)
        return success();
      auto timeAttr = llhd::TimeAttr::get(resultType.getContext(), 0U,
                                          llvm::StringRef("ns"), 0, 1);
      auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
      llhd::DrvOp::create(rewriter, loc, signal, assignedValue, time, Value{});
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // FIXME: Discard unknown bits and map them to 0 for now.
    auto value = op.getValue().toAPInt(false);
    auto type = rewriter.getIntegerType(value.getBitWidth());
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, type, rewriter.getIntegerAttr(type, value));
    return success();
  }
};

struct RealLiteralOpConv : public OpConversionPattern<RealLiteralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RealLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    auto attr = op.getValueAttr();
    if (attr.getType() != resultType)
      attr = FloatAttr::get(resultType, attr.getValueAsDouble());

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, resultType, attr);
    return success();
  }
};

struct ConstantTimeOpConv : public OpConversionPattern<ConstantTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::ConstantTimeOp>(
        op, llhd::TimeAttr::get(op->getContext(), op.getValue(),
                                StringRef("fs"), 0, 0));
    return success();
  }
};

struct TimeBIOpConv : public OpConversionPattern<TimeBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TimeBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.replaceOpWithNewOp<llhd::NowOp>(op);
    return success();
  }
};

static mlir::func::FuncOp getOrCreateExternFunc(mlir::ModuleOp module,
                                                StringRef name,
                                                FunctionType fnType,
                                                ConversionPatternRewriter &rewriter) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name))
    return existing;
  PatternRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto fn = rewriter.create<mlir::func::FuncOp>(module.getLoc(), name, fnType);
  fn.setPrivate();
  return fn;
}

struct UrandomBIOpConv : public OpConversionPattern<UrandomBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UrandomBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands;
    StringRef fnName = "circt_sv_urandom_u32";
    if (Value seed = adaptor.getSeed()) {
      operands.push_back(seed);
      fnName = "circt_sv_urandom_seed_u32";
    }

    SmallVector<Type> inputTypes;
    inputTypes.reserve(operands.size());
    for (Value v : operands)
      inputTypes.push_back(v.getType());
    auto fnType = FunctionType::get(op.getContext(), inputTypes, {resultType});
    auto fn = getOrCreateExternFunc(module, fnName, fnType, rewriter);
    auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), fn, operands);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct RandomBIOpConv : public OpConversionPattern<RandomBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RandomBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands;
    StringRef fnName = "circt_sv_random_i32";
    if (Value seed = adaptor.getSeed()) {
      operands.push_back(seed);
      fnName = "circt_sv_random_seed_i32";
    }

    SmallVector<Type> inputTypes;
    inputTypes.reserve(operands.size());
    for (Value v : operands)
      inputTypes.push_back(v.getType());
    auto fnType = FunctionType::get(op.getContext(), inputTypes, {resultType});
    auto fn = getOrCreateExternFunc(module, fnName, fnType, rewriter);
    auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), fn, operands);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct TimeToLogicOpConv : public OpConversionPattern<TimeToLogicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TimeToLogicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::TimeToIntOp>(op, adaptor.getInput());
    return success();
  }
};

struct LogicToTimeOpConv : public OpConversionPattern<LogicToTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LogicToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::IntToTimeOp>(op, adaptor.getInput());
    return success();
  }
};

struct StringConstantOpConv : public OpConversionPattern<StringConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(moore::StringConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto resultType =
        typeConverter->convertType(op.getResult().getType());
    const auto intType = mlir::cast<IntegerType>(resultType);

    const auto str = op.getValue();
    const unsigned byteWidth = intType.getWidth();
    APInt value(byteWidth, 0);

    // Pack ascii chars from the end of the string, until it fits.
    const size_t maxChars =
        std::min(str.size(), static_cast<size_t>(byteWidth / 8));
    for (size_t i = 0; i < maxChars; i++) {
      const size_t pos = str.size() - 1 - i;
      const auto asciiChar = static_cast<uint8_t>(str[pos]);
      value |= APInt(byteWidth, asciiChar) << (8 * i);
    }

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, resultType, rewriter.getIntegerAttr(resultType, value));
    return success();
  }
};

template <typename MooreOp, typename MathOp>
struct RealMathBuiltinOpConversion : public OpConversionPattern<MooreOp> {
  using OpConversionPattern<MooreOp>::OpConversionPattern;
  using OpAdaptor = typename MooreOp::Adaptor;

  LogicalResult
  matchAndRewrite(MooreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    rewriter.replaceOpWithNewOp<MathOp>(op, resultType, adaptor.getValue());
    return success();
  }
};

struct ReadInterfaceSignalOpConversion
    : public OpConversionPattern<sv::ReadInterfaceSignalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(sv::ReadInterfaceSignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    Type newType = converter->convertType(op.getResult().getType());
    if (!newType)
      return rewriter.notifyMatchFailure(op, "unable to convert result type");

    auto newOp = rewriter.create<sv::ReadInterfaceSignalOp>(
        op.getLoc(), newType, adaptor.getIface(), op.getSignalNameAttr());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConcatOpConversion : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType,
                                                   adaptor.getValue());
    return success();
  }
};

struct ExtractOpConversion : public OpConversionPattern<ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: return X if the domain is four-valued for out-of-bounds accesses
    // once we support four-valued lowering
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType = adaptor.getInput().getType();
    int32_t low = adaptor.getLowBit();

    if (isa<IntegerType>(inputType)) {
      int32_t inputWidth = inputType.getIntOrFloatBitWidth();
      int32_t resultWidth = hw::getBitWidth(resultType);
      int32_t high = low + resultWidth;

      SmallVector<Value> toConcat;
      if (low < 0)
        toConcat.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(std::min(-low, resultWidth), 0)));

      if (low < inputWidth && high > 0) {
        int32_t lowIdx = std::max(low, 0);
        Value middle = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(),
            rewriter.getIntegerType(
                std::min(resultWidth, std::min(high, inputWidth) - lowIdx)),
            adaptor.getInput(), lowIdx);
        toConcat.push_back(middle);
      }

      int32_t diff = high - inputWidth;
      if (diff > 0) {
        Value val =
            hw::ConstantOp::create(rewriter, op.getLoc(), APInt(diff, 0));
        toConcat.push_back(val);
      }

      Value concat =
          rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), toConcat);
      rewriter.replaceOp(op, concat);
      return success();
    }

    if (auto arrTy = dyn_cast<hw::ArrayType>(inputType)) {
      int32_t width = llvm::Log2_64_Ceil(arrTy.getNumElements());
      int32_t inputWidth = arrTy.getNumElements();

      if (auto resArrTy = dyn_cast<hw::ArrayType>(resultType)) {
        int32_t elementWidth = hw::getBitWidth(arrTy.getElementType());
        if (elementWidth < 0)
          return failure();

        int32_t high = low + resArrTy.getNumElements();
        int32_t resWidth = resArrTy.getNumElements();

        SmallVector<Value> toConcat;
        if (low < 0) {
          Value val = hw::ConstantOp::create(
              rewriter, op.getLoc(),
              APInt(std::min((-low) * elementWidth, resWidth * elementWidth),
                    0));
          Value res = rewriter.createOrFold<hw::BitcastOp>(
              op.getLoc(), hw::ArrayType::get(arrTy.getElementType(), -low),
              val);
          toConcat.push_back(res);
        }

        if (low < inputWidth && high > 0) {
          int32_t lowIdx = std::max(0, low);
          Value lowIdxVal = hw::ConstantOp::create(
              rewriter, op.getLoc(), rewriter.getIntegerType(width), lowIdx);
          Value middle = rewriter.createOrFold<hw::ArraySliceOp>(
              op.getLoc(),
              hw::ArrayType::get(
                  arrTy.getElementType(),
                  std::min(resWidth, std::min(inputWidth, high) - lowIdx)),
              adaptor.getInput(), lowIdxVal);
          toConcat.push_back(middle);
        }

        int32_t diff = high - inputWidth;
        if (diff > 0) {
          Value constZero = hw::ConstantOp::create(
              rewriter, op.getLoc(), APInt(diff * elementWidth, 0));
          Value val = hw::BitcastOp::create(
              rewriter, op.getLoc(),
              hw::ArrayType::get(arrTy.getElementType(), diff), constZero);
          toConcat.push_back(val);
        }

        Value concat =
            rewriter.createOrFold<hw::ArrayConcatOp>(op.getLoc(), toConcat);
        rewriter.replaceOp(op, concat);
        return success();
      }

      // Otherwise, it has to be the array's element type
      if (low < 0 || low >= inputWidth) {
        int32_t bw = hw::getBitWidth(resultType);
        if (bw < 0)
          return failure();

        Value val = hw::ConstantOp::create(rewriter, op.getLoc(), APInt(bw, 0));
        Value bitcast =
            rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), resultType, val);
        rewriter.replaceOp(op, bitcast);
        return success();
      }

      Value idx = hw::ConstantOp::create(rewriter, op.getLoc(),
                                         rewriter.getIntegerType(width),
                                         adaptor.getLowBit());
      rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(), idx);
      return success();
    }

    return failure();
  }
};

struct ExtractRefOpConversion : public OpConversionPattern<ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType =
        cast<hw::InOutType>(adaptor.getInput().getType()).getElementType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      int64_t width = hw::getBitWidth(inputType);
      if (width == -1)
        return failure();

      Value lowBit = hw::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerType(llvm::Log2_64_Ceil(width)),
          adaptor.getLowBit());
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultType, adaptor.getInput(), lowBit);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      Value lowBit = hw::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerType(llvm::Log2_64_Ceil(arrType.getNumElements())),
          adaptor.getLowBit());

      // If the result type is not the same as the array's element type, then
      // it has to be a slice.
      if (arrType.getElementType() !=
          cast<hw::InOutType>(resultType).getElementType()) {
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, adaptor.getInput(), lowBit);
        return success();
      }

      rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, adaptor.getInput(),
                                                       lowBit);
      return success();
    }

    return failure();
  }
};

struct DynExtractOpConversion : public OpConversionPattern<DynExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType = adaptor.getInput().getType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      Value amount = adjustIntegerWidth(rewriter, adaptor.getLowBit(),
                                        intType.getWidth(), op->getLoc());
      Value value = comb::ShrUOp::create(rewriter, op->getLoc(),
                                         adaptor.getInput(), amount);

      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, value, 0);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
      Value idx = adjustIntegerWidth(rewriter, adaptor.getLowBit(), idxWidth,
                                     op->getLoc());

      if (isa<hw::ArrayType>(resultType)) {
        rewriter.replaceOpWithNewOp<hw::ArraySliceOp>(op, resultType,
                                                      adaptor.getInput(), idx);
        return success();
      }

      rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(), idx);
      return success();
    }

    return failure();
  }
};

struct DynExtractRefOpConversion : public OpConversionPattern<DynExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    Type resultType = typeConverter->convertType(op.getResult().getType());
    Type inputType =
        cast<hw::InOutType>(adaptor.getInput().getType()).getElementType();

    if (auto intType = dyn_cast<IntegerType>(inputType)) {
      int64_t width = hw::getBitWidth(inputType);
      if (width == -1)
        return failure();

      Value amount =
          adjustIntegerWidth(rewriter, adaptor.getLowBit(),
                             llvm::Log2_64_Ceil(width), op->getLoc());
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultType, adaptor.getInput(), amount);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      Value idx = adjustIntegerWidth(
          rewriter, adaptor.getLowBit(),
          llvm::Log2_64_Ceil(arrType.getNumElements()), op->getLoc());

      if (isa<hw::ArrayType>(
              cast<hw::InOutType>(resultType).getElementType())) {
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, adaptor.getInput(), idx);
        return success();
      }

      rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, adaptor.getInput(),
                                                       idx);
      return success();
    }

    return failure();
  }
};

struct ArrayCreateOpConversion : public OpConversionPattern<ArrayCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<hw::ArrayCreateOp>(op, resultType,
                                                   adaptor.getElements());
    return success();
  }
};

struct StructCreateOpConversion : public OpConversionPattern<StructCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<hw::StructCreateOp>(op, resultType,
                                                    adaptor.getFields());
    return success();
  }
};

struct StructExtractOpConversion : public OpConversionPattern<StructExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct StructExtractRefOpConversion
    : public OpConversionPattern<StructExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StructExtractRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<llhd::SigStructExtractOp>(
        op, adaptor.getInput(), adaptor.getFieldNameAttr());
    return success();
  }
};

struct ReduceAndOpConversion : public OpConversionPattern<ReduceAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value max = hw::ConstantOp::create(rewriter, op->getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                              adaptor.getInput(), max);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    Value zero = hw::ConstantOp::create(rewriter, op->getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                              adaptor.getInput(), zero);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<comb::ParityOp>(op, adaptor.getInput());
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getInput().getType());
    if (isa_and_nonnull<IntegerType>(resultType)) {
      Value zero =
          hw::ConstantOp::create(rewriter, op->getLoc(), resultType, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                adaptor.getInput(), zero);
      return success();
    }
    return failure();
  }
};

struct NotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value max = hw::ConstantOp::create(rewriter, op.getLoc(), resultType, -1);

    rewriter.replaceOpWithNewOp<comb::XorOp>(op, adaptor.getInput(), max);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    Value zero = hw::ConstantOp::create(rewriter, op.getLoc(), resultType, 0);

    rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, adaptor.getInput());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs(), false);
    return success();
  }
};

template <typename SourceOp, ICmpPredicate pred>
struct ICmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, resultType, pred, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename SourceOp, bool withoutX>
struct CaseXZEqOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check each operand if it is a known constant and extract the X and/or Z
    // bits to be ignored.
    // TODO: Once the core dialects support four-valued integers, we will have
    // to create ops that extract X and Z bits from the operands, since we also
    // have to do the right casez/casex comparison on non-constant inputs.
    unsigned bitWidth = op.getLhs().getType().getWidth();
    auto ignoredBits = APInt::getZero(bitWidth);
    auto detectIgnoredBits = [&](Value value) {
      auto constOp = value.getDefiningOp<ConstantOp>();
      if (!constOp)
        return;
      auto constValue = constOp.getValue();
      if (withoutX)
        ignoredBits |= constValue.getZBits();
      else
        ignoredBits |= constValue.getUnknownBits();
    };
    detectIgnoredBits(op.getLhs());
    detectIgnoredBits(op.getRhs());

    // If we have detected any bits to be ignored, mask them in the operands for
    // the comparison.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (!ignoredBits.isZero()) {
      ignoredBits.flipAllBits();
      auto maskOp = hw::ConstantOp::create(rewriter, op.getLoc(), ignoredBits);
      lhs = rewriter.createOrFold<comb::AndOp>(op.getLoc(), lhs, maskOp);
      rhs = rewriter.createOrFold<comb::AndOp>(op.getLoc(), rhs, maskOp);
    }

    rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, ICmpPredicate::ceq, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

struct ConversionOpConversion : public OpConversionPattern<ConversionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConversionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    if (auto fmtTy = dyn_cast<sim::FormatStringType>(resultType)) {
      if (auto strConst =
              op.getInput().getDefiningOp<moore::StringConstantOp>()) {
        rewriter.replaceOpWithNewOp<sim::FormatLitOp>(
            op, rewriter.getStringAttr(strConst.getValue()));
        return success();
      }

      if (auto hwConst =
              adaptor.getInput().getDefiningOp<hw::ConstantOp>()) {
        auto value = hwConst.getValue();
        unsigned numBytes = (value.getBitWidth() + 7) / 8;
        std::string decoded;
        decoded.reserve(numBytes);
        for (unsigned i = 0; i < numBytes; ++i) {
          auto byte = value.extractBits(8, i * 8).getZExtValue();
          decoded.push_back(static_cast<char>(byte));
        }
        while (!decoded.empty() && decoded.back() == '\0')
          decoded.pop_back();
        std::reverse(decoded.begin(), decoded.end());
        rewriter.replaceOpWithNewOp<sim::FormatLitOp>(
            op, rewriter.getStringAttr(decoded));
        return success();
      }

      if (adaptor.getInput().getType() == fmtTy) {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
      }

      return failure();
    }

    if (auto strTy = dyn_cast<hw::StringType>(resultType)) {
      auto emitLiteral = [&](StringRef text) {
        rewriter.replaceOpWithNewOp<sv::ConstantStrOp>(
            op, strTy, rewriter.getStringAttr(text));
      };

      if (auto strConst =
              op.getInput().getDefiningOp<moore::StringConstantOp>()) {
        emitLiteral(strConst.getValue());
        return success();
      }

      if (auto hwConst =
              adaptor.getInput().getDefiningOp<hw::ConstantOp>()) {
        auto value = hwConst.getValue();
        unsigned numBytes = (value.getBitWidth() + 7) / 8;
        std::string decoded;
        decoded.reserve(numBytes);
        for (unsigned i = 0; i < numBytes; ++i) {
          auto byte = value.extractBits(8, i * 8).getZExtValue();
          decoded.push_back(static_cast<char>(byte));
        }
        while (!decoded.empty() && decoded.back() == '\0')
          decoded.pop_back();
        std::reverse(decoded.begin(), decoded.end());
        emitLiteral(decoded);
        return success();
      }

      if (adaptor.getInput().getType() == strTy) {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
      }

      return failure();
    }

    // Integer <-> floating point conversions.
    if (auto floatResTy = dyn_cast<FloatType>(resultType)) {
      if (auto floatInTy = dyn_cast<FloatType>(adaptor.getInput().getType())) {
        if (floatInTy == floatResTy) {
          rewriter.replaceOp(op, adaptor.getInput());
          return success();
        }
        if (floatInTy.getWidth() < floatResTy.getWidth()) {
          rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(
              op, floatResTy, adaptor.getInput());
          return success();
        }
        if (floatInTy.getWidth() > floatResTy.getWidth()) {
          rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(
              op, floatResTy, adaptor.getInput());
          return success();
        }
      }

      if (isa<IntegerType>(adaptor.getInput().getType())) {
        rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(
            op, floatResTy, adaptor.getInput());
        return success();
      }
    }

    if (auto intResTy = dyn_cast<IntegerType>(resultType)) {
      if (isa<FloatType>(adaptor.getInput().getType())) {
        rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(
            op, intResTy, adaptor.getInput());
        return success();
      }
    }

    int64_t inputBw = hw::getBitWidth(adaptor.getInput().getType());
    int64_t resultBw = hw::getBitWidth(resultType);
    if (inputBw == -1 || resultBw == -1)
      return failure();

    Value input = rewriter.createOrFold<hw::BitcastOp>(
        loc, rewriter.getIntegerType(inputBw), adaptor.getInput());
    Value amount = adjustIntegerWidth(rewriter, input, resultBw, loc);

    Value result =
        rewriter.createOrFold<hw::BitcastOp>(loc, resultType, amount);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp>
struct BitcastConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConversionPattern::typeConverter;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (type == adaptor.getInput().getType())
      rewriter.replaceOp(op, adaptor.getInput());
    else
      rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, type, adaptor.getInput());
    return success();
  }
};

struct TruncOpConversion : public OpConversionPattern<TruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, adaptor.getInput(), 0,
                                                 op.getType().getWidth());
    return success();
  }
};

struct ZExtOpConversion : public OpConversionPattern<ZExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetWidth = op.getType().getWidth();
    auto inputWidth = op.getInput().getType().getWidth();

    auto zeroExt = hw::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerType(targetWidth - inputWidth), 0);

    rewriter.replaceOpWithNewOp<comb::ConcatOp>(
        op, ValueRange{zeroExt, adaptor.getInput()});
    return success();
  }
};

struct SExtOpConversion : public OpConversionPattern<SExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getType());
    auto value =
        comb::createOrFoldSExt(op.getLoc(), adaptor.getInput(), type, rewriter);
    rewriter.replaceOp(op, value);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Statement Conversion
//===----------------------------------------------------------------------===//

struct HWInstanceOpConversion : public OpConversionPattern<hw::InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, convResTypes, op.getInstanceName(), op.getModuleName(),
        adaptor.getOperands(), op.getArgNames(),
        op.getResultNames(), /*Parameter*/
        rewriter.getArrayAttr({}), /*InnerSymbol*/ nullptr);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), /*branch_weights=*/nullptr,
        op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(),
                                              adaptor.getDestOperands());
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());
    return success();
  }
};

struct UnrealizedConversionCastConversion
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();

    // Drop the cast if the operand and result types agree after type
    // conversion.
    if (convResTypes == adaptor.getOperands().getTypes()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, convResTypes, adaptor.getOperands());
    return success();
  }
};

struct ShlOpConversion : public OpConversionPattern<ShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, adaptor.getValue(),
                                             amount, false);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrUOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct PowUOpConversion : public OpConversionPattern<PowUOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    Location loc = op->getLoc();

    Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    // zero extend both LHS & RHS to ensure the unsigned integers are
    // interpreted correctly when calculating power
    auto lhs = comb::ConcatOp::create(rewriter, loc, zeroVal, adaptor.getLhs());
    auto rhs = comb::ConcatOp::create(rewriter, loc, zeroVal, adaptor.getRhs());

    // lower the exponentiation via MLIR's math dialect
    auto pow = mlir::math::IPowIOp::create(rewriter, loc, lhs, rhs);

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, resultType, pow, 0);
    return success();
  }
};

struct PowSOpConversion : public OpConversionPattern<PowSOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // utilize MLIR math dialect's math.ipowi to handle the exponentiation of
    // expression
    rewriter.replaceOpWithNewOp<mlir::math::IPowIOp>(
        op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    // Comb shift operations require the same bit-width for value and amount
    Value amount =
        adjustIntegerWidth(rewriter, adaptor.getAmount(),
                           resultType.getIntOrFloatBitWidth(), op->getLoc());
    rewriter.replaceOpWithNewOp<comb::ShrSOp>(
        op, resultType, adaptor.getValue(), amount, false);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<llhd::PtrType>(adaptor.getInput().getType()))
      rewriter.replaceOpWithNewOp<llhd::LoadOp>(op, adaptor.getInput());
    else
      rewriter.replaceOpWithNewOp<llhd::PrbOp>(op, adaptor.getInput());
    return success();
  }
};

struct AssignedVariableOpConversion
    : public OpConversionPattern<AssignedVariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AssignedVariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::WireOp>(op, adaptor.getInput(),
                                            adaptor.getNameAttr());
    return success();
  }
};

template <typename OpTy, unsigned DeltaTime, unsigned EpsilonTime>
struct AssignOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<llhd::PtrType>(adaptor.getDst().getType())) {
      rewriter.replaceOpWithNewOp<llhd::StoreOp>(op, adaptor.getDst(),
                                                 adaptor.getSrc());
      return success();
    }

    // TODO: When we support delay control in Moore dialect, we need to update
    // this conversion.
    auto timeAttr = llhd::TimeAttr::get(
        op->getContext(), 0U, llvm::StringRef("ns"), DeltaTime, EpsilonTime);
    auto time = llhd::ConstantTimeOp::create(rewriter, op->getLoc(), timeAttr);
    rewriter.replaceOpWithNewOp<llhd::DrvOp>(op, adaptor.getDst(),
                                             adaptor.getSrc(), time, Value{});
    return success();
  }
};

struct ConditionalOpConversion : public OpConversionPattern<ConditionalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConditionalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: This lowering is only correct if the condition is two-valued. If
    // the condition is X or Z, both branches of the conditional must be
    // evaluated and merged with the appropriate lookup table. See documentation
    // for `ConditionalOp`.
    auto type = typeConverter->convertType(op.getType());

    auto hasNoWriteEffect = [](Region &region) {
      auto result = region.walk([](Operation *operation) {
        if (auto memOp = dyn_cast<MemoryEffectOpInterface>(operation))
          if (!memOp.hasEffect<MemoryEffects::Write>() &&
              !memOp.hasEffect<MemoryEffects::Free>())
            return WalkResult::advance();

        if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return WalkResult::advance();

        return WalkResult::interrupt();
      });
      return !result.wasInterrupted();
    };

    if (hasNoWriteEffect(op.getTrueRegion()) &&
        hasNoWriteEffect(op.getFalseRegion())) {
      Operation *trueTerm = op.getTrueRegion().front().getTerminator();
      Operation *falseTerm = op.getFalseRegion().front().getTerminator();

      rewriter.inlineBlockBefore(&op.getTrueRegion().front(), op);
      rewriter.inlineBlockBefore(&op.getFalseRegion().front(), op);

      Value convTrueVal = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), type, trueTerm->getOperand(0));
      Value convFalseVal = typeConverter->materializeTargetConversion(
          rewriter, op.getLoc(), type, falseTerm->getOperand(0));

      rewriter.eraseOp(trueTerm);
      rewriter.eraseOp(falseTerm);

      rewriter.replaceOpWithNewOp<comb::MuxOp>(op, adaptor.getCondition(),
                                               convTrueVal, convFalseVal);
      return success();
    }

    auto ifOp =
        scf::IfOp::create(rewriter, op.getLoc(), type, adaptor.getCondition());
    rewriter.inlineRegionBefore(op.getTrueRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getFalseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());
    rewriter.replaceOp(op, ifOp);
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getResult());
    return success();
  }
};

template <typename SourceOp>
struct InPlaceOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

template <typename MooreOpTy, typename VerifOpTy>
struct AssertLikeOpConversion : public OpConversionPattern<MooreOpTy> {
  using OpConversionPattern<MooreOpTy>::OpConversionPattern;
  using OpAdaptor = typename MooreOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(MooreOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr label =
        op.getLabel().has_value()
            ? StringAttr::get(op->getContext(), op.getLabel().value())
            : StringAttr::get(op->getContext());
    rewriter.replaceOpWithNewOp<VerifOpTy>(op, adaptor.getCond(), mlir::Value(),
                                           label);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Format String Conversion
//===----------------------------------------------------------------------===//

struct FormatLiteralOpConversion : public OpConversionPattern<FormatLiteralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::FormatLitOp>(op, adaptor.getLiteral());
    return success();
  }
};

struct FormatStringOpConversion : public OpConversionPattern<FormatStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::FormatStrOp>(op, adaptor.getString());
    return success();
  }
};

struct FormatStringToStringOpConversion
    : public OpConversionPattern<FormatStringToStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatStringToStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto strTy = dyn_cast<hw::StringType>(resultType);
    if (!strTy)
      return failure();

    Value input = adaptor.getFmtstring();

    // If the format string is just a string fragment, the conversion is a no-op.
    if (auto strOp = input.getDefiningOp<sim::FormatStrOp>()) {
      rewriter.replaceOp(op, strOp.getValue());
      return success();
    }

    // If the format string is a literal fragment, materialize a string literal.
    if (auto litOp = input.getDefiningOp<sim::FormatLitOp>()) {
      rewriter.replaceOpWithNewOp<sv::ConstantStrOp>(
          op, strTy, rewriter.getStringAttr(litOp.getLiteral()));
      return success();
    }

    // Best-effort: if the concatenation is entirely literal fragments, fold.
    if (auto concat = input.getDefiningOp<sim::FormatStringConcatOp>()) {
      SmallVector<Value> fragments;
      if (succeeded(concat.getFlattenedInputs(fragments))) {
        std::string text;
        bool allLiteral = true;
        for (Value fragment : fragments) {
          auto fragOp = fragment.getDefiningOp<sim::FormatLitOp>();
          if (!fragOp) {
            allLiteral = false;
            break;
          }
          text += fragOp.getLiteral().str();
        }
        if (allLiteral) {
          rewriter.replaceOpWithNewOp<sv::ConstantStrOp>(
              op, strTy, rewriter.getStringAttr(text));
          return success();
        }
      }
    }

    // Otherwise, drop formatting and produce an empty string for bring-up.
    rewriter.replaceOpWithNewOp<sv::ConstantStrOp>(op, strTy,
                                                   rewriter.getStringAttr(""));
    return success();
  }
};

struct FormatConcatOpConversion : public OpConversionPattern<FormatConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::FormatStringConcatOp>(op,
                                                           adaptor.getInputs());
    return success();
  }
};

struct FormatIntOpConversion : public OpConversionPattern<FormatIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the formatted value can be evaluated to a constant four-valued value,
    // emit a literal formatted string. This preserves correct X/Z printing and
    // avoids lossy conversion of four-valued integers to core dialects.
    DenseMap<Value, std::optional<FVInt>> cache;
    SmallDenseSet<Value, 8> visiting;
    if (auto fv = tryEvaluateMooreIntValue(op.getValue(), cache, visiting)) {
      auto text = formatFourValuedConstant(*fv, op.getFormat());
      rewriter.replaceOpWithNewOp<sim::FormatLitOp>(
          op, rewriter.getStringAttr(text));
      return success();
    }

    // TODO: These should honor the width, alignment, and padding.
    switch (op.getFormat()) {
    case IntFormat::Decimal:
      rewriter.replaceOpWithNewOp<sim::FormatDecOp>(op, adaptor.getValue());
      return success();
    case IntFormat::Binary:
      rewriter.replaceOpWithNewOp<sim::FormatBinOp>(op, adaptor.getValue());
      return success();
    case IntFormat::HexLower:
    case IntFormat::HexUpper:
      rewriter.replaceOpWithNewOp<sim::FormatHexOp>(op, adaptor.getValue());
      return success();
    default:
      return rewriter.notifyMatchFailure(op, "unsupported int format");
    }
  }
};

struct FormatRealOpConversion : public OpConversionPattern<FormatRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef fmt = "float";
    switch (op.getFormat()) {
    case moore::RealFormat::Float:
      fmt = "float";
      break;
    case moore::RealFormat::Exponential:
      fmt = "exponential";
      break;
    case moore::RealFormat::General:
      fmt = "general";
      break;
    }
    auto fmtAttr = rewriter.getStringAttr(fmt);
    rewriter.replaceOpWithNewOp<sim::FormatRealOp>(op, adaptor.getValue(),
                                                   fmtAttr);
    return success();
  }
};

struct DisplayBIOpConversion : public OpConversionPattern<DisplayBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DisplayBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(
        op, adaptor.getMessage());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Simulation Control Conversion
//===----------------------------------------------------------------------===//

// moore.builtin.stop -> sim.pause
static LogicalResult convert(StopBIOp op, StopBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::PauseOp>(op, /*verbose=*/false);
  return success();
}

// moore.builtin.finish -> sim.terminate
static LogicalResult convert(FinishBIOp op, FinishBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::TerminateOp>(op, op.getExitCode() == 0,
                                                /*verbose=*/false);
  return success();
}

// moore.builtin.severity -> sim.proc.print
static LogicalResult convert(SeverityBIOp op, SeverityBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {

  std::string severityString;

  switch (op.getSeverity()) {
  case (Severity::Fatal):
    severityString = "Fatal: ";
    break;
  case (Severity::Error):
    severityString = "Error: ";
    break;
  case (Severity::Warning):
    severityString = "Warning: ";
    break;
  default:
    return failure();
  }

  auto prefix = rewriter.create<sim::FormatLitOp>(op.getLoc(), severityString);
  // SystemVerilog severity tasks (`$fatal`, `$error`, `$warning`) emit a
  // newline, unlike `$write`. The importer models `$display` by explicitly
  // appending a newline fragment, but severity tasks currently do not.
  auto newline = rewriter.create<sim::FormatLitOp>(op.getLoc(), "\n");
  auto message = rewriter.create<sim::FormatStringConcatOp>(
      op.getLoc(), ValueRange{prefix, adaptor.getMessage(), newline});
  rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(op, message);
  return success();
}

// moore.builtin.finish_message
static LogicalResult convert(FinishMessageBIOp op,
                             FinishMessageBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // We don't support printing termination/pause messages yet.
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target,
                             const TypeConverter &converter) {
  target.addIllegalDialect<MooreDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<llhd::LLHDDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<sim::SimDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addLegalOp<sv::InterfaceOp, sv::InterfaceSignalOp,
                    sv::InterfaceModportOp, sv::InterfaceInstanceOp,
                    sv::AssignInterfaceSignalOp, sv::GetModportOp>();

  target.addDynamicallyLegalOp<sv::ReadInterfaceSignalOp>(
      [&](sv::ReadInterfaceSignalOp op) {
        return converter.isLegal(op.getResult().getType());
      });

  target.addLegalOp<debug::ScopeOp>();

  target.addDynamicallyLegalOp<
      cf::CondBranchOp, cf::BranchOp, scf::YieldOp, func::CallOp,
      func::ReturnOp, UnrealizedConversionCastOp, hw::OutputOp, hw::InstanceOp,
      debug::ArrayOp, debug::StructOp, debug::VariableOp>(
      [&](Operation *op) { return converter.isLegal(op); });

  target.addDynamicallyLegalOp<scf::IfOp, scf::ForOp, scf::ExecuteRegionOp,
                               scf::WhileOp, scf::ForallOp>([&](Operation *op) {
    return converter.isLegal(op) && !op->getParentOfType<llhd::ProcessOp>();
  });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType()) &&
           converter.isLegal(&op.getFunctionBody());
  });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
    return converter.isSignatureLegal(op.getModuleType().getFuncType()) &&
           converter.isLegal(&op.getBody());
  });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](IntType type) {
    return IntegerType::get(type.getContext(), type.getWidth());
  });

  typeConverter.addConversion(
      [&](TimeType type) { return llhd::TimeType::get(type.getContext()); });

  typeConverter.addConversion([&](RealType type) -> std::optional<Type> {
    switch (type.getWidth()) {
    case RealWidth::f32:
      return Float32Type::get(type.getContext());
    case RealWidth::f64:
      return Float64Type::get(type.getContext());
    }
    return {};
  });

  typeConverter.addConversion([&](FormatStringType type) {
    return sim::FormatStringType::get(type.getContext());
  });
  typeConverter.addConversion([&](StringType type) {
    return hw::StringType::get(type.getContext());
  });

  typeConverter.addConversion([&](ArrayType type) -> std::optional<Type> {
    if (auto elementType = typeConverter.convertType(type.getElementType()))
      return hw::ArrayType::get(elementType, type.getSize());
    return {};
  });

  // FIXME: Unpacked arrays support more element types than their packed
  // variants, and as such, mapping them to hw::Array is somewhat naive. See
  // also the analogous note below concerning unpacked struct type conversion.
  typeConverter.addConversion(
      [&](UnpackedArrayType type) -> std::optional<Type> {
        if (auto elementType = typeConverter.convertType(type.getElementType()))
          return hw::ArrayType::get(elementType, type.getSize());
        return {};
      });

  typeConverter.addConversion([&](StructType type) -> std::optional<Type> {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::StructType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      fields.push_back(info);
    }
    return hw::StructType::get(type.getContext(), fields);
  });

  // FIXME: Mapping unpacked struct type to struct type in hw dialect may be a
  // plain solution. The packed and unpacked data structures have some
  // differences though they look similarily. The packed data structure is
  // contiguous in memory but another is opposite. The differences will affect
  // data layout and granularity of event tracking in simulation.
  typeConverter.addConversion(
      [&](UnpackedStructType type) -> std::optional<Type> {
        SmallVector<hw::StructType::FieldInfo> fields;
        for (auto field : type.getMembers()) {
          hw::StructType::FieldInfo info;
          info.type = typeConverter.convertType(field.type);
          if (!info.type)
            return {};
          info.name = field.name;
          fields.push_back(info);
        }
        return hw::StructType::get(type.getContext(), fields);
      });

  typeConverter.addConversion([&](RefType type) -> std::optional<Type> {
    if (auto innerType = typeConverter.convertType(type.getNestedType())) {
      if (!hw::isHWValueType(innerType))
        return {};
      // Strings are lowered to pointer-like runtime values. Treat references to
      // strings as stack memory so blocking assignments behave immediately.
      if (isa<moore::StringType>(type.getNestedType()))
        return llhd::PtrType::get(innerType);
      return hw::InOutType::get(innerType);
    }
    return {};
  });

  // Valid target types.
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });
  typeConverter.addConversion([](llhd::TimeType type) { return type; });
  typeConverter.addConversion([&](llhd::PtrType type) -> std::optional<Type> {
    if (auto elem = typeConverter.convertType(type.getElementType()))
      return llhd::PtrType::get(elem);
    return {};
  });
  typeConverter.addConversion([](hw::StringType type) { return type; });
  typeConverter.addConversion([](debug::ArrayType type) { return type; });
  typeConverter.addConversion([](debug::ScopeType type) { return type; });
  typeConverter.addConversion([](debug::StructType type) { return type; });

  typeConverter.addConversion([&](hw::InOutType type) -> std::optional<Type> {
    if (auto innerType = typeConverter.convertType(type.getElementType()))
      return hw::InOutType::get(innerType);
    return {};
  });

  typeConverter.addConversion([&](hw::ArrayType type) -> std::optional<Type> {
    if (auto elementType = typeConverter.convertType(type.getElementType()))
      return hw::ArrayType::get(elementType, type.getNumElements());
    return {};
  });

  typeConverter.addConversion([&](hw::StructType type) -> std::optional<Type> {
    SmallVector<hw::StructType::FieldInfo> fields;
    for (auto field : type.getElements()) {
      hw::StructType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      fields.push_back(info);
    }
    return hw::StructType::get(type.getContext(), fields);
  });

  // Pass SV interface types through unchanged so modules can keep interface
  // ports all the way to HW/SV dialects.
  typeConverter.addConversion(
      [&](sv::InterfaceType type) -> std::optional<Type> { return type; });
  typeConverter.addConversion(
      [&](sv::ModportType type) -> std::optional<Type> { return type; });

  typeConverter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1 || !inputs[0])
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            .getResult(0);
      });

  typeConverter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });
}

static void populateOpConversion(ConversionPatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // clang-format off
  patterns.add<
    // Patterns of declaration operations.
    VariableOpConversion,
    NetOpConversion,

    // Patterns for conversion operations.
    ConversionOpConversion,
    BitcastConversion<PackedToSBVOp>,
    BitcastConversion<SBVToPackedOp>,
    BitcastConversion<LogicToIntOp>,
    BitcastConversion<IntToLogicOp>,
    BitcastConversion<ToBuiltinBoolOp>,
    TruncOpConversion,
    ZExtOpConversion,
    SExtOpConversion,

    // Patterns of miscellaneous operations.
    ConstantOpConv,
    RealLiteralOpConv,
    ConcatOpConversion,
    ReplicateOpConversion,
    ConstantTimeOpConv,
    TimeBIOpConv,
    UrandomBIOpConv,
    RandomBIOpConv,
    TimeToLogicOpConv,
    LogicToTimeOpConv,
    ExtractOpConversion,
    DynExtractOpConversion,
    DynExtractRefOpConversion,
    ReadOpConversion,
    StructExtractOpConversion,
    StructExtractRefOpConversion,
    ExtractRefOpConversion,
    StructCreateOpConversion,
    ConditionalOpConversion,
    ArrayCreateOpConversion,
    YieldOpConversion,
    OutputOpConversion,
    StringConstantOpConv,
    RealMathBuiltinOpConversion<LnBIOp, mlir::math::LogOp>,
    RealMathBuiltinOpConversion<Log10BIOp, mlir::math::Log10Op>,
    RealMathBuiltinOpConversion<ExpBIOp, mlir::math::ExpOp>,
    RealMathBuiltinOpConversion<SqrtBIOp, mlir::math::SqrtOp>,
    RealMathBuiltinOpConversion<FloorBIOp, mlir::math::FloorOp>,
    RealMathBuiltinOpConversion<CeilBIOp, mlir::math::CeilOp>,
    RealMathBuiltinOpConversion<SinBIOp, mlir::math::SinOp>,
    RealMathBuiltinOpConversion<CosBIOp, mlir::math::CosOp>,
    RealMathBuiltinOpConversion<TanBIOp, mlir::math::TanOp>,
    RealMathBuiltinOpConversion<AsinBIOp, mlir::math::AsinOp>,
    RealMathBuiltinOpConversion<AcosBIOp, mlir::math::AcosOp>,
    RealMathBuiltinOpConversion<AtanBIOp, mlir::math::AtanOp>,
    RealMathBuiltinOpConversion<SinhBIOp, mlir::math::SinhOp>,
    RealMathBuiltinOpConversion<CoshBIOp, mlir::math::CoshOp>,
    RealMathBuiltinOpConversion<TanhBIOp, mlir::math::TanhOp>,
    RealMathBuiltinOpConversion<AsinhBIOp, mlir::math::AsinhOp>,
    RealMathBuiltinOpConversion<AcoshBIOp, mlir::math::AcoshOp>,
    RealMathBuiltinOpConversion<AtanhBIOp, mlir::math::AtanhOp>,
    ReadInterfaceSignalOpConversion,

    // Patterns of unary operations.
    ReduceAndOpConversion,
    ReduceOrOpConversion,
    ReduceXorOpConversion,
    BoolCastOpConversion,
    NotOpConversion,
    NegOpConversion,

    // Patterns of binary operations.
    BinaryOpConversion<AddOp, comb::AddOp>,
    BinaryOpConversion<SubOp, comb::SubOp>,
    BinaryOpConversion<MulOp, comb::MulOp>,
    BinaryOpConversion<DivUOp, comb::DivUOp>,
    BinaryOpConversion<DivSOp, comb::DivSOp>,
    BinaryOpConversion<ModUOp, comb::ModUOp>,
    BinaryOpConversion<ModSOp, comb::ModSOp>,
    BinaryOpConversion<AndOp, comb::AndOp>,
    BinaryOpConversion<OrOp, comb::OrOp>,
    BinaryOpConversion<XorOp, comb::XorOp>,

    // Patterns of power operations.
    PowUOpConversion, PowSOpConversion,

    // Patterns of relational operations.
    ICmpOpConversion<UltOp, ICmpPredicate::ult>,
    ICmpOpConversion<SltOp, ICmpPredicate::slt>,
    ICmpOpConversion<UleOp, ICmpPredicate::ule>,
    ICmpOpConversion<SleOp, ICmpPredicate::sle>,
    ICmpOpConversion<UgtOp, ICmpPredicate::ugt>,
    ICmpOpConversion<SgtOp, ICmpPredicate::sgt>,
    ICmpOpConversion<UgeOp, ICmpPredicate::uge>,
    ICmpOpConversion<SgeOp, ICmpPredicate::sge>,
    ICmpOpConversion<EqOp, ICmpPredicate::eq>,
    ICmpOpConversion<NeOp, ICmpPredicate::ne>,
    ICmpOpConversion<CaseEqOp, ICmpPredicate::ceq>,
    ICmpOpConversion<CaseNeOp, ICmpPredicate::cne>,
    ICmpOpConversion<WildcardEqOp, ICmpPredicate::weq>,
    ICmpOpConversion<WildcardNeOp, ICmpPredicate::wne>,
    CaseXZEqOpConversion<CaseZEqOp, true>,
    CaseXZEqOpConversion<CaseXZEqOp, false>,
    StringCmpOpConversion,

    // Patterns of structural operations.
    SVModuleOpConversion,
    InstanceOpConversion,
    ProcedureOpConversion,
    WaitEventOpConversion,

    // Patterns of shifting operations.
    ShrOpConversion,
    ShlOpConversion,
    AShrOpConversion,

    // Patterns of assignment operations.
    AssignOpConversion<ContinuousAssignOp, 0, 1>,
    AssignOpConversion<BlockingAssignOp, 0, 1>,
    AssignOpConversion<NonBlockingAssignOp, 1, 0>,
    AssignedVariableOpConversion,

    // Patterns of branch operations.
    CondBranchOpConversion,
    BranchOpConversion,

    // Patterns of other operations outside Moore dialect.
    HWInstanceOpConversion,
    ReturnOpConversion,
    CallOpConversion,
    UnrealizedConversionCastConversion,
    InPlaceOpConversion<debug::ArrayOp>,
    InPlaceOpConversion<debug::StructOp>,
    InPlaceOpConversion<debug::VariableOp>,

	    // Patterns of assert-like operations
	    AssertLikeOpConversion<AssertOp, verif::AssertOp>,
	    AssertLikeOpConversion<AssumeOp, verif::AssumeOp>,
	    AssertLikeOpConversion<CoverOp, verif::CoverOp>,

		    // Format strings.
		    FormatLiteralOpConversion,
		    FormatStringOpConversion,
		    FormatStringToStringOpConversion,
		    FormatConcatOpConversion,
		    FormatIntOpConversion,
		    FormatRealOpConversion,
		    DisplayBIOpConversion
	  >(typeConverter, patterns.getContext());
  // clang-format on

  // Structural operations
  patterns.add<WaitDelayOp>(convert);
  patterns.add<UnreachableOp>(convert);

  // Simulation control
  patterns.add<StopBIOp>(convert);
  patterns.add<SeverityBIOp>(convert);
  patterns.add<FinishBIOp>(convert);
  patterns.add<FinishMessageBIOp>(convert);

  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                            typeConverter);
  hw::populateHWModuleLikeTypeConversionPattern(
      hw::HWModuleOp::getOperationName(), patterns, typeConverter);
  populateSCFToControlFlowConversionPatterns(patterns);
  populateArithToCombPatterns(patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Moore to Core Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct MooreToCorePass
    : public circt::impl::ConvertMooreToCoreBase<MooreToCorePass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Moore to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertMooreToCorePass() {
  return std::make_unique<MooreToCorePass>();
}

/// This is the main entrypoint for the Moore to Core conversion pass.
void MooreToCorePass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  context.getOrLoadDialect<sv::SVDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  IRRewriter rewriter(module);
  (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());

  TypeConverter typeConverter;
  populateTypeConversion(typeConverter);

  ConversionTarget target(context);
  populateLegality(target, typeConverter);

  ConversionPatternSet patterns(&context, typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
