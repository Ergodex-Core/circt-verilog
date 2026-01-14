//===- LowerInterfaces.cpp - SV interface lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower SystemVerilog interfaces/modports to plain HW structs and inouts so
// that downstream passes (Arc/LLVM) operate without interface types. Ports are
// upgraded to `hw.inout` when any transitive user writes through them, and
// instances are rewritten to match the lowered signatures.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "llvm/ADT/SmallString.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>

namespace circt {
namespace sv {
#define GEN_PASS_DEF_LOWERINTERFACES
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
using namespace circt::hw;
using namespace circt::sv;

namespace {

static constexpr const char kArcilatorSigIdAttr[] = "arcilator.sig_id";

static bool isInProceduralRegion(Operation *op) {
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (parent->hasTrait<sv::ProceduralRegion>() ||
        parent->hasTrait<llhd::ProceduralRegion>() ||
        isa<moore::ProcedureOp>(parent))
      return true;
  }
  return false;
}

static Value getInterfaceBase(Value v) {
  while (auto mp = v.getDefiningOp<sv::GetModportOp>())
    v = mp.getIface();
  return v;
}

static bool isFourValuedIntStruct(Type type, Type &bitsType) {
  auto structTy = dyn_cast<hw::StructType>(type);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (elements[0].name.getValue() != "value" ||
      elements[1].name.getValue() != "unknown")
    return false;
  if (elements[0].type != elements[1].type)
    return false;
  bitsType = elements[0].type;
  return hw::getBitWidth(bitsType) > 0;
}

static Value materializeTwoFourStateConversion(OpBuilder &builder, Location loc,
                                               Value input, Type dstType) {
  if (!input || input.getType() == dstType)
    return input;

  Type dstBitsType;
  Type srcBitsType;
  bool dstIsFour = isFourValuedIntStruct(dstType, dstBitsType);
  bool srcIsFour = isFourValuedIntStruct(input.getType(), srcBitsType);

  auto getOrBitcast = [&](Value v, Type ty) -> Value {
    if (!v || v.getType() == ty)
      return v;
    if (hw::getBitWidth(v.getType()) != hw::getBitWidth(ty))
      return {};
    return builder.create<hw::BitcastOp>(loc, ty, v);
  };

  // 2-state integer -> 4-state struct {value, unknown=0}
  if (dstIsFour && !srcIsFour) {
    int64_t w = hw::getBitWidth(dstBitsType);
    if (w <= 0)
      return {};
    Value valueBits = getOrBitcast(input, dstBitsType);
    if (!valueBits)
      return {};
    Type computeTy = IntegerType::get(builder.getContext(), w);
    Value zero = builder.create<hw::ConstantOp>(loc, computeTy, 0);
    Value unknownBits = getOrBitcast(zero, dstBitsType);
    if (!unknownBits)
      return {};
    auto structTy = cast<hw::StructType>(dstType);
    return builder.create<hw::StructCreateOp>(loc, structTy,
                                              ValueRange{valueBits, unknownBits});
  }

  // 4-state struct -> 2-state integer (treat unknown bits as 0).
  if (!dstIsFour && srcIsFour) {
    int64_t w = hw::getBitWidth(srcBitsType);
    if (w <= 0)
      return {};
    auto structTy = cast<hw::StructType>(input.getType());
    Value valueBits =
        builder.create<hw::StructExtractOp>(loc, input,
                                            builder.getStringAttr("value"));
    Value unknownBits =
        builder.create<hw::StructExtractOp>(loc, input,
                                            builder.getStringAttr("unknown"));

    Type computeTy = IntegerType::get(builder.getContext(), w);
    valueBits = getOrBitcast(valueBits, computeTy);
    unknownBits = getOrBitcast(unknownBits, computeTy);
    if (!valueBits || !unknownBits)
      return {};

    Value ones = builder.create<hw::ConstantOp>(loc, computeTy, -1);
    Value notUnknown = builder.create<comb::XorOp>(loc, unknownBits, ones);
    Value masked = builder.create<comb::AndOp>(loc, valueBits, notUnknown);
    return getOrBitcast(masked, dstType);
  }

  return {};
}

struct InterfaceInfo {
  struct ModportInfo {
    SmallVector<StringAttr> signalOrder;
    DenseMap<StringAttr, sv::ModportDirection> directions;
  };

  InterfaceOp op;
  MLIRContext *context = nullptr;
  SmallVector<hw::StructType::FieldInfo> fields;
  DenseMap<StringAttr, unsigned> fieldIndex;
  DenseMap<StringAttr, ModportInfo> modports;

  hw::StructType getStructType() const {
    return hw::StructType::get(context, fields);
  }

  hw::StructType getModportStructType(StringAttr modportName) const {
    auto it = modports.find(modportName);
    if (it == modports.end())
      return hw::StructType();
    SmallVector<hw::StructType::FieldInfo> modFields;
    for (auto sig : it->second.signalOrder) {
      auto fieldIt = fieldIndex.find(sig);
      if (fieldIt == fieldIndex.end())
        continue;
      modFields.push_back(fields[fieldIt->second]);
    }
    return hw::StructType::get(context, modFields);
  }
};

struct InterfacePortPlan {
  InterfaceInfo *info = nullptr;
  StringAttr modportName;
  hw::StructType loweredType;
  bool needsInOut = false;
};

static Value buildZeroValue(OpBuilder &builder, Location loc, Type type);

static LogicalResult lowerInterfaceValueImpl(InterfaceInfo *info,
                                            StringAttr declaredModportName,
                                            Value oldValue, Value loweredBase,
                                            StringAttr directModport);

static bool modportAllowsWrite(InterfaceInfo *info, StringAttr modportName) {
  if (!modportName)
    return false;
  auto mpIt = info->modports.find(modportName);
  if (mpIt == info->modports.end())
    return false;
  for (auto [sig, dir] : mpIt->second.directions) {
    (void)sig;
    if (dir != sv::ModportDirection::input)
      return true;
  }
  return false;
}

static bool interfaceValueHasWrite(Value ifaceVal) {
  SmallVector<Value, 4> worklist{ifaceVal};
  SmallPtrSet<Value, 8> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      if (isa<sv::AssignInterfaceSignalOp>(user))
        return true;
      if (auto modport = dyn_cast<sv::GetModportOp>(user))
        worklist.push_back(modport.getResult());
    }
  }
  return false;
}

static LogicalResult verifyReadableSignal(InterfaceInfo *info,
                                          StringAttr modportName,
                                          StringAttr signal, Operation *diagOp) {
  if (!modportName)
    return success();
  auto mpIt = info->modports.find(modportName);
  if (mpIt == info->modports.end())
    return success();
  auto dirIt = mpIt->second.directions.find(signal);
  if (dirIt == mpIt->second.directions.end())
    return success();
  if (dirIt->second == sv::ModportDirection::output) {
    diagOp->emitOpError()
        << "cannot read signal " << signal.getValue() << " from modport "
        << modportName.getValue() << " because it is declared as output";
    return failure();
  }
  return success();
}

static LogicalResult verifyWritableSignal(InterfaceInfo *info,
                                          StringAttr modportName,
                                          StringAttr signal, Operation *diagOp) {
  if (!modportName)
    return success();
  auto mpIt = info->modports.find(modportName);
  if (mpIt == info->modports.end())
    return success();
  auto dirIt = mpIt->second.directions.find(signal);
  if (dirIt == mpIt->second.directions.end())
    return success();
  if (dirIt->second == sv::ModportDirection::input) {
    diagOp->emitOpError()
        << "cannot write signal " << signal.getValue() << " from modport "
        << modportName.getValue() << " because it is declared as input";
    return failure();
  }
  return success();
}

class LowerInterfacesPass
    : public circt::sv::impl::LowerInterfacesBase<LowerInterfacesPass> {
public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hw::HWDialect, comb::CombDialect, llhd::LLHDDialect,
                    sv::SVDialect>();
  }
  void runOnOperation() override;

  FailureOr<InterfaceInfo *> lookupInterfaceInfo(InterfaceType type,
                                                 hw::HWModuleOp mod);
  FailureOr<InterfaceInfo *> lookupInterfaceInfo(ModportType type,
                                                 hw::HWModuleOp mod,
                                                 StringAttr &modportName);

  DenseMap<StringAttr, InterfaceInfo> interfaces;
  DenseMap<Operation *, SmallVector<InterfacePortPlan>> plans;
  DenseMap<Value, Value> loweredInterfaceInstances;
  bool encounteredFailure = false;

  FailureOr<Value>
  getOrCreateLoweredInterfaceInstance(sv::InterfaceInstanceOp inst,
                                      hw::HWModuleOp parentModule);

	private:
  LogicalResult populateInterfaceInfo(ModuleOp module);
  LogicalResult buildPlans(hw::InstanceGraph &graph);
  LogicalResult lowerModules(hw::InstanceGraph &graph);
  LogicalResult lowerInterfaceInstances(ModuleOp module);
  LogicalResult lowerUnrealizedInterfaceCasts(ModuleOp module);
  LogicalResult lowerInterfaceHandleCasts(ModuleOp module);

  uint32_t nextInterfaceSigId = 0;
};

// Port conversion that rewrites interface/modport ports to lowered structs.
class InterfacePortConversion : public hw::PortConversion {
public:
  InterfacePortConversion(hw::PortConverterImpl &converter,
                          hw::PortInfo origPort, LowerInterfacesPass &pass,
                          InterfacePortPlan &plan)
      : PortConversion(converter, origPort), pass(pass), plan(plan) {}

  LogicalResult init() override;
  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;
  LogicalResult lowerInterfaceValue(Value oldValue, Value loweredBase,
                                    StringAttr directModport);

  LowerInterfacesPass &pass;
  InterfacePortPlan &plan;
  hw::PortInfo loweredPort;
};

class InterfacePortConversionBuilder : public hw::PortConversionBuilder {
public:
  InterfacePortConversionBuilder(hw::PortConverterImpl &converter,
                                 LowerInterfacesPass &pass,
                                 SmallVector<InterfacePortPlan> &plans)
      : PortConversionBuilder(converter), pass(pass), plans(plans) {}

  FailureOr<std::unique_ptr<hw::PortConversion>>
  build(hw::PortInfo port) override {
    if (port.argNum >= plans.size() || !plans[port.argNum].info)
      return PortConversionBuilder::build(port);
    return {std::make_unique<InterfacePortConversion>(converter, port, pass,
                                                      plans[port.argNum])};
  }

private:
  LowerInterfacesPass &pass;
  SmallVector<InterfacePortPlan> &plans;
};

LogicalResult InterfacePortConversion::init() {
  if (!plan.info)
    return success();
  if (!body)
    return success();
  BlockArgument arg = body->getArgument(origPort.argNum);
  if (isa<InterfaceType>(arg.getType()) || isa<ModportType>(arg.getType()))
    return success();
  return emitError(arg.getLoc(),
                   "expected interface-typed argument prior to lowering");
}

void InterfacePortConversion::buildInputSignals() {
  auto dir = plan.needsInOut ? hw::ModulePort::Direction::InOut
                             : hw::ModulePort::Direction::Input;
  Value newValue = converter.createNewInput(origPort, "", plan.loweredType,
                                            loweredPort, dir);

  if (!body)
    return;

  BlockArgument oldArg = body->getArgument(origPort.argNum);
  if (failed(lowerInterfaceValue(oldArg, newValue, plan.modportName))) {
    pass.encounteredFailure = true;
    return;
  }
  if (!oldArg.use_empty())
    oldArg.replaceAllUsesWith(newValue);
}

void InterfacePortConversion::buildOutputSignals() {
  if (!plan.info)
    return;
  Type outType =
      plan.needsInOut ? Type(hw::InOutType::get(plan.loweredType))
                      : Type(plan.loweredType);

  Value outputValue;
  if (body) {
    Operation *terminator = body->getTerminator();
    Value oldValue = terminator->getOperand(origPort.argNum);
    Type oldTy = oldValue.getType();

    if (isa<InterfaceType>(oldTy) || isa<ModportType>(oldTy)) {
      terminator->emitOpError()
          << "interface output " << origPort.name.getValue()
          << " is not lowered inside module body";
      pass.encounteredFailure = true;
      return;
    }

    if (oldTy != outType) {
      terminator->emitOpError()
          << "expected lowered output type " << outType
          << " for interface port " << origPort.name.getValue() << " but got "
          << oldTy;
      pass.encounteredFailure = true;
      return;
    }

    terminator->setOperand(origPort.argNum, oldValue);
    outputValue = oldValue;
  }

  converter.createNewOutput(origPort, "", outType, outputValue, loweredPort);
}

void InterfacePortConversion::mapInputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  (void)newResults;
  Location loc = inst->getLoc();
  Type expectedType = plan.needsInOut
                          ? Type(hw::InOutType::get(plan.loweredType))
                          : Type(plan.loweredType);

  Value mapped = instValue;

  // Allow connecting an inout<struct> value to an input struct port by
  // implicitly reading it.
  if (!plan.needsInOut) {
    if (auto inout = dyn_cast<hw::InOutType>(mapped.getType()))
      if (inout.getElementType() == expectedType)
        mapped = llhd::PrbOp::create(b, loc, mapped);
  }

  // If the operand is still an interface value (e.g. a local
  // `sv.interface.instance`), materialize a lowered storage signal on-demand.
  if (mapped.getType() != expectedType &&
      (isa<InterfaceType>(mapped.getType()) || isa<ModportType>(mapped.getType()))) {
    Value base = getInterfaceBase(mapped);
    if (auto ifaceInst = base.getDefiningOp<sv::InterfaceInstanceOp>()) {
      hw::HWModuleOp parentModule = inst->getParentOfType<hw::HWModuleOp>();
      if (!parentModule) {
        inst->emitOpError("cannot lower interface instance outside hw.module");
        pass.encounteredFailure = true;
      } else {
        auto lowered =
            pass.getOrCreateLoweredInterfaceInstance(ifaceInst, parentModule);
        if (succeeded(lowered)) {
          Value loweredInOut = *lowered;
          if (expectedType == loweredInOut.getType()) {
            mapped = loweredInOut;
          } else if (!plan.needsInOut) {
            if (auto loweredInoutType =
                    dyn_cast<hw::InOutType>(loweredInOut.getType())) {
              Type elemType = loweredInoutType.getElementType();
              if (elemType == expectedType) {
                mapped = llhd::PrbOp::create(b, loc, loweredInOut);
              } else if (auto fullStruct = dyn_cast<hw::StructType>(elemType);
                         fullStruct && isa<hw::StructType>(expectedType)) {
                // Build a subset struct from the full interface bundle.
                Value fullValue = llhd::PrbOp::create(b, loc, loweredInOut);
                auto expectedStruct = cast<hw::StructType>(expectedType);
                SmallVector<Value> elems;
                elems.reserve(expectedStruct.getElements().size());
                for (auto field : expectedStruct.getElements())
                  elems.push_back(
                      hw::StructExtractOp::create(b, loc, fullValue, field.name));
                mapped =
                    hw::StructCreateOp::create(b, loc, expectedStruct, elems);
              }
            }
          }
        } else {
          pass.encounteredFailure = true;
        }
      }
    }
  }

  if (mapped.getType() != expectedType) {
    inst->emitOpError("expected operand type ")
        << expectedType << " after interface lowering, got "
        << instValue.getType();
    pass.encounteredFailure = true;
    mapped = mlir::UnrealizedConversionCastOp::create(b, loc, expectedType,
                                                      ValueRange{})
                 .getResult(0);
  }

  newOperands[loweredPort.argNum] = mapped;
}

void InterfacePortConversion::mapOutputSignals(
    OpBuilder &, Operation *, Value instValue, SmallVectorImpl<Value> &,
    ArrayRef<Backedge> newResults) {
  if (!plan.info)
    return;
  instValue.replaceAllUsesWith(newResults[loweredPort.argNum]);
}

static LogicalResult lowerInterfaceValueImpl(InterfaceInfo *info,
                                            StringAttr declaredModportName,
                                            Value oldValue, Value loweredBase,
                                            StringAttr directModport) {
  SmallVector<std::tuple<Value, Value, StringAttr>> worklist;
  worklist.emplace_back(oldValue, loweredBase, directModport);
  SmallVector<Operation *> toErase;

  while (!worklist.empty()) {
    auto [ifaceVal, baseValue, modportName] = worklist.pop_back_val();
    for (Operation *user :
         llvm::make_early_inc_range(ifaceVal.getUsers())) {
      if (auto modport = dyn_cast<sv::GetModportOp>(user)) {
        worklist.emplace_back(modport.getResult(), baseValue,
                              modport.getFieldAttr().getAttr());
        toErase.push_back(modport);
        continue;
      }

      if (auto read = dyn_cast<sv::ReadInterfaceSignalOp>(user)) {
        StringAttr fieldAttr = read.getSignalNameAttr().getAttr();
        if (failed(verifyReadableSignal(info, declaredModportName, fieldAttr,
                                        read)) ||
            failed(verifyReadableSignal(info, modportName, fieldAttr, read)))
          return failure();
        ImplicitLocOpBuilder builder(read.getLoc(), read);
        Value replacement;
        if (auto inout = dyn_cast<hw::InOutType>(baseValue.getType())) {
          auto structType = dyn_cast<hw::StructType>(inout.getElementType());
          if (!structType)
            return read.emitOpError()
                   << "expected lowered interface to wrap a struct, got "
                   << baseValue.getType();
          Value fieldHandle = builder.create<sv::StructFieldInOutOp>(
              baseValue, fieldAttr);
          replacement = llhd::PrbOp::create(builder, read.getLoc(), fieldHandle);
        } else if (llvm::isa<hw::StructType>(baseValue.getType())) {
          replacement =
              builder.create<hw::StructExtractOp>(baseValue, fieldAttr);
        } else {
          return read.emitOpError()
                 << "expected lowered interface value to be a struct or "
                    "inout struct, got "
                 << baseValue.getType();
        }
        if (replacement.getType() != read.getType()) {
          Value converted =
              materializeTwoFourStateConversion(builder, read.getLoc(),
                                                replacement, read.getType());
          if (!converted)
            return read.emitOpError()
                   << "unsupported interface read type conversion from "
                   << replacement.getType() << " to " << read.getType();
          replacement = converted;
        }
        read.replaceAllUsesWith(replacement);
        read.erase();
        continue;
      }

      if (auto assign = dyn_cast<sv::AssignInterfaceSignalOp>(user)) {
        StringAttr fieldAttr = assign.getSignalNameAttr().getAttr();
        if (failed(verifyWritableSignal(info, declaredModportName, fieldAttr,
                                        assign)) ||
            failed(verifyWritableSignal(info, modportName, fieldAttr, assign)))
          return failure();
        auto inoutType = dyn_cast<hw::InOutType>(baseValue.getType());
        if (!inoutType ||
            !llvm::isa<hw::StructType>(inoutType.getElementType()))
          return assign.emitOpError()
                 << "cannot assign to flattened interface port because it is "
                    "not lowered to an inout";
        ImplicitLocOpBuilder builder(assign.getLoc(), assign);
        Value fieldHandle =
            builder.create<sv::StructFieldInOutOp>(baseValue, fieldAttr);
        Value rhs = assign.getRhs();
        Type fieldType = cast<hw::InOutType>(fieldHandle.getType()).getElementType();
        if (rhs.getType() != fieldType) {
          Value converted =
              materializeTwoFourStateConversion(builder, assign.getLoc(), rhs,
                                                fieldType);
          if (!converted)
            return assign.emitOpError()
                   << "unsupported interface assign type conversion from "
                   << rhs.getType() << " to " << fieldType;
          rhs = converted;
        }
        bool inProceduralRegion = isInProceduralRegion(assign);
        // Prefer LLHD drives over SV assign ops so downstream passes (Arc/LLVM)
        // never see residual SV dialect statements.
        //
        // - Structural interface bindings (e.g. interface port hookups) are
        //   modeled as epsilon-delayed continuous drives.
        // - Procedural writes (e.g. inside `llhd.process` / always blocks) are
        //   modeled as delta-delayed drives to preserve scheduling semantics.
        auto delay = llhd::ConstantTimeOp::create(
            builder, assign.getLoc(), 0, "ns",
            /*delta=*/inProceduralRegion ? 1 : 0,
            /*epsilon=*/inProceduralRegion ? 0 : 1);
        llhd::DrvOp::create(builder, assign.getLoc(), fieldHandle, rhs, delay,
                            Value{});
        assign.erase();
        continue;
      }

      // Allow interface handles to flow through `unrealized_conversion_cast`.
      // This is used by higher-level shims (e.g. virtual interfaces stored in
      // runtime tables) to cast interface values to opaque pointer-like types
      // such as `moore.chandle`, and back.
      if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(user)) {
        ImplicitLocOpBuilder builder(cast.getLoc(), cast);
        auto repl = mlir::UnrealizedConversionCastOp::create(
            builder, cast.getLoc(), cast.getResultTypes(), ValueRange{baseValue});
        for (auto [oldRes, newRes] :
             llvm::zip_equal(cast.getResults(), repl.getResults()))
          oldRes.replaceAllUsesWith(newRes);
        toErase.push_back(cast);
        continue;
      }

      return user->emitOpError("unsupported interface use during lowering");
    }
  }

  for (Operation *op : toErase)
    op->erase();
  return success();
}

LogicalResult InterfacePortConversion::lowerInterfaceValue(
    Value oldValue, Value loweredBase, StringAttr directModport) {
  return lowerInterfaceValueImpl(plan.info, plan.modportName, oldValue,
                                 loweredBase, directModport);
}

LogicalResult LowerInterfacesPass::populateInterfaceInfo(ModuleOp module) {
  for (auto iface : module.getOps<InterfaceOp>()) {
    InterfaceInfo info;
    info.op = iface;
    info.context = iface.getContext();
    for (auto sig : iface.getOps<InterfaceSignalOp>()) {
      hw::StructType::FieldInfo field;
      field.name = sig.getSymNameAttr();
      field.type = sig.getType();
      info.fieldIndex[field.name] = info.fields.size();
      info.fields.push_back(field);
    }
    for (auto modport : iface.getOps<InterfaceModportOp>()) {
      InterfaceInfo::ModportInfo mpInfo;
      for (Attribute attr : modport.getPortsAttr()) {
        auto port = cast<ModportStructAttr>(attr);
        auto sig = port.getSignal().getAttr();
        auto dirAttr = port.getDirection();
        mpInfo.signalOrder.push_back(sig);
        mpInfo.directions[sig] =
            dirAttr ? dirAttr.getValue() : sv::ModportDirection::inout;
      }
      info.modports[modport.getSymNameAttr()] = std::move(mpInfo);
    }
    auto [it, inserted] =
        interfaces.try_emplace(iface.getSymNameAttr(), std::move(info));
    if (!inserted)
      return iface.emitOpError()
             << "duplicate interface definition named " << iface.getSymName();
  }
  return success();
}

FailureOr<InterfaceInfo *>
LowerInterfacesPass::lookupInterfaceInfo(InterfaceType type,
                                         hw::HWModuleOp mod) {
  auto symName = type.getInterface().getAttr();
  auto it = interfaces.find(symName);
  if (it == interfaces.end()) {
    mod.emitOpError() << "references unknown interface " << symName;
    return failure();
  }
  InterfaceInfo &info = it->second;
  if (info.fields.empty()) {
    mod.emitOpError() << "interface " << info.op.getSymName()
                      << " has no signals; lowering is not meaningful";
    return failure();
  }
  return &info;
}

FailureOr<InterfaceInfo *>
LowerInterfacesPass::lookupInterfaceInfo(ModportType type, hw::HWModuleOp mod,
                                         StringAttr &modportName) {
  SymbolRefAttr modportRef = type.getModport();
  modportName = modportRef.getLeafReference();
  auto symName = modportRef.getRootReference();
  auto it = interfaces.find(symName);
  if (it == interfaces.end()) {
    mod.emitOpError() << "references unknown interface " << symName;
    return failure();
  }
  InterfaceInfo &info = it->second;
  if (info.fields.empty()) {
    mod.emitOpError() << "interface " << info.op.getSymName()
                      << " has no signals; lowering is not meaningful";
    return failure();
  }
  return &info;
}

LogicalResult LowerInterfacesPass::buildPlans(hw::InstanceGraph &graph) {
  auto initPlansForModule = [&](hw::HWModuleOp mod) -> LogicalResult {
    auto &modulePlans = plans[mod.getOperation()];
    if (!modulePlans.empty())
      return success();
    ModulePortInfo ports(mod.getPortList());
    modulePlans.resize(ports.size());
    for (auto [idx, port] : llvm::enumerate(ports)) {
      if (auto ifaceType = dyn_cast<InterfaceType>(port.type)) {
        auto ifaceInfo = lookupInterfaceInfo(ifaceType, mod);
        if (failed(ifaceInfo))
          return failure();
        modulePlans[idx].info = *ifaceInfo;
        modulePlans[idx].loweredType = (*ifaceInfo)->getStructType();
      } else if (auto modportType = dyn_cast<ModportType>(port.type)) {
        StringAttr modportName;
        auto ifaceInfo = lookupInterfaceInfo(modportType, mod, modportName);
        if (failed(ifaceInfo))
          return failure();
        modulePlans[idx].info = *ifaceInfo;
        modulePlans[idx].modportName = modportName;
        modulePlans[idx].loweredType =
            (*ifaceInfo)->getModportStructType(modportName);
        if (!modulePlans[idx].loweredType)
          return mod.emitOpError()
                 << "modport " << modportName
                 << " does not reference any signals";
        if (modportAllowsWrite(*ifaceInfo, modportName))
          modulePlans[idx].needsInOut = true;
      }
    }
    return success();
  };

  for (auto *node : graph)
    if (auto mod = dyn_cast<hw::HWModuleOp>(*node->getModule()))
      if (failed(initPlansForModule(mod)))
        return failure();

  bool changed = true;
  while (changed) {
    changed = false;
    if (failed(graph.walkPostOrder([&](igraph::InstanceGraphNode &node) {
          auto mod = dyn_cast<hw::HWModuleOp>(*node.getModule());
          if (!mod)
            return success();
          auto &modulePlans = plans[mod.getOperation()];
          ModulePortInfo ports(mod.getPortList());
          SmallVector<PortInfo> inputPorts(ports.getInputs().begin(),
                                           ports.getInputs().end());
          Block *body = mod.getBodyBlock();

          if (body) {
            for (auto [idx, port] : llvm::enumerate(inputPorts)) {
              auto &plan = modulePlans[port.argNum];
              if (!plan.info)
                continue;
              if (!plan.needsInOut &&
                  interfaceValueHasWrite(body->getArgument(idx))) {
                plan.needsInOut = true;
                changed = true;
              }
            }
          }

          // Propagate child write requirements to operands.
          for (auto *record : node) {
            auto inst = dyn_cast<hw::InstanceOp>(record->getInstance());
            if (!inst)
              continue;
            auto targetMod =
                dyn_cast<hw::HWModuleOp>(*record->getTarget()->getModule());
            if (!targetMod)
              continue;
            ModulePortInfo calleePorts(targetMod.getPortList());
            SmallVector<PortInfo> calleeInputs(calleePorts.getInputs().begin(),
                                               calleePorts.getInputs().end());
            auto &calleePlans = plans[targetMod.getOperation()];
            for (auto [opIdx, operand] : llvm::enumerate(inst.getOperands())) {
              auto &calleePort = calleeInputs[opIdx];
              auto &calleePlan = calleePlans[calleePort.argNum];
              if (!calleePlan.info || !calleePlan.needsInOut)
                continue;
              Value base = getInterfaceBase(operand);
              auto barg = dyn_cast<BlockArgument>(base);
              if (!barg || barg.getOwner() != inst->getBlock())
                continue;
              auto &parentPort = inputPorts[barg.getArgNumber()];
              auto &parentPlan = modulePlans[parentPort.argNum];
              if (parentPlan.info && !parentPlan.needsInOut) {
                parentPlan.needsInOut = true;
                changed = true;
              }
            }
          }
          return success();
        })))
      return failure();
  }

  return success();
}

LogicalResult LowerInterfacesPass::lowerModules(hw::InstanceGraph &graph) {
  if (failed(graph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
        igraph::ModuleOpInterface module = node.getModule();
        if (!module)
          return success();
        auto mutableModule =
            dyn_cast<hw::HWMutableModuleLike>(module.getOperation());
        if (!mutableModule)
          return success();
        auto it = plans.find(mutableModule.getOperation());
        if (it == plans.end())
          return success();
        auto &modulePlans = it->second;
        hw::PortConverter<InterfacePortConversionBuilder> converter(
            graph, mutableModule, *this, modulePlans);
        return converter.run();
      })))
    return failure();
  if (encounteredFailure)
    return failure();
  return success();
}

FailureOr<Value> LowerInterfacesPass::getOrCreateLoweredInterfaceInstance(
    sv::InterfaceInstanceOp inst, hw::HWModuleOp parentModule) {
  Value key = inst.getResult();
  auto it = loweredInterfaceInstances.find(key);
  if (it != loweredInterfaceInstances.end())
    return it->second;

  auto ifaceInfo = lookupInterfaceInfo(inst.getInterfaceType(), parentModule);
  if (failed(ifaceInfo))
    return failure();

  InterfaceInfo *info = *ifaceInfo;
  hw::StructType structTy = info->getStructType();

  OpBuilder builder(inst);
  builder.setInsertionPointAfter(inst);
  Value init = buildZeroValue(builder, inst.getLoc(), structTy);
  if (!init) {
    inst.emitOpError("unsupported interface instance type for lowering: ")
        << structTy;
    return failure();
  }

  Value lowered =
      llhd::SignalOp::create(builder, inst.getLoc(),
                             builder.getStringAttr(inst.getName()), init);
  if (auto sigOp = lowered.getDefiningOp<llhd::SignalOp>()) {
    if (!sigOp->hasAttr(kArcilatorSigIdAttr))
      sigOp->setAttr(kArcilatorSigIdAttr,
                     builder.getI32IntegerAttr(nextInterfaceSigId++));
  }
  loweredInterfaceInstances[key] = lowered;
  return lowered;
}

static Value buildZeroValue(OpBuilder &builder, Location loc, Type type) {
  int64_t width = hw::getBitWidth(type);
  if (width < 0)
    return {};
  Value constZero = hw::ConstantOp::create(builder, loc, APInt(width, 0));
  return builder.createOrFold<hw::BitcastOp>(loc, type, constZero);
}

LogicalResult LowerInterfacesPass::lowerInterfaceInstances(ModuleOp module) {
  for (auto mod : module.getOps<hw::HWModuleOp>()) {
    if (!mod.getBodyBlock())
      continue;

    SmallVector<sv::InterfaceInstanceOp> instances;
    for (auto inst : mod.getOps<sv::InterfaceInstanceOp>())
      instances.push_back(inst);
    if (instances.empty())
      continue;

    for (auto inst : instances) {
      auto ifaceInfo = lookupInterfaceInfo(inst.getInterfaceType(), mod);
      if (failed(ifaceInfo))
        return failure();
      auto loweredOr = getOrCreateLoweredInterfaceInstance(inst, mod);
      if (failed(loweredOr))
        return failure();
      Value lowered = *loweredOr;

      if (failed(lowerInterfaceValueImpl(*ifaceInfo, /*declaredModportName=*/{},
                                         inst.getResult(), lowered,
                                         /*directModport=*/{})))
        return failure();

      if (!inst->use_empty()) {
        inst.emitOpError("unsupported uses remain after instance lowering");
        return failure();
      }
      inst.erase();
    }
  }
  return success();
}

LogicalResult
LowerInterfacesPass::lowerUnrealizedInterfaceCasts(ModuleOp module) {
  SmallVector<mlir::UnrealizedConversionCastOp> ifaceCasts;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    Type resultTy = cast.getResult(0).getType();
    if (isa<sv::InterfaceType>(resultTy) || isa<sv::ModportType>(resultTy))
      ifaceCasts.push_back(cast);
  });
  if (ifaceCasts.empty())
    return success();

  for (auto cast : ifaceCasts) {
    Type resultTy = cast.getResult(0).getType();
    InterfaceInfo *info = nullptr;
    StringAttr declaredModport;

    if (auto ifaceTy = dyn_cast<sv::InterfaceType>(resultTy)) {
      auto symName = ifaceTy.getInterface().getAttr();
      auto it = interfaces.find(symName);
      if (it == interfaces.end())
        return cast.emitOpError() << "references unknown interface " << symName;
      info = &it->second;
    } else if (auto modportTy = dyn_cast<sv::ModportType>(resultTy)) {
      SymbolRefAttr modportRef = modportTy.getModport();
      declaredModport = modportRef.getLeafReference();
      auto symName = modportRef.getRootReference();
      auto it = interfaces.find(symName);
      if (it == interfaces.end())
        return cast.emitOpError() << "references unknown interface " << symName;
      info = &it->second;
    } else {
      continue;
    }

    if (!info || info->fields.empty())
      return cast.emitOpError()
             << "interface lowering is not meaningful for empty interface";

    // Treat these casts as producing a reference-like interface handle.
    // Materialize a lowered inout<struct> base from the opaque operand.
    hw::StructType structTy = info->getStructType();
    Type loweredType = hw::InOutType::get(structTy);

    OpBuilder builder(cast);
    builder.setInsertionPointAfter(cast);
    Value loweredBase =
        mlir::UnrealizedConversionCastOp::create(builder, cast.getLoc(),
                                                 loweredType,
                                                 ValueRange{cast.getOperand(0)})
            .getResult(0);

    if (failed(lowerInterfaceValueImpl(
            info, declaredModport, cast.getResult(0), loweredBase,
            /*directModport=*/declaredModport)))
      return failure();

    if (!cast.getResult(0).use_empty())
      return cast.emitOpError("unsupported uses remain after lowering");
    cast.erase();
  }
  return success();
}

LogicalResult LowerInterfacesPass::lowerInterfaceHandleCasts(ModuleOp module) {
  SmallVector<mlir::UnrealizedConversionCastOp> casts;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    if (!isa<IntegerType>(cast.getResult(0).getType()))
      return;
    casts.push_back(cast);
  });
  if (casts.empty())
    return success();

  for (auto cast : casts) {
    auto resultIntTy = dyn_cast<IntegerType>(cast.getResult(0).getType());
    if (!resultIntTy)
      continue;
    unsigned width = resultIntTy.getWidth();
    if (width == 0 || width > 64)
      continue;

    Value input = cast.getOperand(0);
    if (auto inout = dyn_cast<hw::InOutType>(input.getType()))
      input = getInterfaceBase(input);

    // Strip trivial conversion casts and locate a lowered interface signal.
    Value base = input;
    while (auto inner = base.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (inner.getNumOperands() != 1 || inner.getNumResults() != 1)
        break;
      base = inner.getOperand(0);
    }

    auto sigOp = base.getDefiningOp<llhd::SignalOp>();
    if (!sigOp)
      continue;

    auto sigIdAttr = sigOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr);
    if (!sigIdAttr) {
      OpBuilder b(sigOp);
      sigIdAttr = b.getI32IntegerAttr(nextInterfaceSigId++);
      sigOp->setAttr(kArcilatorSigIdAttr, sigIdAttr);
    }

    int64_t sigId = sigIdAttr.getInt();
    APInt sigBits(width, static_cast<uint64_t>(sigId), /*isSigned=*/false);
    ImplicitLocOpBuilder b(cast.getLoc(), cast);
    Value handle = hw::ConstantOp::create(b, cast.getLoc(), sigBits);
    if (handle.getType() != cast.getResult(0).getType())
      handle =
          b.createOrFold<hw::BitcastOp>(cast.getLoc(), cast.getResult(0).getType(),
                                        handle);
    cast.getResult(0).replaceAllUsesWith(handle);
    cast.erase();
  }

  return success();
}

void LowerInterfacesPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  if (failed(populateInterfaceInfo(module)))
    return signalPassFailure();
  if (failed(buildPlans(instanceGraph)))
    return signalPassFailure();
  if (failed(lowerModules(instanceGraph)))
    return signalPassFailure();
  if (failed(lowerInterfaceInstances(module)))
    return signalPassFailure();
  if (failed(lowerUnrealizedInterfaceCasts(module)))
    return signalPassFailure();
  if (failed(lowerInterfaceHandleCasts(module)))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}
