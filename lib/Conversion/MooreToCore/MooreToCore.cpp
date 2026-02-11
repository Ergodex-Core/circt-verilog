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
#include "circt/Dialect/LLHD/LLHDOps.h"
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
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"

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

/// Cache for identified structs and field GEP paths keyed by class symbol.
struct ClassTypeCache {
  struct ClassStructInfo {
    LLVM::LLVMStructType classBody;

    // field name -> GEP path inside ident (excluding the leading pointer index)
    DenseMap<StringRef, SmallVector<unsigned, 2>> propertyPath;

    // TODO: Add classVTable in here.
    /// Record/overwrite the field path to a single property for a class.
    void setFieldPath(StringRef propertyName, ArrayRef<unsigned> path) {
      this->propertyPath[propertyName] =
          SmallVector<unsigned, 2>(path.begin(), path.end());
    }

    /// Lookup the full GEP path for a (class, field).
    std::optional<ArrayRef<unsigned>>
    getFieldPath(StringRef propertySym) const {
      if (auto prop = this->propertyPath.find(propertySym);
          prop != this->propertyPath.end())
        return ArrayRef<unsigned>(prop->second);
      return std::nullopt;
    }
  };

  /// Record the identified struct body for a class.
  /// Implicitly finalizes the class to struct conversion.
  void setClassInfo(SymbolRefAttr classSym, const ClassStructInfo &info) {
    auto &dst = classToStructMap[classSym];
    dst = info;
  }

  /// Lookup the identified struct body for a class.
  std::optional<ClassStructInfo> getStructInfo(SymbolRefAttr classSym) const {
    if (auto it = classToStructMap.find(classSym); it != classToStructMap.end())
      return it->second;
    return std::nullopt;
  }

private:
  // Keyed by the SymbolRefAttr of the class.
  // Kept private so all accesses are done with helpers which preserve
  // invariants
  DenseMap<Attribute, ClassStructInfo> classToStructMap;
};

/// Ensure we have `declare i8* @malloc(i64)` (opaque ptr prints as !llvm.ptr).
static LLVM::LLVMFuncOp getOrCreateMalloc(ModuleOp mod, OpBuilder &b) {
  if (auto f = mod.lookupSymbol<LLVM::LLVMFuncOp>("malloc"))
    return f;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(mod.getBody());

  auto i64Ty = IntegerType::get(mod.getContext(), 64);
  auto ptrTy = LLVM::LLVMPointerType::get(mod.getContext()); // opaque pointer
  auto fnTy = LLVM::LLVMFunctionType::get(ptrTy, {i64Ty}, false);

  auto fn = LLVM::LLVMFuncOp::create(b, mod.getLoc(), "malloc", fnTy);
  // Link this in from somewhere else.
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

/// Helper function to create an opaque LLVM Struct Type which corresponds
/// to the sym
static LLVM::LLVMStructType getOrCreateOpaqueStruct(MLIRContext *ctx,
                                                    SymbolRefAttr className) {
  return LLVM::LLVMStructType::getIdentified(ctx, className.getRootReference());
}

static LogicalResult resolveClassStructBody(ClassDeclOp op,
                                            TypeConverter const &typeConverter,
                                            ClassTypeCache &cache) {

  auto classSym = SymbolRefAttr::get(op.getSymNameAttr());
  auto structInfo = cache.getStructInfo(classSym);
  if (structInfo)
    // We already have a resolved class struct body.
    return success();

  // Otherwise we need to resolve.
  ClassTypeCache::ClassStructInfo structBody;
  SmallVector<Type> structBodyMembers;

  // Base-first (prefix) layout for single inheritance.
  unsigned derivedStartIdx = 0;

  if (auto baseClass = op.getBaseAttr()) {

    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto *opSym = mod.lookupSymbol(baseClass);
    auto classDeclOp = cast<ClassDeclOp>(opSym);

    if (failed(resolveClassStructBody(classDeclOp, typeConverter, cache)))
      return failure();

    // Process base class' struct layout first
    auto baseClassStruct = cache.getStructInfo(baseClass);
    structBodyMembers.push_back(baseClassStruct->classBody);
    derivedStartIdx = 1;

    // Inherit base field paths with a leading 0.
    for (auto &kv : baseClassStruct->propertyPath) {
      SmallVector<unsigned, 2> path;
      path.push_back(0); // into base subobject
      path.append(kv.second.begin(), kv.second.end());
      structBody.setFieldPath(kv.first, path);
    }
  }

  // Properties in source order.
  unsigned iterator = derivedStartIdx;
  auto &block = op.getBody().front();
  for (Operation &child : block) {
    if (auto prop = dyn_cast<ClassPropertyDeclOp>(child)) {
      Type mooreTy = prop.getPropertyType();
      Type llvmTy = typeConverter.convertType(mooreTy);
      if (!llvmTy)
        return prop.emitOpError()
               << "failed to convert property type " << mooreTy;

      structBodyMembers.push_back(llvmTy);

      // Derived field path: either {i} or {1+i} if base is present.
      SmallVector<unsigned, 2> path{iterator};
      structBody.setFieldPath(prop.getSymName(), path);
      ++iterator;
    }
  }

  // TODO: Handle vtable generation over ClassMethodDeclOp here.
  auto llvmStructTy = getOrCreateOpaqueStruct(op.getContext(), classSym);
  // Empty structs may be kept opaque
  if (!structBodyMembers.empty() &&
      failed(llvmStructTy.setBody(structBodyMembers, false)))
    return op.emitOpError() << "Failed to set LLVM Struct body";

  structBody.classBody = llvmStructTy;
  cache.setClassInfo(classSym, structBody);

  return success();
}

/// Convenience overload that looks up ClassDeclOp
static LogicalResult resolveClassStructBody(ModuleOp mod, SymbolRefAttr op,
                                            TypeConverter const &typeConverter,
                                            ClassTypeCache &cache) {
  auto classDeclOp = cast<ClassDeclOp>(*mod.lookupSymbol(op));
  return resolveClassStructBody(classDeclOp, typeConverter, cache);

static constexpr llvm::StringLiteral kFourValuedValueField = "value";
static constexpr llvm::StringLiteral kFourValuedUnknownField = "unknown";

static bool isFourValuedIntType(Type type) {
  auto structTy = dyn_cast<hw::StructType>(type);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (elements[0].name != kFourValuedValueField ||
      elements[1].name != kFourValuedUnknownField)
    return false;
  return elements[0].type == elements[1].type &&
         isa<IntegerType>(elements[0].type);
}

static Value getFourValuedValue(OpBuilder &builder, Location loc,
                                Value fourValued) {
  return builder.createOrFold<hw::StructExtractOp>(
      loc, fourValued, builder.getStringAttr(kFourValuedValueField));
}

static Value getFourValuedUnknown(OpBuilder &builder, Location loc,
                                  Value fourValued) {
  return builder.createOrFold<hw::StructExtractOp>(
      loc, fourValued, builder.getStringAttr(kFourValuedUnknownField));
}

static Value createFourValued(OpBuilder &builder, Location loc, Type type,
                              Value valueBits, Value unknownBits) {
  auto structTy = dyn_cast<hw::StructType>(type);
  if (!structTy)
    return {};
  return builder.create<hw::StructCreateOp>(loc, structTy,
                                            ValueRange{valueBits, unknownBits});
}

static std::optional<FVInt>
tryEvaluateMooreIntValue(Value value,
                         DenseMap<Value, std::optional<FVInt>> &cache,
                         SmallDenseSet<Value, 8> &visiting);

static std::string formatFourValuedConstant(const FVInt &value,
                                            moore::IntFormat format,
                                            unsigned minWidth,
                                            moore::IntAlign alignment,
                                            moore::IntPadding padding) {
  auto bitWidth = value.getBitWidth();
  bool leftJustify = alignment == moore::IntAlign::Left;
  char padChar = padding == moore::IntPadding::Zero ? '0' : ' ';

  auto applyPadding = [&](StringRef text) {
    if (minWidth == 0 || text.size() >= minWidth)
      return text.str();
    std::string out;
    out.reserve(minWidth);
    unsigned padCount = minWidth - text.size();
    if (!leftJustify)
      out.append(padCount, padChar);
    out.append(text.begin(), text.end());
    if (leftJustify)
      out.append(padCount, padChar);
    return out;
  };

  if (bitWidth == 0) {
    if (format == moore::IntFormat::Decimal)
      return applyPadding("0");
    return "";
  }

  switch (format) {
  case moore::IntFormat::Decimal: {
    if (value.hasUnknown()) {
      char digit = value.isAllZ() ? 'z' : 'x';
      return applyPadding(StringRef(&digit, 1));
    }

    auto apint = value.toAPInt(false);
    SmallString<16> strBuf;
    apint.toString(strBuf, /*Radix=*/10, /*Signed=*/false);
    return applyPadding(strBuf);
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
    // Trim leading zeros for minimum-width formatting.
    size_t firstNonZero = out.find_first_not_of('0');
    if (firstNonZero == std::string::npos)
      firstNonZero = out.size() - 1;
    StringRef trimmed(out.data() + firstNonZero, out.size() - firstNonZero);
    return applyPadding(trimmed);
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
    // Trim leading zeros for minimum-width formatting.
    size_t firstNonZero = out.find_first_not_of('0');
    if (firstNonZero == std::string::npos)
      firstNonZero = out.size() - 1;
    StringRef trimmed(out.data() + firstNonZero, out.size() - firstNonZero);
    return applyPadding(trimmed);
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
    // Trim leading zeros for minimum-width formatting.
    size_t firstNonZero = out.find_first_not_of('0');
    if (firstNonZero == std::string::npos)
      firstNonZero = out.size() - 1;
    StringRef trimmed(out.data() + firstNonZero, out.size() - firstNonZero);
    return applyPadding(trimmed);
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
  auto intType = dyn_cast<IntegerType>(value.getType());
  if (!intType)
    return {};
  uint32_t intWidth = intType.getWidth();
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
    if (!isa<llhd::RefType>(value.getType()))
      return value;
    return llhd::ProbeOp::create(rewriter, loc, value);
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
    //
    // `always_comb` and `always_latch` have an implicit sensitivity list and
    // must suspend until an observed value changes.
    //
    // Some SV constructs (notably concurrent assertions lowered to
    // `verif.clocked_*`) currently appear as `moore.procedure always` without
    // an explicit `moore.wait_*`. Lowering those directly to an LLHD process
    // loop would create a zero-delay infinite loop and hang cycle-driven
    // simulation. Treat such procedures like `always_comb`: observe their
    // external dependencies and insert an implicit wait point.
    bool needsImplicitWait = false;
    if (op.getKind() == ProcedureKind::AlwaysComb ||
        op.getKind() == ProcedureKind::AlwaysLatch)
      needsImplicitWait = true;
    if (!needsImplicitWait &&
        (op.getKind() == ProcedureKind::Always ||
         op.getKind() == ProcedureKind::AlwaysFF)) {
      bool hasExplicitWait = false;
      op.getBody().walk([&](Operation *inner) {
        if (isa<WaitEventOp, WaitDelayOp>(inner))
          hasExplicitWait = true;
      });
      needsImplicitWait = !hasExplicitWait;
    }

    SmallVector<Value> observedValues;
    if (needsImplicitWait) {
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

    // Add special handling for procedures with an implicit sensitivity list.
    // These run once at simulation startup and then implicitly wait for any of
    // the values they access to change before running again. To implement this,
    // we create another basic block that contains the implicit wait, and make
    // all `moore.return` ops branch to that wait block instead of immediately
    // jumping back up to the body.
    if (needsImplicitWait) {
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
    // have now been materialized in the wait block as `valuesBefore`. Convert
    // them to HW value types and:
    //   1) Observe them so `llhd.wait` wakes up on changes, and
    //   2) Pass them as dest operands into the check block so edge detection can
    //      use the true pre-wait values even after later scheduler lowering.
    SmallVector<Value> beforeValues;
    SmallVector<Value> beforeArgs;
    for (auto value : valuesBefore) {
      auto type = typeConverter->convertType(value.getType());
      if (!type || !hw::isHWValueType(type))
        continue;
      Value converted = value;
      if (converted.getType() != type) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(waitOp);
        converted = typeConverter->materializeTargetConversion(rewriter, loc, type,
                                                               converted);
      }
      beforeValues.push_back(converted);
      beforeArgs.push_back(checkBlock->addArgument(type, value.getLoc()));
    }
    if (!beforeValues.empty()) {
      waitOp.getObservedMutable().append(beforeValues);
      waitOp.getDestOperandsMutable().append(beforeValues);
    }

    // Collect a list of all detect ops and inline the `wait_event` body into
    // the check block.
    SmallVector<DetectEventOp> detectOps(op.getBody().getOps<DetectEventOp>());
    rewriter.inlineBlockBefore(&op.getBody().front(), checkBlock,
                               checkBlock->end());
    rewriter.eraseOp(op);

    // Helper function to detect if a certain change occurred between a value
    // captured before the `llhd.wait` and the value after.
    auto computeTrigger = [&](Value before, Value after, Edge edge) -> Value {
      assert(before.getType() == after.getType() &&
             "mismatched types after conversion");
      Value trueVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 1));

      auto lowerToLsb = [&](Value v) -> Value {
        auto intTy = dyn_cast<IntegerType>(v.getType());
        if (!intTy || intTy.getWidth() == 1)
          return v;
        return comb::ExtractOp::create(rewriter, loc, v, /*lowBit=*/0,
                                       /*width=*/1);
      };

      // Two-valued values are plain integers.
      if (auto intTy = dyn_cast<IntegerType>(before.getType())) {
        Value beforeInt = before;
        Value afterInt = after;
        if (edge != Edge::AnyChange && intTy.getWidth() != 1) {
          beforeInt = lowerToLsb(beforeInt);
          afterInt = lowerToLsb(afterInt);
        }

        if (edge == Edge::AnyChange)
          return comb::ICmpOp::create(rewriter, loc, ICmpPredicate::ne, beforeInt,
                                      afterInt, true);

        SmallVector<Value> disjuncts;
        if (edge == Edge::PosEdge || edge == Edge::BothEdges) {
          Value notOld =
              comb::XorOp::create(rewriter, loc, beforeInt, trueVal, true);
          disjuncts.push_back(
              comb::AndOp::create(rewriter, loc, notOld, afterInt, true));
        }
        if (edge == Edge::NegEdge || edge == Edge::BothEdges) {
          Value notCurr =
              comb::XorOp::create(rewriter, loc, afterInt, trueVal, true);
          disjuncts.push_back(
              comb::AndOp::create(rewriter, loc, beforeInt, notCurr, true));
        }
        return rewriter.createOrFold<comb::OrOp>(loc, disjuncts, true);
      }

      // Four-valued values are represented as `{value, unknown}`.
      auto structTy = dyn_cast<hw::StructType>(before.getType());
      if (!structTy)
        return {};

      auto valueField = rewriter.getStringAttr(kFourValuedValueField);
      auto unknownField = rewriter.getStringAttr(kFourValuedUnknownField);
      Value beforeValue =
          rewriter.createOrFold<hw::StructExtractOp>(loc, before, valueField);
      Value beforeUnknown =
          rewriter.createOrFold<hw::StructExtractOp>(loc, before, unknownField);
      Value afterValue =
          rewriter.createOrFold<hw::StructExtractOp>(loc, after, valueField);
      Value afterUnknown =
          rewriter.createOrFold<hw::StructExtractOp>(loc, after, unknownField);
      auto fieldTy = dyn_cast<IntegerType>(beforeValue.getType());
      if (!fieldTy || beforeUnknown.getType() != fieldTy ||
          afterValue.getType() != fieldTy || afterUnknown.getType() != fieldTy)
        return {};

      // Any-change event: detect changes in the canonical 4-state value where
      // unknown bits mask out the corresponding value bits.
      if (edge == Edge::AnyChange) {
        Value ones = hw::ConstantOp::create(rewriter, loc,
                                            APInt(fieldTy.getWidth(), -1,
                                                  /*isSigned=*/true));
        Value beforeKnownMask =
            comb::XorOp::create(rewriter, loc, beforeUnknown, ones, true);
        Value afterKnownMask =
            comb::XorOp::create(rewriter, loc, afterUnknown, ones, true);
        Value beforeCanon = comb::AndOp::create(rewriter, loc, beforeValue,
                                                beforeKnownMask, true);
        Value afterCanon = comb::AndOp::create(rewriter, loc, afterValue,
                                               afterKnownMask, true);
        Value unknownChanged = comb::ICmpOp::create(
            rewriter, loc, ICmpPredicate::ne, beforeUnknown, afterUnknown, true);
        Value valueChanged = comb::ICmpOp::create(
            rewriter, loc, ICmpPredicate::ne, beforeCanon, afterCanon, true);
        return rewriter.createOrFold<comb::OrOp>(
            loc, ValueRange{unknownChanged, valueChanged}, true);
      }

      // 9.4.2 IEEE 1800-2017: Edge events consider only the LSB.
      Value beforeValueBit = beforeValue;
      Value beforeUnknownBit = beforeUnknown;
      Value afterValueBit = afterValue;
      Value afterUnknownBit = afterUnknown;
      if (fieldTy.getWidth() != 1) {
        beforeValueBit = lowerToLsb(beforeValueBit);
        beforeUnknownBit = lowerToLsb(beforeUnknownBit);
        afterValueBit = lowerToLsb(afterValueBit);
        afterUnknownBit = lowerToLsb(afterUnknownBit);
      }

      // Posedge: after==1 and before!=1 (before can be 0, X, or Z).
      // Negedge: after==0 and before!=0 (before can be 1, X, or Z).
      Value afterKnown =
          comb::XorOp::create(rewriter, loc, afterUnknownBit, trueVal, true);
      Value beforeNot1 = rewriter.createOrFold<comb::OrOp>(
          loc, ValueRange{beforeUnknownBit,
                          comb::XorOp::create(rewriter, loc, beforeValueBit,
                                              trueVal, true)},
          true);
      Value beforeNot0 = rewriter.createOrFold<comb::OrOp>(
          loc, ValueRange{beforeUnknownBit, beforeValueBit}, true);

      SmallVector<Value> disjuncts;
      if (edge == Edge::PosEdge || edge == Edge::BothEdges) {
        Value lhs =
            comb::AndOp::create(rewriter, loc, afterKnown, afterValueBit, true);
        disjuncts.push_back(
            comb::AndOp::create(rewriter, loc, lhs, beforeNot1, true));
      }
      if (edge == Edge::NegEdge || edge == Edge::BothEdges) {
        Value notAfter =
            comb::XorOp::create(rewriter, loc, afterValueBit, trueVal, true);
        Value lhs =
            comb::AndOp::create(rewriter, loc, afterKnown, notAfter, true);
        disjuncts.push_back(
            comb::AndOp::create(rewriter, loc, lhs, beforeNot0, true));
      }
      return rewriter.createOrFold<comb::OrOp>(loc, disjuncts, true);
    };

    // Convert all `detect_event` ops into a check for the corresponding event
    // between the value before and after the `llhd.wait`. The "before" value
    // has been collected into `valuesBefore` in the "wait" block; the "after"
    // value corresponds to the detect op's input.
    SmallVector<Value> triggers;
    for (auto [detectOp, before] : llvm::zip(detectOps, beforeArgs)) {
      if (!allDetectsAreAnyChange) {
        rewriter.setInsertionPoint(detectOp);
        auto afterType = before.getType();
        Value after = detectOp.getInput();
        if (after.getType() != afterType)
          after = typeConverter->materializeTargetConversion(rewriter, loc,
                                                             afterType, after);
        auto trigger = computeTrigger(before, after, detectOp.getEdge());
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

static Value createZeroValue(Type type, Location loc,
                             ConversionPatternRewriter &rewriter) {
  // Handle pointers.
  if (isa<mlir::LLVM::LLVMPointerType>(type))
    return mlir::LLVM::ZeroOp::create(rewriter, loc, type);

  // Handle time values.
  if (isa<llhd::TimeType>(type)) {
    auto timeAttr =
        llhd::TimeAttr::get(type.getContext(), 0U, llvm::StringRef("ns"), 0, 0);
    return llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
  }

  // Handle real values.
  if (auto floatType = dyn_cast<FloatType>(type)) {
    auto floatAttr = rewriter.getFloatAttr(floatType, 0.0);
    return mlir::arith::ConstantOp::create(rewriter, loc, floatAttr);
  }

  // Handle dynamic strings
  if (auto strType = dyn_cast<sim::DynamicStringType>(type))
    return sim::StringConstantOp::create(rewriter, loc, strType, "");

  // Handle queues
  if (auto queueType = dyn_cast<sim::QueueType>(type))
    return sim::QueueEmptyOp::create(rewriter, loc, queueType);

  // Otherwise try to create a zero integer and bitcast it to the result type.
  int64_t width = hw::getBitWidth(type);
  if (width == -1)
    return {};

  // TODO: Once the core dialects support four-valued integers, this code
  // will additionally need to generate an all-X value for four-valued
  // variables.
  Value constZero = hw::ConstantOp::create(rewriter, loc, APInt(width, 0));
  return rewriter.createOrFold<hw::BitcastOp>(loc, type, constZero);
}

struct ClassPropertyRefOpConversion
    : public OpConversionPattern<circt::moore::ClassPropertyRefOp> {
  ClassPropertyRefOpConversion(TypeConverter &tc, MLIRContext *ctx,
                               ClassTypeCache &cache)
      : OpConversionPattern(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(circt::moore::ClassPropertyRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Convert result type; we expect !llhd.ref<someT>.
    Type dstTy = getTypeConverter()->convertType(op.getPropertyRef().getType());
    // Operand is a !llvm.ptr
    Value instRef = adaptor.getInstance();

    // Resolve identified struct from cache.
    auto classRefTy =
        cast<circt::moore::ClassHandleType>(op.getInstance().getType());
    SymbolRefAttr classSym = classRefTy.getClassSym();
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    if (failed(resolveClassStructBody(mod, classSym, *typeConverter, cache)))
      return rewriter.notifyMatchFailure(op,
                                         "Could not resolve class struct for " +
                                             classSym.getRootReference().str());

    auto structInfo = cache.getStructInfo(classSym);
    assert(structInfo && "class struct info must exist");
    auto structTy = structInfo->classBody;

    // Look up cached GEP path for the property.
    auto propSym = op.getProperty();
    auto pathOpt = structInfo->getFieldPath(propSym);
    if (!pathOpt)
      return rewriter.notifyMatchFailure(op,
                                         "no GEP path for property " + propSym);

    auto i32Ty = IntegerType::get(ctx, 32);
    SmallVector<Value> idxVals;
    for (unsigned idx : *pathOpt)
      idxVals.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(idx)));

    // GEP to the field (opaque ptr mode requires element type).
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto gep =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, structTy, instRef, idxVals);

    // Wrap pointer back to !llhd.ref<someT>.
    Value fieldRef = UnrealizedConversionCastOp::create(rewriter, loc, dstTy,
                                                        gep.getResult())
                         .getResult(0);

    rewriter.replaceOp(op, fieldRef);
    return success();
  }

private:
  ClassTypeCache &cache;
};

struct ClassUpcastOpConversion : public OpConversionPattern<ClassUpcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassUpcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect lowered types like !llvm.ptr
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    Type srcTy = adaptor.getInstance().getType();

    if (!dstTy)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // If the types are already identical (opaque pointer mode), just forward.
    if (dstTy == srcTy && isa<LLVM::LLVMPointerType>(srcTy)) {
      rewriter.replaceOp(op, adaptor.getInstance());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "Upcast applied to non-opaque pointers!");
  }
};

/// moore.class.new lowering: heap-allocate storage for the class object.
struct ClassNewOpConversion : public OpConversionPattern<ClassNewOp> {
  ClassNewOpConversion(TypeConverter &tc, MLIRContext *ctx,
                       ClassTypeCache &cache)
      : OpConversionPattern<ClassNewOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto handleTy = cast<ClassHandleType>(op.getResult().getType());
    auto sym = handleTy.getClassSym();

    ModuleOp mod = op->getParentOfType<ModuleOp>();

    if (failed(resolveClassStructBody(mod, sym, *typeConverter, cache)))
      return op.emitError() << "Could not resolve class struct for " << sym;

    auto structTy = cache.getStructInfo(sym)->classBody;

    DataLayout dl(mod);
    // DataLayout::getTypeSize gives a byte count for LLVM types.
    uint64_t byteSize = dl.getTypeSize(structTy);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto cSize = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(byteSize));

    // Get or declare malloc and call it.
    auto mallocFn = getOrCreateMalloc(mod, rewriter);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx); // opaque pointer result
    auto call =
        LLVM::CallOp::create(rewriter, loc, TypeRange{ptrTy},
                             SymbolRefAttr::get(mallocFn), ValueRange{cSize});

    // Replace the new op with the malloc pointer (no cast needed with opaque
    // ptrs).
    rewriter.replaceOp(op, call.getResult());
    return success();
  }

private:
  ClassTypeCache &cache; // shared, owned by the pass
};

struct ClassDeclOpConversion : public OpConversionPattern<ClassDeclOp> {
  ClassDeclOpConversion(TypeConverter &tc, MLIRContext *ctx,
                        ClassTypeCache &cache)
      : OpConversionPattern<ClassDeclOp>(tc, ctx), cache(cache) {}

  LogicalResult
  matchAndRewrite(ClassDeclOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(resolveClassStructBody(op, *typeConverter, cache)))
      return failure();
    // The declaration itself is a no-op
    rewriter.eraseOp(op);
    return success();
  }

private:
  ClassTypeCache &cache; // shared, owned by the pass
};

struct VariableOpConversion : public OpConversionPattern<VariableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(VariableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.empty())
      return rewriter.notifyMatchFailure(op.getLoc(), "invalid variable type");

    // Special handling for four-valued integer variables: store the signal as
    // a `!llhd.ref<!hw.struct<value, unknown>>` but expose it as two
    // independent subsignal references (`!llhd.ref<value>` and
    // `!llhd.ref<unknown>`) so that bit slicing works using existing
    // `llhd.sig.*` ops.
    if (resultTypes.size() == 2) {
      auto valueRefTy = dyn_cast<llhd::RefType>(resultTypes[0]);
      auto unknownRefTy = dyn_cast<llhd::RefType>(resultTypes[1]);
      if (!valueRefTy || !unknownRefTy)
        return failure();
      auto iTy = dyn_cast<IntegerType>(valueRefTy.getNestedType());
      if (!iTy || unknownRefTy.getNestedType() != iTy)
        return failure();

      SmallVector<hw::StructType::FieldInfo> fields;
      fields.push_back(
          {StringAttr::get(op->getContext(), kFourValuedValueField), iTy});
      fields.push_back(
          {StringAttr::get(op->getContext(), kFourValuedUnknownField), iTy});
      auto elemStructTy = hw::StructType::get(op->getContext(), fields);
      auto signalTy = llhd::RefType::get(elemStructTy);

      Value init = adaptor.getInitial();
      if (!init) {
        Value valueZero = hw::ConstantOp::create(rewriter, loc, iTy, 0);
        Value unknownOnes = hw::ConstantOp::create(rewriter, loc, iTy, -1);
        init = createFourValued(rewriter, loc, elemStructTy, valueZero,
                                unknownOnes);
        if (!init)
          return failure();
      }

      Value signal = rewriter.create<llhd::SignalOp>(loc, signalTy,
                                                     op.getNameAttr(), init);
      Value valueRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, signal, rewriter.getStringAttr(kFourValuedValueField));
      Value unknownRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, signal, rewriter.getStringAttr(kFourValuedUnknownField));
      SmallVector<SmallVector<Value>> replacements;
      replacements.push_back({valueRef, unknownRef});
      rewriter.replaceOpWithMultiple(op, std::move(replacements));
      return success();
    }

    if (resultTypes.size() != 1)
      return failure();
    Type resultType = resultTypes.front();

    // Determine the initial value of the signal.
    Value init = adaptor.getInitial();
    if (!init) {
      auto refTy = dyn_cast<llhd::RefType>(resultType);
      if (!refTy)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "unexpected converted variable type");

      Type elementType = refTy.getNestedType();
      if (isa<hw::StringType>(elementType)) {
        rewriter.getContext()->getOrLoadDialect<sv::SVDialect>();
        init = rewriter.create<sv::ConstantStrOp>(
            loc, cast<hw::StringType>(elementType), rewriter.getStringAttr(""));
      } else {
        init = createZeroValue(elementType, loc, rewriter);
        if (!init)
          return failure();
      }
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

struct UArrayCmpOpConversion : public OpConversionPattern<UArrayCmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UArrayCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsTy = dyn_cast<hw::ArrayType>(lhs.getType());
    auto rhsTy = dyn_cast<hw::ArrayType>(rhs.getType());
    if (!lhsTy || !rhsTy || lhsTy != rhsTy)
      return rewriter.notifyMatchFailure(loc, "expected hw::ArrayType operands");

    auto elementType = lhsTy.getElementType();
    bool isIntElem = isa<IntegerType>(elementType);
    bool isFourValuedElem = isFourValuedIntType(elementType);
    if (!isIntElem && !isFourValuedElem)
      return rewriter.notifyMatchFailure(
          loc, "unsupported unpacked array element type");

    unsigned numElems = lhsTy.getNumElements();
    unsigned idxWidth = std::max<unsigned>(llvm::Log2_64_Ceil(numElems), 1);
    auto idxType = rewriter.getIntegerType(idxWidth);
    auto i1Type = rewriter.getI1Type();

    Value allEq = hw::ConstantOp::create(rewriter, loc, i1Type, 1);
    for (unsigned i = 0; i < numElems; ++i) {
      Value idx = hw::ConstantOp::create(rewriter, loc, idxType, i);
      Value lhsElem = rewriter.create<hw::ArrayGetOp>(loc, lhs, idx);
      Value rhsElem = rewriter.create<hw::ArrayGetOp>(loc, rhs, idx);

      Value elemEq;
      if (isFourValuedElem) {
        Value lhsVal = getFourValuedValue(rewriter, loc, lhsElem);
        Value lhsUnk = getFourValuedUnknown(rewriter, loc, lhsElem);
        Value rhsVal = getFourValuedValue(rewriter, loc, rhsElem);
        Value rhsUnk = getFourValuedUnknown(rewriter, loc, rhsElem);
        Value valEq = rewriter.create<comb::ICmpOp>(loc, i1Type,
                                                    comb::ICmpPredicate::eq,
                                                    lhsVal, rhsVal);
        Value unkEq = rewriter.create<comb::ICmpOp>(loc, i1Type,
                                                    comb::ICmpPredicate::eq,
                                                    lhsUnk, rhsUnk);
        elemEq = rewriter.create<comb::AndOp>(loc, valEq, unkEq);
      } else {
        elemEq = rewriter.create<comb::ICmpOp>(
            loc, i1Type, comb::ICmpPredicate::eq, lhsElem, rhsElem);
      }

      allEq = rewriter.create<comb::AndOp>(loc, allEq, elemEq);
    }

    Value result = allEq;
    if (op.getPredicate() == moore::UArrayCmpPredicate::ne) {
      Value zero = hw::ConstantOp::create(rewriter, loc, i1Type, 0);
      result = rewriter.create<comb::ICmpOp>(loc, i1Type,
                                             comb::ICmpPredicate::eq, result,
                                             zero);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NetOpConversion : public OpConversionPattern<NetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.empty())
      return rewriter.notifyMatchFailure(loc, "invalid net type");

    // Four-valued nets are stored as `!llhd.ref<!hw.struct<value, unknown>>`
    // but exposed as two independent subsignal references (`!llhd.ref<value>`
    // and `!llhd.ref<unknown>}`) so bit slicing works.
    if (resultTypes.size() == 2) {
      auto valueRefTy = dyn_cast<llhd::RefType>(resultTypes[0]);
      auto unknownRefTy = dyn_cast<llhd::RefType>(resultTypes[1]);
      if (!valueRefTy || !unknownRefTy)
        return failure();
      auto iTy = dyn_cast<IntegerType>(valueRefTy.getNestedType());
      if (!iTy || unknownRefTy.getNestedType() != iTy)
        return failure();

      SmallVector<hw::StructType::FieldInfo> fields;
      fields.push_back(
          {StringAttr::get(op->getContext(), kFourValuedValueField), iTy});
      fields.push_back(
          {StringAttr::get(op->getContext(), kFourValuedUnknownField), iTy});
      auto elemStructTy = hw::StructType::get(op->getContext(), fields);
      auto signalTy = llhd::RefType::get(elemStructTy);

      // Many sv-tests simulation cases declare constant-driven wires via a net
      // initializer (e.g. `wire [N:0] a = <const>;`) and then immediately
      // compute expressions from those nets inside a `final` block. Later in the
      // pipeline we intentionally drop LLHD inout storage semantics for
      // `llhd.signal`/`llhd.drv`/`llhd.prb` in graph regions; if we always
      // initialize nets to Z and model the initializer as an epsilon-timed drive
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
        if (assignedValue.getType() == elemStructTy &&
            isTriviallyConstant(assignedValue))
          init = assignedValue;

      if (!init) {
        // Undriven nets default to Z.
        Value valueOnes = hw::ConstantOp::create(rewriter, loc, iTy, -1);
        Value unknownOnes = hw::ConstantOp::create(rewriter, loc, iTy, -1);
        init = createFourValued(rewriter, loc, elemStructTy, valueOnes,
                                unknownOnes);
        if (!init)
          return failure();
      }

      Value signal = rewriter.create<llhd::SignalOp>(loc, signalTy,
                                                     op.getNameAttr(), init);
      Value valueRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, signal, rewriter.getStringAttr(kFourValuedValueField));
      Value unknownRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, signal, rewriter.getStringAttr(kFourValuedUnknownField));
      SmallVector<SmallVector<Value>> replacements;
      replacements.push_back({valueRef, unknownRef});
      rewriter.replaceOpWithMultiple(op, std::move(replacements));

      if (auto assignedValue = adaptor.getAssignment()) {
        if (assignedValue == init)
          return success();
        auto timeAttr = llhd::TimeAttr::get(signalTy.getContext(), 0U,
                                            llvm::StringRef("ns"), 0, 1);
        auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
        llhd::DriveOp::create(rewriter, loc, signal, assignedValue, time,
                              Value{});
      }
      return success();
    }

    if (resultTypes.size() != 1)
      return failure();
    Type resultType = resultTypes.front();
    auto elementType = cast<llhd::RefType>(resultType).getNestedType();
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

    Value init =
        createInitialValue(op.getKind(), rewriter, loc, width, elementType);
    if (auto assignedValue = adaptor.getAssignment())
      if (assignedValue.getType() == elementType &&
          isTriviallyConstant(assignedValue))
        init = assignedValue;
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
      llhd::DriveOp::create(rewriter, loc, signal, assignedValue, time,
                            Value{});
    }

    return success();
  }

  static mlir::Value createInitialValue(NetKind kind,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, int64_t width,
                                        Type elementType) {
    // TODO: Once the core dialects support four-valued integers, this code
    // will additionally need to generate an all-X value for four-valued nets.
    //
    // If no driver is connected to a net, its value shall be high-impedance (z)
    // unless the net is a trireg, in which case it shall hold the previously
    // driven value.
    //
    // See IEEE 1800-2017 § 6.6 "Net types".
    auto theInt = [&] {
      if (kind == NetKind::Supply1 || kind == NetKind::Tri1)
        return APInt::getAllOnes(width);
      return APInt::getZero(width);
    }();
    auto theConst = hw::ConstantOp::create(rewriter, loc, theInt);
    return rewriter.createOrFold<hw::BitcastOp>(loc, elementType, theConst);
  }
};

// moore.global_variable -> llhd.global_signal
static LogicalResult convert(GlobalVariableOp op,
                             GlobalVariableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TypeConverter &typeConverter) {
  auto type = typeConverter.convertType(op.getType());
  auto sig = llhd::GlobalSignalOp::create(rewriter, op.getLoc(),
                                          op.getSymNameAttr(), type);
  sig.getInitRegion().takeBody(op.getInitRegion());
  rewriter.eraseOp(op);
  return success();
}

// moore.get_global_variable -> llhd.get_global_signal
static LogicalResult convert(GetGlobalVariableOp op,
                             GetGlobalVariableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TypeConverter &typeConverter) {
  auto type = typeConverter.convertType(op.getType());
  rewriter.replaceOpWithNewOp<llhd::GetGlobalSignalOp>(op, type,
                                                       op.getGlobalNameAttr());
  return success();
}

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct ConstantOpConv : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    // Two-valued constants lower to an integer constant.
    if (isa<IntegerType>(resultType)) {
      auto value = op.getValue().toAPInt(false);
      auto type = cast<IntegerType>(resultType);
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, value));
      return success();
    }

    // Four-valued constants lower to `{value, unknown}`.
    if (isFourValuedIntType(resultType)) {
      auto fv = op.getValue();
      auto structTy = cast<hw::StructType>(resultType);
      auto elemTy = cast<IntegerType>(structTy.getElements()[0].type);
      auto valueAttr = rewriter.getIntegerAttr(elemTy, fv.getRawValue());
      auto unknownAttr = rewriter.getIntegerAttr(elemTy, fv.getRawUnknown());
      auto fieldsAttr = rewriter.getArrayAttr({valueAttr, unknownAttr});
      rewriter.replaceOpWithNewOp<hw::AggregateConstantOp>(op, structTy,
                                                           fieldsAttr);
      return success();
    }

    return failure();
  }
};

struct ConstantRealOpConv : public OpConversionPattern<ConstantRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantRealOp op, OpAdaptor adaptor,
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
    rewriter.replaceOpWithNewOp<llhd::CurrentTimeOp>(op);
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
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value timeInt =
        rewriter.create<llhd::TimeToIntOp>(op.getLoc(), adaptor.getInput());

    if (isa<IntegerType>(resultType)) {
      rewriter.replaceOp(op, timeInt);
      return success();
    }

    if (!isFourValuedIntType(resultType))
      return failure();

    auto intTy = cast<IntegerType>(
        cast<hw::StructType>(resultType).getElements()[0].type);
    Value unknownZero = hw::ConstantOp::create(rewriter, op.getLoc(), intTy, 0);
    Value result =
        createFourValued(rewriter, op.getLoc(), resultType, timeInt, unknownZero);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LogicToTimeOpConv : public OpConversionPattern<LogicToTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LogicToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();

    if (isFourValuedIntType(input.getType())) {
      Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
      Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);
      auto intTy = cast<IntegerType>(valueBits.getType());
      Value allOnes = hw::ConstantOp::create(rewriter, op.getLoc(), intTy, -1);
      Value knownMask =
          rewriter.createOrFold<comb::XorOp>(op.getLoc(), unknownBits, allOnes);
      Value knownValue =
          rewriter.createOrFold<comb::AndOp>(op.getLoc(), valueBits, knownMask);
      input = knownValue;
    }

    rewriter.replaceOpWithNewOp<llhd::IntToTimeOp>(op, input);
    return success();
  }
};

struct ConstantStringOpConv : public OpConversionPattern<ConstantStringOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstantStringOp op, OpAdaptor adaptor,
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
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    bool anyFourValued = false;
    for (Value v : adaptor.getValues())
      anyFourValued |= isFourValuedIntType(v.getType());

    if (!anyFourValued) {
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getValues());
      return success();
    }

    if (!isFourValuedIntType(resultType))
      return failure();

    SmallVector<Value> valueParts;
    SmallVector<Value> unknownParts;
    valueParts.reserve(adaptor.getValues().size());
    unknownParts.reserve(adaptor.getValues().size());

    for (Value v : adaptor.getValues()) {
      if (isFourValuedIntType(v.getType())) {
        valueParts.push_back(getFourValuedValue(rewriter, op.getLoc(), v));
        unknownParts.push_back(getFourValuedUnknown(rewriter, op.getLoc(), v));
      } else {
        auto intTy = cast<IntegerType>(v.getType());
        valueParts.push_back(v);
        unknownParts.push_back(
            hw::ConstantOp::create(rewriter, op.getLoc(), intTy, 0));
      }
    }

    Value valueConcat =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), valueParts);
    Value unknownConcat =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), unknownParts);
    Value result = createFourValued(rewriter, op.getLoc(), resultType,
                                    valueConcat, unknownConcat);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReplicateOpConversion : public OpConversionPattern<ReplicateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());

    Value input = adaptor.getValue();
    if (isa<IntegerType>(input.getType())) {
      rewriter.replaceOpWithNewOp<comb::ReplicateOp>(op, resultType, input);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);

    auto fieldTy =
        cast<IntegerType>(cast<hw::StructType>(resultType).getElements()[0].type);
    Value valueRep =
        rewriter.createOrFold<comb::ReplicateOp>(op.getLoc(), fieldTy, valueBits);
    Value unknownRep = rewriter.createOrFold<comb::ReplicateOp>(
        op.getLoc(), fieldTy, unknownBits);

    Value result = createFourValued(rewriter, op.getLoc(), resultType, valueRep,
                                    unknownRep);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
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

    if (isFourValuedIntType(inputType)) {
      if (!isFourValuedIntType(resultType))
        return failure();

      Value valueBits = getFourValuedValue(rewriter, op.getLoc(), adaptor.getInput());
      Value unknownBits =
          getFourValuedUnknown(rewriter, op.getLoc(), adaptor.getInput());

      int32_t inputWidth = valueBits.getType().getIntOrFloatBitWidth();
      int32_t resultWidth = cast<IntegerType>(
                                cast<hw::StructType>(resultType).getElements()[0]
                                    .type)
                                .getWidth();
      int32_t high = low + resultWidth;

      SmallVector<Value> valueConcatParts;
      SmallVector<Value> unknownConcatParts;
      if (low < 0) {
        int32_t padWidth = std::min(-low, resultWidth);
        valueConcatParts.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(padWidth, 0)));
        unknownConcatParts.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(padWidth, -1)));
      }

      if (low < inputWidth && high > 0) {
        int32_t lowIdx = std::max(low, 0);
        int32_t takeWidth =
            std::min(resultWidth, std::min(high, inputWidth) - lowIdx);
        Value midValue = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(), rewriter.getIntegerType(takeWidth), valueBits, lowIdx);
        Value midUnknown = rewriter.createOrFold<comb::ExtractOp>(
            op.getLoc(), rewriter.getIntegerType(takeWidth), unknownBits,
            lowIdx);
        valueConcatParts.push_back(midValue);
        unknownConcatParts.push_back(midUnknown);
      }

      int32_t diff = high - inputWidth;
      if (diff > 0) {
        valueConcatParts.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(diff, 0)));
        unknownConcatParts.push_back(hw::ConstantOp::create(
            rewriter, op.getLoc(), APInt(diff, -1)));
      }

      Value concatValue = rewriter.createOrFold<comb::ConcatOp>(
          op.getLoc(), valueConcatParts);
      Value concatUnknown = rewriter.createOrFold<comb::ConcatOp>(
          op.getLoc(), unknownConcatParts);
      Value result = createFourValued(rewriter, op.getLoc(), resultType,
                                      concatValue, concatUnknown);
      if (!result)
        return failure();
      rewriter.replaceOp(op, result);
      return success();
    }

    if (auto arrTy = dyn_cast<hw::ArrayType>(inputType)) {
      int32_t width = llvm::Log2_64_Ceil(arrTy.getNumElements());
      int32_t inputWidth = arrTy.getNumElements();

      if (auto resArrTy = dyn_cast<hw::ArrayType>(resultType);
          resArrTy && resArrTy != arrTy.getElementType()) {
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
  using OneToNOpAdaptor =
      typename OpConversionPattern<ExtractRefOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ExtractRefOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.empty())
      return failure();
    auto loc = op.getLoc();
    ValueRange input = adaptor.getInput();

    // Four-valued integer refs are represented as two independent subsignal
    // references: `{!llhd.ref<value>, !llhd.ref<unknown>}`.
    if (input.size() == 2) {
      if (resultTypes.size() != 2)
        return failure();

      Value valueRef = input[0];
      Value unknownRef = input[1];
      auto valueRefTy = dyn_cast<llhd::RefType>(valueRef.getType());
      if (!valueRefTy)
        return failure();

      int64_t width = hw::getBitWidth(valueRefTy.getNestedType());
      if (width == -1)
        return failure();

      Value lowBit = hw::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerType(llvm::Log2_64_Ceil(width)),
          adaptor.getLowBit());

      Value slicedValueRef = rewriter.create<llhd::SigExtractOp>(
          loc, resultTypes[0], valueRef, lowBit);
      Value slicedUnknownRef = rewriter.create<llhd::SigExtractOp>(
          loc, resultTypes[1], unknownRef, lowBit);
      SmallVector<SmallVector<Value>> replacements;
      replacements.push_back({slicedValueRef, slicedUnknownRef});
      rewriter.replaceOpWithMultiple(op, std::move(replacements));
      return success();
    }

    if (input.size() != 1)
      return failure();
    Value inputValue = input.front();
    auto inputRefTy = dyn_cast<llhd::RefType>(inputValue.getType());
    if (!inputRefTy)
      return failure();
    Type elemType = inputRefTy.getNestedType();

    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (resultTypes.size() != 1)
        return failure();
      int64_t width = hw::getBitWidth(elemType);
      if (width == -1)
        return failure();

      Value lowBit = hw::ConstantOp::create(
          rewriter, loc, rewriter.getIntegerType(llvm::Log2_64_Ceil(width)),
          adaptor.getLowBit());
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(op, resultTypes.front(),
                                                      inputValue,
                                                      lowBit);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(elemType)) {
      Value lowBit = hw::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerType(llvm::Log2_64_Ceil(arrType.getNumElements())),
          adaptor.getLowBit());

      // Extracting an element that is a four-valued integer yields a pair of
      // subsignal references.
      if (resultTypes.size() == 2) {
        Value elementRef =
            rewriter.create<llhd::SigArrayGetOp>(loc, inputValue, lowBit);
        Value valueRef = rewriter.create<llhd::SigStructExtractOp>(
            loc, elementRef, rewriter.getStringAttr(kFourValuedValueField));
        Value unknownRef = rewriter.create<llhd::SigStructExtractOp>(
            loc, elementRef,
            rewriter.getStringAttr(kFourValuedUnknownField));
        SmallVector<SmallVector<Value>> replacements;
        replacements.push_back({valueRef, unknownRef});
        rewriter.replaceOpWithMultiple(op, std::move(replacements));
        return success();
      }

      if (resultTypes.size() != 1)
        return failure();
      auto resultType = resultTypes.front();
      auto resultRefTy = dyn_cast<llhd::RefType>(resultType);
      if (!resultRefTy)
        return failure();

      // If the result type is not the same as the array's element type, then
      // it has to be a slice.
      if (arrType.getElementType() != resultRefTy.getNestedType()) {
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, inputValue, lowBit);
        return success();
      }

      rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, inputValue, lowBit);
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
      auto loc = op.getLoc();
      Value lowBit = adaptor.getLowBit();
      Value amountUnknownAny =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
      if (isFourValuedIntType(lowBit.getType())) {
        Value lowValue = getFourValuedValue(rewriter, loc, lowBit);
        Value lowUnknown = getFourValuedUnknown(rewriter, loc, lowBit);
        Value zeroVec = hw::ConstantOp::create(
            rewriter, loc, cast<IntegerType>(lowUnknown.getType()), 0);
        amountUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
            loc, comb::ICmpPredicate::ne, lowUnknown, zeroVec);
        lowBit = lowValue;
      }

      Value amount =
          adjustIntegerWidth(rewriter, lowBit, intType.getWidth(), loc);
      if (!amount)
        return failure();
      Value value = comb::ShrUOp::create(rewriter, loc, adaptor.getInput(),
                                         amount);

      if (!isa<IntegerType>(resultType))
        return failure();
      Value extracted =
          rewriter.createOrFold<comb::ExtractOp>(loc, resultType, value, 0);
      Value zero =
          hw::ConstantOp::create(rewriter, loc, cast<IntegerType>(resultType), 0);
      Value mux =
          rewriter.createOrFold<comb::MuxOp>(loc, amountUnknownAny, zero, extracted);
      rewriter.replaceOp(op, mux);
      return success();
    }

    if (isFourValuedIntType(inputType)) {
      if (!isFourValuedIntType(resultType))
        return failure();

      auto loc = op.getLoc();
      Value input = adaptor.getInput();
      Value valueBits = getFourValuedValue(rewriter, loc, input);
      Value unknownBits = getFourValuedUnknown(rewriter, loc, input);
      auto vecTy = cast<IntegerType>(valueBits.getType());

      Value lowBit = adaptor.getLowBit();
      Value amountUnknownAny =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
      if (isFourValuedIntType(lowBit.getType())) {
        Value lowValue = getFourValuedValue(rewriter, loc, lowBit);
        Value lowUnknown = getFourValuedUnknown(rewriter, loc, lowBit);
        Value zeroVec =
            hw::ConstantOp::create(rewriter, loc, cast<IntegerType>(lowUnknown.getType()), 0);
        amountUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
            loc, comb::ICmpPredicate::ne, lowUnknown, zeroVec);
        lowBit = lowValue;
      }

      Value amount = adjustIntegerWidth(rewriter, lowBit, vecTy.getWidth(), loc);
      if (!amount)
        return failure();
      Value valueShift =
          rewriter.createOrFold<comb::ShrUOp>(loc, valueBits, amount, false);
      Value unknownShift =
          rewriter.createOrFold<comb::ShrUOp>(loc, unknownBits, amount, false);

      auto outTy = cast<IntegerType>(
          cast<hw::StructType>(resultType).getElements()[0].type);
      Value outValue =
          rewriter.createOrFold<comb::ExtractOp>(loc, outTy, valueShift, 0);
      Value outUnknown =
          rewriter.createOrFold<comb::ExtractOp>(loc, outTy, unknownShift, 0);

      Value shifted =
          createFourValued(rewriter, loc, resultType, outValue, outUnknown);

      Value allX = createFourValued(
          rewriter, loc, resultType,
          hw::ConstantOp::create(rewriter, loc, outTy, 0),
          hw::ConstantOp::create(rewriter, loc, outTy, -1));
      if (!shifted || !allX)
        return failure();

      Value mux =
          rewriter.createOrFold<comb::MuxOp>(loc, amountUnknownAny, allX, shifted);
      rewriter.replaceOp(op, mux);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(inputType)) {
      auto loc = op.getLoc();
      unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getNumElements());
      Value lowBit = adaptor.getLowBit();
      Value idxUnknownAny =
          hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
      if (isFourValuedIntType(lowBit.getType())) {
        Value lowValue = getFourValuedValue(rewriter, loc, lowBit);
        Value lowUnknown = getFourValuedUnknown(rewriter, loc, lowBit);
        Value zeroVec = hw::ConstantOp::create(
            rewriter, loc, cast<IntegerType>(lowUnknown.getType()), 0);
        idxUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
            loc, comb::ICmpPredicate::ne, lowUnknown, zeroVec);
        lowBit = lowValue;
      }

      Value idx = adjustIntegerWidth(rewriter, lowBit, idxWidth, loc);
      if (!idx)
        return failure();

      if (isa<hw::ArrayType>(resultType)) {
        rewriter.replaceOpWithNewOp<hw::ArraySliceOp>(op, resultType,
                                                      adaptor.getInput(), idx);
        return success();
      }

      Value extracted = rewriter.createOrFold<hw::ArrayGetOp>(
          loc, adaptor.getInput(), idx);
      if (!extracted)
        return failure();

      Value lowered = extracted;
      if (isFourValuedIntType(extracted.getType())) {
        auto outTy =
            cast<IntegerType>(cast<hw::StructType>(extracted.getType())
                                  .getElements()[0]
                                  .type);
        Value allX = createFourValued(
            rewriter, loc, extracted.getType(),
            hw::ConstantOp::create(rewriter, loc, outTy, 0),
            hw::ConstantOp::create(rewriter, loc, outTy, -1));
        if (!allX)
          return failure();
        lowered =
            rewriter.createOrFold<comb::MuxOp>(loc, idxUnknownAny, allX, extracted);
      } else if (auto intResult = dyn_cast<IntegerType>(extracted.getType())) {
        Value zero = hw::ConstantOp::create(rewriter, loc, intResult, 0);
        lowered = rewriter.createOrFold<comb::MuxOp>(loc, idxUnknownAny, zero,
                                                     extracted);
      }

      rewriter.replaceOp(op, lowered);
      return success();
    }

    return failure();
  }
};

struct DynExtractRefOpConversion : public OpConversionPattern<DynExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor =
      typename OpConversionPattern<DynExtractRefOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(DynExtractRefOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: properly handle out-of-bounds accesses
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.empty())
      return failure();
    auto loc = op.getLoc();
    ValueRange input = adaptor.getInput();
    Value lowBit = adaptor.getLowBit().front();

    // Four-valued integer refs are represented as `{!llhd.ref<value>,
    // !llhd.ref<unknown>}`.
    if (input.size() == 2) {
      if (resultTypes.size() != 2)
        return failure();
      Value valueRef = input[0];
      Value unknownRef = input[1];
      auto valueRefTy = dyn_cast<llhd::RefType>(valueRef.getType());
      if (!valueRefTy)
        return failure();
      int64_t width = hw::getBitWidth(valueRefTy.getNestedType());
      if (width == -1)
        return failure();

      if (isFourValuedIntType(lowBit.getType()))
        lowBit = getFourValuedValue(rewriter, loc, lowBit);

      Value amount = adjustIntegerWidth(rewriter, lowBit,
                                        llvm::Log2_64_Ceil(width), loc);
      if (!amount)
        return failure();
      Value slicedValueRef = rewriter.create<llhd::SigExtractOp>(
          loc, resultTypes[0], valueRef, amount);
      Value slicedUnknownRef = rewriter.create<llhd::SigExtractOp>(
          loc, resultTypes[1], unknownRef, amount);
      SmallVector<SmallVector<Value>> replacements;
      replacements.push_back({slicedValueRef, slicedUnknownRef});
      rewriter.replaceOpWithMultiple(op, std::move(replacements));
      return success();
    }

    if (input.size() != 1)
      return failure();
    Value inputValue = input.front();
    auto inputRefTy = dyn_cast<llhd::RefType>(inputValue.getType());
    if (!inputRefTy)
      return failure();
    Type elemType = inputRefTy.getNestedType();

    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      if (resultTypes.size() != 1)
        return failure();
      int64_t width = hw::getBitWidth(elemType);
      if (width == -1)
        return failure();

      if (isFourValuedIntType(lowBit.getType()))
        lowBit = getFourValuedValue(rewriter, loc, lowBit);

      Value amount =
          adjustIntegerWidth(rewriter, lowBit,
                             llvm::Log2_64_Ceil(width), loc);
      if (!amount)
        return failure();
      rewriter.replaceOpWithNewOp<llhd::SigExtractOp>(
          op, resultTypes.front(), inputValue, amount);
      return success();
    }

    if (auto arrType = dyn_cast<hw::ArrayType>(elemType)) {
      if (isFourValuedIntType(lowBit.getType()))
        lowBit = getFourValuedValue(rewriter, loc, lowBit);
      Value idx = adjustIntegerWidth(
          rewriter, lowBit,
          llvm::Log2_64_Ceil(arrType.getNumElements()), loc);
      if (!idx)
        return failure();

      if (resultTypes.size() == 2) {
        Value elementRef =
            rewriter.create<llhd::SigArrayGetOp>(loc, inputValue, idx);
        Value valueRef = rewriter.create<llhd::SigStructExtractOp>(
            loc, elementRef, rewriter.getStringAttr(kFourValuedValueField));
        Value unknownRef = rewriter.create<llhd::SigStructExtractOp>(
            loc, elementRef,
            rewriter.getStringAttr(kFourValuedUnknownField));
        SmallVector<SmallVector<Value>> replacements;
        replacements.push_back({valueRef, unknownRef});
        rewriter.replaceOpWithMultiple(op, std::move(replacements));
        return success();
      }

      if (resultTypes.size() != 1)
        return failure();
      auto resultType = resultTypes.front();
      auto resultRefTy = dyn_cast<llhd::RefType>(resultType);
      if (!resultRefTy)
        return failure();
      if (isa<hw::ArrayType>(resultRefTy.getNestedType())) {
        rewriter.replaceOpWithNewOp<llhd::SigArraySliceOp>(
            op, resultType, inputValue, idx);
        return success();
      }

      rewriter.replaceOpWithNewOp<llhd::SigArrayGetOp>(op, inputValue, idx);
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
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(), resultTypes)) ||
        resultTypes.empty())
      return failure();
    auto loc = op.getLoc();

    // If the extracted field is a four-valued integer, expose it as
    // `{inout<value>, inout<unknown>}` by extracting the subsignals.
    if (resultTypes.size() == 2) {
      Value fieldInout = rewriter.create<llhd::SigStructExtractOp>(
          loc, adaptor.getInput(), adaptor.getFieldNameAttr());
      Value valueRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, fieldInout, rewriter.getStringAttr(kFourValuedValueField));
      Value unknownRef = rewriter.create<llhd::SigStructExtractOp>(
          loc, fieldInout, rewriter.getStringAttr(kFourValuedUnknownField));
      SmallVector<SmallVector<Value>> replacements;
      replacements.push_back({valueRef, unknownRef});
      rewriter.replaceOpWithMultiple(op, std::move(replacements));
      return success();
    }

    rewriter.replaceOpWithNewOp<llhd::SigStructExtractOp>(op, adaptor.getInput(),
                                                          adaptor.getFieldNameAttr());
    return success();
  }
};

struct UnionCreateOpConversion : public OpConversionPattern<UnionCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<hw::UnionCreateOp>(
        op, resultType, adaptor.getFieldNameAttr(), adaptor.getInput());
    return success();
  }
};

struct UnionExtractOpConversion : public OpConversionPattern<UnionExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::UnionExtractOp>(op, adaptor.getInput(),
                                                    adaptor.getFieldNameAttr());
    return success();
  }
};

struct UnionExtractRefOpConversion
    : public OpConversionPattern<UnionExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnionExtractRefOp op, OpAdaptor adaptor,
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
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    auto loc = op.getLoc();

    if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
      Value max = hw::ConstantOp::create(rewriter, loc, intTy, -1);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::eq,
                                                input, max);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, input);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);
    Value allOnesVec = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(loc, unknownBits, allOnesVec);
    Value invValue = rewriter.createOrFold<comb::XorOp>(loc, valueBits, allOnesVec);
    Value knownZeroMask =
        rewriter.createOrFold<comb::AndOp>(loc, knownMask, invValue);

    Value hasKnownZero = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, knownZeroMask, zeroVec);
    Value hasUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, unknownBits, zeroVec);

    Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
    Value noKnownZero = rewriter.createOrFold<comb::XorOp>(loc, hasKnownZero, one);
    Value unknownOut = rewriter.createOrFold<comb::AndOp>(loc, noKnownZero, hasUnknown);
    Value noUnknown = rewriter.createOrFold<comb::XorOp>(loc, hasUnknown, one);
    Value valueOut = rewriter.createOrFold<comb::AndOp>(loc, noKnownZero, noUnknown);

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReduceOrOpConversion : public OpConversionPattern<ReduceOrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    auto loc = op.getLoc();

    if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
      Value zero = hw::ConstantOp::create(rewriter, loc, intTy, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                input, zero);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, input);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);
    Value allOnesVec = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(loc, unknownBits, allOnesVec);
    Value knownOneMask =
        rewriter.createOrFold<comb::AndOp>(loc, knownMask, valueBits);

    Value hasKnownOne = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, knownOneMask, zeroVec);
    Value hasUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, unknownBits, zeroVec);

    Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
    Value noKnownOne = rewriter.createOrFold<comb::XorOp>(loc, hasKnownOne, one);
    Value unknownOut = rewriter.createOrFold<comb::AndOp>(loc, noKnownOne, hasUnknown);
    Value valueOut = hasKnownOne;

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReduceXorOpConversion : public OpConversionPattern<ReduceXorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReduceXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    auto loc = op.getLoc();

    if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
      rewriter.replaceOpWithNewOp<comb::ParityOp>(op, input);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, input);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);

    Value hasUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, unknownBits, zeroVec);
    Value parity = rewriter.createOrFold<comb::ParityOp>(loc, valueBits);

    Value one = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
    Value noUnknown = rewriter.createOrFold<comb::XorOp>(loc, hasUnknown, one);
    Value valueOut = rewriter.createOrFold<comb::AndOp>(loc, parity, noUnknown);
    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, hasUnknown);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BoolCastOpConversion : public OpConversionPattern<BoolCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BoolCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    auto loc = op.getLoc();

    if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
      Value zero = hw::ConstantOp::create(rewriter, loc, intTy, 0);
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, comb::ICmpPredicate::ne,
                                                input, zero);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, input);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);

    Value hasUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, unknownBits, zeroVec);
    Value nonZero = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, valueBits, zeroVec);
    Value zeroBit = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    Value valueOut =
        rewriter.createOrFold<comb::MuxOp>(loc, hasUnknown, zeroBit, nonZero);

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, hasUnknown);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct NotOpConversion : public OpConversionPattern<NotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();

    if (isa<IntegerType>(input.getType())) {
      Value max = hw::ConstantOp::create(rewriter, op.getLoc(),
                                         cast<IntegerType>(resultType), -1);
      rewriter.replaceOpWithNewOp<comb::XorOp>(op, input, max);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, op.getLoc(), vecTy, -1);

    Value notValue =
        rewriter.createOrFold<comb::XorOp>(op.getLoc(), valueBits, allOnes);
    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(op.getLoc(), unknownBits, allOnes);
    Value valueOut =
        rewriter.createOrFold<comb::AndOp>(op.getLoc(), notValue, knownMask);

    Value result =
        createFourValued(rewriter, op.getLoc(), resultType, valueOut, unknownBits);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
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
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();

    if (isa<IntegerType>(input.getType())) {
      Value zero = hw::ConstantOp::create(rewriter, op.getLoc(),
                                          cast<IntegerType>(resultType), 0);
      rewriter.replaceOpWithNewOp<comb::SubOp>(op, zero, input);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);
    auto vecTy = cast<IntegerType>(valueBits.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, op.getLoc(), vecTy, 0);
    Value allOnesVec = hw::ConstantOp::create(rewriter, op.getLoc(), vecTy, -1);

    Value anyUnknown = rewriter.createOrFold<comb::ICmpOp>(
        op.getLoc(), comb::ICmpPredicate::ne, unknownBits, zeroVec);
    Value negValue =
        rewriter.createOrFold<comb::SubOp>(op.getLoc(), zeroVec, valueBits, false);
    Value known =
        createFourValued(rewriter, op.getLoc(), resultType, negValue, zeroVec);
    Value allX =
        createFourValued(rewriter, op.getLoc(), resultType, zeroVec, allOnesVec);
    if (!known || !allX)
      return failure();

    Value mux =
        rewriter.createOrFold<comb::MuxOp>(op.getLoc(), anyUnknown, allX, known);
    rewriter.replaceOp(op, mux);
    return success();
  }
};

struct NegRealOpConversion : public OpConversionPattern<NegRealOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NegRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::NegFOp>(op, adaptor.getInput());
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
    Type resultType = this->typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<TargetOp>(op, lhs, rhs, false);
      return success();
    }

    if (!isFourValuedIntType(lhs.getType()) || !isFourValuedIntType(rhs.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    Value lhsValue = getFourValuedValue(rewriter, op.getLoc(), lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, op.getLoc(), lhs);
    Value rhsValue = getFourValuedValue(rewriter, op.getLoc(), rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, op.getLoc(), rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, op.getLoc(), vecTy, 0);
    Value allOnesVec = hw::ConstantOp::create(rewriter, op.getLoc(), vecTy, -1);

    Value unknownAnyVec = rewriter.createOrFold<comb::OrOp>(
        op.getLoc(), lhsUnknown, rhsUnknown);
    Value anyUnknown = rewriter.createOrFold<comb::ICmpOp>(
        op.getLoc(), comb::ICmpPredicate::ne, unknownAnyVec, zeroVec);

    Value valueKnown =
        rewriter.createOrFold<TargetOp>(op.getLoc(), lhsValue, rhsValue, false);
    Value known =
        createFourValued(rewriter, op.getLoc(), resultType, valueKnown, zeroVec);
    Value allX =
        createFourValued(rewriter, op.getLoc(), resultType, zeroVec, allOnesVec);
    if (!known || !allX)
      return failure();

    Value mux =
        rewriter.createOrFold<comb::MuxOp>(op.getLoc(), anyUnknown, allX, known);
    rewriter.replaceOp(op, mux);
    return success();
  }
};

struct BitwiseAndOpConversion : public OpConversionPattern<AndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<comb::AndOp>(op, lhs, rhs, false);
      return success();
    }

    if (!isFourValuedIntType(lhs.getType()) || !isFourValuedIntType(rhs.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    auto loc = op.getLoc();
    Value lhsValue = getFourValuedValue(rewriter, loc, lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhs);
    Value rhsValue = getFourValuedValue(rewriter, loc, rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    Value lhsKnownMask =
        rewriter.createOrFold<comb::XorOp>(loc, lhsUnknown, allOnes);
    Value rhsKnownMask =
        rewriter.createOrFold<comb::XorOp>(loc, rhsUnknown, allOnes);
    Value lhsInvValue =
        rewriter.createOrFold<comb::XorOp>(loc, lhsValue, allOnes);
    Value rhsInvValue =
        rewriter.createOrFold<comb::XorOp>(loc, rhsValue, allOnes);

    Value lhsKnown0 =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnownMask, lhsInvValue);
    Value rhsKnown0 =
        rewriter.createOrFold<comb::AndOp>(loc, rhsKnownMask, rhsInvValue);
    Value lhsKnown1 =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnownMask, lhsValue);
    Value rhsKnown1 =
        rewriter.createOrFold<comb::AndOp>(loc, rhsKnownMask, rhsValue);

    Value known0Out = rewriter.createOrFold<comb::OrOp>(loc, lhsKnown0, rhsKnown0);
    Value known1Out =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnown1, rhsKnown1);
    Value determined =
        rewriter.createOrFold<comb::OrOp>(loc, known0Out, known1Out);
    Value unknownOut =
        rewriter.createOrFold<comb::XorOp>(loc, determined, allOnes);
    Value valueOut = known1Out;

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BitwiseOrOpConversion : public OpConversionPattern<OrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<comb::OrOp>(op, lhs, rhs, false);
      return success();
    }

    if (!isFourValuedIntType(lhs.getType()) || !isFourValuedIntType(rhs.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    auto loc = op.getLoc();
    Value lhsValue = getFourValuedValue(rewriter, loc, lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhs);
    Value rhsValue = getFourValuedValue(rewriter, loc, rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    Value lhsKnownMask =
        rewriter.createOrFold<comb::XorOp>(loc, lhsUnknown, allOnes);
    Value rhsKnownMask =
        rewriter.createOrFold<comb::XorOp>(loc, rhsUnknown, allOnes);
    Value lhsInvValue =
        rewriter.createOrFold<comb::XorOp>(loc, lhsValue, allOnes);
    Value rhsInvValue =
        rewriter.createOrFold<comb::XorOp>(loc, rhsValue, allOnes);

    Value lhsKnown0 =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnownMask, lhsInvValue);
    Value rhsKnown0 =
        rewriter.createOrFold<comb::AndOp>(loc, rhsKnownMask, rhsInvValue);
    Value lhsKnown1 =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnownMask, lhsValue);
    Value rhsKnown1 =
        rewriter.createOrFold<comb::AndOp>(loc, rhsKnownMask, rhsValue);

    Value known1Out = rewriter.createOrFold<comb::OrOp>(loc, lhsKnown1, rhsKnown1);
    Value known0Out =
        rewriter.createOrFold<comb::AndOp>(loc, lhsKnown0, rhsKnown0);
    Value determined =
        rewriter.createOrFold<comb::OrOp>(loc, known0Out, known1Out);
    Value unknownOut =
        rewriter.createOrFold<comb::XorOp>(loc, determined, allOnes);
    Value valueOut = known1Out;

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BitwiseXorOpConversion : public OpConversionPattern<XorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<comb::XorOp>(op, lhs, rhs, false);
      return success();
    }

    if (!isFourValuedIntType(lhs.getType()) || !isFourValuedIntType(rhs.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    auto loc = op.getLoc();
    Value lhsValue = getFourValuedValue(rewriter, loc, lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhs);
    Value rhsValue = getFourValuedValue(rewriter, loc, rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    Value unknownOut =
        rewriter.createOrFold<comb::OrOp>(loc, lhsUnknown, rhsUnknown);
    Value xorValue =
        rewriter.createOrFold<comb::XorOp>(loc, lhsValue, rhsValue);
    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(loc, unknownOut, allOnes);
    Value valueOut =
        rewriter.createOrFold<comb::AndOp>(loc, xorValue, knownMask);

    Value result =
        createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct BinaryRealOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, adaptor.getLhs(),
                                          adaptor.getRhs());
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
    if (!resultType)
      return failure();

    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    bool lhsFV = isFourValuedIntType(lhs.getType());
    bool rhsFV = isFourValuedIntType(rhs.getType());
    if (!lhsFV && !rhsFV) {
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, resultType, pred, lhs, rhs);
      return success();
    }

    if (lhsFV != rhsFV)
      return failure();

    Value lhsValue = getFourValuedValue(rewriter, loc, lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhs);
    Value rhsValue = getFourValuedValue(rewriter, loc, rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);
    Value allOnesVec = hw::ConstantOp::create(rewriter, loc, vecTy, -1);
    Value zeroBit = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    Value oneBit = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);

    auto invBit = [&](Value v) {
      return rewriter.createOrFold<comb::XorOp>(loc, v, oneBit);
    };
    auto invVec = [&](Value v) {
      return rewriter.createOrFold<comb::XorOp>(loc, v, allOnesVec);
    };

    // Case equality / inequality (`===` / `!==`) return a 2-state bit.
    if (pred == ICmpPredicate::ceq || pred == ICmpPredicate::cne) {
      Value eqValue = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::eq, lhsValue, rhsValue);
      Value eqUnknown = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::eq, lhsUnknown, rhsUnknown);
      Value eq = rewriter.createOrFold<comb::AndOp>(loc, eqValue, eqUnknown);
      if (pred == ICmpPredicate::cne)
        eq = invBit(eq);
      rewriter.replaceOp(op, eq);
      return success();
    }

    if (!isFourValuedIntType(resultType))
      return failure();

    // Compute any-unknown for relational comparisons.
    Value unknownOr = rewriter.createOrFold<comb::OrOp>(loc, lhsUnknown, rhsUnknown);
    Value anyUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::ne, unknownOr, zeroVec);

    auto buildResult = [&](Value valueOut, Value unknownOut) -> LogicalResult {
      Value result = createFourValued(rewriter, loc, resultType, valueOut, unknownOut);
      if (!result)
        return failure();
      rewriter.replaceOp(op, result);
      return success();
    };

    // Logical equality / inequality.
    if (pred == ICmpPredicate::eq || pred == ICmpPredicate::ne) {
      Value knownMask = invVec(unknownOr);
      Value mismatchBits =
          rewriter.createOrFold<comb::XorOp>(loc, lhsValue, rhsValue);
      Value mismatchKnown =
          rewriter.createOrFold<comb::AndOp>(loc, mismatchBits, knownMask);
      Value mismatchExists = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, mismatchKnown, zeroVec);
      Value noMismatch = invBit(mismatchExists);
      Value unknownOut =
          rewriter.createOrFold<comb::AndOp>(loc, noMismatch, anyUnknown);

      if (pred == ICmpPredicate::ne)
        return buildResult(mismatchExists, unknownOut);

      Value noUnknown = invBit(anyUnknown);
      Value valueOut =
          rewriter.createOrFold<comb::AndOp>(loc, noMismatch, noUnknown);
      return buildResult(valueOut, unknownOut);
    }

    // Wildcard equality / inequality: RHS X/Z are wildcards.
    if (pred == ICmpPredicate::weq || pred == ICmpPredicate::wne) {
      Value careMask = invVec(rhsUnknown);
      Value lhsKnownMask = invVec(lhsUnknown);
      Value knownMask =
          rewriter.createOrFold<comb::AndOp>(loc, careMask, lhsKnownMask);
      Value mismatchBits =
          rewriter.createOrFold<comb::XorOp>(loc, lhsValue, rhsValue);
      Value mismatchKnown =
          rewriter.createOrFold<comb::AndOp>(loc, mismatchBits, knownMask);
      Value mismatchExists = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, mismatchKnown, zeroVec);
      Value unknownRelevantBits =
          rewriter.createOrFold<comb::AndOp>(loc, lhsUnknown, careMask);
      Value unknownRelevant = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, unknownRelevantBits, zeroVec);

      Value noMismatch = invBit(mismatchExists);
      Value unknownOut =
          rewriter.createOrFold<comb::AndOp>(loc, noMismatch, unknownRelevant);

      if (pred == ICmpPredicate::wne)
        return buildResult(mismatchExists, unknownOut);

      Value noUnknown = invBit(unknownRelevant);
      Value valueOut =
          rewriter.createOrFold<comb::AndOp>(loc, noMismatch, noUnknown);
      return buildResult(valueOut, unknownOut);
    }

    // Relational comparisons: any unknown produces X.
    if (pred == ICmpPredicate::ult || pred == ICmpPredicate::ule ||
        pred == ICmpPredicate::ugt || pred == ICmpPredicate::uge ||
        pred == ICmpPredicate::slt || pred == ICmpPredicate::sle ||
        pred == ICmpPredicate::sgt || pred == ICmpPredicate::sge) {
      Value compare =
          rewriter.createOrFold<comb::ICmpOp>(loc, pred, lhsValue, rhsValue);
      Value valueOut =
          rewriter.createOrFold<comb::MuxOp>(loc, anyUnknown, zeroBit, compare);
      return buildResult(valueOut, anyUnknown);
    }

    // Anything else is unsupported for four-valued lowering.
    return failure();
  }
};

template <typename SourceOp, arith::CmpFPredicate pred>
struct FCmpOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        ConversionPattern::typeConverter->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<arith::CmpFOp>(
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
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<IntegerType>(lhs.getType())) {
      rewriter.replaceOpWithNewOp<comb::ICmpOp>(op, ICmpPredicate::ceq, lhs, rhs);
      return success();
    }

    if (!isFourValuedIntType(lhs.getType()) || !isFourValuedIntType(rhs.getType()))
      return failure();

    auto loc = op.getLoc();
    Value lhsValue = getFourValuedValue(rewriter, loc, lhs);
    Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhs);
    Value rhsValue = getFourValuedValue(rewriter, loc, rhs);
    Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhs);

    auto vecTy = cast<IntegerType>(lhsValue.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, loc, vecTy, -1);

    // `casez`: ignore Z bits in either operand. `casexz`: ignore X/Z bits.
    Value ignored = Value{};
    if constexpr (withoutX) {
      // Z bits are `unknown & value` in the FVInt encoding.
      Value lhsZ = rewriter.createOrFold<comb::AndOp>(loc, lhsUnknown, lhsValue);
      Value rhsZ = rewriter.createOrFold<comb::AndOp>(loc, rhsUnknown, rhsValue);
      ignored = rewriter.createOrFold<comb::OrOp>(loc, lhsZ, rhsZ);
    } else {
      ignored = rewriter.createOrFold<comb::OrOp>(loc, lhsUnknown, rhsUnknown);
    }

    Value keepMask = rewriter.createOrFold<comb::XorOp>(loc, ignored, allOnes);

    Value lhsValueMasked =
        rewriter.createOrFold<comb::AndOp>(loc, lhsValue, keepMask);
    Value lhsUnknownMasked =
        rewriter.createOrFold<comb::AndOp>(loc, lhsUnknown, keepMask);
    Value rhsValueMasked =
        rewriter.createOrFold<comb::AndOp>(loc, rhsValue, keepMask);
    Value rhsUnknownMasked =
        rewriter.createOrFold<comb::AndOp>(loc, rhsUnknown, keepMask);

    Value eqValue = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, lhsValueMasked, rhsValueMasked);
    Value eqUnknown = rewriter.createOrFold<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, lhsUnknownMasked, rhsUnknownMasked);
    Value eq = rewriter.createOrFold<comb::AndOp>(loc, eqValue, eqUnknown);
    rewriter.replaceOp(op, eq);
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

      if (auto arrTy = dyn_cast<hw::ArrayType>(adaptor.getInput().getType())) {
        // Best-effort formatting for fixed-size unpacked arrays. This is
        // primarily used by `%p` formatting in UVM tests; provide a stable
        // representation rather than failing legalization.
        auto elemTy = arrTy.getElementType();
        if (isa<IntegerType>(elemTy)) {
          SmallVector<Value> frags;
          frags.push_back(rewriter.create<sim::FormatLitOp>(loc, "{"));

          unsigned numElems = arrTy.getNumElements();
          unsigned idxWidth = std::max<unsigned>(llvm::Log2_64_Ceil(numElems), 1);
          auto idxType = rewriter.getIntegerType(idxWidth);
          auto baseAttr = rewriter.getI32IntegerAttr(10);
          auto widthAttr = rewriter.getI32IntegerAttr(0);
          auto flagsAttr = rewriter.getI32IntegerAttr(0);

          for (unsigned i = 0; i < numElems; ++i) {
            if (i != 0)
              frags.push_back(rewriter.create<sim::FormatLitOp>(loc, ", "));
            Value idx = hw::ConstantOp::create(rewriter, loc, idxType, i);
            Value elem = rewriter.create<hw::ArrayGetOp>(loc, adaptor.getInput(), idx);
            frags.push_back(
                rewriter.create<sim::FormatIntOp>(loc, elem, baseAttr, widthAttr,
                                                  flagsAttr));
          }

          frags.push_back(rewriter.create<sim::FormatLitOp>(loc, "}"));
          rewriter.replaceOpWithNewOp<sim::FormatStringConcatOp>(op, frags);
          return success();
        }

        rewriter.replaceOpWithNewOp<sim::FormatLitOp>(
            op, rewriter.getStringAttr("<array>"));
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
    if (!amount)
      return failure();

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

/// For casts that are automatically resolved by type conversion.
template <typename SourceOp>
struct NoOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConversionPattern::typeConverter;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct IntToLogicOpConversion : public OpConversionPattern<IntToLogicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntToLogicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType || !isFourValuedIntType(resultType))
      return failure();

    auto intTy = cast<IntegerType>(
        cast<hw::StructType>(resultType).getElements()[0].type);
    Value unknownZero = hw::ConstantOp::create(rewriter, op.getLoc(), intTy, 0);
    Value result = createFourValued(rewriter, op.getLoc(), resultType,
                                    adaptor.getInput(), unknownZero);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LogicToIntOpConversion : public OpConversionPattern<LogicToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LogicToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType || !isa<IntegerType>(resultType))
      return failure();

    Value input = adaptor.getInput();
    if (!isFourValuedIntType(input.getType()))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);
    auto intTy = cast<IntegerType>(valueBits.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, op.getLoc(), intTy, -1);
    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(op.getLoc(), unknownBits, allOnes);
    Value knownValue =
        rewriter.createOrFold<comb::AndOp>(op.getLoc(), valueBits, knownMask);
    rewriter.replaceOp(op, knownValue);
    return success();
  }
};

struct ToBuiltinBoolOpConversion : public OpConversionPattern<ToBuiltinBoolOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToBuiltinBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType || !isa<IntegerType>(resultType))
      return failure();

    Value input = adaptor.getInput();
    if (isa<IntegerType>(input.getType())) {
      rewriter.replaceOp(op, input);
      return success();
    }

    if (!isFourValuedIntType(input.getType()))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);
    auto intTy = cast<IntegerType>(valueBits.getType());
    Value allOnes = hw::ConstantOp::create(rewriter, op.getLoc(), intTy, -1);
    Value knownMask =
        rewriter.createOrFold<comb::XorOp>(op.getLoc(), unknownBits, allOnes);
    Value knownValue =
        rewriter.createOrFold<comb::AndOp>(op.getLoc(), valueBits, knownMask);
    rewriter.replaceOp(op, knownValue);
    return success();
  }
};

struct TruncOpConversion : public OpConversionPattern<TruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    uint32_t targetWidth = op.getType().getWidth();

    if (auto intTy = dyn_cast<IntegerType>(input.getType())) {
      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, input, 0, targetWidth);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);

    Value valueTrunc = rewriter.createOrFold<comb::ExtractOp>(
        op.getLoc(), valueBits, 0, targetWidth);
    Value unknownTrunc = rewriter.createOrFold<comb::ExtractOp>(
        op.getLoc(), unknownBits, 0, targetWidth);
    Value result = createFourValued(rewriter, op.getLoc(), resultType,
                                    valueTrunc, unknownTrunc);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ZExtOpConversion : public OpConversionPattern<ZExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    auto targetWidth = op.getType().getWidth();
    auto inputWidth = op.getInput().getType().getWidth();

    Value input = adaptor.getInput();
    if (isa<IntegerType>(input.getType())) {
      auto zeroExt = hw::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerType(targetWidth - inputWidth), 0);
      rewriter.replaceOpWithNewOp<comb::ConcatOp>(op,
                                                  ValueRange{zeroExt, input});
      return success();
    }

    if (!isFourValuedIntType(input.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);

    auto zextTy = rewriter.getIntegerType(targetWidth - inputWidth);
    Value zeros = hw::ConstantOp::create(rewriter, op.getLoc(), zextTy, 0);

    Value valueExt =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(),
                                              ValueRange{zeros, valueBits});
    Value unknownExt =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(),
                                              ValueRange{zeros, unknownBits});

    Value result = createFourValued(rewriter, op.getLoc(), resultType, valueExt,
                                    unknownExt);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SExtOpConversion : public OpConversionPattern<SExtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    if (isa<IntegerType>(input.getType())) {
      auto value =
          comb::createOrFoldSExt(op.getLoc(), input, resultType, rewriter);
      rewriter.replaceOp(op, value);
      return success();
    }

    if (!isFourValuedIntType(input.getType()) ||
        !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), input);
    Value unknownBits = getFourValuedUnknown(rewriter, op.getLoc(), input);

    auto fieldTy =
        cast<IntegerType>(cast<hw::StructType>(resultType).getElements()[0].type);
    Value valueExt =
        comb::createOrFoldSExt(op.getLoc(), valueBits, fieldTy, rewriter);
    Value unknownExt =
        comb::createOrFoldSExt(op.getLoc(), unknownBits, fieldTy, rewriter);

    Value result = createFourValued(rewriter, op.getLoc(), resultType, valueExt,
                                    unknownExt);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SIntToRealOpConversion : public OpConversionPattern<SIntToRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SIntToRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

struct UIntToRealOpConversion : public OpConversionPattern<UIntToRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UIntToRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::UIToFPOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
    return success();
  }
};

struct IntToStringOpConversion : public OpConversionPattern<IntToStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntToStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::IntToStringOp>(op, adaptor.getInput());
    return success();
  }
};

struct RealToIntOpConversion : public OpConversionPattern<RealToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RealToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::FPToSIOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getInput());
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
    if (!resultType)
      return failure();

    Value value = adaptor.getValue();
    Value amount = adaptor.getAmount();
    auto loc = op.getLoc();

    if (isa<IntegerType>(value.getType())) {
      // Comb shift operations require the same bit-width for value and amount.
      Value adj =
          adjustIntegerWidth(rewriter, amount, resultType.getIntOrFloatBitWidth(), loc);
      if (!adj)
        return failure();
      rewriter.replaceOpWithNewOp<comb::ShlOp>(op, resultType, value, adj, false);
      return success();
    }

    if (!isFourValuedIntType(value.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, value);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, value);
    auto vecTy = cast<IntegerType>(valueBits.getType());

    // If the shift amount contains unknown bits, all results are X.
    Value amountUnknownAny = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    if (isFourValuedIntType(amount.getType())) {
      Value amtValue = getFourValuedValue(rewriter, loc, amount);
      Value amtUnknown = getFourValuedUnknown(rewriter, loc, amount);
      Value zeroVec = hw::ConstantOp::create(rewriter, loc, cast<IntegerType>(amtUnknown.getType()), 0);
      amountUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, amtUnknown, zeroVec);
      amount = amtValue;
    }

    Value adj = adjustIntegerWidth(rewriter, amount, vecTy.getWidth(), loc);
    if (!adj)
      return failure();
    Value shiftedValue =
        rewriter.createOrFold<comb::ShlOp>(loc, valueBits, adj, false);
    Value shiftedUnknown =
        rewriter.createOrFold<comb::ShlOp>(loc, unknownBits, adj, false);

    Value shifted = createFourValued(rewriter, loc, resultType, shiftedValue,
                                     shiftedUnknown);
    Value allX = createFourValued(
        rewriter, loc, resultType,
        hw::ConstantOp::create(rewriter, loc, vecTy, 0),
        hw::ConstantOp::create(rewriter, loc, vecTy, -1));
    if (!shifted || !allX)
      return failure();

    Value mux =
        rewriter.createOrFold<comb::MuxOp>(loc, amountUnknownAny, allX, shifted);
    rewriter.replaceOp(op, mux);
    return success();
  }
};

struct ShrOpConversion : public OpConversionPattern<ShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value value = adaptor.getValue();
    Value amount = adaptor.getAmount();
    auto loc = op.getLoc();

    if (isa<IntegerType>(value.getType())) {
      Value adj =
          adjustIntegerWidth(rewriter, amount, resultType.getIntOrFloatBitWidth(), loc);
      if (!adj)
        return failure();
      rewriter.replaceOpWithNewOp<comb::ShrUOp>(op, resultType, value, adj, false);
      return success();
    }

    if (!isFourValuedIntType(value.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, value);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, value);
    auto vecTy = cast<IntegerType>(valueBits.getType());

    Value amountUnknownAny = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    if (isFourValuedIntType(amount.getType())) {
      Value amtValue = getFourValuedValue(rewriter, loc, amount);
      Value amtUnknown = getFourValuedUnknown(rewriter, loc, amount);
      Value zeroVec = hw::ConstantOp::create(rewriter, loc, cast<IntegerType>(amtUnknown.getType()), 0);
      amountUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, amtUnknown, zeroVec);
      amount = amtValue;
    }

    Value adj = adjustIntegerWidth(rewriter, amount, vecTy.getWidth(), loc);
    if (!adj)
      return failure();
    Value shiftedValue =
        rewriter.createOrFold<comb::ShrUOp>(loc, valueBits, adj, false);
    Value shiftedUnknown =
        rewriter.createOrFold<comb::ShrUOp>(loc, unknownBits, adj, false);

    Value shifted = createFourValued(rewriter, loc, resultType, shiftedValue,
                                     shiftedUnknown);
    Value allX = createFourValued(
        rewriter, loc, resultType,
        hw::ConstantOp::create(rewriter, loc, vecTy, 0),
        hw::ConstantOp::create(rewriter, loc, vecTy, -1));
    if (!shifted || !allX)
      return failure();

    Value mux =
        rewriter.createOrFold<comb::MuxOp>(loc, amountUnknownAny, allX, shifted);
    rewriter.replaceOp(op, mux);
    return success();
  }
};

struct PowUOpConversion : public OpConversionPattern<PowUOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Location loc = op->getLoc();

    Value lhsIn = adaptor.getLhs();
    Value rhsIn = adaptor.getRhs();

    // Four-valued: any unknown propagates to an all-X result.
    if (isFourValuedIntType(lhsIn.getType())) {
      if (!isFourValuedIntType(rhsIn.getType()) || !isFourValuedIntType(resultType))
        return failure();

      Value lhsValue = getFourValuedValue(rewriter, loc, lhsIn);
      Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhsIn);
      Value rhsValue = getFourValuedValue(rewriter, loc, rhsIn);
      Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhsIn);

      auto vecTy = cast<IntegerType>(lhsValue.getType());
      Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);
      Value allOnesVec = hw::ConstantOp::create(rewriter, loc, vecTy, -1);
      Value unknownOr =
          rewriter.createOrFold<comb::OrOp>(loc, lhsUnknown, rhsUnknown);
      Value anyUnknown = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, unknownOr, zeroVec);

      Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
      auto lhs =
          comb::ConcatOp::create(rewriter, loc, zeroVal, lhsValue);
      auto rhs =
          comb::ConcatOp::create(rewriter, loc, zeroVal, rhsValue);
      auto pow = mlir::math::IPowIOp::create(rewriter, loc, lhs, rhs);
      Value trunc = rewriter.createOrFold<comb::ExtractOp>(loc, vecTy, pow, 0);

      Value known = createFourValued(rewriter, loc, resultType, trunc, zeroVec);
      Value allX =
          createFourValued(rewriter, loc, resultType, zeroVec, allOnesVec);
      if (!known || !allX)
        return failure();

      Value mux = rewriter.createOrFold<comb::MuxOp>(loc, anyUnknown, allX, known);
      rewriter.replaceOp(op, mux);
      return success();
    }

    Value zeroVal = hw::ConstantOp::create(rewriter, loc, APInt(1, 0));
    auto lhs = comb::ConcatOp::create(rewriter, loc, zeroVal, lhsIn);
    auto rhs = comb::ConcatOp::create(rewriter, loc, zeroVal, rhsIn);
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
    if (!resultType)
      return failure();

    auto loc = op.getLoc();
    Value lhsIn = adaptor.getLhs();
    Value rhsIn = adaptor.getRhs();

    if (isFourValuedIntType(lhsIn.getType())) {
      if (!isFourValuedIntType(rhsIn.getType()) || !isFourValuedIntType(resultType))
        return failure();

      Value lhsValue = getFourValuedValue(rewriter, loc, lhsIn);
      Value lhsUnknown = getFourValuedUnknown(rewriter, loc, lhsIn);
      Value rhsValue = getFourValuedValue(rewriter, loc, rhsIn);
      Value rhsUnknown = getFourValuedUnknown(rewriter, loc, rhsIn);

      auto vecTy = cast<IntegerType>(lhsValue.getType());
      Value zeroVec = hw::ConstantOp::create(rewriter, loc, vecTy, 0);
      Value allOnesVec = hw::ConstantOp::create(rewriter, loc, vecTy, -1);
      Value unknownOr =
          rewriter.createOrFold<comb::OrOp>(loc, lhsUnknown, rhsUnknown);
      Value anyUnknown = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, unknownOr, zeroVec);

      auto pow = mlir::math::IPowIOp::create(rewriter, loc, lhsValue, rhsValue);
      Value known = createFourValued(rewriter, loc, resultType, pow, zeroVec);
      Value allX =
          createFourValued(rewriter, loc, resultType, zeroVec, allOnesVec);
      if (!known || !allX)
        return failure();

      Value mux = rewriter.createOrFold<comb::MuxOp>(loc, anyUnknown, allX, known);
      rewriter.replaceOp(op, mux);
      return success();
    }

    // utilize MLIR math dialect's math.ipowi to handle the exponentiation of
    // expression
    rewriter.replaceOpWithNewOp<mlir::math::IPowIOp>(
        op, resultType, lhsIn, rhsIn);
    return success();
  }
};

struct AShrOpConversion : public OpConversionPattern<AShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value value = adaptor.getValue();
    Value amount = adaptor.getAmount();
    auto loc = op.getLoc();

    if (isa<IntegerType>(value.getType())) {
      Value adj =
          adjustIntegerWidth(rewriter, amount, resultType.getIntOrFloatBitWidth(), loc);
      if (!adj)
        return failure();
      rewriter.replaceOpWithNewOp<comb::ShrSOp>(op, resultType, value, adj, false);
      return success();
    }

    if (!isFourValuedIntType(value.getType()) || !isFourValuedIntType(resultType))
      return failure();

    Value valueBits = getFourValuedValue(rewriter, loc, value);
    Value unknownBits = getFourValuedUnknown(rewriter, loc, value);
    auto vecTy = cast<IntegerType>(valueBits.getType());

    Value amountUnknownAny = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 0);
    if (isFourValuedIntType(amount.getType())) {
      Value amtValue = getFourValuedValue(rewriter, loc, amount);
      Value amtUnknown = getFourValuedUnknown(rewriter, loc, amount);
      Value zeroVec = hw::ConstantOp::create(rewriter, loc, cast<IntegerType>(amtUnknown.getType()), 0);
      amountUnknownAny = rewriter.createOrFold<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ne, amtUnknown, zeroVec);
      amount = amtValue;
    }

    Value adj = adjustIntegerWidth(rewriter, amount, vecTy.getWidth(), loc);
    if (!adj)
      return failure();
    Value shiftedValue =
        rewriter.createOrFold<comb::ShrSOp>(loc, valueBits, adj, false);
    Value shiftedUnknown =
        rewriter.createOrFold<comb::ShrSOp>(loc, unknownBits, adj, false);

    Value shifted = createFourValued(rewriter, loc, resultType, shiftedValue,
                                     shiftedUnknown);
    Value allX = createFourValued(
        rewriter, loc, resultType,
        hw::ConstantOp::create(rewriter, loc, vecTy, 0),
        hw::ConstantOp::create(rewriter, loc, vecTy, -1));
    if (!shifted || !allX)
      return failure();

    Value mux =
        rewriter.createOrFold<comb::MuxOp>(loc, amountUnknownAny, allX, shifted);
    rewriter.replaceOp(op, mux);
    return success();
  }
};

struct ReadOpConversion : public OpConversionPattern<ReadOp> {
  using OpConversionPattern::OpConversionPattern;
  using OneToNOpAdaptor = typename OpConversionPattern<ReadOp>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(ReadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange input = adaptor.getInput();

    if (input.size() == 2) {
      Type resultType = typeConverter->convertType(op.getResult().getType());
      if (!resultType || !isFourValuedIntType(resultType))
        return failure();

      auto loc = op.getLoc();
      Value valueBits = llhd::ProbeOp::create(rewriter, loc, input[0]);
      Value unknownBits = llhd::ProbeOp::create(rewriter, loc, input[1]);
      Value combined =
          createFourValued(rewriter, loc, resultType, valueBits, unknownBits);
      if (!combined)
        return failure();
      rewriter.replaceOp(op, combined);
      return success();
    }

    if (input.size() != 1)
      return failure();

    rewriter.replaceOpWithNewOp<llhd::ProbeOp>(op, input.front());
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

// Blocking and continuous assignments get a 0ns 0d 1e delay.
static llhd::TimeAttr
getBlockingOrContinuousAssignDelay(mlir::MLIRContext *context) {
  return llhd::TimeAttr::get(context, 0U, "ns", 0, 1);
}

template <typename OpTy>
struct AssignOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  using OneToNOpAdaptor = typename OpConversionPattern<OpTy>::OneToNOpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Determine the delay for the assignment.
    Value delay;
    if constexpr (std::is_same_v<OpTy, ContinuousAssignOp> ||
                  std::is_same_v<OpTy, BlockingAssignOp>) {
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          getBlockingOrContinuousAssignDelay(op->getContext()));
    } else if constexpr (std::is_same_v<OpTy, NonBlockingAssignOp>) {
      // Non-blocking assignments get a 0ns 1d 0e delay.
      delay = llhd::ConstantTimeOp::create(
          rewriter, op->getLoc(),
          llhd::TimeAttr::get(op->getContext(), 0U, "ns", 1, 0));
    } else {
      // Delayed assignments have a delay operand.
      delay = adaptor.getDelay();
    }

    rewriter.replaceOpWithNewOp<llhd::DriveOp>(
        op, adaptor.getDst(), adaptor.getSrc(), delay, Value{});
    return success();
  }

  LogicalResult
  matchAndRewrite(OpTy op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange dst = adaptor.getDst();
    ValueRange srcRange = adaptor.getSrc();
    if (srcRange.size() != 1)
      return failure();
    Value src = srcRange.front();

    // Four-valued integer refs are represented as `{!llhd.ref<value>,
    // !llhd.ref<unknown>}`.
    if (dst.size() == 2) {
      auto loc = op.getLoc();

      Value srcValueBits;
      Value srcUnknownBits;
      if (isFourValuedIntType(src.getType())) {
        srcValueBits = getFourValuedValue(rewriter, loc, src);
        srcUnknownBits = getFourValuedUnknown(rewriter, loc, src);
      } else if (auto intTy = dyn_cast<IntegerType>(src.getType())) {
        srcValueBits = src;
        srcUnknownBits = hw::ConstantOp::create(rewriter, loc, intTy, 0);
      } else {
        return failure();
      }

      auto timeAttr = llhd::TimeAttr::get(op->getContext(), 0U,
                                          llvm::StringRef("ns"), DeltaTime,
                                          EpsilonTime);
      auto time = llhd::ConstantTimeOp::create(rewriter, loc, timeAttr);
      llhd::DrvOp::create(rewriter, loc, dst[0], srcValueBits, time, Value{});
      llhd::DrvOp::create(rewriter, loc, dst[1], srcUnknownBits, time,
                          Value{});
      rewriter.eraseOp(op);
      return success();
    }

    if (dst.size() != 1)
      return failure();

    // TODO: When we support delay control in Moore dialect, we need to update
    // this conversion.
    auto timeAttr = llhd::TimeAttr::get(
        op->getContext(), 0U, llvm::StringRef("ns"), DeltaTime, EpsilonTime);
    auto time = llhd::ConstantTimeOp::create(rewriter, op->getLoc(), timeAttr);
    rewriter.replaceOpWithNewOp<llhd::DrvOp>(op, dst.front(), src, time,
                                             Value{});
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
    auto loc = op.getLoc();

    Value condition = adaptor.getCondition();
    if (isFourValuedIntType(condition.getType()))
      condition = getFourValuedValue(rewriter, loc, condition);
    if (auto intTy = dyn_cast<IntegerType>(condition.getType())) {
      if (intTy.getWidth() != 1) {
        Value zero = hw::ConstantOp::create(rewriter, loc, intTy, 0);
        condition = rewriter.createOrFold<comb::ICmpOp>(
            loc, comb::ICmpPredicate::ne, condition, zero);
      }
    } else {
      return failure();
    }

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

      rewriter.replaceOpWithNewOp<comb::MuxOp>(op, condition,
                                               convTrueVal, convFalseVal);
      return success();
    }

    auto ifOp = scf::IfOp::create(rewriter, loc, type, condition);
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
    if (isa<llhd::GlobalSignalOp>(op->getParentOp()))
      rewriter.replaceOpWithNewOp<llhd::YieldOp>(op, adaptor.getResult());
    else
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
    rewriter.replaceOpWithNewOp<sim::FormatLiteralOp>(op, adaptor.getLiteral());
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

    // Otherwise, evaluate the format string at runtime (used for `$sformatf`).
    rewriter.replaceOpWithNewOp<sim::FormatToStringOp>(op, input);
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
      auto text = formatFourValuedConstant(*fv, op.getFormat(), op.getWidth(),
                                           op.getAlignment(), op.getPadding());
      rewriter.replaceOpWithNewOp<sim::FormatLitOp>(
          op, rewriter.getStringAttr(text));
      return success();
    }

    int32_t base = 10;
    int32_t flags = 0;
    switch (op.getFormat()) {
    case IntFormat::Binary:
      base = 2;
      break;
    case IntFormat::Octal:
      base = 8;
      break;
    case IntFormat::Decimal:
      base = 10;
      break;
    case IntFormat::HexLower:
      base = 16;
      break;
    case IntFormat::HexUpper:
      base = 16;
      flags |= 1 << 0; // uppercase
      break;
    default:
      return rewriter.notifyMatchFailure(op, "unsupported int format");
    }

    if (op.getAlignment() == IntAlign::Left)
      flags |= 1 << 1; // leftJustify
    if (op.getPadding() == IntPadding::Zero)
      flags |= 1 << 2; // padZero

    if (op.getValue().getType().getDomain() == Domain::TwoValued) {
      rewriter.replaceOpWithNewOp<sim::FormatIntOp>(
          op, adaptor.getValue(), rewriter.getI32IntegerAttr(base),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getWidth())),
          rewriter.getI32IntegerAttr(flags));
      return success();
    }

    // Dynamic four-valued formatting: pass `{value, unknown}` to Sim.
    if (!isFourValuedIntType(adaptor.getValue().getType()))
      return failure();
    Value valueBits = getFourValuedValue(rewriter, op.getLoc(), adaptor.getValue());
    Value unknownBits =
        getFourValuedUnknown(rewriter, op.getLoc(), adaptor.getValue());
    rewriter.replaceOpWithNewOp<sim::FormatFVIntOp>(
        op, valueBits, unknownBits, rewriter.getI32IntegerAttr(base),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getWidth())),
        rewriter.getI32IntegerAttr(flags));
    return success();
  }
};

struct FormatRealOpConversion : public OpConversionPattern<FormatRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fracDigitsAttr = adaptor.getFracDigitsAttr();

    auto fieldWidthAttr = adaptor.getFieldWidthAttr();
    bool isLeftAligned = adaptor.getAlignment() == IntAlign::Left;
    mlir::BoolAttr isLeftAlignedAttr = rewriter.getBoolAttr(isLeftAligned);

    switch (op.getFormat()) {
    case RealFormat::General:
      rewriter.replaceOpWithNewOp<sim::FormatGeneralOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    case RealFormat::Float:
      rewriter.replaceOpWithNewOp<sim::FormatFloatOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    case RealFormat::Exponential:
      rewriter.replaceOpWithNewOp<sim::FormatScientificOp>(
          op, adaptor.getValue(), isLeftAlignedAttr, fieldWidthAttr,
          fracDigitsAttr);
      return success();
    }
  }
};

struct StringLenOpConversion : public OpConversionPattern<StringLenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringLenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::StringLengthOp>(op, adaptor.getStr());
    return success();
  }
};

struct StringConcatOpConversion : public OpConversionPattern<StringConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StringConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<sim::StringConcatOp>(op, adaptor.getInputs());
    return success();
  }
};

struct QueueSizeBIOpConversion : public OpConversionPattern<QueueSizeBIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueueSizeBIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto readQueue =
        llhd::ProbeOp::create(rewriter, op->getLoc(), adaptor.getQueue());

    rewriter.replaceOpWithNewOp<sim::QueueSizeOp>(op, readQueue);
    return success();
  }
};

struct DynQueueExtractOpConversion
    : public OpConversionPattern<DynQueueExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DynQueueExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool isSingleElementExtract =
        op.getInput().getType().getElementType() == op.getResult().getType();

    if (isSingleElementExtract) {
      rewriter.replaceOpWithNewOp<sim::QueueGetOp>(op, adaptor.getInput(),
                                                   adaptor.getLowerIdx());
    } else {
      rewriter.replaceOpWithNewOp<sim::QueueSliceOp>(
          op, adaptor.getInput(), adaptor.getLowerIdx(), adaptor.getUpperIdx());
    }

    return success();
  }
};

struct FormatTimeOpConversion : public OpConversionPattern<FormatTimeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormatTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Format `%t` by converting `!llhd.time` into an `i64` femtoseconds count
    // and delegating to Sim's time formatter (which honors `$timeformat`).
    Value timeFs =
        rewriter.create<llhd::TimeToIntOp>(op.getLoc(), adaptor.getValue());

    if (auto widthAttr = op.getWidthAttr()) {
      rewriter.replaceOpWithNewOp<sim::FormatTimeOp>(op, timeFs, widthAttr);
      return success();
    }
    rewriter.replaceOpWithNewOp<sim::FormatTimeOp>(op, timeFs);
    return success();
  }
};

// Given a reference `ref` to some Moore type, this function emits a
// `ProbeOp` to read the contained value, then passes it to the function `func`.
// It finally emits a `DriveOp` to write the result of the function back to
// the referenced signal.
//
// This is useful for converting impure operations (such as the Moore ops for
// manipulating queues) into pure operations. (Which do not mutate the source
// value, instead returning a modified value.)
static void
probeRefAndDriveWithResult(OpBuilder &builder, Location loc, Value ref,
                           const std::function<Value(Value)> &func) {

  Value v = llhd::ProbeOp::create(builder, loc, ref);

  // Drive using the same delay as a blocking assignment
  Value delay = llhd::ConstantTimeOp::create(
      builder, loc, getBlockingOrContinuousAssignDelay(builder.getContext()));

  llhd::DriveOp::create(builder, loc, ref, func(v), delay, Value{});
}

struct QueuePushBackOpConversion : public OpConversionPattern<QueuePushBackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueuePushBackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          return sim::QueuePushBackOp::create(rewriter, op->getLoc(), queue,
                                              adaptor.getElement());
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct QueuePushFrontOpConversion
    : public OpConversionPattern<QueuePushFrontOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueuePushFrontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          return sim::QueuePushFrontOp::create(rewriter, op->getLoc(), queue,
                                               adaptor.getElement());
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct QueuePopBackOpConversion : public OpConversionPattern<QueuePopBackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueuePopBackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          auto popBack =
              sim::QueuePopBackOp::create(rewriter, op->getLoc(), queue);

          op.replaceAllUsesWith(popBack.getPopped());
          return popBack.getOutQueue();
        });
    rewriter.eraseOp(op);

    return success();
  }
};

struct QueuePopFrontOpConversion : public OpConversionPattern<QueuePopFrontOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueuePopFrontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          auto popFront =
              sim::QueuePopFrontOp::create(rewriter, op->getLoc(), queue);

          op.replaceAllUsesWith(popFront.getPopped());
          return popFront.getOutQueue();
        });
    rewriter.eraseOp(op);

    return success();
  }
};

struct QueueClearOpConversion : public OpConversionPattern<QueueClearOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueueClearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto refType = cast<llhd::RefType>(adaptor.getQueue().getType());
    auto queueType = refType.getNestedType();
    Value emptyQueue =
        sim::QueueEmptyOp::create(rewriter, op->getLoc(), queueType);

    // Replace with an assignment to an empty queue
    Value delay = llhd::ConstantTimeOp::create(
        rewriter, op.getLoc(),
        getBlockingOrContinuousAssignDelay(rewriter.getContext()));

    llhd::DriveOp::create(rewriter, op.getLoc(), adaptor.getQueue(), emptyQueue,
                          delay, Value{});

    rewriter.eraseOp(op);
    return success();
  }
};

struct QueueInsertOpConversion : public OpConversionPattern<QueueInsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueueInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          auto insert =
              sim::QueueInsertOp::create(rewriter, op->getLoc(), queue,
                                         adaptor.getIndex(), adaptor.getItem());

          return insert.getOutQueue();
        });
    rewriter.eraseOp(op);

    return success();
  }
};

struct QueueDeleteOpConversion : public OpConversionPattern<QueueDeleteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QueueDeleteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    probeRefAndDriveWithResult(
        rewriter, op.getLoc(), adaptor.getQueue(), [&](Value queue) {
          auto delOp = sim::QueueDeleteOp::create(rewriter, op->getLoc(), queue,
                                                  adaptor.getIndex());

          return delOp.getOutQueue();
        });
    rewriter.eraseOp(op);

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
  case (Severity::Info):
    severityString = "Info: ";
    break;
  }

  auto prefix =
      sim::FormatLiteralOp::create(rewriter, op.getLoc(), severityString);
  // SystemVerilog severity tasks (`$fatal`, `$error`, `$warning`) emit a
  // newline, unlike `$write`. Model this by appending a newline fragment.
  auto newline =
      sim::FormatLiteralOp::create(rewriter, op.getLoc(), "\n");
  auto message = sim::FormatStringConcatOp::create(
      rewriter, op.getLoc(),
      ValueRange{prefix, adaptor.getMessage(), newline});
  rewriter.replaceOpWithNewOp<sim::PrintFormattedProcOp>(op, message);
  return success();
}

// moore.builtin.timeformat -> sim.proc.timeformat
static LogicalResult convert(TimeFormatBIOp op, TimeFormatBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<sim::TimeFormatProcOp>(
      op, op.getUnitAttr(), op.getPrecisionAttr(), op.getSuffixAttr(),
      op.getMinWidthAttr());
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
// Timing Control Conversion
//===----------------------------------------------------------------------===//

// moore.builtin.time
static LogicalResult convert(TimeBIOp op, TimeBIOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::CurrentTimeOp>(op);
  return success();
}

// moore.logic_to_time
static LogicalResult convert(LogicToTimeOp op, LogicToTimeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::IntToTimeOp>(op, adaptor.getInput());
  return success();
}

// moore.time_to_logic
static LogicalResult convert(TimeToLogicOp op, TimeToLogicOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<llhd::TimeToIntOp>(op, adaptor.getInput());
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
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  target.addLegalOp<sv::InterfaceOp, sv::InterfaceSignalOp,
                    sv::InterfaceModportOp, sv::InterfaceInstanceOp,
                    sv::AssignInterfaceSignalOp, sv::GetModportOp>();

  target.addDynamicallyLegalOp<sv::ReadInterfaceSignalOp>(
      [&](sv::ReadInterfaceSignalOp op) {
        return converter.isLegal(op.getResult().getType());
      });

  target.addLegalOp<debug::ScopeOp>();

  target.addDynamicallyLegalOp<scf::YieldOp, func::CallOp, func::ReturnOp,
                               UnrealizedConversionCastOp, hw::OutputOp,
                               hw::InstanceOp, debug::ArrayOp, debug::StructOp,
                               debug::VariableOp>(
      [&](Operation *op) { return converter.isLegal(op); });

  target.addDynamicallyLegalOp<scf::IfOp, scf::ForOp, scf::ExecuteRegionOp,
                               scf::WhileOp, scf::ForallOp>([&](Operation *op) {
    return converter.isLegal(op) && !op->getParentOfType<llhd::ProcessOp>();
  });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<hw::HWModuleOp>([&](hw::HWModuleOp op) {
    return converter.isSignatureLegal(op.getModuleType().getFuncType()) &&
           converter.isLegal(&op.getBody());
  });
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](IntType type) -> Type {
    auto iTy = IntegerType::get(type.getContext(), type.getWidth());
    if (type.getDomain() == Domain::TwoValued)
      return iTy;

    // Represent four-valued integers as `{value, unknown}` bitvectors.
    SmallVector<hw::StructType::FieldInfo> fields;
    fields.push_back(
        {StringAttr::get(type.getContext(), kFourValuedValueField), iTy});
    fields.push_back(
        {StringAttr::get(type.getContext(), kFourValuedUnknownField), iTy});
    return hw::StructType::get(type.getContext(), fields);
  });

  typeConverter.addConversion([&](RealType type) -> mlir::Type {
    MLIRContext *ctx = type.getContext();
    switch (type.getWidth()) {
    case moore::RealWidth::f32:
      return mlir::Float32Type::get(ctx);
    case moore::RealWidth::f64:
      return mlir::Float64Type::get(ctx);
    }
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

  typeConverter.addConversion([&](StringType type) {
    return sim::DynamicStringType::get(type.getContext());
  });

  typeConverter.addConversion([&](QueueType type) {
    return sim::QueueType::get(type.getContext(),
                               typeConverter.convertType(type.getElementType()),
                               type.getBound());
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

  // UnionType -> hw::UnionType
  typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
    SmallVector<hw::UnionType::FieldInfo> fields;
    for (auto field : type.getMembers()) {
      hw::UnionType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      info.offset = 0; // packed union, all fields start at bit 0
      fields.push_back(info);
    }
    auto result = hw::UnionType::get(type.getContext(), fields);
    return result;
  });

  // UnpackedUnionType -> hw::UnionType
  typeConverter.addConversion(
      [&](UnpackedUnionType type) -> std::optional<Type> {
        SmallVector<hw::UnionType::FieldInfo> fields;
        for (auto field : type.getMembers()) {
          hw::UnionType::FieldInfo info;
          info.type = typeConverter.convertType(field.type);
          if (!info.type)
            return {};
          info.name = field.name;
          info.offset = 0;
          fields.push_back(info);
        }
        return hw::UnionType::get(type.getContext(), fields);
      });

  // Conversion of CHandle to LLVMPointerType
  typeConverter.addConversion([&](ChandleType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  // Explicitly mark LLVMPointerType as a legal target
  typeConverter.addConversion(
      [](LLVM::LLVMPointerType t) -> std::optional<Type> { return t; });

  // ClassHandleType  ->  !llvm.ptr
  typeConverter.addConversion([&](ClassHandleType type) -> std::optional<Type> {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  typeConverter.addConversion(
      [&](RefType type, SmallVectorImpl<Type> &results) -> LogicalResult {
        // Represent references to four-valued integers as two independent
        // subsignal references: `{!llhd.ref<value>, !llhd.ref<unknown>}`.
        if (auto intType = dyn_cast<IntType>(type.getNestedType())) {
          if (intType.getDomain() == Domain::FourValued) {
            auto iTy = IntegerType::get(type.getContext(), intType.getWidth());
            results.push_back(llhd::RefType::get(iTy));
            results.push_back(llhd::RefType::get(iTy));
            return success();
          }
        }

        if (auto innerType = typeConverter.convertType(type.getNestedType())) {
          results.push_back(llhd::RefType::get(innerType));
          return success();
        }
        return failure();
      });

  // Valid target types.
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });
  typeConverter.addConversion([](sim::DynamicStringType type) { return type; });
  typeConverter.addConversion([](llhd::TimeType type) { return type; });
  typeConverter.addConversion([](hw::StringType type) { return type; });
  typeConverter.addConversion([](debug::ArrayType type) { return type; });
  typeConverter.addConversion([](debug::ScopeType type) { return type; });
  typeConverter.addConversion([](debug::StructType type) { return type; });

  typeConverter.addConversion([&](llhd::RefType type) -> std::optional<Type> {
    if (auto innerType = typeConverter.convertType(type.getNestedType()))
      return llhd::RefType::get(innerType);
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

  typeConverter.addConversion([&](hw::UnionType type) -> std::optional<Type> {
    SmallVector<hw::UnionType::FieldInfo> fields;
    for (auto field : type.getElements()) {
      hw::UnionType::FieldInfo info;
      info.type = typeConverter.convertType(field.type);
      if (!info.type)
        return {};
      info.name = field.name;
      info.offset = field.offset;
      fields.push_back(info);
    }
    return hw::UnionType::get(type.getContext(), fields);
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
        // Materialize reads from `inout<T>` values as explicit probes instead
        // of leaving an unresolved cast behind. This is important for SV
        // interface lowering, where interface instances are backed by
        // `inout<struct<...>>` storage, but some users need the current value
        // (`struct<...>`) for extraction.
        if (auto inoutTy = dyn_cast<hw::InOutType>(inputs[0].getType())) {
          auto stripAlias = [](Type t) -> Type {
            while (auto alias = dyn_cast<hw::TypeAliasType>(t))
              t = alias.getInnerType();
            return t;
          };
          if (stripAlias(inoutTy.getElementType()) == stripAlias(resultType))
            return llhd::PrbOp::create(builder, loc, inputs[0]);
        }
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
                                 TypeConverter &typeConverter,
                                 ClassTypeCache &classCache) {

  patterns.add<ClassDeclOpConversion>(typeConverter, patterns.getContext(),
                                      classCache);
  patterns.add<ClassNewOpConversion>(typeConverter, patterns.getContext(),
                                     classCache);
  patterns.add<ClassPropertyRefOpConversion>(typeConverter,
                                             patterns.getContext(), classCache);

  // clang-format off
  patterns.add<
    ClassUpcastOpConversion,
    // Patterns of declaration operations.
    VariableOpConversion,
    NetOpConversion,

    // Patterns for conversion operations.
    ConversionOpConversion,
    BitcastConversion<PackedToSBVOp>,
    BitcastConversion<SBVToPackedOp>,
    LogicToIntOpConversion,
    IntToLogicOpConversion,
    ToBuiltinBoolOpConversion,
    NoOpConversion<ToBuiltinIntOp>,
    NoOpConversion<FromBuiltinIntOp>,
    TruncOpConversion,
    ZExtOpConversion,
    SExtOpConversion,
    SIntToRealOpConversion,
    UIntToRealOpConversion,
    IntToStringOpConversion,
    RealToIntOpConversion,

    // Patterns of miscellaneous operations.
    ConstantOpConv,
    ConstantRealOpConv,
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
    UnionCreateOpConversion,
    UnionExtractOpConversion,
    UnionExtractRefOpConversion,
    ConditionalOpConversion,
    ArrayCreateOpConversion,
    YieldOpConversion,
    OutputOpConversion,
    ConstantStringOpConv,
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
    BitwiseAndOpConversion,
    BitwiseOrOpConversion,
    BitwiseXorOpConversion,

    // Patterns for unary real operations.
    NegRealOpConversion,

    // Patterns for binary real operations.
    BinaryRealOpConversion<AddRealOp, arith::AddFOp>,
    BinaryRealOpConversion<SubRealOp, arith::SubFOp>,
    BinaryRealOpConversion<DivRealOp, arith::DivFOp>,
    BinaryRealOpConversion<MulRealOp, arith::MulFOp>,
    BinaryRealOpConversion<PowRealOp, math::PowFOp>,

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
    FCmpOpConversion<NeRealOp, arith::CmpFPredicate::ONE>,
    FCmpOpConversion<FltOp, arith::CmpFPredicate::OLT>,
    FCmpOpConversion<FleOp, arith::CmpFPredicate::OLE>,
    FCmpOpConversion<FgtOp, arith::CmpFPredicate::OGT>,
    FCmpOpConversion<FgeOp, arith::CmpFPredicate::OGE>,
    FCmpOpConversion<EqRealOp, arith::CmpFPredicate::OEQ>,
    CaseXZEqOpConversion<CaseZEqOp, true>,
    CaseXZEqOpConversion<CaseXZEqOp, false>,
    UArrayCmpOpConversion,
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
    AssignOpConversion<ContinuousAssignOp>,
    AssignOpConversion<DelayedContinuousAssignOp>,
    AssignOpConversion<BlockingAssignOp>,
    AssignOpConversion<NonBlockingAssignOp>,
    AssignOpConversion<DelayedNonBlockingAssignOp>,
    AssignedVariableOpConversion,

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
    FormatTimeOpConversion,
    FormatRealOpConversion,
    DisplayBIOpConversion,

    // Dynamic string operations.
    StringLenOpConversion,
    StringConcatOpConversion,

    // Queue operations.
    QueueSizeBIOpConversion,
    QueuePushBackOpConversion,
    QueuePushFrontOpConversion,
    QueuePopBackOpConversion,
    QueuePopFrontOpConversion,
    QueueDeleteOpConversion,
    QueueInsertOpConversion,
    QueueClearOpConversion,
    DynQueueExtractOpConversion
  >(typeConverter, patterns.getContext());
  // clang-format on

  // Structural operations
  patterns.add<WaitDelayOp>(convert);
  patterns.add<UnreachableOp>(convert);
  patterns.add<GlobalVariableOp>(convert);
  patterns.add<GetGlobalVariableOp>(convert);

  // Simulation control
  patterns.add<StopBIOp>(convert);
  patterns.add<SeverityBIOp>(convert);
  patterns.add<TimeFormatBIOp>(convert);
  patterns.add<FinishBIOp>(convert);
  patterns.add<FinishMessageBIOp>(convert);

  // Timing control
  patterns.add<TimeBIOp>(convert);
  patterns.add<LogicToTimeOp>(convert);
  patterns.add<TimeToLogicOp>(convert);

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
  ClassTypeCache classCache;

  context.getOrLoadDialect<sv::SVDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  IRRewriter rewriter(module);
  (void)mlir::eraseUnreachableBlocks(rewriter, module->getRegions());

  TypeConverter typeConverter;
  populateTypeConversion(typeConverter);

  ConversionTarget target(context);
  populateLegality(target, typeConverter);

  ConversionPatternSet patterns(&context, typeConverter);
  populateOpConversion(patterns, typeConverter, classCache);
  mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                           patterns, target);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
