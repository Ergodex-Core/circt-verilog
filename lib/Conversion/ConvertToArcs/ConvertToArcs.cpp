//===- ConvertToArcs.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Support/Namespace.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>

#define DEBUG_TYPE "convert-to-arcs"

using namespace circt;
using namespace arc;
using namespace hw;
using llvm::MapVector;
using llvm::SmallSetVector;
using mlir::ConversionConfig;

static constexpr StringLiteral kArcilatorProcIdAttr = "arcilator.proc_id";
static constexpr StringLiteral kArcilatorWaitIdAttr = "arcilator.wait_id";
static constexpr StringLiteral kArcilatorSigIdAttr = "arcilator.sig_id";
static constexpr StringLiteral kArcilatorSigOffsetAttr = "arcilator.sig_offset";
static constexpr StringLiteral kArcilatorSigTotalWidthAttr =
    "arcilator.sig_total_width";
static constexpr StringLiteral kArcilatorSigInitU64Attr =
    "arcilator.sig_init_u64";
static constexpr StringLiteral kArcilatorSigDynExtractAttr =
    "arcilator.sig_dyn_extract";
static constexpr StringLiteral kArcilatorNeedsSchedulerAttr =
    "arcilator.needs_scheduler";
static constexpr StringLiteral kArcilatorSigInitsAttr =
    "arcilator.sig_inits";

struct ArcilatorRuntimeSigInit {
  uint64_t initU64 = 0;
  uint64_t totalWidth = 0;
};

static Value stripCasts(Value value) {
  while (auto castOp =
             value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() != 1)
      break;
    value = castOp.getInputs().front();
  }
  return value;
}

static bool isFourStateValueUnknownStruct(Type ty, unsigned &fieldWidth) {
  auto structTy = dyn_cast<hw::StructType>(ty);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (elements[0].name.getValue() != "value" ||
      elements[1].name.getValue() != "unknown")
    return false;
  auto valueTy = dyn_cast<IntegerType>(elements[0].type);
  auto unknownTy = dyn_cast<IntegerType>(elements[1].type);
  if (!valueTy || !unknownTy || valueTy.getWidth() != unknownTy.getWidth())
    return false;
  fieldWidth = static_cast<unsigned>(valueTy.getWidth());
  return true;
}

static std::optional<APInt> tryEvalIntConstant(Value value, unsigned bitWidth) {
  value = stripCasts(value);
  APInt bits;
  if (matchPattern(value, mlir::m_ConstantInt(&bits)))
    return bits.zextOrTrunc(bitWidth);
  return std::nullopt;
}

static Type stripInOutAndAliasTypes(Type ty) {
  if (auto inoutTy = dyn_cast<hw::InOutType>(ty))
    ty = inoutTy.getElementType();
  while (auto alias = dyn_cast<hw::TypeAliasType>(ty))
    ty = alias.getInnerType();
  return ty;
}

static std::optional<APInt> tryEvalRuntimeSignalInitAttr(Attribute initAttr,
                                                         Type elemTy);

static std::optional<APInt>
tryEvalRuntimeSignalInitStructFields(ArrayRef<Attribute> fields,
                                     hw::StructType structTy) {
  const unsigned structWidth =
      static_cast<unsigned>(hw::getBitWidth(structTy));
  if (structWidth == 0 || structWidth > 64)
    return std::nullopt;

  auto elements = structTy.getElements();
  if (fields.size() != elements.size())
    return std::nullopt;

  APInt packed(structWidth, 0);
  for (unsigned idx = 0, e = elements.size(); idx < e; ++idx) {
    Type fieldTy = elements[idx].type;
    int64_t fieldWidthSigned = hw::getBitWidth(fieldTy);
    if (fieldWidthSigned <= 0)
      return std::nullopt;
    unsigned fieldWidth = static_cast<unsigned>(fieldWidthSigned);
    if (fieldWidth > 64)
      return std::nullopt;

    // Match HW struct packing (MSB-first). Keep the special-case 4-state
    // `{value, unknown}` encoding stable by treating `value` as low bits.
    uint64_t fieldOffsetLSB = 0;
    if (elements.size() == 2 && elements[0].name.getValue() == "value" &&
        elements[1].name.getValue() == "unknown") {
      if (idx == 0) {
        fieldOffsetLSB = 0;
      } else {
        fieldOffsetLSB = static_cast<uint64_t>(fieldWidth);
      }
    } else {
      uint64_t prefixWidth = 0;
      for (unsigned i = 0; i < idx; ++i) {
        int64_t w = hw::getBitWidth(elements[i].type);
        if (w <= 0)
          return std::nullopt;
        prefixWidth += static_cast<uint64_t>(w);
      }
      uint64_t structWidthU = static_cast<uint64_t>(structWidth);
      uint64_t fieldWidthU = static_cast<uint64_t>(fieldWidth);
      if (prefixWidth + fieldWidthU > structWidthU)
        return std::nullopt;
      fieldOffsetLSB = structWidthU - prefixWidth - fieldWidthU;
    }

    auto fieldBits = tryEvalRuntimeSignalInitAttr(fields[idx], fieldTy);
    if (!fieldBits)
      return std::nullopt;
    APInt slice = fieldBits->zextOrTrunc(fieldWidth);
    packed |= slice.zext(structWidth).shl(fieldOffsetLSB);
  }
  return packed;
}

static std::optional<APInt> tryEvalRuntimeSignalInitAttr(Attribute initAttr,
                                                         Type elemTy) {
  elemTy = stripInOutAndAliasTypes(elemTy);

  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    auto intAttr = dyn_cast<IntegerAttr>(initAttr);
    if (!intAttr)
      return std::nullopt;
    return intAttr.getValue().zextOrTrunc(static_cast<unsigned>(intTy.getWidth()));
  }

  unsigned fieldWidth = 0;
  if (isFourStateValueUnknownStruct(elemTy, fieldWidth)) {
    auto fields = dyn_cast<ArrayAttr>(initAttr);
    if (!fields || fields.size() != 2)
      return std::nullopt;
    auto valueAttr = dyn_cast<IntegerAttr>(fields[0]);
    auto unknownAttr = dyn_cast<IntegerAttr>(fields[1]);
    if (!valueAttr || !unknownAttr)
      return std::nullopt;
    APInt valueBits = valueAttr.getValue().zextOrTrunc(fieldWidth);
    APInt unknownBits = unknownAttr.getValue().zextOrTrunc(fieldWidth);
    unsigned totalWidth = 2 * fieldWidth;
    APInt packed =
        unknownBits.zext(totalWidth).shl(fieldWidth) | valueBits.zext(totalWidth);
    return packed;
  }

  if (auto structTy = dyn_cast<hw::StructType>(elemTy)) {
    auto fieldsAttr = dyn_cast<ArrayAttr>(initAttr);
    if (!fieldsAttr)
      return std::nullopt;
    return tryEvalRuntimeSignalInitStructFields(fieldsAttr.getValue(), structTy);
  }

  // Note: arrays/memories are currently not supported for runtime-managed
  // signal init evaluation. Extend as needed.
  return std::nullopt;
}

static std::optional<APInt> tryEvalRuntimeSignalInit(Value init, Type elemTy) {
  init = stripCasts(init);
  elemTy = stripInOutAndAliasTypes(elemTy);

  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    auto bits = tryEvalIntConstant(init, static_cast<unsigned>(intTy.getWidth()));
    if (!bits)
      return std::nullopt;
    return *bits;
  }

  unsigned fieldWidth = 0;
  if (!isFourStateValueUnknownStruct(elemTy, fieldWidth))
    return std::nullopt;

  // struct<value,unknown> initializer: pack as [value (low), unknown (high)].
  if (auto create = init.getDefiningOp<hw::StructCreateOp>()) {
    if (create.getOperands().size() != 2)
      return std::nullopt;
    auto valueBits = tryEvalIntConstant(create.getOperands()[0], fieldWidth);
    auto unknownBits = tryEvalIntConstant(create.getOperands()[1], fieldWidth);
    if (!valueBits || !unknownBits)
      return std::nullopt;
    unsigned totalWidth = 2 * fieldWidth;
    APInt packed = unknownBits->zext(totalWidth).shl(fieldWidth) |
                   valueBits->zext(totalWidth);
    return packed;
  }

  if (auto agg = init.getDefiningOp<hw::AggregateConstantOp>()) {
    return tryEvalRuntimeSignalInitAttr(agg.getFields(), elemTy);
  }

  // Already-packed integer constant initializer.
  if (auto cst = init.getDefiningOp<hw::ConstantOp>()) {
    unsigned totalWidth = 2 * fieldWidth;
    return cst.getValue().zextOrTrunc(totalWidth);
  }

  return std::nullopt;
}

static std::optional<APInt> tryDefaultRuntimeSignalInit(Type elemTy) {
  elemTy = stripInOutAndAliasTypes(elemTy);

  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return APInt(static_cast<unsigned>(intTy.getWidth()), 0);

  unsigned fieldWidth = 0;
  if (isFourStateValueUnknownStruct(elemTy, fieldWidth)) {
    unsigned totalWidth = 2 * fieldWidth;
    return APInt::getAllOnes(totalWidth);
  }

  if (auto arrayTy = dyn_cast<hw::ArrayType>(elemTy)) {
    Type elem = arrayTy.getElementType();
    int64_t elemWidthSigned = hw::getBitWidth(elem);
    if (elemWidthSigned <= 0)
      return std::nullopt;
    unsigned elemWidth = static_cast<unsigned>(elemWidthSigned);
    uint64_t numElems = arrayTy.getNumElements();
    uint64_t totalWidthU = numElems * static_cast<uint64_t>(elemWidth);
    if (totalWidthU == 0 || totalWidthU > 64)
      return std::nullopt;
    unsigned totalWidth = static_cast<unsigned>(totalWidthU);
    APInt packed(totalWidth, 0);
    for (uint64_t idx = 0; idx < numElems; ++idx) {
      auto bits = tryDefaultRuntimeSignalInit(elem);
      if (!bits)
        return std::nullopt;
      APInt slice = bits->zextOrTrunc(elemWidth);
      // HW array packing is MSB-first (element 0 at the top).
      uint64_t offsetLSB = totalWidthU - (idx + 1) * static_cast<uint64_t>(elemWidth);
      packed |= slice.zext(totalWidth).shl(offsetLSB);
    }
    return packed;
  }

  if (auto arrayTy = dyn_cast<hw::UnpackedArrayType>(elemTy)) {
    Type elem = arrayTy.getElementType();
    int64_t elemWidthSigned = hw::getBitWidth(elem);
    if (elemWidthSigned <= 0)
      return std::nullopt;
    unsigned elemWidth = static_cast<unsigned>(elemWidthSigned);
    uint64_t numElems = arrayTy.getNumElements();
    uint64_t totalWidthU = numElems * static_cast<uint64_t>(elemWidth);
    if (totalWidthU == 0 || totalWidthU > 64)
      return std::nullopt;
    unsigned totalWidth = static_cast<unsigned>(totalWidthU);
    APInt packed(totalWidth, 0);
    for (uint64_t idx = 0; idx < numElems; ++idx) {
      auto bits = tryDefaultRuntimeSignalInit(elem);
      if (!bits)
        return std::nullopt;
      APInt slice = bits->zextOrTrunc(elemWidth);
      uint64_t offsetLSB = totalWidthU - (idx + 1) * static_cast<uint64_t>(elemWidth);
      packed |= slice.zext(totalWidth).shl(offsetLSB);
    }
    return packed;
  }

  if (auto structTy = dyn_cast<hw::StructType>(elemTy)) {
    const unsigned structWidth =
        static_cast<unsigned>(hw::getBitWidth(structTy));
    if (structWidth == 0 || structWidth > 64)
      return std::nullopt;

    APInt packed(structWidth, 0);
    auto elements = structTy.getElements();
    for (unsigned idx = 0, e = elements.size(); idx < e; ++idx) {
      Type fieldTy = elements[idx].type;
      int64_t fieldWidthSigned = hw::getBitWidth(fieldTy);
      if (fieldWidthSigned <= 0)
        return std::nullopt;
      unsigned fieldWidth = static_cast<unsigned>(fieldWidthSigned);
      if (fieldWidth > 64)
        return std::nullopt;

      uint64_t fieldOffsetLSB = 0;
      uint64_t prefixWidth = 0;
      for (unsigned i = 0; i < idx; ++i) {
        int64_t w = hw::getBitWidth(elements[i].type);
        if (w <= 0)
          return std::nullopt;
        prefixWidth += static_cast<uint64_t>(w);
      }
      uint64_t structWidthU = static_cast<uint64_t>(structWidth);
      uint64_t fieldWidthU = static_cast<uint64_t>(fieldWidth);
      if (prefixWidth + fieldWidthU > structWidthU)
        return std::nullopt;
      fieldOffsetLSB = structWidthU - prefixWidth - fieldWidthU;

      auto fieldBits = tryDefaultRuntimeSignalInit(fieldTy);
      if (!fieldBits)
        return std::nullopt;
      APInt slice = fieldBits->zextOrTrunc(fieldWidth);
      packed |= slice.zext(structWidth).shl(fieldOffsetLSB);
    }
    return packed;
  }

  return std::nullopt;
}

static Attribute getZeroHWAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  if (auto arrayTy = dyn_cast<hw::ArrayType>(type)) {
    Attribute elementZero = getZeroHWAttr(builder, arrayTy.getElementType());
    if (!elementZero)
      return {};
    SmallVector<Attribute> elements(arrayTy.getNumElements(), elementZero);
    return builder.getArrayAttr(elements);
  }
  if (auto arrayTy = dyn_cast<hw::UnpackedArrayType>(type)) {
    Attribute elementZero = getZeroHWAttr(builder, arrayTy.getElementType());
    if (!elementZero)
      return {};
    SmallVector<Attribute> elements(arrayTy.getNumElements(), elementZero);
    return builder.getArrayAttr(elements);
  }
  if (auto structTy = dyn_cast<hw::StructType>(type)) {
    SmallVector<Attribute> fields;
    fields.reserve(structTy.getElements().size());
    for (auto field : structTy.getElements()) {
      Attribute fieldZero = getZeroHWAttr(builder, field.type);
      if (!fieldZero)
        return {};
      fields.push_back(fieldZero);
    }
    return builder.getArrayAttr(fields);
  }
  return {};
}

static Value createZeroHWConstant(OpBuilder &builder, Location loc, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return hw::ConstantOp::create(builder, loc, type,
                                  builder.getIntegerAttr(intTy, 0));
  Attribute zero = getZeroHWAttr(builder, type);
  if (auto arrayAttr = dyn_cast_or_null<ArrayAttr>(zero))
    return hw::AggregateConstantOp::create(builder, loc, type, arrayAttr);
  return {};
}

static mlir::func::FuncOp getOrInsertFunc(mlir::ModuleOp module, StringRef name,
                                          mlir::FunctionType type) {
  if (!module)
    return {};
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name))
    return fn;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<mlir::func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static Value buildI32Constant(OpBuilder &builder, Location loc, uint32_t value) {
  return hw::ConstantOp::create(builder, loc, APInt(32, value));
}

static Value buildI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return hw::ConstantOp::create(builder, loc, APInt(64, value));
}

static FailureOr<uint64_t> tryExtractDelayFs(Value delay) {
  if (!delay)
    return failure();
  auto cst = delay.getDefiningOp<llhd::ConstantTimeOp>();
  if (!cst)
    return failure();
  auto timeAttr = cst.getValue();
  uint64_t scale = llvm::StringSwitch<uint64_t>(timeAttr.getTimeUnit())
                       .Case("fs", 1ULL)
                       .Case("ps", 1000ULL)
                       .Case("ns", 1000ULL * 1000ULL)
                       .Case("us", 1000ULL * 1000ULL * 1000ULL)
                       .Case("ms", 1000ULL * 1000ULL * 1000ULL * 1000ULL)
                       .Case("s", 1000ULL * 1000ULL * 1000ULL * 1000ULL *
                                     1000ULL)
                       .Default(0);
  if (scale == 0)
    return failure();
  return timeAttr.getTime() * scale;
}

static bool needsCycleScheduler(llhd::ProcessOp op) {
  // The current scheduler lowering does not model LLHD process results (wait/halt
  // yields). Pre-lowering may rewrite some resultful processes into direct
  // signal drives, at which point they become eligible for cycle scheduling.
  if (!op.getResults().empty())
    return false;
  // Also schedule one-shot processes that end in `llhd.halt` so we can model the
  // "run once then stop" semantics via the runtime PC state.
  bool hasHalt = !op.getOps<llhd::HaltOp>().empty();
  bool hasWait = false;
  bool hasDelay = false;
  bool hasObserved = false;
  op.walk([&](llhd::WaitOp wait) {
    hasWait = true;
    if (wait.getDelay())
      hasDelay = true;
    if (!wait.getObserved().empty())
      hasObserved = true;
  });
  return hasHalt || hasWait || hasDelay || hasObserved;
}

static bool isRematerializableCallForPolling(mlir::func::CallOp call) {
  auto callee = call.getCallee();
  if (callee.empty())
    return false;

  // Observed values in scheduled waits often depend on runtime "getter" shims
  // (e.g. virtual interface indirections). Treat these calls as rematerializable
  // since they are side-effect free and deterministic with respect to runtime
  // state.
  return callee == "circt_sv_class_get_ptr" ||
         callee == "circt_sv_class_get_i32" ||
         callee == "circt_sv_class_get_type" ||
         callee == "circt_sv_dynarray_size_i32" ||
         callee == "circt_sv_dynarray_get_i32" ||
         callee == "circt_sv_queue_size_i32" ||
         callee == "circt_sv_mailbox_num_i32" ||
         callee == "circt_sv_assoc_exists_str_i32" ||
         callee == "circt_sv_assoc_get_str_i32" ||
         callee == "__arcilator_sig_load_u64" ||
         callee == "__arcilator_frame_load_u64" ||
         callee == "circt_uvm_coreservice_get" ||
         callee == "circt_uvm_coreservice_get_root" ||
         callee == "circt_uvm_root_get" ||
         callee == "circt_uvm_report_server_get_server" ||
         callee == "circt_uvm_get_severity_count" ||
         callee == "circt_uvm_phase_all_done" ||
         callee == "circt_uvm_component_count" ||
         callee == "circt_uvm_component_get" ||
         callee == "circt_uvm_component_get_full_name" ||
         callee == "circt_uvm_resource_db_get_i32" ||
         callee == "circt_uvm_resource_db_get_ptr";
}

static bool isRematerializableForPolling(Operation *op) {
  if (!op)
    return false;
  if (op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (auto call = dyn_cast<mlir::func::CallOp>(op))
    return isRematerializableCallForPolling(call);
  // Treat LLHD signal declarations as rematerializable handles. For scheduled
  // processes we lower signals into runtime-managed storage keyed by a stable
  // id, so cloning the declaration is equivalent to duplicating the handle.
  if (isa<llhd::SignalOp>(op))
    return true;
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (mlir::isMemoryEffectFree(op))
    return true;
  auto effects = dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
      effectList;
  effects.getEffects(effectList);
  for (auto &effect : effectList)
    if (!isa<mlir::MemoryEffects::Read>(effect.getEffect()))
      return false;
  return true;
}

static FailureOr<Value>
rematerializeValueForPolling(Value value, DenseMap<Value, Value> &memo,
                             mlir::RewriterBase &rewriter) {
  if (!value)
    return failure();
  if (auto it = memo.find(value); it != memo.end())
    return it->second;

  if (isa<BlockArgument>(value)) {
    memo.try_emplace(value, value);
    return value;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return failure();
  if (!isRematerializableForPolling(defOp))
    return failure();

  IRMapping mapping;
  for (Value operand : defOp->getOperands()) {
    auto remat = rematerializeValueForPolling(operand, memo, rewriter);
    if (failed(remat))
      return failure();
    mapping.map(operand, *remat);
  }

  Operation *cloned = rewriter.clone(*defOp, mapping);
  for (auto [from, to] :
       llvm::zip(defOp->getResults(), cloned->getResults()))
    memo.try_emplace(from, to);

  auto it = memo.find(value);
  if (it == memo.end())
    return failure();
  return it->second;
}

static LogicalResult lowerCycleScheduler(ExecuteOp execOp, uint32_t procId,
                                         mlir::RewriterBase &rewriter) {
  Region &region = execOp.getBody();
  if (region.empty())
    return success();

  Block *entryBlock = &region.front();

  auto module = execOp->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return execOp.emitOpError() << "missing module for scheduler lowering";

  // Runtime hooks (implemented by the autogenerated driver).
  (void)getOrInsertFunc(
      module, "__arcilator_get_pc",
      rewriter.getFunctionType({rewriter.getI32Type()}, {rewriter.getI32Type()}));
  (void)getOrInsertFunc(
      module, "__arcilator_set_pc",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                               {}));
  (void)getOrInsertFunc(
      module, "__arcilator_wait_delay",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI64Type()},
                               {rewriter.getI1Type()}));
  (void)getOrInsertFunc(
      module, "__arcilator_wait_change",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI64Type()},
                               {rewriter.getI1Type()}));
  (void)getOrInsertFunc(
      module, "__arcilator_frame_store_u64",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type(),
                                rewriter.getI64Type()},
                               {}));
  (void)getOrInsertFunc(
      module, "__arcilator_frame_load_u64",
      rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                               {rewriter.getI64Type()}));

  // String frame hooks are inserted lazily with the exact string type used in
  // the scheduler frame (to avoid type-alias mismatches).
  Type frameStringTy;
  auto ensureFrameStringFuncs = [&](Type ty) -> LogicalResult {
    Type baseTy = ty;
    while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
      baseTy = alias.getInnerType();
    if (!isa<hw::StringType>(baseTy))
      return success();

    // These are module-global runtime hooks. If another scheduled process has
    // already inserted them, require the same string type to avoid generating
    // invalid IR (func.call operand/result types must match the callee type).
    if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(
            "__arcilator_frame_store_str")) {
      if (fn.getFunctionType().getNumInputs() != 3)
        return execOp.emitOpError()
               << "__arcilator_frame_store_str has unexpected signature";
      Type existingTy = fn.getFunctionType().getInput(2);
      if (existingTy != ty)
        return execOp.emitOpError()
               << "string frame type mismatch for __arcilator_frame_store_str ("
               << existingTy << " vs " << ty << ")";
      frameStringTy = existingTy;
      return success();
    }

    if (frameStringTy && frameStringTy != ty)
      return execOp.emitOpError()
             << "cannot lower scheduled process with multiple distinct string "
                "types ("
             << frameStringTy << " vs " << ty << ")";
    if (!frameStringTy) {
      frameStringTy = ty;
      OpBuilder builder(module.getBodyRegion());
      builder.setInsertionPointToStart(module.getBody());
      auto storeTy = rewriter.getFunctionType(
          {rewriter.getI32Type(), rewriter.getI32Type(), frameStringTy}, {});
      auto loadTy = rewriter.getFunctionType(
          {rewriter.getI32Type(), rewriter.getI32Type()}, {frameStringTy});
      auto fnStore = builder.create<mlir::func::FuncOp>(
          module.getLoc(), "__arcilator_frame_store_str", storeTy);
      fnStore.setPrivate();
      auto fnLoad = builder.create<mlir::func::FuncOp>(
          module.getLoc(), "__arcilator_frame_load_str", loadTy);
      fnLoad.setPrivate();
    }
    return success();
  };

  // Insert the dispatch block as the new entry. If the original entry block has
  // captured values as arguments, move those captures to the dispatch block so
  // they remain available when the scheduler jumps directly to a wait state.
  SmallVector<Type> entryArgTypes;
  SmallVector<Location> entryArgLocs;
  entryArgTypes.reserve(entryBlock->getNumArguments());
  entryArgLocs.reserve(entryBlock->getNumArguments());
  for (BlockArgument arg : entryBlock->getArguments()) {
    entryArgTypes.push_back(arg.getType());
    entryArgLocs.push_back(arg.getLoc());
  }
  Block *dispatchBlock =
      rewriter.createBlock(&region, region.begin(), entryArgTypes, entryArgLocs);
  for (auto [oldArg, newArg] :
       llvm::zip(entryBlock->getArguments(), dispatchBlock->getArguments()))
    oldArg.replaceAllUsesWith(newArg);
  while (entryBlock->getNumArguments() != 0)
    entryBlock->eraseArgument(0);

  rewriter.setInsertionPointToEnd(dispatchBlock);
  Location loc = execOp.getLoc();
  Value procIdVal = buildI32Constant(rewriter, loc, procId);

  // Create a common exit block that returns from the execute region.
  Block *exitBlock = rewriter.createBlock(&region);
  rewriter.setInsertionPointToEnd(exitBlock);
  arc::OutputOp::create(rewriter, loc, ValueRange{});

  // Helpers to pack/unpack values stored in the cycle-scheduler frame.
  auto canFrameType = [&](Type ty) -> bool {
    Type baseTy = ty;
    while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
      baseTy = alias.getInnerType();
    if (isa<hw::InOutType>(baseTy))
      return false;
    if (isa<hw::StringType>(baseTy))
      return true;
    int64_t bw = hw::getBitWidth(baseTy);
    if (bw < 0 || bw > 64)
      return false;
    // Zero-width values (e.g. array indices for 1-element arrays) still need to
    // be treated as spillable when we add new scheduler dispatch predecessors.
    // They can be represented as a constant 0 in the frame.
    if (bw == 0)
      return isa<IntegerType>(ty);
    return true;
  };
  auto packToI64 = [&](Value value) -> FailureOr<Value> {
    Type ty = value.getType();
    int64_t bw = hw::getBitWidth(ty);
    if (bw < 0 || bw > 64)
      return failure();
    Location loc = value.getLoc();
    if (bw == 0)
      return buildI64Constant(rewriter, loc, 0);
    Value bits = value;
    auto intTy = rewriter.getIntegerType(static_cast<unsigned>(bw));
    if (!isa<IntegerType>(ty))
      bits = rewriter.createOrFold<hw::BitcastOp>(loc, intTy, value);
    if (bw < 64)
      bits = comb::createZExt(rewriter, loc, bits, 64);
    return bits;
  };
  auto unpackFromI64 = [&](Value packed, Type ty,
                           Location loc) -> FailureOr<Value> {
    int64_t bw = hw::getBitWidth(ty);
    if (bw < 0 || bw > 64)
      return failure();
    if (bw == 0) {
      Value zero = createZeroHWConstant(rewriter, loc, ty);
      if (!zero)
        return failure();
      return zero;
    }
    Value bits = packed;
    if (bw < 64)
      bits = comb::ExtractOp::create(rewriter, loc, packed, 0,
                                     static_cast<unsigned>(bw));
    if (isa<IntegerType>(ty))
      return bits;
    auto intTy = rewriter.getIntegerType(static_cast<unsigned>(bw));
    return rewriter.createOrFold<hw::BitcastOp>(loc, ty, bits);
  };

  // Lower SSA block arguments (phi nodes) by modeling branch operands as frame
  // stores and replacing the block arguments with frame loads. This avoids
  // having to pass operands when the scheduler jumps directly to a state.
  DenseMap<Block *, SmallVector<std::pair<BlockArgument, uint32_t>>>
      phiArgsByBlock;
  uint32_t nextPhiSlot = 0;
  for (Block &block : region) {
    if (&block == dispatchBlock || &block == exitBlock)
      continue;
    if (block.getArguments().empty())
      continue;
    auto &args = phiArgsByBlock[&block];
    for (BlockArgument arg : block.getArguments()) {
      if (!canFrameType(arg.getType()))
        return execOp.emitOpError()
               << "cannot lower scheduled process with block argument of type "
               << arg.getType();
      args.push_back({arg, nextPhiSlot++});
    }
  }

  auto emitPhiStores = [&](Location storeLoc, Block *dest,
                           ValueRange operands) -> LogicalResult {
    auto it = phiArgsByBlock.find(dest);
    if (it == phiArgsByBlock.end())
      return success();
    auto &args = it->second;
    if (operands.size() != args.size())
      return failure();
    for (auto [operand, argInfo] : llvm::zip(operands, args)) {
      Value slotVal = buildI32Constant(rewriter, storeLoc, argInfo.second);
      Value argVal = argInfo.first;
      if (!operand || !argVal)
        return execOp.emitOpError()
               << "null operand while lowering scheduler frame phi stores";
      Type argTy = argVal.getType();
      Type operandTy = operand.getType();
      if (!operandTy || !argTy)
        return execOp.emitOpError()
               << "scheduler frame phi operand has null type";
      if (operandTy != argTy)
        return execOp.emitOpError()
               << "scheduler frame phi operand type mismatch";
      Type baseTy = argTy;
      while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
        baseTy = alias.getInnerType();
      if (isa<hw::StringType>(baseTy)) {
        if (failed(ensureFrameStringFuncs(argTy)))
          return failure();
        rewriter.create<mlir::func::CallOp>(
            storeLoc, "__arcilator_frame_store_str", TypeRange{},
            ValueRange{procIdVal, slotVal, operand});
      } else {
        auto packed = packToI64(operand);
        if (failed(packed))
          return failure();
        rewriter.create<mlir::func::CallOp>(
            storeLoc, "__arcilator_frame_store_u64", TypeRange{},
            ValueRange{procIdVal, slotVal, *packed});
      }
    }
    return success();
  };

  if (!phiArgsByBlock.empty()) {
    // Rewrite branches to perform frame stores along edges that would have
    // supplied phi operands.
    SmallVector<Operation *> terminators;
    terminators.reserve(region.getBlocks().size());
    for (Block &block : region)
      terminators.push_back(block.getTerminator());

    for (Operation *terminator : terminators) {
      if (auto br = dyn_cast<mlir::cf::BranchOp>(terminator)) {
        Block *dest = br.getDest();
        if (phiArgsByBlock.contains(dest)) {
          rewriter.setInsertionPoint(br);
          if (failed(emitPhiStores(br.getLoc(), dest, br.getDestOperands())))
            return br.emitOpError() << "failed to lower branch block arguments";
          rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(br, dest);
        }
        continue;
      }

      if (auto br = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
        Block *trueDest = br.getTrueDest();
        Block *falseDest = br.getFalseDest();
        bool needsTrueStores = phiArgsByBlock.contains(trueDest);
        bool needsFalseStores = phiArgsByBlock.contains(falseDest);
        if (!needsTrueStores && !needsFalseStores)
          continue;

        auto insertIt = std::next(Region::iterator(br->getBlock()));

        Block *newTrueDest = trueDest;
        if (needsTrueStores) {
          auto *storeBlock = rewriter.createBlock(&region, insertIt);
          rewriter.setInsertionPointToEnd(storeBlock);
          if (failed(emitPhiStores(br.getLoc(), trueDest,
                                  br.getTrueDestOperands())))
            return br.emitOpError()
                   << "failed to lower true-branch block arguments";
          mlir::cf::BranchOp::create(rewriter, br.getLoc(), trueDest);
          newTrueDest = storeBlock;
        }

        Block *newFalseDest = falseDest;
        if (needsFalseStores) {
          auto *storeBlock = rewriter.createBlock(&region, insertIt);
          rewriter.setInsertionPointToEnd(storeBlock);
          if (failed(emitPhiStores(br.getLoc(), falseDest,
                                  br.getFalseDestOperands())))
            return br.emitOpError()
                   << "failed to lower false-branch block arguments";
          mlir::cf::BranchOp::create(rewriter, br.getLoc(), falseDest);
          newFalseDest = storeBlock;
        }

        rewriter.setInsertionPoint(br);
        rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
            br, br.getCondition(), newTrueDest, ValueRange{}, newFalseDest,
            ValueRange{});
        continue;
      }

      if (auto sw = dyn_cast<mlir::cf::SwitchOp>(terminator)) {
        bool needsRewrite = phiArgsByBlock.contains(sw.getDefaultDestination());
        for (Block *dest : sw.getCaseDestinations())
          needsRewrite |= phiArgsByBlock.contains(dest);
        if (!needsRewrite)
          continue;

        auto insertIt = std::next(Region::iterator(sw->getBlock()));

        auto makeStoreBlock = [&](Block *dest, ValueRange operands)
            -> FailureOr<Block *> {
          if (!phiArgsByBlock.contains(dest))
            return dest;
          auto *storeBlock = rewriter.createBlock(&region, insertIt);
          rewriter.setInsertionPointToEnd(storeBlock);
          if (failed(emitPhiStores(sw.getLoc(), dest, operands)))
            return failure();
          mlir::cf::BranchOp::create(rewriter, sw.getLoc(), dest);
          return storeBlock;
        };

        auto newDefault = makeStoreBlock(sw.getDefaultDestination(),
                                         sw.getDefaultOperands());
        if (failed(newDefault))
          return sw.emitOpError() << "failed to lower default block arguments";

        SmallVector<Block *> newCaseDests;
        SmallVector<ValueRange> newCaseOperands;
        for (auto [dest, operands] :
             llvm::zip(sw.getCaseDestinations(), sw.getCaseOperands())) {
          auto newDest = makeStoreBlock(dest, operands);
          if (failed(newDest))
            return sw.emitOpError() << "failed to lower case block arguments";
          newCaseDests.push_back(*newDest);
          newCaseOperands.push_back(ValueRange{});
        }

        rewriter.setInsertionPoint(sw);
        auto caseValuesAttr = sw.getCaseValuesAttr();
        rewriter.replaceOpWithNewOp<mlir::cf::SwitchOp>(
            sw, sw.getFlag(), *newDefault, ValueRange{}, caseValuesAttr,
            mlir::BlockRange(newCaseDests), newCaseOperands);
        continue;
      }
    }

    // Replace the phi arguments with loads in their destination blocks and
    // erase the arguments.
    for (auto &it : phiArgsByBlock) {
      Block *block = it.first;
      auto &args = it.second;
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(block);
      for (auto [arg, slot] : args) {
        Value slotVal = buildI32Constant(rewriter, arg.getLoc(), slot);
        Type argTy = arg.getType();
        Type baseTy = argTy;
        while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
          baseTy = alias.getInnerType();
        if (isa<hw::StringType>(baseTy)) {
          if (failed(ensureFrameStringFuncs(argTy)))
            return failure();
          Value loaded =
              rewriter
                  .create<mlir::func::CallOp>(arg.getLoc(),
                                              "__arcilator_frame_load_str",
                                              argTy,
                                              ValueRange{procIdVal, slotVal})
                  .getResult(0);
          arg.replaceAllUsesWith(loaded);
        } else {
          Value loaded =
              rewriter
                  .create<mlir::func::CallOp>(arg.getLoc(),
                                              "__arcilator_frame_load_u64",
                                              rewriter.getI64Type(),
                                              ValueRange{procIdVal, slotVal})
                  .getResult(0);
          auto unpacked = unpackFromI64(loaded, arg.getType(), arg.getLoc());
          if (failed(unpacked))
            return execOp.emitOpError()
                   << "cannot reload block argument of type " << arg.getType();
          arg.replaceAllUsesWith(*unpacked);
        }
      }
      while (block->getNumArguments() != 0)
        block->eraseArgument(0);
    }
  }
  uint32_t spillSlotBase = nextPhiSlot;

  // If there are no waits, avoid the full PC-based state dispatch and hoisting.
  // We only need to model "run once then stop" semantics for one-shot processes
  // ending in `llhd.halt`.
  SmallVector<llhd::WaitOp> waits;
  execOp.walk([&](llhd::WaitOp w) { waits.push_back(w); });

  // Replace halts with a pc update + exit.
  SmallVector<llhd::HaltOp> halts;
  execOp.walk([&](llhd::HaltOp h) { halts.push_back(h); });
  for (auto haltOp : halts) {
    if (!haltOp.getYieldOperands().empty())
      return haltOp.emitOpError() << "scheduled halt with yield operands unsupported";
    rewriter.setInsertionPoint(haltOp);
    // Use an out-of-range state id to make the dispatch default to exit.
    Value doneStateVal = buildI32Constant(rewriter, haltOp.getLoc(), 0xFFFFFFFFu);
    rewriter.create<mlir::func::CallOp>(haltOp.getLoc(), "__arcilator_set_pc",
                                        TypeRange{},
                                        ValueRange{procIdVal, doneStateVal});
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(haltOp, exitBlock);
  }

  if (waits.empty()) {
    rewriter.setInsertionPointToEnd(dispatchBlock);
    Value pc =
        rewriter
            .create<mlir::func::CallOp>(loc, "__arcilator_get_pc",
                                        rewriter.getI32Type(),
                                        ValueRange{procIdVal})
            .getResult(0);

    // Only state 0 is valid for one-shot processes without waits.
    rewriter.create<mlir::cf::SwitchOp>(loc, pc, exitBlock, ValueRange{},
                                        ArrayRef<int32_t>{0},
                                        ArrayRef<Block *>{entryBlock},
                                        ArrayRef<ValueRange>{ValueRange{}});
    return success();
  }

  // The scheduler dispatch can jump directly to any wait state, bypassing the
  // original entry block. Hoist any constant-like definitions that are shared
  // across blocks into the dispatch block so they dominate all possible entry
  // points.
  SmallVector<Operation *> hoistable;
  for (Block &block : region) {
    if (&block == dispatchBlock || &block == exitBlock)
      continue;
    for (Operation &op : block.without_terminator()) {
      if (!isRematerializableForPolling(&op))
        continue;
      // Avoid hoisting computed dataflow values into the dispatch block. Those
      // values may be semantically tied to a particular suspension point (e.g.
      // edge detection across `llhd.wait`) and are handled via frame-spilling
      // below. Only hoist constants and "handle" computations (inout handles /
      // signal declarations) to keep dominance valid without changing meaning.
      bool isHandleLike = isa<llhd::SignalOp>(op) || op.hasTrait<OpTrait::ConstantLike>();
      if (!isHandleLike) {
        isHandleLike = llvm::any_of(op.getResultTypes(), [](Type ty) {
          if (auto alias = dyn_cast<hw::TypeAliasType>(ty))
            ty = alias.getInnerType();
          // Treat opaque runtime "handle" types as hoistable. In particular,
          // sim formatting ops (`sim.fmt.*`) produce `!sim.fstring` values that
          // may be shared across scheduler states (e.g. error messages). These
          // are side-effect free and must dominate all possible state entries.
          return isa<hw::InOutType, hw::StringType, sim::FormatStringType>(ty);
        });
      }
      if (!isHandleLike)
        continue;
      bool usedOutsideBlock = false;
      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (use.getOwner()->getBlock() != &block) {
            usedOutsideBlock = true;
            break;
          }
        }
        if (usedOutsideBlock)
          break;
      }
      if (!usedOutsideBlock)
        continue;

      // Only hoist operations whose operands are themselves rematerializable
      // (or already dominate the dispatch block). Otherwise moving the op can
      // introduce new dominance violations.
      bool operandsOk = true;
      for (Value operand : op.getOperands()) {
        if (isa<BlockArgument>(operand))
          continue;
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          operandsOk = false;
          break;
        }
        if (defOp->getBlock() == dispatchBlock || defOp->getBlock() == exitBlock)
          continue;
        if (!isRematerializableForPolling(defOp)) {
          operandsOk = false;
          break;
        }
      }
      if (operandsOk)
        hoistable.push_back(&op);
    }
  }

  // Hoisting to the dispatch block changes the "used outside block" property
  // of all transitively referenced operands. Ensure we hoist a closed set of
  // rematerializable dependency ops to avoid introducing dominance violations.
  DenseSet<Operation *> hoistSet;
  SmallVector<Operation *> worklist;
  for (Operation *op : hoistable)
    if (hoistSet.insert(op).second)
      worklist.push_back(op);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (defOp->getBlock() == dispatchBlock || defOp->getBlock() == exitBlock)
        continue;
      if (!isRematerializableForPolling(defOp))
        continue;
      if (hoistSet.insert(defOp).second)
        worklist.push_back(defOp);
    }
  }

  SmallVector<Operation *> hoistOrder;
  DenseSet<Operation *> hoistVisited;
  auto visit = [&](Operation *op, auto &self) -> void {
    if (!op || !hoistSet.contains(op) || !hoistVisited.insert(op).second)
      return;
    for (Value operand : op->getOperands())
      if (Operation *defOp = operand.getDefiningOp())
        if (hoistSet.contains(defOp))
          self(defOp, self);
    hoistOrder.push_back(op);
  };

  // Visit hoist candidates in region order for deterministic output.
  for (Block &block : region) {
    if (&block == dispatchBlock || &block == exitBlock)
      continue;
    for (Operation &op : block.without_terminator())
      if (hoistSet.contains(&op))
        visit(&op, visit);
  }

  for (Operation *op : hoistOrder)
    op->moveBefore(dispatchBlock, dispatchBlock->end());

  // Split each `llhd.wait` into its own block so we don't re-run side effects in
  // the pre-wait block while waiting.
  SmallVector<std::pair<Block *, llhd::WaitOp>> waitBlocks;
  for (auto w : waits) {
    auto *parent = w->getBlock();
    auto insertIt = std::next(Region::iterator(parent));
    Block *waitBlock = rewriter.createBlock(&region, insertIt);
    w->moveBefore(waitBlock, waitBlock->end());
    rewriter.setInsertionPointToEnd(parent);
    mlir::cf::BranchOp::create(rewriter, loc, waitBlock);
    waitBlocks.push_back({waitBlock, w});
  }

  // Collect the blocks that represent resumable scheduler states.
  SmallVector<Block *> stateBlocks;
  stateBlocks.reserve(region.getBlocks().size());
  for (auto &block : region) {
    if (&block == dispatchBlock || &block == exitBlock)
      continue;
    stateBlocks.push_back(&block);
  }

  DenseMap<Block *, uint32_t> stateIds;
  for (auto [idx, block] : llvm::enumerate(stateBlocks))
    stateIds[block] = static_cast<uint32_t>(idx);

  // Ensure suspension always records the current wait state, regardless of how
  // control enters the wait block (conditional branches, loops, etc.).
  for (auto [waitBlock, waitOp] : waitBlocks) {
    uint32_t waitState = stateIds.lookup(waitBlock);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(waitBlock);
    Value waitStateVal = buildI32Constant(rewriter, loc, waitState);
    rewriter.create<mlir::func::CallOp>(waitOp.getLoc(), "__arcilator_set_pc",
                                        TypeRange{},
                                        ValueRange{procIdVal, waitStateVal});
  }

  // Replace wait terminators with runtime polling + branching.
  for (auto [waitBlock, waitOp] : waitBlocks) {
    if (!waitOp.getYieldOperands().empty())
      return waitOp.emitOpError() << "scheduled wait with yield operands unsupported";

    rewriter.setInsertionPoint(waitOp);

    auto waitIdAttr = waitOp->getAttrOfType<IntegerAttr>(kArcilatorWaitIdAttr);
    if (!waitIdAttr)
      return waitOp.emitOpError() << "missing wait id attribute";
    Value waitIdVal =
        buildI32Constant(rewriter, waitOp.getLoc(), waitIdAttr.getInt());

    Value ready;
    // If the wait resumes into a block with (lowered) phi arguments, store the
    // destination operands into the scheduler frame so the destination block's
    // frame loads observe the correct values when we resume.
    if (failed(emitPhiStores(waitOp.getLoc(), waitOp.getDest(),
                             waitOp.getDestOperands())))
      return waitOp.emitOpError()
             << "failed to lower scheduled wait dest operands";
    if (auto delay = waitOp.getDelay()) {
      auto delayFs = tryExtractDelayFs(delay);
      if (failed(delayFs))
        return waitOp.emitOpError() << "unsupported non-constant delay";
      Value delayFsVal =
          buildI64Constant(rewriter, waitOp.getLoc(), *delayFs);
      ready = rewriter
                  .create<mlir::func::CallOp>(waitOp.getLoc(),
                                              "__arcilator_wait_delay",
                                              rewriter.getI1Type(),
                                              ValueRange{waitIdVal, delayFsVal})
                  .getResult(0);
    } else if (!waitOp.getObserved().empty()) {
      // Combine observed values into a single signature to keep the runtime API
      // simple. Rematerialize observed reads into the wait block so each poll
      // sees the current value even when the scheduler jumps directly here.
      // NOTE: The signature must change for any observed change. A plain XOR
      // across observed fields is commutative and can cancel (e.g. 4-state clocks
      // transitioning X->1 flip both `{value, unknown}` bits, and `value ^ unknown`
      // may remain constant). Use an order-dependent hash combine instead.
      constexpr uint64_t kFNVOffsetBasis64 = 0xcbf29ce484222325ULL;
      constexpr uint64_t kFNVPrime64 = 0x100000001b3ULL;
      Value sig = buildI64Constant(rewriter, waitOp.getLoc(), kFNVOffsetBasis64);
      Value fnvPrime = buildI64Constant(rewriter, waitOp.getLoc(), kFNVPrime64);
      DenseMap<Value, Value> rematCache;
      for (Value obs : waitOp.getObserved()) {
        auto remat = rematerializeValueForPolling(obs, rematCache, rewriter);
        if (failed(remat))
          return waitOp.emitOpError() << "could not rematerialize observed value";
        Value curObs = *remat;

        auto foldIntIntoSig = [&](Value intVal) -> LogicalResult {
          auto intTy = dyn_cast<IntegerType>(intVal.getType());
          if (!intTy || intTy.getWidth() > 64)
            return failure();
          Value ext = intVal;
          if (intTy.getWidth() < 64)
            ext = comb::createZExt(rewriter, waitOp.getLoc(), intVal, 64);
          sig = comb::XorOp::create(rewriter, waitOp.getLoc(), sig, ext, true);
          sig = comb::MulOp::create(rewriter, waitOp.getLoc(),
                                    ValueRange{sig, fnvPrime}, true);
          return success();
        };

        // Common SV/Moore lowering represents 4-state values as a struct of
        // `{value, unknown}`. Fold all integer struct fields into the wait
        // signature so change detection works for `logic`-typed clocks.
        if (succeeded(foldIntIntoSig(curObs)))
          continue;
        if (auto structTy = dyn_cast<hw::StructType>(curObs.getType())) {
          for (auto field : structTy.getElements()) {
            Value fieldVal = rewriter.createOrFold<hw::StructExtractOp>(
                waitOp.getLoc(), curObs, field);
            if (failed(foldIntIntoSig(fieldVal)))
              return waitOp.emitOpError() << "unsupported observed value type";
          }
          continue;
        }

        return waitOp.emitOpError() << "unsupported observed value type";
      }
      ready = rewriter
                  .create<mlir::func::CallOp>(waitOp.getLoc(),
                                              "__arcilator_wait_change",
                                              rewriter.getI1Type(),
                                              ValueRange{waitIdVal, sig})
                  .getResult(0);
    } else {
      // Waits with neither delay nor observed values never resume.
      rewriter.setInsertionPoint(waitOp);
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(waitOp, exitBlock);
      continue;
    }

    Block *dest = waitOp.getDest();
    auto destIt = stateIds.find(dest);
    if (destIt == stateIds.end())
      return waitOp.emitOpError() << "wait dest block is not a scheduler state";

    // Resume block: update PC and branch to the successor.
    Block *resumeBlock =
        rewriter.createBlock(&region, std::next(Region::iterator(waitBlock)));
    rewriter.setInsertionPointToEnd(resumeBlock);
    Value destStateVal = buildI32Constant(rewriter, waitOp.getLoc(), destIt->second);
    rewriter.create<mlir::func::CallOp>(waitOp.getLoc(), "__arcilator_set_pc",
                                        TypeRange{},
                                        ValueRange{procIdVal, destStateVal});
    mlir::cf::BranchOp::create(rewriter, waitOp.getLoc(), dest);

    // Conditional branch: resume or yield.
    rewriter.setInsertionPoint(waitOp);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(waitOp, ready, resumeBlock,
                                                        exitBlock);
  }

  // Replace halts with a pc update + exit.
  rewriter.setInsertionPointToEnd(dispatchBlock);
  Value pc =
      rewriter
          .create<mlir::func::CallOp>(loc, "__arcilator_get_pc",
                                      rewriter.getI32Type(),
                                      ValueRange{procIdVal})
          .getResult(0);

  SmallVector<int32_t> caseValues;
  SmallVector<Block *> caseDests;
  SmallVector<ValueRange> caseOperands;
  caseValues.reserve(stateBlocks.size());
  caseDests.reserve(stateBlocks.size());
  caseOperands.reserve(stateBlocks.size());
  for (auto [idx, block] : llvm::enumerate(stateBlocks)) {
    caseValues.push_back(static_cast<int32_t>(idx));
    caseDests.push_back(block);
    caseOperands.push_back(ValueRange{});
  }
  rewriter.create<mlir::cf::SwitchOp>(loc, pc, exitBlock, ValueRange{},
                                      caseValues, caseDests, caseOperands);

  // Spilling: any SSA value defined in one scheduler state and used in another
  // must be preserved across suspension/resume. Model this with a simple
  // per-process frame array managed by the runtime.
  DenseMap<Value, uint32_t> spillSlots;
  uint32_t nextSlot = spillSlotBase;
  for (Block *block : stateBlocks) {
    for (Operation &op : block->getOperations()) {
      for (Value result : op.getResults()) {
        if (!canFrameType(result.getType()))
          continue;
        bool crossBlockUse = llvm::any_of(result.getUses(), [&](OpOperand &use) {
          return use.getOwner()->getBlock() != block;
        });
        if (crossBlockUse)
          spillSlots.try_emplace(result, nextSlot++);
      }
    }
  }

  // Emit stores at the end of the defining blocks.
  DenseMap<Block *, SmallVector<std::pair<Value, uint32_t>>> storesByBlock;
  for (auto [value, slot] : spillSlots) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      continue;
    Block *defBlock = defOp->getBlock();
    storesByBlock[defBlock].push_back({value, slot});
  }
  for (auto &it : storesByBlock) {
    Block *defBlock = it.first;
    auto &values = it.second;
    rewriter.setInsertionPoint(defBlock->getTerminator());
    for (auto [value, slot] : values) {
      Value slotVal = buildI32Constant(rewriter, value.getLoc(), slot);
      Type ty = value.getType();
      Type baseTy = ty;
      while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
        baseTy = alias.getInnerType();
      if (isa<hw::StringType>(baseTy)) {
        if (failed(ensureFrameStringFuncs(ty)))
          return failure();
        rewriter.create<mlir::func::CallOp>(value.getLoc(),
                                            "__arcilator_frame_store_str",
                                            TypeRange{},
                                            ValueRange{procIdVal, slotVal, value});
      } else {
        auto packed = packToI64(value);
        if (failed(packed))
          return execOp.emitOpError() << "cannot spill value of type "
                                      << value.getType();
        rewriter.create<mlir::func::CallOp>(
            value.getLoc(), "__arcilator_frame_store_u64", TypeRange{},
            ValueRange{procIdVal, slotVal, *packed});
      }
    }
  }

  // Replace cross-block uses with loads from the frame.
  DenseMap<Block *, DenseMap<uint32_t, Value>> loadCache;
  auto getSpillLoad = [&](Block *block, uint32_t slot, Type ty,
                          Location loc) -> FailureOr<Value> {
    auto &blockCache = loadCache[block];
    if (auto it = blockCache.find(slot); it != blockCache.end())
      return it->second;

    OpBuilder::InsertionGuard g(rewriter);
    if (block == dispatchBlock) {
      Operation *procIdDef = procIdVal.getDefiningOp();
      if (!procIdDef)
        return failure();
      rewriter.setInsertionPointAfter(procIdDef);
    } else {
      rewriter.setInsertionPointToStart(block);
    }

    Value slotVal = buildI32Constant(rewriter, loc, slot);
    Type baseTy = ty;
    while (auto alias = dyn_cast<hw::TypeAliasType>(baseTy))
      baseTy = alias.getInnerType();
    if (isa<hw::StringType>(baseTy)) {
      if (failed(ensureFrameStringFuncs(ty)))
        return failure();
      Value loaded =
          rewriter
              .create<mlir::func::CallOp>(loc, "__arcilator_frame_load_str",
                                          ty, ValueRange{procIdVal, slotVal})
              .getResult(0);
      blockCache.try_emplace(slot, loaded);
      return loaded;
    }

    Value loaded =
        rewriter
            .create<mlir::func::CallOp>(loc, "__arcilator_frame_load_u64",
                                        rewriter.getI64Type(),
                                        ValueRange{procIdVal, slotVal})
            .getResult(0);
    auto unpacked = unpackFromI64(loaded, ty, loc);
    if (failed(unpacked))
      return failure();
    blockCache.try_emplace(slot, *unpacked);
    return *unpacked;
  };

  for (auto [value, slot] : spillSlots) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      continue;
    Block *defBlock = defOp->getBlock();
    for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
      Operation *user = use.getOwner();
      Block *useBlock = user->getBlock();
      if (useBlock == defBlock)
        continue;
      auto repl = getSpillLoad(useBlock, slot, value.getType(), user->getLoc());
      if (failed(repl))
        return execOp.emitOpError()
               << "cannot reload spilled value of type " << value.getType();
      use.set(*repl);
    }
  }

  return success();
}

static bool isArcBreakingOp(Operation *op) {
  if (isa<TapOp>(op))
    return false;
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<hw::InstanceOp, seq::CompRegOp, MemoryOp, MemoryReadPortOp,
             ClockedOpInterface, seq::InitialOp, seq::ClockGateOp,
             sim::DPICallOp>(op) ||
         op->getNumResults() > 1 || op->getNumRegions() > 0 ||
         !mlir::isMemoryEffectFree(op);
}

static LogicalResult convertInitialValue(seq::CompRegOp reg,
                                         SmallVectorImpl<Value> &values) {
  if (!reg.getInitialValue())
    return values.push_back({}), success();

  // Use from_immutable cast to convert the seq.immutable type to the reg's
  // type.
  OpBuilder builder(reg);
  auto init = seq::FromImmutableOp::create(builder, reg.getLoc(), reg.getType(),
                                           reg.getInitialValue());

  values.push_back(init);
  return success();
}

//===----------------------------------------------------------------------===//
// LLHD pre-lowering
//===----------------------------------------------------------------------===//

static bool isEpsilonTime(Value time) {
  auto timeOp = time.getDefiningOp<llhd::ConstantTimeOp>();
  if (!timeOp)
    return false;
  auto delay = timeOp.getValueAttr();
  return delay.getTime() == 0 && delay.getDelta() == 0 && delay.getEpsilon() == 1;
}

static std::optional<uint64_t> getConstantLowBit(Value value) {
  if (auto cst = value.getDefiningOp<hw::ConstantOp>())
    return cst.getValue().getZExtValue();
  if (auto intTy = dyn_cast<IntegerType>(value.getType()))
    if (intTy.getWidth() == 0)
      return 0;
  return std::nullopt;
}

struct ResolvedRuntimeSignal {
  IntegerAttr sigIdAttr;
  // Dynamic signal id (e.g. opaque interface handles cast back from runtime
  // pointer-like storage). Expected to be an integer value (typically i64/i32)
  // that names a runtime-managed signal group.
  Value dynSigId;
  uint64_t baseOffset = 0;
  uint64_t totalWidth = 0;
  Value dynamicOffset;
  Operation *dynamicOffsetOp = nullptr;
};

static LogicalResult resolveRuntimeSignal(Value handle, ResolvedRuntimeSignal &out) {
  // Strip trivial `unrealized_conversion_cast` wrappers, but preserve cast nodes
  // that carry semantic meaning for the runtime signal representation.
  while (auto cast = handle.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->hasAttr(kArcilatorSigDynExtractAttr))
      break;
    // Preserve casts used by SV interface lowering to model "opaque handle"
    // round-trips (e.g. virtual interfaces stored as i64 values). Note that
    // `hw.inout<T>` is dropped by type conversion, so these casts may appear as
    // `i64 -> T` during dialect conversion.
    if (cast.getInputs().size() == 1 && cast.getResults().size() == 1) {
      Value input = cast.getInputs().front();
      auto inputIntTy = dyn_cast<IntegerType>(input.getType());
      if (inputIntTy && inputIntTy.getWidth() == 64) {
        Type outTy = cast.getResult(0).getType();
        if (auto inoutTy = dyn_cast<hw::InOutType>(outTy))
          outTy = inoutTy.getElementType();
        while (auto alias = dyn_cast<hw::TypeAliasType>(outTy))
          outTy = alias.getInnerType();
        int64_t bw = hw::getBitWidth(outTy);
        if (bw > 0 && bw <= 64 && isa<hw::StructType, hw::ArrayType>(outTy))
          break;
        if (isa<hw::InOutType>(cast.getResult(0).getType()))
          break;
      }
    }
    if (cast.getInputs().size() != 1)
      break;
    handle = cast.getInputs().front();
  }

  Operation *defOp = handle.getDefiningOp();
  if (!defOp)
    return failure();

  if (auto sigIdAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
    out.sigIdAttr = sigIdAttr;
    if (auto offAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigOffsetAttr))
      out.baseOffset = static_cast<uint64_t>(offAttr.getInt());
    if (auto widthAttr =
            defOp->getAttrOfType<IntegerAttr>(kArcilatorSigTotalWidthAttr))
      out.totalWidth = static_cast<uint64_t>(widthAttr.getInt());
    if (out.totalWidth == 0) {
      Type ty = handle.getType();
      if (auto inoutTy = dyn_cast<hw::InOutType>(ty))
        ty = inoutTy.getElementType();
      int64_t bw = hw::getBitWidth(ty);
      if (bw > 0)
        out.totalWidth = static_cast<uint64_t>(bw);
    }
    return success();
  }

  if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
    // Opaque interface handles: interpret `i64 -> hw.inout<struct>` casts as
    // dynamic runtime signal ids. This is produced by SV interface lowering for
    // virtual interfaces stored in runtime shims (e.g. UVM config/resource DB).
    if (!cast->hasAttr(kArcilatorSigDynExtractAttr) &&
        cast.getInputs().size() == 1 && cast.getResults().size() == 1) {
      Value input = cast.getInputs().front();
      auto inputIntTy = dyn_cast<IntegerType>(input.getType());
      if (inputIntTy && inputIntTy.getWidth() == 64) {
        Type outTy = cast.getResult(0).getType();
        if (auto outInOutTy = dyn_cast<hw::InOutType>(outTy))
          outTy = outInOutTy.getElementType();
        while (auto alias = dyn_cast<hw::TypeAliasType>(outTy))
          outTy = alias.getInnerType();
        int64_t bw = hw::getBitWidth(outTy);
        if (bw > 0 && bw <= 64 && isa<hw::StructType, hw::ArrayType>(outTy)) {
          out.dynSigId = input;
          out.totalWidth = static_cast<uint64_t>(bw);
          return success();
        }
      }
    }

    if (!cast->hasAttr(kArcilatorSigDynExtractAttr))
      return failure();
    if (cast.getInputs().size() != 2)
      return failure();
    ResolvedRuntimeSignal base;
    if (failed(resolveRuntimeSignal(cast.getInputs().front(), base)))
      return failure();
    out = base;

    Value dynOffset = cast.getInputs()[1];
    if (auto lowBit = getConstantLowBit(dynOffset)) {
      out.baseOffset += *lowBit;
      return success();
    }

    if (out.dynamicOffset)
      return failure();
    out.dynamicOffset = dynOffset;
    out.dynamicOffsetOp = defOp;
    return success();
  }

  if (auto ex = dyn_cast<llhd::SigExtractOp>(defOp)) {
    ResolvedRuntimeSignal base;
    if (failed(resolveRuntimeSignal(ex.getInput(), base)))
      return failure();
    out = base;

    if (auto lowBit = getConstantLowBit(ex.getLowBit())) {
      out.baseOffset += *lowBit;
      return success();
    }

    // Only support a single dynamic extract in the chain.
    if (out.dynamicOffset)
      return failure();
    out.dynamicOffset = ex.getLowBit();
    out.dynamicOffsetOp = defOp;
    return success();
  }

  auto accumulateStructFieldOffset = [&](Type structTy,
                                         StringAttr field) -> FailureOr<uint64_t> {
    if (auto inoutTy = dyn_cast<hw::InOutType>(structTy))
      structTy = inoutTy.getElementType();
    auto hwStructTy = dyn_cast<hw::StructType>(structTy);
    if (!hwStructTy)
      return failure();
    auto fieldIndexOpt = hwStructTy.getFieldIndex(field);
    if (!fieldIndexOpt)
      return failure();

    auto elements = hwStructTy.getElements();

    // Special-case the ubiquitous 4-state `{value, unknown}` struct encoding.
    // We canonicalize this layout separately (value in low bits), so keep the
    // field offsets consistent here.
    if (elements.size() == 2 &&
        elements[0].name.getValue() == "value" &&
        elements[1].name.getValue() == "unknown") {
      int64_t w = hw::getBitWidth(elements[0].type);
      if (w <= 0 || hw::getBitWidth(elements[1].type) != w)
        return failure();
      return *fieldIndexOpt == 0 ? 0ULL : static_cast<uint64_t>(w);
    }

    // Match HW struct packing (MSB-first). Special-case the 4-state
    // `{value, unknown}` encoding which we canonicalize as value in low bits.
    uint64_t prefixWidth = 0;
    for (uint32_t i = 0; i < *fieldIndexOpt; ++i) {
      int64_t w = hw::getBitWidth(elements[i].type);
      if (w <= 0)
        return failure();
      prefixWidth += static_cast<uint64_t>(w);
    }
    int64_t fieldWidthI64 = hw::getBitWidth(elements[*fieldIndexOpt].type);
    if (fieldWidthI64 <= 0)
      return failure();
    uint64_t fieldWidth = static_cast<uint64_t>(fieldWidthI64);
    int64_t totalWidthI64 = hw::getBitWidth(hwStructTy);
    if (totalWidthI64 <= 0)
      return failure();
    uint64_t totalWidth = static_cast<uint64_t>(totalWidthI64);
    if (prefixWidth + fieldWidth > totalWidth)
      return failure();
    return totalWidth - prefixWidth - fieldWidth;
  };

  if (auto field = dyn_cast<llhd::SigStructExtractOp>(defOp)) {
    ResolvedRuntimeSignal base;
    if (failed(resolveRuntimeSignal(field.getInput(), base)))
      return failure();
    auto fieldOffset =
        accumulateStructFieldOffset(field.getInput().getType(), field.getFieldAttr());
    if (failed(fieldOffset))
      return failure();
    out = base;
    out.baseOffset += *fieldOffset;
    return success();
  }

  if (auto field = dyn_cast<sv::StructFieldInOutOp>(defOp)) {
    ResolvedRuntimeSignal base;
    if (failed(resolveRuntimeSignal(field.getInput(), base)))
      return failure();
    auto fieldOffset =
        accumulateStructFieldOffset(field.getInput().getType(), field.getFieldAttr());
    if (failed(fieldOffset))
      return failure();
    out = base;
    out.baseOffset += *fieldOffset;
    return success();
  }

  return failure();
}

static std::optional<bool> getConstantBoolValue(Value value) {
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy || intTy.getWidth() != 1)
    return std::nullopt;
  auto lowBit = getConstantLowBit(value);
  if (!lowBit)
    return std::nullopt;
  return (*lowBit & 1ULL) != 0;
}

static bool isCloneableConstant(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
    return false;
  if (defOp->getNumOperands() != 0 || defOp->getNumRegions() != 0)
    return false;
  return true;
}

static LogicalResult
cloneExternalConstantsIntoProcess(llhd::ProcessOp proc,
                                  ArrayRef<Value> externalValues) {
  if (externalValues.empty())
    return success();
  if (proc.getBody().empty())
    return failure();

  Block &entry = proc.getBody().front();
  OpBuilder builder(&entry, entry.begin());

  DenseMap<Operation *, Operation *> clonedOps;
  DenseMap<Value, Value> valueMap;
  for (Value value : externalValues) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return failure();
    Operation *&clonedOp = clonedOps[defOp];
    if (!clonedOp) {
      clonedOp = builder.clone(*defOp);
      for (auto [from, to] :
           llvm::zip(defOp->getResults(), clonedOp->getResults()))
        valueMap.try_emplace(from, to);
    }
  }

  proc.getBody().walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = valueMap.find(operand.get());
      if (it != valueMap.end())
        operand.set(it->second);
    }
  });

  return success();
}

static LogicalResult convertOneShotProcessToInitial(llhd::ProcessOp proc) {
  if (proc.getNumResults() != 0)
    return failure();
  if (!proc.getOps<llhd::WaitOp>().empty())
    return failure();

  // `seq.initial` cannot capture arbitrary values from above. Only convert this
  // process if it depends solely on constant-like values (which we can clone
  // into the region).
  SetVector<Value> externalValues;
  mlir::getUsedValuesDefinedAbove(proc.getBody(), externalValues);
  for (Value v : externalValues)
    if (!isCloneableConstant(v))
      return failure();

  // All exit paths must be a `llhd.halt` without yields.
  for (auto halt : proc.getOps<llhd::HaltOp>())
    if (!halt.getYieldOperands().empty())
      return failure();

  if (failed(cloneExternalConstantsIntoProcess(proc, externalValues.getArrayRef())))
    return failure();

  // Ensure no external values remain after cloning constants.
  externalValues.clear();
  mlir::getUsedValuesDefinedAbove(proc.getBody(), externalValues);
  if (!externalValues.empty())
    return failure();

  OpBuilder builder(proc);
  auto loc = proc.getLoc();
  seq::InitialOp::create(builder, loc, TypeRange{}, [&]() {
    auto exec = mlir::scf::ExecuteRegionOp::create(builder, loc, TypeRange{});
    exec.getRegion().takeBody(proc.getBody());
    SmallVector<llhd::HaltOp> halts;
    exec.walk([&](llhd::HaltOp halt) { halts.push_back(halt); });
    for (auto halt : halts) {
      OpBuilder b(halt);
      mlir::scf::YieldOp::create(b, halt.getLoc());
      halt.erase();
    }
    seq::YieldOp::create(builder, loc);
  });

  proc.erase();
  return success();
}

/// Lower LLHD process results that are used exclusively by epsilon-time drives
/// into direct `llhd.drv` ops within the process. This matches the common
/// lowering shape for `->event` triggers where the process yields a new event
/// "bump" value and an enable bit to a module-level `llhd.drv`.
///
/// This rewrite intentionally only supports constant enable values at each
/// suspension point (wait/halt). That is sufficient for M3 bring-up tests while
/// avoiding the need to model enabled drives in the later best-effort lowering.
static LogicalResult sinkProcessResultDrives(llhd::ProcessOp proc) {
  if (proc.getNumResults() == 0)
    return failure();

  llvm::SmallDenseSet<Operation *, 4> driveOps;
  for (Value result : proc.getResults()) {
    for (OpOperand &use : result.getUses()) {
      auto drv = dyn_cast<llhd::DrvOp>(use.getOwner());
      if (!drv)
        return failure();
      if (proc->isAncestor(drv))
        return failure();
      driveOps.insert(drv.getOperation());
    }
  }
  if (driveOps.empty())
    return failure();

  SmallVector<llhd::DrvOp> drives;
  drives.reserve(driveOps.size());
  for (Operation *op : driveOps)
    drives.push_back(cast<llhd::DrvOp>(op));

  for (llhd::DrvOp drv : drives) {
    if (!isEpsilonTime(drv.getTime()))
      return failure();
  }

  unsigned numResults = proc.getNumResults();
  SmallVector<Operation *> suspendOps;
  for (Block &block : proc.getBody()) {
    Operation *term = block.getTerminator();
    if (isa<llhd::WaitOp, llhd::HaltOp>(term))
      suspendOps.push_back(term);
  }
  if (suspendOps.empty())
    return failure();

  auto remapValue = [](Value value,
                       const DenseMap<Value, Value> &yieldMap) -> Value {
    if (!value)
      return value;
    auto it = yieldMap.find(value);
    if (it != yieldMap.end())
      return it->second;
    return value;
  };

  for (Operation *term : suspendOps) {
    ValueRange yieldOperands;
    if (auto waitOp = dyn_cast<llhd::WaitOp>(term))
      yieldOperands = waitOp.getYieldOperands();
    else if (auto haltOp = dyn_cast<llhd::HaltOp>(term))
      yieldOperands = haltOp.getYieldOperands();
    else
      return failure();

    if (yieldOperands.size() != numResults)
      return failure();

    DenseMap<Value, Value> yieldMap;
    yieldMap.reserve(numResults);
    for (unsigned i = 0; i != numResults; ++i)
      yieldMap.try_emplace(proc.getResult(i), yieldOperands[i]);

    OpBuilder builder(term);
    builder.setInsertionPoint(term);
    Location loc = term->getLoc();
    for (llhd::DrvOp drv : drives) {
      Value signal = remapValue(drv.getSignal(), yieldMap);
      Value value = remapValue(drv.getValue(), yieldMap);
      Value time = remapValue(drv.getTime(), yieldMap);
      Value enable = remapValue(drv.getEnable(), yieldMap);

      if (enable) {
        auto enableConst = getConstantBoolValue(enable);
        if (!enableConst)
          return failure();
        if (!*enableConst)
          continue;
        enable = Value{};
      }

      llhd::DrvOp::create(builder, loc, signal, value, time, enable);
    }

    if (auto waitOp = dyn_cast<llhd::WaitOp>(term)) {
      Value delay = waitOp.getDelay();
      ValueRange observed = waitOp.getObserved();
      ValueRange destOperands = waitOp.getDestOperands();
      Block *dest = waitOp.getDest();
      (void)llhd::WaitOp::create(builder, loc, ValueRange{}, delay, observed,
                                 destOperands, dest);
      waitOp.erase();
      continue;
    }

    if (auto haltOp = dyn_cast<llhd::HaltOp>(term)) {
      (void)llhd::HaltOp::create(builder, loc, ValueRange{});
      haltOp.erase();
      continue;
    }

    return failure();
  }

  for (llhd::DrvOp drv : drives)
    drv.erase();

  OpBuilder builder(proc);
  auto newProc =
      llhd::ProcessOp::create(builder, proc.getLoc(), TypeRange{},
                              proc->getOperands(), proc->getAttrs());
  newProc.getBody().takeBody(proc.getBody());
  proc.erase();

  return success();
}

/// Move simple module-level LLHD drives into a single-shot process.
///
/// Moore-to-LLHD lowering can represent time-0 initialization effects (e.g.
/// variable declaration initializers) as module-level `llhd.drv` ops. Our
/// best-effort `llhd.drv` lowering treats module-level drives as combinational
/// updates and re-applies them every evaluation, which clobbers procedural
/// state and diverges from event-driven simulators (Questa).
///
/// Hoist constant, time-0 drives into an `llhd.process` that runs once and
/// halts. The cycle scheduler can then model their "run once" semantics via
/// per-process PC state.
static LogicalResult lowerModuleLevelInitDrives(hw::HWModuleOp module) {
  SmallVector<llhd::DrvOp> initDrives;
  module.walk([&](llhd::DrvOp drv) {
    if (drv->getParentOp() != module.getOperation())
      return;

    auto timeOp = drv.getTime().getDefiningOp<llhd::ConstantTimeOp>();
    if (!timeOp)
      return;
    auto delay = timeOp.getValueAttr();
    // Moore-to-LLHD sometimes schedules initialization effects into epsilon
    // slots at time 0. Treat any <0, 0d, *> drive of a constant as a
    // single-shot initializer.
    if (delay.getTime() != 0 || delay.getDelta() != 0)
      return;

    if (Value enable = drv.getEnable()) {
      auto enableConst = getConstantBoolValue(enable);
      if (!enableConst || !*enableConst)
        return;
    }

    // Restrict to drives of constant-like values. This matches the common
    // shape for declaration initializers and avoids changing the semantics of
    // continuous assignments.
    if (!isCloneableConstant(drv.getValue()))
      return;

    initDrives.push_back(drv);
  });

  if (initDrives.empty())
    return success();

  // Insert the one-shot process into the HW module body (not next to the
  // module op itself). Otherwise later Arc/LLVM lowering may treat it as a
  // top-level process and produce illegal IR.
  OpBuilder builder(module.getBodyBlock()->getTerminator());
  auto loc = module.getLoc();
  auto proc = llhd::ProcessOp::create(builder, loc, TypeRange{}, ValueRange{},
                                      ArrayRef<NamedAttribute>{});
  // Ensure the cycle-scheduler lowering treats this as a "run once" scheduled
  // process (PC state) rather than lowering it as pure combinational logic.
  proc->setAttr(kArcilatorNeedsSchedulerAttr, builder.getUnitAttr());
  Block &entry = proc.getBody().emplaceBlock();

  // Preserve source order for deterministic initialization behavior.
  for (llhd::DrvOp drv : initDrives)
    drv->moveBefore(&entry, entry.end());

  OpBuilder bodyBuilder(&entry, entry.end());
  (void)llhd::HaltOp::create(bodyBuilder, loc, ValueRange{});
  return success();
}

/// Best-effort lowering for "simple" LLHD signal semantics within a single
/// `llhd.process` (commonly used for `initial` blocks without delays). This
/// rewrites `llhd.prb`/`llhd.drv`/`llhd.sig.extract` to pure SSA updates within
/// the process region so that later conversions do not have to model inout
/// storage for these cases.
static LogicalResult lowerSimpleProcessSignals(llhd::ProcessOp proc) {
  if (!proc.getOps<llhd::WaitOp>().empty())
    return failure();

  // Keep this transformation simple: bail out if the process contains nested
  // regions that can access values from above. This is a best-effort conversion
  // intended for single-shot processes.
  WalkResult regionCheck = proc.getBody().walk([](Operation *op) -> WalkResult {
    if (op->getNumRegions() > 0 &&
        !op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (regionCheck.wasInterrupted())
    return failure();

  Block &entryBlock = proc.getBody().front();

  auto module = proc->getParentOfType<hw::HWModuleOp>();
  if (!module)
    return failure();

  DenseMap<Value, llhd::SignalOp> signalOps;
  for (auto sig : module.getOps<llhd::SignalOp>())
    signalOps.try_emplace(sig.getResult(), sig);

  // Collect LLHD signal values used in this process and ensure they are not
  // referenced elsewhere. This transformation does not model cross-process
  // storage semantics.
  llvm::SmallDenseSet<Value, 8> baseSignals;
  SmallVector<llhd::SigExtractOp> extracts;
  SmallVector<llhd::DrvOp> drives;
  SmallVector<llhd::PrbOp> probes;
  SmallVector<llhd::ConstantTimeOp> times;

  for (Block &block : proc.getBody()) {
    for (Operation &op : block) {
      if (auto ex = dyn_cast<llhd::SigExtractOp>(op)) {
        extracts.push_back(ex);
        baseSignals.insert(ex.getInput());
      } else if (auto drv = dyn_cast<llhd::DrvOp>(op)) {
        drives.push_back(drv);
        if (auto timeOp = drv.getTime().getDefiningOp<llhd::ConstantTimeOp>())
          times.push_back(timeOp);
      } else if (auto prb = dyn_cast<llhd::PrbOp>(op)) {
        probes.push_back(prb);
      } else if (isa<llhd::WaitOp>(op)) {
        return failure();
      }
    }
  }

  struct AliasInfo {
    Value base;
    Value lowBit;
    std::optional<uint64_t> constLowBit;
    unsigned width = 0;
  };
  DenseMap<Value, AliasInfo> aliasInfo;

  for (llhd::SigExtractOp ex : extracts) {
    for (OpOperand &use : ex.getResult().getUses()) {
      if (!proc->isAncestor(use.getOwner()))
        return failure();
      if (!isa<llhd::PrbOp, llhd::DrvOp>(use.getOwner()))
        return failure();
    }

    auto inoutTy = dyn_cast<hw::InOutType>(ex.getResult().getType());
    if (!inoutTy)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(inoutTy.getElementType());
    if (!elemTy)
      return failure();
    if (!isa<IntegerType>(ex.getLowBit().getType()))
      return failure();

    aliasInfo[ex.getResult()] = {ex.getInput(), ex.getLowBit(),
                                 getConstantLowBit(ex.getLowBit()),
                                 static_cast<unsigned>(elemTy.getWidth())};
  }

  // Now that aliasInfo is known, collect base signal references.
  for (llhd::SigExtractOp ex : extracts)
    baseSignals.insert(ex.getInput());
  for (llhd::DrvOp drv : drives) {
    Value sig = drv.getSignal();
    if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end())
      baseSignals.insert(alias->second.base);
    else
      baseSignals.insert(sig);
  }
  for (llhd::PrbOp prb : probes) {
    Value sig = prb.getSignal();
    if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end())
      baseSignals.insert(alias->second.base);
    else
      baseSignals.insert(sig);
  }

  auto isAllowedSimpleSignalUser = [&](Operation *op) {
    return proc->isAncestor(op) &&
           isa<llhd::PrbOp, llhd::DrvOp, llhd::SigExtractOp>(op);
  };

  for (Value base : baseSignals) {
    // Direct `llhd.signal`.
    if (signalOps.find(base) != signalOps.end()) {
      for (OpOperand &use : base.getUses())
        if (!isAllowedSimpleSignalUser(use.getOwner()))
          return failure();
      continue;
    }

    // `llhd.sig.struct_extract` of a process-local struct signal (commonly used
    // for four-valued storage). The root signal may be referenced by multiple
    // field-extract ops, but all extracted handles must be local to this
    // process.
    if (auto field = base.getDefiningOp<llhd::SigStructExtractOp>()) {
      for (OpOperand &use : base.getUses())
        if (!isAllowedSimpleSignalUser(use.getOwner()))
          return failure();

      Value root = field.getInput();
      auto rootIt = signalOps.find(root);
      if (rootIt == signalOps.end())
        return failure();
      if (!isa<hw::StructType>(rootIt->second.getInit().getType()))
        return failure();

      for (OpOperand &use : root.getUses()) {
        auto otherField = dyn_cast<llhd::SigStructExtractOp>(use.getOwner());
        if (!otherField)
          return failure();
        for (OpOperand &fieldUse : otherField.getResult().getUses())
          if (!isAllowedSimpleSignalUser(fieldUse.getOwner()))
            return failure();
      }

      continue;
    }

    return failure();
  }

  for (llhd::DrvOp drv : drives) {
    if (drv.getEnable())
      return failure();
    if (!isEpsilonTime(drv.getTime()))
      return failure();
  }

  if (baseSignals.empty())
    return failure();

  SmallVector<Value> baseList;
  baseList.reserve(baseSignals.size());
  for (Value base : baseSignals)
    baseList.push_back(base);

  llvm::sort(baseList, [](Value a, Value b) {
    Operation *aDef = a.getDefiningOp();
    Operation *bDef = b.getDefiningOp();
    if (aDef && bDef && aDef->getBlock() == bDef->getBlock())
      return aDef->isBeforeInBlock(bDef);
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  });

  DenseMap<Value, unsigned> baseIndex;
  baseIndex.reserve(baseList.size());
  for (auto it : llvm::enumerate(baseList))
    baseIndex.try_emplace(it.value(), static_cast<unsigned>(it.index()));

  SmallVector<Type> baseElemTypes;
  baseElemTypes.reserve(baseList.size());
  for (Value base : baseList) {
    auto inoutTy = dyn_cast<hw::InOutType>(base.getType());
    if (!inoutTy)
      return failure();
    baseElemTypes.push_back(inoutTy.getElementType());
  }

  DenseMap<Block *, unsigned> baseArgStart;
  for (Block &block : proc.getBody()) {
    if (&block == &entryBlock)
      continue;
    unsigned start = block.getNumArguments();
    baseArgStart.try_emplace(&block, start);
    for (Type ty : baseElemTypes)
      block.addArgument(ty, proc.getLoc());
  }

  SmallVector<Value> entryInit;
  entryInit.reserve(baseList.size());
  OpBuilder entryBuilder(&entryBlock, entryBlock.begin());
  for (Value base : baseList) {
    Value init;
    if (auto sigIt = signalOps.find(base); sigIt != signalOps.end()) {
      init = sigIt->second.getInit();
    } else if (auto field = base.getDefiningOp<llhd::SigStructExtractOp>()) {
      Value root = field.getInput();
      auto rootIt = signalOps.find(root);
      if (rootIt == signalOps.end())
        return failure();
      Value rootInit = rootIt->second.getInit();
      init = entryBuilder.createOrFold<hw::StructExtractOp>(
          proc.getLoc(), rootInit, field.getFieldAttr());
    } else {
      return failure();
    }
    entryInit.push_back(init);
  }

  auto getCurrentForBlock = [&](Block &block) -> SmallVector<Value> {
    SmallVector<Value> cur;
    cur.reserve(baseList.size());
    if (&block == &entryBlock) {
      cur.append(entryInit.begin(), entryInit.end());
      return cur;
    }
    auto it = baseArgStart.find(&block);
    if (it == baseArgStart.end())
      return cur;
    unsigned start = it->second;
    for (unsigned i = 0, e = baseList.size(); i != e; ++i)
      cur.push_back(block.getArgument(start + i));
    return cur;
  };

  auto appendValuesToTerminator = [&](Operation *terminator, Block *succ,
                                     ValueRange values) -> LogicalResult {
    if (auto br = dyn_cast<mlir::cf::BranchOp>(terminator)) {
      if (br.getDest() != succ)
        return failure();
      for (Value v : values)
        br.getDestOperandsMutable().append(v);
      return success();
    }
    if (auto condBr = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      if (condBr.getTrueDest() == succ) {
        for (Value v : values)
          condBr.getTrueDestOperandsMutable().append(v);
        return success();
      }
      if (condBr.getFalseDest() == succ) {
        for (Value v : values)
          condBr.getFalseDestOperandsMutable().append(v);
        return success();
      }
      return failure();
    }
    if (auto sw = dyn_cast<mlir::cf::SwitchOp>(terminator)) {
      if (sw.getDefaultDestination() == succ) {
        for (Value v : values)
          sw.getDefaultOperandsMutable().append(v);
        return success();
      }
      for (auto it : llvm::enumerate(sw.getCaseDestinations())) {
        if (it.value() != succ)
          continue;
        for (Value v : values)
          sw.getCaseOperandsMutable(it.index()).append(v);
        return success();
      }
      return failure();
    }
    if (isa<llhd::HaltOp>(terminator))
      return success();
    return failure();
  };

  SmallVector<Operation *> toErase;
  OpBuilder builder(proc);
  for (Block &block : proc.getBody()) {
    SmallVector<Value> currentVals = getCurrentForBlock(block);
    if (currentVals.size() != baseList.size())
      return failure();

    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto prb = dyn_cast<llhd::PrbOp>(op)) {
        builder.setInsertionPoint(prb);
        Location loc = prb.getLoc();
        Value sig = prb.getSignal();
        Value replacement;

        if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end()) {
          auto idxIt = baseIndex.find(alias->second.base);
          if (idxIt == baseIndex.end())
            return failure();
          Value baseCur = currentVals[idxIt->second];
          auto baseTy = dyn_cast<IntegerType>(baseCur.getType());
          if (!baseTy)
            return failure();

          if (alias->second.constLowBit) {
            replacement = builder.createOrFold<comb::ExtractOp>(
                loc, builder.getIntegerType(alias->second.width), baseCur,
                *alias->second.constLowBit);
          } else {
            Value shiftAmt = alias->second.lowBit;
            auto shiftTy = dyn_cast<IntegerType>(shiftAmt.getType());
            if (!shiftTy)
              return failure();

            unsigned bw = baseTy.getWidth();
            if (shiftTy.getWidth() < bw) {
              Value pad = hw::ConstantOp::create(
                  builder, loc,
                  builder.getIntegerAttr(
                      builder.getIntegerType(bw - shiftTy.getWidth()), 0));
              shiftAmt =
                  builder.createOrFold<comb::ConcatOp>(loc, pad, shiftAmt);
            } else if (shiftTy.getWidth() > bw) {
              shiftAmt = comb::ExtractOp::create(builder, loc, shiftAmt, 0, bw);
            }
            if (bw > 0) {
              Value mask = hw::ConstantOp::create(
                  builder, loc,
                  builder.getIntegerAttr(builder.getIntegerType(bw),
                                         APInt(bw, bw - 1)));
              shiftAmt = builder.createOrFold<comb::AndOp>(loc, shiftAmt, mask);
            }

            Value shifted =
                builder.createOrFold<comb::ShrUOp>(loc, baseCur, shiftAmt);
            replacement = builder.createOrFold<comb::ExtractOp>(
                loc, builder.getIntegerType(alias->second.width), shifted, 0);
          }
        } else {
          auto idxIt = baseIndex.find(sig);
          if (idxIt == baseIndex.end())
            return failure();
          replacement = currentVals[idxIt->second];
        }

        prb.replaceAllUsesWith(replacement);
        toErase.push_back(prb);
        continue;
      }

      if (auto drv = dyn_cast<llhd::DrvOp>(op)) {
        builder.setInsertionPoint(drv);
        Location loc = drv.getLoc();
        Value sig = drv.getSignal();
        Value value = drv.getValue();

        if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end()) {
          auto idxIt = baseIndex.find(alias->second.base);
          if (idxIt == baseIndex.end())
            return failure();
          unsigned baseIdx = idxIt->second;

          Value baseCur = currentVals[baseIdx];
          auto baseTy = dyn_cast<IntegerType>(baseCur.getType());
          auto valTy = dyn_cast<IntegerType>(value.getType());
          if (!baseTy || !valTy)
            return failure();

          unsigned bw = baseTy.getWidth();
          unsigned sliceWidth = alias->second.width;
          if (sliceWidth == 0 || bw == 0 || sliceWidth > bw)
            return failure();

          Value widened = value;
          if (bw > sliceWidth) {
            Value pad = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(
                    builder.getIntegerType(bw - sliceWidth), 0));
            widened = builder.createOrFold<comb::ConcatOp>(loc, pad, widened);
          }

          if (alias->second.constLowBit) {
            uint64_t offset = *alias->second.constLowBit;
            if (offset + sliceWidth > bw)
              return failure();

            APInt sliceMask = APInt::getAllOnes(sliceWidth).zext(bw) << offset;
            APInt clearMask = APInt::getAllOnes(bw) ^ sliceMask;
            Value clearCst = hw::ConstantOp::create(
                builder, loc, builder.getIntegerAttr(baseTy, clearMask));
            Value cleared =
                builder.createOrFold<comb::AndOp>(loc, baseCur, clearCst);

            Value shiftAmt = hw::ConstantOp::create(
                builder, loc, builder.getIntegerAttr(baseTy, offset));
            Value shifted =
                builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
            Value updated =
                builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
            currentVals[baseIdx] = updated;
          } else {
            Value shiftAmt = alias->second.lowBit;
            auto shiftTy = dyn_cast<IntegerType>(shiftAmt.getType());
            if (!shiftTy)
              return failure();

            if (shiftTy.getWidth() < bw) {
              Value pad = hw::ConstantOp::create(
                  builder, loc,
                  builder.getIntegerAttr(
                      builder.getIntegerType(bw - shiftTy.getWidth()), 0));
              shiftAmt =
                  builder.createOrFold<comb::ConcatOp>(loc, pad, shiftAmt);
            } else if (shiftTy.getWidth() > bw) {
              shiftAmt = comb::ExtractOp::create(builder, loc, shiftAmt, 0, bw);
            }
            if (bw > 0) {
              Value mask = hw::ConstantOp::create(
                  builder, loc,
                  builder.getIntegerAttr(builder.getIntegerType(bw),
                                         APInt(bw, bw - 1)));
              shiftAmt = builder.createOrFold<comb::AndOp>(loc, shiftAmt, mask);
            }

            Value sliceMaskBase = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(
                    baseTy, APInt::getAllOnes(sliceWidth).zext(bw)));
            Value sliceMask =
                builder.createOrFold<comb::ShlOp>(loc, sliceMaskBase, shiftAmt);
            Value allOnes = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(baseTy, APInt::getAllOnes(bw)));
            Value clearMask =
                builder.createOrFold<comb::XorOp>(loc, allOnes, sliceMask, true);
            Value cleared =
                builder.createOrFold<comb::AndOp>(loc, baseCur, clearMask);

            Value shifted =
                builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
            Value updated =
                builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
            currentVals[baseIdx] = updated;
          }
        } else {
          auto idxIt = baseIndex.find(sig);
          if (idxIt == baseIndex.end())
            return failure();
          currentVals[idxIt->second] = value;
        }

        toErase.push_back(drv);
        continue;
      }
    }

    Operation *term = block.getTerminator();
    for (Block *succ : block.getSuccessors()) {
      if (failed(appendValuesToTerminator(term, succ, currentVals)))
        return failure();
    }
  }

  for (Operation *op : toErase)
    op->erase();

  for (llhd::SigExtractOp ex : llvm::make_early_inc_range(extracts)) {
    if (ex.getResult().use_empty())
      ex.erase();
  }

  for (llhd::ConstantTimeOp timeOp : llvm::make_early_inc_range(times)) {
    if (timeOp->use_empty())
      timeOp.erase();
  }

  return success();
}

/// Best-effort lowering for "simple" LLHD signal semantics within a single
/// `llhd.final`. This mirrors `lowerSimpleProcessSignals` but targets teardown
/// code that runs without waits/delays. Many sv-tests simulation cases use
/// `final` blocks to compute a value and print `:assert:` lines; those are
/// commonly lowered by Moore/LLHD to a sequence of `llhd.drv`/`llhd.prb`
/// operations in a one-block `llhd.final`. Rewriting these to pure SSA updates
/// avoids needing full inout storage modeling in later conversions.
static LogicalResult lowerSimpleFinalSignals(llhd::FinalOp fin) {
  if (!fin.getBody().hasOneBlock())
    return failure();

  Block &block = fin.getBody().front();
  if (!fin.getOps<llhd::WaitOp>().empty())
    return failure();

  auto module = fin->getParentOfType<hw::HWModuleOp>();
  if (!module)
    return failure();

  DenseMap<Value, llhd::SignalOp> signalOps;
  for (auto sig : module.getOps<llhd::SignalOp>())
    signalOps.try_emplace(sig.getResult(), sig);

  llvm::SmallDenseSet<Value, 8> baseSignals;
  SmallVector<llhd::SigExtractOp> extracts;
  SmallVector<llhd::DrvOp> drives;
  SmallVector<llhd::PrbOp> probes;
  SmallVector<llhd::ConstantTimeOp> times;

  for (Operation &op : block) {
    if (auto ex = dyn_cast<llhd::SigExtractOp>(op)) {
      extracts.push_back(ex);
      baseSignals.insert(ex.getInput());
    } else if (auto drv = dyn_cast<llhd::DrvOp>(op)) {
      drives.push_back(drv);
      if (auto timeOp = drv.getTime().getDefiningOp<llhd::ConstantTimeOp>())
        times.push_back(timeOp);
    } else if (auto prb = dyn_cast<llhd::PrbOp>(op)) {
      probes.push_back(prb);
    } else if (isa<llhd::WaitOp>(op)) {
      return failure();
    }
  }

  struct AliasInfo {
    Value base;
    Value lowBit;
    std::optional<uint64_t> constLowBit;
    unsigned width = 0;
  };
  DenseMap<Value, AliasInfo> aliasInfo;

  for (llhd::SigExtractOp ex : extracts) {
    for (OpOperand &use : ex.getResult().getUses()) {
      if (!fin->isAncestor(use.getOwner()))
        return failure();
      if (!isa<llhd::PrbOp, llhd::DrvOp>(use.getOwner()))
        return failure();
    }

    auto inoutTy = dyn_cast<hw::InOutType>(ex.getResult().getType());
    if (!inoutTy)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(inoutTy.getElementType());
    if (!elemTy)
      return failure();
    if (!isa<IntegerType>(ex.getLowBit().getType()))
      return failure();

    aliasInfo[ex.getResult()] = {ex.getInput(), ex.getLowBit(),
                                 getConstantLowBit(ex.getLowBit()),
                                 static_cast<unsigned>(elemTy.getWidth())};
  }

  for (llhd::SigExtractOp ex : extracts)
    baseSignals.insert(ex.getInput());
  for (llhd::DrvOp drv : drives) {
    Value sig = drv.getSignal();
    if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end())
      baseSignals.insert(alias->second.base);
    else
      baseSignals.insert(sig);
  }
  for (llhd::PrbOp prb : probes) {
    Value sig = prb.getSignal();
    if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end())
      baseSignals.insert(alias->second.base);
    else
      baseSignals.insert(sig);
  }

  auto isAllowedSimpleSignalUser = [&](Operation *op) {
    return fin->isAncestor(op) &&
           isa<llhd::PrbOp, llhd::DrvOp, llhd::SigExtractOp>(op);
  };

  for (Value base : baseSignals) {
    if (signalOps.find(base) != signalOps.end()) {
      for (OpOperand &use : base.getUses())
        if (!isAllowedSimpleSignalUser(use.getOwner()))
          return failure();
      continue;
    }

    if (auto field = base.getDefiningOp<llhd::SigStructExtractOp>()) {
      for (OpOperand &use : base.getUses())
        if (!isAllowedSimpleSignalUser(use.getOwner()))
          return failure();

      Value root = field.getInput();
      auto rootIt = signalOps.find(root);
      if (rootIt == signalOps.end())
        return failure();
      if (!isa<hw::StructType>(rootIt->second.getInit().getType()))
        return failure();

      for (OpOperand &use : root.getUses()) {
        auto otherField = dyn_cast<llhd::SigStructExtractOp>(use.getOwner());
        if (!otherField)
          return failure();
        for (OpOperand &fieldUse : otherField.getResult().getUses())
          if (!isAllowedSimpleSignalUser(fieldUse.getOwner()))
            return failure();
      }

      continue;
    }

    return failure();
  }

  for (llhd::DrvOp drv : drives) {
    if (drv.getEnable())
      return failure();
    if (!isEpsilonTime(drv.getTime()))
      return failure();
  }

  DenseMap<Value, Value> current;
  auto getCurrent = [&](OpBuilder &builder, Location loc, Value base) -> Value {
    if (auto it = current.find(base); it != current.end())
      return it->second;

    Value init;
    if (auto sigIt = signalOps.find(base); sigIt != signalOps.end()) {
      init = sigIt->second.getInit();
    } else if (auto field = base.getDefiningOp<llhd::SigStructExtractOp>()) {
      Value root = field.getInput();
      auto rootIt = signalOps.find(root);
      if (rootIt == signalOps.end())
        return {};
      Value rootInit = rootIt->second.getInit();
      init = builder.createOrFold<hw::StructExtractOp>(loc, rootInit,
                                                       field.getFieldAttr());
    } else {
      return {};
    }

    current[base] = init;
    return init;
  };

  SmallVector<Operation *> toErase;
  OpBuilder builder(fin);
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (auto prb = dyn_cast<llhd::PrbOp>(op)) {
      builder.setInsertionPoint(prb);
      Location loc = prb.getLoc();
      Value sig = prb.getSignal();
      Value replacement;
      if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end()) {
        Value baseCur = getCurrent(builder, loc, alias->second.base);
        if (!baseCur)
          return failure();
        auto baseTy = dyn_cast<IntegerType>(baseCur.getType());
        if (!baseTy)
          return failure();

        if (alias->second.constLowBit) {
          replacement = builder.createOrFold<comb::ExtractOp>(
              loc, builder.getIntegerType(alias->second.width), baseCur,
              *alias->second.constLowBit);
        } else {
          Value shiftAmt = alias->second.lowBit;
          auto shiftTy = dyn_cast<IntegerType>(shiftAmt.getType());
          if (!shiftTy)
            return failure();

          unsigned bw = baseTy.getWidth();
          if (shiftTy.getWidth() < bw) {
            Value pad = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getIntegerType(bw - shiftTy.getWidth()),
                                       0));
            shiftAmt = builder.createOrFold<comb::ConcatOp>(loc, pad, shiftAmt);
          } else if (shiftTy.getWidth() > bw) {
            shiftAmt = comb::ExtractOp::create(builder, loc, shiftAmt, 0, bw);
          }
          if (bw > 0) {
            Value mask = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getIntegerType(bw),
                                       APInt(bw, bw - 1)));
            shiftAmt = builder.createOrFold<comb::AndOp>(loc, shiftAmt, mask);
          }

          Value shifted =
              builder.createOrFold<comb::ShrUOp>(loc, baseCur, shiftAmt);
          replacement = builder.createOrFold<comb::ExtractOp>(
              loc, builder.getIntegerType(alias->second.width), shifted, 0);
        }
      } else {
        replacement = getCurrent(builder, loc, sig);
        if (!replacement)
          return failure();
      }
      prb.replaceAllUsesWith(replacement);
      toErase.push_back(prb);
      continue;
    }

    if (auto drv = dyn_cast<llhd::DrvOp>(op)) {
      builder.setInsertionPoint(drv);
      Location loc = drv.getLoc();
      Value sig = drv.getSignal();
      Value value = drv.getValue();
      if (auto alias = aliasInfo.find(sig); alias != aliasInfo.end()) {
        Value base = alias->second.base;
        Value baseCur = getCurrent(builder, loc, base);
        if (!baseCur)
          return failure();
        auto baseTy = dyn_cast<IntegerType>(baseCur.getType());
        auto valTy = dyn_cast<IntegerType>(value.getType());
        if (!baseTy || !valTy)
          return failure();

        unsigned bw = baseTy.getWidth();
        unsigned sliceWidth = alias->second.width;
        if (sliceWidth == 0 || bw == 0 || sliceWidth > bw)
          return failure();

        Value widened = value;
        if (bw > sliceWidth) {
          Value pad = hw::ConstantOp::create(
              builder, loc,
              builder.getIntegerAttr(builder.getIntegerType(bw - sliceWidth),
                                     0));
          widened = builder.createOrFold<comb::ConcatOp>(loc, pad, widened);
        }
        if (alias->second.constLowBit) {
          uint64_t offset = *alias->second.constLowBit;
          if (offset + sliceWidth > bw)
            return failure();

          APInt sliceMask = APInt::getAllOnes(sliceWidth).zext(bw) << offset;
          APInt clearMask = APInt::getAllOnes(bw) ^ sliceMask;
          Value clearCst = hw::ConstantOp::create(
              builder, loc, builder.getIntegerAttr(baseTy, clearMask));
          Value cleared =
              builder.createOrFold<comb::AndOp>(loc, baseCur, clearCst);

          Value shiftAmt = hw::ConstantOp::create(
              builder, loc, builder.getIntegerAttr(baseTy, offset));
          Value shifted =
              builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
          Value updated =
              builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
          current[base] = updated;
        } else {
          Value shiftAmt = alias->second.lowBit;
          auto shiftTy = dyn_cast<IntegerType>(shiftAmt.getType());
          if (!shiftTy)
            return failure();

          if (shiftTy.getWidth() < bw) {
            Value pad = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getIntegerType(bw - shiftTy.getWidth()),
                                       0));
            shiftAmt = builder.createOrFold<comb::ConcatOp>(loc, pad, shiftAmt);
          } else if (shiftTy.getWidth() > bw) {
            shiftAmt = comb::ExtractOp::create(builder, loc, shiftAmt, 0, bw);
          }
          if (bw > 0) {
            Value mask = hw::ConstantOp::create(
                builder, loc,
                builder.getIntegerAttr(builder.getIntegerType(bw),
                                       APInt(bw, bw - 1)));
            shiftAmt = builder.createOrFold<comb::AndOp>(loc, shiftAmt, mask);
          }

          Value sliceMaskBase = hw::ConstantOp::create(
              builder, loc,
              builder.getIntegerAttr(baseTy,
                                     APInt::getAllOnes(sliceWidth).zext(bw)));
          Value sliceMask =
              builder.createOrFold<comb::ShlOp>(loc, sliceMaskBase, shiftAmt);
          Value allOnes = hw::ConstantOp::create(
              builder, loc, builder.getIntegerAttr(baseTy, APInt::getAllOnes(bw)));
          Value clearMask =
              builder.createOrFold<comb::XorOp>(loc, allOnes, sliceMask, true);
          Value cleared =
              builder.createOrFold<comb::AndOp>(loc, baseCur, clearMask);

          Value shifted =
              builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
          Value updated =
              builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
          current[base] = updated;
        }
      } else {
        current[sig] = value;
      }
      toErase.push_back(drv);
      continue;
    }
  }

  for (Operation *op : toErase)
    op->erase();

  for (llhd::SigExtractOp ex : llvm::make_early_inc_range(extracts)) {
    if (ex.getResult().use_empty())
      ex.erase();
  }

  for (llhd::ConstantTimeOp timeOp : llvm::make_early_inc_range(times)) {
    if (timeOp->use_empty())
      timeOp.erase();
  }

  return success();
}

static Value cloneValueIntoModule(Value value, OpBuilder &builder,
                                  IRMapping &mapping) {
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;

  if (auto barg = dyn_cast<BlockArgument>(value)) {
    auto *owner = barg.getOwner()->getParentOp();
    auto hwModule = dyn_cast<hw::HWModuleOp>(owner);
    if (!hwModule)
      return {};
    Value moduleArg = hwModule.getBodyBlock()->getArgument(barg.getArgNumber());
    mapping.map(value, moduleArg);
    return moduleArg;
  }

  auto *defOp = value.getDefiningOp();
  // LLHD probes/signals carry memory effects but are safe to clone when
  // sinking simple processes into an arc state. Allow those through.
  bool allowSideEffects = isa<llhd::PrbOp>(defOp) || isa<llhd::SignalOp>(defOp);
  if (!defOp || (!isMemoryEffectFree(defOp) && !allowSideEffects))
    return {};

  for (auto operand : defOp->getOperands()) {
    if (!cloneValueIntoModule(operand, builder, mapping))
      return {};
  }

  Operation *cloned = builder.clone(*defOp, mapping);
  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult)
    return {};
  Value clonedResult = cloned->getResult(opResult.getResultNumber());
  mapping.map(value, clonedResult);
  return clonedResult;
}

/// Collapse the canonical Moore always block emitted through LLHD into an
/// explicit arc state that triggers on the observed clock. This keeps the
/// sequential intent intact without relying on the stubby LLHD->Arc patterns
/// below.
static LogicalResult lowerProcessToArcState(llhd::ProcessOp proc,
                                            Namespace &ns) {
  if (!proc.getResults().empty())
    return failure();

  auto waits = llvm::to_vector(proc.getOps<llhd::WaitOp>());
  if (waits.size() != 1)
    return failure();

  llhd::WaitOp wait = waits.front();
  if (wait.getDelay() || !wait.getYieldOperands().empty() ||
      !wait.getDestOperands().empty() || wait.getObserved().size() != 1)
    return failure();

  Block *resumeBlock = wait.getDest();
  auto condBr =
      dyn_cast<mlir::cf::CondBranchOp>(resumeBlock->getTerminator());
  if (!condBr)
    return failure();

  auto succHasWait = [](Block *block) {
    return llvm::any_of(*block,
                        [](Operation &op) { return isa<llhd::WaitOp>(op); });
  };
  Block *bodyBlock = condBr.getTrueDest();
  if (succHasWait(bodyBlock))
    bodyBlock = condBr.getFalseDest();

  auto module = proc->getParentOfType<hw::HWModuleOp>();
  if (!module)
    return failure();
  auto parentModule = module->getParentOfType<mlir::ModuleOp>();
  if (!parentModule)
    return failure();

  // Create the arc definition that will run on the clock edges.
  auto *moduleBlock = module.getBodyBlock();
  SmallVector<Type> argTypes(moduleBlock->getArgumentTypes().begin(),
                             moduleBlock->getArgumentTypes().end());
  auto funcType = FunctionType::get(proc.getContext(), argTypes, {});

  SymbolTable symTable(parentModule);
  auto arcName = ns.newName(module.getModuleName().str() + "_proc");
  OpBuilder topBuilder(parentModule.getBodyRegion());
  topBuilder.setInsertionPoint(module);
  auto defOp = arc::DefineOp::create(topBuilder, proc.getLoc(),
                                     topBuilder.getStringAttr(arcName),
                                     funcType);
  symTable.insert(defOp);

  auto *entry = new Block();
  for (Type type : argTypes)
    entry->addArgument(type, proc.getLoc());
  defOp.getBody().push_back(entry);

  IRMapping mapping;
  for (auto [idx, arg] : llvm::enumerate(moduleBlock->getArguments()))
    mapping.map(arg, entry->getArgument(idx));

  OpBuilder bodyBuilder(entry, entry->end());
  for (Operation &op : bodyBlock->without_terminator())
    bodyBuilder.clone(op, mapping);
  bodyBuilder.create<arc::OutputOp>(proc.getLoc());

  // Materialize the clock value in the module body.
  OpBuilder stateBuilder(module.getBodyBlock()->getTerminator());
  IRMapping cloned;
  Value clockValue =
      cloneValueIntoModule(wait.getObserved().front(), stateBuilder, cloned);
  if (!clockValue)
    return failure();
  if (!isa<seq::ClockType>(clockValue.getType()))
    clockValue = stateBuilder.create<seq::ToClockOp>(proc.getLoc(), clockValue);

  // Instantiate the arc as a stateful element clocked by the observed signal.
  auto stateOp = arc::StateOp::create(
      stateBuilder, proc.getLoc(), defOp, clockValue, Value{},
      /*latency=*/1, moduleBlock->getArguments(), ValueRange{});
  stateBuilder.insert(stateOp.getOperation());

  proc.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

namespace {
struct Converter {
  LogicalResult run(ModuleOp module);
  LogicalResult runOnModule(HWModuleOp module);
  LogicalResult analyzeFanIn();
  void extractArcs(HWModuleOp module);
  LogicalResult absorbRegs(HWModuleOp module);

  /// The global namespace used to create unique definition names.
  Namespace globalNamespace;

  /// All arc-breaking operations in the current module.
  SmallVector<Operation *> arcBreakers;
  SmallDenseMap<Operation *, unsigned> arcBreakerIndices;

  /// A post-order traversal of the operations in the current module.
  SmallVector<Operation *> postOrder;

  /// The set of arc-breaking ops an operation in the current module
  /// contributes to, represented as a bit mask.
  MapVector<Operation *, APInt> faninMasks;

  /// The sets of operations that contribute to the same arc-breaking ops.
  MapVector<APInt, DenseSet<Operation *>> faninMaskGroups;

  /// The arc uses generated by `extractArcs`.
  SmallVector<mlir::CallOpInterface> arcUses;

  /// Whether registers should be made observable by assigning their arcs a
  /// "name" attribute.
  bool tapRegisters;
};
} // namespace

LogicalResult Converter::run(ModuleOp module) {
  for (auto &op : module.getOps())
    if (auto sym =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      globalNamespace.newName(sym.getValue());
  for (auto module : module.getOps<HWModuleOp>())
    if (failed(runOnModule(module)))
      return failure();
  return success();
}

LogicalResult Converter::runOnModule(HWModuleOp module) {
  // Find all arc-breaking operations in this module and assign them an index.
  arcBreakers.clear();
  arcBreakerIndices.clear();
  for (Operation &op : *module.getBodyBlock()) {
    if (isa<seq::InitialOp>(&op))
      continue;
    if (!isArcBreakingOp(&op) && !isa<hw::OutputOp>(&op))
      continue;
    arcBreakerIndices[&op] = arcBreakers.size();
    arcBreakers.push_back(&op);
  }
  // Skip modules with only `OutputOp`.
  if (module.getBodyBlock()->without_terminator().empty() &&
      isa<hw::OutputOp>(module.getBodyBlock()->getTerminator()))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "[convert-to-arcs] module "
                          << module.getModuleName() << " breakers="
                          << arcBreakers.size() << "\n");
  // Defensive: if we somehow collected an absurd number of breakers, bail out
  // with a clear diagnostic instead of letting downstream APInt/SmallVector
  // explode.
  constexpr size_t kArcBreakerSanityLimit = 1u << 20; // 1M breakers is plenty.
  if (arcBreakers.size() > kArcBreakerSanityLimit) {
    module.emitError("convert-to-arcs: collected ")
        << arcBreakers.size()
        << " arc-breaking operations in module `"
        << module.getModuleName().str()
        << "`; this exceeds the sanity limit and likely indicates a bug in "
           "arc-breaker detection.";
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << module.getModuleNameAttr() << " ("
                          << arcBreakers.size() << " breakers)\n");

  // For each operation, figure out the set of breaker ops it contributes to,
  // in the form of a bit mask. Then group operations together that contribute
  // to the same set of breaker ops.
  if (failed(analyzeFanIn()))
    return failure();

  // Extract the fanin mask groups into separate combinational arcs and
  // combine them with the registers in the design.
  extractArcs(module);
  if (failed(absorbRegs(module)))
    return failure();

  return success();
}

LogicalResult Converter::analyzeFanIn() {
  SmallVector<std::tuple<Operation *, SmallVector<Value, 2>>> worklist;
  SetVector<Value> seenOperands;
  auto addToWorklist = [&](Operation *op) {
    seenOperands.clear();
    for (auto operand : op->getOperands())
      seenOperands.insert(operand);
    mlir::getUsedValuesDefinedAbove(op->getRegions(), seenOperands);
    worklist.emplace_back(op, seenOperands.getArrayRef());
  };

  // Seed the worklist and fanin masks with the arc breaking operations.
  faninMasks.clear();
  for (auto *op : arcBreakers) {
    unsigned index = arcBreakerIndices.lookup(op);
    auto mask = APInt::getOneBitSet(arcBreakers.size(), index);
    faninMasks[op] = mask;
    addToWorklist(op);
  }

  // Establish a post-order among the operations.
  DenseSet<Operation *> seen;
  DenseSet<Operation *> finished;
  postOrder.clear();
  while (!worklist.empty()) {
    auto &[op, operands] = worklist.back();
    if (operands.empty()) {
      if (!isArcBreakingOp(op) && !isa<hw::OutputOp>(op))
        postOrder.push_back(op);
      finished.insert(op);
      seen.erase(op);
      worklist.pop_back();
      continue;
    }
    auto operand = operands.pop_back_val(); // advance to next operand
    auto *definingOp = operand.getDefiningOp();
    if (!definingOp || isArcBreakingOp(definingOp) ||
        finished.contains(definingOp))
      continue;
    if (!seen.insert(definingOp).second) {
      definingOp->emitError("combinational loop detected");
      return failure();
    }
    addToWorklist(definingOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "- Sorted " << postOrder.size() << " ops\n");

  // Compute fanin masks in reverse post-order, which will compute the mask
  // for an operation's uses before it computes it for the operation itself.
  // This allows us to compute the set of arc breakers an operation
  // contributes to in one pass.
  for (auto *op : llvm::reverse(postOrder)) {
    auto mask = APInt::getZero(arcBreakers.size());
    for (auto *user : op->getUsers()) {
      while (user->getParentOp() != op->getParentOp())
        user = user->getParentOp();
      auto it = faninMasks.find(user);
      if (it != faninMasks.end())
        mask |= it->second;
    }

    auto duplicateOp = faninMasks.insert({op, mask});
    (void)duplicateOp;
    assert(duplicateOp.second && "duplicate op in order");
  }

  // Group the operations by their fan-in mask.
  faninMaskGroups.clear();
  for (auto [op, mask] : faninMasks)
    if (!isArcBreakingOp(op) && !isa<hw::OutputOp>(op))
      faninMaskGroups[mask].insert(op);
  LLVM_DEBUG(llvm::dbgs() << "- Found " << faninMaskGroups.size()
                          << " fanin mask groups\n");

  return success();
}

void Converter::extractArcs(HWModuleOp module) {
  DenseMap<Value, Value> valueMapping;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  SmallVector<std::pair<OpOperand *, unsigned>> externalUses;

  arcUses.clear();
  for (auto &group : faninMaskGroups) {
    auto &opSet = group.second;
    OpBuilder builder(module);

    auto block = std::make_unique<Block>();
    builder.setInsertionPointToStart(block.get());
    valueMapping.clear();
    inputs.clear();
    outputs.clear();
    inputTypes.clear();
    outputTypes.clear();
    externalUses.clear();

    Operation *lastOp = nullptr;
    // TODO: Remove the elements from the post order as we go.
    for (auto *op : postOrder) {
      if (!opSet.contains(op))
        continue;
      lastOp = op;
      op->remove();
      builder.insert(op);
      for (auto &operand : op->getOpOperands()) {
        if (opSet.contains(operand.get().getDefiningOp()))
          continue;
        auto &mapped = valueMapping[operand.get()];
        if (!mapped) {
          mapped = block->addArgument(operand.get().getType(),
                                      operand.get().getLoc());
          inputs.push_back(operand.get());
          inputTypes.push_back(mapped.getType());
        }
        operand.set(mapped);
      }
      for (auto result : op->getResults()) {
        bool anyExternal = false;
        for (auto &use : result.getUses()) {
          if (!opSet.contains(use.getOwner())) {
            anyExternal = true;
            externalUses.push_back({&use, outputs.size()});
          }
        }
        if (anyExternal) {
          outputs.push_back(result);
          outputTypes.push_back(result.getType());
        }
      }
    }
    assert(lastOp);
    arc::OutputOp::create(builder, lastOp->getLoc(), outputs);

    // Create the arc definition.
    builder.setInsertionPoint(module);
    auto defOp =
        DefineOp::create(builder, lastOp->getLoc(),
                         builder.getStringAttr(globalNamespace.newName(
                             module.getModuleName() + "_arc")),
                         builder.getFunctionType(inputTypes, outputTypes));
    defOp.getBody().push_back(block.release());

    // Create the call to the arc definition to replace the operations that
    // we have just extracted.
    builder.setInsertionPoint(module.getBodyBlock()->getTerminator());
    auto arcOp = CallOp::create(builder, lastOp->getLoc(), defOp, inputs);
    arcUses.push_back(arcOp);
    for (auto [use, resultIdx] : externalUses)
      use->set(arcOp.getResult(resultIdx));
  }
}

LogicalResult Converter::absorbRegs(HWModuleOp module) {
  // Handle the trivial cases where all of an arc's results are used by
  // exactly one register each.
  unsigned outIdx = 0;
  unsigned numTrivialRegs = 0;
  for (auto callOp : arcUses) {
    auto stateOp = dyn_cast<StateOp>(callOp.getOperation());
    Value clock = stateOp ? stateOp.getClock() : Value{};
    Value reset;
    SmallVector<Value> initialValues;
    SmallVector<seq::CompRegOp> absorbedRegs;
    SmallVector<Attribute> absorbedNames(callOp->getNumResults(), {});
    if (auto names = callOp->getAttrOfType<ArrayAttr>("names"))
      absorbedNames.assign(names.getValue().begin(), names.getValue().end());

    // Go through all every arc result and collect the single register that uses
    // it. If a result has multiple uses or is used by something other than a
    // register, skip the arc for now and handle it later.
    bool isTrivial = true;
    for (auto result : callOp->getResults()) {
      if (!result.hasOneUse()) {
        isTrivial = false;
        break;
      }
      auto regOp = dyn_cast<seq::CompRegOp>(result.use_begin()->getOwner());
      if (!regOp || regOp.getInput() != result ||
          (clock && clock != regOp.getClk())) {
        isTrivial = false;
        break;
      }

      clock = regOp.getClk();
      reset = regOp.getReset();

      // Check that if the register has a reset, it is to a constant zero
      if (reset) {
        Value resetValue = regOp.getResetValue();
        Operation *op = resetValue.getDefiningOp();
        if (!op)
          return regOp->emitOpError(
              "is reset by an input; not supported by ConvertToArcs");
        if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
          if (constant.getValue() != 0)
            return regOp->emitOpError("is reset to a constant non-zero value; "
                                      "not supported by ConvertToArcs");
        } else {
          return regOp->emitOpError("is reset to a value that is not clearly "
                                    "constant; not supported by ConvertToArcs");
        }
      }

      if (failed(convertInitialValue(regOp, initialValues)))
        return failure();

      absorbedRegs.push_back(regOp);
      // If we absorb a register into the arc, the arc effectively produces that
      // register's value. So if the register had a name, ensure that we assign
      // that name to the arc's output.
      absorbedNames[result.getResultNumber()] = regOp.getNameAttr();
    }

    // If this wasn't a trivial case keep the arc around for a second iteration.
    if (!isTrivial) {
      arcUses[outIdx++] = callOp;
      continue;
    }
    ++numTrivialRegs;

    // Set the arc's clock to the clock of the registers we've absorbed, bump
    // the latency up by one to account for the registers, add the reset if
    // present and update the output names. Then replace the registers.

    auto arc = dyn_cast<StateOp>(callOp.getOperation());
    if (arc) {
      arc.getClockMutable().assign(clock);
      arc.setLatency(arc.getLatency() + 1);
    } else {
      mlir::IRRewriter rewriter(module->getContext());
      rewriter.setInsertionPoint(callOp);
      arc = rewriter.replaceOpWithNewOp<StateOp>(
          callOp.getOperation(),
          llvm::cast<SymbolRefAttr>(callOp.getCallableForCallee()),
          callOp->getResultTypes(), clock, Value{}, 1, callOp.getArgOperands());
    }

    if (reset) {
      if (arc.getReset())
        return arc.emitError(
            "StateOp tried to infer reset from CompReg, but already "
            "had a reset.");
      arc.getResetMutable().assign(reset);
    }

    bool onlyDefaultInitializers =
        llvm::all_of(initialValues, [](auto val) -> bool { return !val; });

    if (!onlyDefaultInitializers) {
      if (!arc.getInitials().empty()) {
        return arc.emitError(
            "StateOp tried to infer initial values from CompReg, but already "
            "had an initial value.");
      }
      // Create 0 constants for default initialization
      for (unsigned i = 0; i < initialValues.size(); ++i) {
        if (!initialValues[i]) {
          OpBuilder zeroBuilder(arc);
          initialValues[i] = zeroBuilder.createOrFold<hw::ConstantOp>(
              arc.getLoc(),
              zeroBuilder.getIntegerAttr(arc.getResult(i).getType(), 0));
        }
      }
      arc.getInitialsMutable().assign(initialValues);
    }

    if (tapRegisters && llvm::any_of(absorbedNames, [](auto name) {
          return !cast<StringAttr>(name).getValue().empty();
        }))
      arc->setAttr("names", ArrayAttr::get(module.getContext(), absorbedNames));
    for (auto [arcResult, reg] : llvm::zip(arc.getResults(), absorbedRegs)) {
      auto it = arcBreakerIndices.find(reg);
      arcBreakers[it->second] = {};
      arcBreakerIndices.erase(it);
      reg.replaceAllUsesWith(arcResult);
      reg.erase();
    }
  }
  if (numTrivialRegs > 0)
    LLVM_DEBUG(llvm::dbgs() << "- Trivially converted " << numTrivialRegs
                            << " regs to arcs\n");
  arcUses.truncate(outIdx);

  // Group the remaining registers by their clock, their reset and the operation
  // they use as input. This will allow us to generally collapse registers
  // derived from the same arc into one shuffling arc.
  MapVector<std::tuple<Value, Value, Operation *>, SmallVector<seq::CompRegOp>>
      regsByInput;
  for (auto *op : arcBreakers)
    if (auto regOp = dyn_cast_or_null<seq::CompRegOp>(op)) {
      regsByInput[{regOp.getClk(), regOp.getReset(),
                   regOp.getInput().getDefiningOp()}]
          .push_back(regOp);
    }

  unsigned numMappedRegs = 0;
  for (auto [clockAndResetAndOp, regOps] : regsByInput) {
    numMappedRegs += regOps.size();
    OpBuilder builder(module);
    auto block = std::make_unique<Block>();
    builder.setInsertionPointToStart(block.get());

    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Attribute> names;
    SmallVector<Type> types;
    SmallVector<Value> initialValues;
    SmallDenseMap<Value, unsigned> mapping;
    SmallVector<unsigned> regToOutputMapping;
    for (auto regOp : regOps) {
      auto it = mapping.find(regOp.getInput());
      if (it == mapping.end()) {
        it = mapping.insert({regOp.getInput(), inputs.size()}).first;
        inputs.push_back(regOp.getInput());
        types.push_back(regOp.getType());
        outputs.push_back(block->addArgument(regOp.getType(), regOp.getLoc()));
        names.push_back(regOp->getAttrOfType<StringAttr>("name"));
        if (failed(convertInitialValue(regOp, initialValues)))
          return failure();
      }
      regToOutputMapping.push_back(it->second);
    }

    auto loc = regOps.back().getLoc();
    arc::OutputOp::create(builder, loc, outputs);

    builder.setInsertionPoint(module);
    auto defOp = DefineOp::create(builder, loc,
                                  builder.getStringAttr(globalNamespace.newName(
                                      module.getModuleName() + "_arc")),
                                  builder.getFunctionType(types, types));
    defOp.getBody().push_back(block.release());

    builder.setInsertionPoint(module.getBodyBlock()->getTerminator());

    bool onlyDefaultInitializers =
        llvm::all_of(initialValues, [](auto val) -> bool { return !val; });

    if (onlyDefaultInitializers)
      initialValues.clear();
    else
      for (unsigned i = 0; i < initialValues.size(); ++i) {
        if (!initialValues[i])
          initialValues[i] = builder.createOrFold<hw::ConstantOp>(
              loc, builder.getIntegerAttr(types[i], 0));
      }

    auto arcOp =
        StateOp::create(builder, loc, defOp, std::get<0>(clockAndResetAndOp),
                        /*enable=*/Value{}, 1, inputs, initialValues);
    auto reset = std::get<1>(clockAndResetAndOp);
    if (reset)
      arcOp.getResetMutable().assign(reset);
    if (tapRegisters && llvm::any_of(names, [](auto name) {
          return !cast<StringAttr>(name).getValue().empty();
        }))
      arcOp->setAttr("names", builder.getArrayAttr(names));
    for (auto [reg, resultIdx] : llvm::zip(regOps, regToOutputMapping)) {
      reg.replaceAllUsesWith(arcOp.getResult(resultIdx));
      reg.erase();
    }
  }

  if (numMappedRegs > 0)
    LLVM_DEBUG(llvm::dbgs() << "- Mapped " << numMappedRegs << " regs to "
                            << regsByInput.size() << " shuffling arcs\n");

  return success();
}

//===----------------------------------------------------------------------===//
// LLHD Conversion
//===----------------------------------------------------------------------===//

/// `llhd.combinational` -> `arc.execute`
static LogicalResult convert(llhd::CombinationalOp op,
                            llhd::CombinationalOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter,
                            const TypeConverter &converter) {
  // Convert the result types.
  SmallVector<Type> resultTypes;
  if (failed(converter.convertTypes(op.getResultTypes(), resultTypes)))
    return failure();

  // Collect the SSA values defined outside but used inside the body region.
  auto cloneIntoBody = [](Operation *op) {
    // Keep runtime signal handle tokens local to the region. Those handles carry
    // `arcilator.sig_*` attributes on their defining ops which are required by
    // the best-effort LLHD `{prb,drv}` lowerings. Passing them as block
    // arguments would drop the attributes.
    if (op->hasTrait<OpTrait::ConstantLike>())
      return true;
    if (op->hasAttr(kArcilatorSigIdAttr) ||
        op->hasAttr(kArcilatorSigOffsetAttr) ||
        op->hasAttr(kArcilatorSigTotalWidthAttr))
      return true;
    // Avoid capturing LLHD signal handles (and derived inout field handles) as
    // `arc.execute` operands. Those handles carry `arcilator.sig_*` attributes
    // on their defining ops which are required by the best-effort
    // `llhd.{prb,drv}` lowerings; passing them as block arguments would drop
    // the attributes and cause drives/reads to be erased.
    return isa<llhd::SignalOp, llhd::PrbOp, llhd::SigExtractOp,
               llhd::SigStructExtractOp, llhd::SigArrayGetOp,
               llhd::SigArraySliceOp, sv::StructFieldInOutOp>(op);
  };
  auto operands =
      mlir::makeRegionIsolatedFromAbove(rewriter, op.getBody(), cloneIntoBody);
  SmallVector<Value> convertedOperands;
  convertedOperands.reserve(operands.size());
  for (Value operand : operands) {
    SmallVector<Type> types;
    if (failed(converter.convertType(operand, types)) || types.size() != 1)
      return failure();
    auto convertedType = types.front();
    if (convertedType == operand.getType())
      convertedOperands.push_back(operand);
    else
      convertedOperands.push_back(rewriter
                                      .create<mlir::UnrealizedConversionCastOp>(
                                          op.getLoc(), convertedType, operand)
                                      .getResult(0));
  }

  // Create a replacement `arc.execute` op.
  auto executeOp =
      ExecuteOp::create(rewriter, op.getLoc(), resultTypes, convertedOperands);
  Block &entryBlock = op.getBody().front();
  unsigned captureOffset = entryBlock.getNumArguments() - operands.size();
  TypeConverter::SignatureConversion signature(entryBlock.getNumArguments());
  for (unsigned i = 0; i < captureOffset; ++i) {
    SmallVector<Type> types;
    if (failed(converter.convertType(entryBlock.getArgument(i), types)) ||
        types.size() != 1)
      return failure();
    signature.addInputs(i, types.front());
  }
  for (auto [idx, operand] : llvm::enumerate(convertedOperands))
    signature.addInputs(captureOffset + idx, operand.getType());
  // Apply signature conversion before moving the body. This keeps the rewrite
  // rollback-safe: moving regions via `takeBody` is not tracked by the
  // conversion rewriter and can crash if the pattern needs to fail.
  if (!rewriter.applySignatureConversion(&entryBlock, signature, &converter))
    return failure();
  rewriter.inlineRegionBefore(op.getBody(), executeOp.getBody(),
                              executeOp.getBody().begin());
  rewriter.replaceOp(op, executeOp.getResults());
  return success();
}

/// `llhd.process` -> `arc.execute` (drop scheduling; treat body as comb)
static LogicalResult convert(llhd::ProcessOp op, llhd::ProcessOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const TypeConverter &converter) {
  SmallVector<Type> resultTypes;
  if (failed(converter.convertTypes(op.getResultTypes(), resultTypes)))
    return failure();

  auto cloneIntoBody = [](Operation *inner) {
    // Avoid capturing LLHD signal handles (and derived inout field handles) as
    // `arc.execute` operands. Those handles carry `arcilator.sig_*` attributes
    // on their defining ops which are required by the best-effort
    // `llhd.{prb,drv}` lowerings; passing them as block arguments would drop
    // the attributes and cause drives/reads to be erased.
    if (inner->hasTrait<OpTrait::ConstantLike>())
      return true;
    if (inner->hasAttr(kArcilatorSigIdAttr) ||
        inner->hasAttr(kArcilatorSigOffsetAttr) ||
        inner->hasAttr(kArcilatorSigTotalWidthAttr))
      return true;
    return isa<llhd::SignalOp, llhd::PrbOp, llhd::SigExtractOp,
               llhd::SigStructExtractOp, llhd::SigArrayGetOp,
               llhd::SigArraySliceOp, sv::StructFieldInOutOp>(inner);
  };
  auto operands =
      mlir::makeRegionIsolatedFromAbove(rewriter, op.getBody(), cloneIntoBody);
  SmallVector<Value> convertedOperands;
  convertedOperands.reserve(operands.size());
  for (Value operand : operands) {
    SmallVector<Type> types;
    if (failed(converter.convertType(operand, types)) || types.size() != 1)
      return failure();
    auto convertedType = types.front();
    if (convertedType == operand.getType())
      convertedOperands.push_back(operand);
    else
      convertedOperands.push_back(rewriter
                                      .create<mlir::UnrealizedConversionCastOp>(
                                          op.getLoc(), convertedType, operand)
                                      .getResult(0));
  }

  auto executeOp =
      ExecuteOp::create(rewriter, op.getLoc(), resultTypes, convertedOperands);
  Block &entryBlock = op.getBody().front();
  unsigned captureOffset = entryBlock.getNumArguments() - operands.size();
  TypeConverter::SignatureConversion signature(entryBlock.getNumArguments());
  for (unsigned i = 0; i < captureOffset; ++i) {
    SmallVector<Type> types;
    if (failed(converter.convertType(entryBlock.getArgument(i), types)) ||
        types.size() != 1)
      return failure();
    signature.addInputs(i, types.front());
  }
  for (auto [idx, operand] : llvm::enumerate(convertedOperands))
    signature.addInputs(captureOffset + idx, operand.getType());
  if (!rewriter.applySignatureConversion(&entryBlock, signature, &converter))
    return failure();
  rewriter.inlineRegionBefore(op.getBody(), executeOp.getBody(),
                              executeOp.getBody().begin());

  if (auto procIdAttr = op->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
    executeOp->setAttr(kArcilatorProcIdAttr, procIdAttr);

  if (needsCycleScheduler(op)) {
    auto procIdAttr = op->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr);
    uint32_t procId = procIdAttr ? static_cast<uint32_t>(procIdAttr.getInt()) : 0;
    if (failed(lowerCycleScheduler(executeOp, procId, rewriter)))
      return failure();
  }

  rewriter.replaceOp(op, executeOp.getResults());
  return success();
}

/// `llhd.yield` -> `arc.output`
static LogicalResult convert(llhd::YieldOp op, llhd::YieldOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<arc::OutputOp>(op, adaptor.getOperands());
  return success();
}

/// `llhd.sig` -> forward the initializer as a plain SSA value (drop inout)
struct SignalOpConversion : public OpConversionPattern<llhd::SignalOp> {
  using OpConversionPattern<llhd::SignalOp>::OpConversionPattern;

  SignalOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     llvm::DenseMap<uint64_t, ArcilatorRuntimeSigInit> *sigInits)
      : OpConversionPattern(typeConverter, context), sigInits(sigInits) {}

  LogicalResult
  matchAndRewrite(llhd::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For scheduled simulation, treat some LLHD signals as runtime-managed
    // storage, using a stable integer id as the "signal handle" after type
    // conversion. This enables delay/wait processes and interface field storage
    // without implementing full LLHD signal semantics.
    auto sigIdAttr = op->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr);
    if (sigIdAttr) {
      bool needsRuntimeStorage = false;
      for (Operation *user : op.getResult().getUsers()) {
        if (isa<llhd::PrbOp, llhd::DrvOp, llhd::SigExtractOp,
                llhd::SigStructExtractOp, llhd::SigArrayGetOp,
                sv::StructFieldInOutOp>(user)) {
          needsRuntimeStorage = true;
          break;
        }
      }

      Type convertedTy = typeConverter->convertType(op.getType());
      int64_t totalWidth = convertedTy ? hw::getBitWidth(convertedTy) : -1;
      if (needsRuntimeStorage && totalWidth > 0 && totalWidth <= 64) {
        Value handle;
        if (auto intTy = dyn_cast<IntegerType>(convertedTy)) {
          uint64_t sigId = static_cast<uint64_t>(sigIdAttr.getInt());
          APInt sigIdBits(intTy.getWidth(), sigId);
          handle = hw::ConstantOp::create(rewriter, op.getLoc(), sigIdBits);
        } else {
          handle = createZeroHWConstant(rewriter, op.getLoc(), convertedTy);
        }
        if (handle) {
          if (auto *defOp = handle.getDefiningOp()) {
            defOp->setAttr(kArcilatorSigIdAttr, sigIdAttr);
            defOp->setAttr(kArcilatorSigOffsetAttr,
                           rewriter.getI32IntegerAttr(0));
            defOp->setAttr(kArcilatorSigTotalWidthAttr,
                           rewriter.getI32IntegerAttr(totalWidth));
            // Preserve the LLHD signal initializer as a constant i64 payload so
            // the later Arc state-lowering pipeline can seed runtime-managed
            // signals at time 0 (important for 2-state SV types).
            auto initBits =
                tryEvalRuntimeSignalInit(adaptor.getInit(), convertedTy);
            if (!initBits) {
              // Default-initialize runtime-managed signals to match SV
              // semantics:
              // - 2-state integers default to 0,
              // - 4-state values default to X (unknown mask set).
              //
              // This is particularly important for composite runtime signals
              // (e.g. interface packs) that contain 2-state fields such as
              // `bit` clocks; the driver runtime initializes to all-ones by
              // default, which would otherwise start those fields at 1 and
              // trigger spurious time-0 events.
              initBits = tryDefaultRuntimeSignalInit(convertedTy);
            }
            if (initBits) {
              APInt bits =
                  initBits->zextOrTrunc(static_cast<unsigned>(totalWidth));
              uint64_t initU64 = bits.getZExtValue();
              defOp->setAttr(kArcilatorSigInitU64Attr,
                             rewriter.getI64IntegerAttr(initU64));
              if (sigInits) {
                uint64_t sigId = static_cast<uint64_t>(sigIdAttr.getInt());
                auto [it, inserted] = sigInits->try_emplace(
                    sigId, ArcilatorRuntimeSigInit{
                               initU64,
                               static_cast<uint64_t>(totalWidth)});
                if (!inserted) {
                  // Keep the first initializer and width; later duplicates must
                  // agree.
                  if (it->second.initU64 != initU64)
                    op->emitWarning("conflicting runtime init value for sigId ")
                        << sigId;
                  if (it->second.totalWidth != static_cast<uint64_t>(totalWidth))
                    op->emitWarning("conflicting runtime width for sigId ")
                        << sigId;
                }
              }
            }
          }
          rewriter.replaceOp(op, handle);
          return success();
        }
      }
    }

    rewriter.replaceOp(op, adaptor.getInit());
    return success();
  }

  llvm::DenseMap<uint64_t, ArcilatorRuntimeSigInit> *sigInits = nullptr;
};

/// `llhd.prb` -> pass-through (signal already converted to plain SSA)
struct ProbeOpConversion : public OpConversionPattern<llhd::PrbOp> {
  using OpConversionPattern<llhd::PrbOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::PrbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value handle = adaptor.getSignal();
    ResolvedRuntimeSignal resolved;
    if (failed(resolveRuntimeSignal(handle, resolved)) ||
        (!resolved.sigIdAttr && !resolved.dynSigId)) {
      rewriter.replaceOp(op, adaptor.getSignal());
      return success();
    }
    uint64_t sliceOffset = resolved.baseOffset;
    uint64_t totalWidth = resolved.totalWidth;

    Type convertedTy = typeConverter->convertType(op.getType());
    int64_t resultWidth = convertedTy ? hw::getBitWidth(convertedTy) : -1;
    if (!convertedTy || resultWidth <= 0 || resultWidth > 64)
      return rewriter.notifyMatchFailure(op, "unsupported probed signal type");
    if (totalWidth == 0 || totalWidth > 64)
      return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "missing module for runtime hook");

    (void)getOrInsertFunc(
        module, "__arcilator_sig_load_u64",
        rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                                 {rewriter.getI64Type()}));

    uint32_t procId = 0xFFFFFFFFu;
    if (auto exec = op->getParentOfType<arc::ExecuteOp>()) {
      if (auto attr =
              exec->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    } else if (auto proc = op->getParentOfType<llhd::ProcessOp>()) {
      if (auto attr =
              proc->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    }
    Value procIdVal = buildI32Constant(rewriter, op.getLoc(), procId);

    Value sigIdVal;
    if (resolved.sigIdAttr) {
      sigIdVal = buildI32Constant(rewriter, op.getLoc(), resolved.sigIdAttr.getInt());
    } else {
      Value dyn = resolved.dynSigId;
      auto dynTy = dyn_cast<IntegerType>(dyn.getType());
      if (!dynTy)
        return rewriter.notifyMatchFailure(op, "unsupported dynamic signal id type");
      if (dynTy.getWidth() < 32)
        dyn = comb::createZExt(rewriter, op.getLoc(), dyn, 32);
      else if (dynTy.getWidth() > 32)
        dyn = comb::ExtractOp::create(rewriter, op.getLoc(), dyn, 0, 32);
      sigIdVal = dyn;
    }

    Value loaded =
        rewriter
            .create<mlir::func::CallOp>(op.getLoc(), "__arcilator_sig_load_u64",
                                        rewriter.getI64Type(),
                                        ValueRange{sigIdVal, procIdVal})
            .getResult(0);
    Value bitsVal = loaded;
    if (resolved.dynamicOffset) {
      Value offsetVal = resolved.dynamicOffset;
      auto offsetTy = dyn_cast<IntegerType>(offsetVal.getType());
      if (!offsetTy)
        return rewriter.notifyMatchFailure(op, "unsupported dynamic extract index");
      if (offsetTy.getWidth() < 64)
        offsetVal = comb::createZExt(rewriter, op.getLoc(), offsetVal, 64);
      else if (offsetTy.getWidth() > 64)
        offsetVal = comb::ExtractOp::create(rewriter, op.getLoc(), offsetVal, 0, 64);

      if (sliceOffset != 0) {
        Value baseOff = buildI64Constant(rewriter, op.getLoc(), sliceOffset);
        offsetVal =
            comb::AddOp::create(rewriter, op.getLoc(), baseOff, offsetVal, true);
      }
      offsetVal = rewriter.createOrFold<comb::AndOp>(
          op.getLoc(), offsetVal, buildI64Constant(rewriter, op.getLoc(), 63));

      Value shifted =
          rewriter.createOrFold<comb::ShrUOp>(op.getLoc(), loaded, offsetVal);
      bitsVal = shifted;
      if (resultWidth != 64) {
        bitsVal = comb::ExtractOp::create(rewriter, op.getLoc(),
                                          rewriter.getIntegerType(resultWidth),
                                          shifted, 0);
      }
    } else {
      if (sliceOffset + static_cast<uint64_t>(resultWidth) > 64)
        return rewriter.notifyMatchFailure(
            op, "signal slice exceeds 64-bit runtime storage");
      if (resultWidth != 64 || sliceOffset != 0) {
        bitsVal = comb::ExtractOp::create(rewriter, op.getLoc(),
                                          rewriter.getIntegerType(resultWidth),
                                          loaded, sliceOffset);
      }
    }

    if (bitsVal.getType() == convertedTy) {
      rewriter.replaceOp(op, bitsVal);
      return success();
    }

    Value casted =
        rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), convertedTy, bitsVal);
    rewriter.replaceOp(op, casted);

    if (resolved.dynamicOffsetOp &&
        llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                     [](Value v) { return v.use_empty(); }))
      rewriter.eraseOp(resolved.dynamicOffsetOp);
    return success();
  }
};

/// `llhd.drv` -> best-effort runtime store (ignore precise delay; honor enable).
struct DrvOpConversion : public OpConversionPattern<llhd::DrvOp> {
  using OpConversionPattern<llhd::DrvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::DrvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value handle = adaptor.getSignal();
    ResolvedRuntimeSignal resolved;
    if (failed(resolveRuntimeSignal(handle, resolved)) ||
        (!resolved.sigIdAttr && !resolved.dynSigId)) {
      rewriter.eraseOp(op);
      return success();
    }

    // For runtime-managed signals we need to preserve LLHD delta-time semantics
    // for SystemVerilog nonblocking assignments. Moore-to-LLHD lowering encodes
    // NBAs as `llhd.drv` with `delta=1, epsilon=0` (and `time=0`).
    //
    // We still ignore the absolute time component here (it is generally modeled
    // via explicit waits), but we must distinguish delta drives so we can apply
    // them at the end of the current delta cycle rather than immediately.
    bool isNbaDrive = false;
    if (auto timeOp = op.getTime().getDefiningOp<llhd::ConstantTimeOp>()) {
      auto t = timeOp.getValueAttr();
      if (t.getTime() == 0 && t.getDelta() > 0 && t.getEpsilon() == 0)
        isNbaDrive = true;
    }

    Type rawValueTy = adaptor.getValue().getType();
    int64_t valueWidth = hw::getBitWidth(rawValueTy);
    if (valueWidth <= 0 || valueWidth > 64)
      return rewriter.notifyMatchFailure(op, "unsupported driven value type");

    auto valueTy = rewriter.getIntegerType(valueWidth);
    Value valueInt = adaptor.getValue();
    if (valueInt.getType() != valueTy) {
      valueInt =
          rewriter.createOrFold<hw::BitcastOp>(op.getLoc(), valueTy, valueInt);
    }

    uint64_t sliceOffset = resolved.baseOffset;
    uint64_t totalWidth = resolved.totalWidth;
    if (totalWidth == 0 || totalWidth > 64)
      return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");
    if (!resolved.dynamicOffset &&
        sliceOffset + static_cast<uint64_t>(valueTy.getWidth()) > totalWidth)
      return rewriter.notifyMatchFailure(op, "signal slice exceeds runtime width");

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "missing module for runtime hook");

    (void)getOrInsertFunc(
        module, "__arcilator_sig_load_u64",
        rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                                 {rewriter.getI64Type()}));
    (void)getOrInsertFunc(
        module, "__arcilator_sig_load_nba_u64",
        rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI32Type()},
                                 {rewriter.getI64Type()}));
    (void)getOrInsertFunc(
        module, "__arcilator_sig_store_u64",
        rewriter.getFunctionType(
            {rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI32Type()},
            {}));
    (void)getOrInsertFunc(
        module, "__arcilator_sig_store_nba_u64",
        rewriter.getFunctionType(
            {rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI32Type()},
            {}));
    (void)getOrInsertFunc(
        module, "__arcilator_sig_store_nba_masked_u64",
        rewriter.getFunctionType({rewriter.getI32Type(), rewriter.getI64Type(),
                                  rewriter.getI64Type(), rewriter.getI32Type()},
                                 {}));

    const StringRef loadCallee =
        isNbaDrive ? "__arcilator_sig_load_nba_u64" : "__arcilator_sig_load_u64";
    const StringRef storeCallee =
        isNbaDrive ? "__arcilator_sig_store_nba_u64" : "__arcilator_sig_store_u64";

    uint32_t procId = 0xFFFFFFFFu;
    if (auto exec = op->getParentOfType<arc::ExecuteOp>()) {
      if (auto attr =
              exec->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    } else if (auto proc = op->getParentOfType<llhd::ProcessOp>()) {
      if (auto attr =
              proc->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    }
    Value procIdVal = buildI32Constant(rewriter, op.getLoc(), procId);

    Value sigIdVal;
    if (resolved.sigIdAttr) {
      sigIdVal = buildI32Constant(rewriter, op.getLoc(), resolved.sigIdAttr.getInt());
    } else {
      Value dyn = resolved.dynSigId;
      auto dynTy = dyn_cast<IntegerType>(dyn.getType());
      if (!dynTy)
        return rewriter.notifyMatchFailure(op, "unsupported dynamic signal id type");
      if (dynTy.getWidth() < 32)
        dyn = comb::createZExt(rewriter, loc, dyn, 32);
      else if (dynTy.getWidth() > 32)
        dyn = comb::ExtractOp::create(rewriter, loc, dyn, 0, 32);
      sigIdVal = dyn;
    }

    Value enable = adaptor.getEnable();
    std::optional<bool> constEnable;
    if (!enable) {
      constEnable = true;
      enable = hw::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1);
    } else {
      constEnable = getConstantBoolValue(enable);
      if (constEnable && !*constEnable) {
        rewriter.eraseOp(op);
        if (resolved.dynamicOffsetOp &&
            llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                         [](Value v) { return v.use_empty(); }))
          rewriter.eraseOp(resolved.dynamicOffsetOp);
        return success();
      }
    }

    Value value64 = valueInt;
    if (static_cast<uint64_t>(valueWidth) < 64)
      value64 = comb::createZExt(rewriter, loc, value64, 64);

    if (resolved.dynamicOffset) {
      Value offsetVal = resolved.dynamicOffset;
      auto offsetTy = dyn_cast<IntegerType>(offsetVal.getType());
      if (!offsetTy)
        return rewriter.notifyMatchFailure(op, "unsupported dynamic extract index");
      if (offsetTy.getWidth() < 64)
        offsetVal = comb::createZExt(rewriter, loc, offsetVal, 64);
      else if (offsetTy.getWidth() > 64)
        offsetVal = comb::ExtractOp::create(rewriter, loc, offsetVal, 0, 64);

      if (sliceOffset != 0) {
        Value baseOff = buildI64Constant(rewriter, loc, sliceOffset);
        offsetVal = comb::AddOp::create(rewriter, loc, baseOff, offsetVal, true);
      }
      offsetVal = rewriter.createOrFold<comb::AndOp>(
          loc, offsetVal, buildI64Constant(rewriter, loc, 63));

      APInt totalMask = APInt::getAllOnes(64);
      if (totalWidth < 64)
        totalMask = APInt::getAllOnes(totalWidth).zext(64);
      Value totalMaskCst = hw::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(rewriter.getI64Type(), totalMask));

      APInt sliceMaskBase = APInt::getAllOnes(static_cast<unsigned>(valueWidth));
      if (valueWidth < 64)
        sliceMaskBase = sliceMaskBase.zext(64);
      Value sliceMaskBaseCst = hw::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(rewriter.getI64Type(), sliceMaskBase));
      Value sliceMask =
          rewriter.createOrFold<comb::ShlOp>(loc, sliceMaskBaseCst, offsetVal);
      sliceMask = rewriter.createOrFold<comb::AndOp>(loc, sliceMask, totalMaskCst);
      Value clearMask =
          rewriter.createOrFold<comb::XorOp>(loc, totalMaskCst, sliceMask, true);

      if (isNbaDrive) {
        Value shiftedVal =
            rewriter.createOrFold<comb::ShlOp>(loc, value64, offsetVal);
        Value inserted =
            rewriter.createOrFold<comb::AndOp>(loc, shiftedVal, sliceMask);
        Value mask = sliceMask;
        if (!constEnable || !*constEnable) {
          Value zero = buildI64Constant(rewriter, loc, 0);
          mask = rewriter.createOrFold<comb::MuxOp>(loc, enable, mask, zero);
        }
        rewriter.create<mlir::func::CallOp>(
            loc, "__arcilator_sig_store_nba_masked_u64", TypeRange{},
            ValueRange{sigIdVal, mask, inserted, procIdVal});
        rewriter.eraseOp(op);
        if (resolved.dynamicOffsetOp &&
            llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                         [](Value v) { return v.use_empty(); }))
          rewriter.eraseOp(resolved.dynamicOffsetOp);
        return success();
      }

      Value cur =
          rewriter
              .create<mlir::func::CallOp>(loc, loadCallee,
                                          rewriter.getI64Type(),
                                          ValueRange{sigIdVal, procIdVal})
              .getResult(0);
      Value curMasked = rewriter.createOrFold<comb::AndOp>(loc, cur, totalMaskCst);
      Value cleared = rewriter.createOrFold<comb::AndOp>(loc, curMasked, clearMask);

      Value shiftedVal = rewriter.createOrFold<comb::ShlOp>(loc, value64, offsetVal);
      Value inserted = rewriter.createOrFold<comb::AndOp>(loc, shiftedVal, sliceMask);
      Value storeSlice = inserted;
      if (!constEnable || !*constEnable) {
        Value oldSlice = rewriter.createOrFold<comb::AndOp>(loc, curMasked, sliceMask);
        storeSlice =
            rewriter.createOrFold<comb::MuxOp>(loc, enable, storeSlice, oldSlice);
      }
      Value updated = rewriter.createOrFold<comb::OrOp>(loc, cleared, storeSlice);
      Value storeVal = rewriter.createOrFold<comb::AndOp>(loc, updated, totalMaskCst);

      rewriter.create<mlir::func::CallOp>(loc, "__arcilator_sig_store_u64",
                                          TypeRange{},
                                          ValueRange{sigIdVal, storeVal, procIdVal});
      rewriter.eraseOp(op);
      if (resolved.dynamicOffsetOp &&
          llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                       [](Value v) { return v.use_empty(); }))
        rewriter.eraseOp(resolved.dynamicOffsetOp);
      return success();
    }

    if (sliceOffset != 0) {
      Value shiftAmt = buildI64Constant(rewriter, loc, sliceOffset);
      value64 = rewriter.createOrFold<comb::ShlOp>(loc, value64, shiftAmt);
    }

    APInt totalMask = APInt::getAllOnes(64);
    if (totalWidth < 64)
      totalMask = APInt::getAllOnes(totalWidth).zext(64);
    Value totalMaskCst = hw::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(rewriter.getI64Type(), totalMask));

    const bool isFullWrite =
        (sliceOffset == 0 && static_cast<uint64_t>(valueWidth) == totalWidth);
    Value storeVal;
    if (isFullWrite) {
      Value newVal = rewriter.createOrFold<comb::AndOp>(loc, value64, totalMaskCst);
      if (isNbaDrive) {
        Value mask = totalMaskCst;
        if (!constEnable || !*constEnable) {
          Value zero = buildI64Constant(rewriter, loc, 0);
          mask = rewriter.createOrFold<comb::MuxOp>(loc, enable, mask, zero);
        }
        rewriter.create<mlir::func::CallOp>(
            loc, "__arcilator_sig_store_nba_masked_u64", TypeRange{},
            ValueRange{sigIdVal, mask, newVal, procIdVal});
        rewriter.eraseOp(op);
        if (resolved.dynamicOffsetOp &&
            llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                         [](Value v) { return v.use_empty(); }))
          rewriter.eraseOp(resolved.dynamicOffsetOp);
        return success();
      }
      storeVal = newVal;
      if (!constEnable || !*constEnable) {
        Value cur =
            rewriter
                .create<mlir::func::CallOp>(loc, loadCallee,
                                            rewriter.getI64Type(),
                                            ValueRange{sigIdVal, procIdVal})
                .getResult(0);
        Value curMasked = rewriter.createOrFold<comb::AndOp>(loc, cur, totalMaskCst);
        storeVal =
            rewriter.createOrFold<comb::MuxOp>(loc, enable, newVal, curMasked);
      }
    } else {
      Value cur =
          rewriter
              .create<mlir::func::CallOp>(loc, loadCallee,
                                          rewriter.getI64Type(),
                                          ValueRange{sigIdVal, procIdVal})
              .getResult(0);

      unsigned sliceWidth = static_cast<unsigned>(valueWidth);
      APInt sliceMask = APInt::getAllOnes(sliceWidth).zext(64) << sliceOffset;
      Value sliceMaskCst = hw::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(rewriter.getI64Type(), sliceMask));
      if (isNbaDrive) {
        Value mask = sliceMaskCst;
        if (!constEnable || !*constEnable) {
          Value zero = buildI64Constant(rewriter, loc, 0);
          mask = rewriter.createOrFold<comb::MuxOp>(loc, enable, mask, zero);
        }
        Value inserted =
            rewriter.createOrFold<comb::AndOp>(loc, value64, sliceMaskCst);
        rewriter.create<mlir::func::CallOp>(
            loc, "__arcilator_sig_store_nba_masked_u64", TypeRange{},
            ValueRange{sigIdVal, mask, inserted, procIdVal});
        rewriter.eraseOp(op);
        if (resolved.dynamicOffsetOp &&
            llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                         [](Value v) { return v.use_empty(); }))
          rewriter.eraseOp(resolved.dynamicOffsetOp);
        return success();
      }
      APInt clearMask = APInt::getAllOnes(64) ^ sliceMask;
      Value clearMaskCst = hw::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(rewriter.getI64Type(), clearMask));

      Value cleared = rewriter.createOrFold<comb::AndOp>(loc, cur, clearMaskCst);
      Value inserted = rewriter.createOrFold<comb::AndOp>(loc, value64, sliceMaskCst);
      Value storeSlice = inserted;
      if (!constEnable || !*constEnable) {
        Value oldSlice = rewriter.createOrFold<comb::AndOp>(loc, cur, sliceMaskCst);
        storeSlice =
            rewriter.createOrFold<comb::MuxOp>(loc, enable, storeSlice, oldSlice);
      }
      Value updated = rewriter.createOrFold<comb::OrOp>(loc, cleared, storeSlice);
      storeVal = rewriter.createOrFold<comb::AndOp>(loc, updated, totalMaskCst);
    }

    rewriter.create<mlir::func::CallOp>(loc, storeCallee,
                                        TypeRange{},
                                        ValueRange{sigIdVal, storeVal, procIdVal});
    rewriter.eraseOp(op);

    if (resolved.dynamicOffsetOp &&
        llvm::all_of(resolved.dynamicOffsetOp->getResults(),
                     [](Value v) { return v.use_empty(); }))
      rewriter.eraseOp(resolved.dynamicOffsetOp);
    return success();
  }
};

/// `llhd.sig.extract` -> approximate lowering to `comb.extract` when the low
/// bit is constant (the inout-ness is dropped by the type converter).
struct SigExtractOpConversion : public OpConversionPattern<llhd::SigExtractOp> {
  using OpConversionPattern<llhd::SigExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::SigExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto lowBit = getConstantLowBit(op.getLowBit());
    auto outInout = dyn_cast<hw::InOutType>(op.getResult().getType());
    if (!outInout)
      return rewriter.notifyMatchFailure(op, "expected inout result type");
    auto outTy = dyn_cast<IntegerType>(outInout.getElementType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "expected integer element type");

    Value inputVal = adaptor.getInput();
    Value inputHandle = stripCasts(inputVal);
    if (auto *defOp = inputHandle.getDefiningOp()) {
      if (auto sigIdAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
        auto lowBit = getConstantLowBit(op.getLowBit());
        if (!lowBit) {
          // Preserve dynamic bit selects on runtime-managed signals for the
          // probe/drive conversions, which lower them to load/shift/extract or
          // load/modify/store sequences.
          auto dyn = rewriter.create<mlir::UnrealizedConversionCastOp>(
              op.getLoc(), TypeRange{outTy},
              ValueRange{inputVal, adaptor.getLowBit()});
          dyn->setAttr(kArcilatorSigDynExtractAttr, rewriter.getUnitAttr());
          rewriter.replaceOp(op, dyn.getResult(0));
          return success();
        }

        uint64_t baseOffset = 0;
        if (auto offAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigOffsetAttr))
          baseOffset = static_cast<uint64_t>(offAttr.getInt());
        uint64_t totalWidth = 0;
        if (auto widthAttr =
                defOp->getAttrOfType<IntegerAttr>(kArcilatorSigTotalWidthAttr))
          totalWidth = static_cast<uint64_t>(widthAttr.getInt());
        if (totalWidth == 0)
          totalWidth = static_cast<uint64_t>(hw::getBitWidth(inputVal.getType()));

        uint64_t newOffset = baseOffset + *lowBit;
        if (totalWidth == 0 || totalWidth > 64 ||
            newOffset + outTy.getWidth() > totalWidth)
          return rewriter.notifyMatchFailure(op, "extract slice exceeds runtime width");

        Value handleBits = hw::ConstantOp::create(
            rewriter, op.getLoc(),
            rewriter.getIntegerAttr(outTy, 0));
        if (auto *newDef = handleBits.getDefiningOp()) {
          newDef->setAttr(kArcilatorSigIdAttr, sigIdAttr);
          newDef->setAttr(kArcilatorSigOffsetAttr,
                          rewriter.getI32IntegerAttr(newOffset));
          newDef->setAttr(kArcilatorSigTotalWidthAttr,
                          rewriter.getI32IntegerAttr(totalWidth));
        }
        rewriter.replaceOp(op, handleBits);
        return success();
      }
    }

    auto inTy = dyn_cast<IntegerType>(inputVal.getType());
    if (!inTy)
      return rewriter.notifyMatchFailure(op, "expected integer input type");

    // Handle constant bit indices with a direct extract, otherwise lower to a
    // variable shift and then extract bit 0.
    if (lowBit) {
      rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, outTy, inputVal, *lowBit);
      return success();
    }

    Value lowBitVal = adaptor.getLowBit();
    auto lowBitTy = dyn_cast<IntegerType>(lowBitVal.getType());
    if (!lowBitTy)
      return rewriter.notifyMatchFailure(op, "expected integer lowBit type");

    // comb.shru requires uniform operand widths.
    if (lowBitTy.getWidth() != inTy.getWidth()) {
      Location loc = op.getLoc();
      if (lowBitTy.getWidth() < inTy.getWidth()) {
        unsigned padWidth = inTy.getWidth() - lowBitTy.getWidth();
        Value pad = hw::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(rewriter.getIntegerType(padWidth), 0));
        lowBitVal = comb::ConcatOp::create(rewriter, loc, pad, lowBitVal);
      } else {
        lowBitVal = comb::ExtractOp::create(rewriter, loc, lowBitVal, 0,
                                            inTy.getWidth());
      }
    }

    Value shifted = rewriter.createOrFold<comb::ShrUOp>(op.getLoc(), inputVal,
                                                        lowBitVal);
    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, outTy, shifted, 0);
    return success();
  }
};

/// `llhd.sig.struct_extract` -> `hw.struct_extract` (drop inout semantics)
struct SigStructExtractOpConversion
    : public OpConversionPattern<llhd::SigStructExtractOp> {
  using OpConversionPattern<llhd::SigStructExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::SigStructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<hw::StructType>(adaptor.getInput().getType()))
      return rewriter.notifyMatchFailure(
          op, "expected struct input after conversion");
    Value inputHandle = stripCasts(adaptor.getInput());
    if (auto *defOp = inputHandle.getDefiningOp()) {
      if (auto sigIdAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
        uint64_t baseOffset = 0;
        if (auto offAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigOffsetAttr))
          baseOffset = static_cast<uint64_t>(offAttr.getInt());
        uint64_t totalWidth = 0;
        if (auto widthAttr =
                defOp->getAttrOfType<IntegerAttr>(kArcilatorSigTotalWidthAttr))
          totalWidth = static_cast<uint64_t>(widthAttr.getInt());
        if (totalWidth == 0)
          totalWidth =
              static_cast<uint64_t>(hw::getBitWidth(adaptor.getInput().getType()));
        if (totalWidth == 0 || totalWidth > 64)
          return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");

        auto structTy = type_cast<hw::StructType>(adaptor.getInput().getType());
        auto fieldIndexOpt = structTy.getFieldIndex(op.getFieldAttr());
        if (!fieldIndexOpt)
          return rewriter.notifyMatchFailure(op, "unknown struct field");
        auto elements = structTy.getElements();
        Type fieldTy = elements[*fieldIndexOpt].type;
        int64_t fieldWidth = hw::getBitWidth(fieldTy);
        if (fieldWidth <= 0)
          return rewriter.notifyMatchFailure(op, "unsupported extracted field type");

        // Match HW struct packing (MSB-first). Special-case the 4-state
        // `{value, unknown}` encoding which we canonicalize as value in low
        // bits.
        uint64_t fieldOffsetLSB = 0;
        if (elements.size() == 2 &&
            elements[0].name.getValue() == "value" &&
            elements[1].name.getValue() == "unknown") {
          int64_t w = hw::getBitWidth(elements[0].type);
          if (w <= 0 || hw::getBitWidth(elements[1].type) != w)
            return rewriter.notifyMatchFailure(op, "unsupported 4-state struct width");
          fieldOffsetLSB = *fieldIndexOpt == 0 ? 0ULL : static_cast<uint64_t>(w);
        } else {
          uint64_t prefixWidth = 0;
          for (uint32_t i = 0; i < *fieldIndexOpt; ++i) {
            int64_t w = hw::getBitWidth(elements[i].type);
            if (w <= 0)
              return rewriter.notifyMatchFailure(op,
                                                 "unsupported struct field width");
            prefixWidth += static_cast<uint64_t>(w);
          }
          uint64_t structWidth = static_cast<uint64_t>(hw::getBitWidth(structTy));
          uint64_t uFieldWidth = static_cast<uint64_t>(fieldWidth);
          if (prefixWidth + uFieldWidth > structWidth)
            return rewriter.notifyMatchFailure(op, "struct field offset overflow");
          fieldOffsetLSB = structWidth - prefixWidth - uFieldWidth;
        }

        uint64_t newOffset = baseOffset + fieldOffsetLSB;
        if (newOffset + static_cast<uint64_t>(fieldWidth) > totalWidth)
          return rewriter.notifyMatchFailure(op, "struct field slice exceeds runtime width");

        Value handleToken = createZeroHWConstant(rewriter, op.getLoc(), fieldTy);
        if (!handleToken)
          return rewriter.notifyMatchFailure(op, "failed to materialize handle token");
        if (auto *newDef = handleToken.getDefiningOp()) {
          newDef->setAttr(kArcilatorSigIdAttr, sigIdAttr);
          newDef->setAttr(kArcilatorSigOffsetAttr,
                          rewriter.getI32IntegerAttr(newOffset));
          newDef->setAttr(kArcilatorSigTotalWidthAttr,
                          rewriter.getI32IntegerAttr(totalWidth));
        }
        rewriter.replaceOp(op, handleToken);
        return success();
      }
    }

    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(op, adaptor.getInput(),
                                                     op.getFieldAttr());
    return success();
  }
};

/// `llhd.sig.array_get` -> `hw.array_get` (drop inout semantics)
struct SigArrayGetOpConversion
    : public OpConversionPattern<llhd::SigArrayGetOp> {
  using OpConversionPattern<llhd::SigArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::SigArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<hw::ArrayType>(adaptor.getInput().getType()))
      return rewriter.notifyMatchFailure(
          op, "expected array input after conversion");
    Value inputHandle = stripCasts(adaptor.getInput());
    if (auto *defOp = inputHandle.getDefiningOp()) {
      if (auto sigIdAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
        auto indexOpt = getConstantLowBit(op.getIndex());
        if (!indexOpt)
          return rewriter.notifyMatchFailure(op, "non-constant array index on runtime signal");

        uint64_t baseOffset = 0;
        if (auto offAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigOffsetAttr))
          baseOffset = static_cast<uint64_t>(offAttr.getInt());
        uint64_t totalWidth = 0;
        if (auto widthAttr =
                defOp->getAttrOfType<IntegerAttr>(kArcilatorSigTotalWidthAttr))
          totalWidth = static_cast<uint64_t>(widthAttr.getInt());
        if (totalWidth == 0)
          totalWidth =
              static_cast<uint64_t>(hw::getBitWidth(adaptor.getInput().getType()));
        if (totalWidth == 0 || totalWidth > 64)
          return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");

        auto arrayTy = type_cast<hw::ArrayType>(adaptor.getInput().getType());
        Type elementTy = arrayTy.getElementType();
        int64_t elemWidth = hw::getBitWidth(elementTy);
        if (elemWidth <= 0)
          return rewriter.notifyMatchFailure(op, "unsupported array element width");
        uint64_t idx = *indexOpt;
        if (idx >= arrayTy.getNumElements())
          return rewriter.notifyMatchFailure(op, "array index out of bounds");
        uint64_t newOffset = baseOffset + idx * static_cast<uint64_t>(elemWidth);
        if (newOffset + static_cast<uint64_t>(elemWidth) > totalWidth)
          return rewriter.notifyMatchFailure(op, "array element slice exceeds runtime width");

        Value handleToken = createZeroHWConstant(rewriter, op.getLoc(), elementTy);
        if (!handleToken)
          return rewriter.notifyMatchFailure(op, "failed to materialize handle token");
        if (auto *newDef = handleToken.getDefiningOp()) {
          newDef->setAttr(kArcilatorSigIdAttr, sigIdAttr);
          newDef->setAttr(kArcilatorSigOffsetAttr,
                          rewriter.getI32IntegerAttr(newOffset));
          newDef->setAttr(kArcilatorSigTotalWidthAttr,
                          rewriter.getI32IntegerAttr(totalWidth));
        }
        rewriter.replaceOp(op, handleToken);
        return success();
      }
    }

    rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, adaptor.getInput(),
                                                adaptor.getIndex());
    return success();
  }
};

/// `sv.struct_field_inout` -> `hw.struct_extract` (drop inout semantics)
struct StructFieldInOutOpConversion
    : public OpConversionPattern<sv::StructFieldInOutOp> {
  using OpConversionPattern<sv::StructFieldInOutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(sv::StructFieldInOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<hw::StructType>(adaptor.getInput().getType()))
      return rewriter.notifyMatchFailure(op, "expected struct input after conversion");
    Value inputHandle = stripCasts(adaptor.getInput());
    if (auto *defOp = inputHandle.getDefiningOp()) {
      if (auto sigIdAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
        uint64_t baseOffset = 0;
        if (auto offAttr = defOp->getAttrOfType<IntegerAttr>(kArcilatorSigOffsetAttr))
          baseOffset = static_cast<uint64_t>(offAttr.getInt());
        uint64_t totalWidth = 0;
        if (auto widthAttr =
                defOp->getAttrOfType<IntegerAttr>(kArcilatorSigTotalWidthAttr))
          totalWidth = static_cast<uint64_t>(widthAttr.getInt());
        if (totalWidth == 0)
          totalWidth =
              static_cast<uint64_t>(hw::getBitWidth(adaptor.getInput().getType()));
        if (totalWidth == 0 || totalWidth > 64)
          return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");

        auto structTy = type_cast<hw::StructType>(adaptor.getInput().getType());
        auto fieldIndexOpt = structTy.getFieldIndex(op.getFieldAttr());
        if (!fieldIndexOpt)
          return rewriter.notifyMatchFailure(op, "unknown struct field");
        auto elements = structTy.getElements();

        Type fieldTy = elements[*fieldIndexOpt].type;
        int64_t fieldWidth = hw::getBitWidth(fieldTy);
        if (fieldWidth <= 0)
          return rewriter.notifyMatchFailure(op, "unsupported extracted field type");

        // Match HW struct packing (MSB-first). Special-case the 4-state
        // `{value, unknown}` encoding which we canonicalize as value in low
        // bits.
        uint64_t fieldOffsetLSB = 0;
        if (elements.size() == 2 &&
            elements[0].name.getValue() == "value" &&
            elements[1].name.getValue() == "unknown") {
          int64_t w = hw::getBitWidth(elements[0].type);
          if (w <= 0 || hw::getBitWidth(elements[1].type) != w)
            return rewriter.notifyMatchFailure(op, "unsupported 4-state struct width");
          fieldOffsetLSB = *fieldIndexOpt == 0 ? 0ULL : static_cast<uint64_t>(w);
        } else {
          uint64_t prefixWidth = 0;
          for (uint32_t i = 0; i < *fieldIndexOpt; ++i) {
            int64_t w = hw::getBitWidth(elements[i].type);
            if (w <= 0)
              return rewriter.notifyMatchFailure(op, "unsupported struct field width");
            prefixWidth += static_cast<uint64_t>(w);
          }
          uint64_t structWidth = static_cast<uint64_t>(hw::getBitWidth(structTy));
          uint64_t uFieldWidth = static_cast<uint64_t>(fieldWidth);
          if (prefixWidth + uFieldWidth > structWidth)
            return rewriter.notifyMatchFailure(op, "struct field offset overflow");
          fieldOffsetLSB = structWidth - prefixWidth - uFieldWidth;
        }

        uint64_t newOffset = baseOffset + fieldOffsetLSB;
        if (newOffset + static_cast<uint64_t>(fieldWidth) > totalWidth)
          return rewriter.notifyMatchFailure(op, "struct field slice exceeds runtime width");

        Value handleToken = createZeroHWConstant(rewriter, op.getLoc(), fieldTy);
        if (!handleToken)
          return rewriter.notifyMatchFailure(op, "failed to materialize handle token");
        if (auto *newDef = handleToken.getDefiningOp()) {
          newDef->setAttr(kArcilatorSigIdAttr, sigIdAttr);
          newDef->setAttr(kArcilatorSigOffsetAttr,
                          rewriter.getI32IntegerAttr(newOffset));
          newDef->setAttr(kArcilatorSigTotalWidthAttr,
                          rewriter.getI32IntegerAttr(totalWidth));
        }
        rewriter.replaceOp(op, handleToken);
        return success();
      }
    }

    // Handle dynamic runtime signal ids (e.g. virtual interfaces stored in i64
    // fields). Encode the struct field select as a constant dynamic extract so
    // probe/drive conversions can still resolve a runtime slice.
    ResolvedRuntimeSignal resolved;
    if (succeeded(resolveRuntimeSignal(adaptor.getInput(), resolved)) &&
        resolved.dynSigId) {
      uint64_t baseOffset = resolved.baseOffset;
      uint64_t totalWidth = resolved.totalWidth;
      if (totalWidth == 0)
        totalWidth = static_cast<uint64_t>(hw::getBitWidth(adaptor.getInput().getType()));
      if (totalWidth == 0 || totalWidth > 64)
        return rewriter.notifyMatchFailure(op, "unsupported runtime signal width");

      auto structTy = type_cast<hw::StructType>(adaptor.getInput().getType());
      auto fieldIndexOpt = structTy.getFieldIndex(op.getFieldAttr());
      if (!fieldIndexOpt)
        return rewriter.notifyMatchFailure(op, "unknown struct field");
      auto elements = structTy.getElements();

      Type fieldTy = elements[*fieldIndexOpt].type;
      int64_t fieldWidth = hw::getBitWidth(fieldTy);
      if (fieldWidth <= 0)
        return rewriter.notifyMatchFailure(op, "unsupported extracted field type");

      uint64_t fieldOffsetLSB = 0;
      if (elements.size() == 2 &&
          elements[0].name.getValue() == "value" &&
          elements[1].name.getValue() == "unknown") {
        int64_t w = hw::getBitWidth(elements[0].type);
        if (w <= 0 || hw::getBitWidth(elements[1].type) != w)
          return rewriter.notifyMatchFailure(op, "unsupported 4-state struct width");
        fieldOffsetLSB = *fieldIndexOpt == 0 ? 0ULL : static_cast<uint64_t>(w);
      } else {
        uint64_t prefixWidth = 0;
        for (uint32_t i = 0; i < *fieldIndexOpt; ++i) {
          int64_t w = hw::getBitWidth(elements[i].type);
          if (w <= 0)
            return rewriter.notifyMatchFailure(op, "unsupported struct field width");
          prefixWidth += static_cast<uint64_t>(w);
        }
        uint64_t structWidth = static_cast<uint64_t>(hw::getBitWidth(structTy));
        uint64_t uFieldWidth = static_cast<uint64_t>(fieldWidth);
        if (prefixWidth + uFieldWidth > structWidth)
          return rewriter.notifyMatchFailure(op, "struct field offset overflow");
        fieldOffsetLSB = structWidth - prefixWidth - uFieldWidth;
      }

      uint64_t newOffset = baseOffset + fieldOffsetLSB;
      if (newOffset + static_cast<uint64_t>(fieldWidth) > totalWidth)
        return rewriter.notifyMatchFailure(op, "struct field slice exceeds runtime width");

      Value offsetVal = buildI64Constant(rewriter, op.getLoc(), fieldOffsetLSB);
      auto dyn = rewriter.create<mlir::UnrealizedConversionCastOp>(
          op.getLoc(), TypeRange{fieldTy}, ValueRange{adaptor.getInput(), offsetVal});
      dyn->setAttr(kArcilatorSigDynExtractAttr, rewriter.getUnitAttr());
      rewriter.replaceOp(op, dyn.getResult(0));
      return success();
    }

    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(op, adaptor.getInput(),
                                                     op.getFieldAttr());
    return success();
  }
};

/// `sv.assign` -> erase (drop inout storage semantics)
struct SVAssignOpConversion : public OpConversionPattern<sv::AssignOp> {
  using OpConversionPattern<sv::AssignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(sv::AssignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.eraseOp(op);
    return success();
  }
};

/// `sv.bpassign` -> erase (drop inout storage semantics)
struct SVBPAssignOpConversion : public OpConversionPattern<sv::BPAssignOp> {
  using OpConversionPattern<sv::BPAssignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(sv::BPAssignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.eraseOp(op);
    return success();
  }
};

/// `llhd.wait` -> `arc.output` (surface the yielded values; drop scheduling)
struct WaitOpConversion : public OpConversionPattern<llhd::WaitOp> {
  using OpConversionPattern<llhd::WaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arc::OutputOp>(op, adaptor.getYieldOperands());
    return success();
  }
};

/// `llhd.now` -> runtime read + time struct materialization
struct NowOpConversion : public OpConversionPattern<llhd::NowOp> {
  using OpConversionPattern<llhd::NowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::NowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto structTy = dyn_cast<hw::StructType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!structTy)
      return rewriter.notifyMatchFailure(op, "expected time struct type");
    auto timeFieldTy = structTy.getFieldType("time");
    auto deltaFieldTy = structTy.getFieldType("delta");
    auto epsilonFieldTy = structTy.getFieldType("epsilon");
    if (!timeFieldTy || !deltaFieldTy || !epsilonFieldTy)
      return rewriter.notifyMatchFailure(op, "malformed time struct layout");

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto fnType = rewriter.getFunctionType({}, {rewriter.getI64Type()});
    (void)getOrInsertFunc(module, "__arcilator_now_fs", fnType);
    auto nowFs = rewriter.create<mlir::func::CallOp>(op.getLoc(),
                                                     "__arcilator_now_fs",
                                                     rewriter.getI64Type())
                     .getResult(0);

    Value delta = hw::ConstantOp::create(rewriter, op.getLoc(), deltaFieldTy, 0);
    Value eps = hw::ConstantOp::create(rewriter, op.getLoc(), epsilonFieldTy, 0);
    rewriter.replaceOpWithNewOp<hw::StructCreateOp>(
        op, structTy, ValueRange{nowFs, delta, eps});
    return success();
  }
};

/// `llhd.time_to_int` -> extract `time` field from struct
struct TimeToIntOpConversion : public OpConversionPattern<llhd::TimeToIntOp> {
  using OpConversionPattern<llhd::TimeToIntOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::TimeToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = dyn_cast<hw::StructType>(
        typeConverter->convertType(op.getInput().getType()));
    if (!structTy)
      return rewriter.notifyMatchFailure(op, "expected time struct type");
    rewriter.replaceOpWithNewOp<hw::StructExtractOp>(op, adaptor.getInput(),
                                                     rewriter.getStringAttr("time"));
    return success();
  }
};

/// `llhd.int_to_time` -> build a time struct from fs + zero delta/epsilon
struct IntToTimeOpConversion : public OpConversionPattern<llhd::IntToTimeOp> {
  using OpConversionPattern<llhd::IntToTimeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::IntToTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto structTy = dyn_cast<hw::StructType>(
        typeConverter->convertType(op.getResult().getType()));
    if (!structTy)
      return rewriter.notifyMatchFailure(op, "expected time struct type");
    auto deltaFieldTy = structTy.getFieldType("delta");
    auto epsilonFieldTy = structTy.getFieldType("epsilon");
    if (!deltaFieldTy || !epsilonFieldTy)
      return rewriter.notifyMatchFailure(op, "malformed time struct layout");
    Value delta = hw::ConstantOp::create(rewriter, op.getLoc(), deltaFieldTy, 0);
    Value eps = hw::ConstantOp::create(rewriter, op.getLoc(), epsilonFieldTy, 0);
    rewriter.replaceOpWithNewOp<hw::StructCreateOp>(
        op, structTy, ValueRange{adaptor.getInput(), delta, eps});
    return success();
  }
};

/// `llhd.halt` -> `arc.output` (surface the yielded values; drop scheduling)
struct HaltOpConversion : public OpConversionPattern<llhd::HaltOp> {
  using OpConversionPattern<llhd::HaltOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::HaltOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arc::OutputOp>(op, adaptor.getYieldOperands());
    return success();
  }
};

/// Helper to materialize the converted time struct type used to lower
/// `llhd.constant_time`. Store the components separately so downstream passes
/// can preserve delta/epsilon information even before a proper time semantics
/// layer exists in Arc.
static hw::StructType getTimeStructType(MLIRContext *ctx) {
  SmallVector<hw::StructType::FieldInfo> fields = {
      {StringAttr::get(ctx, "time"), IntegerType::get(ctx, 64)},
      {StringAttr::get(ctx, "delta"), IntegerType::get(ctx, 32)},
      {StringAttr::get(ctx, "epsilon"), IntegerType::get(ctx, 32)},
  };
  return hw::StructType::get(ctx, fields);
}

/// `llhd.constant_time` -> `hw.aggregate_constant` (time, delta, epsilon)
struct ConstantTimeOpConversion
    : public OpConversionPattern<llhd::ConstantTimeOp> {
  using OpConversionPattern<llhd::ConstantTimeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::ConstantTimeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the LLHD time type into the tuple backing we use in Arc.
    auto convertedType =
        typeConverter->convertType(op.getResult().getType());
    auto structTy = dyn_cast_or_null<hw::StructType>(convertedType);
    if (!structTy)
      return rewriter.notifyMatchFailure(op, "expected struct type for time");

    // Decode the time attribute into a unified integer value in femtoseconds.
    auto timeAttr = op.getValue();
    uint64_t scale = llvm::StringSwitch<uint64_t>(timeAttr.getTimeUnit())
                         .Case("fs", 1ULL)
                         .Case("ps", 1000ULL)
                         .Case("ns", 1000ULL * 1000ULL)
                         .Case("us", 1000ULL * 1000ULL * 1000ULL)
                         .Case("ms", 1000ULL * 1000ULL * 1000ULL * 1000ULL)
                         .Case("s", 1000ULL * 1000ULL * 1000ULL * 1000ULL *
                                       1000ULL)
                         .Default(0);
    if (scale == 0)
      return rewriter.notifyMatchFailure(op, "unsupported time unit");

    auto timeField = structTy.getFieldType("time");
    auto deltaField = structTy.getFieldType("delta");
    auto epsilonField = structTy.getFieldType("epsilon");
    if (!timeField || !deltaField || !epsilonField)
      return rewriter.notifyMatchFailure(op, "malformed time struct layout");

    SmallVector<Attribute> fields;
    fields.push_back(rewriter.getIntegerAttr(
        timeField, timeAttr.getTime() * scale));
    fields.push_back(rewriter.getIntegerAttr(deltaField, timeAttr.getDelta()));
    fields.push_back(
        rewriter.getIntegerAttr(epsilonField, timeAttr.getEpsilon()));

    auto agg = rewriter.getArrayAttr(fields);
    rewriter.replaceOpWithNewOp<hw::AggregateConstantOp>(op, structTy, agg);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace circt {
#define GEN_PASS_DEF_CONVERTTOARCSPASS
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

namespace {
struct ConvertToArcsPass
    : public circt::impl::ConvertToArcsPassBase<ConvertToArcsPass> {
  using ConvertToArcsPassBase::ConvertToArcsPassBase;
  void runOnOperation() override;
};
} // namespace

void ConvertToArcsPass::runOnOperation() {
  llvm::DenseMap<uint64_t, ArcilatorRuntimeSigInit> runtimeSigInits;

  // Drop dead string labels early so they don't leak SV constants through the
  // pipeline.
  SmallVector<Operation *> toErase;
  getOperation().walk([&](llhd::SignalOp sig) {
    if (sig.getResult().use_empty()) {
      if (auto inout = dyn_cast<hw::InOutType>(sig.getType()))
        if (isa<hw::StringType>(inout.getElementType()))
          toErase.push_back(sig);
    }
  });
  getOperation().walk([&](sv::ConstantStrOp cst) {
    if (cst.use_empty())
      toErase.push_back(cst);
  });
  for (Operation *op : toErase)
    op->erase();

  // Before running the (still incomplete) LLHD-to-Arc conversion, try to
  // simplify away LLHD signal storage for common "single-shot" processes such
  // as `initial` blocks without waits or delays.
  getOperation().walk([&](hw::HWModuleOp module) {
    (void)lowerModuleLevelInitDrives(module);
    for (auto fin :
         llvm::make_early_inc_range(module.getOps<llhd::FinalOp>()))
      (void)lowerSimpleFinalSignals(fin);
    for (auto proc :
         llvm::make_early_inc_range(module.getOps<llhd::ProcessOp>())) {
      (void)lowerSimpleProcessSignals(proc);
      if (succeeded(sinkProcessResultDrives(proc)))
        continue;
      // Keep one-shot `llhd.process` ops (no waits/delays) in the scheduled
      // process pipeline. The cycle scheduler models their "run once then stop"
      // semantics via the runtime PC state. Converting them to `seq.initial`
      // causes the body to execute on every evaluation, clobbering procedural
      // state (e.g. counters in `always @(posedge ...)` blocks) and leading to
      // waveform mismatches vs Questa.
      if (!needsCycleScheduler(proc))
        (void)convertOneShotProcessToInitial(proc);
    }
  });

  // NOTE: LLHD `llhd.process` pre-lowering is currently disabled. The existing
  // heuristic lowering to `arc.state` is incomplete and can lead to non-
  // terminating type legalization. A proper scheduler/stateful lowering should
  // live in LLHD transforms (e.g. `llhd-deseq`) or be implemented here as a
  // dedicated conversion.

  // Assign stable ids for scheduled-process lowering. These ids are consumed by
  // the arcilator-generated driver runtime to retain per-process state (PC and
  // wait bookkeeping) in a cycle-driven execution model.
  uint32_t nextProcId = 0;
  uint32_t nextWaitId = 0;
  uint32_t nextSigId = 0;
  OpBuilder idBuilder(&getContext());
  getOperation().walk([&](llhd::ProcessOp proc) {
    if (!proc->hasAttr(kArcilatorProcIdAttr))
      proc->setAttr(kArcilatorProcIdAttr,
                    idBuilder.getI32IntegerAttr(nextProcId++));
    if (needsCycleScheduler(proc) && !proc->hasAttr(kArcilatorNeedsSchedulerAttr))
      proc->setAttr(kArcilatorNeedsSchedulerAttr, idBuilder.getUnitAttr());
    proc.walk([&](llhd::WaitOp wait) {
      if (!wait->hasAttr(kArcilatorWaitIdAttr))
        wait->setAttr(kArcilatorWaitIdAttr,
                      idBuilder.getI32IntegerAttr(nextWaitId++));
    });
  });
  getOperation().walk([&](llhd::SignalOp sig) {
    if (auto sigIdAttr = sig->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)) {
      uint64_t id = static_cast<uint64_t>(sigIdAttr.getInt());
      if (id + 1 > nextSigId)
        nextSigId = static_cast<uint32_t>(id + 1);
    }
  });
  getOperation().walk([&](llhd::SignalOp sig) {
    if (!sig->hasAttr(kArcilatorSigIdAttr))
      sig->setAttr(kArcilatorSigIdAttr, idBuilder.getI32IntegerAttr(nextSigId++));
  });

  // Pre-lower scheduled LLHD processes into `arc.execute` state machines before
  // running dialect conversion. Dialect conversion is type-driven and would
  // otherwise eagerly convert `llhd.wait` terminators (dropping control flow)
  // before the process-level scheduler rewrite can run.
  bool schedulerFailed = false;
  mlir::PatternRewriter schedulerRewriter(&getContext());
  getOperation().walk([&](hw::HWModuleOp module) {
    for (auto proc :
         llvm::make_early_inc_range(module.getOps<llhd::ProcessOp>())) {
      if (!proc->hasAttr(kArcilatorNeedsSchedulerAttr))
        continue;
      auto procIdAttr = proc->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr);
      uint32_t procId =
          procIdAttr ? static_cast<uint32_t>(procIdAttr.getInt()) : 0;
	      auto cloneIntoBody = [](Operation *inner) {
		        // Clone constants and LLHD signal/probe declarations into the body so
		        // scheduled-process pre-lowering does not capture `!hw.inout` values as
		        // `arc.execute` operands (those would require unresolved
		        // materializations during type conversion).
		        if (inner->hasTrait<OpTrait::ConstantLike>())
		          return true;
		        if (inner->hasAttr(kArcilatorSigIdAttr) ||
		            inner->hasAttr(kArcilatorSigOffsetAttr) ||
		            inner->hasAttr(kArcilatorSigTotalWidthAttr))
		          return true;
		        return isa<llhd::SignalOp, llhd::PrbOp, llhd::SigExtractOp,
		                   llhd::SigStructExtractOp, llhd::SigArrayGetOp,
		                   llhd::SigArraySliceOp, sv::StructFieldInOutOp>(inner);
		      };
      schedulerRewriter.setInsertionPoint(proc);
      auto operands = mlir::makeRegionIsolatedFromAbove(
          schedulerRewriter, proc.getBody(), cloneIntoBody);
      auto executeOp =
          ExecuteOp::create(schedulerRewriter, proc.getLoc(), TypeRange{}, operands);
      if (procIdAttr)
        executeOp->setAttr(kArcilatorProcIdAttr, procIdAttr);
      schedulerRewriter.inlineRegionBefore(proc.getBody(), executeOp.getBody(),
                                           executeOp.getBody().begin());
      if (failed(lowerCycleScheduler(executeOp, procId, schedulerRewriter))) {
        proc.emitOpError() << "failed to lower scheduled process";
        schedulerFailed = true;
        continue;
      }
      schedulerRewriter.eraseOp(proc);
    }
  });
  if (schedulerFailed) {
    emitError(getOperation().getLoc())
        << "failed to pre-lower scheduled LLHD processes";
    return signalPassFailure();
  }

  // Setup the type conversion.
  TypeConverter converter;

  // Define legal types.
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<llhd::LLHDDialect>(type.getDialect()))
      return std::nullopt;
    return type;
  });
  converter.addConversion(
      [](hw::InOutType type) -> std::optional<Type> {
        return type.getElementType();
  });
  converter.addConversion([](llhd::TimeType type) -> std::optional<Type> {
    return getTimeStructType(type.getContext());
  });
  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange inputs,
         Location loc) -> Value {
        if (inputs.size() != 1)
          return {};
        Value input = inputs.front();
        auto timeStruct = getTimeStructType(builder.getContext());
        if (isa<llhd::TimeType>(type) && input.getType() == timeStruct)
          return builder
              .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        if (type == timeStruct && isa<llhd::TimeType>(input.getType()))
          return builder
              .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        if (auto inout = dyn_cast<hw::InOutType>(input.getType())) {
          auto stripAlias = [](Type t) -> Type {
            while (auto alias = dyn_cast<hw::TypeAliasType>(t))
              t = alias.getInnerType();
            return t;
          };
          if (stripAlias(inout.getElementType()) == stripAlias(type))
            return builder.create<llhd::PrbOp>(loc, input).getResult();
        }
        if (auto desiredInOut = dyn_cast<hw::InOutType>(type)) {
          if (desiredInOut.getElementType() == input.getType())
            return builder
                .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
        }
        return {};
      });

  // When dropping `hw.inout` types to plain SSA values, materialize reads as
  // explicit probes rather than leaving unresolved casts behind.
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs,
         Location loc) -> Value {
        if (inputs.size() != 1)
          return {};
        Value input = inputs.front();
        auto inout = dyn_cast<hw::InOutType>(input.getType());
        if (!inout)
          return {};
        auto stripAlias = [](Type t) -> Type {
          while (auto alias = dyn_cast<hw::TypeAliasType>(t))
            t = alias.getInnerType();
          return t;
        };
        if (stripAlias(inout.getElementType()) != stripAlias(resultType))
          return {};
        return builder.create<llhd::PrbOp>(loc, input).getResult();
      });

  converter.addTargetMaterialization(
      [](OpBuilder &builder, hw::InOutType type, ValueRange inputs,
         Location loc) -> Value {
        if (inputs.size() != 1)
          return {};
        if (inputs.front().getType() != type.getElementType())
          return {};
        auto mat = builder.create<mlir::UnrealizedConversionCastOp>(
            loc, TypeRange{type}, inputs);
        return mat.getResult(0);
      });

  // Gather the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add<llhd::CombinationalOp>(convert);
  patterns.add<llhd::ProcessOp>(convert);
  patterns.add<llhd::YieldOp>(convert);
  patterns.add<NowOpConversion>(converter, &getContext());
  patterns.add<TimeToIntOpConversion>(converter, &getContext());
  patterns.add<IntToTimeOpConversion>(converter, &getContext());
  patterns.add<ConstantTimeOpConversion>(converter, &getContext());
  patterns.add<SignalOpConversion>(converter, &getContext(), &runtimeSigInits);
  patterns.add<ProbeOpConversion>(converter, &getContext());
  patterns.add<DrvOpConversion>(converter, &getContext());
  patterns.add<SigExtractOpConversion>(converter, &getContext());
  patterns.add<SigStructExtractOpConversion>(converter, &getContext());
  patterns.add<SigArrayGetOpConversion>(converter, &getContext());
  patterns.add<StructFieldInOutOpConversion>(converter, &getContext());
  patterns.add<SVAssignOpConversion>(converter, &getContext());
  patterns.add<SVBPAssignOpConversion>(converter, &getContext());
  patterns.add<WaitOpConversion>(converter, &getContext());
  patterns.add<HaltOpConversion>(converter, &getContext());

  // Setup the legal ops. (Sort alphabetically.)
  ConversionTarget target(getContext());
  target.addIllegalDialect<llhd::LLHDDialect>();
  // Keep `llhd.final` around for `arc::LowerState`, which lowers it into the
  // model's `arc.final` clock tree. Similarly, keep `llhd.halt` terminators
  // within `llhd.final` regions so they can be replaced with `scf.yield` by the
  // finalization lowering.
  target.addLegalOp<llhd::FinalOp>();
  target.addDynamicallyLegalOp<llhd::HaltOp>(
      [](llhd::HaltOp op) { return isa<llhd::FinalOp>(op->getParentOp()); });
  target.addDynamicallyLegalOp<llhd::WaitOp>([](llhd::WaitOp op) {
    auto parent = op->getParentOfType<llhd::ProcessOp>();
    return parent && parent->hasAttr(kArcilatorNeedsSchedulerAttr);
  });
  target.addIllegalOp<llhd::DrvOp, llhd::ProcessOp, llhd::SignalOp,
                      llhd::PrbOp, llhd::NowOp,
                      llhd::TimeToIntOp, llhd::IntToTimeOp,
                      sv::StructFieldInOutOp, sv::AssignOp, sv::BPAssignOp>();
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isa<llhd::LLHDDialect>(op->getDialect()); });

  // Disable pattern rollback to use the faster one-shot dialect conversion.
  ConversionConfig config;
  // Keep rollback enabled: this pass still contains partial LLHD lowering and
  // should fail gracefully instead of aborting.
  config.allowPatternRollback = true;

  // Apply the dialect conversion patterns.
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns),
                                    config))) {
    emitError(getOperation().getLoc()) << "conversion to arcs failed";
    return signalPassFailure();
  }

  // Collapse trivial inout<->SSA round-trips that may have been introduced as
  // materializations during conversion.
  SmallVector<Operation *> materializationsToErase;
  getOperation().walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    Value input = cast.getInputs().front();
    Value result = cast.getResult(0);
    auto inputInOut = dyn_cast<hw::InOutType>(input.getType());
    if (!inputInOut || result.getType() != inputInOut.getElementType())
      return;
    auto producer =
        input.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!producer || producer->getNumOperands() != 1 ||
        producer->getNumResults() != 1)
      return;
    if (producer.getResult(0).getType() != input.getType())
      return;
    if (producer.getInputs().front().getType() != result.getType())
      return;
    result.replaceAllUsesWith(producer.getInputs().front());
    materializationsToErase.push_back(cast);
    if (producer.getResult(0).use_empty())
      materializationsToErase.push_back(producer);
  });
  for (Operation *op : materializationsToErase)
    op->erase();

  // Collapse any remaining immediate conversion-cast round-trips (e.g. time
  // structs materialized back to `llhd.time` and then re-converted) so the
  // conversion driver doesn't trip over unresolved materializations.
  SmallVector<Operation *> toEraseGeneric;
  getOperation().walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    Value input = cast.getInputs().front();
    Value result = cast.getResult(0);
    auto producer = input.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!producer || producer->getNumOperands() != 1 ||
        producer->getNumResults() != 1)
      return;
    Value producerInput = producer.getInputs().front();
    if (producer.getResult(0).getType() != input.getType())
      return;
    if (producerInput.getType() != result.getType())
      return;
    result.replaceAllUsesWith(producerInput);
    toEraseGeneric.push_back(cast);
    if (producer.getResult(0).use_empty())
      toEraseGeneric.push_back(producer);
  });
  for (Operation *op : toEraseGeneric)
    op->erase();

  // Lower boolean conversions from 4-state `{value, unknown}` structs. These
  // commonly arise from `moore.to_builtin_bool` and must be resolved before
  // lowering to LLVM.
  SmallVector<mlir::UnrealizedConversionCastOp> boolCastsToErase;
  getOperation().walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    Value input = cast.getInputs().front();
    auto structTy = dyn_cast<hw::StructType>(input.getType());
    if (!structTy)
      return;
    auto outTy = dyn_cast<IntegerType>(cast.getResult(0).getType());
    if (!outTy || outTy.getWidth() != 1)
      return;
    auto valueFieldTy = dyn_cast_or_null<IntegerType>(structTy.getFieldType("value"));
    auto unknownFieldTy =
        dyn_cast_or_null<IntegerType>(structTy.getFieldType("unknown"));
    if (!valueFieldTy || !unknownFieldTy ||
        valueFieldTy.getWidth() != unknownFieldTy.getWidth())
      return;

    OpBuilder b(cast);
    Location loc = cast.getLoc();
    Value valueBits = b.createOrFold<hw::StructExtractOp>(
        loc, input, b.getStringAttr("value"));
    Value unknownBits = b.createOrFold<hw::StructExtractOp>(
        loc, input, b.getStringAttr("unknown"));

    Value zeroValue = hw::ConstantOp::create(b, loc, valueFieldTy, 0);
    Value zeroUnknown = hw::ConstantOp::create(b, loc, unknownFieldTy, 0);
    Value valueNeZero =
        comb::ICmpOp::create(b, loc, comb::ICmpPredicate::ne, valueBits,
                             zeroValue, /*twoState=*/true);
    Value unknownEqZero =
        comb::ICmpOp::create(b, loc, comb::ICmpPredicate::eq, unknownBits,
                             zeroUnknown, /*twoState=*/true);
    Value boolVal =
        comb::AndOp::create(b, loc, valueNeZero, unknownEqZero, true);

    cast.getResult(0).replaceAllUsesWith(boolVal);
    boolCastsToErase.push_back(cast);
  });
  for (auto cast : boolCastsToErase)
    cast.erase();

  // Outline operations into arcs.
  Converter outliner;
  outliner.tapRegisters = tapRegisters;
  if (failed(outliner.run(getOperation())))
    return signalPassFailure();

  // Canonicalize packing/unpacking of 4-state structs when bitcasting to/from
  // integers.
  //
  // The cycle scheduler, edge detectors, and runtime signal packing assume a
  // stable layout for any 4-state `{value, unknown}` structs (including when
  // they appear nested inside interface bundles):
  //   [value (low bits), unknown (high bits)]
  //
  // Materialize this layout explicitly to avoid relying on `hw.bitcast`
  // field-order semantics for aggregates.
  SmallVector<hw::BitcastOp> stableBitcastsToErase;
  auto stripInoutType = [](Type ty) -> Type {
    if (auto inoutTy = dyn_cast<hw::InOutType>(ty))
      return inoutTy.getElementType();
    return ty;
  };
  struct FourStateLayout {
    unsigned fieldWidth = 0;
    unsigned valueIdx = 0;
    unsigned unknownIdx = 0;
  };

  auto getFourStateLayout = [&](Type ty) -> std::optional<FourStateLayout> {
    auto structTy = dyn_cast<hw::StructType>(stripInoutType(ty));
    if (!structTy)
      return std::nullopt;
    auto elements = structTy.getElements();
    if (elements.size() != 2)
      return std::nullopt;
    auto int0 = dyn_cast<IntegerType>(elements[0].type);
    auto int1 = dyn_cast<IntegerType>(elements[1].type);
    if (!int0 || !int1 || int0.getWidth() != int1.getWidth())
      return std::nullopt;

    auto classify = [](StringRef name) -> std::optional<bool> {
      // Return `true` for value, `false` for unknown, and `nullopt` otherwise.
      if (name == "value" || name == "aval")
        return true;
      if (name == "unknown" || name == "bval")
        return false;
      return std::nullopt;
    };

    std::optional<unsigned> valueIdx;
    std::optional<unsigned> unknownIdx;
    for (auto [idx, element] : llvm::enumerate(elements)) {
      auto kind = classify(element.name.getValue());
      if (!kind)
        continue;
      if (*kind)
        valueIdx = static_cast<unsigned>(idx);
      else
        unknownIdx = static_cast<unsigned>(idx);
    }
    if (!valueIdx || !unknownIdx || *valueIdx == *unknownIdx)
      return std::nullopt;

    FourStateLayout layout;
    layout.fieldWidth = static_cast<unsigned>(int0.getWidth());
    layout.valueIdx = *valueIdx;
    layout.unknownIdx = *unknownIdx;
    return layout;
  };

  std::function<bool(Type)> containsFourState;
  containsFourState = [&](Type ty) -> bool {
    ty = stripInoutType(ty);
    if (auto structTy = dyn_cast<hw::StructType>(ty)) {
      if (getFourStateLayout(structTy))
        return true;
      for (auto element : structTy.getElements())
        if (containsFourState(element.type))
          return true;
      return false;
    }
    if (auto arrTy = dyn_cast<hw::ArrayType>(ty))
      return containsFourState(arrTy.getElementType());
    return false;
  };

  std::function<Value(OpBuilder &, Location, Value, Type)> packStable;
  std::function<Value(OpBuilder &, Location, Value, Type)> unpackStable;

  packStable = [&](OpBuilder &b, Location loc, Value v, Type ty) -> Value {
    ty = stripInoutType(ty);
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      if (v.getType() == intTy)
        return v;
      if (auto vIntTy = dyn_cast<IntegerType>(v.getType())) {
        if (vIntTy.getWidth() > intTy.getWidth())
          return comb::ExtractOp::create(b, loc, v, /*lowBit=*/0,
                                         /*width=*/intTy.getWidth());
        Value zeros =
            hw::ConstantOp::create(b, loc,
                                   b.getIntegerType(intTy.getWidth() -
                                                    vIntTy.getWidth()),
                                   0);
        return b.createOrFold<comb::ConcatOp>(loc, zeros, v);
      }
      return {};
    }

    if (auto arrTy = dyn_cast<hw::ArrayType>(ty)) {
      SmallVector<Value> parts;
      parts.reserve(arrTy.getNumElements());
      unsigned idxWidth = llvm::Log2_64_Ceil(arrTy.getNumElements());
      for (uint64_t idx = 0, e = arrTy.getNumElements(); idx < e; ++idx) {
        Value i = hw::ConstantOp::create(b, loc, b.getIntegerType(idxWidth), idx);
        Value elem = hw::ArrayGetOp::create(b, loc, v, i);
        Value packedElem = packStable(b, loc, elem, arrTy.getElementType());
        if (!packedElem)
          return {};
        parts.push_back(packedElem);
      }
      if (parts.empty())
        return {};
      Value packed = parts.front();
      for (Value part : llvm::drop_begin(parts))
        packed = b.createOrFold<comb::ConcatOp>(loc, packed, part);
      return packed;
    }

    auto structTy = dyn_cast<hw::StructType>(ty);
    if (!structTy)
      return {};

    if (auto fourState = getFourStateLayout(structTy)) {
      auto elements = structTy.getElements();
      Value valueBits = b.createOrFold<hw::StructExtractOp>(
          loc, v, elements[fourState->valueIdx].name);
      Value unknownBits = b.createOrFold<hw::StructExtractOp>(
          loc, v, elements[fourState->unknownIdx].name);
      return b.createOrFold<comb::ConcatOp>(loc, unknownBits, valueBits);
    }

    SmallVector<Value> parts;
    parts.reserve(structTy.getElements().size());
    for (auto element : structTy.getElements()) {
      Value field = b.createOrFold<hw::StructExtractOp>(loc, v, element.name);
      Value packedField = packStable(b, loc, field, element.type);
      if (!packedField)
        return {};
      parts.push_back(packedField);
    }

    if (parts.empty())
      return {};
    Value packed = parts.front();
    for (Value part : llvm::drop_begin(parts))
      packed = b.createOrFold<comb::ConcatOp>(loc, packed, part);
    return packed;
  };

  unpackStable = [&](OpBuilder &b, Location loc, Value bits, Type ty) -> Value {
    ty = stripInoutType(ty);
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      if (bits.getType() == intTy)
        return bits;
      if (auto bitsIntTy = dyn_cast<IntegerType>(bits.getType())) {
        if (bitsIntTy.getWidth() > intTy.getWidth())
          return comb::ExtractOp::create(b, loc, bits, /*lowBit=*/0,
                                         /*width=*/intTy.getWidth());
        Value zeros = hw::ConstantOp::create(
            b, loc, b.getIntegerType(intTy.getWidth() - bitsIntTy.getWidth()),
            0);
        return b.createOrFold<comb::ConcatOp>(loc, zeros, bits);
      }
      return {};
    }

    if (auto arrTy = dyn_cast<hw::ArrayType>(ty)) {
      int64_t elemWidth = hw::getBitWidth(stripInoutType(arrTy.getElementType()));
      if (elemWidth <= 0)
        return {};
      SmallVector<Value> elems(arrTy.getNumElements());
      unsigned offset = 0;
      for (int64_t i = static_cast<int64_t>(arrTy.getNumElements()) - 1; i >= 0;
           --i) {
        Value elemBits = comb::ExtractOp::create(b, loc, bits, /*lowBit=*/offset,
                                                 /*width=*/elemWidth);
        Value elemVal =
            unpackStable(b, loc, elemBits, arrTy.getElementType());
        if (!elemVal)
          return {};
        elems[static_cast<size_t>(i)] = elemVal;
        offset += static_cast<unsigned>(elemWidth);
      }
      return hw::ArrayCreateOp::create(b, loc, arrTy, elems);
    }

    auto structTy = dyn_cast<hw::StructType>(ty);
    if (!structTy)
      return {};

    if (auto fourState = getFourStateLayout(structTy)) {
      unsigned w = fourState->fieldWidth;
      Value valueBits = comb::ExtractOp::create(b, loc, bits, /*lowBit=*/0,
                                                /*width=*/w);
      Value unknownBits = comb::ExtractOp::create(b, loc, bits, /*lowBit=*/w,
                                                  /*width=*/w);
      SmallVector<Value> elems(structTy.getElements().size());
      elems[fourState->valueIdx] = valueBits;
      elems[fourState->unknownIdx] = unknownBits;
      return b.createOrFold<hw::StructCreateOp>(loc, structTy, elems);
    }

    auto elements = structTy.getElements();
    SmallVector<Value> fieldVals(elements.size());
    unsigned offset = 0;
    for (auto [idx, element] :
         llvm::enumerate(llvm::reverse(elements))) {
      (void)idx;
      int64_t fieldWidth = hw::getBitWidth(stripInoutType(element.type));
      if (fieldWidth <= 0)
        return {};
      Value fieldBits = comb::ExtractOp::create(b, loc, bits, /*lowBit=*/offset,
                                                /*width=*/fieldWidth);
      Value fieldVal = unpackStable(b, loc, fieldBits, element.type);
      if (!fieldVal)
        return {};
      fieldVals[static_cast<size_t>(elements.size()) -
                static_cast<size_t>(idx) - 1] = fieldVal;
      offset += static_cast<unsigned>(fieldWidth);
    }
    return b.createOrFold<hw::StructCreateOp>(loc, structTy, fieldVals);
  };

  getOperation().walk([&](hw::BitcastOp cast) {
    Location loc = cast.getLoc();
    Type inTy = stripInoutType(cast.getInput().getType());
    Type outTy = stripInoutType(cast.getType());

    // Only rewrite pure value bitcasts; inout bitcasts would require explicit
    // read/write ops which are outside this canonicalization's scope.
    if (cast.getInput().getType() != inTy || cast.getType() != outTy)
      return;

    // Only rewrite bitcasts that involve 4-state structs somewhere in the
    // aggregate. Otherwise preserve default `hw.bitcast` semantics.
    bool needsStableLayout = containsFourState(inTy) || containsFourState(outTy);
    if (!needsStableLayout)
      return;

    auto inIntTy = dyn_cast<IntegerType>(inTy);
    auto outIntTy = dyn_cast<IntegerType>(outTy);

    OpBuilder b(cast);

    // aggregate -> int
    if (outIntTy && isa<hw::StructType, hw::ArrayType>(inTy)) {
      Value packed = packStable(b, loc, cast.getInput(), inTy);
      if (!packed || packed.getType() != outIntTy)
        return;
      cast.getResult().replaceAllUsesWith(packed);
      stableBitcastsToErase.push_back(cast);
      return;
    }

    // int -> aggregate
    if (inIntTy && isa<hw::StructType, hw::ArrayType>(outTy)) {
      Value unpacked = unpackStable(b, loc, cast.getInput(), outTy);
      if (!unpacked || unpacked.getType() != outTy)
        return;
      cast.getResult().replaceAllUsesWith(unpacked);
      stableBitcastsToErase.push_back(cast);
      return;
    }
  });

  for (auto cast : stableBitcastsToErase)
    cast.erase();

  // Some scheduled `wait_event` patterns end up stashing signal-derived
  // expressions into the per-process frame exactly once (typically at time 0)
  // and then reusing that frozen value on every wake-up. This breaks edge
  // detection for interface clocks and similar 4-state signals.
  //
  // Detect frame slots with a single store that depends on a runtime signal load
  // and replace frame loads *outside* polling blocks with a rematerialized
  // version of the stored expression.
  getOperation().walk([&](arc::ExecuteOp execOp) {
    Region &region = execOp.getBody();
    if (region.empty())
      return;

    DenseSet<Block *> waitBlocks;
    for (Block &block : region) {
      for (Operation &op : block) {
        auto call = dyn_cast<mlir::func::CallOp>(op);
        if (!call)
          continue;
        if (call.getCallee() == "__arcilator_wait_change" ||
            call.getCallee() == "__arcilator_wait_delay") {
          waitBlocks.insert(&block);
          break;
        }
      }
    }

    auto makeKey = [](int64_t procId, int64_t slot) -> uint64_t {
      return (static_cast<uint64_t>(static_cast<uint32_t>(procId)) << 32) |
             static_cast<uint32_t>(slot);
    };

    DenseMap<uint64_t, SmallVector<mlir::func::CallOp>> storesByKey;
    DenseMap<uint64_t, SmallVector<mlir::func::CallOp>> loadsByKey;

    execOp.walk([&](mlir::func::CallOp call) {
      StringRef callee = call.getCallee();
      if (callee != "__arcilator_frame_store_u64" &&
          callee != "__arcilator_frame_load_u64")
        return;

      auto procIdBits = tryEvalIntConstant(call.getOperand(0), /*bitWidth=*/32);
      auto slotBits = tryEvalIntConstant(call.getOperand(1), /*bitWidth=*/32);
      if (!procIdBits || !slotBits)
        return;
      uint64_t key = makeKey(procIdBits->getSExtValue(), slotBits->getSExtValue());

      if (callee == "__arcilator_frame_store_u64")
        storesByKey[key].push_back(call);
      else
        loadsByKey[key].push_back(call);
    });

    auto dependsOnCall = [&](Value value, StringRef name) -> bool {
      DenseSet<Value> visited;
      SmallVector<Value> worklist;
      worklist.push_back(value);
      while (!worklist.empty()) {
        Value v = stripCasts(worklist.pop_back_val());
        if (!v || !visited.insert(v).second)
          continue;
        if (auto call = v.getDefiningOp<mlir::func::CallOp>())
          if (call.getCallee() == name)
            return true;
        if (Operation *defOp = v.getDefiningOp())
          for (Value operand : defOp->getOperands())
            worklist.push_back(operand);
      }
      return false;
    };

    SmallVector<mlir::func::CallOp> loadsToErase;
    for (auto &it : storesByKey) {
      uint64_t key = it.first;
      auto &stores = it.second;
      if (stores.size() != 1)
        continue;
      auto loadsIt = loadsByKey.find(key);
      if (loadsIt == loadsByKey.end())
        continue;
      auto store = stores.front();
      Value storedValue = store.getOperand(2);
      if (!storedValue)
        continue;
      if (!dependsOnCall(storedValue, "__arcilator_sig_load_u64"))
        continue;
      if (dependsOnCall(storedValue, "__arcilator_frame_load_u64"))
        continue;

      for (auto load : loadsIt->second) {
        if (waitBlocks.contains(load->getBlock()))
          continue;
        mlir::IRRewriter rewriter(load.getContext());
        rewriter.setInsertionPoint(load);
        DenseMap<Value, Value> memo;
        auto remat = rematerializeValueForPolling(storedValue, memo, rewriter);
        if (failed(remat) || (*remat).getType() != load.getResult(0).getType())
          continue;
        load.getResult(0).replaceAllUsesWith(*remat);
        loadsToErase.push_back(load);
      }
    }

    for (auto load : loadsToErase)
      if (load && load->use_empty())
        load.erase();
  });

  // Work around a scheduler lowering bug for `wait_event`-lowered edge
  // detectors.
  //
  // Some UVM-style testbenches rely on `always @(posedge clk)` sequential logic
  // where `clk` is a 4-state struct `{value, unknown}`. After `lowerCycleScheduler`
  // spills the pre-wait `notOld` value into the runtime frame, a later rewrite
  // can accidentally recompute `notOld` from the *post-wait* value instead of
  // reloading it, producing a vacuous `(!after) & after` trigger that never
  // fires. Detect that pattern in the resulting `arc.execute` blocks and patch
  // the trigger to use the spilled frame load.
  //
  // This keeps the lowering honest (no stubbing) and restores proper clocked
  // behavior for head-to-head VCD parity with Verilator.
  getOperation().walk([&](arc::ExecuteOp execOp) {
    bool debugEdgePatch = std::getenv("CIRCT_ARC_EDGE_PATCH_DEBUG") != nullptr;
    bool debugEdgePatchVerbose =
        std::getenv("CIRCT_ARC_EDGE_PATCH_DEBUG_VERBOSE") != nullptr;
    IntegerAttr procIdAttr =
        execOp->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr);
    int64_t debugProcId = procIdAttr ? procIdAttr.getInt() : -1;

    Region &region = execOp.getBody();
    if (region.empty())
      return;

    // Group frame-load calls by block so we can patch within a block without
    // guessing dominance. Prefer unused loads when selecting the pre-wait
    // trigger value (those are a strong indicator the lowering became vacuous).
    DenseMap<Block *, SmallVector<mlir::func::CallOp>> frameLoadsByBlock;
    for (Block &block : region) {
      for (Operation &op : block) {
        auto call = dyn_cast<mlir::func::CallOp>(op);
        if (!call)
          continue;
        if (call.getCallee() != "__arcilator_frame_load_u64")
          continue;
        if (call.getNumResults() != 1)
          continue;
        if (!call.getResult(0).getType().isInteger(64))
          continue;
        frameLoadsByBlock[&block].push_back(call);
      }
    }

    for (Block &block : region) {
      auto loadsIt = frameLoadsByBlock.find(&block);
      if (loadsIt == frameLoadsByBlock.end() || loadsIt->second.empty())
        continue;

      // Only patch the first matching broken trigger per block to keep the
      // rewrite conservative.
      mlir::func::CallOp frameLoad;
      for (auto call : loadsIt->second) {
        if (call.getResult(0).use_empty()) {
          frameLoad = call;
          break;
        }
      }
      if (!frameLoad)
        frameLoad = loadsIt->second.front();
      bool patched = false;
      if (debugEdgePatch) {
        llvm::errs() << "[convert-to-arcs] edgepatch proc_id=" << debugProcId
                     << " block=" << &block
                     << " frame_loads=" << loadsIt->second.size()
                     << " selected_unused=" << frameLoad.getResult(0).use_empty()
                     << "\n";
      }

      auto isOne = [](Value v) -> bool {
        auto cst = v.getDefiningOp<hw::ConstantOp>();
        return cst && cst.getValue().isOne();
      };
      auto xorNotOperand = [&](comb::XorOp xorOp) -> Value {
        auto inputs = xorOp.getInputs();
        if (inputs.size() != 2)
          return {};
        if (isOne(inputs[0]))
          return inputs[1];
        if (isOne(inputs[1]))
          return inputs[0];
        return {};
      };

      // Treat two values as equivalent if they are structurally identical up
      // to CSE misses (e.g. repeated `__arcilator_sig_load_u64` calls feeding
      // identical `comb.extract` / `xor 1` patterns).
      std::function<bool(Value, Value, unsigned)> equiv =
          [&](Value a, Value b, unsigned depth) -> bool {
        if (a == b)
          return true;
        if (depth > 6)
          return false;

        auto sameConst = [](Value x, Value y) -> bool {
          auto xc = x.getDefiningOp<hw::ConstantOp>();
          auto yc = y.getDefiningOp<hw::ConstantOp>();
          return xc && yc && xc.getValue() == yc.getValue();
        };
        if (sameConst(a, b))
          return true;

        auto callA = a.getDefiningOp<mlir::func::CallOp>();
        auto callB = b.getDefiningOp<mlir::func::CallOp>();
        if (callA && callB && callA.getCallee() == callB.getCallee() &&
            callA.getNumOperands() == callB.getNumOperands() &&
            callA.getNumResults() == 1 && callB.getNumResults() == 1) {
          bool argsMatch = true;
          for (auto [opA, opB] :
               llvm::zip(callA.getOperands(), callB.getOperands())) {
            if (opA == opB || sameConst(opA, opB))
              continue;
            argsMatch = false;
            break;
          }
          if (argsMatch)
            return true;
        }

        auto extA = a.getDefiningOp<comb::ExtractOp>();
        auto extB = b.getDefiningOp<comb::ExtractOp>();
        if (extA && extB && extA.getLowBit() == extB.getLowBit() &&
            extA.getType() == extB.getType())
          return equiv(extA.getInput(), extB.getInput(), depth + 1);

        auto xorA = a.getDefiningOp<comb::XorOp>();
        auto xorB = b.getDefiningOp<comb::XorOp>();
        if (xorA && xorB && xorA.getTwoState() && xorB.getTwoState()) {
          Value aOther = xorNotOperand(xorA);
          Value bOther = xorNotOperand(xorB);
          if (aOther && bOther)
            return equiv(aOther, bOther, depth + 1);
        }

        auto icmpA = a.getDefiningOp<comb::ICmpOp>();
        auto icmpB = b.getDefiningOp<comb::ICmpOp>();
        if (icmpA && icmpB && icmpA.getTwoState() == icmpB.getTwoState() &&
            icmpA.getPredicate() == icmpB.getPredicate()) {
          if (equiv(icmpA.getLhs(), icmpB.getLhs(), depth + 1) &&
              equiv(icmpA.getRhs(), icmpB.getRhs(), depth + 1))
            return true;
          switch (icmpA.getPredicate()) {
          case comb::ICmpPredicate::eq:
          case comb::ICmpPredicate::ne:
            if (equiv(icmpA.getLhs(), icmpB.getRhs(), depth + 1) &&
                equiv(icmpA.getRhs(), icmpB.getLhs(), depth + 1))
              return true;
            break;
          default:
            break;
          }
        }

        auto andA = a.getDefiningOp<comb::AndOp>();
        auto andB = b.getDefiningOp<comb::AndOp>();
        if (andA && andB && andA.getTwoState() == andB.getTwoState()) {
          auto inputsA = andA.getInputs();
          auto inputsB = andB.getInputs();
          if (inputsA.size() != inputsB.size())
            return false;
          SmallVector<bool> matched(inputsB.size(), false);
          for (Value inputA : inputsA) {
            bool found = false;
            for (size_t i = 0, e = inputsB.size(); i != e; ++i) {
              if (matched[i])
                continue;
              if (!equiv(inputA, inputsB[i], depth + 1))
                continue;
              matched[i] = true;
              found = true;
              break;
            }
            if (!found)
              return false;
          }
          return true;
        }

        return false;
      };

      // Prefer an existing unpacked `oldNot` (i1) extract from a frame load in
      // this block. If none exists, we will materialize one from `frameLoad`.
      Value oldNotValue;
      for (auto call : loadsIt->second) {
        for (Operation *user : call.getResult(0).getUsers()) {
          auto extract = dyn_cast<comb::ExtractOp>(user);
          if (!extract)
            continue;
          if (extract.getLowBit() != 0)
            continue;
          if (!extract.getType().isInteger(1))
            continue;
          oldNotValue = extract.getResult();
          break;
        }
        if (oldNotValue)
          break;
      }

      for (Operation &op : block.without_terminator()) {
        auto andOp = dyn_cast<comb::AndOp>(op);
        if (!andOp || !andOp.getTwoState())
          continue;

        // Look for `and(!A, <operands-of-A...>)` where `A` itself is an AND.
        // This is always false and indicates the edge trigger got rewritten
        // from `(!before) & after` into `(!after) & after`.
        Value notA;
        for (Value candidate : andOp.getInputs()) {
          auto xorOp = candidate.getDefiningOp<comb::XorOp>();
          if (!xorOp || !xorOp.getTwoState())
            continue;
          // Must be `xor <x>, 1` or `xor 1, <x>`.
          Value aVal = xorNotOperand(xorOp);
          if (!aVal)
            continue;
          auto aAnd = aVal.getDefiningOp<comb::AndOp>();
          if (!aAnd || !aAnd.getTwoState())
            continue;

          // Prefer matching `and(!A, A)` (where `A` may have been recomputed
          // with redundant signal loads) over the later-flattened
          // `and(!A, <operands-of-A...>)` form.
          SmallVector<Value> outerInputs;
          for (Value v : andOp.getInputs()) {
            if (v != candidate)
              outerInputs.push_back(v);
          }

          bool matchesByValue = llvm::any_of(
              outerInputs, [&](Value v) { return equiv(aVal, v, /*depth=*/0); });
          bool matches = matchesByValue;

          SmallVector<bool> used(outerInputs.size(), false);
          auto hasEquivalent = [&](Value needle) -> bool {
            for (size_t i = 0, e = outerInputs.size(); i != e; ++i) {
              if (used[i])
                continue;
              if (!equiv(needle, outerInputs[i], /*depth=*/0))
                continue;
              used[i] = true;
              return true;
            }
            return false;
          };

          bool matchesByOperands = false;
          if (!matches) {
            matchesByOperands = llvm::all_of(
                aAnd.getInputs(), [&](Value v) { return hasEquivalent(v); });
            matches = matchesByOperands;
          }

          if (debugEdgePatch && procIdAttr) {
            llvm::errs() << "[convert-to-arcs] edgepatch proc_id=" << debugProcId
                         << " candidate_matches value=" << matchesByValue
                         << " operands=" << matchesByOperands
                         << " outer_inputs=" << outerInputs.size() << "\n";
            if (debugEdgePatchVerbose) {
              llvm::errs() << "  andOp: ";
              andOp.print(llvm::errs());
              llvm::errs() << "\n";
              if (Operation *aDef = aVal.getDefiningOp()) {
                llvm::errs() << "  aVal: ";
                aDef->print(llvm::errs());
                llvm::errs() << "\n";
              }
              if (!outerInputs.empty()) {
                if (Operation *bDef = outerInputs.front().getDefiningOp()) {
                  llvm::errs() << "  bVal: ";
                  bDef->print(llvm::errs());
                  llvm::errs() << "\n";
                }
                auto aAndDbg = aVal.getDefiningOp<comb::AndOp>();
                auto bAndDbg = outerInputs.front().getDefiningOp<comb::AndOp>();
                if (aAndDbg && bAndDbg) {
                  llvm::errs() << "  aVal inputs:\n";
                  for (Value in : aAndDbg.getInputs()) {
                    llvm::errs() << "    ";
                    if (Operation *def = in.getDefiningOp())
                      def->print(llvm::errs());
                    else
                      llvm::errs() << "<blockarg>";
                    llvm::errs() << "\n";
                  }
                  llvm::errs() << "  bVal inputs:\n";
                  for (Value in : bAndDbg.getInputs()) {
                    llvm::errs() << "    ";
                    if (Operation *def = in.getDefiningOp())
                      def->print(llvm::errs());
                    else
                      llvm::errs() << "<blockarg>";
                    llvm::errs() << "\n";
                  }
                  llvm::errs() << "  equiv matrix:\n";
                  for (Value inA : aAndDbg.getInputs()) {
                    for (Value inB : bAndDbg.getInputs()) {
                      auto icmpA = inA.getDefiningOp<comb::ICmpOp>();
                      auto icmpB = inB.getDefiningOp<comb::ICmpOp>();
                      if (icmpA && icmpB) {
                        llvm::errs() << "    icmp attrs: twoState "
                                     << icmpA.getTwoState() << " vs "
                                     << icmpB.getTwoState() << " pred "
                                     << static_cast<int>(icmpA.getPredicate())
                                     << " vs "
                                     << static_cast<int>(icmpB.getPredicate())
                                     << "\n";
                      }
                      llvm::errs() << "    equiv=" << equiv(inA, inB, /*depth=*/0)
                                   << "  ";
                      if (Operation *defA = inA.getDefiningOp())
                        defA->print(llvm::errs());
                      else
                        llvm::errs() << "<blockarg>";
                      llvm::errs() << "  vs  ";
                      if (Operation *defB = inB.getDefiningOp())
                        defB->print(llvm::errs());
                      else
                        llvm::errs() << "<blockarg>";
                      llvm::errs() << "\n";
                    }
                  }
                }
              }
            }
          }

          if (!matches)
            continue;

          notA = candidate;
          break;
        }
        if (!notA)
          continue;

        Value oldNot = oldNotValue;
        if (!oldNot) {
          // Materialize `oldNot` from the spilled i64: extract bit0.
          OpBuilder b(andOp);
          Value oldNotPacked = frameLoad.getResult(0);
          oldNot = comb::ExtractOp::create(b, andOp.getLoc(), oldNotPacked,
                                           /*lowBit=*/0, /*width=*/1);
        }

        // Replace the `!after` operand with the spilled `!before`.
        andOp->replaceUsesOfWith(notA, oldNot);
        patched = true;
        if (debugEdgePatch)
          llvm::errs() << "[convert-to-arcs] edgepatch proc_id=" << debugProcId
                       << " patched block=" << &block << "\n";
        break;
      }

      (void)patched;
    }
  });

  // At this point the Arc pipeline must not contain any remaining LLHD probe
  // operations, otherwise the Arc-to-LLVM lowering will fail to legalize them.
  //
  // `llhd.prb` operations can still appear as type-conversion materializations
  // when SV interface lowering introduces opaque interface handles (e.g.
  // virtual interfaces stored as i64 values). Lower these probes to arcilator
  // runtime signal loads.
  SmallVector<llhd::PrbOp> leftoverProbes;
  getOperation().walk([&](llhd::PrbOp prb) { leftoverProbes.push_back(prb); });
  bool probeLoweringFailed = false;
  for (llhd::PrbOp prb : leftoverProbes) {
    ResolvedRuntimeSignal resolved;
    if (failed(resolveRuntimeSignal(prb.getSignal(), resolved)) ||
        (!resolved.sigIdAttr && !resolved.dynSigId)) {
      prb.emitOpError() << "unsupported probe source for runtime lowering";
      probeLoweringFailed = true;
      continue;
    }

    Type resultTy = prb.getType();
    int64_t resultWidth = hw::getBitWidth(resultTy);
    if (resultWidth <= 0 || resultWidth > 64) {
      prb.emitOpError() << "unsupported probed type for runtime lowering: "
                        << resultTy;
      probeLoweringFailed = true;
      continue;
    }

    uint64_t totalWidth = resolved.totalWidth;
    if (totalWidth == 0) {
      int64_t bw = hw::getBitWidth(resultTy);
      if (bw > 0)
        totalWidth = static_cast<uint64_t>(bw);
    }
    if (totalWidth == 0 || totalWidth > 64) {
      prb.emitOpError() << "unsupported runtime signal width for probe: "
                        << totalWidth;
      probeLoweringFailed = true;
      continue;
    }

    auto module = prb->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      prb.emitOpError() << "missing module for runtime hook";
      probeLoweringFailed = true;
      continue;
    }

    OpBuilder b(prb);
    Location loc = prb.getLoc();

    (void)getOrInsertFunc(
        module, "__arcilator_sig_load_u64",
        b.getFunctionType({b.getI32Type(), b.getI32Type()}, {b.getI64Type()}));

    uint32_t procId = 0xFFFFFFFFu;
    if (auto exec = prb->getParentOfType<arc::ExecuteOp>()) {
      if (auto attr = exec->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    } else if (auto proc = prb->getParentOfType<llhd::ProcessOp>()) {
      if (auto attr = proc->getAttrOfType<IntegerAttr>(kArcilatorProcIdAttr))
        procId = static_cast<uint32_t>(attr.getInt());
    }
    Value procIdVal = buildI32Constant(b, loc, procId);

    Value sigIdVal;
    if (resolved.sigIdAttr) {
      sigIdVal = buildI32Constant(b, loc, resolved.sigIdAttr.getInt());
    } else {
      Value dyn = resolved.dynSigId;
      auto dynTy = dyn_cast<IntegerType>(dyn.getType());
      if (!dynTy) {
        prb.emitOpError() << "unsupported dynamic signal id type: "
                          << dyn.getType();
        probeLoweringFailed = true;
        continue;
      }
      if (dynTy.getWidth() < 32)
        dyn = comb::createZExt(b, loc, dyn, 32);
      else if (dynTy.getWidth() > 32)
        dyn = comb::ExtractOp::create(b, loc, dyn, 0, 32);
      sigIdVal = dyn;
    }

    Value loaded =
        b.create<mlir::func::CallOp>(loc, "__arcilator_sig_load_u64",
                                     b.getI64Type(),
                                     ValueRange{sigIdVal, procIdVal})
            .getResult(0);

    Value bitsVal = loaded;
    uint64_t sliceOffset = resolved.baseOffset;
    if (resolved.dynamicOffset) {
      Value offsetVal = resolved.dynamicOffset;
      auto offsetTy = dyn_cast<IntegerType>(offsetVal.getType());
      if (!offsetTy) {
        prb.emitOpError() << "unsupported dynamic extract index type: "
                          << offsetVal.getType();
        probeLoweringFailed = true;
        continue;
      }
      if (offsetTy.getWidth() < 64)
        offsetVal = comb::createZExt(b, loc, offsetVal, 64);
      else if (offsetTy.getWidth() > 64)
        offsetVal = comb::ExtractOp::create(b, loc, offsetVal, 0, 64);

      if (sliceOffset != 0) {
        Value baseOff = buildI64Constant(b, loc, sliceOffset);
        offsetVal = comb::AddOp::create(b, loc, baseOff, offsetVal, true);
      }
      offsetVal = b.createOrFold<comb::AndOp>(
          loc, offsetVal, buildI64Constant(b, loc, 63));

      Value shifted = b.createOrFold<comb::ShrUOp>(loc, loaded, offsetVal);
      bitsVal = shifted;
      if (resultWidth != 64) {
        bitsVal = comb::ExtractOp::create(b, loc, b.getIntegerType(resultWidth),
                                          shifted, 0);
      }
    } else {
      if (sliceOffset + static_cast<uint64_t>(resultWidth) > 64) {
        prb.emitOpError()
            << "signal slice exceeds 64-bit runtime storage for probe";
        probeLoweringFailed = true;
        continue;
      }
      if (resultWidth != 64 || sliceOffset != 0) {
        bitsVal = comb::ExtractOp::create(b, loc, b.getIntegerType(resultWidth),
                                          loaded, sliceOffset);
      }
    }

    Value replacement = bitsVal;
    if (replacement.getType() != resultTy)
      replacement = b.createOrFold<hw::BitcastOp>(loc, resultTy, replacement);
    prb.replaceAllUsesWith(replacement);
    prb.erase();
  }
  if (probeLoweringFailed) {
    emitError(getOperation().getLoc())
        << "failed to lower LLHD probes after arc conversion";
    return signalPassFailure();
  }

  // Persist runtime-managed signal initial values on the Arc model op so they
  // survive later canonicalization/DCE and can be exported via state.json.
  if (!runtimeSigInits.empty()) {
    OpBuilder b(&getContext());
    SmallVector<std::pair<uint64_t, ArcilatorRuntimeSigInit>> items;
    items.reserve(runtimeSigInits.size());
    for (auto &it : runtimeSigInits)
      items.push_back({it.first, it.second});
    llvm::sort(items, [](auto &a, auto &b) { return a.first < b.first; });

    SmallVector<Attribute> entries;
    entries.reserve(items.size());
    for (auto &it : items) {
      uint64_t sigId = it.first;
      const auto &rec = it.second;
      NamedAttrList dict;
      dict.append("sigId", b.getI64IntegerAttr(sigId));
      dict.append("initU64", b.getI64IntegerAttr(rec.initU64));
      dict.append("totalWidth", b.getI64IntegerAttr(rec.totalWidth));
      entries.push_back(DictionaryAttr::get(&getContext(), dict));
    }
    auto arr = b.getArrayAttr(entries);
    // ConvertToArcs runs before `arc::LowerState` creates `arc.model` ops from
    // HW modules. Stash the summary attribute on any existing models and on
    // the originating HW modules so later passes can propagate it onto the
    // final `arc.model` (consumed by ModelInfo/state.json export).
    getOperation().walk([&](arc::ModelOp modelOp) {
      modelOp->setAttr(kArcilatorSigInitsAttr, arr);
    });
    getOperation().walk([&](hw::HWModuleOp moduleOp) {
      moduleOp->setAttr(kArcilatorSigInitsAttr, arr);
    });
  }

}
