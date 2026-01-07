//===- LowerSimConsole.cpp - Lower sim console ops to LLVM ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers a subset of Sim dialect operations used for interactive
// debugging (`$display`, `$error`, `$fatal`, `$finish`) to LLVM calls (`printf`,
// `exit`) after the main Arc-to-LLVM lowering has finished.
//
// Specifically, it lowers:
//   - sim.proc.print + sim.fmt.* DAGs
//   - sim.terminate
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "sim-lower-console"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_LOWERSIMCONSOLE
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct LowerSimConsolePass
    : public sim::impl::LowerSimConsoleBase<LowerSimConsolePass> {
  void runOnOperation() override;

private:
  LLVM::GlobalOp getOrCreateStringGlobal(Location loc, StringRef bytes,
                                         StringRef prefix);
  Value getStringGlobalPtr(Location loc, LLVM::GlobalOp global,
                           OpBuilder &builder);

  FailureOr<LLVM::LLVMFuncOp> getOrCreatePrintf(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreatePrintInt(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreatePrintFVInt(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreatePrintTime(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateSetTimeformat(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateExit(OpBuilder &builder);

  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderNew(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendBytes(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendCStr(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendInt(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendFVInt(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendTime(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendChar(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderAppendReal(OpBuilder &builder);
  FailureOr<LLVM::LLVMFuncOp> getOrCreateStrBuilderFinish(OpBuilder &builder);

  LogicalResult lowerFormatToString(sim::FormatToStringOp op);
  LogicalResult lowerPrint(sim::PrintFormattedProcOp op);
  LogicalResult lowerTimeformat(sim::TimeFormatProcOp op);
  LogicalResult lowerTerminate(sim::TerminateOp op);

  LLVM::GlobalOp getOrCreateFormatString(Location loc, StringRef fmt);

  llvm::StringMap<LLVM::GlobalOp> stringGlobals;
  unsigned nextStringId = 0;
};
} // namespace

LLVM::GlobalOp LowerSimConsolePass::getOrCreateStringGlobal(Location loc,
                                                            StringRef bytes,
                                                            StringRef prefix) {
  auto module = getOperation();
  auto it = stringGlobals.find(bytes);
  if (it != stringGlobals.end())
    return it->second;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  SmallString<32> name;
  name.append(prefix);
  name.append("_");
  name.append(Twine(nextStringId++).str());

  // Use an i8 array global.
  auto i8Ty = builder.getI8Type();
  auto globalType = LLVM::LLVMArrayType::get(i8Ty, bytes.size());
  auto global = LLVM::GlobalOp::create(
      builder, loc, globalType, /*isConstant=*/true, LLVM::Linkage::Internal,
      name, builder.getStringAttr(bytes), /*alignment=*/0);

  stringGlobals.insert({bytes, global});
  return global;
}

Value LowerSimConsolePass::getStringGlobalPtr(Location loc, LLVM::GlobalOp global,
                                              OpBuilder &builder) {
  return LLVM::AddressOfOp::create(builder, loc, global);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreatePrintf(
    OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  return LLVM::lookupOrCreateFn(builder, module, "printf", ptrTy, i32Ty,
                                /*isVarArg=*/true);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreatePrintInt(
    OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_print_int", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreatePrintFVInt(
    OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_print_fvint", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreatePrintTime(
    OpBuilder &builder) {
  auto module = getOperation();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{i64Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_print_time", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreateSetTimeformat(
    OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{i32Ty, i32Ty, ptrTy, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_set_timeformat",
                                params, voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreateExit(
    OpBuilder &builder) {
  auto module = getOperation();
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  return LLVM::lookupOrCreateFn(builder, module, "exit", i32Ty, voidTy,
                                /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreateStrBuilderNew(
    OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  SmallVector<Type> params;
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_strbuilder_new",
                                params, ptrTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendBytes(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, ptrTy, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_bytes", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendCStr(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, ptrTy};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_cstr", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendInt(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_int", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendFVInt(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_fvint", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendTime(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, i64Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_time", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendChar(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_char", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderAppendReal(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  auto f64Ty = builder.getF64Type();
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  SmallVector<Type> params{ptrTy, f64Ty, i32Ty};
  return LLVM::lookupOrCreateFn(builder, module,
                                "circt_sv_strbuilder_append_real", params,
                                voidTy, /*isVarArg=*/false);
}

FailureOr<LLVM::LLVMFuncOp>
LowerSimConsolePass::getOrCreateStrBuilderFinish(OpBuilder &builder) {
  auto module = getOperation();
  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
  SmallVector<Type> params{ptrTy};
  return LLVM::lookupOrCreateFn(builder, module, "circt_sv_strbuilder_finish",
                                params, ptrTy, /*isVarArg=*/false);
}

LLVM::GlobalOp LowerSimConsolePass::getOrCreateFormatString(Location loc,
                                                            StringRef fmt) {
  SmallVector<char> bytes(fmt.begin(), fmt.end());
  bytes.push_back(0);
  return getOrCreateStringGlobal(loc, StringRef(bytes.data(), bytes.size()),
                                 "_sim_fmt");
}

static Value castToI64(Value value, bool isSigned, OpBuilder &builder,
                       Location loc, bool &truncated) {
  truncated = false;
  auto intTy = dyn_cast<IntegerType>(value.getType());
  if (!intTy)
    return {};

  auto i64Ty = builder.getI64Type();
  unsigned width = intTy.getWidth();
  if (width == 0) {
    return LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
  }

  Value v = value;
  if (width > 64) {
    v = LLVM::TruncOp::create(builder, loc, builder.getI64Type(), v);
    truncated = true;
    width = 64;
  }

  if (width < 64) {
    if (isSigned)
      v = LLVM::SExtOp::create(builder, loc, i64Ty, v);
    else
      v = LLVM::ZExtOp::create(builder, loc, i64Ty, v);
  }
  return v;
}

static Value castToF64(Value value, OpBuilder &builder, Location loc,
                       bool &truncated) {
  truncated = false;
  auto floatTy = dyn_cast<FloatType>(value.getType());
  if (!floatTy)
    return {};

  auto f64Ty = builder.getF64Type();
  if (value.getType() == f64Ty)
    return value;

  if (floatTy.getWidth() < 64)
    return LLVM::FPExtOp::create(builder, loc, f64Ty, value);

  if (floatTy.getWidth() > 64) {
    truncated = true;
    return LLVM::FPTruncOp::create(builder, loc, f64Ty, value);
  }

  return {};
}

LogicalResult LowerSimConsolePass::lowerFormatToString(sim::FormatToStringOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto ptrTy = LLVM::LLVMPointerType::get(&getContext());

  auto newFn = getOrCreateStrBuilderNew(builder);
  auto appendBytesFn = getOrCreateStrBuilderAppendBytes(builder);
  auto appendCStrFn = getOrCreateStrBuilderAppendCStr(builder);
  auto appendIntFn = getOrCreateStrBuilderAppendInt(builder);
  auto appendFVIntFn = getOrCreateStrBuilderAppendFVInt(builder);
  auto appendTimeFn = getOrCreateStrBuilderAppendTime(builder);
  auto appendCharFn = getOrCreateStrBuilderAppendChar(builder);
  auto appendRealFn = getOrCreateStrBuilderAppendReal(builder);
  auto finishFn = getOrCreateStrBuilderFinish(builder);

  if (failed(newFn) || failed(appendBytesFn) || failed(appendCStrFn) ||
      failed(appendIntFn) || failed(appendFVIntFn) || failed(appendTimeFn) ||
      failed(appendCharFn) || failed(appendRealFn) || failed(finishFn)) {
    op.emitOpError("failed to lookup or create runtime string builder hooks");
    return failure();
  }

  Value builderHandle =
      LLVM::CallOp::create(builder, loc, newFn.value(), ValueRange{})
          .getResult();

  auto appendBytes = [&](StringRef bytes) -> LogicalResult {
    if (bytes.empty())
      return success();
    auto global = getOrCreateStringGlobal(loc, bytes, "_sim_bytes");
    Value ptr = getStringGlobalPtr(loc, global, builder);
    Value lenVal = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                           static_cast<int32_t>(bytes.size()));
    LLVM::CallOp::create(builder, loc, appendBytesFn.value(),
                         ValueRange{builderHandle, ptr, lenVal});
    return success();
  };

  auto appendCStrValue = [&](Value strVal) -> LogicalResult {
    Value strPtr;
    if (auto cst = strVal.getDefiningOp<circt::sv::ConstantStrOp>()) {
      SmallVector<char> bytes(cst.getStr().begin(), cst.getStr().end());
      bytes.push_back(0);
      auto global = getOrCreateStringGlobal(loc,
                                            StringRef(bytes.data(), bytes.size()),
                                            "_sim_str");
      strPtr = getStringGlobalPtr(loc, global, builder);
    } else {
      while (auto castOp =
                 strVal.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() != 1)
          break;
        strVal = castOp.getInputs().front();
      }

      if (isa<LLVM::LLVMPointerType>(strVal.getType())) {
        strPtr = strVal;
      } else {
        strPtr =
            builder.create<UnrealizedConversionCastOp>(loc, ptrTy, strVal)
                .getResult(0);
      }
    }

    LLVM::CallOp::create(builder, loc, appendCStrFn.value(),
                         ValueRange{builderHandle, strPtr});
    return success();
  };

  auto appendInt = [&](Value value, int32_t base, int32_t minWidth,
                       int32_t flags) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(value.getType());
    if (!intTy)
      return failure();
    int32_t bitWidth = static_cast<int32_t>(intTy.getWidth());
    if (bitWidth == 0) {
      if (base == 10)
        return appendBytes("0");
      return success();
    }

    Value one = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), 1);
    Value slot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, value.getType(), one);
    LLVM::StoreOp::create(builder, loc, value, slot);

    Value bitWidthVal = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                                 bitWidth);
    Value baseVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), base);
    Value widthVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), minWidth);
    Value flagsVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), flags);

    LLVM::CallOp::create(builder, loc, appendIntFn.value(),
                         ValueRange{builderHandle, slot, bitWidthVal, baseVal,
                                    widthVal, flagsVal});
    return success();
  };

  auto appendFVInt = [&](Value value, Value unknown, int32_t base,
                         int32_t minWidth, int32_t flags) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(value.getType());
    auto unkTy = dyn_cast<IntegerType>(unknown.getType());
    if (!intTy || !unkTy || intTy.getWidth() != unkTy.getWidth())
      return failure();
    int32_t bitWidth = static_cast<int32_t>(intTy.getWidth());
    if (bitWidth == 0) {
      if (base == 10)
        return appendBytes("0");
      return success();
    }

    Value one = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), 1);
    Value valueSlot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, value.getType(), one);
    LLVM::StoreOp::create(builder, loc, value, valueSlot);
    Value unknownSlot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, unknown.getType(), one);
    LLVM::StoreOp::create(builder, loc, unknown, unknownSlot);

    Value bitWidthVal = LLVM::ConstantOp::create(
        builder, loc, builder.getI32Type(), bitWidth);
    Value baseVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), base);
    Value widthVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), minWidth);
    Value flagsVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), flags);

    LLVM::CallOp::create(builder, loc, appendFVIntFn.value(),
                         ValueRange{builderHandle, valueSlot, unknownSlot,
                                    bitWidthVal, baseVal, widthVal, flagsVal});
    return success();
  };

  auto appendTime = [&](Value timeFs, int32_t widthOverride) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(timeFs.getType());
    if (!intTy)
      return failure();
    Value t = timeFs;
    if (intTy.getWidth() != 64) {
      if (intTy.getWidth() < 64)
        t = LLVM::ZExtOp::create(builder, loc, builder.getI64Type(), t);
      else
        t = LLVM::TruncOp::create(builder, loc, builder.getI64Type(), t);
    }
    Value widthVal = LLVM::ConstantOp::create(
        builder, loc, builder.getI32Type(), widthOverride);
    LLVM::CallOp::create(builder, loc, appendTimeFn.value(),
                         ValueRange{builderHandle, t, widthVal});
    return success();
  };

  auto appendChar = [&](Value value) -> LogicalResult {
    bool truncated = false;
    Value v64 = castToI64(value, /*isSigned=*/false, builder, loc, truncated);
    if (!v64)
      return failure();
    Value v32 = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), v64);
    LLVM::CallOp::create(builder, loc, appendCharFn.value(),
                         ValueRange{builderHandle, v32});
    return success();
  };

  auto appendReal = [&](Value value, StringRef mode) -> LogicalResult {
    bool truncated = false;
    Value v64 = castToF64(value, builder, loc, truncated);
    if (!v64)
      return failure();

    int32_t kind = 0;
    if (mode == "exponential")
      kind = 1;
    else if (mode == "general")
      kind = 2;

    Value kindVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), kind);
    LLVM::CallOp::create(builder, loc, appendRealFn.value(),
                         ValueRange{builderHandle, v64, kindVal});
    return success();
  };

  SmallVector<Value> fragments;
  if (auto concat = op.getInput().getDefiningOp<sim::FormatStringConcatOp>()) {
    if (failed(concat.getFlattenedInputs(fragments))) {
      op.emitOpError("cyclic format string cannot be lowered");
      return failure();
    }
  } else {
    fragments.push_back(op.getInput());
  }

  for (Value fragment : fragments) {
    Operation *defOp = fragment.getDefiningOp();
    if (!defOp) {
      op.emitOpError("unsupported block argument format fragment");
      return failure();
    }

    auto res = TypeSwitch<Operation *, LogicalResult>(defOp)
                   .Case<sim::FormatLitOp>([&](auto litOp) {
                     return appendBytes(litOp.getLiteral());
                   })
                   .Case<sim::FormatStrOp>(
                       [&](auto strOp) { return appendCStrValue(strOp.getValue()); })
                   .Case<sim::FormatDecOp>([&](auto decOp) {
                     auto intTy =
                         cast<IntegerType>(decOp.getValue().getType());
                     unsigned bits = intTy.getWidth();
                     bool isSigned = decOp.getIsSigned();
                     int32_t width = static_cast<int32_t>(
                         sim::FormatDecOp::getDecimalWidth(bits, isSigned));
                     int32_t flags = isSigned ? (1 << 3) : 0;
                     return appendInt(decOp.getValue(), /*base=*/10, width,
                                      flags);
                   })
                   .Case<sim::FormatHexOp>([&](auto hexOp) {
                     auto intTy =
                         cast<IntegerType>(hexOp.getValue().getType());
                     int32_t width =
                         static_cast<int32_t>((intTy.getWidth() + 3) / 4);
                     if (width == 0)
                       return success();
                     return appendInt(hexOp.getValue(), /*base=*/16, width,
                                      /*flags=*/1 << 2);
                   })
                   .Case<sim::FormatBinOp>([&](auto binOp) {
                     auto intTy =
                         cast<IntegerType>(binOp.getValue().getType());
                     int32_t width = static_cast<int32_t>(intTy.getWidth());
                     if (width == 0)
                       return success();
                     return appendInt(binOp.getValue(), /*base=*/2, width,
                                      /*flags=*/1 << 2);
                   })
                   .Case<sim::FormatIntOp>([&](auto intOp) {
                     return appendInt(intOp.getValue(), intOp.getBase(),
                                      intOp.getWidth(), intOp.getFlags());
                   })
                   .Case<sim::FormatFVIntOp>([&](auto fvOp) {
                     return appendFVInt(fvOp.getValue(), fvOp.getUnknown(),
                                        fvOp.getBase(), fvOp.getWidth(),
                                        fvOp.getFlags());
                   })
                   .Case<sim::FormatTimeOp>([&](auto timeOp) {
                     int32_t width = -1;
                     if (auto w = timeOp.getWidth())
                       width = static_cast<int32_t>(*w);
                     return appendTime(timeOp.getValue(), width);
                   })
                   .Case<sim::FormatCharOp>(
                       [&](auto charOp) { return appendChar(charOp.getValue()); })
                   .Case<sim::FormatRealOp>([&](auto realOp) {
                     return appendReal(realOp.getValue(), realOp.getFormat());
                   })
                   .Default([&](Operation *) {
                     return appendBytes("<unsupported sim.fmt.* fragment>");
                   });
    if (failed(res))
      return failure();
  }

  Value outPtr =
      LLVM::CallOp::create(builder, loc, finishFn.value(),
                           ValueRange{builderHandle})
          .getResult();

  Value result = outPtr;
  if (op.getResult().getType() != ptrTy) {
    result = builder.create<UnrealizedConversionCastOp>(
                        loc, op.getResult().getType(), outPtr)
                 .getResult(0);
  }

  op.replaceAllUsesWith(result);
  op.erase();
  return success();
}

LogicalResult LowerSimConsolePass::lowerPrint(sim::PrintFormattedProcOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto printfFn = getOrCreatePrintf(builder);
  if (failed(printfFn)) {
    op.emitOpError("failed to lookup or create `printf`");
    return failure();
  }

  auto printCString = [&](StringRef str) -> LogicalResult {
    SmallVector<char> bytes(str.begin(), str.end());
    bytes.push_back(0);
    auto literalGlobal =
        getOrCreateStringGlobal(loc, StringRef(bytes.data(), bytes.size()),
                                "_sim_str");

    auto fmtGlobal = getOrCreateFormatString(loc, "%s");
    Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
    Value litPtr = getStringGlobalPtr(loc, literalGlobal, builder);
    LLVM::CallOp::create(builder, loc, printfFn.value(), ValueRange{fmtPtr, litPtr});
    return success();
  };

  auto printIntFn = getOrCreatePrintInt(builder);
  if (failed(printIntFn)) {
    op.emitOpError("failed to lookup or create `circt_sv_print_int`");
    return failure();
  }

  auto printFVIntFn = getOrCreatePrintFVInt(builder);
  if (failed(printFVIntFn)) {
    op.emitOpError("failed to lookup or create `circt_sv_print_fvint`");
    return failure();
  }

  auto printTimeFn = getOrCreatePrintTime(builder);
  if (failed(printTimeFn)) {
    op.emitOpError("failed to lookup or create `circt_sv_print_time`");
    return failure();
  }

  auto printInt = [&](Value value, int32_t base, int32_t minWidth,
                      int32_t flags) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(value.getType());
    if (!intTy)
      return failure();
    int32_t bitWidth = static_cast<int32_t>(intTy.getWidth());
    if (bitWidth == 0) {
      if (base == 10)
        return printCString("0");
      return success();
    }

    auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
    Value one = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), 1);
    Value slot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, value.getType(), one);
    LLVM::StoreOp::create(builder, loc, value, slot);

    Value bitWidthVal = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                                 bitWidth);
    Value baseVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), base);
    Value widthVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), minWidth);
    Value flagsVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), flags);

    LLVM::CallOp::create(builder, loc, printIntFn.value(),
                         ValueRange{slot, bitWidthVal, baseVal, widthVal,
                                    flagsVal});
    return success();
  };

  auto printFVInt = [&](Value value, Value unknown, int32_t base,
                        int32_t minWidth, int32_t flags) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(value.getType());
    auto unkTy = dyn_cast<IntegerType>(unknown.getType());
    if (!intTy || !unkTy || intTy.getWidth() != unkTy.getWidth())
      return failure();
    int32_t bitWidth = static_cast<int32_t>(intTy.getWidth());
    if (bitWidth == 0) {
      if (base == 10)
        return printCString("0");
      return success();
    }

    auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
    Value one =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), 1);

    Value valueSlot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, value.getType(), one);
    LLVM::StoreOp::create(builder, loc, value, valueSlot);

    Value unknownSlot =
        LLVM::AllocaOp::create(builder, loc, ptrTy, unknown.getType(), one);
    LLVM::StoreOp::create(builder, loc, unknown, unknownSlot);

    Value bitWidthVal = LLVM::ConstantOp::create(
        builder, loc, builder.getI32Type(), bitWidth);
    Value baseVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), base);
    Value widthVal = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                              minWidth);
    Value flagsVal =
        LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), flags);

    LLVM::CallOp::create(builder, loc, printFVIntFn.value(),
                         ValueRange{valueSlot, unknownSlot, bitWidthVal,
                                    baseVal, widthVal, flagsVal});
    return success();
  };

  auto printTime = [&](Value timeFs, int32_t widthOverride) -> LogicalResult {
    auto intTy = dyn_cast<IntegerType>(timeFs.getType());
    if (!intTy)
      return failure();
    Value t = timeFs;
    if (intTy.getWidth() != 64) {
      // Time is treated as an unsigned femtoseconds count.
      if (intTy.getWidth() < 64)
        t = LLVM::ZExtOp::create(builder, loc, builder.getI64Type(), t);
      else
        t = LLVM::TruncOp::create(builder, loc, builder.getI64Type(), t);
    }

    Value widthVal = LLVM::ConstantOp::create(
        builder, loc, builder.getI32Type(), widthOverride);
    LLVM::CallOp::create(builder, loc, printTimeFn.value(),
                         ValueRange{t, widthVal});
    return success();
  };

  // Flatten the format string into primitive fragments.
  SmallVector<Value> fragments;
  if (auto concat = op.getInput().getDefiningOp<sim::FormatStringConcatOp>()) {
    if (failed(concat.getFlattenedInputs(fragments))) {
      op.emitOpError("cyclic format string cannot be lowered");
      return failure();
    }
  } else {
    fragments.push_back(op.getInput());
  }

  for (Value fragment : fragments) {
    Operation *defOp = fragment.getDefiningOp();
    if (!defOp) {
      op.emitOpError("unsupported block argument format fragment");
      return failure();
    }

    auto res = TypeSwitch<Operation *, LogicalResult>(defOp)
	                   .Case<sim::FormatLitOp>([&](auto litOp) {
	                     return printCString(litOp.getLiteral());
	                   })
	                   .Case<sim::FormatStrOp>([&](auto strOp) {
	                     auto fmtGlobal = getOrCreateFormatString(loc, "%s");
	                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
	                     Value strVal = strOp.getValue();
	                     Value strPtr;

	                     if (auto cst =
	                             strVal.getDefiningOp<circt::sv::ConstantStrOp>()) {
	                       SmallVector<char> bytes(cst.getStr().begin(),
	                                               cst.getStr().end());
	                       bytes.push_back(0);
	                       auto global = getOrCreateStringGlobal(
	                           loc, StringRef(bytes.data(), bytes.size()), "_sim_str");
	                       strPtr = getStringGlobalPtr(loc, global, builder);
	                     } else {
	                       // Peel single-input conversion casts; `sim.fmt.str` may
	                       // survive Arc-to-LLVM as a legal op and carry type casts
	                       // from `!hw.string` to `!llvm.ptr`.
	                       while (auto castOp =
	                                  strVal.getDefiningOp<UnrealizedConversionCastOp>()) {
	                         if (castOp.getInputs().size() != 1)
	                           break;
	                         strVal = castOp.getInputs().front();
	                       }

	                       if (isa<LLVM::LLVMPointerType>(strVal.getType())) {
	                         strPtr = strVal;
	                       } else {
	                         auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
	                         strPtr =
	                             builder
	                                 .create<UnrealizedConversionCastOp>(loc, ptrTy,
	                                                                     strVal)
	                                 .getResult(0);
	                       }
	                     }
	                     LLVM::CallOp::create(builder, loc, printfFn.value(),
	                                          ValueRange{fmtPtr, strPtr});
	                     return success();
	                   })
	                   .Case<sim::FormatDecOp>([&](auto decOp) {
	                     // Implement `sim.fmt.dec` via the runtime printer. Use the
	                     // Sim op's documented minimum field width.
	                     auto intTy = cast<IntegerType>(decOp.getValue().getType());
	                     unsigned bits = intTy.getWidth();
	                     bool isSigned = decOp.getIsSigned();
	                     int32_t width = static_cast<int32_t>(
	                         sim::FormatDecOp::getDecimalWidth(bits, isSigned));
	                     int32_t flags = isSigned ? (1 << 3) : 0;
	                     return printInt(decOp.getValue(), /*base=*/10, width,
	                                     flags);
	                   })
	                   .Case<sim::FormatHexOp>([&](auto hexOp) {
	                     auto intTy = cast<IntegerType>(hexOp.getValue().getType());
	                     int32_t width =
	                         static_cast<int32_t>((intTy.getWidth() + 3) / 4);
	                     if (width == 0)
	                       return success();
	                     return printInt(hexOp.getValue(), /*base=*/16, width,
	                                     /*flags=*/1 << 2);
	                   })
	                   .Case<sim::FormatBinOp>([&](auto binOp) {
	                     auto intTy = cast<IntegerType>(binOp.getValue().getType());
	                     int32_t width = static_cast<int32_t>(intTy.getWidth());
	                     if (width == 0)
	                       return success();
	                     return printInt(binOp.getValue(), /*base=*/2, width,
	                                     /*flags=*/1 << 2);
	                   })
	                   .Case<sim::FormatIntOp>([&](auto intOp) {
	                     return printInt(intOp.getValue(), intOp.getBase(),
	                                     intOp.getWidth(), intOp.getFlags());
	                   })
	                   .Case<sim::FormatFVIntOp>([&](auto fvOp) {
	                     return printFVInt(fvOp.getValue(), fvOp.getUnknown(),
	                                       fvOp.getBase(), fvOp.getWidth(),
	                                       fvOp.getFlags());
	                   })
	                   .Case<sim::FormatTimeOp>([&](auto timeOp) {
	                     int32_t width = -1;
	                     if (auto w = timeOp.getWidth())
	                       width = static_cast<int32_t>(*w);
	                     return printTime(timeOp.getValue(), width);
	                   })
                   .Case<sim::FormatCharOp>([&](auto charOp) {
                     // Use printf("%c", value) to avoid needing `putchar`.
                     bool truncated = false;
                     Value v64 = castToI64(charOp.getValue(), /*isSigned=*/false,
                                           builder, loc, truncated);
                     if (!v64) {
                       charOp.emitOpError(
                           "unsupported value type for char formatting");
                       return failure();
                     }
                     auto fmtGlobal = getOrCreateFormatString(loc, "%c");
                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
                     LLVM::CallOp::create(builder, loc, printfFn.value(),
                                          ValueRange{fmtPtr, v64});
                     return success();
                   })
                   .Case<sim::FormatRealOp>([&](auto realOp) {
                     bool truncated = false;
                     Value v64 =
                         castToF64(realOp.getValue(), builder, loc, truncated);
                     if (!v64) {
                       realOp.emitOpError(
                           "unsupported value type for real formatting");
                       return failure();
                     }

                     StringRef fmt = "%f";
                     auto mode = realOp.getFormat();
                     if (mode == "exponential")
                       fmt = "%e";
                     else if (mode == "general")
                       fmt = "%g";

                     auto fmtGlobal = getOrCreateFormatString(loc, fmt);
                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
                     LLVM::CallOp::create(builder, loc, printfFn.value(),
                                          ValueRange{fmtPtr, v64});
                     return success();
                   })
                   .Default([&](Operation *) {
                     return printCString("<unsupported sim.fmt.* fragment>");
                   });
    if (failed(res))
      return failure();
  }

  op.erase();
  return success();
}

LogicalResult LowerSimConsolePass::lowerTimeformat(sim::TimeFormatProcOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto setFn = getOrCreateSetTimeformat(builder);
  if (failed(setFn)) {
    op.emitOpError("failed to lookup or create `circt_sv_set_timeformat`");
    return failure();
  }

  int32_t unit = static_cast<int32_t>(op.getUnit());
  int32_t precision = static_cast<int32_t>(op.getPrecision());
  int32_t minWidth = static_cast<int32_t>(op.getMinWidth());

  Value unitVal =
      LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), unit);
  Value precisionVal =
      LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), precision);
  Value minWidthVal =
      LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), minWidth);

  SmallVector<char> bytes(op.getSuffix().begin(), op.getSuffix().end());
  bytes.push_back(0);
  auto suffixGlobal =
      getOrCreateStringGlobal(loc, StringRef(bytes.data(), bytes.size()),
                              "_sim_timefmt");
  Value suffixPtr = getStringGlobalPtr(loc, suffixGlobal, builder);

  LLVM::CallOp::create(builder, loc, setFn.value(),
                       ValueRange{unitVal, precisionVal, suffixPtr, minWidthVal});

  op.erase();
  return success();
}

LogicalResult LowerSimConsolePass::lowerTerminate(sim::TerminateOp op) {
  OpBuilder builder(op);
  auto loc = op.getLoc();

  auto exitFn = getOrCreateExit(builder);
  if (failed(exitFn)) {
    op.emitOpError("failed to lookup or create `exit`");
    return failure();
  }

  int32_t code = op.getSuccess() ? 0 : 1;
  Value codeVal = LLVM::ConstantOp::create(builder, loc, builder.getI32Type(), code);
  LLVM::CallOp::create(builder, loc, exitFn.value(), ValueRange{codeVal});
  op.erase();
  return success();
}

void LowerSimConsolePass::runOnOperation() {
  auto module = getOperation();

  SmallVector<sim::TimeFormatProcOp> timeformats;
  module.walk([&](sim::TimeFormatProcOp op) { timeformats.push_back(op); });
  for (auto tfOp : llvm::make_early_inc_range(timeformats))
    if (failed(lowerTimeformat(tfOp)))
      return signalPassFailure();

  SmallVector<sim::FormatToStringOp> toStrings;
  module.walk([&](sim::FormatToStringOp op) { toStrings.push_back(op); });
  for (auto toStrOp : llvm::make_early_inc_range(toStrings))
    if (failed(lowerFormatToString(toStrOp)))
      return signalPassFailure();

  // Lower prints first, since they may be used to report errors before exit.
  SmallVector<sim::PrintFormattedProcOp> prints;
  module.walk([&](sim::PrintFormattedProcOp op) { prints.push_back(op); });
  for (auto printOp : llvm::make_early_inc_range(prints))
    if (failed(lowerPrint(printOp)))
      return signalPassFailure();

  SmallVector<sim::TerminateOp> terminates;
  module.walk([&](sim::TerminateOp op) { terminates.push_back(op); });
  for (auto termOp : llvm::make_early_inc_range(terminates))
    if (failed(lowerTerminate(termOp)))
      return signalPassFailure();

  // Erase dead formatting ops left behind by lowering.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
		    module.walk([&](Operation *op) {
		      if (!isa<sim::FormatLitOp, sim::FormatHexOp, sim::FormatBinOp,
		               sim::FormatDecOp, sim::FormatRealOp, sim::FormatCharOp,
		               sim::FormatIntOp, sim::FormatFVIntOp, sim::FormatTimeOp,
		               sim::FormatStrOp, sim::FormatToStringOp,
		               sim::FormatStringConcatOp>(op))
		        return;
	      if (op->getNumResults() != 1)
	        return;
	      if (op->getResult(0).use_empty())
        toErase.push_back(op);
    });
    if (!toErase.empty()) {
      changed = true;
      for (Operation *op : toErase)
        op->erase();
    }
  }

  // If any sim ops remain, fail with a clear diagnostic so we don't crash in
  // LLVM translation.
  Operation *firstSimOp = nullptr;
  module.walk([&](Operation *op) {
    if (auto *dialect = op->getDialect();
        dialect && dialect->getNamespace() == sim::SimDialect::getDialectNamespace()) {
      firstSimOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (firstSimOp) {
    firstSimOp->emitError(
        "unlowered Sim dialect operation; arcilator requires all `sim.*` ops "
        "to be lowered before LLVM translation");
    return signalPassFailure();
  }
}
