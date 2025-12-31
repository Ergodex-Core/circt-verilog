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
  FailureOr<LLVM::LLVMFuncOp> getOrCreateExit(OpBuilder &builder);

  LogicalResult lowerPrint(sim::PrintFormattedProcOp op);
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

FailureOr<LLVM::LLVMFuncOp> LowerSimConsolePass::getOrCreateExit(
    OpBuilder &builder) {
  auto module = getOperation();
  auto i32Ty = builder.getI32Type();
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  return LLVM::lookupOrCreateFn(builder, module, "exit", i32Ty, voidTy,
                                /*isVarArg=*/false);
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
	                     Value strPtr = strOp.getValue();
	                     LLVM::CallOp::create(builder, loc, printfFn.value(),
	                                          ValueRange{fmtPtr, strPtr});
	                     return success();
	                   })
	                   .Case<sim::FormatDecOp>([&](auto decOp) {
	                     bool truncated = false;
	                     bool isSigned = decOp.getIsSigned();
                     Value v64 = castToI64(decOp.getValue(), isSigned, builder,
                                           loc, truncated);
                     if (!v64) {
                       decOp.emitOpError(
                           "unsupported value type for decimal formatting");
                       return failure();
                     }
                     StringRef fmt = isSigned ? "%lld" : "%llu";
                     auto fmtGlobal = getOrCreateFormatString(loc, fmt);
                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
                     LLVM::CallOp::create(builder, loc, printfFn.value(),
                                          ValueRange{fmtPtr, v64});
                     return success();
                   })
                   .Case<sim::FormatHexOp>([&](auto hexOp) {
                     bool truncated = false;
                     Value v64 = castToI64(hexOp.getValue(), /*isSigned=*/false,
                                           builder, loc, truncated);
                     if (!v64) {
                       hexOp.emitOpError(
                           "unsupported value type for hex formatting");
                       return failure();
                     }
                     auto intTy = cast<IntegerType>(hexOp.getValue().getType());
                     unsigned digits = (intTy.getWidth() + 3) / 4;
                     if (digits == 0)
                       return success();
                     SmallString<16> fmt;
                     fmt.append("%0");
                     fmt.append(Twine(digits).str());
                     fmt.append("llx");
                     auto fmtGlobal = getOrCreateFormatString(loc, fmt);
                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);
                     LLVM::CallOp::create(builder, loc, printfFn.value(),
                                          ValueRange{fmtPtr, v64});
                     return success();
                   })
                   .Case<sim::FormatBinOp>([&](auto binOp) {
                     bool truncated = false;
                     Value v64 = castToI64(binOp.getValue(), /*isSigned=*/false,
                                           builder, loc, truncated);
                     if (!v64) {
                       binOp.emitOpError(
                           "unsupported value type for binary formatting");
                       return failure();
                     }

                     auto intTy = cast<IntegerType>(binOp.getValue().getType());
                     unsigned width = intTy.getWidth();
                     if (width == 0)
                       return success();
                     if (width > 64)
                       width = 64;

                     auto i64Ty = builder.getI64Type();
                     auto fmtGlobal = getOrCreateFormatString(loc, "%c");
                     Value fmtPtr = getStringGlobalPtr(loc, fmtGlobal, builder);

                     Value zero = LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
                     Value one = LLVM::ConstantOp::create(builder, loc, i64Ty, 1);
                     Value ch0 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                                          static_cast<int64_t>('0'));
                     Value ch1 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                                          static_cast<int64_t>('1'));

                     // Emit one printf("%c") per bit. This is not optimized for
                     // speed, but keeps the lowering simple and deterministic.
                     for (int bit = static_cast<int>(width) - 1; bit >= 0; --bit) {
                       Value shiftAmt =
                           LLVM::ConstantOp::create(builder, loc, i64Ty, bit);
                       Value shifted =
                           LLVM::LShrOp::create(builder, loc, i64Ty, v64, shiftAmt);
                       Value masked =
                           LLVM::AndOp::create(builder, loc, i64Ty, shifted, one);
                       Value isOne = LLVM::ICmpOp::create(
                           builder, loc, LLVM::ICmpPredicate::ne, masked, zero);
                       Value ch =
                           LLVM::SelectOp::create(builder, loc, isOne, ch1, ch0);
                       LLVM::CallOp::create(builder, loc, printfFn.value(),
                                            ValueRange{fmtPtr, ch});
                     }
                     return success();
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
	               sim::FormatStrOp,
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
