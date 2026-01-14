//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "moore-inline-calls"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_INLINECALLS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {
/// Inliner interface that allows calls in SSACFG regions nested within
/// `moore.procedure` ops to be inlined.
struct FunctionInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    auto funcOp = dyn_cast<func::FuncOp>(callable);
    if (!funcOp || funcOp.isExternal())
      return false;

    if (!mayHaveSSADominance(*call->getParentRegion()))
      return false;
    return call->getParentOfType<ProcedureOp>() != nullptr;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }

  bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const override {
    // The generic `InlinerInterface` implementation delegates to the dialect
    // handler of the destination operation. Moore does not currently register a
    // dialect inliner interface, so always disable this optional fast path.
    return false;
  }
};

struct InlineCallsPass
    : public circt::moore::impl::InlineCallsBase<InlineCallsPass> {
  void runOnOperation() override;
};
} // namespace

void InlineCallsPass::runOnOperation() {
  FunctionInliner inliner(&getContext());
  InlinerConfig config;
  auto module = getOperation();

  // Inline `func.call` sites that occur anywhere under a `moore.procedure`. We
  // iteratively inline to a fixed point so newly inlined calls get folded in as
  // well.
  bool didInline = false;
  unsigned iteration = 0;
  constexpr unsigned maxIterations = 128;
  do {
    didInline = false;
    SmallVector<func::CallOp, 32> callOps;
    module.walk([&](func::CallOp callOp) {
      if (!callOp->getParentOfType<ProcedureOp>())
        return;
      if (callOp.getNoInline())
        return;
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (!callee || callee.isExternal())
        return;
      callOps.push_back(callOp);
    });

    for (auto callOp : callOps) {
      // This call may have been erased/moved by a previous inlining step.
      if (!callOp || !callOp->getBlock())
        continue;

      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (!callee || callee.isExternal() || callOp.getNoInline())
        continue;

      LLVM_DEBUG(llvm::dbgs() << "- Inlining " << callOp << "\n");
      if (failed(inlineCall(inliner, config.getCloneCallback(), callOp, callee,
                            callee.getCallableRegion()))) {
        callOp.emitError("function call cannot be inlined");
        signalPassFailure();
        return;
      }
      callOp.erase();
      didInline = true;
    }
  } while (didInline && ++iteration < maxIterations);

  if (didInline) {
    module.emitError("inlining did not converge");
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::moore::createInlineCallsPass() {
  return std::make_unique<InlineCallsPass>();
}
