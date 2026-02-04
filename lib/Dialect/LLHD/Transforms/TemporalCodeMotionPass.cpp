//===- TemporalCodeMotionPass.cpp - Implement Temporal Code Motion Pass ---===//
//
// Implement Pass to move all signal drives in a unique exiting block per
// temporal region and coalesce drives to the same signal.
//
//===----------------------------------------------------------------------===//

#include "TemporalRegions.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <queue>

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_TEMPORALCODEMOTION
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace mlir;

static bool isSimProcPrint(Operation *op) {
  return op && op->getName().getStringRef() == "sim.proc.print";
}

static bool isSimTerminate(Operation *op) {
  return op && op->getName().getStringRef() == "sim.terminate";
}

static bool isSimSideEffectOp(Operation *op) {
  return isSimProcPrint(op) || isSimTerminate(op);
}

/// Return the `hw.constant` i1 value for `bit`.
static Value getOrCreateI1Constant(OpBuilder &builder, Location loc, bool bit) {
  return hw::ConstantOp::create(builder, loc, APInt(1, bit ? 1 : 0));
}

static bool isI1Constant(Value value, bool &bitOut) {
  auto cst = value.getDefiningOp<hw::ConstantOp>();
  if (!cst)
    return false;
  auto attr = dyn_cast<IntegerAttr>(cst.getValueAttr());
  if (!attr)
    return false;
  auto intTy = dyn_cast<IntegerType>(attr.getType());
  if (!intTy || intTy.getWidth() != 1)
    return false;
  bitOut = attr.getValue().isAllOnes();
  return true;
}

/// Match `comb.xor(x, true)` (i.e. boolean NOT).
static bool matchI1Not(Value maybeNot, Value &baseOut) {
  auto xorOp = maybeNot.getDefiningOp<comb::XorOp>();
  if (!xorOp)
    return false;
  SmallVector<Value, 2> inputs(xorOp.getInputs().begin(),
                               xorOp.getInputs().end());
  if (inputs.size() != 2)
    return false;

  bool bit = false;
  if (isI1Constant(inputs[0], bit) && bit) {
    baseOut = inputs[1];
    return true;
  }
  if (isI1Constant(inputs[1], bit) && bit) {
    baseOut = inputs[0];
    return true;
  }
  return false;
}

/// Match `comb.and(not x, x)` / `comb.and(x, not x)` which is semantically
/// false, but may arise when we lose edge-detection structure after rewriting
/// waits/probes.
static bool matchI1AndNotSelf(Value cond) {
  auto andOp = cond.getDefiningOp<comb::AndOp>();
  if (!andOp)
    return false;
  SmallVector<Value, 2> inputs(andOp.getInputs().begin(),
                               andOp.getInputs().end());
  if (inputs.size() != 2)
    return false;

  Value base;
  if (matchI1Not(inputs[0], base) && base == inputs[1])
    return true;
  if (matchI1Not(inputs[1], base) && base == inputs[0])
    return true;
  return false;
}

/// Rewrite a boolean expression by treating any `~x & x` sub-expression as
/// `true`. This is intentionally imprecise, and only used to keep sim side
/// effects from being deleted due to wait/probe canonicalization artifacts.
static Value rewriteBrokenEdgeCondition(Value value, Location loc,
                                        OpBuilder &builder,
                                        DenseMap<Value, Value> &memo) {
  if (!value)
    return {};

  if (auto it = memo.find(value); it != memo.end())
    return it->second;

  // Only handle i1 expressions.
  if (!value.getType().isSignlessInteger(1)) {
    memo[value] = value;
    return value;
  }

  if (matchI1AndNotSelf(value)) {
    Value trueVal = getOrCreateI1Constant(builder, loc, /*bit=*/true);
    memo[value] = trueVal;
    return trueVal;
  }

  if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
    // AND: drop `true` operands, short-circuit on `false`.
    SmallVector<Value, 4> filtered;
    for (Value input : andOp.getInputs()) {
      Value rewritten = rewriteBrokenEdgeCondition(input, loc, builder, memo);
      bool bit = false;
      if (isI1Constant(rewritten, bit)) {
        if (!bit) {
          Value falseVal = getOrCreateI1Constant(builder, loc, /*bit=*/false);
          memo[value] = falseVal;
          return falseVal;
        }
        continue;
      }
      filtered.push_back(rewritten);
    }
    if (filtered.empty()) {
      Value trueVal = getOrCreateI1Constant(builder, loc, /*bit=*/true);
      memo[value] = trueVal;
      return trueVal;
    }
    Value acc = filtered[0];
    for (Value v : llvm::drop_begin(filtered, 1))
      acc = comb::AndOp::create(builder, loc, acc, v).getResult();
    memo[value] = acc;
    return acc;
  }

  if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
    // OR: drop `false` operands, short-circuit on `true`.
    SmallVector<Value, 4> filtered;
    for (Value input : orOp.getInputs()) {
      Value rewritten = rewriteBrokenEdgeCondition(input, loc, builder, memo);
      bool bit = false;
      if (isI1Constant(rewritten, bit)) {
        if (bit) {
          Value trueVal = getOrCreateI1Constant(builder, loc, /*bit=*/true);
          memo[value] = trueVal;
          return trueVal;
        }
        continue;
      }
      filtered.push_back(rewritten);
    }
    if (filtered.empty()) {
      Value falseVal = getOrCreateI1Constant(builder, loc, /*bit=*/false);
      memo[value] = falseVal;
      return falseVal;
    }
    Value acc = filtered[0];
    for (Value v : llvm::drop_begin(filtered, 1))
      acc = comb::OrOp::create(builder, loc, acc, v).getResult();
    memo[value] = acc;
    return acc;
  }

  if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
    // XOR: fold constants where possible (binary/variadic).
    SmallVector<Value, 4> inputs;
    bool parity = false;
    for (Value input : xorOp.getInputs()) {
      Value rewritten = rewriteBrokenEdgeCondition(input, loc, builder, memo);
      bool bit = false;
      if (isI1Constant(rewritten, bit)) {
        parity ^= bit;
        continue;
      }
      inputs.push_back(rewritten);
    }
    if (inputs.empty()) {
      Value cst = getOrCreateI1Constant(builder, loc, parity);
      memo[value] = cst;
      return cst;
    }
    Value acc = inputs[0];
    for (Value v : llvm::drop_begin(inputs, 1))
      acc = comb::XorOp::create(builder, loc, acc, v).getResult();
    if (parity) {
      Value one = getOrCreateI1Constant(builder, loc, /*bit=*/true);
      acc = comb::XorOp::create(builder, loc, acc, one).getResult();
    }
    memo[value] = acc;
    return acc;
  }

  memo[value] = value;
  return value;
}

/// Repair broken edge-detection expressions of the form `~x & x` / `x & ~x`
/// that may arise after canonicalizing wait loops. These expressions are
/// semantically false, but if `x` is a wait-observed trigger and we have a
/// corresponding "past" block argument, we can reconstruct the intended
/// `~past & present` / `past & ~present` form.
static void repairBrokenEdgeDetectors(llhd::ProcessOp procOp) {
  // Find the unique wait terminator and derive the mapping from present
  // triggers (destination operands) to their past loop-carried block args.
  llhd::WaitOp waitOp;
  for (auto candidate : procOp.getOps<llhd::WaitOp>()) {
    if (waitOp)
      return;
    waitOp = candidate;
  }
  if (!waitOp)
    return;

  Block *succ = waitOp.getSuccessor();
  if (!succ)
    return;
  if (succ->getNumArguments() != waitOp.getDestOperands().size())
    return;

  DenseMap<Value, Value> presentToPast;
  for (auto [present, past] :
       llvm::zip(waitOp.getDestOperands(), succ->getArguments())) {
    if (!present.getType().isSignlessInteger(1) ||
        !past.getType().isSignlessInteger(1))
      continue;
    presentToPast.try_emplace(present, past);
  }
  if (presentToPast.empty())
    return;

  SmallVector<comb::AndOp> andOps;
  procOp.walk([&](comb::AndOp andOp) { andOps.push_back(andOp); });

  for (comb::AndOp andOp : andOps) {
    if (!andOp)
      continue;

    auto inputs = llvm::to_vector(andOp.getInputs());
    if (inputs.size() != 2)
      continue;

    Value base;
    bool wantPosedge = false;
    if (matchI1Not(inputs[0], base) && base == inputs[1]) {
      wantPosedge = true;
    } else if (matchI1Not(inputs[1], base) && base == inputs[0]) {
      wantPosedge = false;
    } else {
      continue;
    }

    auto it = presentToPast.find(base);
    if (it == presentToPast.end())
      continue;

    OpBuilder builder(andOp);
    builder.setInsertionPoint(andOp);
    Location loc = andOp.getLoc();
    bool twoState = andOp.getTwoState();
    Value past = it->second;

    Value replacement;
    if (wantPosedge) {
      Value notPast = comb::createOrFoldNot(loc, past, builder, twoState);
      replacement =
          comb::AndOp::create(builder, loc, notPast, base, twoState).getResult();
    } else {
      Value notPresent = comb::createOrFoldNot(loc, base, builder, twoState);
      replacement =
          comb::AndOp::create(builder, loc, past, notPresent, twoState).getResult();
    }

    andOp.getResult().replaceAllUsesWith(replacement);
    andOp.erase();
  }
}

struct ClockStripInfo {
  Value simplified;
  bool onlyObserved = false;
  bool hasObserved = false;
};

/// Simplify a boolean (i1) expression by replacing any sub-expression that
/// depends *only* on `observedValues` (and constants) with `true`.
///
/// This is used to drop time/clock gating when preserving sim side effects
/// inside monitor-only processes, since arcilator's current LLHD->Arc
/// conversion does not model `llhd.wait` scheduling.
static ClockStripInfo stripClockOnlyConditions(
    Value value, const DenseSet<Value> &observedValues, OpBuilder &builder,
    DenseMap<Value, ClockStripInfo> &memo) {
  if (!value)
    return {};

  if (auto it = memo.find(value); it != memo.end())
    return it->second;

  ClockStripInfo result;

  // Constants are "onlyObserved" (depend on no non-observed values), but do not
  // count as observed-dependent.
  bool cstBit = false;
  if (isI1Constant(value, cstBit)) {
    result.simplified = value;
    result.onlyObserved = true;
    result.hasObserved = false;
    memo[value] = result;
    return result;
  }

  // Treat the values we explicitly tracked (wait observed signals and their
  // loop-carried block arguments) as clock-only sources.
  if (observedValues.contains(value)) {
    result.simplified = value;
    result.onlyObserved = true;
    result.hasObserved = true;
    memo[value] = result;
    return result;
  }

  // For non-tracked block arguments, conservatively assume non-clock data.
  if (isa<BlockArgument>(value)) {
    result.simplified = value;
    result.onlyObserved = false;
    result.hasObserved = false;
    memo[value] = result;
    return result;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.simplified = value;
    result.onlyObserved = false;
    result.hasObserved = false;
    memo[value] = result;
    return result;
  }

  auto simplifyBinaryBool =
      [&](Value lhs, Value rhs,
          function_ref<Value(Location, Value, Value)> mkOp) -> ClockStripInfo {
    auto lhsInfo = stripClockOnlyConditions(lhs, observedValues, builder, memo);
    auto rhsInfo = stripClockOnlyConditions(rhs, observedValues, builder, memo);

    bool onlyObserved = lhsInfo.onlyObserved && rhsInfo.onlyObserved;
    bool hasObserved = lhsInfo.hasObserved || rhsInfo.hasObserved;

    Value simplifiedLhs = lhsInfo.simplified;
    Value simplifiedRhs = rhsInfo.simplified;

    // If the entire subtree depends only on observed values, drop it.
    if (onlyObserved && hasObserved) {
      ClockStripInfo info;
      info.simplified =
          getOrCreateI1Constant(builder, defOp->getLoc(), /*bit=*/true);
      info.onlyObserved = true;
      info.hasObserved = false;
      return info;
    }

    // Fold simple boolean constants.
    bool lhsBit = false, rhsBit = false;
    if (isI1Constant(simplifiedLhs, lhsBit) &&
        isI1Constant(simplifiedRhs, rhsBit)) {
      // Let MLIR fold it via mkOp so we don't duplicate truth tables.
      ClockStripInfo info;
      info.simplified = mkOp(defOp->getLoc(), simplifiedLhs, simplifiedRhs);
      info.onlyObserved = true;
      info.hasObserved = false;
      return info;
    }

    ClockStripInfo info;
    info.simplified = mkOp(defOp->getLoc(), simplifiedLhs, simplifiedRhs);
    info.onlyObserved = onlyObserved;
    info.hasObserved = hasObserved;
    return info;
  };

  if (auto andOp = dyn_cast<comb::AndOp>(defOp)) {
    SmallVector<Value, 2> inputs(andOp.getInputs().begin(),
                                 andOp.getInputs().end());
    if (inputs.size() != 2) {
      result.simplified = value;
      result.onlyObserved = false;
      result.hasObserved = false;
      memo[value] = result;
      return result;
    }
    result = simplifyBinaryBool(
        inputs[0], inputs[1],
        [&](Location loc, Value lhs, Value rhs) -> Value {
          // AND identity/annihilator
          bool lhsBit = false, rhsBit = false;
          if (isI1Constant(lhs, lhsBit)) {
            if (!lhsBit)
              return getOrCreateI1Constant(builder, loc, false);
            return rhs;
          }
          if (isI1Constant(rhs, rhsBit)) {
            if (!rhsBit)
              return getOrCreateI1Constant(builder, loc, false);
            return lhs;
          }
          return comb::AndOp::create(builder, loc, lhs, rhs).getResult();
        });
    memo[value] = result;
    return result;
  }

  if (auto orOp = dyn_cast<comb::OrOp>(defOp)) {
    SmallVector<Value, 2> inputs(orOp.getInputs().begin(), orOp.getInputs().end());
    if (inputs.size() != 2) {
      result.simplified = value;
      result.onlyObserved = false;
      result.hasObserved = false;
      memo[value] = result;
      return result;
    }
    result = simplifyBinaryBool(
        inputs[0], inputs[1],
        [&](Location loc, Value lhs, Value rhs) -> Value {
          // OR identity/annihilator
          bool lhsBit = false, rhsBit = false;
          if (isI1Constant(lhs, lhsBit)) {
            if (lhsBit)
              return getOrCreateI1Constant(builder, loc, true);
            return rhs;
          }
          if (isI1Constant(rhs, rhsBit)) {
            if (rhsBit)
              return getOrCreateI1Constant(builder, loc, true);
            return lhs;
          }
          return comb::OrOp::create(builder, loc, lhs, rhs).getResult();
        });
    memo[value] = result;
    return result;
  }

  if (auto xorOp = dyn_cast<comb::XorOp>(defOp)) {
    SmallVector<Value, 2> inputs(xorOp.getInputs().begin(),
                                 xorOp.getInputs().end());
    if (inputs.size() != 2) {
      result.simplified = value;
      result.onlyObserved = false;
      result.hasObserved = false;
      memo[value] = result;
      return result;
    }
    result = simplifyBinaryBool(
        inputs[0], inputs[1],
        [&](Location loc, Value lhs, Value rhs) -> Value {
          bool rhsBit = false;
          if (isI1Constant(rhs, rhsBit)) {
            if (!rhsBit)
              return lhs;
            return comb::XorOp::create(builder, loc, lhs,
                                       getOrCreateI1Constant(builder, loc, true))
                .getResult();
          }
          bool lhsBit = false;
          if (isI1Constant(lhs, lhsBit)) {
            if (!lhsBit)
              return rhs;
            return comb::XorOp::create(builder, loc, rhs,
                                       getOrCreateI1Constant(builder, loc, true))
                .getResult();
          }
          return comb::XorOp::create(builder, loc, lhs, rhs).getResult();
        });
    memo[value] = result;
    return result;
  }

  // Conservatively keep unknown boolean producers.
  result.simplified = value;
  result.onlyObserved = false;
  result.hasObserved = false;
  memo[value] = result;
  return result;
}

/// Clone a value into the target block if it is defined in a different block
/// of the same region. This is a best-effort helper for preserving the
/// definitions of branch conditions when structuralizing a process.
static Value cloneValueIntoBlock(Value value, OpBuilder &builder,
                                 Block *targetBlock, IRMapping &mapping,
                                 SmallPtrSetImpl<Operation *> &onStack) {
  if (!value)
    return {};

  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;

  // Block arguments dominate all blocks in the region.
  if (isa<BlockArgument>(value))
    return value;

  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return value;

  // Only clone values defined in the same region. Values defined outside the
  // region already dominate the region.
  if (defOp->getParentRegion() != targetBlock->getParent())
    return value;

  // If already in the target block, nothing to do.
  if (defOp->getBlock() == targetBlock)
    return value;

  // Do not clone ops with regions; this is intended for leaf, SSA-based
  // computations such as comb/hw/llhd probes.
  if (defOp->getNumRegions() != 0)
    return {};

  // LLHD probes/signals carry memory effects but are safe to clone for branch
  // condition reconstruction. Allow those through.
  bool allowSideEffects =
      isa<llhd::PrbOp>(defOp) || isa<llhd::SignalOp>(defOp);
  if (!isMemoryEffectFree(defOp) && !allowSideEffects)
    return {};

  // Cycle breaker.
  if (!onStack.insert(defOp).second)
    return {};

  // Clone operands first.
  for (Value operand : defOp->getOperands()) {
    Value mappedOperand =
        cloneValueIntoBlock(operand, builder, targetBlock, mapping, onStack);
    if (!mappedOperand)
      return {};
    mapping.map(operand, mappedOperand);
  }

  Operation *cloned = builder.clone(*defOp, mapping);
  onStack.erase(defOp);

  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult)
    return {};
  Value clonedResult = cloned->getResult(opResult.getResultNumber());
  mapping.map(value, clonedResult);
  return clonedResult;
}

/// Find an `llhd.prb` in the parent graph region (outside any process) which
/// probes `signal`, or create a new one. Returns the probed value.
static Value getOrCreateGraphProbe(Value signal, Operation *insertBefore,
                                   Location loc) {
  auto *parentRegion = signal.getParentRegion();
  assert(parentRegion && "expected signal to have a parent region");

  // Reuse an existing probe in the parent graph region (module body).
  for (Operation *user : signal.getUsers()) {
    auto probe = dyn_cast<llhd::PrbOp>(user);
    if (!probe)
      continue;
    if (probe->getParentRegion() != parentRegion)
      continue;
    return probe.getResult();
  }

  OpBuilder builder(insertBefore);
  builder.setInsertionPoint(insertBefore);
  return builder.create<llhd::PrbOp>(loc, signal).getResult();
}

/// Rewrite the common 2-TR pattern
///
///   entry -> waitBlock --wait--> loopBlock -> waitBlock
///
/// into a single-block wait loop where the wait targets itself and carries
/// prior observed values as destination operands. This puts drives into the
/// same block as the wait and makes the past/present trigger values explicit,
/// enabling later LLHD lowering passes to convert the process into registers.
static LogicalResult canonicalizeWaitLoop(llhd::ProcessOp procOp) {
  if (procOp.getNumResults() != 0)
    return failure();

  llhd::WaitOp waitOp;
  for (auto &block : procOp.getBody()) {
    if (auto candidate = dyn_cast<llhd::WaitOp>(block.getTerminator())) {
      if (waitOp)
        return failure();
      waitOp = candidate;
    }
  }
  if (!waitOp)
    return failure();

  // Only canonicalize event-driven waits without yields/destination operands.
  if (!waitOp.getYieldOperands().empty() || waitOp.getDelay() ||
      !waitOp.getDestOperands().empty())
    return failure();

  Block *waitBlock = waitOp->getBlock();
  Block *loopBlock = waitOp.getSuccessor();
  if (!waitBlock || !loopBlock)
    return failure();

  // Require the loop block to branch unconditionally back to the wait block
  // without carrying values. The canonical form will instead end in a wait.
  auto loopBranch = dyn_cast<cf::BranchOp>(loopBlock->getTerminator());
  if (!loopBranch || loopBranch.getDest() != waitBlock ||
      !loopBranch.getDestOperands().empty())
    return failure();

  // Require the entry block to branch unconditionally to the wait block.
  Block *entryBlock = &procOp.getBody().front();
  if (entryBlock == waitBlock)
    return failure();
  auto entryBranch = dyn_cast<cf::BranchOp>(entryBlock->getTerminator());
  if (!entryBranch || entryBranch.getDest() != waitBlock ||
      !entryBranch.getDestOperands().empty())
    return failure();

  // Require the wait block to only be reached from entry and the loop back
  // edge. This ensures it becomes dead after the rewrite.
  SmallPtrSet<Block *, 4> waitPreds;
  for (Block *pred : waitBlock->getPredecessors())
    waitPreds.insert(pred);
  if (waitPreds.size() != 2 || !waitPreds.contains(entryBlock) ||
      !waitPreds.contains(loopBlock))
    return failure();

  // The canonical form requires a loop block with block arguments for the
  // past observed values.
  if (!loopBlock->getArguments().empty())
    return failure();
  if (!waitBlock->getArguments().empty())
    return failure();

  // Create/reuse graph probes for the observed signals. These values live
  // outside the process and can be used as register clocks later.
  SmallVector<Value> observedValues;
  SmallVector<Value> observedSignals;
  observedValues.reserve(waitOp.getObserved().size());
  observedSignals.reserve(waitOp.getObserved().size());
  for (Value observed : waitOp.getObserved()) {
    if (!observed.getType().isSignlessInteger(1))
      return failure();

    auto probe = observed.getDefiningOp<llhd::PrbOp>();
    if (!probe)
      return failure();
    Value signal = probe.getSignal();
    if (!signal.getParentRegion()->isProperAncestor(&procOp.getBody()))
      return failure();

    observedValues.push_back(observed);
    observedSignals.push_back(signal);
  }

  // Determine which values need to be cloned out of the old wait block, and
  // ensure we can do so without mutating the IR.
  SmallVector<Value> crossValues;
  for (Operation &op : waitBlock->without_terminator()) {
    for (Value result : op.getResults()) {
      bool usedInLoop = false;
      for (OpOperand &use : result.getUses()) {
        Block *useBlock = use.getOwner()->getBlock();
        if (useBlock == waitBlock)
          continue;
        if (useBlock != loopBlock)
          return failure();
        usedInLoop = true;
      }
      if (usedInLoop)
        crossValues.push_back(result);
    }
  }

  SmallPtrSet<Operation *, 16> canCloneStack;
  DenseSet<Value> observedSet;
  observedSet.insert(observedValues.begin(), observedValues.end());
  auto canCloneValue = [&](auto &&self, Value value) -> bool {
    if (!value)
      return false;
    if (observedSet.contains(value))
      return true;
    auto *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != waitBlock)
      return true;
    if (!isMemoryEffectFree(defOp) || defOp->getNumRegions() != 0)
      return false;
    if (!canCloneStack.insert(defOp).second)
      return false;
    for (Value operand : defOp->getOperands())
      if (!self(self, operand))
        return false;
    canCloneStack.erase(defOp);
    return true;
  };
  for (Value value : crossValues)
    if (!canCloneValue(canCloneValue, value))
      return failure();

  // Create/reuse graph probes for the observed signals. These values live
  // outside the process and can be used as register clocks later.
  SmallVector<Value> presentTriggers;
  presentTriggers.reserve(observedSignals.size());
  for (auto [signal, observed] : llvm::zip(observedSignals, observedValues))
    presentTriggers.push_back(
        getOrCreateGraphProbe(signal, procOp, observed.getLoc()));

  // Add block arguments to hold the past observed values.
  SmallVector<BlockArgument> pastArgs;
  pastArgs.reserve(presentTriggers.size());
  for (Value trigger : presentTriggers)
    pastArgs.push_back(
        loopBlock->addArgument(trigger.getType(), procOp.getLoc()));

  // Clone any values defined in the wait block which are used in the loop
  // block, remapping observed values to the newly created past block
  // arguments.
  IRMapping mapping;
  for (auto [observed, past] : llvm::zip(observedValues, pastArgs))
    mapping.map(observed, past);

  OpBuilder cloneBuilder(procOp.getContext());
  cloneBuilder.setInsertionPointToStart(loopBlock);

  SmallPtrSet<Operation *, 16> cloneInProgress;
  auto cloneValue = [&](auto &&self, Value value) -> Value {
    if (auto mapped = mapping.lookupOrNull(value))
      return mapped;

    auto *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != waitBlock)
      return value;

    if (!isMemoryEffectFree(defOp) || defOp->getNumRegions() != 0)
      return Value{};

    if (!cloneInProgress.insert(defOp).second)
      return Value{};

    SmallVector<Value> newOperands;
    newOperands.reserve(defOp->getNumOperands());
    for (Value operand : defOp->getOperands()) {
      Value mappedOperand = self(self, operand);
      if (!mappedOperand)
        return Value{};
      newOperands.push_back(mappedOperand);
    }

    Operation *cloned = cloneBuilder.clone(*defOp, mapping);
    (void)cloned;
    return mapping.lookup(value);
  };

  for (Value oldValue : crossValues) {
    Value newValue = cloneValue(cloneValue, oldValue);
    if (!newValue)
      return failure();
    oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
      return use.getOwner()->getBlock() == loopBlock;
    });
  }

  // Replace any probes of observed signals in the loop block with the graph
  // probe values, since the wait will observe the graph probe.
  DenseMap<Value, Value> signalToTrigger;
  for (auto [signal, trigger] : llvm::zip(observedSignals, presentTriggers))
    signalToTrigger.insert({signal, trigger});
  for (auto probe :
       llvm::make_early_inc_range(loopBlock->getOps<llhd::PrbOp>())) {
    if (auto it = signalToTrigger.find(probe.getSignal());
        it != signalToTrigger.end()) {
      probe.getResult().replaceAllUsesWith(it->second);
      probe.erase();
    }
  }

  // Replace the loop-back branch with a wait that observes the current trigger
  // values and passes them as destination operands to create past values in the
  // next iteration.
  OpBuilder endBuilder(loopBranch);
  endBuilder.setInsertionPoint(loopBranch);
  llhd::WaitOp::create(endBuilder, loopBranch.getLoc(), ValueRange{}, Value{},
                       presentTriggers, presentTriggers, loopBlock);
  loopBranch.erase();

  // Branch into the loop with the initial past values set to the current
  // trigger values. This avoids spurious edges on simulation start.
  OpBuilder entryBuilder(entryBranch);
  entryBuilder.setInsertionPoint(entryBranch);
  cf::BranchOp::create(entryBuilder, entryBranch.getLoc(), loopBlock,
                       presentTriggers);
  entryBranch.erase();

  // Remove the now-unreachable wait block.
  IRRewriter rewriter(procOp.getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter, procOp->getRegions());

  return success();
}

/// Explore all paths from the 'driveBlock' to the 'dominator' block and
/// construct a boolean expression at the current insertion point of 'builder'
/// to represent all those paths.
static Value
getBranchDecisionsFromDominatorToTarget(OpBuilder &builder, Block *driveBlock,
                                        Block *dominator,
                                        DenseMap<Block *, Value> &mem,
                                        IRMapping &clonedValues,
                                        SmallPtrSetImpl<Operation *> &onStack) {
  Location loc = driveBlock->getTerminator()->getLoc();
  if (mem.count(driveBlock))
    return mem[driveBlock];

  SmallVector<Block *> worklist;
  worklist.push_back(driveBlock);

  while (!worklist.empty()) {
    Block *curr = worklist.back();

    if (curr == dominator || curr->getPredecessors().empty()) {
      if (!mem.count(curr))
        mem[curr] = hw::ConstantOp::create(builder, loc, APInt(1, 1));

      worklist.pop_back();
      continue;
    }

    bool addedSomething = false;
    for (auto *predBlock : curr->getPredecessors()) {
      if (!mem.count(predBlock)) {
        worklist.push_back(predBlock);
        addedSomething = true;
      }
    }

    if (addedSomething)
      continue;

    Value runner = hw::ConstantOp::create(builder, loc, APInt(1, 0));
    for (auto *predBlock : curr->getPredecessors()) {
      if (predBlock->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(predBlock->getTerminator());
        Value cond = cloneValueIntoBlock(condBr.getCondition(), builder,
                                         builder.getBlock(), clonedValues,
                                         onStack);
        if (!cond)
          cond = condBr.getCondition();
        if (condBr.getFalseDest() == curr) {
          Value trueVal = hw::ConstantOp::create(builder, loc, APInt(1, 1));
          cond = comb::XorOp::create(builder, loc, cond, trueVal);
        }
        Value next = comb::AndOp::create(builder, loc, mem[predBlock], cond);
        runner = comb::OrOp::create(builder, loc, runner, next);
      } else {
        runner = comb::OrOp::create(builder, loc, runner, mem[predBlock]);
      }
    }
    mem[curr] = runner;
    worklist.pop_back();
  }

  return mem[driveBlock];
}

/// More a 'llhd.drv' operation before the 'moveBefore' operation by adjusting
/// the 'enable' operand.
static void moveDriveOpBefore(llhd::DrvOp drvOp, Block *dominator,
                              Operation *moveBefore,
                              DenseMap<Block *, Value> &mem,
                              IRMapping &clonedValues,
                              SmallPtrSetImpl<Operation *> &onStack) {
  OpBuilder builder(drvOp);
  builder.setInsertionPoint(moveBefore);
  Block *drvParentBlock = drvOp->getBlock();

  // Find sequence of branch decisions and add them as a sequence of
  // instructions to the TR exiting block
  Value finalValue = getBranchDecisionsFromDominatorToTarget(
      builder, drvParentBlock, dominator, mem, clonedValues, onStack);

  if (drvOp.getEnable())
    finalValue = comb::AndOp::create(builder, drvOp.getLoc(), drvOp.getEnable(),
                                     finalValue);

  drvOp.getEnableMutable().assign(finalValue);
  drvOp->moveBefore(moveBefore);
}

namespace {
struct TemporalCodeMotionPass
    : public llhd::impl::TemporalCodeMotionBase<TemporalCodeMotionPass> {
  void runOnOperation() override;
  LogicalResult runOnProcess(llhd::ProcessOp procOp);
};
} // namespace

void TemporalCodeMotionPass::runOnOperation() {
  for (auto proc : getOperation().getOps<llhd::ProcessOp>())
    (void)runOnProcess(proc); // Ignore processes that could not be lowered
}

static LogicalResult checkForCFGLoop(llhd::ProcessOp procOp) {
  // The temporal region analysis underpinning this pass cannot handle CFG
  // cycles that do *not* pass through a wait terminator. Detect those by doing
  // a DFS and tracking the current recursion stack. Shared successors in the
  // CFG (i.e. join points) are fine and should not be mistaken for loops.

  SmallVector<Block *> roots(
      llvm::map_range(procOp.getOps<llhd::WaitOp>(), [](llhd::WaitOp waitOp) {
        return waitOp.getSuccessor();
      }));
  roots.push_back(&procOp.getBody().front());

  auto isWaitBlock = [](Block *block) {
    return isa<llhd::WaitOp>(block->getTerminator());
  };

  for (Block *root : roots) {
    SmallPtrSet<Block *, 32> visited;
    SmallPtrSet<Block *, 32> onStack;

    std::function<bool(Block *)> dfs = [&](Block *block) -> bool {
      if (isWaitBlock(block))
        return false;
      if (onStack.contains(block))
        return true;
      if (visited.contains(block))
        return false;

      visited.insert(block);
      onStack.insert(block);
      for (Block *succ : block->getSuccessors())
        if (dfs(succ))
          return true;
      onStack.erase(block);
      return false;
    };

    if (dfs(root))
      return failure();
  }

  return success();
}

LogicalResult TemporalCodeMotionPass::runOnProcess(llhd::ProcessOp procOp) {
  // Collect simulation side-effect ops (e.g. from `$display/$fatal/$error`)
  // that we may want to preserve when a process has no drives and would
  // otherwise be optimized away.
  SmallVector<Operation *> simSideEffectOps;
  bool hasDrv = false;
  procOp.walk([&](Operation *op) {
    if (isa<llhd::DrvOp>(op))
      hasDrv = true;
    if (isSimSideEffectOp(op))
      simSideEffectOps.push_back(op);
  });
  bool preserveSimSideEffects = !hasDrv && !simSideEffectOps.empty();

  // Make sure there are no CFG loops that don't contain a block with a wait
  // terminator in the cycle because that's currently not supported by the
  // temporal region analysis and this pass.
  if (failed(checkForCFGLoop(procOp)))
    return failure();

  llhd::TemporalRegionAnalysis trAnalysis =
      llhd::TemporalRegionAnalysis(procOp);
  unsigned numTRs = trAnalysis.getNumTemporalRegions();

  // Only support processes with max. 2 temporal regions and one wait terminator
  // as this is enough to represent flip-flops, registers, etc.
  // NOTE: there always has to be either a wait or halt terminator in a process
  // If the wait block creates the backwards edge, we only have one TR,
  // otherwise we have 2 TRs
  // NOTE: as the wait instruction needs to be on every path around the loop,
  // it has to be the only exiting block of its TR
  // NOTE: the other TR can either have only one exiting block, then we do not
  // need to add an auxillary block, otherwise we have to add one
  // NOTE: All drive operations have to be moved to the single exiting block of
  // its TR. To do so, add the condition under which its block is reached from
  // the TR entry block as a gating condition to the 'llhd.drv' operation
  // NOTE: the entry blocks that are not part of the infinite loop do not count
  // as TR and have TR number -1
  // TODO: need to check that entry blocks that are note part of the loop to not
  // have any instructions that have side effects that should not be allowed
  // outside of the loop (drv, prb, ...)
  // TODO: add support for more TRs and wait terminators (e.g., to represent
  // FSMs)
  if (numTRs > 2)
    return failure();

  bool seenWait = false;
  WalkResult walkResult = procOp.walk([&](llhd::WaitOp op) -> WalkResult {
    if (seenWait)
      return failure();

    // Check that the block containing the wait is the only exiting block of
    // that TR
    int trId = trAnalysis.getBlockTR(op->getBlock());
    if (!trAnalysis.hasSingleExitBlock(trId))
      return failure();

    seenWait = true;
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  //===--------------------------------------------------------------------===//
  // Create unique exit block per TR
  //===--------------------------------------------------------------------===//

  // TODO: consider the case where a wait brances to itself
  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    unsigned numTRSuccs = trAnalysis.getNumTRSuccessors(currTR);
    (void)numTRSuccs;
    // NOTE: Above error checks make this impossible to trigger, but the above
    // are changed this one might have to be promoted to a proper error message.
    assert((numTRSuccs == 0 || numTRSuccs == 1 ||
            (numTRSuccs == 2 && trAnalysis.isOwnTRSuccessor(currTR))) &&
           "only TRs with a single TR as possible successor are "
           "supported for now.");

    if (trAnalysis.hasSingleExitBlock(currTR))
      continue;

    // Get entry block of successor TR
    Block *succTREntry =
        trAnalysis.getTREntryBlock(*trAnalysis.getTRSuccessors(currTR).begin());

    // Create the auxillary block as we currently don't have a single exiting
    // block and give it the same arguments as the entry block of the
    // successor TR
    Block *auxBlock = new Block();
    auxBlock->addArguments(
        succTREntry->getArgumentTypes(),
        SmallVector<Location>(succTREntry->getNumArguments(), procOp.getLoc()));

    // Insert the auxillary block after the last block of the current TR
    procOp.getBody().getBlocks().insertAfter(
        Region::iterator(trAnalysis.getExitingBlocksInTR(currTR).back()),
        auxBlock);

    // Let all current exit blocks branch to the auxillary block instead.
    for (Block *exit : trAnalysis.getExitingBlocksInTR(currTR))
      for (auto [i, succ] : llvm::enumerate(exit->getSuccessors()))
        if (trAnalysis.getBlockTR(succ) != static_cast<int>(currTR))
          exit->getTerminator()->setSuccessor(auxBlock, i);

    // Let the auxiallary block branch to the entry block of the successor
    // temporal region entry block
    OpBuilder b(procOp);
    b.setInsertionPointToEnd(auxBlock);
    cf::BranchOp::create(b, procOp.getLoc(), succTREntry,
                         auxBlock->getArguments());
  }

  //===--------------------------------------------------------------------===//
  // Move drive instructions
  //===--------------------------------------------------------------------===//

  DenseMap<Operation *, Block *> drvPos;

  // Force a new analysis as we have changed the CFG
  trAnalysis = llhd::TemporalRegionAnalysis(procOp);
  numTRs = trAnalysis.getNumTemporalRegions();
  OpBuilder builder(procOp);

  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    DenseMap<Block *, Value> mem;
    IRMapping clonedValues;
    SmallPtrSet<Operation *, 32> cloneStack;

    // We ensured this in the previous phase above.
    assert(trAnalysis.getExitingBlocksInTR(currTR).size() == 1);

    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    Block *entryBlock = trAnalysis.getTREntryBlock(currTR);

    DominanceInfo dom(procOp);
    Block *dominator = exitingBlock;

    // Collect all 'llhd.drv' operations in the process and compute their common
    // dominator block.
    procOp.walk([&](llhd::DrvOp op) {
      if (trAnalysis.getBlockTR(op.getOperation()->getBlock()) ==
          static_cast<int>(currTR)) {
        Block *parentBlock = op.getOperation()->getBlock();
        drvPos[op] = parentBlock;
        dominator = dom.findNearestCommonDominator(dominator, parentBlock);
      }
    });

    // Set insertion point before first 'llhd.drv' op in the exiting block
    Operation *moveBefore = exitingBlock->getTerminator();
    exitingBlock->walk([&](llhd::DrvOp op) { moveBefore = op; });

    assert(dominator &&
           "could not find nearest common dominator for TR exiting "
           "block and the block containing drv");

    // If the dominator isn't already a TR entry block, set it to the nearest
    // dominating TR entry block.
    if (trAnalysis.getBlockTR(dominator) != static_cast<int>(currTR))
      dominator = trAnalysis.getTREntryBlock(currTR);

    std::queue<Block *> workQueue;
    SmallPtrSet<Block *, 32> workDone;

    if (entryBlock != exitingBlock)
      workQueue.push(entryBlock);

    while (!workQueue.empty()) {
      Block *block = workQueue.front();
      workQueue.pop();
      workDone.insert(block);

      builder.setInsertionPoint(moveBefore);
      SmallVector<llhd::DrvOp> drives(block->getOps<llhd::DrvOp>());
      for (auto drive : drives)
        moveDriveOpBefore(drive, dominator, moveBefore, mem, clonedValues,
                          cloneStack);

      for (Block *succ : block->getSuccessors()) {
        if (succ == exitingBlock ||
            trAnalysis.getBlockTR(succ) != static_cast<int>(currTR))
          continue;

        if (llvm::all_of(succ->getPredecessors(), [&](Block *block) {
              return workDone.contains(block);
            }))
          workQueue.push(succ);
      }
    }

    // If this process has no drives, preserve simulation side-effect ops by
    // cloning them into the current TR's exit block under the reachability
    // condition from the TR entry. This prevents `$display/$fatal/$error`
    // statements in "monitor-only" always blocks from being dropped.
    if (preserveSimSideEffects) {
      SmallVector<Operation *> toErase;
      OpBuilder seBuilder(procOp.getContext());
      seBuilder.setInsertionPoint(moveBefore);

      // Use the TR entry as dominator when reconstructing reachability.
      Block *reachDominator = entryBlock;

      // Collect wait-observed values (and their loop-carried block args) as
      // "clock-only" inputs; we will strip them from the enable conditions.
      DenseSet<Value> observedValues;
      DenseSet<Value> observedSignals;
      procOp.walk([&](llhd::WaitOp waitOp) {
        for (Value observed : waitOp.getObserved()) {
          observedValues.insert(observed);
          if (auto prb = observed.getDefiningOp<llhd::PrbOp>())
            observedSignals.insert(prb.getSignal());
        }
        Block *succ = waitOp.getSuccessor();
        if (!succ)
          return;
        for (BlockArgument arg : succ->getArguments())
          observedValues.insert(arg);
      });
      // Also treat *all* probes of the observed signals as clock-only. This
      // ensures edge-detection patterns like `~old & new` (which use multiple
      // probes of the same signal) get stripped before later rewrites (such as
      // canonicalizeWaitLoop) fold them into an impossible `~x & x` predicate.
      if (!observedSignals.empty()) {
        procOp.walk([&](llhd::PrbOp prb) {
          if (observedSignals.contains(prb.getSignal()))
            observedValues.insert(prb.getResult());
        });
      }
      DenseMap<Value, ClockStripInfo> clockStripMemo;

      for (Operation *sideOp : simSideEffectOps) {
        if (!sideOp || sideOp->getParentOp() != procOp)
          continue;
        if (trAnalysis.getBlockTR(sideOp->getBlock()) !=
            static_cast<int>(currTR))
          continue;

        Value enable = getBranchDecisionsFromDominatorToTarget(
            seBuilder, sideOp->getBlock(), reachDominator, mem, clonedValues,
            cloneStack);

        // Drop any gating that depends only on the wait-observed values. This
        // effectively treats sys tasks as being evaluated on each `eval()` step
        // in the current arcilator model, which is sufficient for interactive
        // debugging and avoids clock-gating being folded to false.
        auto stripped =
            stripClockOnlyConditions(enable, observedValues, seBuilder,
                                     clockStripMemo);
        if (stripped.simplified)
          enable = stripped.simplified;

        // Clone side-op operands into the exit block so they stay live after
        // we drop intermediate blocks.
        for (Value operand : sideOp->getOperands()) {
          Value cloned = cloneValueIntoBlock(operand, seBuilder,
                                             seBuilder.getBlock(), clonedValues,
                                             cloneStack);
          if (!cloned)
            cloned = operand;
          if (!clonedValues.lookupOrNull(operand))
            clonedValues.map(operand, cloned);
        }

        // Create `scf.if` guarding the cloned side effect.
        auto ifOp =
            scf::IfOp::create(seBuilder, sideOp->getLoc(), enable, false);
        Block &thenBlock = ifOp.getThenRegion().front();
        OpBuilder thenBuilder(&thenBlock,
                              thenBlock.getTerminator()->getIterator());

        thenBuilder.clone(*sideOp, clonedValues);
        toErase.push_back(sideOp);
      }

      for (Operation *op : toErase)
        op->erase();
    }

    // Merge entry and exit block of each TR, remove all other blocks
    if (entryBlock != exitingBlock) {
      entryBlock->getTerminator()->erase();
      entryBlock->getOperations().splice(entryBlock->end(),
                                         exitingBlock->getOperations());
    }
  }

  IRRewriter rewriter(procOp);
  (void)mlir::eraseUnreachableBlocks(rewriter, procOp->getRegions());

  //===--------------------------------------------------------------------===//
  // Coalesce multiple drives to the same signal
  //===--------------------------------------------------------------------===//

  trAnalysis = llhd::TemporalRegionAnalysis(procOp);
  numTRs = trAnalysis.getNumTemporalRegions();
  DominanceInfo dom(procOp);
  for (unsigned currTR = 0; currTR < numTRs; ++currTR) {
    // We ensured this in the previous phase above.
    assert(trAnalysis.getExitingBlocksInTR(currTR).size() == 1);

    Block *exitingBlock = trAnalysis.getExitingBlocksInTR(currTR)[0];
    DenseMap<std::pair<Value, Value>, llhd::DrvOp> sigToDrv;

    SmallVector<llhd::DrvOp> drives(exitingBlock->getOps<llhd::DrvOp>());
    for (auto op : drives) {
      auto sigTimePair = std::make_pair(op.getSignal(), op.getTime());
      if (!sigToDrv.count(sigTimePair)) {
        sigToDrv[sigTimePair] = op;
        continue;
      }

      OpBuilder builder(op);
      if (op.getEnable()) {
        // Multiplex value to be driven
        auto firstDrive = sigToDrv[sigTimePair];
        Value muxValue =
            comb::MuxOp::create(builder, op.getLoc(), op.getEnable(),
                                op.getValue(), firstDrive.getValue());
        op.getValueMutable().assign(muxValue);

        // Take the disjunction of the enable conditions
        if (firstDrive.getEnable()) {
          Value orVal = comb::OrOp::create(builder, op.getLoc(), op.getEnable(),
                                           firstDrive.getEnable());
          op.getEnableMutable().assign(orVal);
        } else {
          // No enable is equivalent to a constant 'true' enable
          op.getEnableMutable().clear();
        }
      }

      sigToDrv[sigTimePair]->erase();
      sigToDrv[sigTimePair] = op;
    }
  }

  // If this pass produced the common "wait block + loop block" pattern, rewrite
  // it into a single wait loop that is friendlier to drive hoisting and
  // desequentialization. Ignore processes that do not match the pattern.
  (void)canonicalizeWaitLoop(procOp);
  repairBrokenEdgeDetectors(procOp);

  // canonicalizeWaitLoop replaces multiple `llhd.prb` ops of an observed signal
  // with a single graph probe. If we already cloned sim side effects into a
  // unified exit block and structuralized control flow with `scf.if`, this can
  // collapse edge-detection predicates into an impossible `~x & x` form. For
  // interactive `$display/$fatal/$error`, prefer executing the side effect
  // rather than dropping it.
  // Disabled: `rewriteBrokenEdgeCondition` can encounter invalid SSA values in
  // some canonicalizeWaitLoop patterns and crash. Prefer leaving the condition
  // unchanged over crashing the entire simulation pipeline.

  // LLHD processes use SSACFG regions which require the entry block to have no
  // predecessors. Some of the CFG rewriting above can collapse the original
  // entry block and leave a loop header as the first block, violating that
  // invariant. Repair by inserting a fresh empty entry block when needed.
  Block &entry = procOp.getBody().front();
  if (!entry.hasNoPredecessors()) {
    auto *newEntry = new Block();
    procOp.getBody().getBlocks().push_front(newEntry);
    OpBuilder builder(procOp.getContext());
    builder.setInsertionPointToStart(newEntry);
    cf::BranchOp::create(builder, procOp.getLoc(), &entry, ValueRange{});
  }

  return success();
}
