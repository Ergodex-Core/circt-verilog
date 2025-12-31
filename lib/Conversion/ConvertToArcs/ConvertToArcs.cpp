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
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "circt/Support/Namespace.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

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
static constexpr StringLiteral kArcilatorNeedsSchedulerAttr =
    "arcilator.needs_scheduler";

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
  bool hasDelay = false;
  bool hasWideObserved = false;
  op.walk([&](llhd::WaitOp wait) {
    if (wait.getDelay())
      hasDelay = true;
    for (Value obs : wait.getObserved()) {
      auto intTy = dyn_cast<IntegerType>(obs.getType());
      if (!intTy || intTy.getWidth() != 1)
        hasWideObserved = true;
    }
  });
  return hasDelay || hasWideObserved;
}

static bool isRematerializableForPolling(Operation *op) {
  if (!op)
    return false;
  if (op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
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
  for (auto &block : region)
    if (&block != entryBlock && !block.getArguments().empty())
      return execOp.emitOpError()
             << "cannot lower scheduled process with non-entry block arguments";

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
      if (usedOutsideBlock)
        hoistable.push_back(&op);
    }
  }
  for (Operation *op : hoistable)
    op->moveBefore(dispatchBlock, dispatchBlock->end());

  // Split each `llhd.wait` into its own block so we don't re-run side effects in
  // the pre-wait block while waiting.
  SmallVector<std::pair<Block *, llhd::WaitOp>> waitBlocks;
  SmallVector<llhd::WaitOp> waits;
  execOp.walk([&](llhd::WaitOp w) { waits.push_back(w); });
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
    if (!waitOp.getDestOperands().empty())
      return waitOp.emitOpError() << "scheduled wait with dest operands unsupported";

    rewriter.setInsertionPoint(waitOp);

    auto waitIdAttr = waitOp->getAttrOfType<IntegerAttr>(kArcilatorWaitIdAttr);
    if (!waitIdAttr)
      return waitOp.emitOpError() << "missing wait id attribute";
    Value waitIdVal =
        buildI32Constant(rewriter, waitOp.getLoc(), waitIdAttr.getInt());

    Value ready;
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
      Value sig = buildI64Constant(rewriter, waitOp.getLoc(), 0);
      DenseMap<Value, Value> rematCache;
      for (Value obs : waitOp.getObserved()) {
        auto remat = rematerializeValueForPolling(obs, rematCache, rewriter);
        if (failed(remat))
          return waitOp.emitOpError() << "could not rematerialize observed value";
        Value curObs = *remat;
        auto intTy = dyn_cast<IntegerType>(curObs.getType());
        if (!intTy || intTy.getWidth() > 64)
          return waitOp.emitOpError() << "unsupported observed value type";
        Value ext = curObs;
        if (intTy.getWidth() < 64)
          ext = comb::createZExt(rewriter, waitOp.getLoc(), curObs, 64);
        sig = comb::XorOp::create(rewriter, waitOp.getLoc(), sig, ext, true);
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
  return std::nullopt;
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

/// Best-effort lowering for "simple" LLHD signal semantics within a single
/// `llhd.process` (commonly used for `initial` blocks without delays). This
/// rewrites `llhd.prb`/`llhd.drv`/`llhd.sig.extract` to pure SSA updates within
/// the process region so that later conversions do not have to model inout
/// storage for these cases.
static LogicalResult lowerSimpleProcessSignals(llhd::ProcessOp proc) {
  if (!proc.getBody().hasOneBlock())
    return failure();

  Block &block = proc.getBody().front();
  if (!proc.getOps<llhd::WaitOp>().empty())
    return failure();

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
    uint64_t offset = 0;
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

    auto lowBit = getConstantLowBit(ex.getLowBit());
    if (!lowBit)
      return failure();

    auto inoutTy = dyn_cast<hw::InOutType>(ex.getResult().getType());
    if (!inoutTy)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(inoutTy.getElementType());
    if (!elemTy)
      return failure();

    aliasInfo[ex.getResult()] = {ex.getInput(), *lowBit,
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

  for (Value base : baseSignals) {
    auto it = signalOps.find(base);
    if (it == signalOps.end())
      return failure();

    for (OpOperand &use : base.getUses()) {
      if (!proc->isAncestor(use.getOwner()))
        return failure();
      if (!isa<llhd::PrbOp, llhd::DrvOp, llhd::SigExtractOp>(use.getOwner()))
        return failure();
    }
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
    auto sigIt = signalOps.find(base);
    if (sigIt == signalOps.end())
      return {};
    Value init = sigIt->second.getInit();
    current[base] = init;
    return init;
  };

  SmallVector<Operation *> toErase;
  OpBuilder builder(proc);
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
        replacement = builder.createOrFold<comb::ExtractOp>(
            loc, builder.getIntegerType(alias->second.width), baseCur,
            alias->second.offset);
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
        uint64_t offset = alias->second.offset;
        if (sliceWidth == 0 || bw == 0 || sliceWidth > bw ||
            offset + sliceWidth > bw)
          return failure();

        APInt sliceMask = APInt::getAllOnes(sliceWidth).zext(bw) << offset;
        APInt clearMask = APInt::getAllOnes(bw) ^ sliceMask;
        Value clearCst = hw::ConstantOp::create(
            builder, loc, builder.getIntegerAttr(baseTy, clearMask));
        Value cleared = builder.createOrFold<comb::AndOp>(loc, baseCur, clearCst);

        Value widened = value;
        if (bw > sliceWidth) {
          Value pad = hw::ConstantOp::create(
              builder, loc,
              builder.getIntegerAttr(builder.getIntegerType(bw - sliceWidth), 0));
          widened = builder.createOrFold<comb::ConcatOp>(loc, pad, widened);
        }

        Value shiftAmt =
            hw::ConstantOp::create(builder, loc, builder.getIntegerType(bw), offset);
        Value shifted = builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
        Value updated = builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
        current[base] = updated;
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
    uint64_t offset = 0;
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

    auto lowBit = getConstantLowBit(ex.getLowBit());
    if (!lowBit)
      return failure();

    auto inoutTy = dyn_cast<hw::InOutType>(ex.getResult().getType());
    if (!inoutTy)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(inoutTy.getElementType());
    if (!elemTy)
      return failure();

    aliasInfo[ex.getResult()] = {ex.getInput(), *lowBit,
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

  for (Value base : baseSignals) {
    auto it = signalOps.find(base);
    if (it == signalOps.end())
      return failure();

    for (OpOperand &use : base.getUses()) {
      if (!fin->isAncestor(use.getOwner()))
        return failure();
      if (!isa<llhd::PrbOp, llhd::DrvOp, llhd::SigExtractOp>(use.getOwner()))
        return failure();
    }
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
    auto sigIt = signalOps.find(base);
    if (sigIt == signalOps.end())
      return {};
    Value init = sigIt->second.getInit();
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
        replacement = builder.createOrFold<comb::ExtractOp>(
            loc, builder.getIntegerType(alias->second.width), baseCur,
            alias->second.offset);
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
        uint64_t offset = alias->second.offset;
        if (sliceWidth == 0 || bw == 0 || sliceWidth > bw ||
            offset + sliceWidth > bw)
          return failure();

        APInt sliceMask = APInt::getAllOnes(sliceWidth).zext(bw) << offset;
        APInt clearMask = APInt::getAllOnes(bw) ^ sliceMask;
        Value clearCst = hw::ConstantOp::create(
            builder, loc, builder.getIntegerAttr(baseTy, clearMask));
        Value cleared =
            builder.createOrFold<comb::AndOp>(loc, baseCur, clearCst);

        Value widened = value;
        if (bw > sliceWidth) {
          Value pad = hw::ConstantOp::create(
              builder, loc,
              builder.getIntegerAttr(builder.getIntegerType(bw - sliceWidth),
                                     0));
          widened = builder.createOrFold<comb::ConcatOp>(loc, pad, widened);
        }

        Value shiftAmt = hw::ConstantOp::create(builder, loc,
                                                builder.getIntegerType(bw),
                                                offset);
        Value shifted = builder.createOrFold<comb::ShlOp>(loc, widened, shiftAmt);
        Value updated = builder.createOrFold<comb::OrOp>(loc, cleared, shifted);
        current[base] = updated;
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
    return op->hasTrait<OpTrait::ConstantLike>();
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
    return inner->hasTrait<OpTrait::ConstantLike>();
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

  LogicalResult
  matchAndRewrite(llhd::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For scheduled simulation, treat some integer signals as runtime-managed
    // storage, using a stable integer id as the "signal handle" after type
    // conversion. This is a minimal model that enables cross-process variables
    // and events (lowered as integer bumps) without implementing full LLHD
    // signal semantics.
    auto sigIdAttr = op->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr);
    auto convertedTy =
        dyn_cast_or_null<IntegerType>(typeConverter->convertType(op.getType()));
    if (sigIdAttr && convertedTy && convertedTy.getWidth() >= 32 &&
        convertedTy.getWidth() <= 64) {
      uint64_t sigId = static_cast<uint64_t>(sigIdAttr.getInt());
      APInt sigIdBits(convertedTy.getWidth(), sigId);
      Value handle = hw::ConstantOp::create(rewriter, op.getLoc(), sigIdBits);
      if (auto cstOp = handle.getDefiningOp<hw::ConstantOp>())
        cstOp->setAttr(kArcilatorSigIdAttr, sigIdAttr);
      rewriter.replaceOp(op, handle);
      return success();
    }

    rewriter.replaceOp(op, adaptor.getInit());
    return success();
  }
};

/// `llhd.prb` -> pass-through (signal already converted to plain SSA)
struct ProbeOpConversion : public OpConversionPattern<llhd::PrbOp> {
  using OpConversionPattern<llhd::PrbOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::PrbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value handle = adaptor.getSignal();
    while (auto castOp =
               handle.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() != 1)
        break;
      handle = castOp.getInputs().front();
    }
    auto *defOp = handle.getDefiningOp();
    auto sigIdAttr =
        defOp ? defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)
              : IntegerAttr();
    if (!sigIdAttr) {
      rewriter.replaceOp(op, adaptor.getSignal());
      return success();
    }

    auto resultTy =
        dyn_cast_or_null<IntegerType>(typeConverter->convertType(op.getType()));
    if (!resultTy || resultTy.getWidth() > 64)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported probed signal type");

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
    if (handle.getType() == rewriter.getI32Type()) {
      sigIdVal = handle;
    } else if (auto sigIntTy = dyn_cast<IntegerType>(handle.getType())) {
      if (sigIntTy.getWidth() < 32)
        return rewriter.notifyMatchFailure(op,
                                           "signal handle is too narrow");
      sigIdVal = comb::ExtractOp::create(rewriter, op.getLoc(),
                                         rewriter.getI32Type(), handle, 0);
    } else if (isa<llhd::SignalOp>(defOp)) {
      sigIdVal = buildI32Constant(rewriter, op.getLoc(), sigIdAttr.getInt());
    } else {
      return rewriter.notifyMatchFailure(op, "signal handle is not an integer");
    }

    Value loaded =
        rewriter
            .create<mlir::func::CallOp>(op.getLoc(), "__arcilator_sig_load_u64",
                                        rewriter.getI64Type(),
                                        ValueRange{sigIdVal, procIdVal})
            .getResult(0);
    if (resultTy.getWidth() == 64) {
      rewriter.replaceOp(op, loaded);
      return success();
    }

    Value truncated = comb::ExtractOp::create(
        rewriter, op.getLoc(), rewriter.getIntegerType(resultTy.getWidth()),
        loaded, 0);
    rewriter.replaceOp(op, truncated);
    return success();
  }
};

/// `llhd.drv` -> best-effort runtime store (ignore precise delay/enable).
struct DrvOpConversion : public OpConversionPattern<llhd::DrvOp> {
  using OpConversionPattern<llhd::DrvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::DrvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value handle = adaptor.getSignal();
    while (auto castOp =
               handle.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() != 1)
        break;
      handle = castOp.getInputs().front();
    }
    auto *defOp = handle.getDefiningOp();
    auto sigIdAttr =
        defOp ? defOp->getAttrOfType<IntegerAttr>(kArcilatorSigIdAttr)
              : IntegerAttr();
    if (!sigIdAttr) {
      rewriter.eraseOp(op);
      return success();
    }

    auto valueTy = dyn_cast<IntegerType>(adaptor.getValue().getType());
    if (!valueTy || valueTy.getWidth() > 64)
      return rewriter.notifyMatchFailure(op, "unsupported driven value type");

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "missing module for runtime hook");

    (void)getOrInsertFunc(
        module, "__arcilator_sig_store_u64",
        rewriter.getFunctionType(
            {rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI32Type()},
            {}));

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
    if (handle.getType() == rewriter.getI32Type()) {
      sigIdVal = handle;
    } else if (auto sigIntTy = dyn_cast<IntegerType>(handle.getType())) {
      if (sigIntTy.getWidth() < 32)
        return rewriter.notifyMatchFailure(op,
                                           "signal handle is too narrow");
      sigIdVal = comb::ExtractOp::create(rewriter, op.getLoc(),
                                         rewriter.getI32Type(), handle, 0);
    } else if (isa<llhd::SignalOp>(defOp)) {
      sigIdVal = buildI32Constant(rewriter, op.getLoc(), sigIdAttr.getInt());
    } else {
      return rewriter.notifyMatchFailure(op, "signal handle is not an integer");
    }

    Value value64 = adaptor.getValue();
    if (valueTy.getWidth() < 64)
      value64 = comb::createZExt(rewriter, op.getLoc(), value64, 64);

    rewriter.create<mlir::func::CallOp>(op.getLoc(), "__arcilator_sig_store_u64",
                                        TypeRange{},
                                        ValueRange{sigIdVal, value64, procIdVal});
    rewriter.eraseOp(op);
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
    for (auto fin :
         llvm::make_early_inc_range(module.getOps<llhd::FinalOp>()))
      (void)lowerSimpleFinalSignals(fin);
    for (auto proc :
         llvm::make_early_inc_range(module.getOps<llhd::ProcessOp>())) {
      (void)lowerSimpleProcessSignals(proc);
      if (succeeded(sinkProcessResultDrives(proc)))
        continue;
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
        return inner->hasTrait<OpTrait::ConstantLike>() ||
               isa<llhd::SignalOp, llhd::PrbOp>(inner);
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
          if (inout.getElementType() == type)
            return builder
                .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
        }
        if (auto desiredInOut = dyn_cast<hw::InOutType>(type)) {
          if (desiredInOut.getElementType() == input.getType())
            return builder
                .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
        }
        return {};
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
  patterns.add<SignalOpConversion>(converter, &getContext());
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

  // Outline operations into arcs.
  Converter outliner;
  outliner.tapRegisters = tapRegisters;
  if (failed(outliner.run(getOperation())))
    return signalPassFailure();
}
