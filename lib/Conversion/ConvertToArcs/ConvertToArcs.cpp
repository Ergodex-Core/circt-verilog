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
  executeOp.getBody().takeBody(op.getBody());
  TypeConverter::SignatureConversion signature(convertedOperands.size());
  for (auto [idx, operand] : llvm::enumerate(convertedOperands))
    signature.addInputs(idx, operand.getType());
  if (!rewriter.applySignatureConversion(&executeOp.getBody().front(),
                                         signature, &converter))
    return failure();
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
  executeOp.getBody().takeBody(op.getBody());
  TypeConverter::SignatureConversion signature(convertedOperands.size());
  for (auto [idx, operand] : llvm::enumerate(convertedOperands))
    signature.addInputs(idx, operand.getType());
  if (!rewriter.applySignatureConversion(&executeOp.getBody().front(),
                                         signature, &converter))
    return failure();
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
    rewriter.replaceOp(op, adaptor.getSignal());
    return success();
  }
};

/// `llhd.drv` -> pass-through of the driven value (ignore delay/enable for now)
struct DrvOpConversion : public OpConversionPattern<llhd::DrvOp> {
  using OpConversionPattern<llhd::DrvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(llhd::DrvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // `llhd.drv` has no SSA results; the current LLHD shims ignore scheduling
    // and treat drives as side effects on the corresponding `llhd.sig`. Until
    // we have a proper LLHD-to-Arc lowering that models signal storage, simply
    // erase the drive during type legalization.
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
    if (!lowBit)
      return rewriter.notifyMatchFailure(op, "non-constant low bit");

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

    rewriter.replaceOpWithNewOp<comb::ExtractOp>(op, outTy, inputVal, *lowBit);
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
    for (auto proc :
         llvm::make_early_inc_range(module.getOps<llhd::ProcessOp>())) {
      (void)lowerSimpleProcessSignals(proc);
      (void)convertOneShotProcessToInitial(proc);
    }
  });

  // NOTE: LLHD `llhd.process` pre-lowering is currently disabled. The existing
  // heuristic lowering to `arc.state` is incomplete and can lead to non-
  // terminating type legalization. A proper scheduler/stateful lowering should
  // live in LLHD transforms (e.g. `llhd-deseq`) or be implemented here as a
  // dedicated conversion.

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
  patterns.add<ConstantTimeOpConversion>(converter, &getContext());
  patterns.add<SignalOpConversion>(converter, &getContext());
  patterns.add<ProbeOpConversion>(converter, &getContext());
  patterns.add<DrvOpConversion>(converter, &getContext());
  patterns.add<SigExtractOpConversion>(converter, &getContext());
  patterns.add<WaitOpConversion>(converter, &getContext());
  patterns.add<HaltOpConversion>(converter, &getContext());

  // Setup the legal ops. (Sort alphabetically.)
  ConversionTarget target(getContext());
  target.addIllegalDialect<llhd::LLHDDialect>();
  target.addIllegalOp<llhd::DrvOp, llhd::ProcessOp, llhd::SignalOp,
                      llhd::WaitOp, llhd::PrbOp>();
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
