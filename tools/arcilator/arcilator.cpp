//===- arcilator.cpp - An experimental circuit simulator ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'arcilator' compiler, which converts HW designs into
// a corresponding LLVM-based software model.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/ModelInfoExport.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <functional>
#include <optional>

using namespace mlir;
using namespace circt;
using namespace arc;

namespace {
/// Bridge for "top-executed" SV testbenches that have no explicit ports.
///
/// Motivation: arcilator's cycle-driven pipeline intentionally drops LLHD
/// signal/process scheduling semantics. For `sv-tests` "module top" benches
/// that generate clocks/time internally (e.g. `#50 clk = ~clk;`) this makes
/// execution vacuous. In strict mode we also want clocked assertions to be
/// meaningful instead of forcing `--strip-verification`.
///
/// This pass performs a pragmatic, opt-in rewrite for no-port `hw.module @top`
/// testbenches:
///   - Lift `llhd.sig` named `clk`/`rst`/`reset` into new module input ports and
///     drop internal drives/processes that write them.
///   - Inline trivial `llhd.sig` wires that are driven once (epsilon) by an SSA
///     value and only ever probed, to keep the model purely SSA where possible.
///   - Hoist `verif.clocked_*` ops out of tight `llhd.process` loops so the Arc
///     state-lowering pass can lower them (posedge detection + terminate).
///
/// This is not a general SV scheduler; it is a bring-up bridge that makes
/// strict-mode assertion tests measurable while the proper M1 scheduler is
/// implemented.
struct MakeNoPortTopDriveablePass
    : public PassWrapper<MakeNoPortTopDriveablePass,
                         OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;

private:
  static bool isClockLikeName(StringRef name);
  static bool isResetLikeName(StringRef name);
  static Value cloneValueIntoBlock(Value value, Operation *scopeOp,
                                   OpBuilder &builder, IRMapping &mapping);
};

/// Work around unsupported dynamic strings for UVM reporting.
///
/// Some UVM tests call `uvm_info` with `$sformatf` messages (e.g. assertion
/// action blocks). This lowers to `sim.fmt.to_string` feeding a
/// `circt_uvm_report` call. Today, downstream LLHD normalization passes do not
/// preserve these dynamic-string calls reliably, which can drop report events
/// (affecting report counts exported via ports for VCD parity).
///
/// Until we have a robust runtime-string path through the LLHD→Arc pipeline,
/// degrade gracefully by keeping the report event but substituting an empty
/// string for the message. This preserves severity/id counters, which are the
/// primary oracle for UVM internal parity checking.
struct UvmReportMessageFallbackPass
    : public PassWrapper<UvmReportMessageFallbackPass,
                         OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    auto strTy = hw::StringType::get(ctx);

    SmallVector<sim::FormatToStringOp> maybeDeadToStrings;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() != "circt_uvm_report")
        return;
      if (call.getNumOperands() != 4)
        return;
      Value message = call.getOperand(3);
      if (!message)
        return;
      if (message.getDefiningOp<sv::ConstantStrOp>())
        return;

      if (auto toStr = message.getDefiningOp<sim::FormatToStringOp>())
        maybeDeadToStrings.push_back(toStr);

      OpBuilder builder(call);
      auto empty =
          builder.create<sv::ConstantStrOp>(call.getLoc(), strTy,
                                            builder.getStringAttr(""));
      call.setOperand(3, empty);
    });

    for (auto toStr : maybeDeadToStrings)
      if (toStr && toStr->use_empty())
        toStr.erase();
  }
};

bool MakeNoPortTopDriveablePass::isClockLikeName(StringRef name) {
  name = name.trim();
  return name == "clk" || name.ends_with(".clk") || name.ends_with("/clk");
}

bool MakeNoPortTopDriveablePass::isResetLikeName(StringRef name) {
  name = name.trim();
  return name == "rst" || name == "reset" || name.ends_with(".rst") ||
         name.ends_with(".reset") || name.ends_with("/rst") ||
         name.ends_with("/reset");
}

Value MakeNoPortTopDriveablePass::cloneValueIntoBlock(Value value,
                                                      Operation *scopeOp,
                                                      OpBuilder &builder,
                                                      IRMapping &mapping) {
  if (!value)
    return {};
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;

  // Values defined outside `scopeOp` can be referenced directly.
  if (auto barg = dyn_cast<BlockArgument>(value)) {
    mapping.map(value, barg);
    return barg;
  }

  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return {};
  if (!scopeOp->isAncestor(defOp)) {
    mapping.map(value, value);
    return value;
  }

  // Only clone simple, side-effect-free computation (plus LLHD probes).
  if (defOp->getNumRegions() != 0)
    return {};
  bool allowSideEffects = isa<llhd::PrbOp>(defOp);
  if (!defOp->hasTrait<OpTrait::ConstantLike>() && !isMemoryEffectFree(defOp) &&
      !allowSideEffects)
    return {};

  for (auto operand : defOp->getOperands()) {
    if (!cloneValueIntoBlock(operand, scopeOp, builder, mapping))
      return {};
  }

  Operation *cloned = builder.clone(*defOp, mapping);
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};
  auto clonedResult = cloned->getResult(result.getResultNumber());
  mapping.map(value, clonedResult);
  return clonedResult;
}

void MakeNoPortTopDriveablePass::runOnOperation() {
  auto module = getOperation();

  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    if (hwModule.getModuleName() != "top")
      continue;
    if (hwModule.getNumPorts() != 0)
      continue;

    auto *body = hwModule.getBodyBlock();
    auto *terminator = body->getTerminator();
    OpBuilder builder(terminator);

    // Collect signal ops we may want to lift.
    SmallVector<llhd::SignalOp> toExternalize;
    for (auto sig : hwModule.getOps<llhd::SignalOp>()) {
      auto nameAttr = sig->getAttrOfType<StringAttr>("name");
      if (!nameAttr)
        continue;
      auto name = nameAttr.getValue();
      auto inoutTy = dyn_cast<hw::InOutType>(sig.getResult().getType());
      if (!inoutTy || !inoutTy.getElementType().isInteger(1))
        continue;
      if (isClockLikeName(name) || isResetLikeName(name))
        toExternalize.push_back(sig);
    }

    DenseMap<Value, Value> liftedSignals;

    // Lift selected signals into module input ports (i1).
    for (auto sig : toExternalize) {
      auto nameAttr = sig->getAttrOfType<StringAttr>("name");
      if (!nameAttr)
        continue;

      auto [portName, portArg] = hwModule.insertInput(
          hwModule.getNumInputPorts(), nameAttr, builder.getI1Type());
      (void)portName;
      liftedSignals[sig.getResult()] = portArg;

      // Replace all probes of this signal with the new input.
      SmallVector<llhd::PrbOp> probes;
      for (auto *user : sig.getResult().getUsers())
        if (auto prb = dyn_cast<llhd::PrbOp>(user))
          probes.push_back(prb);
      for (auto prb : probes) {
        prb.replaceAllUsesWith(portArg);
        prb.erase();
      }

      // Drop any processes which were responsible for driving this signal.
      // This avoids leaving behind tight loops after the drives are erased.
      SmallVector<llhd::ProcessOp> procs;
      hwModule.walk([&](llhd::ProcessOp proc) {
        bool drivesLifted = false;
        proc.walk([&](llhd::DrvOp drv) {
          if (drv.getSignal() == sig.getResult())
            drivesLifted = true;
        });
        if (drivesLifted)
          procs.push_back(proc);
      });
      for (auto proc : procs)
        proc.erase();

      // Drop any drives into the signal (it is now an input).
      SmallVector<llhd::DrvOp> drives;
      hwModule.walk([&](llhd::DrvOp drv) {
        if (drv.getSignal() == sig.getResult())
          drives.push_back(drv);
      });
      for (auto drv : drives)
        drv.erase();

      if (sig.use_empty())
        sig.erase();
    }

    // Inline trivial signal wires: `sig` + single epsilon drive + only probes.
    SmallVector<llhd::SignalOp> signals;
    for (auto sig : llvm::make_early_inc_range(hwModule.getOps<llhd::SignalOp>()))
      signals.push_back(sig);

    for (auto sig : signals) {
      if (liftedSignals.contains(sig.getResult()))
        continue;

      // Collect drives/probes.
      SmallVector<llhd::DrvOp> drives;
      SmallVector<llhd::PrbOp> probes;
      for (auto *user : sig.getResult().getUsers()) {
        if (auto drv = dyn_cast<llhd::DrvOp>(user)) {
          if (drv.getSignal() == sig.getResult())
            drives.push_back(drv);
          continue;
        }
        if (auto prb = dyn_cast<llhd::PrbOp>(user)) {
          probes.push_back(prb);
          continue;
        }
        drives.clear();
        probes.clear();
        break;
      }
      if (drives.size() != 1 || probes.empty())
        continue;

      auto drv = drives.front();
      // Conservatively avoid inlining signals driven from within procedural
      // regions. Such drives may be stateful (e.g. `x = x + 1;`) and replacing
      // probes with the driven value can introduce cyclic SSA use.
      if (drv->getBlock() != body)
        continue;
      if (drv.getEnable())
        continue;

      // Only inline epsilon drives (treat as combinational wiring).
      auto cstTime = drv.getTime().getDefiningOp<llhd::ConstantTimeOp>();
      if (!cstTime)
        continue;
      auto t = cstTime.getValue();
      if (t.getTime() != 0 || t.getDelta() != 0 || t.getEpsilon() != 1)
        continue;

      Value driven = drv.getValue();
      for (auto prb : probes) {
        prb.replaceAllUsesWith(driven);
        prb.erase();
      }
      drv.erase();
      if (sig.use_empty())
        sig.erase();
    }

    // Hoist verif ops out of tight llhd.process loops (no waits).
    SmallVector<llhd::ProcessOp> procsToErase;
    for (auto proc :
         llvm::make_early_inc_range(hwModule.getOps<llhd::ProcessOp>())) {
      if (!proc.getResults().empty())
        continue;
      if (!proc.getOps<llhd::WaitOp>().empty())
        continue;

      SmallVector<Operation *> verifOps;
      proc.walk([&](Operation *op) {
        if (isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp,
                verif::ClockedAssertOp, verif::ClockedAssumeOp,
                verif::ClockedCoverOp>(op))
          verifOps.push_back(op);
      });
      if (verifOps.empty())
        continue;

      // Refuse to touch processes that do anything else stateful.
      bool hasOtherSideEffects = false;
      proc.walk([&](Operation *op) {
        if (op == proc.getOperation())
          return;
        if (isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp,
                verif::ClockedAssertOp, verif::ClockedAssumeOp,
                verif::ClockedCoverOp>(op))
          return;
        if (op->hasTrait<OpTrait::IsTerminator>())
          return;
        if (!isMemoryEffectFree(op) && !isa<llhd::PrbOp>(op))
          hasOtherSideEffects = true;
      });
      if (hasOtherSideEffects)
        continue;

      // Clone dependencies and the verif op itself into the module body.
      IRMapping mapping;
      bool canHoist = true;
      for (Operation *op : verifOps) {
        // Ensure operands are available in the module body.
        for (auto operand : op->getOperands()) {
          if (!cloneValueIntoBlock(operand, proc, builder, mapping)) {
            canHoist = false;
            break;
          }
        }
        if (!canHoist)
          break;
        builder.clone(*op, mapping);
      }
      if (!canHoist)
        continue;

      procsToErase.push_back(proc);
    }
    for (auto proc : procsToErase)
      proc.erase();
  }
}

/// Strip verification ops that are currently not supported by the arcilator
/// lowering pipeline. This allows us to accept SV sources that include
/// assertions/properties (including LTL-based ones) without failing the HW→Arc
/// conversion.
struct StripVerificationOpsPass
    : public PassWrapper<StripVerificationOpsPass,
                         OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    SmallVector<Operation *> verifOps;
    SmallVector<Operation *> ltlOps;

    getOperation().walk([&](Operation *op) {
      auto *dialect = op->getDialect();
      if (!dialect)
        return;
      auto dialectNS = dialect->getNamespace();
      if (dialectNS == "verif") {
        // `verif.simulation` is handled by `arc-lower-verif-simulations`.
        if (isa<verif::SimulationOp>(op))
          return;
        verifOps.push_back(op);
        return;
      }

      // Strip SV verification operations that may be present after various
      // conversions.
      if (isa<sv::AssertOp, sv::AssumeOp, sv::CoverOp, sv::AssertConcurrentOp,
              sv::AssumeConcurrentOp, sv::CoverConcurrentOp,
              sv::AssertPropertyOp, sv::AssumePropertyOp, sv::CoverPropertyOp>(
              op)) {
        verifOps.push_back(op);
        return;
      }

      if (dialectNS == "ltl") {
        ltlOps.push_back(op);
        return;
      }
    });

    // Erase assertion-like users first, then remove any remaining LTL ops.
    for (auto *op : llvm::reverse(verifOps))
      op->erase();
    for (auto *op : llvm::reverse(ltlOps)) {
      if (!op->use_empty()) {
        op->emitOpError()
            << "unexpected live uses of LTL op after stripping verification "
               "ops";
        return signalPassFailure();
      }
      op->erase();
    }
  }
};

/// Lower any remaining `verif.{assert,assume,cover}` operations to `sim.*` ops
/// so the Arc→LLVM lowering does not fail in strict mode.
///
/// In strict mode (`--strip-verification=false`) we keep `verif.*` ops in the
/// IR, but some can survive Arc state lowering nested inside `arc.execute`
/// regions. The Arc→LLVM conversion does not accept `verif.*` ops, so provide
/// a pragmatic lowering bridge here.
struct LowerVerifAssertLikeToSimPass
    : public PassWrapper<LowerVerifAssertLikeToSimPass,
                         OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();

    auto getI1Constant = [](OpBuilder &builder, Location loc,
                            bool value) -> Value {
      return hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                    static_cast<int64_t>(value))
          .getResult();
    };

    auto getU32FromConstant = [](Value value) -> std::optional<uint32_t> {
      if (!value)
        return std::nullopt;
      if (auto cst = value.getDefiningOp<hw::ConstantOp>()) {
        auto apInt = cst.getValue();
        if (apInt.getBitWidth() > 32)
          return std::nullopt;
        return static_cast<uint32_t>(apInt.getZExtValue());
      }
      return std::nullopt;
    };

    DenseMap<uint32_t, uint32_t> nextFrameSlot;
    uint32_t maxProcId = 0;
    module.walk([&](func::CallOp op) {
      auto callee = op.getCallee();
      if (callee != "__arcilator_frame_load_u64" &&
          callee != "__arcilator_frame_store_u64")
        return;
      if (op.getNumOperands() < 2)
        return;
      auto procId = getU32FromConstant(op.getOperand(0));
      auto slot = getU32FromConstant(op.getOperand(1));
      if (!procId || !slot)
        return;
      auto &next = nextFrameSlot[*procId];
      next = std::max(next, *slot + 1u);
      maxProcId = std::max(maxProcId, *procId);
    });

    module.walk([&](arc::ExecuteOp exec) {
      if (auto procIdAttr =
              exec->getAttrOfType<IntegerAttr>("arcilator.proc_id")) {
        maxProcId = std::max(
            maxProcId,
            static_cast<uint32_t>(procIdAttr.getValue().getZExtValue()));
      }
    });

    uint32_t monitorProcId = maxProcId + 1;
    if (monitorProcId == 0)
      monitorProcId = maxProcId;

    auto canCloneSideEffectCall = [](func::CallOp call) -> bool {
      if (!call)
        return false;
      auto callee = call.getCallee();
      return callee == "__arcilator_sig_load_u64" ||
             callee == "__arcilator_sig_load_nba_u64" ||
             callee == "__arcilator_sig_read_u64" ||
             callee == "__arcilator_frame_load_u64" ||
             callee == "__arcilator_now_fs";
    };

    std::function<Value(Value, Operation *, OpBuilder &, IRMapping &)>
        cloneValueIntoBlock = [&](Value value, Operation *scopeOp,
                                  OpBuilder &builder,
                                  IRMapping &mapping) -> Value {
      if (!value)
        return {};
      if (auto mapped = mapping.lookupOrNull(value))
        return mapped;

      // Values defined outside `scopeOp` can be referenced directly.
      if (auto barg = dyn_cast<BlockArgument>(value)) {
        mapping.map(value, barg);
        return barg;
      }

      auto *defOp = value.getDefiningOp();
      if (!defOp)
        return {};
      if (!scopeOp || !scopeOp->isAncestor(defOp)) {
        mapping.map(value, value);
        return value;
      }

      // Only clone simple computation. Allow cloning calls to known read-only
      // arcilator runtime hooks since they are safe to duplicate for hoisting.
      if (defOp->getNumRegions() != 0)
        return {};
      bool allowSideEffects = isa<llhd::PrbOp>(defOp);
      if (auto call = dyn_cast<func::CallOp>(defOp))
        allowSideEffects |= canCloneSideEffectCall(call);

      if (!defOp->hasTrait<OpTrait::ConstantLike>() && !isMemoryEffectFree(defOp) &&
          !allowSideEffects)
        return {};

      for (auto operand : defOp->getOperands()) {
        if (!cloneValueIntoBlock(operand, scopeOp, builder, mapping))
          return {};
      }

      Operation *cloned = builder.clone(*defOp, mapping);
      auto result = dyn_cast<OpResult>(value);
      if (!result)
        return {};
      auto clonedResult = cloned->getResult(result.getResultNumber());
      mapping.map(value, clonedResult);
      return clonedResult;
    };

    auto lowerUnclockedAssertLike = [&](auto op) {
      if (!op.getProperty().getType().isInteger(1)) {
        op.emitWarning()
            << "unsupported non-i1 verif assert-like property; dropping";
        op.erase();
        return;
      }

      OpBuilder builder(op);
      auto loc = op.getLoc();

      Value enable = op.getEnable();
      if (!enable)
        enable = getI1Constant(builder, loc, /*value=*/true);
      Value notProperty =
          comb::XorOp::create(builder, loc, op.getProperty(),
                              getI1Constant(builder, loc, /*value=*/true));
      Value failCond = comb::AndOp::create(builder, loc, enable, notProperty);

      auto ifFail =
          scf::IfOp::create(builder, loc, failCond, /*withElse=*/false);
      builder.setInsertionPoint(ifFail.thenYield());
      sim::TerminateOp::create(builder, loc, /*success=*/false,
                               /*verbose=*/false);

      op.erase();
    };

    auto lowerClockedAssertLike = [&](auto op) {
      if (op.getEdge() != verif::ClockEdge::Pos &&
          op.getEdge() != verif::ClockEdge::Neg &&
          op.getEdge() != verif::ClockEdge::Both) {
        op.emitWarning() << "unsupported verif clock edge; dropping";
        op.erase();
        return;
      }

      if (!op.getProperty().getType().isInteger(1)) {
        op.emitWarning()
            << "unsupported non-i1 verif assert-like property; dropping";
        op.erase();
        return;
      }

      auto modelOp = op->template getParentOfType<arc::ModelOp>();
      if (!modelOp) {
        op.emitWarning()
            << "clocked verif op not in arcilator model; dropping";
        op.erase();
        return;
      }

      auto execOp = op->template getParentOfType<arc::ExecuteOp>();
      uint32_t slot = nextFrameSlot[monitorProcId]++;

      Block &modelBlock = modelOp.getBody().front();
      OpBuilder builder(&modelBlock, modelBlock.end());
      auto loc = op.getLoc();

      IRMapping mapping;
      Value property =
          cloneValueIntoBlock(op.getProperty(), execOp, builder, mapping);
      Value clock = cloneValueIntoBlock(op.getClock(), execOp, builder, mapping);
      Value clonedEnable;
      if (op.getEnable())
        clonedEnable =
            cloneValueIntoBlock(op.getEnable(), execOp, builder, mapping);

      if (!property || !clock || (op.getEnable() && !clonedEnable)) {
        op.emitWarning() << "unable to hoist clocked verif op operands; dropping";
        op.erase();
        return;
      }

      Value procIdVal = hw::ConstantOp::create(builder, loc, builder.getI32Type(),
                                               monitorProcId);
      Value slotVal =
          hw::ConstantOp::create(builder, loc, builder.getI32Type(), slot);

      // Load `{valid, oldClock}` from a per-process frame slot.
      auto frameLoad = FlatSymbolRefAttr::get(builder.getContext(),
                                              "__arcilator_frame_load_u64");
      auto frameStore = FlatSymbolRefAttr::get(builder.getContext(),
                                               "__arcilator_frame_store_u64");

      Value oldPacked =
          func::CallOp::create(builder, loc, frameLoad, builder.getI64Type(),
                               ValueRange{procIdVal, slotVal})
              .getResult(0);
      Value oldClock =
          comb::ExtractOp::create(builder, loc, oldPacked, /*lowBit=*/0,
                                  /*bitWidth=*/1);
      Value oldValid =
          comb::ExtractOp::create(builder, loc, oldPacked, /*lowBit=*/1,
                                  /*bitWidth=*/1);

      if (isa<seq::ClockType>(clock.getType()))
        clock = seq::FromClockOp::create(builder, loc, clock);

      // Update the stored previous clock value and mark the slot valid. The
      // initial value of the frame slot is zero, so the valid bit prevents a
      // false edge trigger at time 0 when the clock starts high.
      Value zero62 =
          hw::ConstantOp::create(builder, loc, builder.getIntegerType(62), 0);
      Value validBit = getI1Constant(builder, loc, /*value=*/true);
      Value newPacked =
          comb::ConcatOp::create(builder, loc,
                                 ValueRange{zero62, validBit, clock});
      func::CallOp::create(builder, loc, frameStore, TypeRange{},
                           ValueRange{procIdVal, slotVal, newPacked});

      // Detect clock edge.
      Value edge = comb::XorOp::create(builder, loc, oldClock, clock);
      Value event;
      switch (op.getEdge()) {
      case verif::ClockEdge::Pos:
        event = comb::AndOp::create(builder, loc, edge, clock);
        break;
      case verif::ClockEdge::Neg: {
        Value notClock =
            comb::XorOp::create(builder, loc, clock,
                                getI1Constant(builder, loc, /*value=*/true));
        event = comb::AndOp::create(builder, loc, edge, notClock);
        break;
      }
      case verif::ClockEdge::Both:
        event = edge;
        break;
      }
      event = comb::AndOp::create(builder, loc, oldValid, event);

      Value enable = clonedEnable;
      if (!enable)
        enable = getI1Constant(builder, loc, /*value=*/true);

      Value notProperty =
          comb::XorOp::create(builder, loc, property,
                              getI1Constant(builder, loc, /*value=*/true));

      Value gated =
          comb::AndOp::create(builder, loc, event, enable);
      Value failCond =
          comb::AndOp::create(builder, loc, gated, notProperty);

      auto ifFail =
          scf::IfOp::create(builder, loc, failCond, /*withElse=*/false);
      builder.setInsertionPoint(ifFail.thenYield());
      sim::TerminateOp::create(builder, loc, /*success=*/false,
                               /*verbose=*/false);

      op.erase();
    };

    // Collect first to avoid invalidating the walk while rewriting.
    SmallVector<verif::AssertOp> asserts;
    SmallVector<verif::AssumeOp> assumes;
    SmallVector<verif::CoverOp> covers;
    SmallVector<verif::ClockedAssertOp> clockedAsserts;
    SmallVector<verif::ClockedAssumeOp> clockedAssumes;
    SmallVector<verif::ClockedCoverOp> clockedCovers;

    module.walk([&](verif::AssertOp op) { asserts.push_back(op); });
    module.walk([&](verif::AssumeOp op) { assumes.push_back(op); });
    module.walk([&](verif::CoverOp op) { covers.push_back(op); });
    module.walk(
        [&](verif::ClockedAssertOp op) { clockedAsserts.push_back(op); });
    module.walk(
        [&](verif::ClockedAssumeOp op) { clockedAssumes.push_back(op); });
    module.walk(
        [&](verif::ClockedCoverOp op) { clockedCovers.push_back(op); });

    for (auto op : asserts)
      lowerUnclockedAssertLike(op);
    for (auto op : assumes)
      lowerUnclockedAssertLike(op);

    // Coverage is not currently surfaced in simulation; drop.
    for (auto op : covers)
      op.erase();

    for (auto op : clockedAsserts)
      lowerClockedAssertLike(op);
    for (auto op : clockedAssumes)
      lowerClockedAssertLike(op);
    for (auto op : clockedCovers)
      op.erase();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory mainCategory("arcilator Options");

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"),
                                                llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observePorts("observe-ports", llvm::cl::desc("Make all ports observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeWires("observe-wires", llvm::cl::desc("Make all wires observable"),
                 llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> observeNamedValues(
    "observe-named-values",
    llvm::cl::desc("Make values with `sv.namehint` observable"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeRegisters("observe-registers",
                     llvm::cl::desc("Make all registers observable"),
                     llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    observeMemories("observe-memories",
                    llvm::cl::desc("Make all memory contents observable"),
                    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string> stateFile("state-file",
                                            llvm::cl::desc("State file"),
                                            llvm::cl::value_desc("filename"),
                                            llvm::cl::init(""),
                                            llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldInline("inline", llvm::cl::desc("Inline arcs"),
                                        llvm::cl::init(true),
                                        llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDedup("dedup",
                                       llvm::cl::desc("Deduplicate arcs"),
                                       llvm::cl::init(true),
                                       llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectEnables(
    "detect-enables",
    llvm::cl::desc("Infer enable conditions for states to avoid computation"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> shouldDetectResets(
    "detect-resets",
    llvm::cl::desc("Infer reset conditions for states to avoid computation"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    stripVerificationOps("strip-verification",
                         llvm::cl::desc("Strip verif/ltl/SV assertion ops that "
                                        "are not supported by the arcilator "
                                        "lowering pipeline"),
                         llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    shouldMakeLUTs("lookup-tables",
                   llvm::cl::desc("Optimize arcs into lookup tables"),
                   llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool>
    printDebugInfo("print-debug-info",
                   llvm::cl::desc("Print debug information"),
                   llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> verbosePassExecutions(
    "verbose-pass-executions",
    llvm::cl::desc("Log executions of toplevel module passes"),
    llvm::cl::init(false), llvm::cl::cat(mainCategory));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(mainCategory));

static llvm::cl::opt<unsigned> splitFuncsThreshold(
    "split-funcs-threshold",
    llvm::cl::desc(
        "Split large MLIR functions that occur above the given size threshold"),
    llvm::cl::ValueOptional, llvm::cl::cat(mainCategory));

// Options to control early-out from pipeline.
enum Until {
  UntilPreprocessing,
  UntilArcConversion,
  UntilArcOpt,
  UntilStateLowering,
  UntilStateAlloc,
  UntilLLVMLowering,
  UntilEnd
};
static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilPreprocessing, "preproc", "Input preprocessing"),
    clEnumValN(UntilArcConversion, "arc-conv", "Conversion of modules to arcs"),
    clEnumValN(UntilArcOpt, "arc-opt", "Arc optimizations"),
    clEnumValN(UntilStateLowering, "state-lowering", "Stateful arc lowering"),
    clEnumValN(UntilStateAlloc, "state-alloc", "State allocation"),
    clEnumValN(UntilLLVMLowering, "llvm-lowering", "Lowering to LLVM"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));
static llvm::cl::opt<Until> runUntilBefore(
    "until-before", llvm::cl::desc("Stop pipeline before a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));
static llvm::cl::opt<Until> runUntilAfter(
    "until-after", llvm::cl::desc("Stop pipeline after a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));

// Options to control the output format.
enum OutputFormat { OutputMLIR, OutputLLVM, OutputRunJIT, OutputDisabled };
static llvm::cl::opt<OutputFormat> outputFormat(
    llvm::cl::desc("Specify output format"),
    llvm::cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit MLIR dialects"),
                     clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
                     clEnumValN(OutputRunJIT, "run",
                                "Run the simulation and emit its output"),
                     clEnumValN(OutputDisabled, "disable-output",
                                "Do not output anything")),
    llvm::cl::init(OutputLLVM), llvm::cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    jitEntryPoint("jit-entry",
                  llvm::cl::desc("Name of the function containing the "
                                 "simulation to run when output is set to run"),
                  llvm::cl::init("entry"), llvm::cl::cat(mainCategory));

static llvm::cl::list<std::string> sharedLibs{
    "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
    llvm::cl::MiscFlags::CommaSeparated, llvm::cl::cat(mainCategory)};

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

/// Populate a pass manager with the arc simulator pipeline for the given
/// command line options. This pipeline lowers modules to the Arc dialect.
static void populateHwModuleToArcPipeline(PassManager &pm, bool inputHasMoore) {
  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator"));

  // Pre-process the input such that it no longer contains any SV dialect ops
  // and external modules that are relevant to the arc transformation are
  // represented as intrinsic ops.
  if (untilReached(UntilPreprocessing))
    return;
  // In strict mode (`--strip-verification=false`), keep `verif.*` ops in the IR
  // and lower supported assertions later during Arc state lowering.
  if (inputHasMoore) {
    pm.addPass(moore::createInlineCallsPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(createConvertMooreToCorePass());
  }
  pm.addPass(std::make_unique<UvmReportMessageFallbackPass>());
  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(createLowerFirMemPass());
  pm.addPass(createLowerVerifSimulationsPass());
  pm.addPass(sv::createLowerInterfacesPass());
  {
    // Normalize LLHD procedural constructs into structural form so the Arc
    // conversion sees SSA values instead of signal/process primitives.
    pm.addPass(llhd::createProcessLowering());
    pm.nest<hw::HWModuleOp>().addPass(llhd::createEarlyCodeMotion());
    // Temporal code motion is sensitive to how many distinct delay constants
    // show up in a process. CSE helps collapse duplicate `llhd.constant_time`
    // ops so the temporal region analysis can succeed on typical always blocks.
    pm.nest<hw::HWModuleOp>().addPass(createCSEPass());
    pm.nest<hw::HWModuleOp>().addPass(llhd::createTemporalCodeMotion());
    // Promote stack-like memory (`llhd.var` / `llhd.load` / `llhd.store`) to SSA
    // values early so later LLHD and Arc conversions don't have to reason about
    // pointer-typed state.
    pm.nest<hw::HWModuleOp>().addPass(llhd::createMemoryToBlockArgument());
    pm.addPass(llhd::createHoistSignalsPass());
    // Hoist signal accesses (and promote hoistable drives to process results)
    // before lowering processes to combinational ops. Otherwise, drives end up
    // trapped inside `llhd.combinational` regions where downstream promotions
    // (e.g. sig2reg) cannot see them, which can incorrectly collapse outputs to
    // their initial values.
    pm.nest<hw::HWModuleOp>().addPass(llhd::createLowerProcessesPass());
    pm.nest<hw::HWModuleOp>().addPass(llhd::createDeseqPass());
    pm.nest<hw::HWModuleOp>().addPass(llhd::createCombineDrivesPass());
    pm.nest<hw::HWModuleOp>().addPass(llhd::createSig2Reg());
  }
  if (!stripVerificationOps)
    pm.addPass(std::make_unique<MakeNoPortTopDriveablePass>());
  {
    arc::AddTapsOptions opts;
    opts.tapPorts = observePorts;
    opts.tapWires = observeWires;
    opts.tapNamedValues = observeNamedValues;
    pm.addPass(arc::createAddTapsPass(opts));
  }
  pm.addPass(arc::createStripSVPass());
  {
    arc::InferMemoriesOptions opts;
    opts.tapPorts = observePorts;
    opts.tapMemories = observeMemories;
    pm.addPass(arc::createInferMemoriesPass(opts));
  }
  pm.addPass(sim::createLowerDPIFunc());
  if (stripVerificationOps)
    pm.addPass(std::make_unique<StripVerificationOpsPass>());

  // Restructure the input from a `hw.module` hierarchy to a collection of arcs.
  if (untilReached(UntilArcConversion))
    return;
  // Flatten the HW module hierarchy before converting to arcs. This avoids
  // passing reference-like values (e.g. interface bundles lowered to
  // `hw.inout<struct>`) through module ports, which can otherwise leave behind
  // unresolved LLHD probe materializations during Arc conversion.
  pm.addPass(hw::createFlattenModules());
  pm.addPass(createCSEPass());
  {
    ConvertToArcsPassOptions opts;
    opts.tapRegisters = observeRegisters;
    pm.addPass(createConvertToArcsPass(opts));
  }
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  pm.addPass(hw::createFlattenModules());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Perform arc-level optimizations that are not specific to software
  // simulation.
  if (untilReached(UntilArcOpt))
    return;
  pm.addPass(arc::createSplitLoopsPass());
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  {
    arc::InferStatePropertiesOptions opts;
    opts.detectEnables = shouldDetectEnables;
    opts.detectResets = shouldDetectResets;
    pm.addPass(arc::createInferStateProperties(opts));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
  pm.addNestedPass<hw::HWModuleOp>(arc::createMergeTaps());
  if (shouldMakeLUTs)
    pm.addPass(arc::createMakeTablesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Now some arguments may be unused because reset conditions are not passed as
  // inputs anymore pm.addPass(arc::createRemoveUnusedArcArgumentsPass());
  // Because we replace a lot of StateOp inputs with constants in the enable
  // patterns we may be able to sink a lot of them
  // TODO: maybe merge RemoveUnusedArcArguments with SinkInputs?
  // pm.addPass(arc::createSinkInputsPass());
  // pm.addPass(createCSEPass());
  // pm.addPass(createSimpleCanonicalizerPass());
  // Removing some muxes etc. may lead to additional dedup opportunities
  // if (shouldDedup)
  // pm.addPass(arc::createDedupPass());

  // Lower stateful arcs into explicit state reads and writes.
  if (untilReached(UntilStateLowering))
    return;
  pm.addPass(arc::createLowerStatePass());

  // TODO: LowerClocksToFuncsPass might not properly consider scf.if operations
  // (or nested regions in general) and thus errors out when muxes are also
  // converted in the hw.module or arc.model
  // TODO: InlineArcs seems to not properly handle scf.if operations, thus the
  // following is commented out
  // pm.addPass(arc::createMuxToControlFlowPass());
  if (shouldInline)
    pm.addPass(arc::createInlineArcsPass());

  pm.addPass(arc::createMergeIfsPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Allocate states.
  if (untilReached(UntilStateAlloc))
    return;
  pm.addPass(arc::createLowerArcsToFuncsPass());
  pm.nest<arc::ModelOp>().addPass(arc::createAllocateStatePass());
  pm.addPass(arc::createLowerClocksToFuncsPass()); // no CSE between state alloc
                                                   // and clock func lowering
  if (splitFuncsThreshold.getNumOccurrences()) {
    pm.addPass(arc::createSplitFuncs({splitFuncsThreshold}));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

/// Populate a pass manager with the Arc to LLVM pipeline for the given
/// command line options. This pipeline lowers modules to LLVM IR.
static void populateArcToLLVMPipeline(PassManager &pm) {
  // Lower the arcs and update functions to LLVM.
  if (untilReached(UntilLLVMLowering))
    return;
  if (!stripVerificationOps)
    pm.addPass(std::make_unique<LowerVerifAssertLikeToSimPass>());
  pm.addPass(createLowerArcToLLVMPass());
  pm.addPass(sim::createLowerSimConsole());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  }
  if (!module)
    return failure();

  // Lower HwModule to Arc model.
  PassManager pmArc(&context);
  pmArc.enableVerifier(verifyPasses);
  pmArc.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pmArc)))
    return failure();
  bool inputHasMoore = false;
  module->walk([&](Operation *op) {
    if (auto *dialect = op->getDialect())
      inputHasMoore |= dialect->getNamespace() == "moore";
  });
  populateHwModuleToArcPipeline(pmArc, inputHasMoore);

  if (failed(pmArc.run(module.get())))
    return failure();

  // Output state info as JSON if requested.
  if (!stateFile.empty() && !untilReached(UntilStateLowering)) {
    std::error_code ec;
    llvm::ToolOutputFile outputFile(stateFile, ec,
                                    llvm::sys::fs::OpenFlags::OF_None);
    if (ec) {
      llvm::errs() << "unable to open state file: " << ec.message() << '\n';
      return failure();
    }
    if (failed(collectAndExportModelInfo(module.get(), outputFile.os()))) {
      llvm::errs() << "failed to collect model info\n";
      return failure();
    }

    outputFile.keep();
  }

  // Lower Arc model to LLVM IR.
  PassManager pmLlvm(&context);
  pmLlvm.enableVerifier(verifyPasses);
  pmLlvm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pmLlvm)))
    return failure();
  if (verbosePassExecutions)
    pmLlvm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "arcilator"));
  populateArcToLLVMPipeline(pmLlvm);

  if (printDebugInfo && outputFormat == OutputLLVM)
    pmLlvm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());

  if (failed(pmLlvm.run(module.get())))
    return failure();

#ifdef ARCILATOR_ENABLE_JIT
  // Handle JIT execution.
  if (outputFormat == OutputRunJIT) {
    if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd) {
      llvm::errs() << "full pipeline must be run for JIT execution\n";
      return failure();
    }

    Operation *toCall = module->lookupSymbol(jitEntryPoint);
    if (!toCall) {
      llvm::errs() << "entry point not found: '" << jitEntryPoint << "'\n";
      return failure();
    }

    auto toCallFunc = llvm::dyn_cast<LLVM::LLVMFuncOp>(toCall);
    if (!toCallFunc) {
      llvm::errs() << "entry point '" << jitEntryPoint
                   << "' was found but on an operation of type '"
                   << toCall->getName()
                   << "' while an LLVM function was expected\n";
      return failure();
    }

    if (toCallFunc.getNumArguments() != 0) {
      llvm::errs() << "entry point '" << jitEntryPoint
                   << "' must have no arguments\n";
      return failure();
    }

    SmallVector<StringRef, 4> sharedLibraries(sharedLibs.begin(),
                                              sharedLibs.end());

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
    std::function<llvm::Error(llvm::Module *)> transformer =
        mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    engineOptions.transformer = transformer;
    engineOptions.sharedLibPaths = sharedLibraries;

    auto executionEngine =
        mlir::ExecutionEngine::create(module.get(), engineOptions);
    if (!executionEngine) {
      llvm::handleAllErrors(
          executionEngine.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to create execution engine: "
                         << info.message() << "\n";
          });
      return failure();
    }

    auto expectedFunc = (*executionEngine)->lookupPacked(jitEntryPoint);
    if (!expectedFunc) {
      llvm::handleAllErrors(
          expectedFunc.takeError(), [](const llvm::ErrorInfoBase &info) {
            llvm::errs() << "failed to run simulation: " << info.message()
                         << "\n";
          });
      return failure();
    }

    void (*simulationFunc)(void **) = *expectedFunc;
    (*simulationFunc)(nullptr);

    return success();
  }
#endif // ARCILATOR_ENABLE_JIT

  // Handle MLIR output.
  if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd ||
      outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    // Only set the debug info flag to true in order to not overwrite MLIR
    // printer CLI flags when the custom debug info option is not set.
    if (printDebugInfo)
      printingFlags.enableDebugInfo(printDebugInfo);
    auto outputTimer = ts.nest("Print MLIR output");
    module->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  // Handle LLVM output.
  if (outputFormat == OutputLLVM) {
    auto outputTimer = ts.nest("Print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    return success();
  }

  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<llvm::MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeArcilator(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    arc::ArcDialect,
    comb::CombDialect,
    emit::EmitDialect,
    hw::HWDialect,
    llhd::LLHDDialect,
    ltl::LTLDialect,
    moore::MooreDialect,
    mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect,
    mlir::DLTIDialect,
    mlir::func::FuncDialect,
    mlir::index::IndexDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect,
    om::OMDialect,
    seq::SeqDialect,
    sim::SimDialect,
    sv::SVDialect,
    verif::VerifDialect
  >();
  // clang-format on

  arc::initAllExternalInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

/// Main driver for the command. This sets up LLVM and MLIR, and parses command
/// line options before passing off to 'executeArcilator'. This is set up so we
/// can `exit(0)` at the end of the program to avoid teardown of the MLIRContext
/// and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  llvm::cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();

    // Dialect passes:
    arc::registerPasses();
    registerConvertToArcsPass();
    registerLowerArcToLLVMPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  llvm::cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR-based circuit simulator\n");

  if (outputFormat == OutputRunJIT) {
#ifdef ARCILATOR_ENABLE_JIT
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
#else
    llvm::errs() << "This arcilator binary was not built with JIT support.\n";
    llvm::errs() << "To enable JIT features, build arcilator with MLIR's "
                    "execution engine.\n";
    llvm::errs() << "This can be achieved by building arcilator with the "
                    "host's LLVM target enabled.\n";
    exit(1);
#endif // ARCILATOR_ENABLE_JIT
  }

  MLIRContext context;
  auto result = executeArcilator(context);

  // Use "exit" instead of returning to signal completion. This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
