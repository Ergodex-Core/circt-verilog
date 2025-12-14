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
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

static bool isInProceduralRegion(Operation *op) {
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (parent->hasTrait<sv::ProceduralRegion>() ||
        parent->hasTrait<circt::llhd::ProceduralRegion>() ||
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
  void runOnOperation() override;

  FailureOr<InterfaceInfo *> lookupInterfaceInfo(InterfaceType type,
                                                 hw::HWModuleOp mod);
  FailureOr<InterfaceInfo *> lookupInterfaceInfo(ModportType type,
                                                 hw::HWModuleOp mod,
                                                 StringAttr &modportName);

  DenseMap<StringAttr, InterfaceInfo> interfaces;
  DenseMap<Operation *, SmallVector<InterfacePortPlan>> plans;
  bool encounteredFailure = false;

private:
  LogicalResult populateInterfaceInfo(ModuleOp module);
  LogicalResult buildPlans(hw::InstanceGraph &graph);
  LogicalResult lowerModules(hw::InstanceGraph &graph);
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
  Type loweredType = plan.needsInOut
                         ? Type(hw::InOutType::get(plan.loweredType))
                         : Type(plan.loweredType);
  if (instValue.getType() != loweredType) {
    inst->emitOpError("expected operand type ")
        << loweredType << " after interface lowering, got "
        << instValue.getType();
    pass.encounteredFailure = true;
    return;
  }
  newOperands[loweredPort.argNum] = instValue;
}

void InterfacePortConversion::mapOutputSignals(
    OpBuilder &, Operation *, Value instValue, SmallVectorImpl<Value> &,
    ArrayRef<Backedge> newResults) {
  if (!plan.info)
    return;
  instValue.replaceAllUsesWith(newResults[loweredPort.argNum]);
}

LogicalResult InterfacePortConversion::lowerInterfaceValue(
    Value oldValue, Value loweredBase, StringAttr directModport) {
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
        if (failed(verifyReadableSignal(plan.info, plan.modportName, fieldAttr,
                                        read)) ||
            failed(verifyReadableSignal(plan.info, modportName, fieldAttr,
                                        read)))
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
          replacement = builder.create<sv::ReadInOutOp>(read.getType(),
                                                        fieldHandle);
        } else if (llvm::isa<hw::StructType>(baseValue.getType())) {
          replacement =
              builder.create<hw::StructExtractOp>(baseValue, fieldAttr);
        } else {
          return read.emitOpError()
                 << "expected lowered interface value to be a struct or "
                    "inout struct, got "
                 << baseValue.getType();
        }
        read.replaceAllUsesWith(replacement);
        read.erase();
        continue;
      }

      if (auto assign = dyn_cast<sv::AssignInterfaceSignalOp>(user)) {
        StringAttr fieldAttr = assign.getSignalNameAttr().getAttr();
        if (failed(verifyWritableSignal(plan.info, plan.modportName, fieldAttr,
                                        assign)) ||
            failed(verifyWritableSignal(plan.info, modportName, fieldAttr,
                                        assign)))
          return failure();
        auto inoutType = dyn_cast<hw::InOutType>(baseValue.getType());
        if (!inoutType || !llvm::isa<hw::StructType>(inoutType.getElementType()))
          return assign.emitOpError()
                 << "cannot assign to flattened interface port because it is "
                    "not lowered to an inout";
        ImplicitLocOpBuilder builder(assign.getLoc(), assign);
        Value fieldHandle =
            builder.create<sv::StructFieldInOutOp>(baseValue, fieldAttr);
        if (isInProceduralRegion(assign))
          builder.create<sv::BPAssignOp>(fieldHandle, assign.getRhs());
        else
          builder.create<sv::AssignOp>(fieldHandle, assign.getRhs());
        assign.erase();
        continue;
      }

      return user->emitOpError("unsupported interface use during lowering");
    }
  }

  for (Operation *op : toErase)
    op->erase();
  return success();
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

void LowerInterfacesPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  if (failed(populateInterfaceInfo(module)))
    return signalPassFailure();
  if (failed(buildPlans(instanceGraph)))
    return signalPassFailure();
  if (failed(lowerModules(instanceGraph)))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}
