//===- LowerInterfaces.cpp - Minimal SV interface lowering
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass currently implements the first slice of interface lowering used by
// the nostub UVM bring-up: it rewrites leaf HW modules so that interface-typed
// *input* ports become plain HW struct ports.  The conversion deliberately
// fails if interface-typed outputs or interface operations appear in a body.
// This keeps the current scope manageable while we bring the rest of the IR
// into parity with the Arc pipeline.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"

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

struct InterfaceInfo {
  InterfaceOp op;
  SmallVector<hw::StructType::FieldInfo> fields;
  DenseMap<StringRef, unsigned> fieldIndex;
  hw::StructType getStructType() const {
    return hw::StructType::get(op.getContext(), fields);
  }
};

struct LowerInterfacesPass
    : public circt::sv::impl::LowerInterfacesBase<LowerInterfacesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(populateInterfaceInfo(module)))
      return signalPassFailure();

    bool changed = false;
    for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
      bool moduleChanged = false;
      if (failed(rewriteModule(hwMod, moduleChanged)))
        return signalPassFailure();
      changed |= moduleChanged;
    }

    if (!changed)
      markAllAnalysesPreserved();
  }

private:
  LogicalResult populateInterfaceInfo(ModuleOp module) {
    for (auto iface : module.getOps<InterfaceOp>()) {
      InterfaceInfo info;
      info.op = iface;
      for (auto sig : iface.getOps<InterfaceSignalOp>()) {
        hw::StructType::FieldInfo field;
        field.name = sig.getSymNameAttr();
        field.type = sig.getType();
        info.fieldIndex[field.name.getValue()] = info.fields.size();
        info.fields.push_back(field);
      }
      auto [it, inserted] =
          interfaces.try_emplace(iface.getSymNameAttr(), std::move(info));
      if (!inserted)
        return iface.emitOpError()
               << "duplicate interface definition named " << iface.getSymName();
    }
    return success();
  }

  LogicalResult rewriteModule(hw::HWModuleOp mod, bool &changed) {
    ModulePortInfo ports = mod.getPorts();
    SmallVector<PortInfo> newInputs, newOutputs;
    bool needsChange = false;
    DenseMap<Value, InterfaceInfo *> ifaceValueInfo;

    for (const auto &port : ports.getInputs()) {
      auto updated = port;
      InterfaceInfo *ifaceInfo = nullptr;
      Type newType = convertInterfaceType(port.type, port, mod, ifaceInfo);
      if (!newType)
        return failure();
      if (newType != port.type) {
        updated.type = newType;
        needsChange = true;
      }
      newInputs.push_back(updated);
      if (ifaceInfo) {
        BlockArgument arg = mod.getBodyBlock()->getArgument(port.argNum);
        ifaceValueInfo[arg] = ifaceInfo;
      }
    }

    for (const auto &port : ports.getOutputs()) {
      if (port.type.isa<InterfaceType>()) {
        mod.emitOpError()
            << "interface outputs are not supported by sv-lower-interfaces yet "
               "(module "
            << mod.getName() << ", port " << port.name.getValue() << ")";
        return failure();
      }
      newOutputs.push_back(port);
    }

    if (!needsChange) {
      changed = false;
      return success();
    }

    SmallVector<Type> inputTypes, outputTypes;
    for (auto &port : newInputs)
      inputTypes.push_back(port.type);
    for (auto &port : newOutputs)
      outputTypes.push_back(port.type);

    auto newType =
        hw::ModuleType::get(mod.getContext(), inputTypes, outputTypes);
    mod.setHWModuleType(newType);

    Block *body = mod.getBodyBlock();
    for (auto [idx, port] : llvm::enumerate(newInputs))
      body->getArgument(idx).setType(port.type);

    if (failed(rewriteInterfaceReads(mod, ifaceValueInfo)))
      return failure();

    changed = true;
    return success();
  }

  LogicalResult rewriteInterfaceReads(
      hw::HWModuleOp mod,
      DenseMap<Value, InterfaceInfo *> &ifaceValueInfo) {
    bool changed = false;
    WalkResult result = mod.walk([&](sv::ReadInterfaceSignalOp op) {
      auto it = ifaceValueInfo.find(op.getIface());
      if (it == ifaceValueInfo.end())
        return WalkResult::advance();
      InterfaceInfo *info = it->second;
      StringRef fieldName = op.getSignalName().getLeafReference().getValue();
      auto fieldIt = info->fieldIndex.find(fieldName);
      if (fieldIt == info->fieldIndex.end()) {
        op.emitOpError() << "signal " << fieldName
                         << " not found in interface "
                         << info->op.getSymName();
        return WalkResult::interrupt();
      }
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      auto newVal = builder.create<hw::StructExtractOp>(
          info->getStructType().getElementType(fieldIt->second), op.getIface(),
          builder.getStringAttr(fieldName));
      op.replaceAllUsesWith(newVal);
      op.erase();
      changed = true;
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
    return success();
  }

  Type convertInterfaceType(Type type, const PortInfo &port,
                            hw::HWModuleOp mod, InterfaceInfo *&infoOut) {
    infoOut = nullptr;
    if (auto ifaceType = type.dyn_cast<InterfaceType>()) {
      auto symName = StringAttr::get(mod.getContext(),
                                     ifaceType.getInterface().getValue());
      auto it = interfaces.find(symName);
      if (it == interfaces.end()) {
        mod.emitOpError()
            << "references unknown interface " << ifaceType.getInterface();
        return {};
      }
      const InterfaceInfo &info = it->second;
      if (info.fields.empty()) {
        mod.emitOpError() << "interface " << info.op.getSymName()
                          << " has no signals; lowering is not meaningful";
        return {};
      }
      infoOut = const_cast<InterfaceInfo *>(&info);
      return info.getStructType();
    }

    return type;
  }

  DenseMap<StringAttr, InterfaceInfo> interfaces;
};

} // namespace

std::unique_ptr<Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}
