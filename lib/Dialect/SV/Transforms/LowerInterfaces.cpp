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
    for (auto hwMod : module.getOps<hw::HWModuleOp>())
      changed |= succeeded(rewriteModule(hwMod));

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

  LogicalResult rewriteModule(hw::HWModuleOp mod) {
    ModulePortInfo ports = mod.getPorts();
    SmallVector<PortInfo> newInputs, newOutputs;
    bool needsChange = false;

    for (const auto &port : ports.getInputs()) {
      auto updated = port;
      Type newType = convertInterfaceType(port.type, port, mod);
      if (!newType)
        return failure();
      if (newType != port.type) {
        updated.type = newType;
        needsChange = true;
      }
      newInputs.push_back(updated);
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

    if (!needsChange)
      return success();

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

    return success();
  }

  Type convertInterfaceType(Type type, const PortInfo &port,
                            hw::HWModuleOp mod) {
    auto ifaceType = type.dyn_cast<InterfaceType>();
    if (!ifaceType)
      return type;

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
    return info.getStructType();
  }

  DenseMap<StringAttr, InterfaceInfo> interfaces;
};

} // namespace

std::unique_ptr<Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}
