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
  struct ModportInfo {
    SmallVector<StringAttr> signalOrder;
    DenseMap<StringAttr, sv::ModportDirection> directions;
  };

  InterfaceOp op;
  SmallVector<hw::StructType::FieldInfo> fields;
  DenseMap<StringRef, unsigned> fieldIndex;
  DenseMap<StringAttr, ModportInfo> modports;

  hw::StructType getStructType() const {
    return hw::StructType::get(op.getContext(), fields);
  }

  hw::StructType getModportStructType(StringAttr modportName) const {
    SmallVector<hw::StructType::FieldInfo> modFields;
    auto it = modports.find(modportName);
    if (it == modports.end())
      return hw::StructType();
    for (auto sig : it->second.signalOrder) {
      auto fieldIt = fieldIndex.find(sig.getValue());
      if (fieldIt == fieldIndex.end())
        continue;
      modFields.push_back(fields[fieldIt->second]);
    }
    return hw::StructType::get(op.getContext(), modFields);
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
      for (auto modport : iface.getOps<InterfaceModportOp>()) {
        InterfaceInfo::ModportInfo mpInfo;
        for (Attribute attr : modport.getPortsAttr()) {
          auto port = cast<ModportStructAttr>(attr);
          auto sig = port.getSignalAttr();
          mpInfo.signalOrder.push_back(sig);
          mpInfo.directions[sig] = port.getDirection();
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

  LogicalResult rewriteModule(hw::HWModuleOp mod, bool &changed) {
    ModulePortInfo ports = mod.getPorts();
    SmallVector<PortInfo> newInputs, newOutputs;
    bool needsChange = false;
    DenseMap<Value, InterfaceInfo *> ifaceValueInfo;
    DenseMap<Value, StringAttr> ifaceValueModport;
    Block *body = mod.getBodyBlock();

    for (const auto &port : ports.getInputs()) {
      auto updated = port;
      InterfaceInfo *ifaceInfoPtr = nullptr;
      StringAttr modportName;

      if (auto ifaceType = port.type.dyn_cast<InterfaceType>()) {
        auto ifaceInfo = lookupInterfaceInfo(ifaceType, mod);
        if (failed(ifaceInfo))
          return failure();
        ifaceInfoPtr = *ifaceInfo;
        if (ifaceInfoPtr) {
          BlockArgument arg = body->getArgument(port.argNum);
          if (canLowerInterfaceValue(arg)) {
            updated.type = ifaceInfoPtr->getStructType();
            ifaceValueInfo[arg] = ifaceInfoPtr;
            needsChange = true;
          }
        }
      } else if (auto modportType = port.type.dyn_cast<ModportType>()) {
        auto ifaceInfo = lookupInterfaceInfo(modportType, mod, modportName);
        if (failed(ifaceInfo))
          return failure();
        ifaceInfoPtr = *ifaceInfo;
        if (ifaceInfoPtr) {
          BlockArgument arg = body->getArgument(port.argNum);
          if (canLowerInterfaceValue(arg)) {
            auto mpStruct = ifaceInfoPtr->getModportStructType(modportName);
            if (!mpStruct)
              return mod.emitOpError()
                     << "modport " << modportName
                     << " does not reference any signals";
            updated.type = mpStruct;
            ifaceValueInfo[arg] = ifaceInfoPtr;
            ifaceValueModport[arg] = modportName;
            needsChange = true;
          }
        }
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

    for (auto [idx, port] : llvm::enumerate(newInputs))
      body->getArgument(idx).setType(port.type);

    if (failed(rewriteInterfaceReads(mod, ifaceValueInfo, ifaceValueModport)))
      return failure();

    changed = true;
    return success();
  }

  struct ResolvedInterfaceValue {
    Value baseValue;
    InterfaceInfo *info;
    sv::GetModportOp modportOp;
    StringAttr directModport;
  };

  FailureOr<ResolvedInterfaceValue>
  resolveInterfaceValue(Value value,
                        DenseMap<Value, InterfaceInfo *> &ifaceValueInfo,
                        DenseMap<Value, StringAttr> &ifaceValueModport) {
    auto it = ifaceValueInfo.find(value);
    if (it != ifaceValueInfo.end())
      return ResolvedInterfaceValue{value,
                                    it->second,
                                    sv::GetModportOp(nullptr),
                                    ifaceValueModport.lookup(value)};

    if (auto modport = value.getDefiningOp<sv::GetModportOp>()) {
      auto resolved =
          resolveInterfaceValue(modport.getIface(), ifaceValueInfo,
                                ifaceValueModport);
      if (failed(resolved))
        return failure();
      resolved->modportOp = modport;
      return resolved;
    }
    return failure();
  }

  LogicalResult rewriteInterfaceReads(
      hw::HWModuleOp mod,
      DenseMap<Value, InterfaceInfo *> &ifaceValueInfo,
      DenseMap<Value, StringAttr> &ifaceValueModport) {
    bool changed = false;
    WalkResult result = mod.walk([&](sv::ReadInterfaceSignalOp op) {
      auto resolved =
          resolveInterfaceValue(op.getIface(), ifaceValueInfo,
                                ifaceValueModport);
      if (failed(resolved))
        return WalkResult::advance();
      InterfaceInfo *info = resolved->info;
      StringRef fieldName = op.getSignalName().getLeafReference().getValue();
      auto fieldIt = info->fieldIndex.find(fieldName);
      if (fieldIt == info->fieldIndex.end()) {
        op.emitOpError() << "signal " << fieldName
                         << " not found in interface "
                         << info->op.getSymName();
        return WalkResult::interrupt();
      }
      auto checkDirection = [&](StringAttr modportName) -> LogicalResult {
        auto mpIt = info->modports.find(modportName);
        if (mpIt == info->modports.end())
          return success();
        auto sigAttr = StringAttr::get(mod.getContext(), fieldName);
        auto dirIt = mpIt->second.directions.find(sigAttr);
        if (dirIt != mpIt->second.directions.end() &&
            dirIt->second == sv::ModportDirection::Output) {
            op.emitOpError()
                << "cannot read signal " << fieldName
                << " from modport " << modportName
                << " because it is declared as output";
            return WalkResult::interrupt();
        }
        return success();
      };

      if (resolved->directModport)
        if (failed(checkDirection(resolved->directModport)))
          return WalkResult::interrupt();

      if (resolved->modportOp)
        if (failed(checkDirection(resolved->modportOp.getField())))
          return WalkResult::interrupt();

      ImplicitLocOpBuilder builder(op.getLoc(), op);
      auto newVal = builder.create<hw::StructExtractOp>(
          info->getStructType().getElementType(fieldIt->second),
          resolved->baseValue, builder.getStringAttr(fieldName));
      op.replaceAllUsesWith(newVal);
      op.erase();
      changed = true;
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
    return success();
  }

  DenseMap<StringAttr, InterfaceInfo> interfaces;

  FailureOr<InterfaceInfo *> lookupInterfaceByName(StringAttr symName,
                                                   hw::HWModuleOp mod) {
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

  FailureOr<InterfaceInfo *> lookupInterfaceInfo(InterfaceType type,
                                                 hw::HWModuleOp mod) {
    auto symName =
        StringAttr::get(mod.getContext(), type.getInterface().getValue());
    return lookupInterfaceByName(symName, mod);
  }

  FailureOr<InterfaceInfo *> lookupInterfaceInfo(ModportType type,
                                                 hw::HWModuleOp mod,
                                                 StringAttr &modportName) {
    SymbolRefAttr modportRef = type.getModport();
    modportName = modportRef.getLeafReference();
    auto symName =
        StringAttr::get(mod.getContext(), modportRef.getRootReference());
    return lookupInterfaceByName(symName, mod);
  }

  bool canLowerInterfaceValue(Value ifaceVal) {
    return llvm::all_of(ifaceVal.getUsers(), [](Operation *user) {
      return isa<sv::ReadInterfaceSignalOp>(user);
    });
  }
};

} // namespace

std::unique_ptr<Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}
