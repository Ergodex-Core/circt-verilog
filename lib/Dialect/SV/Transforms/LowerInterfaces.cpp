#include "circt/Dialect/SV/SVPasses.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "sv-lower-interfaces"

using namespace circt;
using namespace sv;

namespace {

struct InterfaceFieldInfo {
  StringAttr name;
  Type type;
};

/// Collects and caches flattened interface schemas so we can reuse them while
/// walking the design.  At this stage the data is only gathered; the eventual
/// lowering that rewrites modules to consume these flattened schemas will be
/// added in follow-up patches.
class InterfaceSchemaCache {
public:
  InterfaceSchemaCache(ModuleOp module) : module(module), symTable(module) {}

  /// Returns the fields for an `sv.interface` symbol.
  FailureOr<ArrayRef<InterfaceFieldInfo>>
  getInterfaceSchema(FlatSymbolRefAttr ifaceRef) {
    auto it = interfaceCache.find(ifaceRef);
    if (it != interfaceCache.end())
      return it->second;

    auto iface =
        symTable.lookupNearestSymbolFrom<InterfaceOp>(module, ifaceRef);
    if (!iface)
      return ifaceRef.getAttr().emitError("referenced interface not found"), failure();

    SmallVector<InterfaceFieldInfo> fields;
    for (auto signal : iface.getOps<InterfaceSignalOp>()) {
      fields.push_back({signal.getSymNameAttr(), signal.getType()});
    }

    auto inserted = interfaceCache.try_emplace(ifaceRef, std::move(fields));
    cachedStorage.push_back(inserted.first->second);
    return inserted.first->second;
  }

private:
  ModuleOp module;
  SymbolTable symTable;

  DenseMap<FlatSymbolRefAttr, ArrayRef<InterfaceFieldInfo>> interfaceCache;
  // We own the storage for cached vectors to keep ArrayRef stable.
  SmallVector<SmallVector<InterfaceFieldInfo>> cachedStorage;
};

struct LowerInterfacesPass
    : public impl::LowerInterfacesBase<LowerInterfacesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    InterfaceSchemaCache cache(module);

    module.walk([&](hw::HWModuleLike moduleLike) {
      auto moduleType = moduleLike.getHWModuleType();
      for (auto port : moduleType.getPorts()) {
        Type portType = port.type;
        if (auto ifaceType = dyn_cast<InterfaceType>(portType)) {
          (void)cache.getInterfaceSchema(ifaceType.getInterface());
        } else if (auto modportType = dyn_cast<ModportType>(portType)) {
          auto ifaceRef = FlatSymbolRefAttr::get(
              moduleLike->getContext(), modportType.getModport().getRootReference());
          (void)cache.getInterfaceSchema(ifaceRef);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> circt::sv::createLowerInterfacesPass() {
  return std::make_unique<LowerInterfacesPass>();
}

#include "circt/Dialect/SV/SVPasses.h.inc"
