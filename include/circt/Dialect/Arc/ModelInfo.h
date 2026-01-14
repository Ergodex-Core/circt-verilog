//===- ModelInfo.h - Information about Arc models -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines and computes information about Arc models.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_MODELINFO_H
#define CIRCT_DIALECT_ARC_MODELINFO_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace circt {
namespace arc {

/// Gathers information about a given Arc state.
struct StateInfo {
  enum Type { Input, Output, Register, Memory, Wire } type;
  std::string name;
  unsigned offset;
  /// The semantic bit width of the state. For 4-state values modeled as a
  /// `{unknown, value}` struct, this reflects the bit width of the `value`
  /// field rather than the padded storage size.
  unsigned numBits;
  /// The number of bytes used by the state's in-memory storage representation.
  unsigned storageBytes = 0;
  /// For 4-state values modeled as a `{unknown, value}` struct, these indicate
  /// the byte offsets (relative to `offset`) for the `value` and `unknown`
  /// fields. For all other values these remain 0.
  unsigned valueOffset = 0;
  unsigned unknownOffset = 0;
  unsigned memoryStride = 0; // byte separation between memory words
  unsigned memoryDepth = 0;  // number of words in a memory
};

/// Initialization metadata for runtime-managed signals used by the arcilator
/// scheduler hooks (`__arcilator_sig_*`).
struct RuntimeSignalInitInfo {
  uint64_t sigId = 0;
  uint64_t initU64 = 0;
  uint64_t totalWidth = 0;
};

/// Gathers information about a given Arc model.
struct ModelInfo {
  std::string name;
  size_t numStateBytes;
  llvm::SmallVector<StateInfo> states;
  llvm::SmallVector<RuntimeSignalInitInfo> sigInits;
  mlir::FlatSymbolRefAttr initialFnSym;
  mlir::FlatSymbolRefAttr finalFnSym;

  ModelInfo(std::string name, size_t numStateBytes,
            llvm::SmallVector<StateInfo> states,
            llvm::SmallVector<RuntimeSignalInitInfo> sigInits,
            mlir::FlatSymbolRefAttr initialFnSym,
            mlir::FlatSymbolRefAttr finalFnSym)
      : name(std::move(name)), numStateBytes(numStateBytes),
        states(std::move(states)), sigInits(std::move(sigInits)),
        initialFnSym(initialFnSym), finalFnSym(finalFnSym) {}
};

/// Collects information about states within the provided Arc model storage
/// `storage`,  assuming default `offset`, and adds it to `states`.
mlir::LogicalResult collectStates(mlir::Value storage, unsigned offset,
                                  llvm::SmallVector<StateInfo> &states);

/// Collects information about all Arc models in the provided `module`,
/// and adds it to `models`.
mlir::LogicalResult collectModels(mlir::ModuleOp module,
                                  llvm::SmallVector<ModelInfo> &models);

/// Serializes `models` to `outputStream` in JSON format.
void serializeModelInfoToJson(llvm::raw_ostream &outputStream,
                              llvm::ArrayRef<ModelInfo> models);

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_MODELINFO_H
