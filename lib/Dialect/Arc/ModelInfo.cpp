//===- ModelInfo.cpp - Information about Arc models -----------------------===//
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

#include "circt/Dialect/Arc/ModelInfo.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace circt;
using namespace arc;

/// Detect `{unknown, value}` 4-state structs and compute the semantic bit width
/// and element byte offsets (relative to the struct base).
static bool getFourStateInfo(Type type, unsigned &bitWidth,
                             unsigned &valueOffsetBytes,
                             unsigned &unknownOffsetBytes) {
  auto structType = dyn_cast<hw::StructType>(type);
  if (!structType)
    return false;

  auto elements = structType.getElements();
  if (elements.size() != 2)
    return false;

  auto int0 = dyn_cast<IntegerType>(elements[0].type);
  auto int1 = dyn_cast<IntegerType>(elements[1].type);
  if (!int0 || !int1)
    return false;
  if (int0.getWidth() != int1.getWidth())
    return false;

  bitWidth = int0.getWidth();

  auto classify = [](StringRef name) -> std::optional<bool> {
    // Return `true` for value, `false` for unknown, and `nullopt` if unknown.
    if (name == "value" || name == "aval")
      return true;
    if (name == "unknown" || name == "bval")
      return false;
    return std::nullopt;
  };

  // Determine which element is value/unknown using names, falling back to the
  // conventional `{unknown, value}` order used by Moore lowering.
  unsigned unknownIdx = 0;
  unsigned valueIdx = 1;
  auto c0 = classify(elements[0].name.getValue());
  auto c1 = classify(elements[1].name.getValue());
  if (c0.has_value() && c1.has_value()) {
    if (*c0 == *c1)
      return false; // Both labeled as the same thing.
    valueIdx = *c0 ? 0 : 1;
    unknownIdx = *c0 ? 1 : 0;
  } else if (c0.has_value()) {
    valueIdx = *c0 ? 0 : 1;
    unknownIdx = *c0 ? 1 : 0;
  } else if (c1.has_value()) {
    valueIdx = *c1 ? 1 : 0;
    unknownIdx = *c1 ? 0 : 1;
  }

  // Match the struct element layout in `computeLLVMBitWidth` for integers.
  uint64_t widthBits = std::max<uint64_t>(bitWidth, 8);
  uint64_t alignment =
      llvm::bit_ceil(std::min<uint64_t>(widthBits, 16 * 8));
  uint64_t alignedWidthBits = llvm::alignToPowerOf2(widthBits, alignment);
  unsigned elementBytes = alignedWidthBits / 8;

  // The HW-to-LLVM lowering reverses hw.struct element order to match the
  // little-endian packing used by CIRCT's HW types. Mirror that here so the
  // exported offsets match the actual in-memory layout.
  auto toLLVMIndex = [&](unsigned idx) -> unsigned {
    return static_cast<unsigned>(elements.size()) - idx - 1;
  };
  unsigned llvmUnknownIdx = toLLVMIndex(unknownIdx);
  unsigned llvmValueIdx = toLLVMIndex(valueIdx);

  unsigned off0 = 0;
  unsigned off1 = elementBytes;

  unknownOffsetBytes = (llvmUnknownIdx == 0) ? off0 : off1;
  valueOffsetBytes = (llvmValueIdx == 0) ? off0 : off1;
  return true;
}

LogicalResult circt::arc::collectStates(Value storage, unsigned offset,
                                        SmallVector<StateInfo> &states) {
  struct StateCollectionJob {
    mlir::Value::user_iterator nextToProcess;
    mlir::Value::user_iterator end;
    unsigned offset;

    StateCollectionJob(Value storage, unsigned offset)
        : nextToProcess(storage.user_begin()), end(storage.user_end()),
          offset(offset) {}
  };

  SmallVector<StateCollectionJob, 4> jobStack{{storage, offset}};

  while (!jobStack.empty()) {
    StateCollectionJob &job = jobStack.back();

    if (job.nextToProcess == job.end) {
      jobStack.pop_back();
      continue;
    }

    Operation *op = *job.nextToProcess++;
    unsigned offset = job.offset;

    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      if (!substorage.getOffset().has_value())
        return substorage.emitOpError(
            "without allocated offset; run state allocation first");
      Value substorageOutput = substorage.getOutput();
      jobStack.emplace_back(substorageOutput, offset + *substorage.getOffset());
      continue;
    }

    if (!isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp>(op))
      continue;

    SmallVector<StringAttr> names;

    auto opName = op->getAttrOfType<StringAttr>("name");
    if (opName && !opName.getValue().empty())
      names.push_back(opName);

    if (auto nameAttrs = op->getAttrOfType<ArrayAttr>("names"))
      for (auto attr : nameAttrs)
        if (auto nameAttr = dyn_cast<StringAttr>(attr))
          if (!nameAttr.empty())
            names.push_back(nameAttr);

    if (names.empty())
      continue;

    auto opOffset = op->getAttrOfType<IntegerAttr>("offset");
    if (!opOffset)
      return op->emitOpError(
          "without allocated offset; run state allocation first");

    StateInfo stateInfo;
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      auto stateType = cast<StateType>(result.getType());
      stateInfo.type = StateInfo::Register;
      if (isa<RootInputOp>(op))
        stateInfo.type = StateInfo::Input;
      else if (isa<RootOutputOp>(op))
        stateInfo.type = StateInfo::Output;
      else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
        if (alloc.getTap())
          stateInfo.type = StateInfo::Wire;
      }
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.storageBytes = stateType.getByteWidth();

      // For 4-state `{unknown, value}` structs, the semantic bit width should
      // match the `value` field width, rather than the padded storage size.
      unsigned fourStateBits = 0;
      unsigned valueOff = 0;
      unsigned unknownOff = 0;
      if (getFourStateInfo(stateType.getType(), fourStateBits, valueOff,
                           unknownOff)) {
        stateInfo.numBits = fourStateBits;
        stateInfo.valueOffset = valueOff;
        stateInfo.unknownOffset = unknownOff;
      } else {
        stateInfo.numBits = stateType.getBitWidth();
      }
      for (auto name : names) {
        stateInfo.name = name.getValue();
        states.push_back(stateInfo);
      }
      continue;
    }

    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto stride = op->getAttrOfType<IntegerAttr>("stride");
      if (!stride)
        return op->emitOpError(
            "without allocated stride; run state allocation first");
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      stateInfo.type = StateInfo::Memory;
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.storageBytes = (stateInfo.numBits + 7) / 8;
      stateInfo.memoryStride = stride.getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      for (auto name : names) {
        stateInfo.name = name.getValue();
        states.push_back(stateInfo);
      }
      continue;
    }
  }

  return success();
}

LogicalResult circt::arc::collectModels(mlir::ModuleOp module,
                                        SmallVector<ModelInfo> &models) {

  for (auto modelOp : module.getOps<ModelOp>()) {
    auto storageArg = modelOp.getBody().getArgument(0);
    auto storageType = cast<StorageType>(storageArg.getType());

    SmallVector<StateInfo> states;
    if (failed(collectStates(storageArg, 0, states)))
      return failure();
    llvm::stable_sort(states,
                      [](auto &a, auto &b) { return a.offset < b.offset; });

    models.emplace_back(std::string(modelOp.getName()), storageType.getSize(),
                        std::move(states), modelOp.getInitialFnAttr(),
                        modelOp.getFinalFnAttr());
  }

  return success();
}

void circt::arc::serializeModelInfoToJson(llvm::raw_ostream &outputStream,
                                          ArrayRef<ModelInfo> models) {
  llvm::json::OStream json(outputStream, 2);

  json.array([&] {
    for (const ModelInfo &model : models) {
      json.object([&] {
        json.attribute("name", model.name);
        json.attribute("numStateBytes", model.numStateBytes);
        json.attribute("initialFnSym", !model.initialFnSym
                                           ? ""
                                           : model.initialFnSym.getValue());
        json.attribute("finalFnSym",
                       !model.finalFnSym ? "" : model.finalFnSym.getValue());
        json.attributeArray("states", [&] {
          for (const auto &state : model.states) {
            json.object([&] {
              json.attribute("name", state.name);
              json.attribute("offset", state.offset);
              json.attribute("numBits", state.numBits);
              auto typeStr = [](StateInfo::Type type) {
                switch (type) {
                case StateInfo::Input:
                  return "input";
                case StateInfo::Output:
                  return "output";
                case StateInfo::Register:
                  return "register";
                case StateInfo::Memory:
                  return "memory";
                case StateInfo::Wire:
                  return "wire";
                }
                return "";
              };
              json.attribute("type", typeStr(state.type));
              if (state.type == StateInfo::Memory) {
                json.attribute("stride", state.memoryStride);
                json.attribute("depth", state.memoryDepth);
              }
              json.attribute("storageBytes", state.storageBytes);
              if (state.valueOffset != state.unknownOffset) {
                json.attribute("valueOffset", state.valueOffset);
                json.attribute("unknownOffset", state.unknownOffset);
              }
            });
          }
        });
      });
    }
  });
}

circt::arc::ModelInfoAnalysis::ModelInfoAnalysis(Operation *container) {
  assert(container->getNumRegions() == 1 && "Expected single region");
  assert(container->getRegion(0).getBlocks().size() == 1 &&
         "Expected single body block");

  for (auto modelOp :
       container->getRegion(0).getBlocks().front().getOps<ModelOp>()) {
    auto storageArg = modelOp.getBody().getArgument(0);
    auto storageType = cast<StorageType>(storageArg.getType());

    SmallVector<StateInfo> states;
    if (failed(collectStates(storageArg, 0, states))) {
      assert(false && "Failed to collect model states");
      continue;
    }
    llvm::stable_sort(states,
                      [](auto &a, auto &b) { return a.offset < b.offset; });
    infoMap.try_emplace(modelOp, std::string(modelOp.getName()),
                        storageType.getSize(), std::move(states),
                        modelOp.getInitialFnAttr(), modelOp.getFinalFnAttr());
  }
}
