//===- LLHDValueOps.td - LLHD value operations -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD value creation.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/LLHD/IR/LLHDDialect.td"
include "circt/Dialect/LLHD/IR/LLHDTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantTimeOp : LLHDOp<"constant_time",
                                  [ConstantLike, Pure]> {
  let summary = "Introduce a new time constant.";
  let description = [{
    The `llhd.constant_time` instruction introduces a new constant time value as
    an SSA-operator.

    Example:

    ```mlir
    %1 = llhd.constant_time #llhd.time<1ns, 2d, 3d>
    ```
  }];

  let assemblyFormat = "$value attr-dict";

  let builders = [
    /// Build a ConstantTimeOp from an ArrayRef with three values
    /// (real-time, delta, epsilon) and a string for the real-time unit.
    OpBuilder<(ins "uint64_t":$time,
                   "const StringRef &":$timeUnit,
                   "unsigned":$delta,
                   "unsigned":$epsilon)>
  ];

  let arguments = (ins LLHDTimeAttr: $value);
  let results = (outs LLHDTimeType: $result);

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Simulation Time Helpers
//===----------------------------------------------------------------------===//

def NowOp : LLHDOp<"now", [
  MemoryEffects<[MemRead]>
]> {
  let summary = "Read the current simulation time";
  let description = [{
    Represents the current simulation time. This is used to lower SystemVerilog
    `$time` in flows that execute LLHD processes.

    The operation is modeled as a read effect to prevent CSE / hoisting across
    temporal boundaries.
  }];

  let results = (outs LLHDTimeType:$result);
  let assemblyFormat = "attr-dict";
}

def TimeToIntOp : LLHDOp<"time_to_int", [Pure]> {
  let summary = "Convert a time value to an integer count of femtoseconds";
  let description = [{
    Extracts the physical time component (in femtoseconds) from an `!llhd.time`
    value and returns it as an `i64`. Delta/epsilon components are ignored.
  }];

  let arguments = (ins LLHDTimeType:$input);
  let results = (outs I64:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def IntToTimeOp : LLHDOp<"int_to_time", [Pure]> {
  let summary = "Convert an integer femtoseconds value into a time value";
  let description = [{
    Creates an `!llhd.time` value from an `i64` count of femtoseconds, with
    delta and epsilon set to zero.
  }];

  let arguments = (ins I64:$input);
  let results = (outs LLHDTimeType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
