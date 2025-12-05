// RUN: circt-opt --sv-lower-interfaces %s | FileCheck %s

module {
  sv.interface @chan {
    sv.interface.signal @data : i8
    sv.interface.signal @valid : i1
  }

  // CHECK-LABEL: hw.module @dut
  // CHECK-SAME: (in %iface : !hw.struct<data: i8, valid: i1>, in %other : i1)
  hw.module @dut(in %iface : !sv.interface<@chan>, in %other : i1) {
    // CHECK: %[[VAL:.*]] = hw.struct_extract %iface["data"] : !hw.struct<data: i8, valid: i1>
    %val = sv.interface.signal.read %iface(@chan::@data) : i8
    hw.output
  }
}
