// RUN: circt-opt --sv-lower-interfaces %s | FileCheck %s

module {
  sv.interface @chan {
    sv.interface.signal @data : i8
    sv.interface.signal @valid : i1
    sv.interface.modport @sink (input @data)
  }

  // CHECK-LABEL: hw.module @dut
  // CHECK-SAME: (in %iface : !hw.struct<data: i8, valid: i1>, in %other : i1)
  hw.module @dut(in %iface : !sv.interface<@chan>, in %other : i1) {
    // CHECK: %[[VAL:.*]] = hw.struct_extract %iface["data"] : !hw.struct<data: i8, valid: i1>
    %val = sv.interface.signal.read %iface(@chan::@data) : i8
    hw.output
  }

  // CHECK-LABEL: hw.module @use_modport
  // CHECK-SAME: (in %mp : !hw.struct<data: i8>)
  hw.module @use_modport(in %mp : !sv.modport<@chan::@sink>) {
    // CHECK: hw.struct_extract %mp["data"] : !hw.struct<data: i8>
    %v = sv.interface.signal.read %mp(@chan::@data) : i8
    hw.output
  }
}
