// RUN: circt-opt --sv-lower-interfaces %s | FileCheck %s

module {
  sv.interface @chan {
    sv.interface.signal @data : i8
    sv.interface.signal @valid : i1
    sv.interface.modport @sink (input @data)
    sv.interface.modport @source (output @data)
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
    %v = sv.interface.signal.read %mp(@chan::@sink::@data) : i8
    hw.output
  }

  // CHECK-LABEL: hw.module @write_iface
  // CHECK-SAME: (inout %iface : !hw.struct<data: i8, valid: i1>)
  hw.module @write_iface(in %iface : !sv.interface<@chan>) {
    %c0 = arith.constant 0 : i8
    sv.interface.signal.assign %iface(@chan::@data) = %c0 : i8
    hw.output
  }
  // CHECK:   %[[FIELD:.*]] = sv.struct_field_inout %iface["data"] : !hw.inout<struct<data: i8, valid: i1>>
  // CHECK:   sv.assign %[[FIELD]], %c0_i8 : i8

  // CHECK-LABEL: hw.module @read_write
  // CHECK-SAME: (inout %iface : !hw.struct<data: i8, valid: i1>)
  hw.module @read_write(in %iface : !sv.interface<@chan>) {
    %v = sv.interface.signal.read %iface(@chan::@data) : i8
    sv.interface.signal.assign %iface(@chan::@data) = %v : i8
    hw.output
  }
  // CHECK:   %[[READ_FIELD:.*]] = sv.struct_field_inout %iface["data"] : !hw.inout<struct<data: i8, valid: i1>>
  // CHECK-NEXT:   %[[VAL:.*]] = sv.read_inout %[[READ_FIELD]] : !hw.inout<i8>
  // CHECK:   %[[WRITE_FIELD:.*]] = sv.struct_field_inout %iface["data"] : !hw.inout<struct<data: i8, valid: i1>>
  // CHECK:   sv.assign %[[WRITE_FIELD]], %[[VAL]] : i8

  // CHECK-LABEL: hw.module @write_proc
  // CHECK-SAME: (inout %iface : !hw.struct<data: i8, valid: i1>, in %clk : i1)
  hw.module @write_proc(in %iface : !sv.interface<@chan>, in %clk : i1) {
    sv.always posedge %clk {
      %c0 = arith.constant 0 : i8
      sv.interface.signal.assign %iface(@chan::@data) = %c0 : i8
    }
    hw.output
  }
  // CHECK:   sv.always posedge %clk {
  // CHECK:     %[[PROC_FIELD:.*]] = sv.struct_field_inout %iface["data"] : !hw.inout<struct<data: i8, valid: i1>>
  // CHECK:     sv.bpassign %[[PROC_FIELD]], %c0_i8 : i8

  // CHECK-LABEL: hw.module @drive_modport
  // CHECK-SAME: (inout %mp : !hw.struct<data: i8>)
  hw.module @drive_modport(in %mp : !sv.modport<@chan::@source>) {
    %c1 = arith.constant 1 : i8
    sv.interface.signal.assign %mp(@chan::@source::@data) = %c1 : i8
    hw.output
  }
  // CHECK:   %[[MP_FIELD:.*]] = sv.struct_field_inout %mp["data"] : !hw.inout<struct<data: i8>>
  // CHECK:   sv.assign %[[MP_FIELD]], %c1_i8 : i8
}
