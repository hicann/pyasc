// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_set_fmatrix(int16_t v1, int8_t* v2, int8_t v3) {
// CHECK-NEXT:   AscendC::FmatrixMode v4{v3};
// CHECK-NEXT:   AscendC::SetFmatrix(v1, v1, v2, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_fmatrix(%arg0: i16, %arg1: memref<4xi8>, %arg2: i8) {
  %fm = ascendc.construct !ascendc.fmatrix_mode(%arg2) : i8
  ascendc.set_fmatrix %arg0, %arg0, %arg1, %fm : i16, i16, memref<4xi8>, !ascendc.fmatrix_mode
  return
}

// CHECK-LABEL:void emit_set_load_data_padding_value(float v1) {
// CHECK-NEXT:   AscendC::SetLoadDataPaddingValue(v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_load_data_padding_value(%arg0: f32) {
  ascendc.set_load_data_padding_value %arg0 : f32
  return
}

// CHECK-LABEL:void emit_set_load_data_repeat(int16_t v1, int8_t v2) {
// CHECK-NEXT:   AscendC::LoadDataRepeatParam v3{v1, v2, v2};
// CHECK-NEXT:   AscendC::SetLoadDataRepeat(v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_load_data_repeat(%arg0: i16, %arg1: i8) {
  %param = ascendc.construct !ascendc.load_data_repeat_param(%arg0, %arg1, %arg1) : i16, i8, i8
  ascendc.set_load_data_repeat %param : !ascendc.load_data_repeat_param
  return
}

// CHECK-LABEL:void emit_set_load_data_boundary(int32_t v1) {
// CHECK-NEXT:   AscendC::SetLoadDataBoundary(v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_load_data_boundary(%c0_i32: i32) {
  ascendc.set_load_data_boundary %c0_i32 : i32
  return
}

// CHECK-LABEL:void emit_ffts_cross_core_sync(int32_t v1, int64_t v2) {
// CHECK-NEXT:  constexpr int64_t c3873_i64 = 3873;
// CHECK-NEXT:  constexpr int32_t c5_i32 = 5;
// CHECK-NEXT:  ffts_cross_core_sync(PIPE_MTE3, c3873_i64);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_ffts_cross_core_sync(%arg0: i32, %arg1: i64) {
  %c3873_i64 = arith.constant 3873 : i64
  %c5_i32 = arith.constant 5 : i32
  ascendc.ffts_cross_core_sync %c3873_i64 {pipe = 5 : i32} : i64
  return
}

