// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_select_op(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<int16_t> v2, AscendC::LocalTensor<float> v3, AscendC::LocalTensor<float> v4, int32_t v5, int32_t v6) {
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, v5, v6, AscendC::BinaryRepeatParams(v6, v6, v6, v6, v6, v6));
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_select_op(%dst : !ascendc.local_tensor<1024xf32>, %cond : !ascendc.local_tensor<32xi16>, %src0 : !ascendc.local_tensor<1024xf32>, %src1 : !ascendc.local_tensor<1024xf32>, %mask : i32, %c0_i32 : i32) {
  ascendc.select_l0 %dst, %cond, %src0, %src1, %mask, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32 {mode = 2 : i32}: !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, i32, i32, i32, i32, i32
  ascendc.select_l2 %dst, %cond, %src0, %src1, %c0_i32 {selMode = 2 : i32}: !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  return
}

// CHECK-LABEL:void emit_select_scalar_op(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<int16_t> v2, AscendC::LocalTensor<float> v3, int32_t v4, int32_t v5, int32_t v6) {
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, v5, v6, AscendC::BinaryRepeatParams(v6, v6, v6, v6, v6, v6));
// CHECK-NEXT:   AscendC::Select(v1, v2, v3, v4, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_select_scalar_op(%dst : !ascendc.local_tensor<1024xf32>, %cond : !ascendc.local_tensor<32xi16>, %src0 : !ascendc.local_tensor<1024xf32>, %src1 : i32, %mask : i32, %c0_i32 : i32) {
  ascendc.select_scalar_l0 %dst, %cond, %src0, %src1, %mask, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32 {mode = 2 : i32}: !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, i32, i32, i32, i32, i32, i32
  ascendc.select_scalar_l2 %dst, %cond, %src0, %src1, %c0_i32 {selMode = 2 : i32}: !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xi16>, !ascendc.local_tensor<1024xf32>, i32, i32
  return
}

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

// CHECK-LABEL: void emit_fixpipe(AscendC::GlobalTensor<half> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<int64_t> v3, AscendC::FixpipeParams<int32_t> v4) {
// CHECK-NEXT:   AscendC::Fixpipe(v2, v2, v4);
// CHECK-NEXT:   AscendC::Fixpipe(v2, v2, v3, v4);
// CHECK-NEXT:   AscendC::Fixpipe(v1, v2, v4);
// CHECK-NEXT:   AscendC::Fixpipe(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_fixpipe(%arg0: !ascendc.global_tensor<*xf16>, %arg1: !ascendc.local_tensor<*xf32>, %arg2: !ascendc.local_tensor<*xi64>, %arg3: !ascendc.fixpipe_params<i32>) {
  ascendc.fixpipe %arg1, %arg1, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<i32>
  ascendc.fixpipe %arg1, %arg1, %arg2, %arg3 : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xi64>, !ascendc.fixpipe_params<i32>
  ascendc.fixpipe %arg0, %arg1, %arg3 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf32>, !ascendc.fixpipe_params<i32>
  ascendc.fixpipe %arg0, %arg1, %arg2, %arg3 : !ascendc.global_tensor<*xf16>, !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xi64>, !ascendc.fixpipe_params<i32>
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

