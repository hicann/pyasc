// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_cast(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, uint8_t v3, AscendC::UnaryRepeatParams v4, uint64_t v5, uint64_t v6) {
// CHECK-NEXT:   AscendC::Cast<float, float, 0>(v1, v2, AscendC::RoundMode::CAST_NONE, v5, v3, v4);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Cast<float, float, 0>(v1, v2, AscendC::RoundMode::CAST_NONE, v1_mask_list0, v3, v4);
// CHECK-NEXT:   AscendC::Cast<float, float>(v1, v2, AscendC::RoundMode::CAST_NONE, v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_cast(%dst : !ascendc.local_tensor<1024xf32>, %src : !ascendc.local_tensor<1024xf32>, %repeatTime : ui8, %params: !ascendc.unary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.cast_l0 %dst, %src, %maskArray1_0, %repeatTime, %params {roundMode = 0 : i32} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui8, !ascendc.unary_repeat_params
  ascendc.cast_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %repeatTime, %params {roundMode = 0 : i32} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, ui8, !ascendc.unary_repeat_params
  ascendc.cast_l2 %dst, %src, %repeatTime {roundMode = 0 : i32} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui8
  return
}

// CHECK-LABEL:void emit_vector_binary_l0_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, AscendC::BinaryRepeatParams v5) {
// CHECK-NEXT:   AscendC::AddReluCast<float, float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::SubReluCast<float, float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l0_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %c1_i32 : i32, %params: !ascendc.binary_repeat_params) {
  ascendc.add_relu_cast_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  ascendc.sub_relu_cast_l0 %dst, %src0, %src1, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_binary_l1_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4, AscendC::BinaryRepeatParams v5, uint64_t v6, uint64_t v7) {
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::AddReluCast<float, float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::SubReluCast<float, float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l1_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1: !ascendc.local_tensor<1024xf32>, %c1_i32 : i32, %params: !ascendc.binary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.add_relu_cast_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  ascendc.sub_relu_cast_l1 %dst, %src0, %src1, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.binary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_binary_l2_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, int32_t v4) {
// CHECK-NEXT:   AscendC::AddReluCast(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::SubReluCast(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_binary_l2_ops(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %src1 : !ascendc.local_tensor<1024xf32>, %calCount_i32 : i32) {
  ascendc.add_relu_cast_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.sub_relu_cast_l2 %dst, %src0, %src1, %calCount_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  return
}

// CHECK-LABEL:void emit_set_deq_scale(half v1, float v2, int16_t v3) {
// CHECK-NEXT:   constexpr bool c0_i1 = false;
// CHECK-NEXT:   AscendC::SetDeqScale(v1);
// CHECK-NEXT:   AscendC::SetDeqScale(v2, v3, c0_i1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_deq_scale(%arg0: f16, %arg1: f32, %arg2: i16) {
  %false = arith.constant false
  ascendc.set_deq_scale %arg0 : f16
  ascendc.set_deq_scale %arg1, %arg2, %false : f32, i16, i1
  return
}

// CHECK-LABEL:void emit_set_deq_scale_l4(AscendC::LocalTensor<float> v1, AscendC::VdeqInfo v2) {
// CHECK-NEXT:   AscendC::SetDeqScale(v1, v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_set_deq_scale_l4(%vdeq: !ascendc.local_tensor<32xf32>, %vdeq_info: !ascendc.vdeq_info) {
  ascendc.set_deq_scale_l4 %vdeq, %vdeq_info : !ascendc.local_tensor<32xf32>, !ascendc.vdeq_info
  return
}

// CHECK-LABEL:void emit_cast_deq(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, uint8_t v3, AscendC::UnaryRepeatParams v4, uint64_t v5, uint64_t v6, int32_t v7) {
// CHECK-NEXT:   AscendC::CastDeq<float, float, 0, 0>(v1, v2, v7);
// CHECK-NEXT:   AscendC::CastDeq<float, float, 1, 0, 0>(v1, v2, v5, v3, v4);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::CastDeq<float, float, 1, 0, 0>(v1, v2, v1_mask_list0, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_cast_deq(%dst : !ascendc.local_tensor<1024xf32>, %src : !ascendc.local_tensor<1024xf32>, %repeatTime : ui8, %params: !ascendc.unary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64, %count: i32) {
  ascendc.cast_deq_l2 %dst, %src, %count : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.cast_deq_l0 %dst, %src, %maskArray1_0, %repeatTime, %params {isSetMask} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui8, !ascendc.unary_repeat_params
  ascendc.cast_deq_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %repeatTime, %params {isSetMask} : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, ui8, !ascendc.unary_repeat_params  
  return
}