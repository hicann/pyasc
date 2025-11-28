// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_vector_scalar_l0_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, AscendC::UnaryRepeatParams v5) {
// CHECK-NEXT:   AscendC::Adds<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::LeakyRelu<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Maxs<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Mins<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::Muls<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::ShiftLeft<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   AscendC::ShiftRight<float, 0>(v1, v2, v3, v4, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_scalar_l0_ops(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %scalar : i32, %c1_i32 : i32, %params: !ascendc.unary_repeat_params) {
  ascendc.adds_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.leaky_relu_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.maxs_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.mins_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.muls_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.shift_left_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  ascendc.shift_right_l0 %dst, %src, %scalar, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_scalar_l1_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, AscendC::UnaryRepeatParams v5, uint64_t v6, uint64_t v7) {
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Adds<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::LeakyRelu<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Maxs<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Mins<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::Muls<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::ShiftLeft<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v6, v7};
// CHECK-NEXT:   AscendC::ShiftRight<float, 0>(v1, v2, v3, v1_mask_list0, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_scalar_l1_ops(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %scalar : i32, %c1_i32 : i32, %params: !ascendc.unary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.adds_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.leaky_relu_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.maxs_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.mins_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.muls_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.shift_left_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.shift_right_l1 %dst, %src, %scalar, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_scalar_l2_ops(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4) {
// CHECK-NEXT:   AscendC::Adds<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::LeakyRelu<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Maxs<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Mins<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::Muls<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::ShiftLeft<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   AscendC::ShiftRight<float, 0>(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_scalar_l2_ops(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %scalar : i32, %c1_i32 : i32) {
  ascendc.adds_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.leaky_relu_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.maxs_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.mins_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.muls_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.shift_left_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  ascendc.shift_right_l2 %dst, %src, %scalar, %c1_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32
  return
}
