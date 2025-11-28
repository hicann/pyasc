// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_vector_unary_l0_ops(int32_t v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::UnaryRepeatParams v4) {
// CHECK-NEXT:   AscendC::Abs<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Exp<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Ln<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Not<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Reciprocal<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Relu<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Rsqrt<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   AscendC::Sqrt<float, 0>(v2, v3, v1, v1, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_unary_l0_ops(%c1_i32 : i32, %dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %params: !ascendc.unary_repeat_params) {
  ascendc.abs_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.exp_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.ln_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.not_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.reciprocal_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.relu_l0 %dst, %src, %c1_i32, %c1_i32,%params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.rsqrt_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  ascendc.sqrt_l0 %dst, %src, %c1_i32, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_unary_l1_ops(int32_t v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3, AscendC::UnaryRepeatParams v4, uint64_t v5, uint64_t v6) {
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Abs<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Exp<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Ln<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Not<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Reciprocal<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Relu<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Rsqrt<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   uint64_t v2_mask_list0[] = {v5, v6};
// CHECK-NEXT:   AscendC::Sqrt<float, 0>(v2, v3, v2_mask_list0, v1, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_unary_l1_ops(%c1_i32 : i32, %dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %params: !ascendc.unary_repeat_params, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.abs_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.exp_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.ln_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.not_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.reciprocal_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.relu_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.rsqrt_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  ascendc.sqrt_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %c1_i32, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, ui64, ui64, i32, !ascendc.unary_repeat_params
  return
}

// CHECK-LABEL:void emit_vector_unary_l2_ops(int32_t v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<float> v3) {
// CHECK-NEXT:   AscendC::Abs(v2, v3, v1);
// CHECK-NEXT:   AscendC::Exp(v2, v3, v1);
// CHECK-NEXT:   AscendC::Ln(v2, v3, v1);
// CHECK-NEXT:   AscendC::Neg(v2, v3, v1);
// CHECK-NEXT:   AscendC::Not(v2, v3, v1);
// CHECK-NEXT:   AscendC::Reciprocal(v2, v3, v1);
// CHECK-NEXT:   AscendC::Relu(v2, v3, v1);
// CHECK-NEXT:   AscendC::Rsqrt(v2, v3, v1);
// CHECK-NEXT:   AscendC::Sqrt(v2, v3, v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_vector_unary_l2_ops(%c0_i32 : i32, %dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>) {
  ascendc.abs_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.exp_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.ln_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.neg_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.not_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.reciprocal_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.relu_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.rsqrt_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  ascendc.sqrt_l2 %dst, %src, %c0_i32 : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32
  return
}
