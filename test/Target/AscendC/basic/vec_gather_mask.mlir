// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_gather_mask(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<uint32_t> v3, bool v4, uint32_t v5, AscendC::GatherMaskParams v6, int64_t v7) {
// CHECK-NEXT:   uint64_t v8;
// CHECK-NEXT:   AscendC::GatherMask<float, uint32_t, AscendC::defaultGatherMaskMode>(v1, v2, v3, v4, v5, v6, v8);
// CHECK-NEXT:   constexpr int8_t c1_i8 = 1;
// CHECK-NEXT:   uint64_t v9;
// CHECK-NEXT:   AscendC::GatherMask<float, AscendC::defaultGatherMaskMode>(v1, v2, c1_i8, v4, v5, v6, v9);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_gather_mask(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>,
                           %pattern_tensor: !ascendc.local_tensor<32xui32>, %reduce_mode: i1, %mask: ui32,
                           %params: !ascendc.gather_mask_params, %rsvd_cnt: i64) {
  %0 = ascendc.gather_mask %dst, %src0, %pattern_tensor, %reduce_mode, %mask, %params, default :
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<32xui32>, i1, ui32, !ascendc.gather_mask_params -> ui64
  
  %pattern_const = arith.constant 1 : i8
  ascendc.gather_mask %dst, %src0, %pattern_const, %reduce_mode, %mask, %params, default :
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i8, i1, ui32, !ascendc.gather_mask_params -> ui64
  return
}

// CHECK-LABEL:void emit_get_gather_mask_remain_count() {
// CHECK-NEXT:   int64_t v1 = AscendC::GetGatherMaskRemainCount();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_get_gather_mask_remain_count() {
  %result = ascendc.get_gather_mask_remain_count : i64
  return
}

