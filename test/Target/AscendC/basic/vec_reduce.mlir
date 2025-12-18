// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_block_reduce_sum(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::BlockReduceSum(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::BlockReduceSum(v1, v2, v3, v1_mask_list0, v5, v6, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_block_reduce_sum(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %repeat: i32, %mask: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.block_reduce_sum_l0 %dst, %src, %repeat, %mask, %dstRepStride, %srcBlkStride, %srcRepStride : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, i32, i32
  ascendc.block_reduce_sum_l1 %dst, %src, %repeat, %maskArray1_0, %maskArray1_1, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_block_reduce_max(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::BlockReduceMax(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::BlockReduceMax(v1, v2, v3, v1_mask_list0, v5, v6, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_block_reduce_max(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %repeat: i32, %mask: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.block_reduce_max_l0 %dst, %src, %repeat, %mask, %dstRepStride, %srcBlkStride, %srcRepStride : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, i32, i32
  ascendc.block_reduce_max_l1 %dst, %src, %repeat, %maskArray1_0, %maskArray1_1, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_block_reduce_min(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::BlockReduceMin(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::BlockReduceMin(v1, v2, v3, v1_mask_list0, v5, v6, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_block_reduce_min(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %repeat: i32, %mask: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.block_reduce_min_l0 %dst, %src, %repeat, %mask, %dstRepStride, %srcBlkStride, %srcRepStride : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, i32, i32, i32, i32
  ascendc.block_reduce_min_l1 %dst, %src, %repeat, %maskArray1_0, %maskArray1_1, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, ui64, ui64, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_pair_reduce_sum(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::PairReduceSum(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::PairReduceSum(v1, v2, v3, v1_mask_list0, v5, v6, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_pair_reduce_sum(%dst: !ascendc.local_tensor<*xf16>, %src: !ascendc.local_tensor<*xf16>, %repeatTime: i32, %mask0: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.pair_reduce_sum_l0 %dst, %src, %repeatTime, %mask0, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32, i32
  ascendc.pair_reduce_sum_l1 %dst, %src, %repeatTime, %maskArray1_0, %maskArray1_1, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, ui64, ui64, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_reduce_max() {
// CHECK-NEXT:   AscendC::LocalTensor<float> v1;
// CHECK-NEXT:   AscendC::LocalTensor<float> v2;
// CHECK-NEXT:   AscendC::LocalTensor<float> v3;
// CHECK-NEXT:   constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:   constexpr int64_t c1_i64 = 1;
// CHECK-NEXT:   constexpr int64_t c2_i64 = 2;
// CHECK-NEXT:   constexpr int32_t c3_i32 = 3;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   constexpr int32_t c5_i32 = 5;
// CHECK-NEXT:   constexpr int32_t c6_i32 = 6;
// CHECK-NEXT:   AscendC::ReduceMax(v1, v2, v3, c0_i32, c3_i32, c4_i32, c5_i32);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {c1_i64, c2_i64};
// CHECK-NEXT:   AscendC::ReduceMax(v1, v2, v3, v1_mask_list0, c3_i32, c4_i32, c5_i32);
// CHECK-NEXT:   AscendC::ReduceMax(v1, v2, v3, c6_i32, c5_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_reduce_max() {
  %dst = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %src = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %sharedTmpBuffer = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  
  %mask = arith.constant 0 : i32
  %mask_var_0 = arith.constant 1 : i64
  %mask_var_1 = arith.constant 2 : i64
  %repeatTime = arith.constant 3 : i32
  %srcRepStride = arith.constant 4 : i32
  %calIndex = arith.constant 5 : i32
  %count = arith.constant 6 : i32

  ascendc.reduce_max_l0 %dst, %src, %sharedTmpBuffer, %mask, %repeatTime, %srcRepStride, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32, i32, i32, i32

  ascendc.reduce_max_l1 %dst, %src, %sharedTmpBuffer, %mask_var_0, %mask_var_1, %repeatTime, %srcRepStride, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i64, i64, i32, i32, i32

  ascendc.reduce_max_l2 %dst, %src, %sharedTmpBuffer, %count, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32, i32

  return
}

// CHECK-LABEL:void emit_reduce_min() {
// CHECK-NEXT:   AscendC::LocalTensor<float> v1;
// CHECK-NEXT:   AscendC::LocalTensor<float> v2;
// CHECK-NEXT:   AscendC::LocalTensor<float> v3;
// CHECK-NEXT:   constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:   constexpr int64_t c1_i64 = 1;
// CHECK-NEXT:   constexpr int64_t c2_i64 = 2;
// CHECK-NEXT:   constexpr int32_t c3_i32 = 3;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   constexpr int32_t c5_i32 = 5;
// CHECK-NEXT:   constexpr int32_t c6_i32 = 6;
// CHECK-NEXT:   AscendC::ReduceMin(v1, v2, v3, c0_i32, c3_i32, c4_i32, c5_i32);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {c1_i64, c2_i64};
// CHECK-NEXT:   AscendC::ReduceMin(v1, v2, v3, v1_mask_list0, c3_i32, c4_i32, c5_i32);
// CHECK-NEXT:   AscendC::ReduceMin(v1, v2, v3, c6_i32, c5_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_reduce_min() {
  %dst = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %src = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %sharedTmpBuffer = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  
  %mask = arith.constant 0 : i32
  %mask_var_0 = arith.constant 1 : i64
  %mask_var_1 = arith.constant 2 : i64
  %repeatTime = arith.constant 3 : i32
  %srcRepStride = arith.constant 4 : i32
  %calIndex = arith.constant 5 : i32
  %count = arith.constant 6 : i32

  ascendc.reduce_min_l0 %dst, %src, %sharedTmpBuffer, %mask, %repeatTime, %srcRepStride, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32, i32, i32, i32

  ascendc.reduce_min_l1 %dst, %src, %sharedTmpBuffer, %mask_var_0, %mask_var_1, %repeatTime, %srcRepStride, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i64, i64, i32, i32, i32

  ascendc.reduce_min_l2 %dst, %src, %sharedTmpBuffer, %count, %calIndex : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32, i32

  return
}

// CHECK-LABEL:void emit_reduce_sum() {
// CHECK-NEXT:   AscendC::LocalTensor<float> v1;
// CHECK-NEXT:   AscendC::LocalTensor<float> v2;
// CHECK-NEXT:   AscendC::LocalTensor<float> v3;
// CHECK-NEXT:   constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:   constexpr int64_t c1_i64 = 1;
// CHECK-NEXT:   constexpr int64_t c2_i64 = 2;
// CHECK-NEXT:   constexpr int32_t c3_i32 = 3;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   constexpr int32_t c5_i32 = 5;
// CHECK-NEXT:   AscendC::ReduceSum(v1, v2, v3, c0_i32, c3_i32, c4_i32);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {c1_i64, c2_i64};
// CHECK-NEXT:   AscendC::ReduceSum(v1, v2, v3, v1_mask_list0, c3_i32, c4_i32);
// CHECK-NEXT:   AscendC::ReduceSum(v1, v2, v3, c5_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_reduce_sum() {
  %dst = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %src = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  %sharedTmpBuffer = ascendc.local_tensor : !ascendc.local_tensor<1024xf32>
  
  %mask = arith.constant 0 : i32
  %mask_var_0 = arith.constant 1 : i64
  %mask_var_1 = arith.constant 2 : i64
  %repeatTime = arith.constant 3 : i32
  %srcRepStride = arith.constant 4 : i32
  %count = arith.constant 5 : i32

  ascendc.reduce_sum_l0 %dst, %src, %sharedTmpBuffer, %mask, %repeatTime, %srcRepStride : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32, i32, i32

  ascendc.reduce_sum_l1 %dst, %src, %sharedTmpBuffer, %mask_var_0, %mask_var_1, %repeatTime, %srcRepStride : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i64, i64, i32, i32

  ascendc.reduce_sum_l2 %dst, %src, %sharedTmpBuffer, %count : 
    !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>,
    i32

  return
}

// CHECK-LABEL:void emit_repeat_reduce_sum(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8) {
// CHECK-NEXT:   AscendC::RepeatReduceSum(v1, v2, v3, v4, v5, v6, v7, v8);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_repeat_reduce_sum(%dst: !ascendc.local_tensor<*xf16>, %src: !ascendc.local_tensor<*xf16>, %repeatTime: i32, %mask: i32, %dstBlkStride: i32, %srcBlkStride: i32, %dstRepStride: i32, %srcRepStride: i32) {
  ascendc.repeat_reduce_sum_l0 %dst, %src, %repeatTime, %mask, %dstBlkStride, %srcBlkStride, %dstRepStride, %srcRepStride : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_whole_reduce_max(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:  AscendC::WholeReduceMax(v1, v2, v3, v4, v5, v6, v7, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::WholeReduceMax(v1, v2, v1_mask_list0, v4, v5, v6, v7, AscendC::ReduceOrder::ORDER_VALUE_INDEX);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_whole_reduce_max(%dst: !ascendc.local_tensor<*xf16>, %src: !ascendc.local_tensor<*xf16>, %mask0: i32, %repeatTime: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.whole_reduce_max_l0 %dst, %src, %mask0, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride {order = 1 : i32} : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32, i32
  ascendc.whole_reduce_max_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride {order = 0 : i32} : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, ui64, ui64, i32, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_whole_reduce_min(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:  AscendC::WholeReduceMin(v1, v2, v3, v4, v5, v6, v7, AscendC::ReduceOrder::ORDER_INDEX_VALUE);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::WholeReduceMin(v1, v2, v1_mask_list0, v4, v5, v6, v7, AscendC::ReduceOrder::ORDER_VALUE_INDEX);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_whole_reduce_min(%dst: !ascendc.local_tensor<*xf16>, %src: !ascendc.local_tensor<*xf16>, %mask0: i32, %repeatTime: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.whole_reduce_min_l0 %dst, %src, %mask0, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride {order = 1 : i32} : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32, i32
  ascendc.whole_reduce_min_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride {order = 0 : i32} : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, ui64, ui64, i32, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_whole_reduce_sum(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::WholeReduceSum(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:  uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:  AscendC::WholeReduceSum(v1, v2, v1_mask_list0, v4, v5, v6, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_whole_reduce_sum(%dst: !ascendc.local_tensor<*xf16>, %src: !ascendc.local_tensor<*xf16>, %mask0: i32, %repeatTime: i32, %dstRepStride: i32, %srcBlkStride: i32, %srcRepStride: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.whole_reduce_sum_l0 %dst, %src, %mask0, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32, i32, i32, i32
  ascendc.whole_reduce_sum_l1 %dst, %src, %maskArray1_0, %maskArray1_1, %repeatTime, %dstRepStride, %srcBlkStride, %srcRepStride : 
      !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, ui64, ui64, i32, i32, i32, i32
  return
}
