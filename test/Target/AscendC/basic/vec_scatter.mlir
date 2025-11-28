// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_scatter(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, AscendC::LocalTensor<int32_t> v3, uint32_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, uint64_t v9, uint64_t v10) {
// CHECK-NEXT:   AscendC::Scatter(v1, v2, v3, v4, v5, v6, v7);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v9, v10};
// CHECK-NEXT:   AscendC::Scatter(v1, v2, v3, v4, v1_mask_list0, v6, v7);
// CHECK-NEXT:   AscendC::Scatter(v1, v2, v3, v4, v8);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_scatter(%dst: !ascendc.local_tensor<1024xf32>, %src: !ascendc.local_tensor<1024xf32>, %dstOffset: !ascendc.local_tensor<1024xi32>, %dstBase: ui32, %mask: i32, %repeatTimes: i32, %srcRepStride: i32, %count: i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.scatter_l0 %dst, %src, %dstOffset, %dstBase, %mask, %repeatTimes, %srcRepStride : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xi32>, ui32, i32, i32, i32
  ascendc.scatter_l1 %dst, %src, %dstOffset, %dstBase, %maskArray1_0, %maskArray1_1, %repeatTimes, %srcRepStride : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xi32>, ui32, ui64, ui64, i32, i32
  ascendc.scatter_l2 %dst, %src, %dstOffset, %dstBase, %count : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xi32>, ui32, i32
  return
}
