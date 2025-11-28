// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_duplicate(AscendC::LocalTensor<float> v1, float v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7, uint64_t v8, uint64_t v9) {
// CHECK-NEXT:   AscendC::Duplicate<float, 0>(v1, v2, v3, v4, v5, v6);
// CHECK-NEXT:   uint64_t v1_mask_list0[] = {v8, v9};
// CHECK-NEXT:   AscendC::Duplicate<float, 0>(v1, v2, v1_mask_list0, v4, v5, v6);
// CHECK-NEXT:   AscendC::Duplicate(v1, v2, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_duplicate(%dst: !ascendc.local_tensor<1024xf32>, %scalar : f32, %mask : i32, %repeatTimes : i32, %dstBlkStride : i32, %dstRepStride : i32, %calCount : i32, %maskArray1_0: ui64, %maskArray1_1: ui64) {
  ascendc.duplicate_l0 %dst, %scalar, %mask, %repeatTimes, %dstBlkStride, %dstRepStride : !ascendc.local_tensor<1024xf32>, f32, i32, i32, i32, i32
  ascendc.duplicate_l1 %dst, %scalar, %maskArray1_0, %maskArray1_1, %repeatTimes, %dstBlkStride, %dstRepStride : !ascendc.local_tensor<1024xf32>, f32, ui64, ui64, i32, i32, i32
  ascendc.duplicate_l2 %dst, %scalar, %calCount : !ascendc.local_tensor<1024xf32>, f32, i32
  return
}
