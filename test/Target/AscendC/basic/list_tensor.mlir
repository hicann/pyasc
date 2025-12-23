// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_tensor_desc() {
// CHECK-NEXT:  AscendC::TensorDesc v1 = AscendC::TensorDesc<float>();
// CHECK-NEXT:  uint64_t v2 = v1.GetDim();
// CHECK-NEXT:  uint64_t v3 = v1.GetIndex();
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  int64_t v4 = v1.GetShape(c0_i32);
// CHECK-NEXT:  __gm__ float* v5 = v1.GetDataPtr();
// CHECK-NEXT:  AscendC::GlobalTensor<float> v6 = v1.GetDataObj();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_tensor_desc() {
  %tensor_desc = ascendc.tensor_desc f32 : !ascendc.tensor_desc
  %dim = ascendc.tensor_desc_get_dim %tensor_desc : !ascendc.tensor_desc, ui64
  %index = ascendc.tensor_desc_get_index %tensor_desc : !ascendc.tensor_desc, ui64
  %c0_i32 = arith.constant 0 : i32
  %shape = ascendc.tensor_desc_get_shape %tensor_desc, %c0_i32 : !ascendc.tensor_desc, i64, i32
  %data_ptr = ascendc.tensor_desc_get_data_ptr %tensor_desc : !ascendc.tensor_desc, memref<?xf32, 22>
  %data_obj = ascendc.tensor_desc_get_data_obj %tensor_desc : !ascendc.tensor_desc, !ascendc.global_tensor<*xf32>
  return
}