// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_tensor_desc(uint64_t v1) {
// CHECK-NEXT:  AscendC::TensorDesc v2 = AscendC::TensorDesc<float>();
// CHECK-NEXT:  uint64_t v3 = v2.GetDim();
// CHECK-NEXT:  uint64_t v4 = v2.GetIndex();
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  int64_t v5 = v2.GetShape(c0_i32);
// CHECK-NEXT:  __gm__ float* v6 = v2.GetDataPtr();
// CHECK-NEXT:  AscendC::GlobalTensor<float> v7 = v2.GetDataObj();
// CHECK-NEXT:  v2.SetShapeAddr((uint64_t* )v1)
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_tensor_desc(%arg0 : ui64) {
  %tensor_desc = ascendc.tensor_desc f32 : !ascendc.tensor_desc
  %dim = ascendc.tensor_desc_get_dim %tensor_desc : !ascendc.tensor_desc, ui64
  %index = ascendc.tensor_desc_get_index %tensor_desc : !ascendc.tensor_desc, ui64
  %c0_i32 = arith.constant 0 : i32
  %shape = ascendc.tensor_desc_get_shape %tensor_desc, %c0_i32 : !ascendc.tensor_desc, i64, i32
  %data_ptr = ascendc.tensor_desc_get_data_ptr %tensor_desc : !ascendc.tensor_desc, memref<?xf32, 22>
  %data_obj = ascendc.tensor_desc_get_data_obj %tensor_desc : !ascendc.tensor_desc, !ascendc.global_tensor<*xf32>
  ascendc.tensor_desc_set_shape_addr %tensor_desc, %arg0 : !ascendc.tensor_desc, ui64
  return
}

// CHECK-LABEL:void emit_list_tensor_desc(__gm__ int8_t* v1, uint32_t v2, uint32_t v3, uint32_t v4, AscendC::TensorDesc v5) {
// CHECK-NEXT:   AscendC::ListTensorDesc v6;
// CHECK-NEXT:   AscendC::ListTensorDesc v7 = AscendC::ListTensorDesc(v1, v2, v3);
// CHECK-NEXT:   v6.Init(v1, v2, v3);
// CHECK-NEXT:   __gm__ int32_t* v8 = v6.GetDataPtr<int32_t>(v4);
// CHECK-NEXT:   uint32_t v9 = v6.GetSize();
// CHECK-NEXT:   v6.GetDesc(v5, v2)
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_list_tensor_desc(%alloc_0 : memref<?xi8, 22>, %arg0 : ui32,  %arg1 : ui32, %arg2 : ui32, %tensor_desc : !ascendc.tensor_desc) {
  %tensor0 = ascendc.list_tensor_desc : !ascendc.list_tensor_desc
  %tensor1 = ascendc.list_tensor_desc_v2 %alloc_0, %arg0, %arg1 : !ascendc.list_tensor_desc, memref<?xi8, 22>, ui32, ui32
  ascendc.list_tensor_desc_init %tensor0, %alloc_0, %arg0, %arg1 : !ascendc.list_tensor_desc, memref<?xi8, 22>, ui32, ui32
  %1 = ascendc.list_tensor_desc_get_data_ptr %tensor0, %arg2 {dtype = i32} : memref<?xi32, 22>
  %2 = ascendc.list_tensor_desc_get_size %tensor0 : !ascendc.list_tensor_desc
  ascendc.list_tensor_desc_get_desc %tensor0, %tensor_desc, %arg0 : !ascendc.list_tensor_desc, !ascendc.tensor_desc, ui32
  return
}
