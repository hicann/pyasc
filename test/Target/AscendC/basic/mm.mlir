// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_load_data(AscendC::LocalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, AscendC::GlobalTensor<int32_t> v3, AscendC::LoadData2DParams v4, AscendC::LoadData2DParamsV2 v5, AscendC::LoadData3DParamsV2Pro v6)
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v4);
// CHECK-NEXT:   AscendC::LoadData(v1, v3, v4);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v5);
// CHECK-NEXT:   AscendC::LoadData(v1, v3, v5);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_load_data(%arg0 : !ascendc.local_tensor<i32>, %arg1 : !ascendc.local_tensor<i32>, %arg2 : !ascendc.global_tensor<i32>, %arg3 : !ascendc.load_data_2d_params, %arg4 : !ascendc.load_data_2d_params_v2, %arg5 : !ascendc.load_data_3d_params_v2_pro) {
  ascendc.load_data_l0  %arg0, %arg1, %arg3 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_params
  ascendc.load_data_g2l %arg0, %arg2, %arg3 : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.load_data_2d_params
  ascendc.load_data_l0_v2  %arg0, %arg1, %arg4 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_params_v2
  ascendc.load_data_g2l_v2 %arg0, %arg2, %arg4 : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.load_data_2d_params_v2
  ascendc.load_data_3d_l0_v2pro %arg0, %arg1, %arg5 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_3d_params_v2_pro
  return
}

// CHECK-LABEL: void emit_load_data_with_transpose(AscendC::LocalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, AscendC::LoadData2dTransposeParams v3, AscendC::LoadData2dTransposeParamsV2 v4)
// CHECK-NEXT:   AscendC::LoadDataWithTranspose(v1, v2, v3);
// CHECK-NEXT:   AscendC::LoadDataWithTranspose(v1, v2, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_load_data_with_transpose(%v1 : !ascendc.local_tensor<i32>, %v2 : !ascendc.local_tensor<i32>, %v3 : !ascendc.load_data_2d_transpose_params, %v4 : !ascendc.load_data_2d_transpose_params_v2) {
  ascendc.load_data_with_transpose %v1, %v2, %v3 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_transpose_params
  ascendc.load_data_with_transpose_v2 %v1, %v2, %v4 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_transpose_params_v2
  return
}

// CHECK-LABEL: void emit_mmad(AscendC::LocalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, AscendC::LocalTensor<int32_t> v3, AscendC::LocalTensor<int32_t> v4, AscendC::MmadParams v5)
// CHECK-NEXT:   AscendC::Mmad(v1, v2, v3, v5);
// CHECK-NEXT:   AscendC::Mmad(v1, v2, v3, v4, v5);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_mmad(%dst : !ascendc.local_tensor<i32>, %fm : !ascendc.local_tensor<i32>, %filter : !ascendc.local_tensor<i32>, %bias : !ascendc.local_tensor<i32>, %params : !ascendc.mmad_params) {
  ascendc.mmad %dst, %fm, %filter, %params : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.mmad_params
  ascendc.mmad_with_bias %dst, %fm, %filter, %bias, %params : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.mmad_params
  return
}
