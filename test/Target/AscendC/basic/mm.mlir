// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_init_const_value(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 128;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   half c2_1992_f16 = 2.19920e+00;
// CHECK-NEXT:   constexpr uint16_t v4 = 1;
// CHECK-NEXT:   constexpr uint16_t v5 = 2;
// CHECK-NEXT:   constexpr uint16_t v6 = 0;
// CHECK-NEXT:   AscendC::LocalTensor<half> v7 = AscendC::LocalTensor<half>(AscendC::TPosition::A1, v3, v2);
// CHECK-NEXT:   AscendC::InitConstValueParams v8{v4, v5, v6, c2_1992_f16};
// CHECK-NEXT:   AscendC::InitConstValue(v7, v8);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_init_const_value(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = "emitc.constant"() <{value = 128 : ui32}> : () -> ui32
  %1 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %cst = arith.constant 2.199220e+00 : f16
  %2 = "emitc.constant"() <{value = 1 : ui16}> : () -> ui16
  %3 = "emitc.constant"() <{value = 2 : ui16}> : () -> ui16
  %4 = "emitc.constant"() <{value = 0 : ui16}> : () -> ui16
  %5 = ascendc.local_tensor_v2 a1, %1, %0 : !ascendc.local_tensor<*xf16>
  %6 = ascendc.construct !ascendc.init_const_value_params(%2, %3, %4, %cst) [ui16, ui16, ui16, f16] : ui16, ui16, ui16, f16
  ascendc.init_const_value %5, %6 : !ascendc.local_tensor<*xf16>, !ascendc.init_const_value_params
  return
}

// CHECK-LABEL: void emit_load_data(AscendC::LocalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, AscendC::GlobalTensor<int32_t> v3, AscendC::LoadData2DParams v4, AscendC::LoadData2DParamsV2 v5, AscendC::LoadData3DParamsV1 v6, AscendC::LoadData3DParamsV2 v7, AscendC::LoadData3DParamsV2Pro v8)
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v4);
// CHECK-NEXT:   AscendC::LoadData(v1, v3, v4);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v5);
// CHECK-NEXT:   AscendC::LoadData(v1, v3, v5);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v6);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v7);
// CHECK-NEXT:   AscendC::LoadData(v1, v2, v8);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_load_data(%arg0 : !ascendc.local_tensor<i32>, %arg1 : !ascendc.local_tensor<i32>, %arg2 : !ascendc.global_tensor<i32>, %arg3 : !ascendc.load_data_2d_params, %arg4 : !ascendc.load_data_2d_params_v2, %arg5 : !ascendc.load_data_3d_params_v1, %arg6 : !ascendc.load_data_3d_params_v2, %arg7 : !ascendc.load_data_3d_params_v2_pro) {
  ascendc.load_data_l0      %arg0, %arg1, %arg3 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_params
  ascendc.load_data_g2l     %arg0, %arg2, %arg3 : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.load_data_2d_params
  ascendc.load_data_l0_v2   %arg0, %arg1, %arg4 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_2d_params_v2
  ascendc.load_data_g2l_v2  %arg0, %arg2, %arg4 : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.load_data_2d_params_v2
  ascendc.load_data_3d_l0_v1 %arg0, %arg1, %arg5 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_3d_params_v1
  ascendc.load_data_3d_l0_v2 %arg0, %arg1, %arg6 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_3d_params_v2
  ascendc.load_data_3d_l0_v2pro %arg0, %arg1, %arg7 : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.load_data_3d_params_v2_pro
  return
}

// CHECK-LABEL:void emit_load_data_with_sparse(__gm__ uint64_t* v1) {
// CHECK-NEXT:  set_ffts_base_addr(*v1);
// CHECK-NEXT:  constexpr uint32_t v2 = 512;
// CHECK-NEXT:  constexpr uint32_t v3 = 0;
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  constexpr int32_t c1_i32 = 1;
// CHECK-NEXT:  constexpr bool c0_i1 = false;
// CHECK-NEXT:  AscendC::LocalTensor<int8_t> v4 = AscendC::LocalTensor<int8_t>(AscendC::TPosition::B2, v3, v2);
// CHECK-NEXT:  AscendC::LocalTensor<int8_t> v5 = AscendC::LocalTensor<int8_t>(AscendC::TPosition::B1, v3, v2);
// CHECK-NEXT:  AscendC::LocalTensor<uint8_t> v6 = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B1, v3, v2);
// CHECK-NEXT:  AscendC::LoadData2DParams v7{static_cast<uint16_t>(c0_i32), static_cast<uint8_t>(c1_i32), static_cast<uint16_t>(c0_i32), static_cast<uint8_t>(c0_i32), static_cast<uint16_t>(c0_i32), c0_i1, static_cast<uint8_t>(c0_i32)};
// CHECK-NEXT:  AscendC::LoadDataWithSparse(v4, v5, v6, v7);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_load_data_with_sparse(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = "emitc.constant"() <{value = 512 : ui32}> : () -> ui32
  %1 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %false = arith.constant false
  %2 = ascendc.local_tensor_v2 b2, %1, %0 : !ascendc.local_tensor<*xi8>
  %3 = ascendc.local_tensor_v2 b1, %1, %0 : !ascendc.local_tensor<*xi8>
  %4 = ascendc.local_tensor_v2 b1, %1, %0 : !ascendc.local_tensor<*xui8>
  %5 = ascendc.construct !ascendc.load_data_2d_params(%c0_i32, %c1_i32, %c0_i32, %c0_i32, %c0_i32, %false, %c0_i32) [ui16, ui8, ui16, ui8, ui16, i1, ui8] : i32, i32, i32, i32, i32, i1, i32
  ascendc.load_data_with_sparse %2, %3, %4, %5 : !ascendc.local_tensor<*xi8>, !ascendc.local_tensor<*xi8>, !ascendc.local_tensor<*xui8>, !ascendc.load_data_2d_params
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

// CHECK-LABEL:void emit_mmad_with_sparse(__gm__ uint64_t* v1) {
// CHECK-NEXT:  set_ffts_base_addr(*v1);
// CHECK-NEXT:  constexpr uint32_t v2 = 400;
// CHECK-NEXT:  constexpr uint32_t v3 = 0;
// CHECK-NEXT:  constexpr int32_t c20_i32 = 20;
// CHECK-NEXT:  constexpr bool c0_i1 = false;
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  AscendC::LocalTensor<int32_t> v4 = AscendC::LocalTensor<int32_t>(AscendC::TPosition::CO1, v3, v2);
// CHECK-NEXT:  AscendC::LocalTensor<int8_t> v5 = AscendC::LocalTensor<int8_t>(AscendC::TPosition::A2, v3, v2);
// CHECK-NEXT:  AscendC::LocalTensor<int8_t> v6 = AscendC::LocalTensor<int8_t>(AscendC::TPosition::B2, v3, v2);
// CHECK-NEXT:  AscendC::MmadParams v7{static_cast<uint16_t>(c20_i32), static_cast<uint16_t>(c20_i32), static_cast<uint16_t>(c20_i32), c0_i1, c0_i32, c0_i1, c0_i1, c0_i1};
// CHECK-NEXT:  AscendC::MmadWithSparse(v4, v5, v6, v7);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_mmad_with_sparse(%arg0: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %0 = "emitc.constant"() <{value = 400 : ui32}> : () -> ui32
  %1 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c20_i32 = arith.constant 20 : i32
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %2 = ascendc.local_tensor_v2 co1, %1, %0 : !ascendc.local_tensor<*xi32>
  %3 = ascendc.local_tensor_v2 a2, %1, %0 : !ascendc.local_tensor<*xi8>
  %4 = ascendc.local_tensor_v2 b2, %1, %0 : !ascendc.local_tensor<*xi8>
  %5 = ascendc.construct !ascendc.mmad_params(%c20_i32, %c20_i32, %c20_i32, %false, %c0_i32, %false, %false, %false) [ui16, ui16, ui16, i1, i32, i1, i1, i1] : i32, i32, i32, i1, i32, i1, i1, i1
  ascendc.mmad_with_sparse %2, %3, %4, %5 : !ascendc.local_tensor<*xi32>, !ascendc.local_tensor<*xi8>, !ascendc.local_tensor<*xi8>, !ascendc.mmad_params
  return
}