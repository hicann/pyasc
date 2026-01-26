// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_local_tensor(AscendC::LocalTensor<float> v1, uint32_t v2, int32_t v3, uint32_t v4) {
// CHECK-NEXT:   AscendC::LocalTensor<float> v5 = v1[v4];
// CHECK-NEXT:   uint64_t v6 = v1.GetPhyAddr();
// CHECK-NEXT:   v1.SetBufferLen(v2);
// CHECK-NEXT:   v1.SetSize(v2);
// CHECK-NEXT:   v7 = v1.GetPosition();
// CHECK-NEXT:   v8 = v1.GetLength();
// CHECK-NEXT:   v9 = v1.GetSize();
// CHECK-NEXT:   v1.SetUserTag(v3);
// CHECK-NEXT:   v10 = v1.GetUserTag();
// CHECK-NEXT:   AscendC::LocalTensor<int32_t> v11 = v1.ReinterpretCast<int32_t>();
// CHECK-NEXT:   v12 = v11.GetSize();
// CHECK-NEXT:   v13 = v1.GetValue(v2);
// CHECK-NEXT:   v1.SetValue(v2, v13);
// CHECK-NEXT:   AscendC::LocalTensor<float> v14 = v1(v4);
// CHECK-NEXT:   AscendC::ShapeInfo v15 = v1.GetShapeInfo();
// CHECK-NEXT:   v1.SetShapeInfo(v15);
// CHECK-NEXT:   v1.SetAddrWithOffset(v14, v2);
// CHECK-NEXT:   AscendC::LocalTensor<float> v16 = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, v2, v2);
// CHECK-NEXT:   AscendC::LocalTensor<float> v17;
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_local_tensor(%tensor : !ascendc.local_tensor<*xf32>, %len : ui32,  %tag : i32, %arg1 : index) {
  %subindex = ascendc.local_tensor.subindex %tensor[%arg1] : !ascendc.local_tensor<*xf32>, index, !ascendc.local_tensor<*xf32>
  %ub_ptr = ascendc.local_tensor.get_phy_addr %tensor : !ascendc.local_tensor<*xf32>, ui64
  ascendc.local_tensor.set_buffer_len %tensor, %len : !ascendc.local_tensor<*xf32>, ui32
  ascendc.local_tensor.set_size %tensor, %len : !ascendc.local_tensor<*xf32>, ui32
  %1 = ascendc.local_tensor.get_position %tensor : !ascendc.local_tensor<*xf32>, i32
  %2 = ascendc.local_tensor.get_length %tensor : !ascendc.local_tensor<*xf32>, ui32
  %3 = ascendc.local_tensor.get_size %tensor : !ascendc.local_tensor<*xf32>, i32
  ascendc.local_tensor.set_user_tag %tensor, %tag: !ascendc.local_tensor<*xf32>, i32
  %4 = ascendc.local_tensor.get_user_tag %tensor : !ascendc.local_tensor<*xf32>, i32
  %5 = ascendc.reinterpret_cast %tensor : !ascendc.local_tensor<*xf32> to !ascendc.local_tensor<*xi32>
  %6 = ascendc.local_tensor.get_size %5 : !ascendc.local_tensor<*xi32>, i32
  %7 = ascendc.local_tensor.get_value %tensor, %len: !ascendc.local_tensor<*xf32>, ui32, f32
  ascendc.local_tensor.set_value %tensor, %len, %7 : !ascendc.local_tensor<*xf32>, ui32, f32
  %bracket2 = ascendc.local_tensor.bracket %tensor(%arg1) : !ascendc.local_tensor<*xf32>, index, !ascendc.local_tensor<*xf32>
  %8 = ascendc.local_tensor.get_shape_info %tensor : !ascendc.local_tensor<*xf32>, !ascendc.shape_info
  ascendc.local_tensor.set_shape_info %tensor, %8 : !ascendc.local_tensor<*xf32>, !ascendc.shape_info
  ascendc.local_tensor.set_addr_with_offset %tensor, %bracket2, %len : !ascendc.local_tensor<*xf32>, !ascendc.local_tensor<*xf32>, ui32
  %9 = ascendc.local_tensor_v2 vecin, %len, %len : !ascendc.local_tensor<*xf32>
  %10 = ascendc.local_tensor : !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL:void emit_global_tensor(__gm__ float* v1, int32_t v2, uint32_t v3, uint64_t v4) {
// CHECK-NEXT:   AscendC::GlobalTensor<int32_t> v5;
// CHECK-NEXT:   v5.SetGlobalBuffer(v1);
// CHECK-NEXT:   v5.SetGlobalBuffer(v1, v2);
// CHECK-NEXT:   AscendC::GlobalTensor<int32_t> v6 = v5[v3];
// CHECK-NEXT:   __gm__ float* v7 = v5.GetPhyAddr();
// CHECK-NEXT:   __gm__ float* v8 = v5.GetPhyAddr(v4);
// CHECK-NEXT:   AscendC::ShapeInfo v9 = v5.GetShapeInfo();
// CHECK-NEXT:   v10 = v5.GetSize();
// CHECK-NEXT:   v11 = v5.GetValue(v4);
// CHECK-NEXT:   v5.SetL2CacheHint<AscendC::CacheRwMode::RW>(AscendC::CacheMode::CACHE_MODE_DISABLE);
// CHECK-NEXT:   v5.SetShapeInfo(v9);
// CHECK-NEXT:   v5.SetValue(v4, v2);
// CHECK-NEXT:   AscendC::GlobalTensor<int32_t> v12 = v5(v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_global_tensor(%alloc_0 : memref<1024xf32, 22>, %arg0 : i32, %arg1: index, %arg2: ui64) {
  %global_tensor = ascendc.global_tensor : !ascendc.global_tensor<i32>
  ascendc.global_tensor.set_global_buffer %global_tensor, %alloc_0 : !ascendc.global_tensor<i32>, memref<1024xf32, 22>
  ascendc.global_tensor.set_global_buffer %global_tensor, %alloc_0, %arg0 : !ascendc.global_tensor<i32>, memref<1024xf32, 22>, i32
  %subindex = ascendc.global_tensor.subindex %global_tensor[%arg1] : !ascendc.global_tensor<i32>, index, !ascendc.global_tensor<i32>
  %1 = ascendc.global_tensor.get_phy_addr %global_tensor : !ascendc.global_tensor<i32>,  memref<1024xf32, 22>
  %2 = ascendc.global_tensor.get_phy_addr %global_tensor, %arg2 : !ascendc.global_tensor<i32>, memref<1024xf32, 22>, ui64
  %3 = ascendc.global_tensor.get_shape_info %global_tensor : !ascendc.global_tensor<i32>, !ascendc.shape_info
  %4 = ascendc.global_tensor.get_size %global_tensor : !ascendc.global_tensor<i32>
  %5 = ascendc.global_tensor.get_value %global_tensor, %arg2 : !ascendc.global_tensor<i32>, ui64, f32
  ascendc.global_tensor.set_l2_cache_hint %global_tensor{mode = 0 : i32, rwMode = 3 : i32} : !ascendc.global_tensor<i32>
  ascendc.global_tensor.set_shape_info %global_tensor, %3 : !ascendc.global_tensor<i32>, !ascendc.shape_info
  ascendc.global_tensor.set_value %global_tensor, %arg2, %arg0 : !ascendc.global_tensor<i32>, ui64, i32
  %bracket3 = ascendc.global_tensor.bracket %global_tensor(%arg1) : !ascendc.global_tensor<i32>, index, !ascendc.global_tensor<i32>
  return
}

// CHECK-LABEL:void emit_construct(int8_t v1, int16_t v2, int32_t v3, int64_t v4, float v5) {
// CHECK-NEXT:   constexpr bool c0_i1 = false;
// CHECK-NEXT:   AscendC::DataCopyParams v6;
// CHECK-NEXT:   AscendC::DataCopyParams v7{v2, v2, v2, v2};
// CHECK-NEXT:   AscendC::DataFormat v8{v1};
// CHECK-NEXT:   AscendC::FixpipeParams<int32_t> v9;
// CHECK-NEXT:   AscendC::FixpipeParams<half> v10{v2, v2, v2, v3};
// CHECK-NEXT:   AscendC::LoadData2DParams v11{v2, v1, v2, v1, v2, v1, v1};
// CHECK-NEXT:   AscendC::LoadData3DParamsV2 v12{v1, v2, v2, v2, v2, v2, v2, v2, v1, v1, v1, v1, v1, v1, v1, v1, v5};
// CHECK-NEXT:   AscendC::MmadParams v13{v3, v3, v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::QuantParams v14;
// CHECK-NEXT:   AscendC::QuantParams v15{v3};
// CHECK-NEXT:   AscendC::QuantParams v16{v3, v4};
// CHECK-NEXT:   SoftMaxTiling v17;
// CHECK-NEXT:   AscendC::DataCopyEnhancedParams v18;
// CHECK-NEXT:   AscendC::DataCopyEnhancedParams v19{v1, v1, v4, v1, v1, v3, v4};
// CHECK-NEXT:   AscendC::BinaryRepeatParams v20{v1, v1, v1, v1, v1, v1};
// CHECK-NEXT:   AscendC::UnaryRepeatParams v21{v1, v1, v1, v1};
// CHECK-NEXT:   AscendC::UnaryRepeatParams v22{v1, static_cast<uint32_t>(v2), reinterpret_cast<uint8_t*>(v4)};
// CHECK-NEXT:   AscendC::Nd2NzParams v23;
// CHECK-NEXT:   AscendC::Nd2NzParams v24{v2, v2, v3, v4, v4, v2, v2, v3};
// CHECK-NEXT:   AscendC::Nz2NdParams v25;
// CHECK-NEXT:   AscendC::Nz2NdParams v26{v2, v2, v3};
// CHECK-NEXT:   MatmulConfig v27;
// CHECK-NEXT:   AscendC::GatherMaskParams v28{v2, v2, v2, v2};
// CHECK-NEXT:   AscendC::GatherMaskParams v29;
// CHECK-NEXT:   AscendC::GatherRepeatParams v30{v1, v1};
// CHECK-NEXT:   AscendC::Nz2NdParamsFull v31;
// CHECK-NEXT:   AscendC::Nz2NdParamsFull v32{v2, v2, v2, v2, v2, v2, v2};
// CHECK-NEXT:   AscendC::DataCopyCO12DstParams v33;
// CHECK-NEXT:   AscendC::BrcbRepeatParams v34{v1, v1};
// CHECK-NEXT:   AscendC::TransDataTo5HDParams v35;
// CHECK-NEXT:   AscendC::TransDataTo5HDParams v36{c0_i1, c0_i1, static_cast<uint8_t>(v3), static_cast<uint16_t>(v3), static_cast<uint16_t>(v3)};
// CHECK-NEXT:   AscendC::TransposeParamsExt v37;
// CHECK-NEXT:   AscendC::TransposeParamsExt v38{static_cast<uint16_t>(v3), static_cast<uint16_t>(v3), static_cast<uint16_t>(v3), static_cast<uint16_t>(v3), static_cast<AscendC::TransposeType>(v3)};
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_construct(%arg0: i8, %arg1: i16, %arg2: i32, %arg3: i64, %arg4: f32) {
  %false = arith.constant false
  %0 = ascendc.construct !ascendc.data_copy_params()
  %1 = ascendc.construct !ascendc.data_copy_params(%arg1, %arg1, %arg1, %arg1) : i16, i16, i16, i16
  %2 = ascendc.construct !ascendc.data_format(%arg0) : i8
  %3 = ascendc.construct !ascendc.fixpipe_params<i32>()
  %4 = ascendc.construct !ascendc.fixpipe_params<f16>(%arg1, %arg1, %arg1, %arg2) : i16, i16, i16, i32
  %5 = ascendc.construct !ascendc.load_data_2d_params(%arg1, %arg0, %arg1, %arg0, %arg1, %arg0, %arg0) : i16, i8, i16, i8, i16, i8, i8
  %6 = ascendc.construct !ascendc.load_data_3d_params_v2<f32>(%arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg4) : i8, i16, i16, i16, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, f32
  %7 = ascendc.construct !ascendc.mmad_params(%arg2, %arg2, %arg2, %arg2, %arg2, %arg2) : i32, i32, i32, i32, i32, i32
  %8 = ascendc.construct !ascendc.quant_params()
  %9 = ascendc.construct !ascendc.quant_params(%arg2) : i32
  %10 = ascendc.construct !ascendc.quant_params(%arg2, %arg3) : i32, i64
  %11 = ascendc.construct !ascendc.softmax_tiling()
  %12 = ascendc.construct !ascendc.data_copy_enhanced_params()
  %13 = ascendc.construct !ascendc.data_copy_enhanced_params(%arg0, %arg0, %arg3, %arg0, %arg0, %arg2, %arg3) : i8, i8, i64, i8, i8, i32, i64
  %14 = ascendc.construct !ascendc.binary_repeat_params(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) : i8, i8, i8, i8, i8, i8
  %15 = ascendc.construct !ascendc.unary_repeat_params(%arg0, %arg0, %arg0, %arg0) : i8, i8, i8, i8
  %16 = ascendc.construct !ascendc.unary_repeat_params(%arg0, %arg1, %arg3) [i8, ui32, memref<?xui8>] : i8, i16, i64
  %17 = ascendc.construct !ascendc.nd2nz_params()
  %18 = ascendc.construct !ascendc.nd2nz_params(%arg1, %arg1, %arg2, %arg3, %arg3, %arg1, %arg1, %arg2) : i16, i16, i32, i64, i64, i16, i16, i32
  %19 = ascendc.construct !ascendc.nz2nd_params()
  %20 = ascendc.construct !ascendc.nz2nd_params(%arg1, %arg1, %arg2) : i16, i16, i32
  %21 = ascendc.construct !ascendc.matmul_config()
  %22 = ascendc.construct !ascendc.gather_mask_params(%arg1, %arg1, %arg1, %arg1) : i16, i16, i16, i16
  %23 = ascendc.construct !ascendc.gather_mask_params()
  %24 = ascendc.construct !ascendc.gather_repeat_params(%arg0, %arg0) : i8, i8
  %25 = ascendc.construct !ascendc.nz2nd_params_full()
  %26 = ascendc.construct !ascendc.nz2nd_params_full(%arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1) : i16, i16, i16, i16, i16, i16, i16
  %27 = ascendc.construct !ascendc.data_copy_co12dst_params()
  %28 = ascendc.construct !ascendc.brcb_repeat_params(%arg0, %arg0) : i8, i8
  %29 = ascendc.construct !ascendc.trans_data_to_5hd_params()
  %30 = ascendc.construct !ascendc.trans_data_to_5hd_params(%false, %false, %arg2, %arg2, %arg2) [i1, i1, ui8, ui16, ui16] : i1, i1, i32, i32, i32
  %31 = ascendc.construct !ascendc.transpose_params_ext()
  %32 = ascendc.construct !ascendc.transpose_params_ext(%arg2, %arg2, %arg2, %arg2, %arg2) [ui16, ui16, ui16, ui16, !ascendc.transpose_type] : i32, i32, i32, i32, i32
  return
}

// CHECK-LABEL:void emit_shape_info(int8_t v1, int32_t* v2, int8_t v3, int32_t* v4, int8_t v5, AscendC::LocalTensor<half> v6) {
// CHECK-NEXT:  AscendC::ShapeInfo v7{v1, v2, v3, v4, v5};
// CHECK-NEXT:  AscendC::ShapeInfo v8{v1, v2};
// CHECK-NEXT:  v6.SetShapeInfo(v7);
// CHECK-NEXT:  v6.SetShapeInfo(v8);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_shape_info(%arg0: i8, %arg1: memref<?xi32>, %arg2: i8, %arg3: memref<?xi32>, %arg4: i8, %arg5: !ascendc.local_tensor<*xf16>) {
  %0 = ascendc.construct !ascendc.shape_info (%arg0, %arg1, %arg2, %arg3, %arg4) : i8, memref<?xi32>, i8, memref<?xi32>, i8
  %1 = ascendc.construct !ascendc.shape_info (%arg0, %arg1) : i8, memref<?xi32>
  ascendc.local_tensor.set_shape_info %arg5, %0 : !ascendc.local_tensor<*xf16>, !ascendc.shape_info
  ascendc.local_tensor.set_shape_info %arg5, %1 : !ascendc.local_tensor<*xf16>, !ascendc.shape_info
  return
}
