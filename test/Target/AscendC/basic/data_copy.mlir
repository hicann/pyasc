// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_copy(AscendC::LocalTensor<half> v1, AscendC::LocalTensor<half> v2, uint64_t v3, uint8_t v4, uint8_t v5, uint64_t v6, uint16_t v7, uint16_t v8) {
// CHECK-NEXT:    constexpr int8_t c1_i8 = 1;
// CHECK-NEXT:    AscendC::CopyRepeatParams v9{v7, v7, v8, v8};
// CHECK-NEXT:    AscendC::Copy<half, c1_i8>(v1, v2, v3, v4, v9)
// CHECK-NEXT:    uint64_t v1_mask[] = {v6, v6};
// CHECK-NEXT:    AscendC::Copy<half, c1_i8>(v1, v2, v1_mask, v5, v9);
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
func.func @emit_copy(%dst : !ascendc.local_tensor<*xf16>, %src : !ascendc.local_tensor<*xf16>, 
                    %mask_cont : ui64, %repeat_cont : ui8, %repeat_bit : ui8, %mask_bit : ui64,
                    %stride : ui16, %rep_size : ui16) {
  %is_set_mask_true = arith.constant 1 : i8
  %params = ascendc.construct !ascendc.copy_repeat_params(%stride, %stride, %rep_size, %rep_size) [ui16, ui16, ui16, ui16] : ui16, ui16, ui16, ui16
  "ascendc.copy_l1"(%dst, %src, %mask_cont, %repeat_cont, %params, %is_set_mask_true) 
    : (!ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, ui64, ui8, !ascendc.copy_repeat_params, i8) -> ()
  "ascendc.copy_l0"(%dst, %src, %mask_bit, %mask_bit, %repeat_bit, %params, %is_set_mask_true)
    : (!ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, ui64, ui64, ui8, !ascendc.copy_repeat_params, i8) -> ()
  return
}

// CHECK-LABEL: void emit_data_copy(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, AscendC::DataCopyParams v3, AscendC::DataCopyEnhancedParams v4, AscendC::Nd2NzParams v5, int32_t v6, AscendC::Nz2NdParamsFull v7, AscendC::DataCopyCO12DstParams v8)
// CHECK-NEXT: AscendC::DataCopy(v1, v2, v3);
// CHECK-NEXT: AscendC::DataCopy(v1, v2, v6);
// CHECK-NEXT: AscendC::DataCopy(v1, v2, v3, v4);
// CHECK-NEXT: AscendC::DataCopy(v2, v1, v5);
// CHECK-NEXT: AscendC::DataCopy(v2, v1, v5);
// CHECK-NEXT: AscendC::DataCopy(v1, v2, v7);
// CHECK-NEXT: AscendC::DataCopy(v1, v2, v8);
// CHECK-NEXT: return;
func.func @emit_data_copy(%arg0 : !ascendc.global_tensor<i32>, %arg1 : !ascendc.local_tensor<i32>, %arg2 : !ascendc.data_copy_params, %arg3 : !ascendc.data_copy_enhanced_params, %arg4: !ascendc.nd2nz_params, %arg5 : i32, %arg6: !ascendc.nz2nd_params_full, %arg7: !ascendc.data_copy_co12dst_params) {
  ascendc.data_copy_l0 %arg0, %arg1, %arg2 : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_params
  ascendc.data_copy_l2 %arg0, %arg1, %arg5 : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, i32
  ascendc.data_copy_enhanced %arg0, %arg1, %arg2, %arg3 : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_params, !ascendc.data_copy_enhanced_params
  ascendc.data_copy_nd2nz %arg1, %arg0, %arg4 : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.nd2nz_params
  ascendc.data_copy_nd2nz %arg1, %arg0, %arg4 {enableSmall} : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.nd2nz_params
  ascendc.data_copy_nz2nd %arg0, %arg1, %arg6 : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.nz2nd_params_full
  ascendc.data_copy_co12dst %arg0, %arg1, %arg7 : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_co12dst_params
  return
}

// CHECK-LABEL:void emit_data_copy_pad(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, int32_t v3) {
// CHECK-NEXT:   AscendC::DataCopyParams v4{v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::DataCopyPadParams v5{v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::Nd2NzParams v6{v3, v3, v3, v3, v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::DataCopyPad(v2, v1, v4, v5);
// CHECK-NEXT:   AscendC::DataCopyPad(v1, v2, v4);
// CHECK-NEXT:   AscendC::DataCopyPad(v2, v2, v4, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_data_copy_pad(%global_tensor : !ascendc.global_tensor<i32>, %local_tensor : !ascendc.local_tensor<i32>, %c0_i32 : i32) {
  %data_copy_params = ascendc.construct !ascendc.data_copy_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
  %pad_params = ascendc.construct !ascendc.data_copy_pad_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
  %nd2nz_params = ascendc.construct !ascendc.nd2nz_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32, i32, i32, i32, i32
  ascendc.data_copy_pad_l0 %local_tensor, %global_tensor, %data_copy_params, %pad_params : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.data_copy_params, !ascendc.data_copy_pad_params
  ascendc.data_copy_pad_l2 %global_tensor, %local_tensor, %data_copy_params : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_params
  ascendc.data_copy_pad_nd2nz %local_tensor, %local_tensor, %data_copy_params, %nd2nz_params : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_params, !ascendc.nd2nz_params
  return
}

// CHECK-LABEL:void emit_data_copy_pad_ext(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, int32_t v3) {
// CHECK-NEXT:   AscendC::DataCopyExtParams v4{v3, v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::DataCopyPadExtParams<int32_t> v5{v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::Nd2NzParams v6{v3, v3, v3, v3, v3, v3, v3, v3};
// CHECK-NEXT:   AscendC::DataCopyPad(v2, v1, v4, v5);
// CHECK-NEXT:   AscendC::DataCopyPad(v1, v2, v4);
// CHECK-NEXT:   AscendC::DataCopyPad(v2, v2, v4, v6);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_data_copy_pad_ext(%global_tensor : !ascendc.global_tensor<i32>, %local_tensor : !ascendc.local_tensor<i32>, %c0_i32 : i32) {
  %data_copy_ext_params = ascendc.construct !ascendc.data_copy_ext_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32, i32
  %pad_ext_params = ascendc.construct !ascendc.data_copy_pad_ext_params<i32>(%c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32
  %nd2nz_params = ascendc.construct !ascendc.nd2nz_params(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) : i32, i32, i32, i32, i32, i32, i32, i32
  ascendc.data_copy_pad_l0_ext %local_tensor, %global_tensor, %data_copy_ext_params, %pad_ext_params : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.data_copy_ext_params, !ascendc.data_copy_pad_ext_params<i32>
  ascendc.data_copy_pad_l2_ext %global_tensor, %local_tensor, %data_copy_ext_params : !ascendc.global_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_ext_params
  ascendc.data_copy_pad_nd2nz_ext %local_tensor, %local_tensor, %data_copy_ext_params, %nd2nz_params : !ascendc.local_tensor<i32>, !ascendc.local_tensor<i32>, !ascendc.data_copy_ext_params, !ascendc.nd2nz_params
  return
}

// CHECK-LABEL:void emit_data_copy_slice(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2, int32_t v3, int32_t v4, uint32_t v5) {
// CHECK-NEXT:  AscendC::SliceInfo v6{static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v3)};
// CHECK-NEXT:  AscendC::SliceInfo v7{static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v3), static_cast<uint32_t>(v4)};
// CHECK-NEXT:  AscendC::SliceInfo v2_slice_info[] = {v6, v7};
// CHECK-NEXT:  AscendC::SliceInfo v1_slice_info[] = {v6, v7};
// CHECK-NEXT:  AscendC::DataCopy(v2, v1, v2_slice_info, v1_slice_info, v5);
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_data_copy_slice(%arg0 : !ascendc.global_tensor<i32>, %arg1 : !ascendc.local_tensor<i32>,  %c1_i32 : i32, %c2_i32: i32, %arg4: ui32) {
  %14 = ascendc.construct !ascendc.slice_info(%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32) [ui32, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  %15 = ascendc.construct !ascendc.slice_info(%c1_i32, %c1_i32, %c1_i32, %c1_i32, %c2_i32) [ui32, ui32, ui32, ui32, ui32] : i32, i32, i32, i32, i32
  ascendc.data_copy_slice %arg1, %arg0, %14 ,%15, %14 ,%15,  %arg4 {operandSegmentSizes = array<i32: 1, 1, 2, 2, 1>} : !ascendc.local_tensor<i32>, !ascendc.global_tensor<i32>, !ascendc.slice_info, !ascendc.slice_info, !ascendc.slice_info, !ascendc.slice_info, ui32
  return
}