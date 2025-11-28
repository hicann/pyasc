// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_proposal_concat(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 256;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr int32_t c2_i32 = 2;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   AscendC::LocalTensor<half> v4 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v2);
// CHECK-NEXT:   AscendC::ProposalConcat(v4, v5, c2_i32, c4_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }

func.func @emit_proposal_concat(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %c256 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c0 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %dst = ascendc.local_tensor_v2 vec_out, %c0, %c256 : !ascendc.local_tensor<*xf16>
  %src = ascendc.local_tensor_v2 vec_in, %c0, %c256 : !ascendc.local_tensor<*xf16>
  ascendc.proposal_concat %dst, %src, %c2_i32, %c4_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32
  return
}

// CHECK-LABEL: void emit_proposal_extract(__gm__ uint64_t* v1) {
// CHECK-NEXT:   set_ffts_base_addr(*v1);
// CHECK-NEXT:   constexpr uint32_t v2 = 256;
// CHECK-NEXT:   constexpr uint32_t v3 = 0;
// CHECK-NEXT:   constexpr int32_t c2_i32 = 2;
// CHECK-NEXT:   constexpr int32_t c4_i32 = 4;
// CHECK-NEXT:   AscendC::LocalTensor<half> v4 = AscendC::LocalTensor<half>(AscendC::TPosition::VECOUT, v3, v2);
// CHECK-NEXT:   AscendC::LocalTensor<half> v5 = AscendC::LocalTensor<half>(AscendC::TPosition::VECIN, v3, v2);
// CHECK-NEXT:   AscendC::ProposalExtract(v4, v5, c2_i32, c4_i32);
// CHECK-NEXT:   return;
// CHECK-NEXT: }

func.func @emit_proposal_extract(%arg0: memref<?xui64, 22>) {
  ascendc.set_ffts_base_addr %arg0 : memref<?xui64, 22>
  %c256 = "emitc.constant"() <{value = 256 : ui32}> : () -> ui32
  %c0 = "emitc.constant"() <{value = 0 : ui32}> : () -> ui32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %dst = ascendc.local_tensor_v2 vec_out, %c0, %c256 : !ascendc.local_tensor<*xf16>
  %src = ascendc.local_tensor_v2 vec_in, %c0, %c256 : !ascendc.local_tensor<*xf16>
  ascendc.proposal_extract %dst, %src, %c2_i32, %c4_i32 : !ascendc.local_tensor<*xf16>, !ascendc.local_tensor<*xf16>, i32, i32
  return
}