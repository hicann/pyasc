// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_scalar_cast_kernel(float v1) {
// CHECK-NEXT:  int32_t v2 = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_scalar_cast_kernel(%v1: f32){
  %v2 = ascendc.scalar_cast %v1 {dtype = i32, roundMode = 4 : i32} : f32 -> i32
  return
}

// CHECK-LABEL:void emit_scalar_get_sff_value_kernel(uint64_t v1) {
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  int64_t v2 = AscendC::ScalarGetSFFValue<c0_i32>(v1);
// CHECK-NEXT:  constexpr int32_t c1_i32 = 1;
// CHECK-NEXT:  int64_t v3 = AscendC::ScalarGetSFFValue<c1_i32>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_scalar_get_sff_value_kernel(%v1: ui64) {
  %c0_i32 = arith.constant 0 : i32 
  %v2 = ascendc.scalar_get_sff_value %v1, %c0_i32 : ui64, i32 -> i64 
  %c1_i32 = arith.constant 1 : i32 
  %v3 = ascendc.scalar_get_sff_value %v1, %c1_i32 : ui64, i32 -> i64 
  return
}

// CHECK-LABEL:void emit_scalar_get_count_of_value_kernel(uint64_t v1) {
// CHECK-NEXT:  constexpr int32_t c0_i32 = 0;
// CHECK-NEXT:  int64_t v2 = AscendC::ScalarGetCountOfValue<c0_i32>(v1);
// CHECK-NEXT:  constexpr int32_t c1_i32 = 1;
// CHECK-NEXT:  int64_t v3 = AscendC::ScalarGetCountOfValue<c1_i32>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_scalar_get_count_of_value_kernel(%v1: ui64) {
  %c0_i32 = arith.constant 0 : i32 
  %v2 = ascendc.scalar_get_count_of_value %v1, %c0_i32 : ui64, i32 -> i64 
  %c1_i32 = arith.constant 1 : i32 
  %v3 = ascendc.scalar_get_count_of_value %v1, %c1_i32 : ui64, i32 -> i64 
  return
}

// CHECK-LABEL:void emit_scalar_count_leading_zero_kernel(uint64_t v1) {
// CHECK-NEXT:  int64_t v2 = AscendC::ScalarCountLeadingZero(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_scalar_count_leading_zero_kernel(%v1: ui64) {
  %v2 = ascendc.scalar_count_leading_zero %v1 : ui64 -> i64 
  return
}

// CHECK-LABEL:void emit_count_bits_cnt_same_as_sign_bit_kernel(int64_t v1) {
// CHECK-NEXT:  int64_t v2 = AscendC::CountBitsCntSameAsSignBit(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_count_bits_cnt_same_as_sign_bit_kernel(%v1: i64) {
  %v2 = ascendc.count_bits_cnt_same_as_sign_bit %v1 : i64 -> i64 
  return
}