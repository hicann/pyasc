// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_local_mem_allocator_get_cur_addr() {
// CHECK-NEXT:  constexpr int64_t c0_i64 = 0;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::GM> v1;
// CHECK-NEXT:  uint32_t v2 = v1.GetCurAddr();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_local_mem_allocator_get_cur_addr() {
  %hardware = arith.constant 0 : i64
  %allocator = ascendc.local_mem_allocator %hardware : i64, !ascendc.local_mem_allocator<0>
  %addr = ascendc.local_mem_allocator.get_cur_addr %allocator : !ascendc.local_mem_allocator<0>, ui32
  return
}

// CHECK-LABEL:void emit_local_mem_allocator_alloc_const() {
// CHECK-NEXT:  constexpr int64_t c0_i64 = 0;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::GM> v1;
// CHECK-NEXT:  constexpr int32_t c32_i32 = 32;
// CHECK-NEXT:  AscendC::LocalTensor<float> v2 = v1.Alloc<AscendC::TPosition::VECIN, float, c32_i32>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_local_mem_allocator_alloc_const() {
  %hardware = arith.constant 0 : i64
  %allocator = ascendc.local_mem_allocator %hardware : i64, !ascendc.local_mem_allocator<0>
  %tileSize = arith.constant 32 : i32
  %tensor = ascendc.local_mem_allocator.alloc %allocator, vecin, %tileSize, f32 : !ascendc.local_mem_allocator<0>, i32, !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL:void emit_local_mem_allocator_alloc_dynamic(int32_t v1) {
// CHECK-NEXT:  constexpr int64_t c0_i64 = 0;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::GM> v2;
// CHECK-NEXT:  AscendC::LocalTensor<float> v3 = v2.Alloc<AscendC::TPosition::VECIN, float>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_local_mem_allocator_alloc_dynamic(%tileSize: i32) {
  %hardware = arith.constant 0 : i64
  %allocator = ascendc.local_mem_allocator %hardware : i64, !ascendc.local_mem_allocator<0>
  %tensor = ascendc.local_mem_allocator.alloc_dynamic %allocator, vecin, %tileSize, f32 : !ascendc.local_mem_allocator<0>, i32, !ascendc.local_tensor<*xf32>
  return
}

// CHECK-LABEL:void emit_local_mem_allocator_hardware_types() {
// CHECK-NEXT:  constexpr int64_t c0_i64 = 0;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::GM> v1;
// CHECK-NEXT:  constexpr int64_t c1_i64 = 1;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::UB> v2;
// CHECK-NEXT:  constexpr int64_t c2_i64 = 2;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::L1> v3;
// CHECK-NEXT:  constexpr int64_t c3_i64 = 3;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::L0A> v4;
// CHECK-NEXT:  constexpr int64_t c4_i64 = 4;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::L0B> v5;
// CHECK-NEXT:  constexpr int64_t c5_i64 = 5;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::L0C> v6;
// CHECK-NEXT:  constexpr int64_t c6_i64 = 6;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::BIAS> v7;
// CHECK-NEXT:  constexpr int64_t c7_i64 = 7;
// CHECK-NEXT:  AscendC::LocalMemAllocator<AscendC::Hardware::FIXBUF> v8;
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_local_mem_allocator_hardware_types() {
  %hw0 = arith.constant 0 : i64
  %v1 = ascendc.local_mem_allocator %hw0 : i64, !ascendc.local_mem_allocator<0>

  %hw1 = arith.constant 1 : i64
  %v2 = ascendc.local_mem_allocator %hw1 : i64, !ascendc.local_mem_allocator<1>

  %hw2 = arith.constant 2 : i64
  %v3 = ascendc.local_mem_allocator %hw2 : i64, !ascendc.local_mem_allocator<2>

  %hw3 = arith.constant 3 : i64
  %v4 = ascendc.local_mem_allocator %hw3 : i64, !ascendc.local_mem_allocator<3>

  %hw4 = arith.constant 4 : i64
  %v5 = ascendc.local_mem_allocator %hw4 : i64, !ascendc.local_mem_allocator<4>

  %hw5 = arith.constant 5 : i64
  %v6 = ascendc.local_mem_allocator %hw5 : i64, !ascendc.local_mem_allocator<5>

  %hw6 = arith.constant 6 : i64
  %v7 = ascendc.local_mem_allocator %hw6 : i64, !ascendc.local_mem_allocator<6>

  %hw7 = arith.constant 7 : i64
  %v8 = ascendc.local_mem_allocator %hw7 : i64, !ascendc.local_mem_allocator<7>

  return
}