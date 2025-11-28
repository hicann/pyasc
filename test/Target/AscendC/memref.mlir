// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK:void emit_alloca() {
// CHECK-NEXT:  int32_t v1[777];
// CHECK-NEXT:  uint16_t v2[2];
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_alloca() {
  %0 = memref.alloca() : memref<777xi32>
  %1 = memref.alloca() {ascendc.emit_as_unsigned} : memref<2xi16>
  return
}

// CHECK:void emit_load(int32_t* v1, int64_t* v2, uint32_t v3, uint32_t v4) {
// CHECK-NEXT:  int32_t v5 = v1[v3];
// CHECK-NEXT:  int64_t v6 = v2[v3][v4];
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_load(%arg1: memref<?xi32>, %arg2: memref<?x?xi64>, %arg3: index, %arg4: index) {
  %0 = memref.load %arg1[%arg3] : memref<?xi32>
  %1 = memref.load %arg2[%arg3, %arg4] : memref<?x?xi64>
  return
}

// CHECK:void emit_store(uint32_t v1, int64_t v2, int64_t* v3) {
// CHECK-NEXT:  v3[v1] = v2;
// CHECK-NEXT:  return;
// CHECK-NEXT:}
func.func @emit_store(%c0: index, %c0_i64: i64, %alloca: memref<?xi64>) {
  memref.store %c0_i64, %alloca[%c0] : memref<?xi64>
  return
}

// CHECK-LABEL: void emit_address_spaces(
// CHECK-SAME: int32_t* v1
// CHECK-SAME: __gm__ int32_t* v2
// CHECK-SAME: __ca__ int32_t* v3
// CHECK-SAME: __cb__ int32_t* v4
// CHECK-SAME: __cc__ int32_t* v5
// CHECK-SAME: __ubuf__ int32_t* v6
// CHECK-SAME: __cbuf__ int32_t* v7
// CHECK-SAME: __fbuf__ int32_t* v8
// CHECK-SAME: ) {
func.func @emit_address_spaces(
  %arg0: memref<?xi32>,
  %arg1: memref<?xi32, 22>,
  %arg2: memref<?xi32, 23>,
  %arg3: memref<?xi32, 24>,
  %arg4: memref<?xi32, 25>,
  %arg5: memref<?xi32, 26>,
  %arg6: memref<?xi32, 27>,
  %arg7: memref<?xi32, 28>
) {
  return
}

// CHECK-LABEL:void emit_memref_cast(int32_t* v1) {
// CHECK-NEXT:  int32_t* v2 = reinterpret_cast<int32_t*>(v1);
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_memref_cast(%arg0: memref<2xi32>) {
  %0 = memref.cast %arg0 : memref<2xi32> to memref<?xi32>
  return
}
