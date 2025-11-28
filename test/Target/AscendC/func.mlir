// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s -match-full-lines

// CHECK: extern "C" __global__ __aicore__ void global_function(int32_t v1) {
func.func @global_function(%arg0: i32) attributes {ascendc.aicore, ascendc.global} {
  return
}

// CHECK: __inline__ __attribute__((always_inline)) __aicore__ int32_t inline_function(float v1) {
func.func @inline_function(%arg0: f32) -> i32 attributes {ascendc.aicore} {
  %0 = arith.fptosi %arg0 : f32 to i32
  return %0 : i32
}
