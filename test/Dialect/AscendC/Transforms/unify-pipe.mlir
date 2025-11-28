// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-unify-pipe %s | FileCheck %s

// CHECK: %0 = ascendc.pipe
// CHECK-NOT: ascendc.pipe
// CHECK: ascendc.pipe.init_queue %0, %1, %c1_i32, %c256_i64 : !ascendc.queue<vec_in, 1>, i32, i64
// CHECK: ascendc.pipe.init_queue %0, %2, %c1_i32, %c256_i64 : !ascendc.queue<vec_in, 1>, i32, i64
func.func @unify_pipe(%arg0 : !ascendc.global_tensor<*xf32>, %arg1 : !ascendc.global_tensor<*xf32>, %arg2 : !ascendc.global_tensor<*xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %0 = ascendc.pipe
    %1 = ascendc.queue : <vec_in, 1>
    ascendc.pipe.init_queue %0, %1, %c1_i32, %c256_i64 : !ascendc.queue<vec_in, 1>, i32, i64
    %2 = ascendc.pipe
    %3 = ascendc.queue : <vec_in, 1>
    ascendc.pipe.init_queue %2, %3, %c1_i32, %c256_i64 : !ascendc.queue<vec_in, 1>, i32, i64
    return
}
