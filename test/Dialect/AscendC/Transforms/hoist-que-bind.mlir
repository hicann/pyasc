// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-hoist-que-bind %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_que_bind_for
// CHECK:       %0 = ascendc.pipe
// CHECK-NEXT:  %1 = ascendc.queue : <vec_out, 1>
// CHECK-NEXT:  ascendc.pipe.init_queue %0, %1, %arg0, %arg2 : !ascendc.queue<vec_out, 1>, i32, i64
// CHECK-NEXT:  %2 = ascendc.queue : <vec_in, 1>
// CHECK-NEXT:  ascendc.pipe.init_queue %0, %2, %arg0, %arg2 : !ascendc.queue<vec_in, 1>, i32, i64
// CHECK-NEXT:  scf.for %arg3 = %arg0 to %arg1 step %arg0 : i32
func.func @hoist_que_bind_for(%arg0 : i32, %arg1: i32, %arg2: i64) {
    %0 = ascendc.pipe
    scf.for %arg3 = %arg0 to %arg1 step %arg0 : i32 {
        %1 = ascendc.queue : <vec_in, 1>
        ascendc.pipe.init_queue %0, %1, %arg0, %arg2 : !ascendc.queue<vec_in, 1>, i32, i64
        %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vec_in, 1>, !ascendc.local_tensor<64xf32>
        %3 = ascendc.queue : <vec_out, 1>
        ascendc.pipe.init_queue %0, %3, %arg0, %arg2 : !ascendc.queue<vec_out, 1>, i32, i64
        %4 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vec_out, 1>, !ascendc.local_tensor<64xf32>
        ascendc.que_bind.free_tensor %3, %4 : !ascendc.queue<vec_out, 1>, !ascendc.local_tensor<64xf32>
        ascendc.que_bind.free_tensor %1, %2 : !ascendc.queue<vec_in, 1>, !ascendc.local_tensor<64xf32>
    }
    return
}

// CHECK-LABEL: func.func @hoist_que_bind_nested_for
// CHECK:       %0 = ascendc.pipe
// CHECK-NEXT:  %1 = ascendc.queue : <vec_out, 1>
// CHECK-NEXT:  ascendc.pipe.init_queue %0, %1, %arg0, %arg3 : !ascendc.queue<vec_out, 1>, i32, i64
// CHECK-NEXT:  %2 = ascendc.queue : <vec_in, 1>
// CHECK-NEXT:  ascendc.pipe.init_queue %0, %2, %arg0, %arg3 : !ascendc.queue<vec_in, 1>, i32, i64
// CHECK-NEXT:  scf.for %arg4 = %arg0 to %arg1 step %arg0 : i32
// CHECK-NEXT:    scf.for %arg5 = %arg0 to %arg2 step %arg0 : i32
func.func @hoist_que_bind_nested_for(%arg0 : i32, %arg1: i32, %arg2: i32, %arg3: i64) {
    %0 = ascendc.pipe
    scf.for %arg4 = %arg0 to %arg1 step %arg0 : i32 {
        scf.for %arg5 = %arg0 to %arg2 step %arg0 : i32 {
            %1 = ascendc.queue : <vec_in, 1>
            ascendc.pipe.init_queue %0, %1, %arg0, %arg3 : !ascendc.queue<vec_in, 1>, i32, i64
            %2 = ascendc.que_bind.alloc_tensor %1 : !ascendc.queue<vec_in, 1>, !ascendc.local_tensor<64xf32>
            %3 = ascendc.queue : <vec_out, 1>
            ascendc.pipe.init_queue %0, %3, %arg0, %arg3 : !ascendc.queue<vec_out, 1>, i32, i64
            %4 = ascendc.que_bind.alloc_tensor %3 : !ascendc.queue<vec_out, 1>, !ascendc.local_tensor<64xf32>
            ascendc.que_bind.free_tensor %3, %4 : !ascendc.queue<vec_out, 1>, !ascendc.local_tensor<64xf32>
            ascendc.que_bind.free_tensor %1, %2 : !ascendc.queue<vec_in, 1>, !ascendc.local_tensor<64xf32>
        }
    }
    return
}
