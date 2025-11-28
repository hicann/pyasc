// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_pipe() {
// CHECK-NEXT:   AscendC::TPipe v1;
// CHECK-NEXT:   v2 = v1.AllocEventID<AscendC::HardEvent::MTE2_V>();
// CHECK-NEXT:   AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(v2)
// CHECK-NEXT:   AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(v2)
// CHECK-NEXT:   v1.ReleaseEventID<AscendC::HardEvent::MTE2_V>(v2);
// CHECK-NEXT:   v3 = v1.FetchEventID<AscendC::HardEvent::V_MTE3>();
// CHECK-NEXT:   AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(v3)
// CHECK-NEXT:   AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(v3)
// CHECK-NEXT:   v1.Init();
// CHECK-NEXT:   v1.Reset();
// CHECK-NEXT:   v1.Destroy();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_pipe() {
  %0 = ascendc.pipe
  %id0 = ascendc.pipe.alloc_event_id %0, mte2_v : i32
  ascendc.set_flag mte2_v, %id0 : i32
  ascendc.wait_flag mte2_v, %id0 : i32
  ascendc.pipe.release_event_id %0, %id0 {event = 4 : i32} : !ascendc.pipe, i32
  %id1 = ascendc.pipe.fetch_event_id %0, v_mte3 : i32
  ascendc.set_flag v_mte3, %id1 : i32
  ascendc.wait_flag v_mte3, %id1 : i32
  ascendc.pipe.init %0 : !ascendc.pipe 
  ascendc.pipe.reset %0 : !ascendc.pipe 
  ascendc.pipe.destroy %0 : !ascendc.pipe 
  return
}

// CHECK-LABEL:void emit_tbuf_pool(uint32_t v1) {
// CHECK-NEXT:   AscendC::TPipe v2;
// CHECK-NEXT:   AscendC::TBufPool<AscendC::TPosition::VECIN, 4> v3;
// CHECK-NEXT:   AscendC::TBufPool<AscendC::TPosition::VECIN, 4> v4;
// CHECK-NEXT:   bool v5 = v2.InitBufPool(v4, v1);
// CHECK-NEXT:   bool v6 = v2.InitBufPool(v4, v1, v3);
// CHECK-NEXT:   AscendC::TBufPool<AscendC::TPosition::VECIN, 4> v7;
// CHECK-NEXT:   bool v8 = v3.InitBufPool(v4, v1);
// CHECK-NEXT:   bool v9 = v3.InitBufPool(v4, v1, v7);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_tbuf_pool(%arg0 : ui32) {
  %0 = ascendc.pipe
  %1 = ascendc.tBufPool : <vec_in, 4> 
  %2 = ascendc.tBufPool : <vec_in, 4> 
  %4 = ascendc.pipe.init_buf_pool(%0, %2, %arg0) : !ascendc.pipe, !ascendc.tbuf_pool<vec_in, 4>, ui32 -> i1
  %5 = ascendc.pipe.init_buf_pool(%0, %2, %arg0), %1 : !ascendc.pipe, !ascendc.tbuf_pool<vec_in, 4>, ui32 , !ascendc.tbuf_pool<vec_in, 4> -> i1
  %6 = ascendc.tBufPool : <vec_in, 4> 
  %7 = ascendc.tbuf_pool.init_buf_pool(%1, %2, %arg0) : !ascendc.tbuf_pool<vec_in, 4>, !ascendc.tbuf_pool<vec_in, 4>, ui32 -> i1
  %8 = ascendc.tbuf_pool.init_buf_pool(%1, %2, %arg0), %6 : !ascendc.tbuf_pool<vec_in, 4>, !ascendc.tbuf_pool<vec_in, 4>, ui32 , !ascendc.tbuf_pool<vec_in, 4> -> i1
  return
}

// CHECK-LABEL:void emit_tbuf_pool_init_buffer(int32_t v1) {
// CHECK-NEXT:   AscendC::TBufPool<AscendC::TPosition::VECIN, 4> v2;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::GM> v3;
// CHECK-NEXT:   v2.InitBuffer(v3, v1);
// CHECK-NEXT:   AscendC::TQue<AscendC::TPosition::GM, 1> v4;
// CHECK-NEXT:   v2.InitBuffer(v4, v1, v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_tbuf_pool_init_buffer(%arg0 : i32) {
  %0 = ascendc.tBufPool : <vec_in, 4> 
  %buffer = ascendc.tbuf : !ascendc.tbuf<gm>
  ascendc.tbuf_pool.init_buffer %0, %buffer, %arg0 : !ascendc.tbuf_pool<vec_in, 4>, !ascendc.tbuf<gm> , i32
  %queue = ascendc.queue : !ascendc.queue<gm, 1>
  ascendc.tbuf_pool.init_queue %0, %queue, %arg0, %arg0 : !ascendc.tbuf_pool<vec_in, 4>, !ascendc.queue<gm, 1>, i32, i32
  return
}

// CHECK-LABEL:void emit_buffer() {
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::GM> v1;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::A1> v2;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::A2> v3;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::B1> v4;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::B2> v5;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::C1> v6;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::C2> v7;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::CO1> v8;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::CO2> v9;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::VECIN> v10;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::VECOUT> v11;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::VECCALC> v12;
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_buffer() {
  %0 = ascendc.tbuf : !ascendc.tbuf<gm>
  %1 = ascendc.tbuf : !ascendc.tbuf<a1>
  %2 = ascendc.tbuf : !ascendc.tbuf<a2>
  %3 = ascendc.tbuf : !ascendc.tbuf<b1>
  %4 = ascendc.tbuf : !ascendc.tbuf<b2>
  %5 = ascendc.tbuf : !ascendc.tbuf<c1>
  %6 = ascendc.tbuf : !ascendc.tbuf<c2>
  %7 = ascendc.tbuf : !ascendc.tbuf<co1>
  %8 = ascendc.tbuf : !ascendc.tbuf<co2>
  %9 = ascendc.tbuf : !ascendc.tbuf<vec_in>
  %10 = ascendc.tbuf : !ascendc.tbuf<vec_out>
  %11 = ascendc.tbuf : !ascendc.tbuf<vec_calc>
  return
}

// CHECK-LABEL:void emit_queue() {
// CHECK-NEXT:   AscendC::TQue<AscendC::TPosition::GM, 1> v1;
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_queue() {
  %queue = ascendc.queue : !ascendc.queue<gm, 1>
  return
}

// CHECK-LABEL:void emit_local_tensor_slice(AscendC::LocalTensor<float> v1, uint32_t v2) {
// CHECK-NEXT:   AscendC::LocalTensor<float> v3 = v1[v2];
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_local_tensor_slice(%arg0: !ascendc.local_tensor<1024xf32>, %arg1: index) {
  %subindex = ascendc.local_tensor.subindex %arg0[%arg1] : !ascendc.local_tensor<1024xf32>, index, !ascendc.local_tensor<1024xf32>
  return
}


// CHECK-LABEL:void emit_init_buffer(int32_t v1) {
// CHECK-NEXT:   AscendC::TPipe v2;
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::GM> v3;
// CHECK-NEXT:   v2.InitBuffer(v3, v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_init_buffer(%length : i32) {
  %pipe = ascendc.pipe
  %buffer = ascendc.tbuf : !ascendc.tbuf<gm>
  ascendc.pipe.init_buffer %pipe, %buffer, %length : !ascendc.tbuf<gm> , i32
  return
}

// CHECK-LABEL:void emit_get_tensor() {
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::GM> v1;
// CHECK-NEXT:   AscendC::LocalTensor<int32_t> v2 = v1.Get<int32_t>();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_get_tensor() {
  %buffer = ascendc.tbuf : !ascendc.tbuf<gm>
  %local_tensor = ascendc.tbuf.get_tensor %buffer : !ascendc.tbuf<gm>, !ascendc.local_tensor<i32>
  return
}

// CHECK-LABEL:void emit_get_with_offset(uint32_t v1) {
// CHECK-NEXT:   AscendC::TBuf<AscendC::TPosition::GM> v2;
// CHECK-NEXT:   AscendC::LocalTensor<int32_t> v3 = v2.GetWithOffset<int32_t>(v1, v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_get_with_offset(%arg0 : ui32) {
  %buffer = ascendc.tbuf : !ascendc.tbuf<gm>
  %local_tensor = ascendc.tbuf.get_with_offset %buffer, %arg0, %arg0 : !ascendc.tbuf<gm>, ui32, ui32, !ascendc.local_tensor<*xi32>
  return
}

// CHECK-LABEL:void emit_init_queue(int32_t v1, int32_t v2) {
// CHECK-NEXT:   AscendC::TPipe v3;
// CHECK-NEXT:   AscendC::TQue<AscendC::TPosition::GM, 1> v4;
// CHECK-NEXT:   v3.InitBuffer(v4, v1, v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_init_queue(%num : i32, %length : i32) {
  %pipe = ascendc.pipe
  %queue = ascendc.queue : !ascendc.queue<gm, 1>
  ascendc.pipe.init_queue %pipe, %queue, %num, %length : !ascendc.queue<gm, 1>, i32, i32
  return
}

// CHECK-LABEL:void emit_base_queue_operations(AscendC::TQue<AscendC::TPosition::GM, 1> v1) {
// CHECK-NEXT:   AscendC::LocalTensor<int32_t> v2 = v1.AllocTensor<int32_t>();
// CHECK-NEXT:   v1.EnQue(v2);
// CHECK-NEXT:   v1.EnQue<AscendC::TPosition::GM, AscendC::TPosition::VECIN>(v2);
// CHECK-NEXT:   AscendC::LocalTensor<int32_t> v3 = v1.DeQue<int32_t>();
// CHECK-NEXT:   v1.FreeTensor(v2);
// CHECK-NEXT:   AscendC::TQueBind<AscendC::TPosition::GM, AscendC::TPosition::VECCALC, 1>& v4 = v1;
// CHECK-NEXT:   int32_t v5 = v1.GetTensorCountInQue();
// CHECK-NEXT:   bool v6 = v1.HasIdleBuffer();
// CHECK-NEXT:   bool v7 = v1.HasTensorInQue();
// CHECK-NEXT:   bool v8 = v1.VacantInQue();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_base_queue_operations(%queue : !ascendc.queue<gm, 1>) {
  %local_tensor = ascendc.que_bind.alloc_tensor %queue : !ascendc.queue<gm, 1>, !ascendc.local_tensor<i32>
  ascendc.que_bind.enque_tensor %queue, %local_tensor : !ascendc.queue<gm, 1> , !ascendc.local_tensor<i32>
  ascendc.que_bind.enque_tensor_pos %queue, %local_tensor, gm, vec_in : !ascendc.queue<gm, 1> , !ascendc.local_tensor<i32>
  %deque_tensor = ascendc.que_bind.deque_tensor %queue : !ascendc.queue<gm, 1> , !ascendc.local_tensor<i32>
  ascendc.que_bind.free_tensor %queue, %local_tensor : !ascendc.queue<gm, 1>, !ascendc.local_tensor<i32>
  %que_bind = ascendc.to_que_bind %queue : !ascendc.queue<gm, 1>, !ascendc.que_bind<gm, vec_calc, 1>
  %0 = ascendc.que_bind.get_tensor_count_in_que %queue :  !ascendc.queue<gm, 1>, i32
  %1 = ascendc.que_bind.has_idle_buffer %queue :  !ascendc.queue<gm, 1>, i1
  %2 = ascendc.que_bind.has_tensor_in_que %queue :  !ascendc.queue<gm, 1>, i1
  %3 = ascendc.que_bind.vacant_in_que %queue :  !ascendc.queue<gm, 1>, i1
  return
}

// CHECK-LABEL:void emit_que_bind_in_place(uint32_t v1, uint32_t v2) {
// CHECK-NEXT:   AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECIN, 0> v3;
// CHECK-NEXT:   AscendC::LocalTensor<float> v4 = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, v1, v2);
// CHECK-NEXT:   v3.AllocTensor<float>(v4);
// CHECK-NEXT:   v3.DeQue<float>(v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_que_bind_in_place(%arg0 : ui32, %arg1 : ui32) {
  %que_bind = ascendc.que_bind : <vec_in, vec_in, 0> 
  %tensor = ascendc.local_tensor_v2 vec_in, %arg0, %arg1 : !ascendc.local_tensor<*xf32> 
  ascendc.que_bind.alloc_tensor_in_place %que_bind, %tensor : !ascendc.que_bind<vec_in, vec_in, 0>, !ascendc.local_tensor<*xf32> 
  ascendc.que_bind.deque_tensor_in_place %que_bind, %tensor : !ascendc.que_bind<vec_in, vec_in, 0>, !ascendc.local_tensor<*xf32> 
  return
}
