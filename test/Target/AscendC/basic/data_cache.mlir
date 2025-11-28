// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_data_cache_clean_and_invalid(__gm__ float* v1, __gm__ uint64_t* v2) {
// CHECK-NEXT:   set_ffts_base_addr(*v2);
// CHECK-NEXT:   AscendC::GlobalTensor<float> v3;
// CHECK-NEXT:   v3.SetGlobalBuffer(v1);
// CHECK-NEXT:   AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(v3);
// CHECK-NEXT:   AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE>(v3);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_data_cache_clean_and_invalid(%arg0: memref<?xf32, 22>, %arg1: memref<?xui64, 22>){
  ascendc.set_ffts_base_addr %arg1 : memref<?xui64, 22>
  %0 = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  ascendc.global_tensor.set_global_buffer %0, %arg0 : !ascendc.global_tensor<*xf32>, memref<?xf32, 22>
  ascendc.data_cache_clean_invalid_global %0 {dcciDst = 2 : i32, entireType = 0 : i32} : !ascendc.global_tensor<*xf32>
  ascendc.data_cache_clean_invalid_global_nodccidst %0 {entireType = 0 : i32} : !ascendc.global_tensor<*xf32>
  return
}

// CHECK-LABEL: void emit_get_icache_preload_status() {
// CHECK-NEXT:   int64_t v1 = AscendC::GetICachePreloadStatus();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_get_icache_preload_status() {
  %0 = ascendc.get_icache_preload_status : i64
  return
}

// CHECK-LABEL: void emit_icache_preload(int64_t v1) {
// CHECK-NEXT:   AscendC::ICachePreLoad(v1);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_icache_preload(%v1: i64) {
  ascendc.icache_preload %v1 : i64
  return
}
