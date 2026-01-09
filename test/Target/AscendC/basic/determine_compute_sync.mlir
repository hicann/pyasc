// Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL: void emit_notify_next_block(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2) {
// CHECK-NEXT:   AscendC::NotifyNextBlock(v1, v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
module {
  func.func @emit_notify_next_block(
    %gm_workspace: !ascendc.global_tensor<*xi32>, 
    %ub_workspace: !ascendc.local_tensor<*xi32>
  ) attributes {ascendc.aicore, ascendc.global} {
    ascendc.notify_next_block %gm_workspace, %ub_workspace : !ascendc.global_tensor<*xi32>, !ascendc.local_tensor<*xi32>
    return
  }
}

// CHECK-LABEL: void emit_wait_pre_block(AscendC::GlobalTensor<int32_t> v1, AscendC::LocalTensor<int32_t> v2) {
// CHECK-NEXT:   AscendC::WaitPreBlock(v1, v2);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
module {
  func.func @emit_wait_pre_block(
    %gm_workspace: !ascendc.global_tensor<*xi32>, 
    %ub_workspace: !ascendc.local_tensor<*xi32>
  ) attributes {ascendc.aicore, ascendc.global} {
    ascendc.wait_pre_block %gm_workspace, %ub_workspace : !ascendc.global_tensor<*xi32>, !ascendc.local_tensor<*xi32>
    return
  }
}
