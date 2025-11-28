// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_brcb(AscendC::LocalTensor<float> v1, AscendC::LocalTensor<float> v2, int32_t v3, AscendC::BrcbRepeatParams v4) {
// CHECK-NEXT:   AscendC::Brcb(v1, v2, v3, v4);
// CHECK-NEXT:   return;
// CHECK-NEXT: }
func.func @emit_brcb(%dst: !ascendc.local_tensor<1024xf32>, %src0: !ascendc.local_tensor<1024xf32>, %repeatTimes : i32, %params: !ascendc.brcb_repeat_params) {
  ascendc.brcb_l0 %dst, %src0, %repeatTimes, %params : !ascendc.local_tensor<1024xf32>, !ascendc.local_tensor<1024xf32>, i32, !ascendc.brcb_repeat_params
  return
}