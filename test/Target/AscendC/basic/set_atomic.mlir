// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-translate -mlir-to-ascendc %s | FileCheck %s

// CHECK-LABEL:void emit_set_atomic_add() {
// CHECK-NEXT:  AscendC::SetAtomicAdd<half>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_atomic_add() {
  ascendc.set_atomic_add {dtype = f16} : () -> ()
  return
}

// CHECK-LABEL:void emit_set_atomic_none() {
// CHECK-NEXT:  AscendC::SetAtomicNone();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_atomic_none() {
  ascendc.set_atomic_none : () -> ()
  return
}

// CHECK-LABEL:void emit_set_atomic_max() {
// CHECK-NEXT:  AscendC::SetAtomicMax<half>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_atomic_max() {
  ascendc.set_atomic_max {dtype = f16} : () -> ()
  return
}

// CHECK-LABEL:void emit_set_atomic_min() {
// CHECK-NEXT:  AscendC::SetAtomicMin<half>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_atomic_min() {
  ascendc.set_atomic_min {dtype = f16} : () -> ()
  return
}

// CHECK-LABEL:void emit_set_atomic_type() {
// CHECK-NEXT:  AscendC::SetAtomicType<half>();
// CHECK-NEXT:  return;
// CHECK-NEXT: }
func.func @emit_set_atomic_type() {
  ascendc.set_atomic_type {dtype = f16} : () -> ()
  return
}