// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-privatize-func %s | FileCheck %s -match-full-lines

// CHECK-LABEL: func.func @public_global_function() attributes {ascendc.global} {
func.func public @public_global_function() attributes {ascendc.global} {
    return
}

// CHECK-LABEL: func.func private @public_function() {
func.func public @public_function() {
    return
}

// CHECK-LABEL: func.func @private_global_function() attributes {ascendc.global} {
func.func private @private_global_function() attributes {ascendc.global} {
    return
}

// CHECK-LABEL: func.func private @private_function() {
func.func private @private_function() {
    return
}

// CHECK-LABEL: func.func private @private_global_pure_function() attributes {ascendc.global}
func.func private @private_global_pure_function() attributes {ascendc.global}

// CHECK-LABEL: func.func private @private_pure_function()
func.func private @private_pure_function()
