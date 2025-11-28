// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// RUN: ascir-opt -ascendc-declare-py-struct %s | FileCheck %s

// CHECK: emitasc.declare_py_struct !emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>
// CHECK-NEXT: emitasc.declare_py_struct !emitasc.py_struct<"PyStruct2", [i32], ["count"]>
// CHECK-NEXT: emitasc.declare_py_struct !emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>
module {
    func.func @copy_struct(%arg0: memref<?x!emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>, 22>, %arg1: memref<?x!emitasc.py_struct<"PyStruct2", [i32], ["count"]>, 22>) {
        %0 = emitasc.copy_struct %arg0 : memref<?x!emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>, 22>, !emitasc.py_struct<"KernelConfig", [!emitasc.py_struct<"PyStruct1", [i32, i32], ["height", "width"]>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>], ["struct1", "struct2"]>
        %1 = emitasc.copy_struct %arg1 : memref<?x!emitasc.py_struct<"PyStruct2", [i32], ["count"]>, 22>, !emitasc.py_struct<"PyStruct2", [i32], ["count"]>
        return
    }
}
