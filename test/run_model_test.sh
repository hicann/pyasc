#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

run_test() {
    local test_py_file=$1
    local test_name=${test_py_file##*/}
    echo "[INFO] Running model test: ${test_name}"
    if python3 "$test_py_file"; then
        echo "[INFO] ${test_name} example passed!"
        return 0
    else
        echo "[ERROR] ${test_name} example failed!"
        return 1
    fi
}

test_examples=(
    "./python/test/kernels/test_vadd.py"
    "./python/test/kernels/test_matmul.py"
)

passed_examples=()
failed_examples=()

for example in "${test_examples[@]}"; do
    if run_test "$example"; then
        passed_examples+=("$example")
    else
        failed_examples+=("$example")
    fi
done

echo "[INFO] Passed tests list:"
for test in "${passed_examples[@]}"; do
    echo " ${test##*/}"
done

echo "[INFO] Failed tests list:"
for test in "${failed_examples[@]}"; do
    echo " ${test##*/}"
done

if [ ${#failed_examples[@]} -eq 0 ]; then
    echo "[INFO] All ${#test_examples[@]} tests passed!"
    exit 0
else
    echo "[ERROR] ${#failed_examples[@]} / ${#test_examples[@]} tests failed!"
    exit 1
fi
