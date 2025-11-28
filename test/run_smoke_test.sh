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
    echo "[INFO] Running test: ${test_name}"
    if pytest "$test_py_file"; then
        echo "[INFO] ${test_name} example passed!"
        return 0
    else
        echo "[ERROR] ${test_name} example failed!"
        return 1
    fi
}

test_npu_examples=(
    "./python/test/generalization/adv/test_matmul_iterate_all.py"
    "./python/test/generalization/adv/test_matmul_iterate_batch.py"
    "./python/test/generalization/adv/test_matmul.py"
    "./python/test/generalization/adv/test_matmul_quant.py"
    "./python/test/generalization/adv/test_quant.py"
    "./python/test/generalization/basic/test_aipp.py"
    "./python/test/generalization/basic/test_axpy.py"
    "./python/test/generalization/basic/test_copy.py"
    "./python/test/generalization/basic/test_duplicate.py"
    "./python/test/generalization/basic/test_trans_5hd_addr_tensor.py"
    "./python/test/generalization/basic/test_trans_5hd_tensor_list.py"
    "./python/test/generalization/basic/test_transpose_ext.py"
    "./python/test/generalization/basic/test_transpose.py"
    "./python/test/generalization/basic/test_vadd_bits.py"
    "./python/test/generalization/basic/test_vadd_l0.py"
    "./python/test/generalization/basic/test_vadd_sw.py"
    "./python/test/generalization/basic/test_vadd_tiling.py"
    "./python/test/generalization/basic/test_vadd.py"
    "./python/test/generalization/basic/test_vadds.py"
    "./python/test/generalization/basic/test_vsqrt.py"
)

passed_examples=()
failed_examples=()

for example_npu in "${test_npu_examples[@]}"; do
    if run_test "$example_npu"; then
        passed_examples+=("$example_npu")
    else
        failed_examples+=("$example_npu")
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
    echo "[INFO] All ${#test_npu_examples[@]} tests passed!"
    exit 0
else
    echo "[ERROR] ${#failed_examples[@]} / ${#test_npu_examples[@]} tests failed!"
    exit 1
fi
