# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_softmax(mock_launcher_run):

    @asc.jit
    def kernel_softmax() -> None:
        dst = asc.LocalTensor(dtype=asc.float16)
        sum_local = asc.LocalTensor(dtype=asc.float16)
        max_local = asc.LocalTensor(dtype=asc.float16)
        src = asc.LocalTensor(dtype=asc.float16)
        tiling = asc.adv.SoftmaxTiling(src_m=1, src_k=1, src_size=64)
        asc.adv.softmax(dst, sum_local, max_local, src, tiling)

    kernel_softmax[1]()
    assert mock_launcher_run.call_count == 1
