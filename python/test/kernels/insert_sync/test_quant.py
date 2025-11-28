# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Tuple

import asc
from asc.runtime import config
import asc.lib.runtime as rt
import numpy as np


@asc.jit(kernel_type=config.KernelType.AIV_ONLY, insert_sync=True)
def quant_kernel(src_gm: asc.GlobalAddress, dst_gm: asc.GlobalAddress, size: asc.ConstExpr[int],
                 tmp_min_bytes: asc.ConstExpr[int]):
    src_global = asc.GlobalTensor()
    dst_global = asc.GlobalTensor()
    src_global.set_global_buffer(src_gm)
    dst_global.set_global_buffer(dst_gm)
    src_local = asc.LocalTensorAuto(src_global.dtype, size)
    asc.data_copy(src_local, src_global, count=size)
    dst_local = asc.LocalTensorAuto(asc.int8, size)
    tmp_local = asc.LocalTensorAuto(asc.uint8, tmp_min_bytes)
    scale = 2.0
    offset = 0.9
    asc.adv.quant(dst_local, src_local, scale, offset, size, tmp_local)
    asc.data_copy(dst_global, dst_local, count=size)


def get_min_max_tmp_size(input_size: int) -> Tuple[int, int]:
    repeat_times = 2
    one_repeat_bytes = 256
    memory_calc = 2
    blk_size = 32
    tmp1 = input_size * memory_calc
    tmp2 = repeat_times * one_repeat_bytes
    min_value = repeat_times * one_repeat_bytes
    max_value = (max(tmp1, tmp2) + blk_size - 1) // blk_size * blk_size
    return min_value, max_value


def quant_launch(src: np.ndarray) -> np.ndarray:
    dst = np.zeros_like(src, dtype=np.int8)
    tmp_min_bytes, _ = get_min_max_tmp_size(src.size)
    quant_kernel[1, rt.current_stream()](src, dst, src.size, tmp_min_bytes)
    return dst


def quant_numpy(src: np.ndarray) -> np.ndarray:
    return np.round((src * 2.0) + 0.9).astype(np.int8)


def test_quant(backend: config.Backend):
    config.set_platform(backend)
    rng = np.random.default_rng(seed=2025)
    size = 32
    src = (rng.random(size) * 8.0 - 4.0).astype(np.float16)  # [-4, 4]
    dst = quant_launch(src)
    np.testing.assert_allclose(dst, quant_numpy(src))


if __name__ == "__main__":
    test_quant(config.Backend.Model)
