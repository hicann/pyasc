# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt


@asc.jit  # insert_sync=True will be added implicitly because asc.LocalTensorAuto is used in the kernel
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, block_length: asc.ConstExpr[int],
                tile_length: asc.ConstExpr[int], tile_num: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)
    for i in range(tile_num):
        x_local = asc.LocalTensorAuto(x.dtype, tile_length)
        y_local = asc.LocalTensorAuto(y.dtype, tile_length)
        z_local = asc.LocalTensorAuto(z.dtype, tile_length)
        copy_in(i, x_gm, y_gm, x_local, y_local, tile_length)
        compute(x_local, y_local, z_local, tile_length)
        copy_out(i, z_gm, z_local, tile_length)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, x_local: asc.LocalTensorAuto,
            y_local: asc.LocalTensorAuto, tile_length: asc.ConstExpr[int]):
    asc.data_copy(x_local, x_gm[i * tile_length:], count=tile_length)
    asc.data_copy(y_local, y_gm[i * tile_length:], count=tile_length)


@asc.jit
def compute(x_local: asc.LocalTensorAuto, y_local: asc.LocalTensorAuto, z_local: asc.LocalTensorAuto,
            tile_length: asc.ConstExpr[int]):
    asc.add(z_local, x_local, y_local, tile_length)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, z_local: asc.LocalTensorAuto, tile_length: asc.ConstExpr[int]):
    asc.data_copy(z_gm[i * tile_length:], z_local, count=tile_length)


def vadd_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    z = np.zeros_like(x)
    core_num = 16
    block_length = (z.size + core_num - 1) // core_num
    tile_num = 8
    tile_length = (block_length + tile_num - 1) // tile_num
    vadd_kernel[core_num, rt.current_stream()](x, y, z, block_length, tile_length, tile_num)
    return z


def test_vadd(backend: config.Backend):
    config.set_platform(backend)
    rng = np.random.default_rng(seed=2025)
    size = 8192
    x = rng.random(size, dtype=np.float32)
    y = rng.random(size, dtype=np.float32)
    z = vadd_launch(x, y)
    np.testing.assert_allclose(z, x + y)


if __name__ == "__main__":
    test_vadd(config.Backend.Model)
