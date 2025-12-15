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


@asc.jit
def vadd_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, BLOCK_LENGTH: asc.ConstExpr[int],
                BUFFER_NUM: asc.ConstExpr[int], TILE_LENGTH: asc.ConstExpr[int], TILE_NUM: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * BLOCK_LENGTH
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset)
    y_gm.set_global_buffer(y + offset)
    z_gm.set_global_buffer(z + offset)
    pipe = asc.TPipe()
    in_queue_x = asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)
    in_queue_y = asc.TQue(asc.TPosition.VECIN, BUFFER_NUM)
    out_queue_z = asc.TQue(asc.TPosition.VECOUT, BUFFER_NUM)
    pipe.init_buffer(que=in_queue_x, num=BUFFER_NUM, len=TILE_LENGTH * x.dtype.sizeof())
    pipe.init_buffer(que=in_queue_y, num=BUFFER_NUM, len=TILE_LENGTH * y.dtype.sizeof())
    pipe.init_buffer(que=out_queue_z, num=BUFFER_NUM, len=TILE_LENGTH * z.dtype.sizeof())
    for i in range(TILE_NUM):
        copy_in(i, x_gm, y_gm, in_queue_x, in_queue_y, TILE_LENGTH)
        compute(z_gm, in_queue_x, in_queue_y, out_queue_z, TILE_LENGTH)
        copy_out(i, z_gm, out_queue_z, TILE_LENGTH)


@asc.jit
def copy_in(i: int, x_gm: asc.GlobalAddress, y_gm: asc.GlobalAddress, in_queue_x: asc.TQue, in_queue_y: asc.TQue,
            TILE_LENGTH: asc.ConstExpr[int]):
    x_local = in_queue_x.alloc_tensor(x_gm.dtype)
    y_local = in_queue_y.alloc_tensor(y_gm.dtype)
    asc.data_copy(x_local, x_gm[i * TILE_LENGTH:], count=TILE_LENGTH)
    asc.data_copy(y_local, y_gm[i * TILE_LENGTH:], count=TILE_LENGTH)
    in_queue_x.enque(x_local)
    in_queue_y.enque(y_local)


@asc.jit
def compute(z_gm: asc.GlobalTensor, in_queue_x: asc.TQue, in_queue_y: asc.TQue, out_queue_z: asc.TQue,
            TILE_LENGTH: asc.ConstExpr[int]):
    # "z_gm" is passed here to obtain dtype
    x_local = in_queue_x.deque(z_gm.dtype)
    y_local = in_queue_y.deque(z_gm.dtype)
    z_local = out_queue_z.alloc_tensor(z_gm.dtype)
    asc.add(z_local, x_local, y_local, TILE_LENGTH)
    out_queue_z.enque(z_local)
    in_queue_x.free_tensor(x_local)
    in_queue_y.free_tensor(y_local)


@asc.jit
def copy_out(i: int, z_gm: asc.GlobalTensor, out_queue_z: asc.TQue, TILE_LENGTH: asc.ConstExpr[int]):
    z_local = out_queue_z.deque(z_gm.dtype)
    asc.data_copy(z_gm[i * TILE_LENGTH:], z_local, count=TILE_LENGTH)
    out_queue_z.free_tensor(z_local)


def vadd_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    z = np.zeros_like(x)
    TOTAL_LENGTH = z.size
    USE_CORE_NUM = 16
    BLOCK_LENGTH = (TOTAL_LENGTH + USE_CORE_NUM - 1) // USE_CORE_NUM
    TILE_NUM = 8
    TILE_LENGTH = (BLOCK_LENGTH + TILE_NUM - 1) // TILE_NUM
    BUFFER_NUM = 1
    vadd_kernel[USE_CORE_NUM, rt.current_stream()](x, y, z, BLOCK_LENGTH, BUFFER_NUM, TILE_LENGTH, TILE_NUM)
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
