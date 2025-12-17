# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config


def setup_function():
    config.set_platform(config.Backend.Model, check=False)


def test_adds_kernel(mock_launcher_run):

    @asc.jit
    def adds_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.adds(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.adds(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.adds(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    adds_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_leaky_relu_kernel(mock_launcher_run):

    @asc.jit
    def leaky_relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.leaky_relu(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.leaky_relu(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.leaky_relu(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    leaky_relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_muls_kernel(mock_launcher_run):

    @asc.jit
    def muls_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.muls(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.muls(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.muls(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    muls_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_maxs_kernel(mock_launcher_run):

    @asc.jit
    def maxs_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.maxs(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.maxs(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.maxs(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    maxs_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_mins_kernel(mock_launcher_run):

    @asc.jit
    def mins_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.mins(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.mins(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.mins(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    mins_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_shift_left_kernel(mock_launcher_run):

    @asc.jit
    def shift_left_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.shift_left(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.shift_left(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.shift_left(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    shift_left_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_shift_right_kernel(mock_launcher_run):

    @asc.jit
    def shift_right_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.shift_right(z_local, x_local, 1, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.shift_right(z_local, x_local, 1, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.shift_right(z_local, x_local, 1, mask=mask, repeat_times=1, repeat_params=params)

    shift_right_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_compare_scalar_kernel(mock_launcher_run):

    @asc.jit
    def compare_scalar_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, count=512)
        params = asc.UnaryRepeatParams(1, 1, 8, 8)
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.compare_scalar(z_local, x_local, 1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, 
                           repeat_params=params)

    compare_scalar_kernel[1]()
    assert mock_launcher_run.call_count == 1

