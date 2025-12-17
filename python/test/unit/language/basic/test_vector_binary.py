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


def test_add_kernel(mock_launcher_run):

    @asc.jit
    def add_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.add(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.add(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.add(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    add_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sub_kernel(mock_launcher_run):

    @asc.jit
    def sub_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.sub(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.sub(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.sub(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    sub_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_mul_kernel(mock_launcher_run):

    @asc.jit
    def mul_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.mul(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.mul(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.mul(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    mul_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_div_kernel(mock_launcher_run):

    @asc.jit
    def div_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.div(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.div(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.div(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    div_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_max_kernel(mock_launcher_run):

    @asc.jit
    def max_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.max(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.max(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.max(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    max_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_min_kernel(mock_launcher_run):

    @asc.jit
    def min_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.min(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.min(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.min(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    min_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_and_kernel(mock_launcher_run):

    @asc.jit
    def and_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.bitwise_and(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.bitwise_and(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.bitwise_and(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    and_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_or_kernel(mock_launcher_run):

    @asc.jit
    def or_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.bitwise_or(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.bitwise_or(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.bitwise_or(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    or_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_add_relu_kernel(mock_launcher_run):

    @asc.jit
    def add_relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.add_relu(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.add_relu(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.add_relu(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    add_relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_add_relu_cast_kernel(mock_launcher_run):

    @asc.jit
    def add_relu_cast_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.add_relu_cast(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.add_relu_cast(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.add_relu_cast(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    add_relu_cast_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_add_deq_relu_kernel(mock_launcher_run):

    @asc.jit
    def add_deq_relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.add_deq_relu(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.add_deq_relu(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.add_deq_relu(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    add_deq_relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sub_relu_kernel(mock_launcher_run):

    @asc.jit
    def sub_relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.sub_relu(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.sub_relu(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.sub_relu(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    sub_relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sub_relu_cast_kernel(mock_launcher_run):

    @asc.jit
    def sub_relu_cast_kernel():
        x_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.sub_relu_cast(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.sub_relu_cast(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.sub_relu_cast(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    sub_relu_cast_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_mul_add_dst_kernel(mock_launcher_run):

    @asc.jit
    def mul_add_dst_kernel():
        x_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.mul_add_dst(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.mul_add_dst(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.mul_add_dst(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    mul_add_dst_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_mul_cast_kernel(mock_launcher_run):

    @asc.jit
    def mul_cast_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.mul_cast(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.mul_cast(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.mul_cast(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    mul_cast_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_fused_mul_add_kernel(mock_launcher_run):

    @asc.jit
    def fused_mul_add_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.fused_mul_add(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.fused_mul_add(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.fused_mul_add(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    fused_mul_add_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_fused_mul_add_relu_kernel(mock_launcher_run):

    @asc.jit
    def fused_mul_add_relu_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.fused_mul_add_relu(z_local, x_local, y_local, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.fused_mul_add_relu(z_local, x_local, y_local, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.fused_mul_add_relu(z_local, x_local, y_local, mask=mask, repeat_times=1, repeat_params=params)

    fused_mul_add_relu_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_bilinear_interpolation_kernel(mock_launcher_run):

    @asc.jit
    def bilinear_interpolation_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        x0_local = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        h_repeat = 2
        repeat_mode = False
        dst_blk_stride = 1
        v_r_offset = 128
        v_repeat = 2
        mask = 128
        asc.bilinear_interpolation(z_local, x_local, x0_local, y_local, mask, h_repeat, repeat_mode, dst_blk_stride,
                                   v_r_offset, v_repeat, tmp)
        uint64_max = 2**64 - 1
        mask_bits = [uint64_max, uint64_max]
        asc.bilinear_interpolation(z_local, x_local, x0_local, y_local, mask_bits, h_repeat, repeat_mode,
                                   dst_blk_stride, v_r_offset, v_repeat, tmp)

    bilinear_interpolation_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_compare_kernel(mock_launcher_run):

    @asc.jit
    def compare_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, count=512)
        params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_times=1, repeat_params=params)
        uint64_max = 2**64 - 1
        mask = [uint64_max, uint64_max]
        asc.compare(z_local, x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)
        asc.compare(x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=512, repeat_params=params)
        asc.compare(x_local, y_local, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    compare_kernel[1]()
    assert mock_launcher_run.call_count == 1
