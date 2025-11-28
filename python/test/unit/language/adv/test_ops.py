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


def test_acos_kernel(mock_launcher_run):

    @asc.jit
    def acos_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.acos(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.acos(z_local, x_local, count=512)

    acos_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_acosh_kernel(mock_launcher_run):

    @asc.jit
    def acosh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.acosh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.acosh(z_local, x_local, count=512)

    acosh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_asin_kernel(mock_launcher_run):

    @asc.jit
    def asin_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.asin(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.asin(z_local, x_local, count=512)

    asin_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_asinh_kernel(mock_launcher_run):

    @asc.jit
    def asinh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.asinh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.asinh(z_local, x_local, count=512)

    asinh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_atan_kernel(mock_launcher_run):

    @asc.jit
    def atan_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.atan(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.atan(z_local, x_local, count=512)

    atan_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_atanh_kernel(mock_launcher_run):

    @asc.jit
    def atanh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.atanh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.atanh(z_local, x_local, count=512)

    atanh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_ceil_kernel(mock_launcher_run):

    @asc.jit
    def ceil_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.ceil(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.ceil(z_local, x_local, count=512)

    ceil_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_cos_kernel(mock_launcher_run):

    @asc.jit
    def cos_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.cos(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.cos(z_local, x_local, count=512)

    cos_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_cosh_kernel(mock_launcher_run):

    @asc.jit
    def cosh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.cosh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.cosh(z_local, x_local, count=512)

    cosh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_digamma_kernel(mock_launcher_run):

    @asc.jit
    def digamma_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.digamma(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.digamma(z_local, x_local, count=512)

    digamma_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_erf_kernel(mock_launcher_run):

    @asc.jit
    def erf_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.erf(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.erf(z_local, x_local, count=512)

    erf_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_erfc_kernel(mock_launcher_run):

    @asc.jit
    def erfc_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.erfc(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.erfc(z_local, x_local, count=512)

    erfc_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_exp_kernel(mock_launcher_run):

    @asc.jit
    def exp_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.exp(z_local, x_local, count=512, taylor_expand_level=0, temp_buffer=tmp)
        asc.adv.exp(z_local, x_local, count=512, taylor_expand_level=0, is_reuse_source=True)

    exp_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_floor_kernel(mock_launcher_run):

    @asc.jit
    def floor_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.floor(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.floor(z_local, x_local, count=512)

    floor_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_frac_kernel(mock_launcher_run):

    @asc.jit
    def frac_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.frac(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.frac(z_local, x_local, count=512)

    frac_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_lgamma_kernel(mock_launcher_run):

    @asc.jit
    def lgamma_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.lgamma(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.lgamma(z_local, x_local, count=512)

    lgamma_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_log_kernel(mock_launcher_run):

    @asc.jit
    def log_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.log(z_local, x_local)

    log_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_round_kernel(mock_launcher_run):

    @asc.jit
    def round_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.round(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.round(z_local, x_local, count=512)

    round_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sign_kernel(mock_launcher_run):

    @asc.jit
    def sign_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.sign(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.sign(z_local, x_local, count=512)

    sign_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sin_kernel(mock_launcher_run):

    @asc.jit
    def sin_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.sin(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.sin(z_local, x_local, count=512)

    sin_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_sinh_kernel(mock_launcher_run):

    @asc.jit
    def sinh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.sinh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.sinh(z_local, x_local, count=512)

    sinh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_tan_kernel(mock_launcher_run):

    @asc.jit
    def tan_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.tan(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.tan(z_local, x_local, count=512)

    tan_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_tanh_kernel(mock_launcher_run):

    @asc.jit
    def tanh_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.tanh(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.tanh(z_local, x_local, count=512)

    tanh_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_trunc_kernel(mock_launcher_run):

    @asc.jit
    def trunc_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.trunc(z_local, x_local, count=512, temp_buffer=tmp)
        asc.adv.trunc(z_local, x_local, count=512)

    trunc_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_power_kernel(mock_launcher_run):

    @asc.jit
    def power_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.power(z_local, x_local, y_local, count=512, temp_buffer=tmp)
        asc.adv.power(z_local, x_local, y_local, count=512)

    power_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_xor_kernel(mock_launcher_run):

    @asc.jit
    def xor_kernel():
        x_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        y_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.int16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.xor(z_local, x_local, y_local, count=512, temp_buffer=tmp)
        asc.adv.xor(z_local, x_local, y_local, count=512)

    xor_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_axpy_kernel(mock_launcher_run):

    @asc.jit
    def axpy_kernel():
        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        tmp = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.VECCALC, addr=0, tile_size=512)
        asc.adv.axpy(z_local, x_local, 3.0, count=512, temp_buffer=tmp)

    axpy_kernel[1]()
    assert mock_launcher_run.call_count == 1
