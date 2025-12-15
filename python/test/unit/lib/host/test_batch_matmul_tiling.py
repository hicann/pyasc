# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.lib import host


def test_init(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    tiling = asc.adv.TCubeTiling()
    assert matmul_tiling.get_tiling(tiling) == 0


def test_set_a_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    ret = matmul_tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT, False)
    assert ret == 0


def test_set_b_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    ret = matmul_tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT, False)
    assert ret == 0


def test_set_c_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    ret = matmul_tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    assert ret == 0


def test_set_bias_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    ret = matmul_tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
    assert ret == 0


def test_set_shape(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    ret = matmul_tiling.set_shape(32, 16, 8)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.m == 32
    assert tiling.n == 16
    assert tiling.k_a == 8
    assert tiling.k_b == 8


def test_set_org_shape(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    ret = matmul_tiling.set_org_shape(32, 16, 8)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.m == 32
    assert tiling.n == 16
    assert tiling.k_a == 8
    assert tiling.k_b == 8


def test_set_org_shape_ka_kb(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    ret = matmul_tiling.set_org_shape(32, 16, 8, 4)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.m == 32
    assert tiling.n == 16
    assert tiling.k_a == 8
    assert tiling.k_b == 4


def test_set_fix_split(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_fix_split(32, 16, 8)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.base_m == 32
    assert tiling.base_n == 16
    assert tiling.base_k == 8


def test_set_buffer_space(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_buffer_space(-1, -1, -1)
    assert ret == 0


def test_set_traverse(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_traverse(host.MatrixTraverse.FIRSTM)
    assert ret == 0


def test_set_mad_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_mad_type(host.MatrixMadType.NORMAL)
    assert ret == 0


def test_set_split_range(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_split_range(32, 32, 32, 16, 16, 16)
    assert ret == 0


def test_set_double_buffer(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_double_buffer(True, True, True, True, True)
    assert ret == 0


def test_set_dequant_type(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_dequant_type(host.DequantType.SCALAR)
    assert ret == 0


def test_set_a_layout(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_a_layout(2, 32, 1, 3, 64)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.a_layout_info_b == 2
    assert tiling.a_layout_info_s == 32
    assert tiling.a_layout_info_n == 1
    assert tiling.a_layout_info_g == 3
    assert tiling.a_layout_info_d == 64


def test_set_b_layout(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_b_layout(2, 32, 1, 3, 64)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.b_layout_info_b == 2
    assert tiling.b_layout_info_s == 32
    assert tiling.b_layout_info_n == 1
    assert tiling.b_layout_info_g == 3
    assert tiling.b_layout_info_d == 64


def test_set_c_layout(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_c_layout(2, 32, 1, 3, 64)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.c_layout_info_b == 2
    assert tiling.c_layout_info_s1 == 32
    assert tiling.c_layout_info_n == 1
    assert tiling.c_layout_info_g == 3
    assert tiling.c_layout_info_s2 == 64


def test_set_batch_num(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_batch_num(3)
    tiling = asc.adv.TCubeTiling()
    matmul_tiling.get_tiling(tiling)
    assert ret == 0
    assert tiling.batch_num == 3


def test_set_batch_info_for_normal(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 256, 64)
    ret = matmul_tiling.set_batch_info_for_normal(3, 3, 32, 256, 64)
    assert ret == 0


def test_set_matmul_config_params_init(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 256, 64)
    matmul_config_params = host.MatmulConfigParams(1, False, host.ScheduleType.OUTER_PRODUCT,
                                                   host.MatrixTraverse.FIRSTM, False)
    ret = matmul_tiling.set_matmul_config_params(matmul_config_params)
    assert ret is None


def test_set_matmul_config_params(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 256, 64)
    ret = matmul_tiling.set_matmul_config_params(1, False, host.ScheduleType.OUTER_PRODUCT, host.MatrixTraverse.FIRSTM)
    assert ret is None


def test_set_sparse(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.set_sparse(True)
    assert ret == 0


def test_get_base_m(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    matmul_tiling.set_fix_split(32, 16, 8)
    base_m = matmul_tiling.get_base_m()
    assert base_m == 32


def test_get_base_n(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    matmul_tiling.set_fix_split(32, 16, 8)
    base_n = matmul_tiling.get_base_n()
    assert base_n == 16


def test_get_base_k(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    matmul_tiling.set_fix_split(32, 16, 8)
    base_k = matmul_tiling.get_base_k()
    assert base_k == 8


def test_enable_bias(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 16, 8)
    ret = matmul_tiling.enable_bias(True)
    assert ret == 0


def test_get_core_num(asc_platform):
    matmul_tiling = host.BatchMatmulTiling(asc_platform)
    matmul_tiling.set_shape(32, 32, 32)
    tiling = asc.adv.TCubeTiling()
    ret = matmul_tiling.get_tiling(tiling)
    result = matmul_tiling.get_core_num()
    assert ret == 0
    assert result is not None
    dim, m_dim, n_dim, batch_core_m, batch_core_n = result
    assert dim == 1
    assert m_dim == 1
    assert n_dim == 1
    assert batch_core_m == 1
    assert batch_core_n == 1
