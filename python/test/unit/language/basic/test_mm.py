# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import asc
from asc.runtime import config
from asc.runtime.jit import MockTensor


def test_load_data(mock_launcher_run):

    @asc.jit
    def kernel_load_data(x: asc.GlobalAddress) -> None:
        x_local = asc.LocalTensor(
            dtype=asc.float16,
            pos=asc.TPosition.VECIN,
            addr=0,
            tile_size=512,
        )
        y_local = asc.LocalTensor(
            dtype=asc.float16,
            pos=asc.TPosition.VECOUT,
            addr=0,
            tile_size=512,
        )
        x_gm = asc.GlobalTensor()
        x_gm.set_global_buffer(x)

        params_2d = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)
        params_2d_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)
        params_3d_v2_pro = asc.LoadData3DParamsV2Pro(16, False, False, False, False, False, 0, 0x10101010101)

        asc.load_data(y_local, x_local, params_2d)
        asc.load_data(x_local, x_gm, params_2d)
        asc.load_data(y_local, x_local, params_2d_v2)
        asc.load_data(x_local, x_gm, params_2d_v2)
        asc.load_data(y_local, x_local, params_3d_v2_pro)

    x = MockTensor(asc.float16)
    kernel_load_data[1](x)
    assert mock_launcher_run.call_count == 1


def test_load_data_with_transpose(mock_launcher_run):

    @asc.jit
    def kernel_load_data_transpose() -> None:
        x_local = asc.LocalTensor(
            dtype=asc.float16,
            pos=asc.TPosition.VECIN,
            addr=0,
            tile_size=512,
        )
        y_local = asc.LocalTensor(
            dtype=asc.float16,
            pos=asc.TPosition.VECOUT,
            addr=0,
            tile_size=512,
        )

        params_v1 = asc.LoadData2dTransposeParams(0, 4, 0, 0, 0, 0)
        params_v2 = asc.LoadData2dTransposeParamsV2(0, 4, 0, 0, 0, 0, 0)

        asc.load_data_with_transpose(y_local, x_local, params_v1)
        asc.load_data_with_transpose(y_local, x_local, params_v2)

    kernel_load_data_transpose[1]()
    assert mock_launcher_run.call_count == 1


def test_mmad(mock_launcher_run):

    @asc.jit
    def kernel_mmad():
        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=1024)
        fm = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=1024)
        filter = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=1024)
        bias = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=1024)
        params = asc.MmadParams(4, 4, 4)

        asc.mmad(dst, fm, filter, params)
        asc.mmad(dst, fm, filter, bias, params)


    kernel_mmad[1]()
    assert mock_launcher_run.call_count == 1
