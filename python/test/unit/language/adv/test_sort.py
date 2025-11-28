# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
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


def test_concat_kernel(mock_launcher_run):

    @asc.jit
    def concat_kernel():
        concat_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=128)
        concat_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)

        asc.adv.concat(concat_local, value_local, concat_tmp_local, 4)

    concat_kernel[1]()
    assert mock_launcher_run.call_count == 1


def test_extract_kernel(mock_launcher_run):

    @asc.jit
    def extract_kernel():
        dst_value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        dst_index_local = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
        sort_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)

        asc.adv.extract(dst_value_local, dst_index_local, sort_tmp_local, 8)

    extract_kernel[1]()
    assert mock_launcher_run.call_count == 1
