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


@asc.jit(kernel_type=config.KernelType.AIV_ONLY, insert_sync=True)
def softmax_kernel(src: asc.GlobalAddress, dst: asc.GlobalAddress, height: asc.ConstExpr[int],
                   width: asc.ConstExpr[int], tile_num: asc.ConstExpr[int], block_num: asc.ConstExpr[int]):
    assert src.dtype == dst.dtype, "type not equal"
    total_length = height * width
    block_length = total_length // block_num
    tile_length = block_length // tile_num
    count = tile_length // width
    src_global = asc.GlobalTensor()
    dst_global = asc.GlobalTensor()
    src_global.set_global_buffer(src)
    dst_global.set_global_buffer(dst)
    for i in asc.range(tile_num):
        offset = asc.get_block_idx() * block_length + i * tile_length
        src_local = asc.LocalTensorAuto(src.dtype, tile_length)
        sum_local = asc.LocalTensorAuto(src.dtype, (height // block_num // tile_num) * count)
        max_local = asc.LocalTensorAuto(src.dtype, (height // block_num // tile_num) * count)
        dst_local = asc.LocalTensorAuto(dst.dtype, tile_length)
        asc.data_copy(src_local, src_global[offset:], count=tile_length)
        shape = asc.ShapeInfo(asc.array(asc.int32, (height // block_num // tile_num, width)))
        src_local.set_shape_info(shape)
        dst_local.set_shape_info(shape)
        shape = asc.ShapeInfo(asc.array(asc.int32, (height // block_num // tile_num, count)))
        sum_local.set_shape_info(shape)
        max_local.set_shape_info(shape)
        tiling = asc.adv.SoftmaxTiling()
        asc.adv.softmax(src_local, sum_local, max_local, src_local, tiling)
        asc.data_copy(dst_local, src_local, count=tile_length)
        asc.data_copy(dst_global[offset:], dst_local, count=tile_length)


def softmax_ascend(src: np.ndarray) -> np.ndarray:
    dst = np.zeros_like(src)
    height, width = src.shape
    softmax_kernel[16, rt.current_stream()](src, dst, height, width, tile_num=8, block_num=16)
    return dst


def softmax_numpy(tensor: np.ndarray) -> np.ndarray:
    tensor = tensor - np.max(tensor)
    return np.transpose(np.exp(np.transpose(tensor)) / sum(np.exp(np.transpose(tensor))))


def test_softmax(backend: config.Backend):
    config.set_platform(backend)
    rng = np.random.default_rng(seed=2025)
    height, width = 1024, 1024
    src = rng.random((height, width), dtype=np.float32) * 5.0  # [0, 5]
    np.testing.assert_allclose(softmax_ascend(src), softmax_numpy(src), atol=1 / src.size)


if __name__ == "__main__":
    test_softmax(config.Backend.Model)
