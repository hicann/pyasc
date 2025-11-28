.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

asc.lib.host
==================================

.. currentmodule:: asc.lib.host

Ascend C提供一组Matmul Tiling API，方便用户获取Matmul kernel计算时所需的Tiling参数。
用户只需要传入A/B/C矩阵的Position位置、Format格式和DType数据类型等信息，调用API接口，即可获取到Init中TCubeTiling结构体中的相关参数。

MatmulApiTiling/MultiCoreMatmulTiling/BatchMatmulTiling共有接口列表
---------------------------------------------------------------------
.. autosummary::
    :toctree: generated

    MatmulApiTiling.enable_bias
    MatmulApiTiling.get_base_k
    MatmulApiTiling.get_base_m
    MatmulApiTiling.get_base_n
    MatmulApiTiling.get_tiling
    MatmulApiTiling.set_a_layout
    MatmulApiTiling.set_a_type
    MatmulApiTiling.set_b_layout
    MatmulApiTiling.set_b_type
    MatmulApiTiling.set_batch_info_for_normal
    MatmulApiTiling.set_batch_num
    MatmulApiTiling.set_bias_type
    MatmulApiTiling.set_buffer_space
    MatmulApiTiling.set_c_layout
    MatmulApiTiling.set_c_type
    MatmulApiTiling.set_dequant_type
    MatmulApiTiling.set_double_buffer
    MatmulApiTiling.set_fix_split
    MatmulApiTiling.set_mad_type
    MatmulApiTiling.set_matmul_config_params
    MatmulApiTiling.set_org_shape
    MatmulApiTiling.set_shape
    MatmulApiTiling.set_sparse
    MatmulApiTiling.set_split_range
    MatmulApiTiling.set_traverse


MultiCoreMatmulTiling其他接口
----------------------------------
.. autosummary::
    :toctree: generated

    MultiCoreMatmulTiling.enable_multi_core_split_k
    MultiCoreMatmulTiling.get_core_num
    MultiCoreMatmulTiling.get_single_shape
    MultiCoreMatmulTiling.set_align_split
    MultiCoreMatmulTiling.set_dim
    MultiCoreMatmulTiling.set_single_range
    MultiCoreMatmulTiling.set_single_shape



BatchMatmulTiling其他接口
----------------------------------
.. autosummary::
    :toctree: generated

    BatchMatmulTiling.get_core_num
