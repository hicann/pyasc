.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

asc.language.core
==================================

.. currentmodule:: asc.language.core


GlobalTensor
------------
.. autoclass:: GlobalTensor
.. autosummary::
    :toctree: generated
    :nosignatures:

    GlobalTensor.get_phy_addr
    GlobalTensor.get_shape_info
    GlobalTensor.get_size
    GlobalTensor.get_value
    GlobalTensor.set_global_buffer
    GlobalTensor.set_l2_cache_hint
    GlobalTensor.set_shape_info
    GlobalTensor.set_value


LocalMemAllocator
-----------------
.. autoclass:: LocalMemAllocator
.. autosummary::
    :toctree: generated
    :nosignatures:

    LocalMemAllocator.alloc
    LocalMemAllocator.get_cur_addr


LocalTensor
------------
.. autoclass:: LocalTensor
.. autosummary::
    :toctree: generated
    :nosignatures:

    LocalTensor.get_length
    LocalTensor.get_phy_addr
    LocalTensor.get_position
    LocalTensor.get_shape_info
    LocalTensor.get_size
    LocalTensor.get_user_tag
    LocalTensor.get_value
    LocalTensor.reinterpret_cast
    LocalTensor.set_addr_with_offset
    LocalTensor.set_buffer_len
    LocalTensor.set_shape_info
    LocalTensor.set_size
    LocalTensor.set_user_tag
    LocalTensor.set_value
