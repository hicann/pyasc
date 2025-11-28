.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

asc.language.fwk
==================================

.. currentmodule:: asc.language.fwk


GetTPipePtr
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    get_tpipe_ptr


TBuf
------------
.. autoclass:: TBuf
.. autosummary::
    :toctree: generated
    :nosignatures:

    TBuf.get
    TBuf.get_with_offset


TBufPool
------------
.. autoclass:: TBufPool
.. autosummary::
    :toctree: generated
    :nosignatures:

    TBufPool.init_buf_pool
    TBufPool.init_buffer
    TBufPool.reset


TPipe
------------
.. autoclass:: TPipe
.. autosummary::
    :toctree: generated
    :nosignatures:

    TPipe.alloc_event_id
    TPipe.destroy
    TPipe.fetch_event_id
    TPipe.init
    TPipe.init_buf_pool
    TPipe.init_buffer
    TPipe.release_event_id
    TPipe.reset


TQue
------------
.. autoclass:: TQue
.. autosummary::
    :toctree: generated
    :nosignatures:

    TQue.alloc_tensor
    TQue.deque
    TQue.enque
    TQue.free_tensor
    TQue.get_tensor_count_in_que
    TQue.has_idle_buffer
    TQue.has_tensor_in_que
    TQue.vacant_in_que


TQueBind
------------
.. autoclass:: TQueBind
.. autosummary::
    :toctree: generated
    :nosignatures:

    TQueBind.alloc_tensor
    TQueBind.deque
    TQueBind.enque
    TQueBind.free_all_event
    TQueBind.free_tensor
    TQueBind.get_tensor_count_in_que
    TQueBind.has_idle_buffer
    TQueBind.has_tensor_in_que
    TQueBind.vacant_in_que
