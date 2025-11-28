# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..core.enums import HardEvent, PipeID, MemDsbT
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring


@require_jit
def data_sync_barrier(arg0: MemDsbT) -> None:
    global_builder.get_ir_builder().create_asc_DataSyncBarrierOp(arg0.value)


@require_jit
@set_common_docstring(api_name="pipe_barrier")
def pipe_barrier(pipe: PipeID) -> None:
    global_builder.get_ir_builder().create_asc_PipeBarrierOp(pipe)


@overload
def set_flag(event: HardEvent, event_id: int = 0) -> None:
    ...


@require_jit
@set_common_docstring(api_name="set_flag")
def set_flag(event: HardEvent, event_id: RuntimeInt = 0) -> None:
    event_id = _mat(event_id).to_ir()
    global_builder.get_ir_builder().create_asc_SetFlagOp(event, event_id)


@overload
def wait_flag(event: HardEvent, event_id: int = 0) -> None:
    ...


@require_jit
@set_common_docstring(api_name="wait_flag")
def wait_flag(event: HardEvent, event_id: RuntimeInt = 0) -> None:
    event_id = _mat(event_id).to_ir()
    global_builder.get_ir_builder().create_asc_WaitFlagOp(event, event_id)
