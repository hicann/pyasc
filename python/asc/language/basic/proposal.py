# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload
from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import require_jit, global_builder
from .utils import set_common_docstring


@overload
def proposal_concat(dst: LocalTensor, src: LocalTensor, repeat_time: int, mode_number: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="proposal_concat")
def proposal_concat(dst: LocalTensor, src: LocalTensor, repeat_time: RuntimeInt, mode_number: RuntimeInt) -> None:
    global_builder.get_ir_builder().create_asc_ProposalConcatOp(dst.to_ir(), src.to_ir(),
                                                                _mat(repeat_time).to_ir(), _mat(mode_number).to_ir())


@overload
def proposal_extract(dst: LocalTensor, src: LocalTensor, repeat_time: int, mode_number: int) -> None:
    ...


@require_jit
@set_common_docstring(api_name="proposal_extract")
def proposal_extract(dst: LocalTensor, src: LocalTensor, repeat_time: RuntimeInt, mode_number: RuntimeInt) -> None:
    global_builder.get_ir_builder().create_asc_ProposalExtractOp(dst.to_ir(), src.to_ir(),
                                                                 _mat(repeat_time).to_ir(), _mat(mode_number).to_ir())
