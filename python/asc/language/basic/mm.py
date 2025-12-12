# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from ..core.ir_value import RuntimeInt, materialize_ir_value as _mat
from ..core.tensor import BaseTensor, GlobalTensor, LocalTensor
from ..core.types import LoadData2DParams, LoadData2dTransposeParams, MmadParams
from ..core.utils import OverloadDispatcher, require_jit, global_builder
from .utils import set_common_docstring


@overload
def load_data(dst: LocalTensor, src: LocalTensor, params: LoadData2DParams) -> None:
    ...


@overload
def load_data(dst: LocalTensor, src: GlobalTensor, params: LoadData2DParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="load_data")
def load_data(dst: BaseTensor, src: BaseTensor, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(params: LoadData2DParams):

        if isinstance(dst, LocalTensor) and isinstance(src, LocalTensor):
            builder.create_asc_LoadDataL0Op(
                dst.to_ir(),
                src.to_ir(),
                params.to_ir(),
            )
            return

        if isinstance(dst, LocalTensor) and isinstance(src, GlobalTensor):
            builder.create_asc_LoadDataG2LOp(
                dst.to_ir(),
                src.to_ir(),
                params.to_ir(),
            )
            return

    dispatcher(*args, **kwargs)


@overload
def load_data_with_transpose(dst: LocalTensor, src: LocalTensor, params: LoadData2dTransposeParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="load_data_with_transpose")
def load_data_with_transpose(dst: BaseTensor, src: BaseTensor, *args, **kwargs) -> None:
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(params: LoadData2dTransposeParams):
        builder.create_asc_LoadDataWithTransposeOp(
            dst.to_ir(),
            src.to_ir(),
            params.to_ir(),
        )

    dispatcher(*args, **kwargs)


@overload
def mmad(dst: LocalTensor, fm: LocalTensor, filter: LocalTensor, params: MmadParams) -> None:
    ...


@overload
def mmad(dst: LocalTensor, fm: LocalTensor, filter: LocalTensor, bias: LocalTensor, params: MmadParams) -> None:
    ...


@require_jit
@set_common_docstring(api_name="mmad")
def mmad(dst: BaseTensor, fm: BaseTensor, filter: BaseTensor, *args, **kwargs) -> None:
    """
    Matrix multiply-accumulate:
      Mmad(dst, fm, filter, params)
      Mmad(dst, fm, filter, bias, params)
    """
    dispatcher = OverloadDispatcher(__name__)
    builder = global_builder.get_ir_builder()

    @dispatcher.register_auto
    def _(params: MmadParams):
        builder.create_asc_MmadOp(
            dst.to_ir(),
            fm.to_ir(),
            filter.to_ir(),
            params.to_ir(),
        )
        return

    @dispatcher.register_auto
    def _(bias: LocalTensor, params: MmadParams):
        builder.create_asc_MmadWithBiasOp(
            dst.to_ir(),
            fm.to_ir(),
            filter.to_ir(),
            bias.to_ir(),
            params.to_ir(),
        )
        return

    dispatcher(*args, **kwargs)
