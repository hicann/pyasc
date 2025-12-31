# Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations
from typing import overload

from ..._C import ir
from .utils import set_tensor_docstring
from ..core.dtype import DataType, KnownTypes
from ..core.ir_value import GlobalAddress, IRHandle, IRValue, PlainValue, RuntimeInt, materialize_ir_value as _mat
from ..core.utils import global_builder, require_jit
from ..core.tensor import GlobalTensor


class TensorDesc(IRValue):

    @overload
    def __init__(self, dtype: DataType = KnownTypes.float32) -> None:
        ...

    @overload
    def __init__(self, handle: IRHandle, dtype: DataType = KnownTypes.float32) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        dtype = kwargs.pop("dtype", KnownTypes.float32)
        if 'handle' in kwargs:
            self.handle = kwargs['handle']
            self.dtype = dtype
            return
        if global_builder.get_ir_builder() is not None:
            self.__initjit__(dtype=dtype)
            return
        builder = global_builder.get_ir_builder()
        self.dtype = dtype
        self.handle = builder.create_asc_TensorDescOp(
            builder.get_asc_TensorDescType(),
            dtype.to_ir()
        )


    @require_jit
    def __initjit__(self, dtype: DataType = KnownTypes.float32) -> None:
        builder = global_builder.get_ir_builder()
        self.dtype = dtype
        self.handle = builder.create_asc_TensorDescOp(
            builder.get_asc_TensorDescType(),
            dtype.to_ir()
        )

    @classmethod
    def from_ir(cls, handle: IRHandle, dtype: DataType = KnownTypes.float32) -> TensorDesc:
        return cls(handle=handle, dtype=dtype)

    def to_ir(self) -> IRHandle:
        return self.handle

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_dim")
    def get_dim(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetDimOp(builder.get_ui64_type(), self.to_ir())
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_index")
    def get_index(self) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetIndexOp(builder.get_ui64_type(), self.to_ir())
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_shape")
    def get_shape(self, offset: RuntimeInt) -> RuntimeInt:
        builder = global_builder.get_ir_builder()
        handle = builder.create_asc_TensorDescGetShapeOp(
            builder.get_i64_type(), 
            self.to_ir(),
            _mat(offset, KnownTypes.int32).to_ir()
        )
        return PlainValue(handle)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_data_ptr")
    def get_data_ptr(self) -> GlobalAddress:
        builder = global_builder.get_ir_builder()
        element_type = self.dtype.to_ir()
        handle = builder.create_asc_TensorDescGetDataPtrOp(
            ir.get_unranked_memref_type(element_type, ir.AddressSpace.gm),
            self.to_ir()
        )
        return GlobalAddress(handle, self.dtype)

    @require_jit
    @set_tensor_docstring(tensor_name="TensorDesc", api_name="get_data_obj")
    def get_data_obj(self) -> GlobalTensor:
        builder = global_builder.get_ir_builder()
        element_type = self.dtype.to_ir()
        handle = builder.create_asc_TensorDescGetDataObjOp(
            ir.get_global_tensor_type(element_type),
            self.to_ir()
        )
        return GlobalTensor(handle)
