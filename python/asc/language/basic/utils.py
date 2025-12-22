# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from ..._C import ir
from ..core.dtype import KnownTypes as KT
from ..core.ir_value import RuntimeInt, RuntimeNumeric, materialize_ir_value as _mat
from ..core.tensor import LocalTensor
from ..core.utils import DefaultValued, OverloadDispatcher
from ..core.types import BinaryRepeatParams, UnaryRepeatParams

T = TypeVar("T", bound=Callable)


def print_valid_type(origin):
    converted = {}
    for key, types in origin.items():
        converted[key] = [t.name for t in types]
    return converted


def check_type(callee: str, dst: LocalTensor, src0: LocalTensor, src1: LocalTensor) -> None:
    valids = {"src": [KT.float16, KT.float32, KT.int16, KT.int32], "dst": [KT.float16, KT.float32, KT.int16, KT.int32]}
    valids_relu = {"src": [KT.float16, KT.float32, KT.int16], "dst": [KT.float16, KT.float32, KT.int16]}
    valids_relu_cast = {"src": [KT.float16, KT.float32, KT.int16], "dst": [KT.int8, KT.float16]}
    valids_int = {"src": [KT.int16, KT.uint16], "dst": [KT.int16, KT.uint16]}
    valids_float = {"src": [KT.float16, KT.float32], "dst": [KT.float16, KT.float32]}

    valids_map = {
        "add": valids,
        "add_deq_relu": {"src": [KT.int32], "dst": [KT.float16]},
        "add_relu": valids_relu,
        "add_relu_cast": valids_relu_cast,
        "bilinear_interpolation": {"src": [KT.float16], "dst": [KT.float16]},
        "bitwise_and": valids_int,
        "bitwise_or": valids_int,
        "div": valids_float,
        "fused_mul_add": valids_float,
        "fused_mul_add_relu": valids_float,
        "max": valids,
        "min": valids,
        "mul": valids,
        "mul_add_dst": valids_float,
        "mul_cast": {"src": [KT.float16], "dst": [KT.int8, KT.uint8]},
        "sub": valids,
        "sub_relu": valids_relu,
        "sub_relu_cast": valids_relu_cast,
    }

    check_dst_src = {"add_deq_relu", "add_relu_cast", "bilinear_interpolation", "mul_cast", "sub_relu_cast"}

    if dst.dtype not in valids_map.get(callee).get("dst"):
        raise TypeError(f"Invalid dst data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src0.dtype not in valids_map.get(callee).get("src"):
        raise TypeError(f"Invalid src0 data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src1.dtype not in valids_map.get(callee).get("src"):
        raise TypeError(f"Invalid src1 data type, got {dst.dtype}, expect {print_valid_type(valids_map.get(callee))}")
    if src0.dtype != src1.dtype:
        raise TypeError("Src0 and src1 must be same type.")
    if callee not in check_dst_src:
        if not (dst.dtype == src0.dtype and dst.dtype == src1.dtype):
            raise TypeError("Src0, src1 and dst must be same type.")


def check_type_transpose(callee: str, dst: LocalTensor, src: LocalTensor, *args) -> None:
    if dst.dtype != src.dtype:
        raise TypeError(f"For {callee}, dst and src tensor must have the same dtype, "
                        f"got dst: {dst.dtype}, src: {src.dtype}")

    if args:
        shared_tmp_buffer = args[0]
        if not isinstance(shared_tmp_buffer, LocalTensor):
            raise TypeError("shared_tmp_buffer must be a LocalTensor")
        if shared_tmp_buffer.dtype != KT.uint8:
            raise TypeError(f"shared_tmp_buffer must have dtype uint8, got {shared_tmp_buffer.dtype}")


def check_type_5hd(callee: str, dst_or_list, src_or_list) -> None:
    if isinstance(dst_or_list, LocalTensor):
        if dst_or_list.dtype != KT.uint64 or src_or_list.dtype != KT.uint64:
            raise TypeError(f"For {callee} with LocalTensor inputs, dtype must be uint64.")
    elif isinstance(dst_or_list, list):
        if not dst_or_list or not src_or_list:
            return
        
        if len(dst_or_list) != len(src_or_list):
            raise ValueError("For {callee}, dst_list and src_list must have the same length.")

        if isinstance(dst_or_list[0], LocalTensor):
            first_dtype = dst_or_list[0].dtype
            if any(t.dtype != first_dtype for t in dst_or_list) or \
               any(t.dtype != first_dtype for t in src_or_list):
                raise TypeError(f"For {callee}, all tensors in dst_list and src_list must have the same dtype.")
        else:
            if not all(isinstance(x, RuntimeInt) for x in dst_or_list) or \
               not all(isinstance(x, RuntimeInt) for x in src_or_list):
                raise TypeError(f"For {callee}, address lists must contain only RuntimeInt.")
    else:
        raise TypeError(f"Unsupported input types for {callee}: {type(dst_or_list)}")


def op_impl(callee: str, dst: LocalTensor, src0: LocalTensor, src1: LocalTensor, args: Tuple[Any],
            kwargs: Dict[str, Any], build_l0: Callable, build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    dispatcher = OverloadDispatcher(callee)

    check_type(callee, dst, src0, src1)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src0.to_ir(), src1.to_ir(),
                 _mat(mask, KT.uint64).to_ir(), _mat(repeat_times, KT.int8).to_ir(), 
                repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=BinaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: BinaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src0.to_ir(), src1.to_ir(), mask, _mat(repeat_times, KT.int8).to_ir(), 
                repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(count: RuntimeInt, is_set_mask: bool = True):
        build_l2(dst.to_ir(), src0.to_ir(), src1.to_ir(), _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


def vec_binary_scalar_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, 
                              args: Tuple[Any], kwargs: Dict[str, Any], build_l0: Callable, 
                              build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    scalar = _mat(scalar, src.dtype).to_ir()
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(), scalar,
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)
    
    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), scalar, mask, _mat(repeat_times, KT.int8).to_ir(), 
                repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt, is_set_mask=DefaultValued(bool, True))
    def _(count: RuntimeInt, is_set_mask: bool = True):
        build_l2(dst.to_ir(), src.to_ir(), scalar, _mat(count, KT.int32).to_ir(), is_set_mask)

    dispatcher(*args, **kwargs)
    

def vec_ternary_scalar_op_impl(callee: str, dst: LocalTensor, src: LocalTensor, scalar: RuntimeNumeric, 
                               args: Tuple[Any], kwargs: Dict[str, Any], build_l0: Callable, 
                               build_l1: Callable, build_l2: Callable) -> None:
    builder = build_l0.__self__
    if not isinstance(builder, ir.Builder):
        raise TypeError("Input builder must be ir.Builder")
    scalar = _mat(scalar, src.dtype).to_ir()
    dispatcher = OverloadDispatcher(callee)

    @dispatcher.register(mask=RuntimeInt, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: RuntimeInt, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        build_l0(dst.to_ir(), src.to_ir(), scalar,
                 _mat(mask, KT.uint64).to_ir(),
                 _mat(repeat_times, KT.int8).to_ir(), repeat_params.to_ir(), is_set_mask)
    
    @dispatcher.register(mask=list, repeat_times=RuntimeInt, repeat_params=UnaryRepeatParams, 
                        is_set_mask=DefaultValued(bool, True))
    def _(mask: list, repeat_times: RuntimeInt, repeat_params: UnaryRepeatParams, is_set_mask: bool = True):
        mask = [_mat(v, KT.uint64).to_ir() for v in mask]
        build_l1(dst.to_ir(), src.to_ir(), scalar, mask, _mat(repeat_times, KT.int8).to_ir(), 
                repeat_params.to_ir(), is_set_mask)

    @dispatcher.register(count=RuntimeInt)
    def _(count: RuntimeInt):
        build_l2(dst.to_ir(), src.to_ir(), scalar, _mat(count, KT.int32).to_ir())

    dispatcher(*args, **kwargs)


def copy_docstring():
    func_introduction = """
    在 Vector Core 的不同内部存储单元（VECIN, VECCALC, VECOUT）之间进行数据搬运。

    这是一个矢量指令，支持通过掩码（mask）进行灵活的数据块选择，并通过重复参数（repeat parameters）
    实现高效的间隔操作和高维数据处理。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    该接口支持两种掩码（mask）模式，以进行高维切分计算。

    1. **mask 为逐 bit 模式**

       .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
                                    const uint64_t mask[], const uint8_t repeatTime, 
                                    const CopyRepeatParams& repeatParams)
    
    2. **mask 为连续模式**

       .. code-block:: c++
        
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
                                    const uint64_t mask, const uint8_t repeatTime, 
                                    const CopyRepeatParams& repeatParams)
    """

    param_list = """
    **参数说明**

    - dst (asc.LocalTensor): 目标操作数。

      - 必须是 `LocalTensor`。
      - 支持的 TPosition 为 `asc.Position.VECIN`, `asc.Position.VECCALC`, `asc.Position.VECOUT`。
      - 起始地址需要 32 字节对齐。

    - src (asc.LocalTensor): 源操作数。

      - 必须是 `LocalTensor`，且数据类型与 `dst` 保持一致。
      - 支持的 TPosition 为 `asc.Position.VECIN`, `asc.Position.VECCALC`, `asc.Position.VECOUT`。
      - 起始地址需要 32 字节对齐。

    - mask (Union[int, List[int]]): 掩码，用于控制在单次迭代中哪些元素参与搬运。

      - **连续模式** (当 `mask` 为 `int`): 表示从起始位置开始，连续搬运多少个元素。

        - 当数据类型为 16-bit (如 `fp16`) 时，取值范围是 [1, 128]。
        - 当数据类型为 32-bit (如 `fp32`) 时，取值范围是 [1, 64]。

      - **逐 bit 模式** (当 `mask` 为 `List[int]`): 掩码数组中的每个 bit 对应一个元素，bit 为 1 表示搬运，0 表示跳过。

        - 当数据类型为 16-bit 时，`mask` 是一个长度为 2 的列表，例如 `mask=[mask0, mask1]`。
        - 当数据类型为 32-bit 时，`mask` 是一个长度为 1 的列表，例如 `mask=[mask0]`。

    - repeat_time (int): 重复迭代次数。矢量计算单元每次处理一个数据块（256字节），此参数指定了处理整个 Tensor 需要重复迭代的次数。
    - repeat_params (asc.CopyRepeatParams): 控制地址步长的数据结构，用于处理高维或非连续数据。

      - `dstStride`, `srcStride`: 设置同一次迭代内，不同数据块（DataBlock）之间的地址步长。
      - `dstRepeatSize`, `srcRepeatSize`: 设置相邻两次迭代之间的地址步长。

    - is_set_mask (bool, 可选): 模板参数，默认为 `True`。

      - `True`: 在接口内部设置 `mask` 值。
      - `False`: 在接口外部通过 `asc.set_vector_mask` 接口设置 `mask`，此时 `mask` 参数必须为占位符 `asc.MASK_PLACEHOLDER`。
    """

    constraint_list = """
    **约束说明**

    - 源操作数和目的操作数的起始地址需要保证32字节对齐。
    - Copy和矢量计算API一样，支持和掩码操作API配合使用。但Counter模式配合高维切分计算API时，和通用的Counter模式有一定差异。具体差异如下：
      - 通用的Counter模式：Mask代表整个矢量计算参与计算的元素个数，迭代次数不生效。
      - Counter模式配合Copy高维切分计算API，Mask代表每次Repeat中处理的元素个数，迭代次数生效。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        TILE_LENGTH = 1024
        # 1. 定义源和目标 LocalTensor
        src_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECIN, size=TILE_LENGTH)
        dst_tensor = asc.LocalTensor(asc.fp16, asc.Position.VECOUT, size=TILE_LENGTH)
            
        ...

        # 2. 定义地址步长参数
        # 示例：实现一个交错拷贝，源地址每次迭代跳 256 字节，目标地址连续
        params = asc.CopyRepeatParams(
            dstStride=1,       # 迭代内，目标 datablock 连续
            srcStride=2,       # 迭代内，源 datablock 间隔为 1 个 datablock
            dstRepeatSize=8,   # 迭代间，目标地址步长为 8 个元素
            srcRepeatSize=16   # 迭代间，源地址步长为 16 个元素
        )
            
        # 3. 使用连续模式调用 Copy
        # 每次迭代处理 128 个元素（一个 256 字节的 block），重复 4 次
        asc.copy(dst_tensor, src_tensor, mask=128, repeat_time=4, repeat_params=params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_wait_flag_docstring():
    func_introduction = """
    同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插此同步。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetFlag(TEventID id)
        __aicore__ inline void WaitFlag(TEventID id)

    """

    param_list = """
    **参数说明**

    - id: 事件ID，由用户自己指定。
    """

    constraint_list = """
    **约束说明**

    - set_flag/wait_flag必须成对出现。
    - 禁止用户在使用set_flag和wait_flag时，自行指定event_id，容易与框架同步事件冲突，导致卡死问题。event_id需要通过alloc_event_id或者fetch_event_id来获取。
    """
    
    py_example = """
    **调用示例**

    如data_copy需要等待set_value执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

    .. code-block:: python

        dst = asc.GlobalTensor()
        src = asc.LocalTensor()
        src.set_value(0, 0)
        data_size = 512
        event_id = global_pipe.fetch_event_id(event=asc.HardEvent.S_MTE3)
        asc.set_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
        asc.wait_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
        asc.data_copy(dst, src, data_size)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def pipe_barrier_docstring():
    func_introduction = """
    阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <pipe_t pipe>
        __aicore__ inline void PipeBarrier()

    """

    param_list = """
    **参数说明**

    - pipe: 模板参数，表示阻塞的流水类别。
    """

    constraint_list = """
    **约束说明**

    Scalar流水之间的同步由硬件自动保证，调用pipe_barrier(PIPE_S)会引发硬件错误。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.add(dst0, src0, src1, 512)
        asc.pipe_barrier(asc.PipeID.PIPE_V)
        asc.mul(dst1, dst0, src2, 512)
                
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def data_cache_clean_and_invalid_docstring():
    func_introduction = """
    用来刷新Cache，保证Cache与Global Memory之间的数据一致性。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, CacheLine entireType, DcciDst dcciDst>
        __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

    .. code-block:: c++

        template <typename T, CacheLine entireType, DcciDst dcciDst>
        __aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)

    .. code-block:: c++

        template <typename T, CacheLine entireType>
        __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)

    """

    param_list = """
    **参数说明**

    - entire_type：指令操作模式，类型为CacheLine枚举值：
      - SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line（若非64B对齐，仅操作对齐范围内部分）。
      - ENTIRE_DATA_CACHE：刷新整个Data Cache（耗时较大，性能敏感场景慎用）。
    - dcci_dst：指定Data Cache与哪种存储保持一致性，类型为DcciDst枚举类：
      - CACHELINE_ALL：与CACHELINE_OUT效果一致。
      - CACHELINE_UB：预留参数，暂未支持。
      - CACHELINE_OUT：保证Data Cache与Global Memory一致。
      - CACHELINE_ATOMIC：部分Atlas产品上为预留参数，暂未支持。
    - dst：	需要刷新Cache的Tensor。
    """

    py_example = """
    **调用示例**

    - 支持通过配置dcciDst确保Data Cache与GM存储的一致性

      .. code-block:: python

          asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE,
                                          dcci_dst=asc.DcciDst.CACHELINE_OUT, dst=dst)

    - 不支持配置dcciDst，仅支持保证Data Cache与GM的一致性

      .. code-block:: python

          asc.data_cache_clean_and_invalid(entire_type=asc.CacheLine.SINGLE_CACHE_LINE, dst=dst)

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def data_copy_docstring():
    func_introduction = """
    DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。
    该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, 
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, 
                                            const DataCopyParams& repeatParams)
        
    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const DataCopyParams& repeatParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const DataCopyParams& repeatParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, 
                                        const DataCopyParams& intriParams, 
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const DataCopyParams& intriParams, 
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, 
                                        const DataCopyParams& intriParams, 
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
                                        const DataCopyParams& intriParams, 
                                        const DataCopyEnhancedParams& enhancedParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, 
                                        const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], 
                                        const uint32_t dimValue = 1)
        
    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor<T> &dst, const LocalTensor<T> &src, 
                                        const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[], 
                                        const uint32_t dimValue = 1)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor或GlobalTensor。
    - src：源操作数，类型为LocalTensor或GlobalTensor。
    - params：搬运参数，DataCopyParams类型。
    - count：参与搬运的元素个数。
    - enhanced_params：增强信息参数。
    - slice_list1/slice_list2：目的操作数/源操作数切片信息，SliceInfo类型。
    - dim_value：操作数维度信息，默认值为1。
    """

    constraint_list = """
    **约束说明**

    - 如果需要执行多个data_copy指令，且data_copy的目的地址存在重叠，需要通过调用pipe_barrier(ISASI)来插入同步指令，保证多个data_copy指令的串行化，防止出现异常数据。
    - 在跨卡通信算子开发场景，data_copy类接口支持跨卡数据搬运，仅支持HCCS物理链路，不支持其他通路；开发者开发过程中，需要关注涉及卡间通信的物理通路，可通过npu-smi info -t topo命令查询HCCS物理链路。
    """

    py_example = """
    **调用示例**

    - 基础数据搬运

      .. code-block:: python

          pipe = asc.Tpipe()
          in_queue_src = asc.TQue(asc.TPosition.VECIN, 1)
          out_queue_dst = asc.TQue(asc.TPosition.VECOUT, 1)
          src_global = asc.GlobalTensor()
          dst_global = asc.GlobalTensor()
          pipe.init_buffer(que=in_queue_src, num=1, len=512 * asc.half.sizeof())
          pipe.init_buffer(que=out_queue_dst, num=1,len=512 * asc.half.sizeof())
          src_local = in_queue_src.alloc_tensor(asc.half)
          dst_local = out_queue_dst.alloc_tensor(asc.half)
          # 使用传入count参数的搬运接口，完成连续搬运
          asc.data_copy(src_local, src_global, count=512)
          asc.data_copy(dst_local, src_local, count=512)
          asc.data_copy(dst_global, dst_local, count=512)
          # 使用传入DataCopyParams参数的搬运接口，支持连续和非连续搬运
          intri_params = asc.DataCopyParams()
          asc.data_copy(src_local, src_global, params=intri_params)
          asc.data_copy(dst_local, src_local, params=intri_params)
          asc.data_copy(dst_global, dst_local, params=intri_params)
    
    - 增强数据搬运

      .. code-block:: python

          pipe = asc.Tpipe()
          in_queue_src = asc.TQue(asc.TPosition.CO1, 1)
          out_queue_dst = asc.TQue(asc.TPosition.CO2, 1)
          ...
          src_local = in_queue_src.alloc_tensor(asc.half)
          dst_local = out_queue_dst.alloc_tensor(asc.half)
          intri_params = asc.DataCopyParams()
          enhanced_params = asc.DataCopyEnhancedParams()
          asc.data_copy(dst_local, src_local, params=intri_params, enhanced_params=enhanced_params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def data_copy_pad_docstring():
    func_introduction = """
    DataCopyPad接口提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    通路：Global Memory->Local Memory

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<T> &padParams)

    通路：Local Memory->Global Memory

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams)

    通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                            const DataCopyExtParams &dataCopyParams, const Nd2NzParams &nd2nzParams)

    通路：Global Memory->Local Memory (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams)

    通路：Local Memory->Global Memory (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams)

    通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM (DataCopyParams版本)

    .. code-block:: c++

        template<typename T>
        __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor或GlobalTensor。
      LocalTensor的起始地址需要保证32字节对齐。
      GlobalTensor的起始地址无地址对齐约束。

    - src: 源操作数，类型为LocalTensor或GlobalTensor。
      LocalTensor的起始地址需要保证32字节对齐。
      GlobalTensor的起始地址无地址对齐约束。

    - dataCopyParams: 搬运参数。
      DataCopyExtParams类型：支持更大的操作数步长等参数取值范围
      DataCopyParams类型：标准搬运参数

    - padParams: 从Global Memory搬运数据至Local Memory时，用于控制数据填充过程的参数。
      DataCopyPadExtParams<T>类型：支持泛型填充值
      DataCopyPadParams类型：仅支持uint64_t数据类型且填充值只能为0

    - nd2nzParams: 从VECIN/VECOUT->TSCM进行数据搬运时，用于控制数据格式转换的参数。
      Nd2NzParams类型，ndNum仅支持设置为1。
    """

    constraint_list = """
    **约束说明**

    - leftPadding、rightPadding的字节数均不能超过32Bytes。
    - 当数据类型长度为64位时，paddingValue只能设置为0。
    - 不同产品型号对函数原型的支持存在差异，请参考官方文档选择产品型号支持的函数原型进行开发。
    """

    py_example = """
    **调用示例**

    GM->VECIN搬运数据并填充：

    .. code-block:: python

        # 从GM->VECIN搬运，使用DataCopyParams和DataCopyPadParams
        src_local = in_queue_src.alloc_tensor(asc.half)
        copy_params = asc.DataCopyParams(1, 20 * asc.half.sizeof(), 0, 0)
        pad_params = asc.DataCopyPadParams(True, 0, 2, 0)
        asc.data_copy_pad(src_local, src_global, copy_params, pad_params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def dump_tensor_docstring_docstring():
    func_introduction = """
    基于算子工程开发的算子，可以使用该接口Dump指定Tensor的内容。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
        template <typename T>
        __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc, 
        uint32_t dumpSize, const ShapeInfo& shapeInfo)
        template <typename T>
        __aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, 
        uint32_t dumpSize, const ShapeInfo& shapeInfo)

    """

    param_list = """
    **参数说明**

    - tensor：需要dump的Tensor。
    - desc：用户自定义附加信息（行号或其他自定义数字）。
    - dump_size：需要dump的元素个数。
    - shape_info：传入Tensor的shape信息，可按照shape信息进行打印。
    """

    constraint_list = """
    **约束说明**

    - 该功能仅用于NPU上板调试，且仅在如下场景支持：
      - 通过Kernel直调方式调用算子。
      - 通过单算子API调用方式调用算子。
      - 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景。
    - 当前仅支持打印存储位置为Unified Buffer/L1 Buffer/L0C Buffer/Global Memory的Tensor信息。
    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量（包括信息头）不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
    """

    py_example = """
    **调用示例**

    - 无Tensor shape的打印

      .. code-block:: python

          asc.dump_tensor(src_local, 5, date_len)

    - 带Tensor shape的打印

      .. code-block:: python

          shape_info = asc.ShapeInfo()
          asc.dump_tensor(x, 2, 64, shape_info)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def metrics_prof_start_docstring():
    func_introduction = """
    用于设置性能数据采集信号启动，和asc.metrics_prof_stop()配合使用。
    使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc.metrics_prof_start()和asc.metrics_prof_stop()来指定需要调优的代码段范围。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void MetricsProfStart()
    
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc

        asc.metrics_prof_start()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def metrics_prof_stop_docstring():
    func_introduction = """
    设置性能数据采集信号停止，和asc.metrics_prof_start()配合使用。
    使用msProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc.metrics_prof_start()和asc.metrics_prof_stop()来指定需要调优的代码段范围。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void MetricsProfStop()
    
    """

    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        
        asc.metrics_prof_stop()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def printf_docstring():
    func_introduction = """
    该接口提供CPU域/NPU域调试场景下的格式化输出功能。
    在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        void printf(__gm__ const char* fmt, Args&&... args)
        void PRINTF(__gm__ const char* fmt, Args&&... args)

    """

    param_list = """
    **参数说明**

    - fmt：格式控制字符串，包含两种类型的对象：普通字符和转换说明。
    - args：附加参数，个数和类型可变的参数列表。
    """

    constraint_list = """
    **约束说明**

    - 本接口不支持打印除换行符之外的其他转义字符。
    - 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
    - 算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        #整型打印
        x = 10
        asc.printf("%d", x)
        #浮点型打印
        x = 3.14
        asc.printf("%f", x)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_block_num_docstring():
    func_introduction = """
    获取当前任务配置的核数，用于代码内部的多核逻辑控制等。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetBlockNum()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        loop_size = total_size // asc.get_block_num()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def get_block_idx_docstring():
    func_introduction = """
    获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetBlockIdx()

    """
    
    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    GetBlockIdx为一个系统内置函数，返回当前核的index。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src0_global.set_global_buffer(src0_gm + asc.get_block_idx() * single_core_offset)
        src1_global.set_global_buffer(src1_gm + asc.get_block_idx() * single_core_offset)
        dst_global.set_global_buffer(dst_gm + asc.get_block_idx() * single_core_offset)
        pipe.init_buffer(que=in_queue_src0, num=1, len=256*asc.float.sizeof())
        pipe.init_buffer(que=in_queue_src1, num=1, len=256*asc.float.sizeof())
        pipe.init_buffer(que=sel_mask, num=1, len=256)
        pipe.init_buffer(que=out_queue_dst, num=1, len=256*asc.float.sizeof())

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_data_block_size_in_bytes_docstring():
    func_introduction = """
    获取当前芯片版本一个datablock的大小，单位为byte。
    开发者可以根据datablock的大小来计算API指令中待传入的repeatTime、
    DataBlock Stride、Repeat Stride等参数值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
    """
    
    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    当前芯片版本一个datablock的大小，单位为byte。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        size = asc.get_data_block_size_in_bytes()

    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_icache_preload_status_docstring():
    func_introduction = """
    获取ICACHE的PreLoad的状态。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetICachePreloadStatus();
    """
    
    param_list = """
    **参数说明**

    无。
    """

    return_list = """
    **返回值说明**

    int64_t类型，0表示空闲，1表示忙。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        cache_preload_status = asc.get_icache_preload_status()
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def get_program_counter_docstring():
    func_introduction = """
    获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

         __aicore__ inline int64_t GetProgramCounter()

    """
    
    param_list = """
    **参数说明**

    无。
    """
    py_example = """
    **调用示例**

    .. code-block:: python

        pc = asc.get_program_counter()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def get_system_cycle_docstring():
    func_introduction = """
    获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline int64_t GetSystemCycle()

    """
    
    param_list = """
    **参数说明**

    无。
    """

    constraint_list = """
    **约束说明**

    该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过pipe_barrier插入对应流水的同步
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        cycle = asc.get_system_cycle()

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def icache_preload_docstring():
    func_introduction = """
    从指令所在DDR地址预加载指令到ICache中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void ICachePreLoad(const int64_t preFetchLen);
    """

    param_list = """
    **参数说明**

    - pre_fetch_len：预取长度。
    """

    return_list = """
    **返回值说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python
            
        pre_fetch_len = 2
        asc.icache_preload(pre_fetch_len)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def load_data_docstring():
    func_introduction = """
    源操作数/目的操作数的数据类型为uint8_t/int8_t时，分形矩阵大小在A1/A2上为16*32， 在B1/B2上为32*16。
    源操作数/目的操作数的数据类型为uint16_t/int16_t/half/bfloat16_t时，分形矩阵在A1/B1/A2/B2上的大小为16*16。
    源操作数/目的操作数的数据类型为uint32_t/int32_t/float时，分形矩阵大小在A1/A2上为16*8， 在B1/B2上为8*16。
    支持如下数据通路：
    GM->A1; GM->B1; GM->A2; GM->B2;
    A1->A2; B1->B2。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData2DParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const GlobalTensor<T>& src,
                                        const LoadData2DParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData2DParamsV2& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const GlobalTensor<T>& src,
                                        const LoadData2DParamsV2& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst,
                                        const LocalTensor<T>& src,
                                        const LoadData3DParamsV2Pro& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数，类型为 LocalTensor。
      - 作为二维数据加载的目标 Tensor。
      - 支持的 TPosition 为 VECIN/VECCALC/VECOUT。
      - 起始地址需要 32 字节对齐。

    - src：源操作数，类型为 LocalTensor 或 GlobalTensor。
      - 当为 LocalTensor 时，表示在芯片内部不同本地存储单元之间按 2D 方式搬运。
      - 当为 GlobalTensor 时，表示从 Global Memory 按 2D 方式加载数据到 LocalTensor。
      - 元素数据类型需与 dst 保持一致。

    - params：二维加载参数，类型为 LoadData2DParams 或 LoadData2DParamsV2 或 LoadData3DParamsV2Pro。
      - LoadData2DParams 结构体
        - startIndex：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。
        - repeatTimes：迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。
        - srcStride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
        - sid：预留参数，配置为0即可。
        - dstGap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。
        - ifTranspose：是否启用转置功能，对每个分形矩阵进行转置，默认为false:
        - addrMode：预留参数，配置为0即可。

      - LoadData2DParamsV2 结构体
        - m_start_position：M维起始位置，取值范围：m_start_position∈[0, 65535]。默认为0。
        - k_start_position：K维起始位置，取值范围：k_start_position∈[0, 65535]。默认为0。
        - m_step：M维步长，取值范围：m_step∈[0, 65535]。默认为0。
        - k_step：K维步长，取值范围：k_step∈[0, 65535]。默认为0。
        - src_stride：源操作数步长，取值范围：src_stride∈[-2147483648, 2147483647]。默认为0。
        - dst_stride：目的操作数步长，取值范围：dst_stride∈[0, 65535]。默认为0。
        - if_transpose：是否启用转置功能，默认为false。
        - sid：流ID，取值范围：sid∈[0, 255]。默认为0。

      - LoadData3DParamsV2Pro 结构体
        - channel_size：通道大小，取值范围：channel_size∈[0, 65535]。默认为0。
        - en_transpose：是否启用转置功能，默认为false。
        - en_small_k：是否启用小K优化，默认为false。
        - filter_size_w：是否启用滤波器宽度优化，默认为false。
        - filter_size_h：是否启用滤波器高度优化，默认为false。
        - f_matrix_ctrl：是否启用矩阵控制，默认为false。
        - ext_config：扩展配置，取值范围：ext_config∈[0, 18446744073709551615]。默认为0。
        - filter_config：滤波器配置，取值范围：filter_config∈[0, 18446744073709551615]。默认为0x10101010101。
      """

    constraint_list = """
    **约束说明**

    - dst 与 src 的数据需要满足起始地址对齐要求，具体可查看文档。
    - 不使用或者不想改变的配置，建议保持默认值，有助于性能提升。

    """

    py_example = """
    **调用示例**

    - Local Memory 内部 2D 搬运（Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_l2l(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

              asc.load_data(y_local, x_local, params)

    - Global Memory 到 Local Memory 的 2D 搬运（Global -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_g2l(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params = asc.LoadData2DParams(0, 4, 0, 0, 0, 0, 0)

              asc.load_data(y_local, x_local, params)
              asc.load_data(x_local, x_gm, params)

    - Local Memory 内部 2D 搬运（V2版本，Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_l2l_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

              asc.load_data(y_local, x_local, params_v2)

    - Global Memory 到 Local Memory 的 2D 搬运（V2版本，Global -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_g2l_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              x_gm = asc.GlobalTensor()
              x_gm.set_global_buffer(x)

              params_v2 = asc.LoadData2DParamsV2(0, 0, 16, 16, 0, 0, False, 0)

              asc.load_data(y_local, x_local, params_v2)
              asc.load_data(x_local, x_gm, params_v2)

    - Local Memory 内部 3D 搬运（V2Pro版本，Local -> Local）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_3d_v2pro(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)
              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params_3d_v2_pro = asc.LoadData3DParamsV2Pro(16, False, False, False, False, False, 0, 0x10101010101)

              asc.load_data(y_local, x_local, params_3d_v2_pro)
    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def load_data_with_transpose_docstring():
    func_introduction = """
    该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                                     const LocalTensor<T>& src,
                                                     const LoadData2dTransposeParams& loadDataParams)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst,
                                                     const LocalTensor<T>& src,
                                                     const LoadData2dTransposeParamsV2& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数，类型为 LocalTensor。
      - 用于接收转置后的二维数据。
      - 存储位置需属于 VECIN / VECCALC / VECOUT 中的一种。
      - 起始地址需满足 32 字节对齐要求。

    - src：源操作数，类型为 LocalTensor。
      - 作为 2D 输入块的提供者。
      - 仅支持 Local → Local（A1/B1 → A2/B2），不支持 GlobalTensor。
      - 数据类型必须与 dst 一致。

    - params：二维转置加载参数，类型为 LoadData2dTransposeParams 或 LoadData2dTransposeParamsV2。
      - LoadData2dTransposeParams 结构体
        - startIndex：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：startIndex∈[0, 65535] 。默认为0。
        - repeatTimes：迭代次数，取值范围：repeatTimes∈[0, 255]。默认为0。
        - srcStride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。这里的单位实际上是拼接后的方块矩阵的大小。取值范围：srcStride∈[0, 65535]。默认为0。
        - dstGap：相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。
        - dstFracGap：每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位为512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时有效。取值范围：dstFracGap∈[0, 65535]。默认为0。
        - addrMode：预留参数

      - LoadData2dTransposeParamsV2 结构体
        - start_index：方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：start_index∈[0, 65535] 。默认为0。
        - repeat_times：迭代次数，取值范围：repeat_times∈[0, 255]。默认为0。
        - src_stride：源操作数步长，取值范围：src_stride∈[0, 65535]。默认为0。
        - dst_gap：目的操作数间隔，取值范围：dst_gap∈[0, 65535]。默认为0。
        - dst_frac_gap：分形间隔，取值范围：dst_frac_gap∈[0, 65535]。默认为0。
        - src_frac_gap：源分形间隔，取值范围：src_frac_gap∈[0, 65535]。默认为0。
        - addr_mode：地址模式，取值范围：addr_mode∈[0, 255]。默认为0。
    """

    constraint_list = """
    **约束说明**

    - repeatTimes 为 0 时表示不执行搬运操作。
    - 开发者需要保证目的操作数转置后的分形没有重叠。
    - 操作数地址对齐要求请参见通用地址对齐约束。

    - repeat_times 为 0 时表示不执行搬运操作。
    - 开发者需要保证目的操作数转置后的分形没有重叠。
    - 操作数地址对齐要求请参见通用地址对齐约束。
    """

    py_example = """
    **调用示例**

    - 调用示例（V1版本）
    
      .. code-block:: python

          @asc.jit
          def kernel_load_data_with_transpose(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECIN,
                                        addr=0, tile_size=512)

              y_local = asc.LocalTensor(dtype=asc.float16,
                                        pos=asc.TPosition.VECOUT,
                                        addr=0, tile_size=512)

              params = asc.LoadData2dTransposeParams(0, 4, 0, 0, 0, 0)

              asc.load_data_with_transpose(y_local, x_local, params)

    - 调用示例（V2版本）

      .. code-block:: python

          @asc.jit
          def kernel_load_data_with_transpose_v2(x: asc.GlobalAddress) -> None:
              x_local = asc.LocalTensor(dtype=asc.float16,
                                      pos=asc.TPosition.VECIN,
                                      addr=0, tile_size=512)

              y_local = asc.LocalTensor(dtype=asc.float16,
                                      pos=asc.TPosition.VECOUT,
                                      addr=0, tile_size=512)

              params_v2 = asc.LoadData2dTransposeParamsV2(0, 4, 0, 0, 0, 0, 0)

              asc.load_data_with_transpose(y_local, x_local, params_v2)
    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def mmad_docstring():
    func_introduction = """
    完成矩阵乘加（C += A * B）操作。矩阵ABC分别为A2/B2/CO1中的数据。
    ABC矩阵的数据排布格式分别为ZZ，ZN，NZ。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T, typename U, typename S>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst,
                                    const LocalTensor<U>& fm,
                                    const LocalTensor<S>& filter,
                                    const MmadParams& mmadParams)

    .. code-block:: c++

        template <typename T, typename U, typename S, typename V>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst,
                                    const LocalTensor<U>& fm,
                                    const LocalTensor<S>& filter,
                                    const LocalTensor<V>& bias,
                                    const MmadParams& mmadParams)
    """

    param_list = """
    **参数说明**

    - dst：结果输出 Tensor，类型为 LocalTensor。
      - 用于存放矩阵乘累加的结果。
      - **必须位于 CO1 存储位置（TPosition.CO1）**。
      - 元素数据类型需与累加结果类型匹配。

    - fm：左矩阵（A 矩阵）输入，类型为 LocalTensor。
      - 表示矩阵乘法中的左操作数。
      - **必须位于 A2 存储位置（TPosition.A2）**。
      - 需要按照满足 Mmad 格式要求的 A2 布局存储。

    - filter：右矩阵（B 矩阵）输入，类型为 LocalTensor。
      - 表示矩阵乘法中的右操作数。
      - **必须位于 B2 存储位置（TPosition.B2）**。
      - 需要按照符合指令格式的 B2 分块布局排布。

    - bias（可选）：偏置项，类型为 LocalTensor。
      - 用于执行 `dst += fm × filter + bias` 的计算。
      - 当提供 bias 时，将使用带偏置版本的指令。

    - params：MmadParams 类型的矩阵乘参数。
      - m：左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。
      - n：右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。
      - k：左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。
      - cmatrixInitVal：配置C矩阵初始值是否为0。默认值true。
      - cmatrixSource：配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。
      - isBias：该参数废弃，新开发内容不要使用该参数。
      - fmOffset：预留参数。
      - enSsparse：预留参数。
      - enWinogradA：预留参数。
      - enWinogradB：预留参数。
      - unitFlag：预留参数。
      - kDirectionAlign：预留参数。
    """

    constraint_list = """
    **约束说明**

    - dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
    - 当M、K、N中的任意一个值为0时，该指令不会被执行。
    - 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。
    - 操作数地址对齐要求请参见通用地址对齐约束。
    """

    py_example = """
    **调用示例**

    - 基本 Mmad（无 bias）

      .. code-block:: python

          @asc.jit
          def kernel_mmad_basic():
              dst = asc.LocalTensor(dtype=asc.float16,
                                    pos=asc.TPosition.CO1,
                                    addr=0, tile_size=1024)

              fm = asc.LocalTensor(dtype=asc.float16,
                                   pos=asc.TPosition.A2,
                                   addr=0, tile_size=1024)

              filter = asc.LocalTensor(dtype=asc.float16,
                                       pos=asc.TPosition.B2,
                                       addr=0, tile_size=1024)

              params = asc.MmadParams(4, 4, 4)

              asc.mmad(dst, fm, filter, params)

    - Mmad 带 bias

      .. code-block:: python

          @asc.jit
          def kernel_mmad_bias():
              dst = asc.LocalTensor(dtype=asc.float16,
                                    pos=asc.TPosition.CO1,
                                    addr=0, tile_size=1024)

              fm = asc.LocalTensor(dtype=asc.float16,
                                   pos=asc.TPosition.A2,
                                   addr=0, tile_size=1024)

              filter = asc.LocalTensor(dtype=asc.float16,
                                       pos=asc.TPosition.B2,
                                       addr=0, tile_size=1024)

              bias = asc.LocalTensor(dtype=asc.float16,
                                     pos=asc.TPosition.VECIN,
                                     addr=0, tile_size=1024)

              params = asc.MmadParams(4, 4, 4)

              asc.mmad(dst, fm, filter, bias, params)
    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def load_image_to_local_docstring():
    func_introduction = """
    将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。
    图像预处理的相关参数通过set_aipp_functions进行配置。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
    """

    param_list = """
    **参数说明**

    - dst：输出，目的操作数，类型为LocalTensor，支持的TPosition为A1/B1。LocalTensor的起始地址需要保证32字节对齐。不同产品支持的数据类型不同：
      - Atlas A3 训练/推理系列产品：int8_t/half
      - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：int8_t/half
      - Atlas 200I/500 A2 推理产品：uint8_t/int8_t/half
    - load_data_params：输入，LoadData参数结构体，类型为LoadImageToLocalParams，包含以下子参数：
      - horiz_size：输入，从源图中加载图片的水平宽度，单位为像素，取值范围：horiz_size∈[2, 4095]。
      - vert_size：输入，从源图中加载图片的垂直高度，单位为像素，取值范围：vert_size∈[2, 4095]。
      - horiz_start_pos：输入，加载图片在源图片上的水平起始地址，单位为像素，取值范围：horiz_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP、XRGB8888、RGB888和YUV400格式时，该参数需要是偶数。
      - vert_start_pos：输入，加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vert_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
      - src_horiz_size：输入，源图像水平宽度，单位为像素，取值范围：src_horiz_size∈[2, 4095]。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
      - top_pad_size：输入，目的图像顶部填充的像素数，取值范围：top_pad_size∈[0, 32]，默认为0。进行数据填充时使用，需要先调用SetAippFunctions(ISASI)通过AippPaddingParams配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。
      - bot_pad_size：输入，目的图像底部填充的像素数，取值范围：bot_pad_size∈[0, 32]，默认为0。
      - left_pad_size：输入，目的图像左边填充的像素数，取值范围：left_pad_size∈[0, 32]，默认为0。
      - right_pad_size：输入，目的图像右边填充的像素数，取值范围：right_pad_size∈[0, 32]，默认为0。
      - sid：输入，预留参数，为后续功能保留，开发者暂时无需关注，使用默认值即可。
    """

    return_list = """
    **返回值说明**

    无
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 加载到dst的图片的大小加padding的大小必须小于等于L1的大小。
    - 对于XRGB输入格式的数据，芯片在处理的时候会默认丢弃掉第四个通道的数据，所以需要在set_aipp_functions接口里设置好通道交换的参数后输出RGB格式的数据。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A1, addr=0, tile_size=128)
        load_data_params = asc.LoadImageToLocalParams(2, 2, 0, 0, 2, 0, 0, 0, 0)
        asc.load_image_to_local(dst, load_data_params)
    """
    return [func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example]


def proposal_concat_docstring():
    func_introduction = """
    将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void ProposalConcat(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数。数据类型需要与dst保持一致。
    - repeat_time：重复迭代次数。每次迭代完成16个元素合入到16个Region Proposals里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。取值范围：repeatTime∈[0,255]。

    - mode_number：合入位置参数，取值范围：modeNumber∈[0, 5]
      - 0：合入x1
      - 1：合入y1
      - 2：合入x2
      - 3：合入y2
      - 4：合入score
      - 5：合入label
    """

    constraint_list = """
    **约束说明**

    - 用户需保证dst中存储的proposal数目大于等于实际所需数目，否则会存在tensor越界错误。
    - 用户需保证src中存储的元素大于等于实际所需数目，否则会存在tensor越界错误。
    - 操作数地址对齐要求请参见通用地址对齐约束。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.proposal_concat(dst, src, repeat_time=2, mode_number=4)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def proposal_extract_docstring():
    func_introduction = """
    与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void ProposalExtract(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。 
    - src：源操作数，数据类型需与dst一致。  
    - repeat_time：重复迭代次数。每次迭代处理16个Region Proposals的元素抽取并重排，下次迭代跳至相邻的下一组16个Region Proposals。取值范围：repeatTime∈[0,255]。  
    - mode_number：抽取位置参数，取值范围：modeNumber∈[0,5]  
      - 0：抽取x1
      - 1：抽取y1
      - 2：抽取x2
      - 3：抽取y2
      - 4：抽取score
      - 5：抽取label
    """

    constraint_list = """
    **约束说明**

    - 用户需保证src中存储的proposal数量不小于实际所需数量，否则可能发生tensor越界。  
    - 用户需保证dst中可容纳的元素数量不小于实际抽取数量。  
    - 操作数地址需满足通用对齐约束（32字节对齐）。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.proposal_extract(dst, src, repeat_time=2, mode_number=4)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def trap_docstring():
    func_introduction = """
    在Kernel侧调用，NPU模式下会中断AI Core的运行，CPU模式下等同于assert。可用于Kernel侧异常场景的调试。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void Trap()

    """
    
    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.trap()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def duplicate_docstring():
    func_introduction = """
    将一个变量或立即数复制多次并填充到向量中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)

        template <typename T, bool isSetMask = true>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)

        template <typename T, bool isSetMask = true>
        void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask, const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - scalar：被复制的源操作数，支持输入变量和立即数，数据类型需与dst中元素的数据类型保持一致。
    - count：参与计算的元素个数。
    - mask：mask用于控制每次迭代内参与计算的元素。
    - repeat_time：矢量计算单元，每次读取连续的8个datablock（每个block32Bytes，共256Bytes）数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeat_time表示迭代的次数。
    - dst_block_stride：单次迭代内，矢量目的操作数不同datablock间地址步长。
    - dst_repeat_stride：相邻迭代间，矢量目的操作数相同datablock地址步长。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          scalar = 18.0
          asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          scalar = 18.0
          asc.duplicate(dst_local, scalar, mask=mask, repeat_times=2, dst_block_stride=1, dst_repeat_stride=8)
        
    - tensor前n个数据计算样例，源操作数为标量
    
      .. code-block:: python

          scalar = 18.0
          asc.duplicate(dst_local, scalar, count=src_data_size)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def pair_reduce_sum_docstring():
    func_introduction = """
    PairReduceSum：相邻两个（奇偶）元素求和。例如，对于序列 (a1, a2, a3, a4, a5, a6, ...)，
    相邻两个数据求和为 (a1+a2, a3+a4, a5+a6, ...)。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
                                                
    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void PairReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);
    """

    param_list = """
    **参数说明**

    - dst：输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。

    - src：输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32 字节对齐。

    - repeat_time：迭代次数，取值范围 [0, 255]。关于该参数的具体描述请参考如何使用 Tensor 高维切分计算 API。

    - mask：控制每次迭代内参与计算的元素。

      **逐比特模式**
      数组形式，按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。
      - 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
      - 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
      - 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]

      **连续模式**
      整数形式，表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
      - 16 位：mask ∈ [1, 128]
      - 32 位：mask ∈ [1, 64]
      - 64 位：mask ∈ [1, 32]

    - dst_rep_stride：
      目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
      - PairReduce 完成后，一个 repeat 的长度减半。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长，详细说明请参考 dataBlockStride。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即每次迭代跳过的 data block 数目。详细说明请参考 repeatStride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 如果两两相加的两个元素mask位未配置（即当前两个元素不参与运算）。

      - 对于 Atlas 200I/500 A2 推理产品 ，对应的目的操作数中的值会置为0。
      - 对于其他产品型号，对应的目的操作数中的值不会变化。

    - 比如float场景下对64个数使用当前指令，mask配置为62，表示最后两个元素不参与运算。

      - 对于 Atlas 200I/500 A2 推理产品 ，目的操作数中最后一个值会返回0。
      - 对于其他产品型号，目的操作数中最后一个值不会变化。
    """

    py_example = """
    **调用示例**

    - mask 连续模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=128,
                              dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.pair_reduce_sum(z_local, x_local, repeat_time=2, mask=mask,
                              dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def repeat_reduce_sum_docstring():
    func_introduction = """
    对每个 repeat 内的所有数据进行求和。
    与 WholeReduceSum 接口相比，不支持 mask 逐比特模式。
    建议使用功能更全面的 WholeReduceSum 接口。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                                const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride);
    """

    param_list = """
    **参数说明**

    - dst：
      输出操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 数据类型需与 dst 保持一致。

    - repeat_time：
      重复迭代次数，取值范围 [0, 255]。
      - 矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代才能完成所有数据的读取与计算。repeatTime表示迭代的次数。
      - 具体描述请参考 Tensor 高维切分计算 API。

    - mask：
      控制每次迭代内连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
      - 操作数为 16 位：mask ∈ [1, 128]
      - 操作数为 32 位：mask ∈ [1, 64]

    - dst_blk_stride：此参数无效，可配置任意值。

    - src_blk_stride：单次迭代内数据 datablock 的地址步长。详细说明请参考 dataBlockStride。

    - dst_rep_stride：
      目的操作数相邻迭代间的地址步长，以一个 repeat 归约后的长度为单位。
      - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考 repeatStride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - 对于RepeatReduceSum，其内部的相加方式采用二叉树方式，两两相加
      假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。
      1. data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
      2. data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
      3. 以此类推，得到目的操作数为1个half类型的数据[data]。
      4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.repeat_reduce_sum(z_local, x_local, repeat_time=4, mask=128,
                              dst_blk_stride=0, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_max_docstring():
    func_introduction = """
    每个repeat内所有数据求最大值以及其索引index，返回的索引值为每个repeat内部索引。  
    归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
         __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime,
                                                const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride, 
                                                ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
                                                  

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime,
                                                const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                                ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);                                                
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需 4 字节对齐（half 数据类型）或 8 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]

          - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
          - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长。
      - 以一个 repeat 归约后的长度为单位。
      - 返回索引和最值时，单位为 dst 数据类型所占字节长度的两倍，比如当dst为half时，单位为4Bytes。
      - 仅返回最值时，单位为 dst 数据类型所占字节长度。
      - 仅返回索引时，单位为 uint32_t 类型所占字节长度。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长。详细说明请参考 dataBlockStride。

    - src_rep_stride：源操作数相邻迭代间地址步长，即每次迭代跳过的 datablock 数目。详细说明请参考 repeatStride。

    - order
      可选参数，指定 dst 中 index 与 value 的相对位置以及返回结果行为，类型为 ReduceOrder。
      - 默认值为 asc.ReduceOrder.ORDER_VALUE_INDEX。
      - asc.ReduceOrder.ORDER_VALUE_INDEX：value 位于低半部，返回顺序 [value, index]。
      - asc.ReduceOrder.ORDER_INDEX_VALUE：index 位于低半部，返回顺序 [index, value]。
      - asc.ReduceOrder.ORDER_ONLY_VALUE：只返回最值，顺序 [value]。
      - asc.ReduceOrder.ORDER_ONLY_INDEX：只返回索引，顺序 [index]。
      - 910B，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
      - 910C，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如完整样例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。
    - 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。
    """

    py_example = """
    **调用示例**

    - mask 连续模式，默认 ORDER_VALUE_INDEX

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 连续模式，ORDER_INDEX_VALUE

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_max(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式，ORDER_INDEX_VALUE

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_max(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_min_docstring():
    func_introduction = """
    每个repeat内所有数据求最小值以及其索引index，返回的索引值为每个repeat内部索引。  
    归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐bit模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint64_t mask[], const int32_t repeatTime,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMin(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t mask, const int32_t repeatTime,
                                            const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
                                            ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需 4 字节对齐（half 数据类型）或 8 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

        - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
        - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
        - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
        - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

      - **连续模式**

        mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数 16 位：mask ∈ [1, 128]
        - 操作数 32 位：mask ∈ [1, 64]
        - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长。
      - 以一个 repeat 归约后的长度为单位。
      - 返回索引和最值时，单位为 dst 数据类型所占字节长度的两倍，比如当dst为half时，单位为4Bytes。
      - 仅返回最值时，单位为 dst 数据类型所占字节长度。
      - 仅返回索引时，单位为 uint32_t 类型所占字节长度。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内数据 block 的地址步长。详细说明请参考 dataBlockStride。

    - src_rep_stride：源操作数相邻迭代间地址步长，即每次迭代跳过的 datablock 数目。详细说明请参考 repeatStride。

    - order
      可选参数，指定 dst 中 index 与 value 的相对位置以及返回结果行为，类型为 ReduceOrder。
      - 默认值为 asc.ReduceOrder.ORDER_VALUE_INDEX。
      - asc.ReduceOrder.ORDER_VALUE_INDEX：value 位于低半部，返回顺序 [value, index]。
      - asc.ReduceOrder.ORDER_INDEX_VALUE：index 位于低半部，返回顺序 [index, value]。
      - asc.ReduceOrder.ORDER_ONLY_VALUE：只返回最值，顺序 [value]。
      - asc.ReduceOrder.ORDER_ONLY_INDEX：只返回索引，顺序 [index]。
      - 910B，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
      - 910C，支持ORDER_VALUE_INDEX、ORDER_INDEX_VALUE、ORDER_ONLY_VALUE、ORDER_ONLY_INDEX。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - dst结果存储顺序由order决定，默认为最值、最值索引。返回结果中索引index数据按照dst的数据类型进行存储，比如dst使用half类型时，index按照half类型进行存储，读取时需要使用reinterpret_cast方法转换到整数类型。若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。比如完整样例中，前两个计算结果为[9.980e-01 5.364e-06]，5.364e-06需要使用reinterpret_cast方法转换得到索引值90。
    - 针对不同场景合理使用归约指令可以带来性能提升，相关介绍请参考选择低延迟指令，优化归约操作性能，具体样例请参考ReduceCustom。
    """

    py_example = """
    **调用示例**

    - mask 连续模式，默认 ORDER_VALUE_INDEX

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 连续模式，ORDER_INDEX_VALUE

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_min(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式，ORDER_INDEX_VALUE

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_min(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8,
                               order=asc.ReduceOrder.ORDER_INDEX_VALUE)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def whole_reduce_sum_docstring():
    func_introduction = """
    每个迭代内所有数据求和。归约指令的总体介绍请参考如何使用归约指令。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // mask 逐比特模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const uint64_t mask[],
                                              const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);  

    .. code-block:: c++

        // mask 连续模式
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask,
                                              const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride);                  
    """

    param_list = """
    **参数说明**

    - dst：
      输出，目的操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 的起始地址需保证 2 字节对齐（half 数据类型）或 4 字节对齐（float 数据类型）。
      - 数据类型根据产品支持情况：half / float。

    - src：
      输入，源操作数，类型为 LocalTensor，支持 TPosition 为 VECIN/VECCALC/VECOUT。
      - LocalTensor 起始地址需 32 字节对齐。
      - 源操作数的数据类型需要与目的操作数保持一致。

    - mask：
      控制每次迭代内参与计算的元素。

      - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。

          - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
          - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
          - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
          - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算

        - **连续模式**

          mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
          - 操作数 16 位：mask ∈ [1, 128]
          - 操作数 32 位：mask ∈ [1, 64]
          - 操作数 64 位：mask ∈ [1, 32]

    - repeat_time：迭代次数，取值范围 [0, 255]。具体描述请参考 如何使用Tensor 高维切分计算API。

    - dst_rep_stride：
      目的操作数相邻迭代间地址步长，以一个 repeat 归约后的长度为单位。
      - 单位为 dst 数据类型所占字节长度。比如当dst为half时，单位为2Bytes。
      - 注意：Atlas 训练系列产品不支持配置 0。

    - src_blk_stride：单次迭代内datablock的地址步长。详细说明请参考dataBlockStride。

    - src_rep_stride：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的DataBlock数目。详细说明请参考repeatStride。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - 对于WholeReduceSum，其内部的相加方式采用二叉树方式，两两相加
      假设源操作数为128个half类型的数据[data0,data1,data2...data127]，一个repeat可以计算完，计算过程如下。
      1. data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
      2. data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
      3. 以此类推，得到目的操作数为1个half类型的数据[data]。
      4. 需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。
    """

    py_example = """
    **调用示例**

    - mask 连续模式

      .. code-block:: python

          x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
          z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
          asc.whole_reduce_sum(z_local, x_local, mask=128, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)

    - mask 逐bit模式

      .. code-block:: python

          uint64_max = 2**64 - 1
          mask = [uint64_max, uint64_max]
          asc.whole_reduce_sum(z_local, x_local, mask=mask, repeat_time=4,
                               dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def scatter_docstring():
    func_introduction = """
    给定一个连续的输入张量和一个目的地址偏移张量，scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。
    将源操作数src中的元素按照指定的位置（由dst_offset和dst_base共同作用）分散到目的操作数dst中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                      const LocalTensor<uint32_t>& dstOffset,
                                      const uint32_t dstBaseAddr, const uint32_t count)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const LocalTensor<uint32_t>& dstOffset,
                                        const uint32_t dstBaseAddr, const uint64_t mask[],
                                        const uint8_t repeatTime, const uint8_t srcRepStride)

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const LocalTensor<uint32_t>& dstOffset,
                                        const uint32_t dstBaseAddr, const uint64_t mask,
                                        const uint8_t repeatTime, const uint8_t srcRepStride)
    """

    param_list = """
    **参数说明**

    - dst：目的操作数。
    - src：源操作数，数据类型需与dst保持一致。
    - dst_offset：用于存储源操作数的每个元素在dst中对应的地址偏移,以字节为单位。
      偏移基于dst的基地址dst_base计算，以字节为单位，取值应保证按dst数据类型位宽对齐。
    - dst_base：dst的起始偏移地址，单位是字节。取值应保证按dst数据类型位宽对齐。
    - count：执行处理的数据个数。
    - mask：控制每次迭代内参与计算的元素，支持连续模式或逐bit模式。
    - repeat_times：指令迭代次数，每次迭代完成8个datablock的数据收集。
    - src_rep_stride：相邻迭代间的地址步长，单位是datablock。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          asc.scatter(dst, src, dst_offset, dst_base=0, mask=128, repeat_times=1, src_rep_stride=8)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask_bits = [uint64_max, uint64_max]
          asc.scatter(dst, src, dst_offset, dst_base=0, mask=mask_bits, repeat_times=1, src_rep_stride=8)

    - tensor前n个数据计算样例，源操作数为标量
    
      .. code-block:: python

          asc.scatter(dst, src, dst_offset, dst_base=0, count=128)

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_deq_scale_docstring():
    func_introduction = """
    设置DEQSCALE寄存器的值。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline void SetDeqScale(half scale)

        __aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)

    """

    param_list = """
    **参数说明**

    - scale(half)：scale量化参数，half类型。
    - scale(float)：scale量化参数，float类型。
    - offset：offset量化参数，int16_t类型，只有前9位有效。
    - sign_mode：bool类型，表示量化结果是否带符号。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        # Cast
        scale = 1.0
        asc.set_deq_scale(scale)
        asc.cast(cast_dst_local, cast_dsrc_local, asc.RoundMode.CAST_NONE, src_size)
        # CastDeq
        scale = 1.0
        offset = 0
        sign_mode = True
        asc.set_deq_scale(scale, offset, sign_mode)
        asc.cast_deq(dst_local, src_local, count=src_size, is_vec_deq=False, half_block=False)
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_vector_mask_docstring():
    func_introduction = """
    用于在矢量计算时设置mask。使用前需要先调用 SetMaskCount/SetMaskNorm 设置 mask 模式。
    在不同模式下，mask的含义不同：

    - **Normal 模式**

      mask参数用来控制单次迭代内参与计算的元素个数。此时又可以划分为如下两种模式：

      - **连续模式（len）**：表示单次迭代中前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数为16位时：mask ∈ [1, 128]
        - 操作数为32位时：mask ∈ [1, 64]
        - 操作数为64位时：mask ∈ [1, 32]

      - **逐比特模式（mask_high / mask_low）**：按位控制参与计算的元素，bit位的值为1表示参与计算，0表示不参与。

        分为mask_high（高位mask）和mask_low（低位mask）。参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。

        - 操作数为16位时：mask_low、mask_high ∈ [0, 2⁶⁴-1]，并且不同时为 0
        - 操作数为32位时：mask_high = 0，mask_low ∈ (0, 2⁶⁴-1]
        - 操作数为64位时：mask_high = 0，mask_low ∈ (0, 2³²-1]

    - **Counter 模式**

      mask参数表示整个矢量计算参与计算的元素个数。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, MaskMode mode = MaskMode::NORMAL>
        __aicore__ static inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow);

    .. code-block:: c++

        template <typename T, MaskMode mode = MaskMode::NORMAL>
        __aicore__ static inline void SetVectorMask(int32_t len);
    """

    param_list = """
    **参数说明**

    - mask_high

      - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入高位mask值。 
      - Counter模式：需要置0，本入参不生效。  

    - mask_low

      - Normal模式：对应Normal模式下的逐比特模式，可以按位控制哪些元素参与计算。传入低位mask值。  
      - Counter模式：整个矢量计算过程中，参与计算的元素个数。 

    - len

      - Normal模式：对应Normal模式下的mask连续模式，表示单次迭代内表示前面连续的多少个元素参与计算。 
      - Counter模式：整个矢量计算过程中，参与计算的元素个数。  

    - dtype：矢量计算操作数的数据类型，由 Python 前端显式指定，用于推导 C++ 模板参数 T。  

    - mode：
      mask 模式，类型为 MaskMode 枚举值  
      - asc.MaskMode.NORMAL：Normal 模式，支持连续模式与逐比特模式。  
      - asc.MaskMode.COUNTER：Counter 模式，mask 参数表示整个矢量计算参与的总元素个数。  

    """

    constraint_list = """
    **约束说明**

    该接口仅在矢量计算API的isSetMask模板参数为false时生效，使用完成后需要使用ResetMask将mask恢复为默认值。
    """

    py_example = """
    **调用示例**

    - Counter 模式：整个计算中参与 128 个元素

      .. code-block:: python

          len = 128
          asc.set_mask_count()
          asc.set_vector_mask(len, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
          asc.reset_mask()

    - Normal 模式（逐bit模式）：使用 bitmask 控制参与计算的元素

      .. code-block:: python

          mask_high = 2**64 - 1
          mask_low = 2**64 - 1
          asc.set_mask_norm()
          asc.set_vector_mask(mask_high, mask_low, dtype=asc.float16, mode=asc.MaskMode.NORMAL)
          asc.reset_mask()

    - Normal 模式（连续模式）：前 64 个元素参与每次迭代计算

      .. code-block:: python

          len = 64
          asc.set_mask_norm()
          asc.set_vector_mask(len, dtype=asc.float32, mode=asc.MaskMode.NORMAL)
          asc.reset_mask()
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def get_sys_workspace_docstring():
    func_introduction = """
    获取系统workspace指针。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        __aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()

    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        workspace = asc.get_sys_workspace()

    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def transpose_docstring():
    func_introduction = """
    用于实现16 * 16的二维矩阵数据块转置或者[N,C,H,W]与[N,H,W,C]数据格式互相转换。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // 普通转置，支持16 * 16的二维矩阵数据块进行转置
        template <typename T>
        __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T>& src)
            
        // 增强转置，支持16 * 16的二维矩阵数据块转置，支持[N,C,H,W]与[N,H,W,C]互相转换
        template <typename T>
        __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T> &src, 
                                        const LocalTensor<uint8_t> &sharedTmpBuffer, 
                                        const TransposeParamsExt &transposeParams)
    """

    param_list = """
    **参数说明**

    - dst: 目的操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐
    - src: 源操作数，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，起始地址需要32字节对齐，数据类型需要与dst保持一致
    - shared_tmp_buffer: 共享的临时Buffer，大小根据transposeType确定
    - params: 控制Transpose的数据结构，包含输入的shape信息和transposeType参数
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 普通转置接口支持src和dst复用。
    - 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。
    """

    py_example = """
    **调用示例**

    - 基础转置样例

      .. code-block:: python

          pipe = asc.TPipe()
          in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
          out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
          ...
          x_local = in_queue_x.alloc_tensor(asc.float16)
          z_local = out_queue_z.alloc_tensor(asc.float16)
          asc.transpose(z_local, x_local)

    - 增强转置样例

      .. code-block:: python

        pipe = asc.TPipe()
        in_queue_x = asc.TQue(asc.TPosition.VECIN, buffer_num)
        out_queue_z = asc.TQue(asc.TPosition.VECOUT, buffer_num)
        in_queue_tmp = asc.TQue(asc.TPosition.VECIN, buffer_num)
        ...
        x_local = in_queue_x.alloc_tensor(asc.float16)
        z_local = out_queue_z.alloc_tensor(asc.float16)
        tmp_buffer = in_queue_tmp.alloc_tensor(asc.uint8)
        
        params = asc.TransposeParamsExt(
            n_size=1, 
            c_size=16, 
            h_size=4, 
            w_size=4,
            transpose_type=asc.TransposeType.TRANSPOSE_NCHW2NHWC
        )
        
        asc.transpose(z_local, x_local, tmp_buffer, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def trans_data_to_5hd_docstring():
    func_introduction = """
    数据格式转换，一般用于将NCHW格式转换成NC1HWC0格式，也可用于二维矩阵数据块的转置。
    相比于Transpose接口，本接口单次repeat内可处理512Byte的数据（16个datablock），
    支持不同shape的矩阵转置，还可以支持多次repeat操作。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        // 使用LocalTensor数组版本
        template <typename T>
        __aicore__ inline void TransDataTo5HD(const LocalTensor<T> (&dstList)[NCHW_CONV_ADDR_LIST_SIZE], 
                                            const LocalTensor<T> (&srcList)[NCHW_CONV_ADDR_LIST_SIZE], 
                                            const TransDataTo5HDParams& nchwconvParams)
            
        // 使用地址值数组版本（性能更优）
        template<typename T>
        __aicore__ inline void TransDataTo5HD(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE], 
                                            uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], 
                                            const TransDataTo5HDParams& nchwconvParams)
            
        // 使用连续存储地址值版本
        template <typename T>
        __aicore__ inline void TransDataTo5HD(const LocalTensor<uint64_t>& dst, 
                                            const LocalTensor<uint64_t>& src, 
                                            const TransDataTo5HDParams& nchwconvParams)
    """

    param_list = """
    **参数说明**

    - dst_or_list：目的操作数地址序列，类型为LocalTensor数组、地址值数组或连续存储地址值的LocalTensor
    - src_or_list：源操作数地址序列，类型与dst_or_list对应，数据类型需要与目的操作数保持一致
    - params：控制参数结构体，包含读取写入位置控制、迭代次数、地址步长等参数
      - dst_high_half：指定数据存储到datablock的高半部还是低半部（仅支持int8_t/uint8_t）
      - src_high_half：指定数据从datablock的高半部还是低半部读取（仅支持int8_t/uint8_t）
      - repeat_times：重复迭代次数，取值范围[0,255]
      - dst_rep_stride：相邻迭代间目的操作数相同datablock地址步长
      - src_rep_stride：相邻迭代间源操作数相同datablock地址步长
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 普通转置接口支持src和dst复用。
    - 增强转置接口，transposeType为TRANSPOSE_ND2ND_B16时支持src和dst复用，transposeType为TRANSPOSE_NCHW2NHWC、TRANSPOSE_NHWC2NCHW时不支持src和dst复用。
    """

    py_example = """
    **调用示例**

    此接口通过不同的方式构造源和目的操作数序列，以实现灵活的数据重组。

    - `dst_list`, `src_list`：定义了源数据块和目标数据块。它们可以是包含 `LocalTensor` 物理地址的 `list`/`tuple`，
      也可以是包含 `LocalTensor` 视图对象的 `list`/`tuple`，或者是将地址值连续存储的 `LocalTensor<uint64_t>`。

      .. code-block:: python

          params = asc.TransDataTo5HDParams(
              dst_high_half=False,
              src_high_half=False,
              repeat_times=4,
              dst_rep_stride=8,
              src_rep_stride=8
          )

          asc.trans_data_to_5hd(dst_list, src_list, params)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


NAME_TRANS = {
    "Add": "add",
    "AddDeqRelu": "add_deq_relu",
    "AddRelu": "add_relu",
    "AddReluCast": "add_relu_cast",
    "And": "bitwise_and",
    "Or": "bitwise_or",
    "Div": "div",
    "FusedMulAdd": "fused_mul_add",
    "FusedMulAddRelu": "fused_mul_add_relu",
    "Max": "max",
    "Min": "min",
    "Mul": "mul",
    "MulAddDst": "mul_add_dst",
    "MulCast": "mul_cast",
    "Sub": "sub",
    "SubRelu": "sub_relu",
    "SubReluCast": "sub_relu_cast",
    "Adds": "adds",
    "LeakyRelu": "leaky_relu",
    "Maxs": "maxs",
    "Mins": "mins",
    "Muls": "muls",
    "ShiftLeft": "shift_left",
    "ShiftRight": "shift_right",
    "Abs": "abs",
    "Exp": "exp",
    "Ln": "ln",
    "Not": "bitwise_not",
    "Reciprocal": "reciprocal",
    "Relu": "relu",
    "Rsqrt": "rsqrt",
    "Sqrt": "sqrt",
}


def set_aipp_functions_docstring():
    func_introduction = """
    设置图片预处理（AIPP，AI core pre-process）相关参数。和LoadImageToLocal(ISASI)接口配合使用。
    设置后，调用LoadImageToLocal(ISASI)接口可在搬运过程中完成图像预处理操作。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    输入图片格式为YUV400、RGB888、XRGB8888:

    .. code-block:: c++

        template<typename T, typename U>
        void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)
    
    输入图片格式为YUV420 Semi-Planar:

    .. code-block:: c++

        template<typename T, typename U>
        void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1, AippInputFormat format, AippParams<U> config)
    """

    param_list = """
    **参数说明**
    
    - src0：源图片在Global Memory上的矩阵
    - src1：源图片格式为YUV420SP时，表示UV维度在Global Memory上的矩阵
    - input_format：源图片的图片格式
    - config：图片预处理的相关参数，类型为AippParams
    """

    constraint_list = """
    **约束说明**

    - src0、src1在Global Memory上的地址对齐要求如下：
      - YUV420SP：src0必须2Bytes对齐，src1必须2Bytes对齐
      - XRGB8888：src0必须4Bytes对齐
      - RGB888：src0无对齐要求
      - YUV400：src0无对齐要求
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        swap_settings = asc.AippSwapParams(is_swap_rb=True)
        cpad_settings = asc.AippChannelPaddingParams(c_padding_mode=0, c_padding_value=-1)

        aipp_config_int8 = asc.AippParams(
            dtype=asc.int8,
            swap_params=swap_settings,
            c_padding_params=cpad_settings
        )

        asc.set_aipp_functions(rgb_gm, asc.AippInputFormat.RGB888_U8, aipp_config_int8)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_binary_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, const int32_t& count);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
         __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTimes,
                                            const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                            const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTimes,
                                            const BinaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0, src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count：参与计算的元素个数。
    - mask：用于控制每次迭代内参与计算的元素。
    - repeat_times：重复迭代次数。
    - params：控制操作数地址步长的参数。
    """

    set_mask_param = ""
    if cpp_name != 'MulCast':
        set_mask_param = """
    - is_set_mask: 是否在接口内部设置mask。
        """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src0, src1, count=512)

    """

    if api_name == 'add_deq_relu':
        py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, mask=mask, repeat_times=4, params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          scale = 0.1
          asc.set_deq_scale(scale)
          asc.{api_name}(dst, src0, src1, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - 使用整个tensor参与计算接口符号重载时，运算量为目的LocalTensor的总长度。
    """
    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {set_mask_param}
    {constraint_list}
    {py_example}
    """


    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def set_binary_scalar_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
                                            const T& scalarValue, const int32_t& calCount)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
                                            const T& scalarValue, uint64_t mask[], const uint8_t repeatTimes, 
                                            const UnaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
                                            const T& scalarValue, uint64_t mask, const uint8_t repeatTimes, 
                                            const UnaryRepeatParams& repeatParams)

    """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask模式和mask值。
    - dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - scalar：源操作数，数据类型需要与目的操作数中的元素类型保持一致。
    - count：参与计算的元素个数。
    - mask：用于控制每次迭代内参与计算的元素。
    - repeat_times：重复迭代次数。
    - params：元素操作控制结构信息。
    """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          scalar = 2
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, scalar, mask=mask, repeat_times=4, params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          scalar = 2
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, scalar, mask=mask, repeat_times=4, params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src, scalar, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    """

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def set_unary_docstring(cpp_name: Optional[str] = None, append_text: str = "") -> Callable[[T], T]:
    func_introduction = f"""
    {append_text}
    """

    cpp_signature = f"""
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            const int32_t& calCount)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            uint64_t mask[], const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void {cpp_name}(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                            uint64_t mask, const uint8_t repeatTimes,
                                            const UnaryRepeatParams& repeatParams)

        """

    param_list = """
    **参数说明**

    - is_set_mask：是否在接口内部设置mask。
    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - params: 控制操作数地址步长的参数。
    """
    api_name = NAME_TRANS[cpp_name]
    py_example = f"""
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 256 // asc.half.sizeof()
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, mask=mask, repeat_times=4, params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 4，一次迭代计算128个数，共计算512个数
          # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
          # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
          params = asc.UnaryRepeatParams(1, 1, 8, 8)
          asc.{api_name}(dst, src, mask=mask, repeat_times=4, params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.{api_name}(dst, src, count=512)

    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    """

    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator


def gather_mask_docstring():
    func_introduction = """
    以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
        __aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                          const LocalTensor<U>& src1Pattern, const bool reduceMode,
                                          const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                          uint64_t& rsvdCnt)

    .. code-block:: c++

        template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
        __aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                          const uint8_t src1Pattern, const bool reduceMode,
                                          const uint32_t mask, const GatherMaskParams& gatherMaskParams,
                                          uint64_t& rsvdCnt)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型需要与目的操作数保持一致。
    - src1Pattern: gather mask（数据收集的掩码），分为内置固定模式和用户自定义模式两种：

      - 内置固定模式：src1Pattern数据类型为uint8_t，取值范围为[1,7]，所有repeat迭代使用相同的gather mask。不支持配置src1RepeatStride。

        1：01010101…0101 # 每个repeat取偶数索引元素
        2：10101010…1010 # 每个repeat取奇数索引元素
        3：00010001…0001 # 每个repeat内每四个元素取第一个元素
        4：00100010…0010 # 每个repeat内每四个元素取第二个元素
        5：01000100…0100 # 每个repeat内每四个元素取第三个元素
        6：10001000…1000 # 每个repeat内每四个元素取第四个元素
        7：11111111...1111 # 每个repeat内取全部元素

      - 用户自定义模式：src1Pattern数据类型为LocalTensor，迭代间间隔由src1RepeatStride决定，迭代内src1Pattern连续消耗。

    - reduceMode: 用于选择mask参数模式，数据类型为bool，支持如下取值：

      - false：Normal模式。该模式下，每次repeat操作256Bytes数据，总的数据计算量为repeatTimes * 256Bytes。mask参数无效，建议设置为0。按需配置repeatTimes、src0BlockStride、src0RepeatStride参数。支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。
      - true：Counter模式。根据mask等参数含义的不同，该模式有以下两种配置方式：

        配置方式一：每次repeat操作mask个元素，总的数据计算量为repeatTimes * mask个元素。mask值配置为每一次repeat计算的元素个数。按需配置repeatTimes、src0BlockStride、src0RepeatStride参数。支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。
        配置方式二：总的数据计算量为mask个元素。mask配置为总的数据计算量。repeatTimes值不生效，指令的迭代次数由源操作数和mask共同决定。按需配置src0BlockStride、src0RepeatStride参数。支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。

    - mask: 用于控制每次迭代内参与计算的元素。根据reduceMode，分为两种模式：
      - Normal模式：mask无效，建议设置为0。
      - Counter模式：取值范围[1, 232 – 1]。不同的版本型号Counter模式下，mask参数表示含义不同。具体配置规则参考上文reduceMode参数描述。
    - gatherMaskParams: 控制操作数地址步长的数据结构，GatherMaskParams类型。具体参数包括：
      - src0BlockStride: 用于设置src0同一迭代不同DataBlock间的地址步长。
      - repeatTimes: 迭代次数。
      - src0RepeatStride: 用于设置src0相邻迭代间的地址步长。
      - src1RepeatStride: 用于设置src1相邻迭代间的地址步长。
    - mode: 模板参数，用于指定GatherMask的模式，当前仅支持默认模式GatherMaskMode.DEFAULT，为后续功能做预留。
    - rsvdCnt: 该条指令筛选后保留下来的元素计数，对应dstLocal中有效元素个数，数据类型为uint64_t。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - 若调用该接口前为Counter模式，在调用该接口后需要显式设置回Counter模式（接口内部执行结束后会设置为Normal模式）。
    """

    py_example = """
    **调用示例**
    
    .. code-block:: python

        src0_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        dst_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        pattern_value = 2
        reduce_mode = False
        gather_mask_mode = asc.GatherMaskMode.DEFAULT
        mask = 0
        params = asc.GatherMaskParams(src0_block_stride=1, repeat_times=1, src0_repeat_stride=0, src1_repeat_stride=0)
        rsvd_cnt = 0
        asc.gather_mask(dst_local, src0_local, pattern_value, reduce_mode, mask, params, rsvd_cnt, gather_mask_mode)

    """
    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def scalar_cast_docstring():
    func_introduction = """
    对标量的数据类型进行转换。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U, RoundMode roundMode>
        __aicore__ inline U ScalarCast(T valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：被转换数据类型的标量。
    - dtypeL：
      目标数据类型，由Python前端指定。
      - 支持：asc.half、asc.float16、asc.int32。
    - round_mode：
      精度转换处理模式，类型为RoundMode枚举值。
      - asc.RoundMode.CAST_NONE：在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整。
      - asc.RoundMode.CAST_RINT：rint，四舍六入五成双取整。
      - asc.RoundMode.CAST_FLOOR：floor，向负无穷取整。
      - asc.RoundMode.CAST_CEIL：ceil，向正无穷取整。
      - asc.RoundMode.CAST_ROUND：round，四舍五入取整。
      - asc.RoundMode.CAST_ODD：Von Neumann rounding，最近邻奇数舍入。
      
      - 对应支持关系

        - float -> half(f322f16)：asc.RoundMode.CAST_ODD
        - float -> int32(f322s32)：asc.RoundMode.CAST_ROUND、asc.RoundMode.CAST_CEIL、asc.RoundMode.CAST_FLOOR、asc.RoundMode.CAST_RINT

      - ScalarCast的精度转换规则与Cast保持一致
    """

    return_list = """
    **返回值说明**

    返回值为转换后的标量，类型与dtype一致。

    """

    py_example = """
    **调用示例**

    .. code-block:: python

        value_in = 2.5
        dtype = asc.int32
        round_mode = asc.RoundMode.CAST_ROUND
        value_out = asc.scalar_cast(value_in, dtype, round_mode)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def scalar_get_sff_value_docstring():
    func_introduction = """
    获取一个 uint64_t 类型数字的二进制表示中，从最低有效位（LSB）开始第一个 0 或 1 出现的位置。
    如果未找到指定值，则返回 -1。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <int countValue>
        __aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn);
    """

    param_list = """
    **参数说明**

    - value_in：
      输入数据，类型为 uint64_t。
      - 表示待查找的无符号整数。

    - count_value：
      指定要查找的值，类型为 int。
      - 取值为 0 或 1。
      - 0 表示查找从最低有效位开始的第一个 0 出现的位置；
      - 1 表示查找从最低有效位开始的第一个 1 出现的位置。
    """

    return_list = """
    **返回值说明**

    - 返回 int64 类型的数：
      表示 value_in 的二进制表示中，第一个匹配值（0 或 1）出现的位置。
      - 如果未找到，则返回 -1。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        value_in = 28
        count_value = 1
        one_count = asc.scalar_get_sff_value(value_in, count_value)
    """

    return [func_introduction, cpp_signature, param_list, return_list, "", py_example]


def set_atomic_add_docstring():
    func_introduction = """
    调用该接口后，可对后续的从VECOUT/L0C/L1到GM的数据传输开启原子累加，
    通过dtype参数设定不同类型的数据。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicAdd();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子加操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 累加操作完成后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    - 该指令执行前不会对GM的数据做清零操作，开发者根据实际的算子逻辑判断是否需要清零，如果需要自行进行清零操作。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float32
        asc.set_atomic_add(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_max_docstring():
    func_introduction = """
    原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较：
    将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。
    可通过设置模板参数来设定不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicMax();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子 max 操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 使用完后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    - 对于910B，目前无法对bfloat16_t类型设置inf/nan模式。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.int32
        asc.set_atomic_max(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_min_docstring():
    func_introduction = """
    原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，
    将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。
    可通过设置模板参数来设定不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicMin();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子 min 操作的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    使用完后，建议通过set_atomic_none关闭原子累加，以免影响后续相关指令功能。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float16
        asc.set_atomic_min(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_atomic_none_docstring():
    func_introduction = """
    清空原子操作的状态。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetAtomicNone();
    """

    param_list = """
    **参数说明**

    无。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        asc.set_atomic_none()
    """

    return [func_introduction, cpp_signature, param_list, "", "", py_example]


def set_atomic_type_docstring():
    func_introduction = """
    通过设置模板参数来设定原子操作不同的数据类型。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetAtomicType();
    """

    param_list = """
    **参数说明**

    - dtype：
      原子操作使用的数据类型，由 Python 前端指定。
      - 支持类型：asc.float16、asc.float32、asc.int32、asc.half。
    """

    constraint_list = """
    **约束说明**

    - 需要和set_atomic_add、set_atomic_max、set_atomic_min配合使用。
    - 使用完成后，建议清空原子操作的状态（详见set_atomic_none），以免影响后续相关指令功能。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dtype = asc.float16
        asc.set_atomic_type(dtype)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_boundary_docstring():
    func_introduction = """
    设置 Load3D 时 A1/B1 边界值。
    如果 Load3D 指令在处理源操作数时，源操作数在 A1/B1 上的地址超出设置的边界，则会从 A1/B1 起始地址开始读取数据。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
    """

    param_list = """
    **参数说明**

    - boundaryValue 
        边界值。
        Load3Dv1 指令：单位是 32 字节。
        Load3Dv2 指令：单位是字节。
    """

    constraint_list = """
    **约束说明**

    - 用于 Load3Dv1 时， boundaryValue 的最小值是 16 （单位： 32 字节）；用于 Load3Dv2 时， boundaryValue 的最小值是 1024 （单位：字节）。
    - 如果使用 SetLoadDataBoundary 接口设置了边界值，配合 Load3D 指令使用时， Load3D 指令的 A1/B1 初始地址要在设置的边界内。
    - 如果 boundaryValue 设置为 0 ，则表示无边界，可使用整个 A1/B1 。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        asc.set_load_data_boundary(1024)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_padding_value_docstring():
    func_introduction = """
    用于调用 Load3Dv1接口/Load3Dv2 接口时设置 Pad 填充的数值。 Load3Dv1/Load3Dv2 的模板参数 isSetPadding 设置为 true 时，用户需要通过本接口设置 Pad 填充的数值，设置为 false 时，本接口设置的填充值不生效。
    """ 

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        _template <typename T>
        __aicore__ inline void SetLoadDataPaddingValue(const T padValue)
    """

    param_list = """
    **参数说明**

    - padValue 
        输入， Pad 填充值的数值。
    """

    constraint_list = """
    **约束说明**
        无
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        asc.set_load_data_padding_value(10)    
        asc.set_load_data_padding_value(2.0)  
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def set_load_data_repeat_docstring():
    func_introduction = """
    用于设置 Load3Dv2 接口的 repeat 参数。设置 repeat 参数后，可以通过调用一次 Load3Dv2 接口完成多个迭代的数据搬运。
    """

    cpp_signature = """
    **对应的 Ascend C 函数原型**

    .. code-block:: c++

        __aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
    """

    param_list = """
    **参数说明**

    - repeatParams 
        设置Load3Dv2接口的repeat参数，类型为LoadDataRepeatParam。

    - repeatParams 
        height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1

    - repeatStride 
        height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。
        repeatMode为0，repeatStride的单位为16个元素。
        repeatMode为1，repeatStride的单位和具体型号有关。

    - repeatMode 
        控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。
        0：迭代沿height方向；
        1：迭代沿width方向。
    """

    constraint_list = """
    **约束说明**
        无
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        import asc
        static_param = asc.LoadDataRepeatParam(
            repeatTime=4,
            repeatStride=8,
            repeatMode=0
        )
        asc.set_load_data_repeat(static_param)
    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


def compare_docstring() -> Callable[[T], T]:
    func_introduction = """
    逐元素比较两个tensor大小，如果比较后的结果为真，则输出的结果的对应比特位为1，否则为0。可将结果存入寄存器中。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Copmare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[], 
                                        uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                        const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask, 
                                        uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Copmare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, 
                                        const uint64_t mask[], const BinaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, bool isSetMask = true>
        __aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode, 
                                        const uint64_t mask, const BinaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0, src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - cmp_mode: CMPMODE类型，表示比较模式。
      - LT: src0小于（less than）src1
      - GT: src0大于（greater than）src1
      - GE: src0大于或等于（greater than or equal to）src1
      - EQ: src0等于（equal to）src1
      - NE: src0不等于（not equal to）src1
      - LE: src0小于或等于（less than or equal to）src1
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
    - 使用整个tensor参与计算的运算符重载功能，src0和src1需满足256字节对齐；使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。
    - 将结果存入寄存器的接口没有repeat输入，repeat默认为1，即一条指令计算256B的数据。
    - 将结果存入寄存器的接口会将结果写入128bit的cmpMask寄存器中，可以用GetCmpMask接口获取寄存器保存的数据。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, count=512)

    - tensor高维切分计算样例-mask连续模式，结果存入寄存器中

      .. code-block:: python

          mask = 128
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式，结果存入寄存器中

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def compare_scalar_docstring() -> Callable[[T], T]:
    func_introduction = """
    逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出的结果的对应比特位为1，否则为0。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, uint32_t count);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void CopmareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], 
                                              uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

    .. code-block:: c++

        template <typename T, typename U, bool isSetMask = true>
        __aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                              const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, 
                                              uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src1_scalar: 源操作数，Scalar标量。数据类型和src0保持一致。
    - cmp_mode: CMPMODE类型，表示比较模式。
      - LT: src0小于（less than）src1
      - GT: src0大于（greater than）src1
      - GE: src0大于或等于（greater than or equal to）src1
      - EQ: src0等于（equal to）src1
      - NE: src0不等于（not equal to）src1
      - LE: src0小于或等于（less than or equal to）src1
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
    - 使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。
    """

    py_example = """
    **调用示例**

    - tensor高维切分计算样例-mask连续模式

      .. code-block:: python

          mask = 128
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, 
                             repeat_params=params)

    - tensor高维切分计算样例-mask逐bit模式

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, 
                             repeat_params=params)

    - tensor前n个数据计算样例

      .. code-block:: python

          asc.compare_scalar(dst, src0, src1_scalar, cmp_mode=asc.CMPMODE.LT, count=512)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def get_cmp_mask_docstring() -> Callable[[T], T]:
    func_introduction = """
    用于获取Compare（结果存入寄存器）指令的比较结果。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    """

    constraint_list = """
    **约束说明**

    - dst的空间大小不能少于128字节。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
        asc.get_cmp_mask(dst)

    """

    return func_introduction, cpp_signature, param_list, "", constraint_list, py_example


def set_cmp_mask_docstring() -> Callable[[T], T]:
    func_introduction = """
    为Select不传入mask参数的接口设置比较寄存器。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T>
        __aicore__ inline void SetCmpMask(const LocalTensor<T>& src)

    """

    param_list = """
    **参数说明**

    - src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    """

    py_example = """
    **调用示例**

    .. code-block:: python

        src = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
        asc.set_cmp_mask(src)

    """

    return func_introduction, cpp_signature, param_list, "", "", py_example


def select_docstring() -> Callable[[T], T]:
    func_introduction = """
    给定两个源操作数src0和src1，根据sel_mask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。
    """

    cpp_signature = """
    **对应的Ascend C函数原型**

    .. code-block:: c++

        template <typename T, typename U>
        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, 
                                       const LocalTensor<T>& src0, T src1, SELMODE selMode, uint32_t count)

    .. code-block:: c++

        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, 
                                       const LocalTensor<T>& src0, const LocalTensor<T>& src1, 
                                       SELMODE selMode, uint32_t count)

    .. code-block:: c++

        __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, 
                                       const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask[], 
                                       uint8_t repeatTime, const BinaryRepeatParams& repeatParams)

    """

    param_list = """
    **参数说明**

    - dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - sel_mask: 选取mask，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
    - src1: 源操作数。
      - 当selMode为模式0或模式2时：类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
      - 当selMode为模式1时，类型为T，标量数据类型。
    - sel_mode: SELMODE类型，表示指令模式。
      - VSEL_CMPMASK_SPR: 模式0，根据selMask在两个tensor中选取元素，selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。
      - VSEL_TENSOR_SCALAR_MODE: 模式1，根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。
      - VSEL_TENSOR_TENSOR_MODE: 模式2，根据selMask在两个tensor中选取元素，selMask无有效数据限制。
    - count: 参与计算的元素个数。
    - mask: 用于控制每次迭代内参与计算的元素。
    - repeat_times: 重复迭代次数。
    - repeat_params: 控制操作数地址步长的参数。
    - is_set_mask: 是否在接口内部设置mask。
    """

    constraint_list = """
    **约束说明**

    - 操作数地址对齐要求请参见通用地址对齐约束。
    - 操作数地址重叠约束请参考通用地址重叠约束。
    - 对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。
    """

    py_example = """
    **调用示例**

    - tensor前n个数据计算样例（模式0）

      .. code-block:: python

          asc.select(z_local, y_local, x_local, p_local, sel_mode=asc.SELMODE.VSEL_CMPMASK_SPR, count=512)

    - tensor前n个数据计算样例（模式1）

      .. code-block:: python

          asc.select(z_local, y_local, x_local, 0.0, sel_mode=asc.SELMODE.VSEL_TENSOR_SCALAR_MODE, count=512)

    - tensor高维切分计算样例-mask逐bit模式（模式1）

      .. code-block:: python

          mask = [uint64_max, uint64_max]
          # repeat_times = 1，一次迭代计算128个数
          params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
          asc.select(z_local, y_local, x_local, 0.0, sel_mode=asc.SELMODE.VSEL_TENSOR_SCALAR_MODE, mask=mask, 
                     repeat_times=1, repeat_params=params)

    """

    return [func_introduction, cpp_signature, param_list, "", constraint_list, py_example]


DOC_HANDLES = {
    "copy": copy_docstring,
    "set_flag": set_wait_flag_docstring,
    "get_block_num": get_block_num_docstring,
    "get_block_idx": get_block_idx_docstring,
    "get_data_block_size_in_bytes": get_data_block_size_in_bytes_docstring,
    "get_program_counter": get_program_counter_docstring,
    "get_system_cycle": get_system_cycle_docstring,
    "trap": trap_docstring,
    "data_cache_clean_and_invalid": data_cache_clean_and_invalid_docstring,
    "data_copy": data_copy_docstring,
    "data_copy_pad": data_copy_pad_docstring,
    "duplicate": duplicate_docstring,
    "get_icache_preload_status": get_icache_preload_status_docstring,
    "get_sys_workspace": get_sys_workspace_docstring,
    "icache_preload": icache_preload_docstring,
    "load_data": load_data_docstring,
    "load_data_with_transpose": load_data_with_transpose_docstring,
    "load_image_to_local": load_image_to_local_docstring,
    "mmad": mmad_docstring,
    "pipe_barrier": pipe_barrier_docstring,
    "wait_flag": set_wait_flag_docstring,
    "metrics_prof_start": metrics_prof_start_docstring,
    "metrics_prof_stop": metrics_prof_stop_docstring,
    "printf": printf_docstring,
    "scalar_cast": scalar_cast_docstring,
    "scalar_get_sff_value": scalar_get_sff_value_docstring,
    "dump_tensor": dump_tensor_docstring_docstring,
    "gather_mask": gather_mask_docstring,
    "set_vector_mask": set_vector_mask_docstring,
    "pair_reduce_sum": pair_reduce_sum_docstring,
    "repeat_reduce_sum": repeat_reduce_sum_docstring,
    "whole_reduce_max": whole_reduce_max_docstring,
    "whole_reduce_min": whole_reduce_min_docstring,
    "whole_reduce_sum": whole_reduce_sum_docstring, 
    "scatter": scatter_docstring,
    "set_aipp_functions": set_aipp_functions_docstring,
    "set_deq_scale": set_deq_scale_docstring,
    "transpose": transpose_docstring,
    "trans_data_to_5hd": trans_data_to_5hd_docstring,
    "proposal_concat": proposal_concat_docstring,
    "proposal_extract": proposal_extract_docstring,
    "set_atomic_add": set_atomic_add_docstring,
    "set_atomic_max": set_atomic_max_docstring,
    "set_atomic_min": set_atomic_min_docstring,
    "set_atomic_none": set_atomic_none_docstring,
    "set_atomic_type": set_atomic_type_docstring,
    "compare": compare_docstring,
    "compare_scalar": compare_scalar_docstring,
    "get_cmp_mask": get_cmp_mask_docstring,
    "set_cmp_mask": set_cmp_mask_docstring,
    "select": select_docstring,
    "set_load_data_boundary": set_load_data_boundary_docstring,
    "set_load_data_padding_value": set_load_data_padding_value_docstring,
    "set_load_data_repeat": set_load_data_repeat_docstring,
}


def set_common_docstring(api_name: Optional[str] = None) -> Callable[[T], T]:
    func_introduction = ""
    cpp_signature = ""
    param_list = ""
    return_list = ""
    constraint_list = ""
    py_example = ""

    if DOC_HANDLES.get(api_name) is None:
        raise RuntimeError(f"Invalid api name {api_name}")

    handler = DOC_HANDLES.get(api_name)
    func_introduction, cpp_signature, param_list, return_list, constraint_list, py_example = handler()
    
    docstr = f"""
    {func_introduction}
    {cpp_signature}
    {param_list}
    {return_list}
    {constraint_list}
    {py_example}
    """

    def decorator(fn: T) -> T:
        fn.__doc__ = docstr
        return fn

    return decorator
