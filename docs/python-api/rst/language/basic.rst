.. Copyright (c) 2025 Huawei Technologies Co., Ltd.
.. This program is free software, you can redistribute it and/or modify it under the terms and conditions of
.. CANN Open Software License Agreement Version 2.0 (the "License").
.. Please refer to the License for details. You may not use this file except in compliance with the License.
.. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
.. INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
.. See LICENSE in the root of the software repository for the full text of the License.

asc.language.basic
==============================

.. currentmodule:: asc.language.basic


Common operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    copy
    data_cache_clean_and_invalid
    data_copy
    data_copy_pad
    dump_tensor
    duplicate
    get_block_idx
    get_block_num
    get_cmp_mask
    get_data_block_size_in_bytes
    get_icache_preload_status
    get_program_counter
    get_system_cycle
    get_sys_workspace
    icache_preload
    load_data
    load_data_with_transpose
    load_image_to_local
    metrics_prof_start
    metrics_prof_stop
    mmad
    pipe_barrier
    printf
    proposal_concat
    proposal_extract
    scatter
    select
    set_aipp_functions
    set_atomic_add
    set_atomic_max
    set_atomic_min
    set_atomic_none
    set_atomic_type
    set_cmp_mask
    set_fix_pipe_pre_quant_flag
    set_flag
    set_deq_scale
    set_load_data_boundary
    set_load_data_padding_value
    set_load_data_repeat
    set_vector_mask
    transpose
    trans_data_to_5hd
    trap
    wait_flag


Scalar operations
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    scalar_cast
    scalar_get_sff_value


Vector binary operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    add_deq_relu
    add_relu
    add_relu_cast
    bilinear_interpolation
    bitwise_and
    bitwise_or
    compare
    div
    fused_mul_add
    fused_mul_add_relu
    max
    min
    mul
    mul_add_dst
    mul_cast
    sub
    sub_relu
    sub_relu_cast


Vector reduce operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    pair_reduce_sum
    repeat_reduce_sum
    whole_reduce_max
    whole_reduce_min
    whole_reduce_sum


Vector-scalar operations
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    adds
    compare_scalar
    leaky_relu
    maxs
    mins
    muls
    shift_left
    shift_right


Vector unary operations
-----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    exp
    ln
    bitwise_not
    gather_mask
    reciprocal
    relu
    rsqrt
    sqrt
