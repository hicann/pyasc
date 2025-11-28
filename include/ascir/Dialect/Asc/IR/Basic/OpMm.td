/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASC_BASIC_OP_MM_TD
#define ASC_BASIC_OP_MM_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def AscendC_LoadDataOp
    : APIOp<"load_data", "LoadData", [AscFunc]> {
  let arguments = (ins AscendC_LocalTensor:$dst, AscendC_LocalTensor:$src,
                   AscendC_BaseLoadDataParamsTypeInterface:$loadDataParams);
}

def AscendC_MmadOp : APIOp<"mmad", "Mmad", [AscFunc]> {
  let arguments = (ins AnyType:$dst, AnyType:$fm,
                   AnyType:$filter, Optional<AnyType>:$bias, AnyType:$mmadParams);
}

def AscendC_SetFmatrixOp : APIOp<"set_fmatrix", "SetFmatrix", [AscFunc]> {
  let arguments = (ins AnySignlessIntegerOrIndex:$l1H,
                       AnySignlessIntegerOrIndex:$l1W,
                       AnyRankedOrUnrankedMemRef:$padList,
                       AscendC_FmatrixMode:$fmatrixMode);
}

def AscendC_SetLoadDataPaddingValueOp
    : APIOp<"set_load_data_padding_value", "SetLoadDataPaddingValue", [AscFunc]> {
  let arguments = (ins AnyType:$padValue);
}

def AscendC_SetLoadDataRepeatOp
    : APIOp<"set_load_data_repeat", "SetLoadDataRepeat", [AscFunc]> {
  let arguments = (ins AscendC_LoadDataRepeatParam:$repeatParams);
}

def AscendC_SetLoadDataBoundaryOp
    : APIOp<"set_load_data_boundary", "SetLoadDataBoundary", [AscFunc]> {
  let arguments = (ins AnySignlessIntegerOrIndex:$value);
}


#endif // ASC_BASIC_OP_MM_TD
