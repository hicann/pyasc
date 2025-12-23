/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASC_BASIC_OP_LIST_TENSOR_TD
#define ASC_BASIC_OP_LIST_TENSOR_TD

include "Base.td"
include "Core/Attributes.td"
include "Core/Interfaces.td"
include "Core/Types.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorDesc operations
//===----------------------------------------------------------------------===//

def AscendC_TensorDescOp : AscendC_Op<"tensor_desc"> {
  let arguments = (ins TypeAttr:$elementType);
  let results = (outs AscendC_TensorDesc:$tensorDesc);
  let assemblyFormat = "$elementType attr-dict `:` qualified(type($tensorDesc))";
}

def AscendC_TensorDescGetDimOp : APIOp<"tensor_desc_get_dim", "GetDim", [AscMemberFunc]> {
  let arguments = (ins AscendC_TensorDesc:$tensorDesc);
  let results = (outs UI64:$result);
  let assemblyFormat = "$tensorDesc attr-dict `:` qualified(type($tensorDesc)) `,` type($result)";
}

def AscendC_TensorDescGetIndexOp : APIOp<"tensor_desc_get_index", "GetIndex", [AscMemberFunc]> {
  let arguments = (ins AscendC_TensorDesc:$tensorDesc);
  let results = (outs UI64:$result);
  let assemblyFormat = "$tensorDesc attr-dict `:` qualified(type($tensorDesc)) `,` type($result)";
}

def AscendC_TensorDescGetShapeOp : APIOp<"tensor_desc_get_shape", "GetShape", [AscMemberFunc]> {
  let arguments = (ins AscendC_TensorDesc:$tensorDesc, AnySignlessIntegerOrIndex:$offset);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = "$tensorDesc `,` $offset attr-dict `:` qualified(type($tensorDesc)) `,` type($result) `,` type($offset)";
}

def AscendC_TensorDescGetDataPtrOp : APIOp<"tensor_desc_get_data_ptr", "GetDataPtr", [AscMemberFunc]> {
  let arguments = (ins AscendC_TensorDesc:$tensorDesc);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = "$tensorDesc attr-dict `:` qualified(type($tensorDesc)) `,` type($result)";
}

def AscendC_TensorDescGetDataObjOp : APIOp<"tensor_desc_get_data_obj", "GetDataObj", [AscMemberFunc]> {
  let arguments = (ins AscendC_TensorDesc:$tensorDesc);
  let results = (outs AscendC_GlobalTensor:$result);
  let assemblyFormat = "$tensorDesc attr-dict `:` qualified(type($tensorDesc)) `,` type($result)";
}
#endif //ASC_BASIC_OP_LIST_TENSOR_TD
