/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/VecUnary.h"
#include "ascir/Target/Asc/Basic/VecVconv.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Type conversion operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL0Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode())
       << ", " << emitter.getOrCreateName(op.getMask()) << ", " << emitter.getOrCreateName(op.getRepeatTimes())
       << ", AscendC::UnaryRepeatParams(" << emitter.getOrCreateName(op.getDstBlkStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcRepStride()) << "))";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL1Op op)
{
    auto maskName = printMask(emitter, op);
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode())
       << ", " << maskName << ", " << emitter.getOrCreateName(op.getRepeatTimes())
       << ", AscendC::UnaryRepeatParams(" << emitter.getOrCreateName(op.getDstBlkStride()) << ", "
       << emitter.getOrCreateName(op.getSrcBlkStride()) << ", " << emitter.getOrCreateName(op.getDstRepStride()) << ", "
       << emitter.getOrCreateName(op.getSrcRepStride()) << "))";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastL2Op op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", "
       << "AscendC::RoundMode::" << stringifyRoundMode(op.getRoundMode()) << ", "
       << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL0Op op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printCastDeqL01Template(emitter, op));
    printUnaryL0Params(emitter, op);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL1Op op)
{
    auto& os = emitter.ostream();
    auto maskName = printMask(emitter, op);
    FAIL_OR(printCastDeqL01Template(emitter, op));
    printUnaryL1Params(emitter, op, maskName);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastDeqL2Op op)
{
    auto& os = emitter.ostream();
    FAIL_OR(printCastDeqL2Template(emitter, op));
    printUnaryL2Params(emitter, op);
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter &emitter, ascendc::SetDeqScaleOp op)
{
    auto &os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getScale());
    if (auto Offset = op.getOffset()) {
        os << ", " << emitter.getOrCreateName(Offset);
    }
    if (auto SignMode = op.getOffset()) {
        os << ", " << emitter.getOrCreateName(SignMode);
    }
    os << ")";
    return success();
}
