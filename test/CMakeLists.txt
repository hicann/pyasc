# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(TARGET_NAME check-ascir)
configure_lit_site_cfg(
    ${CMAKE_SOURCE_DIR}/test/lit.site.cfg.in

    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg 
    MAIN_CONFIG
    ${CMAKE_SOURCE_DIR}/test/lit.cfg
)
list(APPEND MLIR_CLANG_TEST_DEPS
    llvm-config
    ascir-opt
    ascir-translate
)
set(LLVM_LIT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")
add_lit_testsuite(check-ascir "Runing the AscIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  ARGS --xunit-xml-output ${CMAKE_CURRENT_BINARY_DIR}/ascir_lit_test_report.xml
)

set_target_properties(check-ascir PROPERTIES FOLDER "AscIR test")