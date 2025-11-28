#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e
CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/output
TEST_PATH=${CURRENT_DIR}/python
UT_PATH=${CURRENT_DIR}/../python/test/unit
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)))
JOB_NUM="-j${CPU_NUM}"
ASAN="false"
COV="false"
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
CUSTOM_OPTION="-DASCIR_LLT_TEST=ON -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR} -Dpybind11_DIR=${PYBIND11_DIR}"

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function clean()
{
    if [[ -n "${BUILD_DIR}" ]];then
        rm -rf ${BUILD_DIR}
    fi

    if [[ -z "${TEST}" ]];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ../..  ${CUSTOM_OPTION} ${extra_option}
}

function run_python_ut()
{
    export PATH="$LLVM_INSTALL_PATH/bin:$PATH"

    if [[ "${COV}" == "true" ]];then
        COV_DIR=${CURRENT_DIR}/cov_py
        SRC_DIR=$(python3 -c "import pkg_resources; print(pkg_resources.resource_filename('asc', ''))")
        log "Info: source directory is ${SRC_DIR}"

        coverage run --source=$SRC_DIR --data-file="${COV_DIR}/.coverage" -m pytest -v $UT_PATH

        log "Info: coverage data directory is ${COV_DIR}"

        cd "${COV_DIR}"
        coverage report
        coverage html
    else
        python3 -m pytest -v $UT_PATH -n auto -k "not host"
        python3 -m pytest -v $UT_PATH -n 1 -k "host"
    fi
}

function run_check_ascir()
{
    clean
    cd ${BUILD_DIR}
    if [[ -z "${LIT_INSTALL_PATH}" ]] || [[ ! -e "${LIT_INSTALL_PATH}" ]];then
        cmake_config "-DLLVM_PREFIX_PATH=${LLVM_INSTALL_PATH} -DLLVM_EXTERNAL_LIT=${LLVM_INSTALL_PATH}/bin/llvm-lit"
    else
        cmake_config "-DLLVM_PREFIX_PATH=${LLVM_INSTALL_PATH} -DLLVM_EXTERNAL_LIT=${LIT_INSTALL_PATH}/bin/lit"
    fi
    make check-ascir ${JOB_NUM}
    cd ${CURRENT_DIR}

    if [[ "${COV}" == "true" ]];then
        cov_file=${BUILD_DIR}/coverage.info
        generate_coverage "${BUILD_DIR}" "${cov_file}"
        filter_coverage   "${cov_file}" "${cov_file}_filtered"
        generate_html     "${cov_file}_filtered" "${CURRENT_DIR}/cov_ascir"
    fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    log "Info: directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    log "Info: directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    log "Info: using default file name to generate coverage"
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mkdir -p "${_path_to_gen}"
  fi
  lcov -c -d "${_source_dir}" -o "${_coverage_file}"
  lcov -r "${_coverage_file}" "/home/jenkins/Ascend/ascend-toolkit/latest/*" -o "${_coverage_file}"
  log "Info: generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    log "Info: coverage data file required"
    exit 1
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: lcov is required to generate coverage data, please install"
    exit 1
  fi
  lcov --remove "${_coverage_file}" '/usr/include/*' '/usr/local/include/*' "*/llvm/include/llvm/*" "*/mlir/include/mlir/*" "${BUILD_DIR}/*" \
                                     -o "${_filtered_file}"
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  \which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_out_path})"
  if [[ ! -d "${_out_path}" ]]; then
    mkdir -p "${_out_path}"
  fi
  genhtml "${_filtered_file}" -o "${_out_path}"
}

function build_test() {
    if [[ "${TEST}" == "all" ]]; then
        run_check_ascir
        run_python_ut
        log "Info: test all success."
    elif [[ "${TEST}" == "lit" ]]; then
        run_check_ascir
        log "Info: test check-ascir success."
    else
        run_python_ut
        log "Info: test run_python_ut success."
    fi
}


TEST="all"
while [[ $# -gt 0 ]]; do
    case $1 in
    --check-ascir)
        TEST="lit"
        shift
        ;;
    --llvm_install_path)
        LLVM_INSTALL_PATH="$2"
        shift 2
        ;;
    --lit_install_path)
        LIT_INSTALL_PATH="$2"
        shift 2
        ;;
    --run_python_ut)
        TEST="python_ut"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    *)
        break
        ;;
    esac
done
if [[ -z "${LLVM_INSTALL_PATH}" ]] || [[ ! -e "${LLVM_INSTALL_PATH}" ]];then
    log "Error: --llvm_install_path need to be set or LLVM_INSTALL_PATH: ${LLVM_INSTALL_PATH} is not exsit."
    exit 1
fi

if [[ -z "${LIT_INSTALL_PATH}" ]] || [[ ! -e "${LIT_INSTALL_PATH}" ]];then
    if [[ "${TEST}" != "python_ut" ]];then
        log "Warning: --lit_install_path need to be set, otherwise it will use default path: ${LLVM_INSTALL_PATH},\
            please make sure lit exsit at default path!"
    fi
fi

if [[ "${ASAN}" == "true" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_ASAN=true"
fi

if [[ "${COV}" == "true" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_COVERAGE=true"
fi
build_test
