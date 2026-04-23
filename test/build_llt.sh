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
PYASC_SETUP_CCACHE="ON"
PYASC_SETUP_CLANG_LLD="OFF"
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

        python3 -m coverage run --source=$SRC_DIR --data-file="${COV_DIR}/.coverage" -m pytest -v $UT_PATH

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
    
    # 设置 Clang 覆盖率环境变量
    if [[ "${COV}" == "true" ]]; then
        export LLVM_PROFILE_FILE="${BUILD_DIR}/%p.profraw"
        log "Info: LLVM_PROFILE_FILE set to ${BUILD_DIR}/%p.profraw"
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

# check clang/lld availability
check_clang_available() {
  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]]; then
    if ! command -v clang >/dev/null 2>&1; then
      log "Error: clang not found, please install clang or run without --clang"
      exit 1
    fi
    if ! command -v clang++ >/dev/null 2>&1; then
      log "Error: clang++ not found, please install clang or run without --clang"
      exit 1
    fi
    if ! command -v lld >/dev/null 2>&1 && ! command -v ld.lld >/dev/null 2>&1; then
      log "Error: lld not found, please install lld or run without --clang"
      exit 1
    fi
    log "Info: clang/lld detected and available"
  fi
}

# detect compiler type
detect_compiler() {
  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]]; then
    echo "clang"
  else
    echo "gcc"
  fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    log "Info: directory required to find coverage files"
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

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mkdir -p "${_path_to_gen}"
  fi

  local compiler=$(detect_compiler)
  log "Info: detected compiler: ${compiler}"

  if [[ "${compiler}" == "clang" ]]; then
    generate_llvm_coverage "${_source_dir}" "${_coverage_file}"
  elif [[ "${compiler}" == "gcc" ]]; then
    generate_gcc_coverage "${_source_dir}" "${_coverage_file}"
  else
    log "Info: unknown compiler, trying gcc coverage generation"
    generate_gcc_coverage "${_source_dir}" "${_coverage_file}"
  fi
}

# generate coverage for gcc compiler
generate_gcc_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    log "Info: lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mkdir -p "${_path_to_gen}"
  fi

  local lcov_ignore_errors=""
  local lcov_version=$(lcov --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
  if [[ "${lcov_version}" -ge 2 ]]; then
    lcov_ignore_errors="mismatch,empty,inconsistent,negative,source,unused"
  else
    lcov_ignore_errors="source,graph"
  fi
  log "Info: lcov version ${lcov_version}, using ignore-errors: ${lcov_ignore_errors}"

  lcov -c -d "${_source_dir}" -o "${_coverage_file}" --rc geninfo_unexecuted_blocks=1 --ignore-errors ${lcov_ignore_errors}
  lcov -r "${_coverage_file}" "/home/jenkins/Ascend/ascend-toolkit/latest/*" -o "${_coverage_file}" --ignore-errors unused
  log "Info: generated coverage file ${_coverage_file}"
}

# generate coverage for clang compiler
generate_llvm_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  local llvm_profdata=""
  local llvm_cov=""

  if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]] && [[ -n "${CLANG_CMD}" ]]; then
    local clang_version=""
    if [[ "${CLANG_CMD}" == "clang" ]]; then
      clang_version=$(clang --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
    else
      clang_version=$(echo "${CLANG_CMD}" | grep -oP 'clang-\K[0-9]+')
    fi
    log "Info: detected clang version: ${clang_version}"

    if [[ -n "${clang_version}" ]] && command -v llvm-profdata-${clang_version} >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-${clang_version}"
      llvm_cov="llvm-cov-${clang_version}"
      log "Info: using version-matched tools: ${llvm_profdata}"
    fi
  fi

  if [[ -z "${llvm_profdata}" ]]; then
    if command -v llvm-profdata >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata"
      llvm_cov="llvm-cov"
    elif command -v llvm-profdata-19 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-19"
      llvm_cov="llvm-cov-19"
    elif command -v llvm-profdata-18 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-18"
      llvm_cov="llvm-cov-18"
    elif command -v llvm-profdata-17 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-17"
      llvm_cov="llvm-cov-17"
    elif command -v llvm-profdata-16 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-16"
      llvm_cov="llvm-cov-16"
    elif command -v llvm-profdata-15 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-15"
      llvm_cov="llvm-cov-15"
    elif command -v llvm-profdata-14 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-14"
      llvm_cov="llvm-cov-14"
    elif command -v llvm-profdata-13 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-13"
      llvm_cov="llvm-cov-13"
    elif command -v llvm-profdata-12 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-12"
      llvm_cov="llvm-cov-12"
    elif command -v llvm-profdata-11 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-11"
      llvm_cov="llvm-cov-11"
    elif command -v llvm-profdata-10 >/dev/null 2>&1; then
      llvm_profdata="llvm-profdata-10"
      llvm_cov="llvm-cov-10"
    else
      log "Error: llvm-profdata not found, please install llvm package"
      exit 1
    fi
    log "Info: using system tools: ${llvm_profdata}"
  fi

  log "Info: using ${llvm_profdata} and ${llvm_cov} for coverage processing"

  local profdata_file="${_source_dir}/coverage.profdata"
  local profraw_files=$(find "${_source_dir}" -name "*.profraw" 2>/dev/null)

  if [[ -z "${profraw_files}" ]]; then
    log "Info: no .profraw files found in ${_source_dir}"
    exit 1
  fi

  log "Info: found .profraw files"

  ${llvm_profdata} merge -sparse ${profraw_files} -o "${profdata_file}"
  if [[ $? -ne 0 ]]; then
    log "Info: failed to merge profdata files"
    exit 1
  fi

  log "Info: generated profdata file ${profdata_file}"

  local temp_coverage="${_source_dir}/coverage_temp.info"
  rm -f "${temp_coverage}"

  for binary in "${_source_dir}/bin/ascir-opt" "${_source_dir}/bin/ascir-translate"; do
    if [[ -f "${binary}" ]]; then
      log "Info: exporting coverage for ${binary}"
      ${llvm_cov} export "${binary}" -instr-profile="${profdata_file}" -format=lcov >> "${temp_coverage}"
    fi
  done

  if [[ ! -f "${temp_coverage}" ]]; then
    log "Info: no coverage data exported"
    exit 1
  fi

  mv "${temp_coverage}" "${_coverage_file}"

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

  local lcov_version=$(lcov --version 2>&1 | grep -oP 'version \K[0-9]+' | head -1)
  local lcov_filter_ignore=""
  if [[ "${lcov_version}" -ge 2 ]]; then
    lcov_filter_ignore="--ignore-errors unused"
  fi

  lcov --remove "${_coverage_file}" \
       '/usr/include/*' '/usr/local/include/*' \
       '*/llvm-install/*' '*/llvm/*' '*/mlir/*' \
       "${BUILD_DIR}/*" \
       -o "${_filtered_file}" ${lcov_filter_ignore}

  log "Info: Coverage statistics:"
  lcov --summary "${_filtered_file}" 2>&1 | grep -E "lines|functions"
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
    --clang)
        PYASC_SETUP_CLANG_LLD="ON"
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

check_clang_available

if [[ "${PYASC_SETUP_CCACHE}" == "ON" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DASCIR_CCACHE=ON"
fi

if [[ "${PYASC_SETUP_CLANG_LLD}" == "ON" ]];then
    CUSTOM_OPTION="${CUSTOM_OPTION}
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_LINKER=lld
    -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld
    -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld
    -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
fi
build_test
