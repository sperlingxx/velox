# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(
  velox_cudf_exec
  BloomFilterKernels.cu
  CudfAssignUniqueId.cpp
  CudfConversion.cpp
  CudfExpand.cpp
  CudfFilterProject.cpp
  CudfHashAggregation.cpp
  DecimalAggregationKernels.cu
  CudfHashJoin.cpp
  CudfNestedLoopJoin.cpp
  CudfLimit.cpp
  CudfLocalPartition.cpp
  CudfOrderBy.cpp
  CudfShufflePartition.cpp
  CudfTopN.cpp
  CudfTopNRowNumber.cpp
  CudfWindow.cpp
  DebugUtil.cpp
  OperatorAdapters.cpp
  PinnedArrowInterop.cpp
  ToCudf.cpp
  Utilities.cpp
  VeloxCudfInterop.cpp
)

set_target_properties(velox_cudf_exec PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

target_link_libraries(
  velox_cudf_exec
  PUBLIC cudf::cudf
  PRIVATE
    velox_cudf_kernels
    arrow
    velox_arrow_bridge
    velox_exception
    velox_common_base
    velox_cudf_vector
    velox_exec
    velox_cudf_hive_connector
    velox_cudf_expression
)

# PinnedArrowInterop.cpp uses nanoarrow headers directly (cudf's private dep).
# nanoarrow is already linked into libcudf.so; we only need the headers + namespace define.
if(DEFINED nanoarrow_SOURCE_DIR)
  # cudf was fetched from source: nanoarrow was fetched as a sub-dependency.
  target_include_directories(velox_cudf_exec PRIVATE
    "${nanoarrow_SOURCE_DIR}/src"
    "${nanoarrow_BINARY_DIR}/src"
  )
else()
  # cudf is system-installed: nanoarrow headers are not bundled.
  # Fetch nanoarrow source for headers only (binary is already in libcudf.so).
  include(FetchContent)
  FetchContent_Declare(
    nanoarrow
    GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
    GIT_TAG        apache-arrow-nanoarrow-0.7.0
    GIT_SHALLOW    TRUE
    UPDATE_DISCONNECTED TRUE
  )
  FetchContent_GetProperties(nanoarrow)
  if(NOT nanoarrow_POPULATED)
    FetchContent_Populate(nanoarrow)
  endif()
  # FetchContent_Populate only downloads the source; it does not run
  # nanoarrow's CMake configure step, so nanoarrow_config.h is missing.
  # Generate it from the template with the matching version.
  set(NANOARROW_VERSION_MAJOR 0)
  set(NANOARROW_VERSION_MINOR 7)
  set(NANOARROW_VERSION_PATCH 0)
  set(NANOARROW_VERSION "0.7.0")
  # NANOARROW_NAMESPACE is set via compile definition below; use a comment
  # placeholder in the config header to avoid a duplicate-macro warning.
  set(NANOARROW_NAMESPACE_DEFINE
      "// NANOARROW_NAMESPACE defined via compile definition")
  configure_file(
    "${nanoarrow_SOURCE_DIR}/src/nanoarrow/nanoarrow_config.h.in"
    "${nanoarrow_BINARY_DIR}/src/nanoarrow/nanoarrow_config.h"
  )
  target_include_directories(velox_cudf_exec PRIVATE
    "${nanoarrow_SOURCE_DIR}/src"
    "${nanoarrow_BINARY_DIR}/src"
  )
endif()
target_compile_definitions(velox_cudf_exec PRIVATE NANOARROW_NAMESPACE=cudf)

target_compile_options(velox_cudf_exec PRIVATE -Wno-missing-field-initializers)
target_compile_options(velox_cudf_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
