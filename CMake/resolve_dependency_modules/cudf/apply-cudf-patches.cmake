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

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(CUDF_PAGE_HEADER "${SOURCE_DIR}/cpp/src/io/parquet/page_hdr.cu")
if(NOT EXISTS "${CUDF_PAGE_HEADER}")
  message(FATAL_ERROR "cuDF page header source not found: ${CUDF_PAGE_HEADER}")
endif()

set(CUDF_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/cudf-empty-dictionary-page.patch")
file(READ "${CUDF_PAGE_HEADER}" CUDF_PAGE_HEADER_CONTENTS)
string(REGEX MATCHALL "parse_valid_page_header" CUDF_PATCH_MARKERS "${CUDF_PAGE_HEADER_CONTENTS}")
list(LENGTH CUDF_PATCH_MARKERS CUDF_PATCH_MARKER_COUNT)

if(CUDF_PATCH_MARKER_COUNT EQUAL 0)
  execute_process(
    COMMAND patch --batch --forward --dry-run -p1 -i "${CUDF_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PATCH_APPLY_CHECK
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT CUDF_PATCH_APPLY_CHECK EQUAL 0)
    message(FATAL_ERROR "cuDF empty dictionary page patch does not apply cleanly")
  endif()

  execute_process(
    COMMAND patch --batch --forward -p1 -i "${CUDF_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PATCH_RESULT
  )
  if(NOT CUDF_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to patch cuDF empty dictionary page handling")
  endif()

  file(READ "${CUDF_PAGE_HEADER}" CUDF_PAGE_HEADER_CONTENTS)
  string(REGEX MATCHALL "parse_valid_page_header" CUDF_PATCH_MARKERS "${CUDF_PAGE_HEADER_CONTENTS}")
  list(LENGTH CUDF_PATCH_MARKERS CUDF_PATCH_MARKER_COUNT)
  if(NOT CUDF_PATCH_MARKER_COUNT EQUAL 4)
    message(FATAL_ERROR "cuDF empty dictionary page patch produced an unexpected result")
  endif()
elseif(CUDF_PATCH_MARKER_COUNT EQUAL 4)
  execute_process(
    COMMAND patch --batch --force --reverse --dry-run -p1 -i "${CUDF_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PATCH_REVERSE_CHECK
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT CUDF_PATCH_REVERSE_CHECK EQUAL 0)
    message(FATAL_ERROR "cuDF empty dictionary page patch is only partially applied")
  endif()
else()
  message(FATAL_ERROR "cuDF empty dictionary page patch is only partially applied")
endif()

set(CUDF_PAGE_INFO_SOURCE "${SOURCE_DIR}/cpp/src/io/parquet/reader_impl_preprocess.cu")
set(CUDF_PAGE_INFO_PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/cudf-page-info-initialization.patch")
file(READ "${CUDF_PAGE_INFO_SOURCE}" CUDF_PAGE_INFO_CONTENTS)
string(
  REGEX MATCHALL
        "make_zeroed_device_uvector_async<PageInfo>"
        CUDF_PAGE_INFO_MARKERS
        "${CUDF_PAGE_INFO_CONTENTS}"
)
list(LENGTH CUDF_PAGE_INFO_MARKERS CUDF_PAGE_INFO_MARKER_COUNT)

if(CUDF_PAGE_INFO_MARKER_COUNT EQUAL 0)
  execute_process(
    COMMAND patch --batch --forward --dry-run -p1 -i "${CUDF_PAGE_INFO_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PAGE_INFO_APPLY_CHECK
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT CUDF_PAGE_INFO_APPLY_CHECK EQUAL 0)
    message(FATAL_ERROR "cuDF PageInfo initialization patch does not apply cleanly")
  endif()

  execute_process(
    COMMAND patch --batch --forward -p1 -i "${CUDF_PAGE_INFO_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PAGE_INFO_RESULT
  )
  if(NOT CUDF_PAGE_INFO_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to patch cuDF PageInfo initialization")
  endif()

  file(READ "${CUDF_PAGE_INFO_SOURCE}" CUDF_PAGE_INFO_CONTENTS)
  string(
    REGEX MATCHALL
          "make_zeroed_device_uvector_async<PageInfo>"
          CUDF_PAGE_INFO_MARKERS
          "${CUDF_PAGE_INFO_CONTENTS}"
  )
  list(LENGTH CUDF_PAGE_INFO_MARKERS CUDF_PAGE_INFO_MARKER_COUNT)
  if(NOT CUDF_PAGE_INFO_MARKER_COUNT EQUAL 1)
    message(FATAL_ERROR "cuDF PageInfo initialization patch produced an unexpected result")
  endif()
elseif(CUDF_PAGE_INFO_MARKER_COUNT EQUAL 1)
  execute_process(
    COMMAND patch --batch --force --reverse --dry-run -p1 -i "${CUDF_PAGE_INFO_PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE CUDF_PAGE_INFO_REVERSE_CHECK
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT CUDF_PAGE_INFO_REVERSE_CHECK EQUAL 0)
    message(FATAL_ERROR "cuDF PageInfo initialization patch is only partially applied")
  endif()
else()
  message(FATAL_ERROR "cuDF PageInfo initialization patch is only partially applied")
endif()
