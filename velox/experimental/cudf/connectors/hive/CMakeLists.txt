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
  velox_cudf_hive_connector
  CudfHiveConfig.cpp
  CudfHiveConnector.cpp
  CudfHiveConnectorSplit.cpp
  CudfHiveDataSink.cpp
  CudfHiveDataSource.cpp
  CudfHiveTableHandle.cpp
  CudfScanPreparation.cpp
  CudfSplitReader.cpp
  CudfSplitReaderHelpers.cpp
  ExecutorReadBroker.cpp
  ExecutorSplitPrefetch.cpp
)

target_link_libraries(
  velox_cudf_hive_connector
  PRIVATE velox_dwio_common velox_hive_connector
  PUBLIC cudf::cudf nvtx3::nvtx3-cpp velox_cudf_expression
)

if(VELOX_ENABLE_ABFS)
  target_link_libraries(velox_cudf_hive_connector PRIVATE velox_abfs)
endif()

if(VELOX_ENABLE_S3)
  target_compile_definitions(
    velox_cudf_hive_connector
    PUBLIC KVIKIO_CUFILE_FOUND KVIKIO_CUFILE_VERSION_API_FOUND KVIKIO_LIBCURL_FOUND
  )
  velox_include_directories(velox_cudf_hive_connector PRIVATE ${AWSSDK_INCLUDE_DIRS})
  target_link_libraries(
    velox_cudf_hive_connector
    PRIVATE velox_s3fs ${AWSSDK_LIBRARIES} CURL::libcurl
  )
endif()

add_subdirectory(iceberg)
