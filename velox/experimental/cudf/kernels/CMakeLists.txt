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

# Shared GPU kernel library.
# These kernels are designed to be reusable across projects (velox-cudf,
# spark-rapids-jni) and may eventually be extracted into an independent repo.

add_library(
  velox_cudf_kernels
  src/TopNRowNumber.cpp)

set_target_properties(
  velox_cudf_kernels
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  velox_cudf_kernels
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(velox_cudf_kernels PUBLIC cudf::cudf)
