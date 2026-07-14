/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/experimental/cudf/tests/utils/CudfStreamTestUtils.h"

#include "velox/common/base/Exceptions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <memory>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test;
using namespace facebook::velox::test;

namespace {

class TestCudaStream {
 public:
  TestCudaStream() {
    auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    VELOX_CHECK(
        status == cudaSuccess,
        "cudaStreamCreateWithFlags failed: {} ({})",
        cudaGetErrorString(status),
        static_cast<int>(status));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  rmm::cuda_stream_view view() const {
    return rmm::cuda_stream_view{stream_};
  }

  cudaStream_t value() const {
    return stream_;
  }

 private:
  cudaStream_t stream_{nullptr};
};

std::unique_ptr<cudf::table> makeTable(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::array<int32_t, 4> values{{1, 2, 3, 4}};
  rmm::device_buffer data(values.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32},
          static_cast<cudf::size_type>(values.size()),
          std::move(data),
          rmm::device_buffer{},
          0));
  return std::make_unique<cudf::table>(std::move(columns));
}

class CudfVectorTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(CudfVectorTest, rebindOwnedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;

  auto table = makeTable(
      allocationStream.view(),
      rmm::to_device_async_resource_ref_checked(&resource));
  allocationStream.view().synchronize();

  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      table->num_rows(),
      std::move(table),
      allocationStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  vector.reset();

  EXPECT_GT(resource.deallocationCount(), 0);
  EXPECT_EQ(resource.lastDeallocationStream(), targetStream.value());
}

TEST_F(CudfVectorTest, rebindPackedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;

  auto table = makeTable(
      allocationStream.view(), cudf::get_current_device_resource_ref());
  auto packedColumns = cudf::pack(
      table->view(),
      allocationStream.view(),
      rmm::to_device_async_resource_ref_checked(&resource));
  allocationStream.view().synchronize();
  auto tableView = cudf::unpack(packedColumns);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});

  // Model the intra-node UCX path: the packed buffer was allocated on
  // allocationStream, but downstream work is associated with targetStream.
  // The CudfVector logical stream is already targetStream, but the packed
  // buffer's deallocation stream is still allocationStream. rebindStream must
  // update the packed buffer even when stream_ already matches targetStream.
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      targetStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  vector.reset();

  EXPECT_GT(resource.deallocationCount(), 0);
  EXPECT_EQ(resource.lastDeallocationStream(), targetStream.value());
}

} // namespace
