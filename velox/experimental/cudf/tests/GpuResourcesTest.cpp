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

#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Exceptions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::test;

namespace {

// deviceMemoryDiagnosticsEnabled() caches its answer in a function-local
// static, so the environment must be set before anything calls it. Static
// initialization is the only point guaranteed to precede every test.
const bool kDiagnosticsRequested = [] {
  ::setenv("GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS", "1", 1);
  return true;
}();

class TestCudaStream {
 public:
  TestCudaStream() {
    const auto status =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
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

 private:
  cudaStream_t stream_{nullptr};
};

std::optional<DeviceAllocationContextStats> findContext(
    const std::vector<DeviceAllocationContextStats>& stats,
    const std::string& context) {
  for (const auto& entry : stats) {
    if (entry.context == context) {
      return entry;
    }
  }
  return std::nullopt;
}

std::size_t totalAttributedBytes(
    const std::vector<DeviceAllocationContextStats>& stats) {
  std::size_t bytes = 0;
  for (const auto& entry : stats) {
    bytes += entry.currentBytes;
  }
  return bytes;
}

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

class GpuResourcesTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    EXPECT_TRUE(kDiagnosticsRequested);
    ASSERT_TRUE(deviceMemoryDiagnosticsEnabled())
        << "the attribution map only exists on the diagnostic path";
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    // Installs the process-wide attribution resource that
    // reattributeDeviceAllocation and captureDeviceAllocationAttribution read.
    // Reinstalled per test so each case starts from an empty context map, and
    // held by the fixture so allocations made through it outlive installation.
    resource_ = wrapDeviceMemoryResourceForDiagnostics(
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            rmm::mr::cuda_memory_resource{}},
        /*outputResource=*/false);
  }

  rmm::device_async_resource_ref resource() {
    return rmm::device_async_resource_ref{resource_.value()};
  }

  /// Packs a freshly built single-column table into a CudfVector, leaving the
  /// packed buffer as the vector's single device allocation.
  std::shared_ptr<CudfVector> makePackedVector(rmm::cuda_stream_view stream) {
    auto table = makeTable(stream, resource());
    auto packedColumns = cudf::pack(table->view(), stream, resource());
    // CudfVector does not join producer streams.
    stream.synchronize();
    return std::make_shared<CudfVector>(
        pool_.get(),
        ROW({"c0"}, {INTEGER()}),
        table->num_rows(),
        std::make_unique<cudf::packed_table>(cudf::packed_table{
            cudf::unpack(packedColumns), std::move(packedColumns)}),
        stream);
  }

  std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> resource_;
};

TEST_F(GpuResourcesTest, traceScopeNestsAndRestores) {
  const auto initial = currentDeviceAllocationContext();
  {
    CudaAllocationTraceScope outer("outer");
    EXPECT_EQ(currentDeviceAllocationContext(), "outer");
    {
      CudaAllocationTraceScope inner("inner");
      EXPECT_EQ(currentDeviceAllocationContext(), "inner");
    }
    EXPECT_EQ(currentDeviceAllocationContext(), "outer");
  }
  EXPECT_EQ(currentDeviceAllocationContext(), initial);
}

TEST_F(GpuResourcesTest, allocationIsBilledToEnclosingScope) {
  TestCudaStream stream;
  constexpr std::size_t kBytes = 4096;
  std::optional<rmm::device_buffer> buffer;
  {
    CudaAllocationTraceScope scope("birth");
    buffer.emplace(kBytes, stream.view(), resource());
  }

  const auto birth = findContext(captureDeviceAllocationAttribution(), "birth");
  ASSERT_TRUE(birth.has_value());
  EXPECT_EQ(birth->currentBytes, kBytes);
  EXPECT_EQ(birth->currentAllocations, 1u);
  EXPECT_EQ(birth->peakBytes, kBytes);

  buffer.reset();
  const auto after = findContext(captureDeviceAllocationAttribution(), "birth");
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->currentBytes, 0u);
  EXPECT_EQ(after->currentAllocations, 0u);
  // Deallocation does not lower a context's peak.
  EXPECT_EQ(after->peakBytes, kBytes);
}

TEST_F(GpuResourcesTest, reattributionMovesBytesAndPreservesTotal) {
  TestCudaStream stream;
  constexpr std::size_t kBytes = 8192;
  std::optional<rmm::device_buffer> buffer;
  {
    CudaAllocationTraceScope scope("producer");
    buffer.emplace(kBytes, stream.view(), resource());
  }
  const auto totalBefore =
      totalAttributedBytes(captureDeviceAllocationAttribution());

  ASSERT_TRUE(reattributeDeviceAllocation(buffer->data(), "holder"));

  const auto stats = captureDeviceAllocationAttribution();
  const auto producer = findContext(stats, "producer");
  ASSERT_TRUE(producer.has_value());
  EXPECT_EQ(producer->currentBytes, 0u);
  EXPECT_EQ(producer->currentAllocations, 0u);
  // The producer keeps its peak, matching deallocation semantics.
  EXPECT_EQ(producer->peakBytes, kBytes);

  const auto holder = findContext(stats, "holder");
  ASSERT_TRUE(holder.has_value());
  EXPECT_EQ(holder->currentBytes, kBytes);
  EXPECT_EQ(holder->currentAllocations, 1u);
  EXPECT_EQ(holder->peakBytes, kBytes);

  // Re-attribution moves bytes between owners; it never creates or drops any.
  EXPECT_EQ(totalAttributedBytes(stats), totalBefore);

  // Freeing through the new holder keeps the accounting balanced.
  buffer.reset();
  const auto freed =
      findContext(captureDeviceAllocationAttribution(), "holder");
  ASSERT_TRUE(freed.has_value());
  EXPECT_EQ(freed->currentBytes, 0u);
  EXPECT_EQ(freed->currentAllocations, 0u);
}

TEST_F(GpuResourcesTest, reattributionToSameContextIsANoOp) {
  TestCudaStream stream;
  constexpr std::size_t kBytes = 2048;
  std::optional<rmm::device_buffer> buffer;
  {
    CudaAllocationTraceScope scope("stable");
    buffer.emplace(kBytes, stream.view(), resource());
  }

  ASSERT_TRUE(reattributeDeviceAllocation(buffer->data(), "stable"));
  ASSERT_TRUE(reattributeDeviceAllocation(buffer->data(), "stable"));

  const auto stable =
      findContext(captureDeviceAllocationAttribution(), "stable");
  ASSERT_TRUE(stable.has_value());
  EXPECT_EQ(stable->currentBytes, kBytes);
  EXPECT_EQ(stable->currentAllocations, 1u);
}

TEST_F(GpuResourcesTest, reattributionRejectsUnknownPointers) {
  EXPECT_FALSE(reattributeDeviceAllocation(nullptr, "holder"));

  // Allocated outside any cuDF resource, so the attribution map has no record
  // of it. This is also the early-out taken when diagnostics are disabled and
  // no attribution resource exists at all.
  void* foreign = nullptr;
  ASSERT_EQ(cudaMalloc(&foreign, 1024), cudaSuccess);
  const auto before = captureDeviceAllocationAttribution();
  EXPECT_FALSE(reattributeDeviceAllocation(foreign, "holder"));
  const auto after = captureDeviceAllocationAttribution();
  EXPECT_EQ(after.size(), before.size());
  EXPECT_EQ(totalAttributedBytes(after), totalAttributedBytes(before));
  ASSERT_EQ(cudaFree(foreign), cudaSuccess);
}

TEST_F(GpuResourcesTest, packedDeviceAllocationKeyedOnPackedBufferOnly) {
  TestCudaStream stream;
  auto packedVector = makePackedVector(stream.view());

  const auto key = packedVector->packedDeviceAllocation();
  ASSERT_TRUE(key.has_value());
  EXPECT_NE(key->pointer, nullptr);
  // flatSize_ is the packed buffer size, so the key covers the whole vector.
  EXPECT_EQ(key->bytes, packedVector->estimateFlatSize());

  // Table-backed vectors have many buffers and no stable single key, so they
  // are deliberately excluded from pointer-keyed re-attribution.
  auto table = makeTable(stream.view(), resource());
  stream.view().synchronize();
  const auto rows = table->num_rows();
  auto tableVector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      rows,
      std::move(table),
      stream.view());
  EXPECT_FALSE(tableVector->packedDeviceAllocation().has_value());
}

TEST_F(GpuResourcesTest, packedVectorAllocationIsReattributable) {
  TestCudaStream stream;
  std::shared_ptr<CudfVector> packedVector;
  {
    CudaAllocationTraceScope scope("packer");
    packedVector = makePackedVector(stream.view());
  }

  const auto key = packedVector->packedDeviceAllocation();
  ASSERT_TRUE(key.has_value());
  ASSERT_TRUE(reattributeDeviceAllocation(key->pointer, "consumer"));

  const auto consumer =
      findContext(captureDeviceAllocationAttribution(), "consumer");
  ASSERT_TRUE(consumer.has_value());
  EXPECT_EQ(consumer->currentBytes, key->bytes);
  EXPECT_EQ(consumer->currentAllocations, 1u);
}

} // namespace
