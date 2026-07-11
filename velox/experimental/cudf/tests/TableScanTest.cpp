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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>
#include <tuple>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::tests::utils;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {
struct StatsFilterMetrics {
  cudf::size_type inputRowGroups{0};
  std::optional<cudf::size_type> rowGroupsAfterStats;
  cudf::size_type outputRows{0};
};

StatsFilterMetrics readParquetWithStatsFilter(
    const std::string& filePath,
    const RowTypePtr& rowType,
    const common::SubfieldFilters& filters,
    bool useJitFilter) {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  auto const& expr =
      createAstFromSubfieldFilters(filters, tree, scalars, rowType);

  auto options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filePath))
          .use_jit_filter(useJitFilter)
          .build();
  options.set_filter(expr);

  auto result = cudf::io::read_parquet(options);
  return {
      result.metadata.num_input_row_groups,
      result.metadata.num_row_groups_after_stats_filter,
      result.tbl->num_rows()};
}

class ExecutorBufferedInput final : public dwio::common::BufferedInput {
 public:
  ExecutorBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(readFile), pool), executor_(executor) {}

  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* const executor_;
};

class PinnedHostAllocation final {
 public:
  explicit PinnedHostAllocation(size_t size) : size_(size) {
    if (size_ > 0) {
      CUDF_CUDA_TRY(
          cudaMallocHost(reinterpret_cast<void**>(&data_), size_));
    }
  }

  ~PinnedHostAllocation() {
    if (data_ != nullptr) {
      (void)cudaFreeHost(data_);
    }
  }

  PinnedHostAllocation(const PinnedHostAllocation&) = delete;
  PinnedHostAllocation& operator=(const PinnedHostAllocation&) = delete;

  size_t size() const {
    return size_;
  }

  uint8_t* data() {
    return data_;
  }

 private:
  uint8_t* data_{nullptr};
  const size_t size_;
};

class HostDataView final : public cudf::io::datasource::buffer {
 public:
  HostDataView(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  size_t size() const override {
    return size_;
  }

  const uint8_t* data() const override {
    return data_;
  }

 private:
  const uint8_t* const data_;
  const size_t size_;
};

class PinnedBufferedInputDataSource final
    : public facebook::velox::cudf_velox::connector::hive::
          BufferedInputDataSource {
 public:
  PinnedBufferedInputDataSource(
      std::shared_ptr<dwio::common::BufferedInput> input,
      std::string data,
      folly::Baton<>* hostReadComplete)
      : BufferedInputDataSource(std::move(input)),
        buffer_(data.size()),
        hostReadComplete_(hostReadComplete) {
    if (!data.empty()) {
      std::memcpy(buffer_.data(), data.data(), data.size());
    }
  }

  using BufferedInputDataSource::host_read;

  std::unique_ptr<cudf::io::datasource::buffer> host_read(
      size_t offset,
      size_t size) override {
    VELOX_CHECK_EQ(offset, 0);
    VELOX_CHECK_EQ(size, buffer_.size());
    hostReadComplete_->post();
    return std::make_unique<HostDataView>(buffer_.data(), buffer_.size());
  }

 private:
  PinnedHostAllocation buffer_;
  folly::Baton<>* const hostReadComplete_;
};

class ThrowingBufferedInputDataSource final
    : public facebook::velox::cudf_velox::connector::hive::
          BufferedInputDataSource {
 public:
  explicit ThrowingBufferedInputDataSource(
      std::shared_ptr<dwio::common::BufferedInput> input)
      : BufferedInputDataSource(std::move(input)) {}

  using BufferedInputDataSource::host_read;

  std::unique_ptr<cudf::io::datasource::buffer> host_read(
      size_t /*offset*/,
      size_t /*size*/) override {
    throw std::runtime_error("injected host read failure");
  }
};

class BlockingHostDataSource final : public cudf::io::datasource {
 public:
  explicit BlockingHostDataSource(std::string data) : buffer_(data.size()) {
    if (!data.empty()) {
      std::memcpy(buffer_.data(), data.data(), data.size());
    }
  }

  size_t size() const override {
    return buffer_.size();
  }

  std::unique_ptr<cudf::io::datasource::buffer> host_read(
      size_t offset,
      size_t size) override {
    VELOX_CHECK_LE(offset + size, buffer_.size());
    hostReadStarted_.post();
    allowHostRead_.wait();
    return std::make_unique<HostDataView>(buffer_.data() + offset, size);
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override {
    VELOX_CHECK_LE(offset + size, buffer_.size());
    std::memcpy(dst, buffer_.data() + offset, size);
    return size;
  }

  bool waitForHostRead(std::chrono::seconds timeout) {
    return hostReadStarted_.try_wait_for(timeout);
  }

  void allowHostRead() {
    allowHostRead_.post();
  }

 private:
  PinnedHostAllocation buffer_;
  folly::Baton<> hostReadStarted_;
  folly::Baton<> allowHostRead_;
};

struct BatchedLoadState {
  explicit BatchedLoadState(cudaStream_t stream) : stream(stream) {}

  const cudaStream_t stream;
  folly::Baton<> firstCopyComplete;
  folly::Baton<> releaseFirstCopy;
  std::atomic<int32_t> secondReadAttempts{0};
  std::atomic<bool> firstCopyObserved{false};
};

void CUDART_CB observeFirstCopy(void* opaque) {
  auto* state = static_cast<BatchedLoadState*>(opaque);
  state->firstCopyComplete.post();
  state->releaseFirstCopy.wait();
}

class BatchedLoadBufferedInput final : public dwio::common::BufferedInput {
 public:
  BatchedLoadBufferedInput(
      memory::MemoryPool& pool,
      std::string firstRange,
      BatchedLoadState* state)
      : BufferedInput(
            std::make_shared<InMemoryReadFile>(
                std::string(firstRange.size() * 2, 'x')),
            pool),
        firstRange_(std::move(firstRange)),
        state_(state) {}

  std::unique_ptr<dwio::common::SeekableInputStream> enqueue(
      common::Region region,
      const dwio::common::StreamIdentifier* /*sid*/) override {
    VELOX_CHECK_EQ(region.length, firstRange_.size());
    if (enqueueCount_++ == 0) {
      return std::make_unique<dwio::common::SeekableArrayInputStream>(
          firstRange_.data(), firstRange_.size());
    }

    VELOX_CHECK_EQ(enqueueCount_, 2);
    return std::make_unique<dwio::common::SeekableArrayInputStream>(
        [state = state_]() -> std::tuple<const char*, uint64_t> {
          ++state->secondReadAttempts;
          CUDF_CUDA_TRY(cudaLaunchHostFunc(
              state->stream, observeFirstCopy, state));
          throw std::runtime_error("injected second range failure");
        });
  }

  void load(const dwio::common::LogType /*logType*/) override {}

 private:
  const std::string firstRange_;
  BatchedLoadState* const state_;
  int32_t enqueueCount_{0};
};

class ThrowOnSecondEnqueueBufferedInput final
    : public dwio::common::BufferedInput {
 public:
  ThrowOnSecondEnqueueBufferedInput(
      memory::MemoryPool& pool,
      size_t rangeSize)
      : BufferedInput(
            std::make_shared<InMemoryReadFile>(
                std::string(rangeSize * 2, 'x')),
            pool),
        rangeSize_(rangeSize) {}

  std::unique_ptr<dwio::common::SeekableInputStream> enqueue(
      common::Region region,
      const dwio::common::StreamIdentifier* /*sid*/) override {
    VELOX_CHECK_EQ(region.length, rangeSize_);
    if (enqueueAttempts_++ == 0) {
      return std::make_unique<dwio::common::SeekableArrayInputStream>(
          [this]() -> std::tuple<const char*, uint64_t> {
            ++firstReadAttempts_;
            throw std::runtime_error("stale first range read");
          });
    }

    throw std::runtime_error("injected second enqueue failure");
  }

  void load(const dwio::common::LogType /*logType*/) override {}

  int32_t enqueueAttempts() const {
    return enqueueAttempts_;
  }

  int32_t firstReadAttempts() const {
    return firstReadAttempts_;
  }

 private:
  const size_t rangeSize_;
  int32_t enqueueAttempts_{0};
  int32_t firstReadAttempts_{0};
};

class GatedDeviceReadDataSource final : public cudf::io::datasource {
 public:
  explicit GatedDeviceReadDataSource(size_t rangeSize)
      : rangeSize_(rangeSize) {}

  size_t size() const override {
    return rangeSize_ * 3;
  }

  std::unique_ptr<cudf::io::datasource::buffer> host_read(
      size_t /*offset*/,
      size_t /*size*/) override {
    VELOX_FAIL("Unexpected host read");
  }

  size_t host_read(
      size_t /*offset*/,
      size_t /*size*/,
      uint8_t* /*dst*/) override {
    VELOX_FAIL("Unexpected host read");
  }

  bool supports_device_read() const override {
    return true;
  }

  std::future<size_t> device_read_async(
      size_t /*offset*/,
      size_t size,
      uint8_t* /*dst*/,
      rmm::cuda_stream_view /*stream*/) override {
    if (deviceReadAttempts_++ == 0) {
      VELOX_CHECK_EQ(size, rangeSize_);
      firstReadScheduled_.post();
      return firstReadPromise_.get_future();
    }

    secondReadScheduled_.post();
    throw std::runtime_error("injected second device read failure");
  }

  bool waitForFirstRead(std::chrono::seconds timeout) {
    return firstReadScheduled_.try_wait_for(timeout);
  }

  bool waitForSecondRead(std::chrono::seconds timeout) {
    return secondReadScheduled_.try_wait_for(timeout);
  }

  void releaseFirstRead() {
    if (!firstReadReleased_.exchange(true)) {
      firstReadPromise_.set_value(rangeSize_);
    }
  }

 private:
  const size_t rangeSize_;
  std::promise<size_t> firstReadPromise_;
  folly::Baton<> firstReadScheduled_;
  folly::Baton<> secondReadScheduled_;
  std::atomic<int32_t> deviceReadAttempts_{0};
  std::atomic<bool> firstReadReleased_{false};
};

void CUDART_CB waitForBaton(void* baton) {
  static_cast<folly::Baton<>*>(baton)->wait();
}
} // namespace

class TableScanTest : public virtual CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    ExchangeSource::factories().clear();
    ExchangeSource::registerFactory(createLocalExchangeSource);
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  Split makeCudfHiveSplit(std::string path, int64_t splitWeight = 0) {
    return Split(makeCudfHiveConnectorSplit(std::move(path), splitWeight));
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>&
          parquetSplit,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {parquetSplit}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const Split&& split,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {split}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& duckDbSql) {
    return CudfHiveConnectorTestBase::assertQuery(plan, filePaths, duckDbSql);
  }

  // Run query with spill enabled.
  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& spillDirectory,
      const std::string& duckDbSql) {
    return AssertQueryBuilder(plan, duckDbQueryRunner_)
        .spillDirectory(spillDirectory)
        .config(core::QueryConfig::kSpillEnabled, false)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults(duckDbSql);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    auto tableHandle = makeTableHandle();
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .endTableScan()
        .planNode();
  }

  static PlanNodeStats getTableScanStats(const std::shared_ptr<Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  static std::unordered_map<std::string, RuntimeMetric>
  getTableScanRuntimeStats(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  static int64_t getSkippedStridesStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedStrides"].sum;
  }

  static int64_t getSkippedSplitsStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedSplits"].sum;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<Task>& task,
      uint32_t n) {
    // Limit wait to 10 seconds.
    size_t iteration{0};
    while (task->numFinishedDrivers() < n and iteration < 100) {
      /* sleep override */
      usleep(100'000); // 0.1 second.
      ++iteration;
    }
    ASSERT_EQ(n, task->numFinishedDrivers());
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           TINYINT(),
           DOUBLE(),
           BIGINT(),
           VARCHAR(),
           REAL()})};
};

class TableScanTestParameterized : public TableScanTest,
                                   public testing::WithParamInterface<bool> {};

TEST_P(TableScanTestParameterized, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const std::string duckDbSql = "SELECT * FROM tmp";

  // Helper to test scan all columns for the given splits
  auto testScanAllColumns =
      [&](const std::vector<std::shared_ptr<
              facebook::velox::connector::ConnectorSplit>>& splits) {
        auto task = AssertQueryBuilder(duckDbQueryRunner_)
                        .plan(plan)
                        .splits(splits)
                        .assertResults(duckDbSql);

        // A quick sanity check for memory usage reporting. Check that peak
        // total memory usage for the project node is > 0.
        auto planStats = toPlanStats(task->taskStats());
        auto scanNodeId = plan->id();
        auto it = planStats.find(scanNodeId);
        ASSERT_TRUE(it != planStats.end());
        // TODO (dm): enable this test once we start to track gpu memory
        // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

        //  Verifies there is no dynamic filter stats.
        ASSERT_TRUE(it->second.dynamicFilterStats.empty());

        // TODO: We are not writing any customStats yet so disable this check
        // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
      };

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  // Test scan all columns with CudfHiveConnectorSplits
  {
    auto splits = makeCudfHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }

  // Test scan all columns with HiveConnectorSplits
  {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    splits.push_back(
        facebook::velox::connector::hive::HiveConnectorSplitBuilder(
            filePath->getPath())
            .connectorId(kCudfHiveConnectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .build());
    testScanAllColumns(splits);
  }
}

TEST_P(TableScanTestParameterized, allColumnsUsingExperimentalReader) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  const std::string duckDbSql =
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp";

  auto splits = makeCudfHiveConnectorSplits(
      {filePath, filePath, filePath, filePath, filePath});

  auto useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseExperimentalCudfReader,
       "true"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(splits)
                  .assertResults(duckDbSql);

  // A quick sanity check for memory usage reporting. Check that peak
  // total memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  // TODO (dm): enable this test once we start to track gpu memory
  // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

  //  Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());

  // TODO: We are not writing any customStats yet so disable this check
  // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TableScanTestParameterized,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "FileDataSource";
    });

TEST_F(TableScanTest, bufferedInputDeviceReadAsyncWaitsForDeviceCopy) {
  constexpr size_t kDataSize = 4 << 10;
  std::string expected(kDataSize, '\0');
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<char>(i % 251);
  }

  folly::CPUThreadPoolExecutor executor(1);
  auto input = std::make_shared<ExecutorBufferedInput>(
      std::make_shared<InMemoryReadFile>(std::string(expected)),
      *pool_,
      &executor);
  folly::Baton<> hostReadComplete;
  PinnedBufferedInputDataSource dataSource(
      std::move(input), expected, &hostReadComplete);

  rmm::cuda_stream gateStream(rmm::cuda_stream::flags::non_blocking);
  rmm::cuda_stream copyStream(rmm::cuda_stream::flags::non_blocking);
  rmm::device_buffer destination(expected.size(), copyStream);
  CudaEvent copyGate(cudaEventDisableTiming);
  folly::Baton<> releaseCopy;
  std::future<size_t> future;
  bool copyReleased = false;
  auto cleanupGuard = folly::makeGuard([&] {
    if (!copyReleased) {
      releaseCopy.post();
    }
    if (future.valid()) {
      future.wait();
    }
    copyStream.synchronize_no_throw();
    gateStream.synchronize_no_throw();
  });

  CUDF_CUDA_TRY(
      cudaLaunchHostFunc(gateStream.value(), waitForBaton, &releaseCopy));
  copyGate.recordFrom(gateStream);
  copyGate.waitOn(copyStream);

  future = dataSource.device_read_async(
      0,
      expected.size(),
      static_cast<uint8_t*>(destination.data()),
      copyStream);
  ASSERT_TRUE(hostReadComplete.try_wait_for(std::chrono::seconds(5)));
  EXPECT_EQ(
      future.wait_for(std::chrono::milliseconds(500)),
      std::future_status::timeout);

  releaseCopy.post();
  copyReleased = true;
  ASSERT_EQ(future.get(), expected.size());

  std::string actual(expected.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      destination.data(),
      actual.size(),
      cudaMemcpyDeviceToHost,
      copyStream.value()));
  copyStream.synchronize();
  EXPECT_EQ(actual, expected);
}

TEST_F(TableScanTest, bufferedInputDeviceReadAsyncPropagatesException) {
  folly::CPUThreadPoolExecutor executor(1);
  auto input = std::make_shared<ExecutorBufferedInput>(
      std::make_shared<InMemoryReadFile>(std::string("x")),
      *pool_,
      &executor);
  ThrowingBufferedInputDataSource dataSource(std::move(input));
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);

  auto future = dataSource.device_read_async(0, 1, nullptr, stream);
  EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(TableScanTest, bufferedInputFetchClearsPendingLoadsAfterFailure) {
  constexpr size_t kRangeSize = 4 << 10;
  std::string firstRange(kRangeSize, '\0');
  for (size_t i = 0; i < firstRange.size(); ++i) {
    firstRange[i] = static_cast<char>(i % 251);
  }

  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  BatchedLoadState state(stream.value());
  auto input = std::make_shared<BatchedLoadBufferedInput>(
      *pool_, firstRange, &state);
  auto dataSource = std::make_shared<
      facebook::velox::cudf_velox::connector::hive::
          BufferedInputDataSource>(std::move(input));
  std::vector<cudf::io::text::byte_range_info> ranges{
      {0, kRangeSize}, {kRangeSize, kRangeSize}};

  auto [deviceBuffers, deviceSpans, loadFuture] =
      facebook::velox::cudf_velox::connector::hive::fetchByteRangesAsync(
          dataSource,
          cudf::host_span<const cudf::io::text::byte_range_info>(
              ranges.data(), ranges.size()),
          stream,
          rmm::mr::get_current_device_resource_ref());

  std::thread releaseThread([&] {
    const auto observed =
        state.firstCopyComplete.try_wait_for(std::chrono::seconds(5));
    state.firstCopyObserved.store(observed);
    state.releaseFirstCopy.post();
  });
  EXPECT_THROW(loadFuture.get(), std::runtime_error);
  releaseThread.join();

  ASSERT_TRUE(state.firstCopyObserved.load());
  ASSERT_EQ(deviceBuffers.size(), 1);
  ASSERT_EQ(deviceSpans.size(), 2);
  std::string actual(firstRange.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      deviceSpans[0].data(),
      actual.size(),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  EXPECT_EQ(actual, firstRange);

  EXPECT_NO_THROW(dataSource->load(stream));
  EXPECT_EQ(state.secondReadAttempts.load(), 1);
}

TEST_F(
    TableScanTest,
    bufferedInputFetchClearsPendingLoadsAfterEnqueueFailure) {
  constexpr size_t kRangeSize = 4 << 10;
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  auto input = std::make_shared<ThrowOnSecondEnqueueBufferedInput>(
      *pool_, kRangeSize);
  auto dataSource = std::make_shared<
      facebook::velox::cudf_velox::connector::hive::
          BufferedInputDataSource>(input);
  std::vector<cudf::io::text::byte_range_info> ranges{
      {0, kRangeSize}, {kRangeSize, kRangeSize}};

  EXPECT_THROW(
      (void)facebook::velox::cudf_velox::connector::hive::
          fetchByteRangesAsync(
              dataSource,
              cudf::host_span<const cudf::io::text::byte_range_info>(
                  ranges.data(), ranges.size()),
              stream,
              rmm::mr::get_current_device_resource_ref()),
      std::runtime_error);
  ASSERT_EQ(input->enqueueAttempts(), 2);
  ASSERT_EQ(input->firstReadAttempts(), 0);

  EXPECT_NO_THROW(dataSource->load(stream));
  EXPECT_EQ(input->firstReadAttempts(), 0);
}

TEST_F(TableScanTest, genericHostFetchWaitsForDeviceCopy) {
  constexpr size_t kDataSize = 4 << 10;
  std::string expected(kDataSize, '\0');
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<char>(i % 251);
  }

  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  auto dataSource = std::make_shared<BlockingHostDataSource>(expected);
  std::vector<cudf::io::text::byte_range_info> ranges{{0, kDataSize}};
  auto [deviceBuffers, deviceSpans, loadFuture] =
      facebook::velox::cudf_velox::connector::hive::fetchByteRangesAsync(
          dataSource,
          cudf::host_span<const cudf::io::text::byte_range_info>(
              ranges.data(), ranges.size()),
          stream,
          rmm::mr::get_current_device_resource_ref());

  folly::Baton<> releaseCopy;
  bool hostReadAllowed = false;
  bool copyReleased = false;
  auto cleanupGuard = folly::makeGuard([&] {
    if (!hostReadAllowed) {
      dataSource->allowHostRead();
    }
    if (!copyReleased) {
      releaseCopy.post();
    }
    stream.synchronize_no_throw();
  });

  EXPECT_TRUE(dataSource->waitForHostRead(std::chrono::seconds(5)));
  CUDF_CUDA_TRY(
      cudaLaunchHostFunc(stream.value(), waitForBaton, &releaseCopy));
  dataSource->allowHostRead();
  hostReadAllowed = true;

  auto completion = std::async(
      std::launch::async,
      [loadFuture = std::move(loadFuture)]() mutable { loadFuture.get(); });
  EXPECT_EQ(
      completion.wait_for(std::chrono::milliseconds(500)),
      std::future_status::timeout);

  releaseCopy.post();
  copyReleased = true;
  EXPECT_NO_THROW(completion.get());

  ASSERT_EQ(deviceBuffers.size(), 1);
  ASSERT_EQ(deviceSpans.size(), 1);
  std::string actual(expected.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      deviceSpans[0].data(),
      actual.size(),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  EXPECT_EQ(actual, expected);
}

TEST_F(
    TableScanTest,
    genericDeviceFetchDrainsStartedReadAfterScheduleFailure) {
  constexpr size_t kRangeSize = 4 << 10;
  rmm::cuda_stream stream(rmm::cuda_stream::flags::non_blocking);
  auto dataSource =
      std::make_shared<GatedDeviceReadDataSource>(kRangeSize);
  std::vector<cudf::io::text::byte_range_info> ranges{
      {0, kRangeSize}, {2 * kRangeSize, kRangeSize}};

  std::future<void> fetchAttempt;
  auto cleanupGuard = folly::makeGuard([&] {
    dataSource->releaseFirstRead();
    if (fetchAttempt.valid()) {
      fetchAttempt.wait();
    }
  });
  fetchAttempt = std::async(std::launch::async, [&] {
    (void)facebook::velox::cudf_velox::connector::hive::
        fetchByteRangesAsync(
            dataSource,
            cudf::host_span<const cudf::io::text::byte_range_info>(
                ranges.data(), ranges.size()),
            stream,
            rmm::mr::get_current_device_resource_ref());
  });

  ASSERT_TRUE(dataSource->waitForFirstRead(std::chrono::seconds(5)));
  ASSERT_TRUE(dataSource->waitForSecondRead(std::chrono::seconds(5)));
  EXPECT_EQ(
      fetchAttempt.wait_for(std::chrono::milliseconds(500)),
      std::future_status::timeout);

  dataSource->releaseFirstRead();
  EXPECT_THROW(fetchAttempt.get(), std::runtime_error);
}

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector});

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(ROW({"c0", "c2"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs = {};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(std::move(config)),
      connectorConfigs,
      nullptr);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(makeCudfHiveConnectorSplits({filePath}))
                  .queryCtx(queryCtx)
                  .assertResults("SELECT c0, c2 FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  // Reduced from 500 to 400 as cudf CudfHive writer seems to be writing smaller
  // files.
  ASSERT_GE(rawInputBytes, 400);

  // TableScan runtime stats not available with CudfHive connector yet
#if 0
  auto overreadBytes =
  getTableScanRuntimeStats(task).at("overreadBytes").sum;
  ASSERT_EQ(overreadBytes, 13);
  ASSERT_EQ(
      getTableScanRuntimeStats(task).at("storageReadBytes").sum,
      rawInputBytes + overreadBytes);
  ASSERT_GT(getTableScanRuntimeStats(task)["totalScanTime"].sum, 0);
  ASSERT_GT(getTableScanRuntimeStats(task)["ioWaitWallNanos"].sum, 0);
#endif
}

TEST_F(TableScanTest, columnAliases) {
  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  std::string tableName = "t";
  std::unordered_map<std::string, std::string> aliases = {{"a", "c0"}};
  auto outputType = ROW({"a"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto op = PlanBuilder(pool_.get())
                .startTableScan()
                .tableHandle(tableHandle)
                .tableName(tableName)
                .outputType(outputType)
                .columnAliases(aliases)
                .endTableScan()
                .planNode();
  assertQuery(op, {filePath}, "SELECT c0 FROM tmp");
}

TEST_F(TableScanTest, filterPushdown) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3"}, {TINYINT(), BIGINT(), DOUBLE(), BOOLEAN()});
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // c1 >= 0 or null and c3 is true
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), true))
          .add("c3", std::make_unique<common::BoolValue>(true, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto task = assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c1", "c3", "c0"}, {BIGINT(), BOOLEAN(), TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0 ) AND c3");

  auto tableScanStats = getTableScanStats(task);
  // EXPECT_EQ(tableScanStats.rawInputRows, 10'000);
  // EXPECT_LT(tableScanStats.inputRows, tableScanStats.rawInputRows);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);

#if 0
  // Repeat the same but do not project out the filtered columns.
  assignments.clear();
  assignments["c0"] =
      facebook::velox::exec::test::HiveConnectorTestBase::regularColumn(
          "c0", TINYINT());
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c0"}, {TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c0 FROM tmp WHERE (c1 >= 0 ) AND c3");

  // TODO: zero column non-empty table is not possible in cudf, need to implement.
  // Do the same for count, no columns projected out.
  assignments.clear();
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp WHERE (c1 >= 0 ) AND c3");

  // Do the same for count, no filter, no projections.
  assignments.clear();
  // subfieldFilters.clear(); // Explicitly clear this.
  tableHandle = makeTableHandle(
      "parquet_table",
      rowType,
      false,
      nullptr,
      nullptr);
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"sum(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp");
#endif
}

// Disable this test and the one below for now, pending a CUDF fix.
// simoneves 2/25/26
// @TODO simoneves/mattgara re-enable once fixed.

TEST_F(TableScanTest, DISABLED_decimalFilterPushdown) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});

  auto vector = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>(
              {123, 500, -250, 300, 400, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {int128_t{200},
               int128_t{200},
               int128_t{700},
               int128_t{700},
               int128_t{900},
               int128_t{-100}},
              DECIMAL(20, 2)),
      });

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // c0 between 1.00 and 4.00 and c1 in (2.00, 7.00)
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{100}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              common::createHugeintValues(
                  {int128_t{200}, int128_t{700}}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 BETWEEN CAST('1.00' AS DECIMAL(12, 2)) "
      "AND CAST('4.00' AS DECIMAL(12, 2)) "
      "AND c1 IN (CAST('2.00' AS DECIMAL(20, 2)), "
      "CAST('7.00' AS DECIMAL(20, 2)))");
}

TEST_F(TableScanTest, DISABLED_decimalStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{1000}, int128_t{2000}}, DECIMAL(20, 2))});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({300, 400}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{3000}, int128_t{4000}}, DECIMAL(20, 2))});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({500, 600}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{5000}, int128_t{6000}}, DECIMAL(20, 2))});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{300}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::HugeintRange>(
                  int128_t{3000}, int128_t{4000}, /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, doubleStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DOUBLE(), DOUBLE()});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({1.0, 2.0}),
       makeFlatVector<double>({10.0, 20.0})});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({3.0, 4.0}),
       makeFlatVector<double>({30.0, 40.0})});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({5.0, 6.0}),
       makeFlatVector<double>({50.0, 60.0})});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::DoubleRange>(
                  3.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  4.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::DoubleRange>(
                  30.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  40.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, splitOffsetAndLength) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // Note that the number of row groups selected within `halfFileSize` may
  // change in the future and this test may start failing. In such a case,
  // just adjust the duckdb sql string accordingly.
  const auto halfFileSize = fs::file_size(filePath->getPath()) / 2;

  // First half of file - OFFSET 0 LIMIT 6000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, halfFileSize),
      "SELECT * FROM tmp OFFSET 0 LIMIT 6000");

  // Second half of file - OFFSET 6000 LIMIT 4000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), halfFileSize),
      "SELECT * FROM tmp OFFSET 6000 LIMIT 4000");

  const auto fileSize = fs::file_size(filePath->getPath());

  // All row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, fileSize),
      "SELECT * FROM tmp");

  // No row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), fileSize),
      "SELECT * FROM tmp LIMIT 0");
}

// Verify that extractFiltersFromRemainingFilter extracts simple single-column
// filters from the remaining filter into subfield filters for pushdown.
// When a filter like "c0 = 1" is fully extracted, remainingFilterExprSet_ is
// null and totalRemainingFilterWallNanos is 0. Without extraction, the filter
// runs post-read on the GPU and the stat is > 0.
TEST_F(TableScanTest, remainingFilterExtraction) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  // "c0 = 1" is a single-column equality that should be fully extracted into
  // a subfield filter, leaving no remaining filter to evaluate post-read.
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = 1")
                  .endTableScan()
                  .planNode();

  auto task = assertQuery(plan, {filePath}, "SELECT * FROM tmp WHERE c0 = 1");

  // Verify the filter was fully extracted: no post-read remaining filter ran.
  auto planStats = toPlanStats(task->taskStats());
  const auto& scanStats = planStats.at(plan->id());
  auto it = scanStats.customStats.find("totalRemainingFilterWallNanos");
  ASSERT_NE(it, scanStats.customStats.end());
  EXPECT_EQ(it->second.sum, 0)
      << "Expected no remaining filter time when filter is fully extracted";
}

TEST_F(TableScanTest, decimalSubfieldFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{-500}, int64_t{-500}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);
  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, decimalRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = CAST('-5.00' AS DECIMAL(5, 2))")
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}
