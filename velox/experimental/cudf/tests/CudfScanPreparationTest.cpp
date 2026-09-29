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

#include "velox/experimental/cudf/connectors/hive/CudfScanPreparation.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfPreparedSplitsStore.h"

#include "velox/common/caching/FileIds.h"
#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <folly/OperationCancelled.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;

class ShortReadFile final : public InMemoryReadFile {
 public:
  explicit ShortReadFile(std::string content)
      : InMemoryReadFile(std::move(content)) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context = {}) const override {
    const auto data = InMemoryReadFile::pread(offset, length, buffer, context);
    return data.empty() ? data : data.substr(0, data.size() - 1);
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context = {}) const override {
    const auto bytes = ReadFile::preadv(offset, buffers, context);
    return bytes == 0 ? 0 : bytes - 1;
  }
};

class CancelOnReadFile final : public InMemoryReadFile {
 public:
  CancelOnReadFile(std::string content, folly::CancellationSource source)
      : InMemoryReadFile(std::move(content)), source_(std::move(source)) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context = {}) const override {
    auto data = InMemoryReadFile::pread(offset, length, buffer, context);
    source_.requestCancellation();
    return data;
  }

 private:
  folly::CancellationSource source_;
};

std::unique_ptr<cudf::column> makeIntColumn(
    const std::vector<int32_t>& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  rmm::device_buffer data(values.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      data.size(),
      cudaMemcpyHostToDevice,
      stream.value()));
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(values.size()),
      std::move(data),
      rmm::device_buffer{},
      0);
}

std::vector<char> writeParquetFile(
    int32_t valueBase = 0,
    int32_t sourceMarker = 7) {
  constexpr cudf::size_type kRows = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  std::vector<int32_t> values(kRows);
  std::iota(values.begin(), values.end(), valueBase);
  std::vector<int32_t> sources(kRows, sourceMarker);

  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(makeIntColumn(values, stream, mr));
  columns.emplace_back(makeIntColumn(sources, stream, mr));
  cudf::table table{std::move(columns)};

  cudf::io::table_input_metadata metadata(table.view());
  metadata.column_metadata[0].set_name("value");
  metadata.column_metadata[1].set_name("source");
  std::vector<char> output;
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{&output}, table.view())
                     .metadata(std::move(metadata))
                     .row_group_size_rows(kRowsPerGroup)
                     .max_page_size_rows(kRowsPerGroup / 4)
                     .compression(cudf::io::compression_type::SNAPPY)
                     .build();
  cudf::io::write_parquet(options);
  return output;
}

std::vector<int32_t> copyIntColumn(
    const cudf::column_view& column,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(
      column.type().id() == cudf::type_id::INT32, "Expected INT32 test column");
  std::vector<int32_t> values(column.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      values.data(),
      column.data<int32_t>(),
      values.size() * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return values;
}

class CudfScanPreparationTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    allocator_ = std::make_shared<memory::MallocAllocator>(
        memory::MemoryAllocator::Options{
            .capacity = 64 << 20, .reservationByteLimit = 0});
    cache_ = cache::AsyncDataCache::create(allocator_.get());
  }

  void TearDown() override {
    cache_->shutdown();
    cache_.reset();
    allocator_.reset();
    pool_.reset();
  }

  std::shared_ptr<BufferedInputDataSource> makeSource(
      std::shared_ptr<ReadFile> readFile,
      uint64_t loadQuantum = 1024) {
    const auto id = nextId_.fetch_add(1);
    auto ioStats = std::make_shared<io::IoStatistics>();
    io::ReaderOptions options(pool_.get());
    options.setDataIoStats(ioStats);
    options.setMetadataIoStats(ioStats);
    options.setLoadQuantum(loadQuantum);
    options.setCacheable(true);
    auto& ids = fileIds();
    auto input = std::make_shared<dwio::common::CachedBufferedInput>(
        std::move(readFile),
        dwio::common::MetricsLog::voidLog(),
        StringIdLease(ids, "strong-ready-file-" + std::to_string(id)),
        cache_.get(),
        nullptr,
        StringIdLease(ids, "strong-ready-group-" + std::to_string(id)),
        ioStats,
        nullptr,
        nullptr,
        options);
    return std::make_shared<BufferedInputDataSource>(std::move(input));
  }

  static std::atomic<uint64_t> nextId_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MallocAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
};

std::atomic<uint64_t> CudfScanPreparationTest::nextId_{0};

TEST_F(CudfScanPreparationTest, plansIndependentRowGroupRanges) {
  auto file = writeParquetFile();
  auto sourceInfo = cudf::io::source_info{
      cudf::host_span<const char>{file.data(), file.size()}};
  auto sources = cudf::io::make_datasources(sourceInfo);
  ASSERT_EQ(sources.size(), 1);
  std::shared_ptr<cudf::io::datasource> source = std::move(sources.front());
  auto options = cudf::io::parquet_reader_options::builder().build();

  auto plan =
      planParquetFileRanges(source, options, cudf::get_default_stream());
  EXPECT_EQ(plan.metadata.num_rows, 60'000);
  ASSERT_EQ(plan.rowGroups.size(), 3);
  for (cudf::size_type i = 0; i < 3; ++i) {
    const auto& rowGroup = plan.rowGroups[i];
    EXPECT_EQ(rowGroup.rowGroup, i);
    EXPECT_EQ(rowGroup.inputRows, 20'000);
    EXPECT_GT(rowGroup.logicalBytes, 0);
    ASSERT_FALSE(rowGroup.ranges.empty());
    for (const auto& range : rowGroup.ranges) {
      EXPECT_GE(range.offset(), 0);
      EXPECT_GT(range.size(), 0);
      EXPECT_LE(
          static_cast<uint64_t>(range.offset() + range.size()), file.size());
    }
  }
  EXPECT_NE(
      plan.rowGroups[0].ranges.front().offset(),
      plan.rowGroups[2].ranges.front().offset());
}

TEST_F(CudfScanPreparationTest, leaseRetainsExactBytesUnderEvictionPressure) {
  std::string content(300 * 1024, '\0');
  for (size_t i = 0; i < content.size(); ++i) {
    content[i] = static_cast<char>(i % 251);
  }
  auto source = makeSource(std::make_shared<InMemoryReadFile>(content));
  const std::vector<cudf::io::text::byte_range_info> ranges{
      {137, 2500}, {4096 + 17, 1800}, {8192, 245835}};
  HostRangeAdmission admission(1 << 20, 2);

  auto lease = source->prepareHostRangeLease(ranges, admission);
  ASSERT_NE(lease, nullptr);
  ASSERT_EQ(lease->ranges().size(), ranges.size());
  EXPECT_EQ(lease->logicalBytes(), 250135);
  EXPECT_GT(lease->retainedBytes(), 0);
  EXPECT_LE(lease->retainedBytes(), lease->reservedBytes());
  auto snapshot = admission.snapshot();
  EXPECT_EQ(snapshot.currentBytes, lease->reservedBytes());
  EXPECT_EQ(snapshot.currentUnits, 1);

  std::vector<uint8_t> first(2500);
  std::vector<uint8_t> second(1800);
  lease->copyRange(0, first.data(), first.size());
  lease->copyRange(1, second.data(), second.size());
  EXPECT_EQ(
      std::string_view(reinterpret_cast<char*>(first.data()), first.size()),
      std::string_view(content).substr(137, first.size()));
  EXPECT_EQ(
      std::string_view(reinterpret_cast<char*>(second.data()), second.size()),
      std::string_view(content).substr(4096 + 17, second.size()));

  const auto pinnedStats = cache_->refreshStats();
  EXPECT_GT(pinnedStats.sharedPinnedBytes, 0);
  EXPECT_EQ(cache_->shrink(64 << 20), 0);
  EXPECT_EQ(cache_->refreshStats().numEntries, pinnedStats.numEntries);
  lease->copyRange(0, first.data(), first.size());

  lease.reset();
  snapshot = admission.snapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.currentUnits, 0);
  EXPECT_GT(cache_->shrink(64 << 20), 0);
  EXPECT_EQ(cache_->refreshStats().numEntries, 0);
}

TEST_F(CudfScanPreparationTest, regularReaderUsesOnlyPreparedLeaseSources) {
  auto firstFile = writeParquetFile(0, 7);
  auto secondFile = writeParquetFile(100'000, 9);
  auto firstReadFile = std::make_shared<tests::utils::CountingReadFile>(
      std::string(firstFile.begin(), firstFile.end()));
  auto secondReadFile = std::make_shared<tests::utils::CountingReadFile>(
      std::string(secondFile.begin(), secondFile.end()));
  auto firstInput = makeSource(firstReadFile);
  auto secondInput = makeSource(secondReadFile);
  HostRangeAdmission admission(64 << 20, 2);

  auto firstLease =
      std::shared_ptr<HostRangeLease>(firstInput->prepareHostRangeLease(
          {{0, static_cast<int64_t>(firstFile.size())}}, admission));
  auto secondLease =
      std::shared_ptr<HostRangeLease>(secondInput->prepareHostRangeLease(
          {{0, static_cast<int64_t>(secondFile.size())}}, admission));
  auto firstSource =
      std::make_shared<HostRangeLeaseDataSource>(firstLease, firstFile.size());
  auto secondSource = std::make_shared<HostRangeLeaseDataSource>(
      secondLease, secondFile.size());
  const auto firstReadsAfterPreparation = firstReadFile->numReads();
  const auto secondReadsAfterPreparation = secondReadFile->numReads();

  std::vector<cudf::io::datasource*> sources{
      firstSource.get(), secondSource.get()};
  auto options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{sources})
          .column_names({"value", "source"})
          .build();
  auto result = cudf::io::read_parquet(options);
  ASSERT_NE(result.tbl, nullptr);
  ASSERT_EQ(result.tbl->num_columns(), 2);
  ASSERT_EQ(result.tbl->num_rows(), 120'000);

  const auto stream = cudf::get_default_stream();
  const auto values = copyIntColumn(result.tbl->view().column(0), stream);
  const auto sourceMarkers =
      copyIntColumn(result.tbl->view().column(1), stream);
  ASSERT_EQ(values.size(), 120'000);
  EXPECT_EQ(values.front(), 0);
  EXPECT_EQ(values[59'999], 59'999);
  EXPECT_EQ(values[60'000], 100'000);
  EXPECT_EQ(values.back(), 159'999);
  EXPECT_TRUE(
      std::all_of(
          sourceMarkers.begin(),
          sourceMarkers.begin() + 60'000,
          [](int32_t value) { return value == 7; }));
  EXPECT_TRUE(
      std::all_of(
          sourceMarkers.begin() + 60'000,
          sourceMarkers.end(),
          [](int32_t value) { return value == 9; }));

  EXPECT_GT(firstSource->hostReadCalls(), 0);
  EXPECT_GT(secondSource->hostReadCalls(), 0);
  EXPECT_EQ(firstReadFile->numReads(), firstReadsAfterPreparation);
  EXPECT_EQ(secondReadFile->numReads(), secondReadsAfterPreparation);

  result.tbl.reset();
  firstSource.reset();
  secondSource.reset();
  firstLease.reset();
  secondLease.reset();
  const auto snapshot = admission.snapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.currentUnits, 0);
}

TEST_F(CudfScanPreparationTest, hybridReaderUsesOnlyPreparedRowGroupLeases) {
  auto firstFile = writeParquetFile(0, 7);
  auto secondFile = writeParquetFile(100'000, 9);
  auto firstReadFile = std::make_shared<tests::utils::CountingReadFile>(
      std::string(firstFile.begin(), firstFile.end()));
  auto secondReadFile = std::make_shared<tests::utils::CountingReadFile>(
      std::string(secondFile.begin(), secondFile.end()));
  auto firstInput = makeSource(firstReadFile);
  auto secondInput = makeSource(secondReadFile);
  auto options = cudf::io::parquet_reader_options::builder().build();
  options.set_column_names({"value", "source"});
  auto firstPlan =
      planParquetFileRanges(firstInput, options, cudf::get_default_stream());
  auto secondPlan =
      planParquetFileRanges(secondInput, options, cudf::get_default_stream());
  ASSERT_EQ(firstPlan.rowGroups.size(), 3);
  ASSERT_EQ(secondPlan.rowGroups.size(), 3);

  HostRangeAdmission admission(64 << 20, 2);
  auto firstLease =
      std::shared_ptr<HostRangeLease>(firstInput->prepareHostRangeLease(
          firstPlan.rowGroups[2].ranges, admission));
  auto secondLease =
      std::shared_ptr<HostRangeLease>(secondInput->prepareHostRangeLease(
          secondPlan.rowGroups[1].ranges, admission));
  auto firstSource =
      std::make_shared<HostRangeLeaseDataSource>(firstLease, firstFile.size());
  auto secondSource = std::make_shared<HostRangeLeaseDataSource>(
      secondLease, secondFile.size());
  const auto firstReadsAfterPreparation = firstReadFile->numReads();
  const auto secondReadsAfterPreparation = secondReadFile->numReads();

  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.push_back(std::move(firstPlan.metadata));
  metadatas.push_back(std::move(secondPlan.metadata));
  cudf::io::parquet::experimental::hybrid_scan_multifile reader{
      cudf::host_span<const cudf::io::parquet::FileMetaData>{metadatas},
      options};
  const std::vector<std::vector<cudf::size_type>> selected{{2}, {1}};
  const auto [ranges, sourceIndices] =
      reader.all_column_chunks_byte_ranges(selected, options);
  ASSERT_EQ(ranges.size(), sourceIndices.size());
  ASSERT_FALSE(ranges.empty());
  std::vector<std::reference_wrapper<cudf::io::datasource>> sources{
      *firstSource, *secondSource};
  std::vector<std::vector<cudf::io::text::byte_range_info>> perSource(2);
  for (size_t i = 0; i < ranges.size(); ++i) {
    ASSERT_GE(sourceIndices[i], 0);
    ASSERT_LT(sourceIndices[i], 2);
    perSource[sourceIndices[i]].push_back(ranges[i]);
  }
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto [buffers, perSourceSpans, reads] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          sources, perSource, stream, mr);
  reads.get();
  EXPECT_EQ(firstReadFile->numReads(), firstReadsAfterPreparation);
  EXPECT_EQ(secondReadFile->numReads(), secondReadsAfterPreparation);
  EXPECT_GT(firstSource->hostReadCalls(), 0);
  EXPECT_GT(secondSource->hostReadCalls(), 0);

  std::vector<cudf::device_span<const uint8_t>> spans;
  spans.reserve(ranges.size());
  for (const auto& sourceSpans : perSourceSpans) {
    spans.insert(spans.end(), sourceSpans.begin(), sourceSpans.end());
  }
  ASSERT_EQ(spans.size(), ranges.size());
  reader.setup_chunking_for_all_columns(
      20'000 * 2 * sizeof(int32_t), 0, selected, spans, options, stream, mr);
  std::vector<std::unique_ptr<cudf::table>> chunks;
  while (reader.has_next_table_chunk()) {
    chunks.push_back(reader.materialize_all_columns_chunk().tbl);
  }
  ASSERT_GE(chunks.size(), 2);
  std::vector<cudf::table_view> views;
  views.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    views.push_back(chunk->view());
  }
  auto output = cudf::concatenate(views, stream, mr);
  ASSERT_EQ(output->num_rows(), 40'000);
  const auto values = copyIntColumn(output->view().column(0), stream);
  const auto sourceMarkers = copyIntColumn(output->view().column(1), stream);
  std::vector<int32_t> expectedValues(40'000);
  std::iota(expectedValues.begin(), expectedValues.begin() + 20'000, 40'000);
  std::iota(expectedValues.begin() + 20'000, expectedValues.end(), 120'000);
  EXPECT_EQ(values, expectedValues);
  EXPECT_TRUE(
      std::all_of(
          sourceMarkers.begin(),
          sourceMarkers.begin() + 20'000,
          [](int32_t value) { return value == 7; }));
  EXPECT_TRUE(
      std::all_of(
          sourceMarkers.begin() + 20'000,
          sourceMarkers.end(),
          [](int32_t value) { return value == 9; }));

  output.reset();
  chunks.clear();
  firstSource.reset();
  secondSource.reset();
  firstLease.reset();
  secondLease.reset();
  EXPECT_EQ(admission.snapshot().currentBytes, 0);
  EXPECT_EQ(admission.snapshot().currentUnits, 0);
}

TEST_F(CudfScanPreparationTest, generatedStoreClaimsCurrentlyReadyFiles) {
  using iceberg::CudfPreparedIcebergFile;
  using iceberg::CudfPreparedIcebergSplit;
  using iceberg::CudfPreparedSplitSource;
  using iceberg::CudfPreparedSplitsStore;
  using HiveIcebergSplit =
      ::facebook::velox::connector::hive::iceberg::HiveIcebergSplit;

  folly::CPUThreadPoolExecutor executor(1);
  auto task = exec::Task::create(
      "prepared-store-task",
      exec::test::PlanBuilder()
          .tableScan(ROW({"value"}, {INTEGER()}))
          .planFragment(),
      0,
      core::QueryCtx::create(&executor),
      exec::Task::ExecutionMode::kSerial,
      exec::Consumer{});
  auto source = std::make_shared<CudfPreparedSplitSource>(task, "0", &executor);
  task->setSplitsStore(
      "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1024));

  std::array<std::vector<std::byte>, 4> contents{
      std::vector<std::byte>(100),
      std::vector<std::byte>(100),
      std::vector<std::byte>(100),
      std::vector<std::byte>(100)};
  auto makeFile = [&](uint64_t ordinal) {
    auto descriptor = std::make_shared<HiveIcebergSplit>(
        "cudf-iceberg",
        fmt::format("file:/tmp/{}", ordinal),
        dwio::common::FileFormat::PARQUET,
        0,
        contents[ordinal].size(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        true,
        std::vector<
            ::facebook::velox::connector::hive::iceberg::IcebergDeleteFile>{},
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        0,
        std::vector<::facebook::velox::connector::hive::iceberg::
                        IcebergCoalescedFile>{});
    auto dataSource =
        std::shared_ptr<cudf::io::datasource>(cudf::io::datasource::create(
            cudf::host_span<const std::byte>{
                contents[ordinal].data(), contents[ordinal].size()}));
    return CudfPreparedIcebergFile{
        ordinal, descriptor, std::move(dataSource), contents[ordinal].size()};
  };

  ASSERT_TRUE(source->publishPrepared(makeFile(0)));
  ASSERT_TRUE(source->publishPrepared(makeFile(2)));

  exec::Split split;
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          0, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  auto prepared =
      std::dynamic_pointer_cast<CudfPreparedIcebergSplit>(split.connectorSplit);
  ASSERT_NE(prepared, nullptr);
  ASSERT_EQ(prepared->files().size(), 2);
  EXPECT_EQ(prepared->files()[0].descriptorOrdinal, 0);
  EXPECT_EQ(prepared->files()[1].descriptorOrdinal, 2);
  task->splitFinished(true, split.connectorSplit->splitWeight);

  ASSERT_TRUE(source->publishPrepared(makeFile(1)));
  ASSERT_TRUE(source->publishPrepared(makeFile(3)));
  split = exec::Split{};
  future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          1, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  prepared =
      std::dynamic_pointer_cast<CudfPreparedIcebergSplit>(split.connectorSplit);
  ASSERT_NE(prepared, nullptr);
  ASSERT_EQ(prepared->files().size(), 2);
  EXPECT_EQ(prepared->files()[0].descriptorOrdinal, 1);
  EXPECT_EQ(prepared->files()[1].descriptorOrdinal, 3);
  task->splitFinished(true, split.connectorSplit->splitWeight);

  const auto stats = task->taskStats();
  EXPECT_EQ(stats.numTotalSplits, 2);
  EXPECT_EQ(stats.numFinishedSplits, 2);
  EXPECT_EQ(stats.numQueuedSplits, 0);
  EXPECT_EQ(stats.numRunningSplits, 0);

  split = exec::Split{};
  future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          0, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kWaitForSplit);
  source->finish();
  EXPECT_TRUE(std::move(future).wait(5s));

  split = exec::Split{};
  future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          0, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  EXPECT_FALSE(split.hasConnectorSplit());
  task->requestCancel().wait();
}

TEST_F(
    CudfScanPreparationTest,
    generatedStoreLimitsDistinctSourcesNotRowGroupUnits) {
  using iceberg::CudfPreparedIcebergFile;
  using iceberg::CudfPreparedIcebergSplit;
  using iceberg::CudfPreparedSplitSource;
  using iceberg::CudfPreparedSplitsStore;
  using HiveIcebergSplit =
      ::facebook::velox::connector::hive::iceberg::HiveIcebergSplit;

  folly::CPUThreadPoolExecutor executor(1);
  auto task = exec::Task::create(
      "prepared-row-group-store-task",
      exec::test::PlanBuilder()
          .tableScan(ROW({"value"}, {INTEGER()}))
          .planFragment(),
      0,
      core::QueryCtx::create(&executor),
      exec::Task::ExecutionMode::kSerial,
      exec::Consumer{});
  auto source = std::make_shared<CudfPreparedSplitSource>(task, "0", &executor);
  task->setSplitsStore(
      "0", std::make_unique<CudfPreparedSplitsStore>(source, 1, 1024));

  std::array<std::vector<std::byte>, 2> contents{
      std::vector<std::byte>(100), std::vector<std::byte>(100)};
  std::array<std::shared_ptr<HiveIcebergSplit>, 2> descriptors;
  for (size_t sourceIndex = 0; sourceIndex < descriptors.size();
       ++sourceIndex) {
    descriptors[sourceIndex] = std::make_shared<HiveIcebergSplit>(
        "cudf-iceberg",
        fmt::format("file:/tmp/row-group-source-{}", sourceIndex),
        dwio::common::FileFormat::PARQUET,
        0,
        contents[sourceIndex].size(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        true,
        std::vector<
            ::facebook::velox::connector::hive::iceberg::IcebergDeleteFile>{},
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        0,
        std::vector<::facebook::velox::connector::hive::iceberg::
                        IcebergCoalescedFile>{});
  }
  const auto metadata =
      std::make_shared<const cudf::io::parquet::FileMetaData>();
  const auto makeUnit = [&](uint64_t sourceIndex, cudf::size_type rowGroup) {
    auto dataSource =
        std::shared_ptr<cudf::io::datasource>(cudf::io::datasource::create(
            cudf::host_span<const std::byte>{
                contents[sourceIndex].data(), contents[sourceIndex].size()}));
    return CudfPreparedIcebergFile{
        sourceIndex,
        descriptors[sourceIndex],
        std::move(dataSource),
        contents[sourceIndex].size(),
        metadata,
        rowGroup,
        100};
  };

  ASSERT_TRUE(source->publishPrepared(makeUnit(0, 0)));
  ASSERT_TRUE(source->publishPrepared(makeUnit(0, 1)));
  ASSERT_TRUE(source->publishPrepared(makeUnit(0, 2)));
  ASSERT_TRUE(source->publishPrepared(makeUnit(1, 0)));

  exec::Split split;
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          0, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  auto prepared =
      std::dynamic_pointer_cast<CudfPreparedIcebergSplit>(split.connectorSplit);
  ASSERT_NE(prepared, nullptr);
  ASSERT_EQ(prepared->files().size(), 3);
  EXPECT_EQ(prepared->sourceCount(), 1);
  EXPECT_EQ(prepared->files()[0].rowGroup, 0);
  EXPECT_EQ(prepared->files()[1].rowGroup, 1);
  EXPECT_EQ(prepared->files()[2].rowGroup, 2);
  task->requestCancel().wait();
}

TEST_F(CudfScanPreparationTest, generatedStoreDelaysBarrierUntilSourceDrains) {
  using iceberg::CudfPreparedIcebergFile;
  using iceberg::CudfPreparedSplitSource;
  using iceberg::CudfPreparedSplitsStore;
  using HiveIcebergSplit =
      ::facebook::velox::connector::hive::iceberg::HiveIcebergSplit;

  folly::CPUThreadPoolExecutor executor(1);
  auto task = exec::Task::create(
      "prepared-store-barrier-task",
      exec::test::PlanBuilder()
          .tableScan(ROW({"value"}, {INTEGER()}))
          .planFragment(),
      0,
      core::QueryCtx::create(&executor),
      exec::Task::ExecutionMode::kSerial,
      exec::Consumer{});
  auto source = std::make_shared<CudfPreparedSplitSource>(task, "0", &executor);
  auto store = std::make_unique<CudfPreparedSplitsStore>(source, 1, 1024);
  auto* storePtr = store.get();
  task->setSplitsStore("0", std::move(store));

  std::array<std::vector<std::byte>, 2> contents{
      std::vector<std::byte>(100), std::vector<std::byte>(100)};
  const auto makeFile = [&](uint64_t ordinal) {
    auto descriptor = std::make_shared<HiveIcebergSplit>(
        "cudf-iceberg",
        fmt::format("file:/tmp/barrier-{}", ordinal),
        dwio::common::FileFormat::PARQUET,
        0,
        contents[ordinal].size(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        true,
        std::vector<
            ::facebook::velox::connector::hive::iceberg::IcebergDeleteFile>{},
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        0,
        std::vector<::facebook::velox::connector::hive::iceberg::
                        IcebergCoalescedFile>{});
    auto dataSource =
        std::shared_ptr<cudf::io::datasource>(cudf::io::datasource::create(
            cudf::host_span<const std::byte>{
                contents[ordinal].data(), contents[ordinal].size()}));
    return CudfPreparedIcebergFile{
        ordinal, descriptor, std::move(dataSource), contents[ordinal].size()};
  };

  ASSERT_TRUE(source->publishPrepared(makeFile(0)));
  std::vector<ContinuePromise> barrierWaiters;
  storePtr->requestBarrier(2, barrierWaiters);
  EXPECT_TRUE(barrierWaiters.empty());

  exec::Split split;
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          0, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  ASSERT_TRUE(split.hasConnectorSplit());
  EXPECT_FALSE(split.isBarrier());
  task->splitFinished(true, split.connectorSplit->splitWeight);

  split = exec::Split{};
  future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          1, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kWaitForSplit);
  ASSERT_TRUE(source->publishPrepared(makeFile(1)));
  EXPECT_TRUE(std::move(future).wait(5s));

  split = exec::Split{};
  future = ContinueFuture::makeEmpty();
  EXPECT_EQ(
      task->getSplitOrFuture(
          1, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
      exec::BlockingReason::kNotBlocked);
  ASSERT_TRUE(split.hasConnectorSplit());
  EXPECT_FALSE(split.isBarrier());
  task->splitFinished(true, split.connectorSplit->splitWeight);

  source->finish();
  for (uint32_t driver = 0; driver < 2; ++driver) {
    split = exec::Split{};
    future = ContinueFuture::makeEmpty();
    EXPECT_EQ(
        task->getSplitOrFuture(
            driver, exec::kUngroupedGroupId, "0", 0, nullptr, split, future),
        exec::BlockingReason::kNotBlocked);
    EXPECT_TRUE(split.isBarrier());
  }
  task->requestCancel().wait();
}

TEST_F(CudfScanPreparationTest, generatedStoreReleasesReadyWorkOnCancel) {
  using iceberg::CudfPreparedSplitSource;
  using iceberg::CudfPreparedSplitsStore;
  folly::CPUThreadPoolExecutor executor(1);
  auto task = exec::Task::create(
      "prepared-store-cancel-task",
      exec::test::PlanBuilder()
          .tableScan(ROW({"value"}, {INTEGER()}))
          .planFragment(),
      0,
      core::QueryCtx::create(&executor),
      exec::Task::ExecutionMode::kSerial,
      exec::Consumer{});
  auto source = std::make_shared<CudfPreparedSplitSource>(task, "0", &executor);
  task->setSplitsStore(
      "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1024));

  task->requestCancel().wait();
  EXPECT_EQ(task->state(), exec::TaskState::kCanceled);
  EXPECT_FALSE(source->publishFallback(
      std::make_shared<::facebook::velox::connector::hive::HiveConnectorSplit>(
          "hive", "file:/tmp/cancelled", dwio::common::FileFormat::PARQUET)));
}

TEST_F(CudfScanPreparationTest, oversizedUnitFallsBackBeforeReading) {
  std::string content(8 * 1024, 'x');
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);
  auto source = makeSource(readFile);
  HostRangeAdmission admission(4096, 1);

  EXPECT_EQ(source->prepareHostRangeLease({{0, 5000}}, admission), nullptr);
  EXPECT_EQ(readFile->numReads(), 0);
  EXPECT_EQ(admission.snapshot().currentBytes, 0);
}

TEST_F(CudfScanPreparationTest, shortReadPublishesNoLease) {
  auto source =
      makeSource(std::make_shared<ShortReadFile>(std::string(4096, 'x')));
  HostRangeAdmission admission(64 << 10, 1);

  EXPECT_ANY_THROW(source->prepareHostRangeLease({{100, 2500}}, admission));
  EXPECT_EQ(admission.snapshot().currentBytes, 0);
  EXPECT_EQ(admission.snapshot().currentUnits, 0);
}

TEST_F(CudfScanPreparationTest, cancellationAfterReadPublishesNoLease) {
  folly::CancellationSource cancellation;
  auto source = makeSource(
      std::make_shared<CancelOnReadFile>(std::string(4096, 'x'), cancellation));
  HostRangeAdmission admission(64 << 10, 1);

  EXPECT_THROW(
      source->prepareHostRangeLease(
          {{100, 1000}}, admission, cancellation.getToken()),
      folly::OperationCancelled);
  EXPECT_EQ(admission.snapshot().currentBytes, 0);
  EXPECT_EQ(admission.snapshot().currentUnits, 0);
}

TEST(HostRangeAdmissionTest, cancellationWakesBlockedReservation) {
  HostRangeAdmission admission(4096, 1);
  auto first = admission.reserve(4096, 1);
  folly::CancellationSource cancellation;
  std::promise<void> started;
  auto waiting = std::async(std::launch::async, [&]() {
    started.set_value();
    return admission.reserve(1, 1, cancellation.getToken());
  });
  started.get_future().wait();
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (admission.snapshot().waiters != 1 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(admission.snapshot().waiters, 1);
  EXPECT_EQ(waiting.wait_for(0s), std::future_status::timeout);

  cancellation.requestCancellation();
  ASSERT_EQ(waiting.wait_for(5s), std::future_status::ready);
  EXPECT_THROW(waiting.get(), folly::OperationCancelled);
  auto snapshot = admission.snapshot();
  EXPECT_EQ(snapshot.currentBytes, 4096);
  EXPECT_EQ(snapshot.currentUnits, 1);
  EXPECT_EQ(snapshot.waiters, 0);

  first.reset();
  snapshot = admission.snapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.currentUnits, 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
