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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorReadBroker.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"

#include "velox/common/caching/FileIds.h"
#include "velox/common/file/tests/TestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/dwio/common/CachedBufferedInput.h"

#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_stream.hpp>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef VELOX_ENABLE_S3
extern "C" bool glutenCrtS3RangeReaderAvailable() {
  return false;
}

extern "C" uint64_t glutenCrtS3ObjectSize(const char*) {
  return 0;
}

extern "C" uint64_t glutenCrtS3ReadRanges(
    const char*,
    uint8_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    size_t) {
  return 0;
}
#endif

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;

class CacheHintRangeStatsTest : public testing::Test {
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

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<memory::MallocAllocator> allocator_;
  std::shared_ptr<cache::AsyncDataCache> cache_;
};

class TrackingArrayInputStream final
    : public dwio::common::SeekableArrayInputStream {
 public:
  TrackingArrayInputStream(
      const char* data,
      uint64_t size,
      uint64_t blockSize,
      std::shared_ptr<std::atomic<uint64_t>> backedUpBytes,
      std::shared_ptr<std::atomic<bool>> destroyed)
      : SeekableArrayInputStream(data, size, blockSize),
        backedUpBytes_(std::move(backedUpBytes)),
        destroyed_(std::move(destroyed)) {}

  ~TrackingArrayInputStream() override {
    destroyed_->store(true, std::memory_order_release);
  }

  void BackUp(int32_t count) override {
    backedUpBytes_->fetch_add(count, std::memory_order_relaxed);
    SeekableArrayInputStream::BackUp(count);
  }

 private:
  const std::shared_ptr<std::atomic<uint64_t>> backedUpBytes_;
  const std::shared_ptr<std::atomic<bool>> destroyed_;
};

class FakeCachedBufferedInput final : public dwio::common::BufferedInput {
 public:
  FakeCachedBufferedInput(
      std::string content,
      memory::MemoryPool& pool,
      folly::Executor* executor,
      uint64_t blockSize,
      std::shared_ptr<std::atomic<uint64_t>> backedUpBytes,
      std::shared_ptr<std::atomic<bool>> streamDestroyed)
      : BufferedInput(std::make_shared<InMemoryReadFile>(content), pool),
        content_(std::move(content)),
        executor_(executor),
        blockSize_(blockSize),
        backedUpBytes_(std::move(backedUpBytes)),
        streamDestroyed_(std::move(streamDestroyed)) {}

  bool hasCache() const override {
    return true;
  }

  folly::Executor* executor() const override {
    return executor_;
  }

  std::unique_ptr<dwio::common::SeekableInputStream> read(
      uint64_t offset,
      uint64_t /*length*/,
      dwio::common::LogType /*logType*/) const override {
    VELOX_CHECK_LT(offset, content_.size());
    return std::make_unique<TrackingArrayInputStream>(
        content_.data() + offset,
        content_.size() - offset,
        blockSize_,
        backedUpBytes_,
        streamDestroyed_);
  }

  const char* contentBegin() const {
    return content_.data();
  }

  const char* contentEnd() const {
    return content_.data() + content_.size();
  }

 private:
  const std::string content_;
  folly::Executor* const executor_;
  const uint64_t blockSize_;
  const std::shared_ptr<std::atomic<uint64_t>> backedUpBytes_;
  const std::shared_ptr<std::atomic<bool>> streamDestroyed_;
};

class ObservedBufferedDataSource : public BufferedInputDataSource {
 public:
  ObservedBufferedDataSource(
      std::shared_ptr<dwio::common::BufferedInput> input,
      std::function<void(size_t)> onRead)
      : BufferedInputDataSource(std::move(input)), onRead_(std::move(onRead)) {}

  size_t host_read(size_t offset, size_t size, uint8_t* destination) override {
    onRead_(++reads_);
    return BufferedInputDataSource::host_read(offset, size, destination);
  }

 private:
  std::function<void(size_t)> onRead_;
  size_t reads_{0};
};

// Hold the CUDA stream without blocking the submitting CPU. Enqueue only
// after the staging allocations, which may themselves synchronize CUDA.
class H2dTestStreamGate {
 public:
  void enqueue(rmm::cuda_stream_view stream) {
    CUDF_CUDA_TRY(cudaLaunchHostFunc(
        stream.value(),
        [](void* opaque) {
          auto& gate = *static_cast<H2dTestStreamGate*>(opaque);
          std::unique_lock lock(gate.mutex_);
          gate.cv_.wait(lock, [&] { return gate.released_; });
        },
        this));
  }

  void release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool released_{false};
};

class HybridH2dStagingTest : public CacheHintRangeStatsTest {
 protected:
  void SetUp() override {
    CacheHintRangeStatsTest::SetUp();
    if (const auto* value = std::getenv(kSetting)) {
      previous_ = value;
    }
    if (const auto* value = std::getenv(kPoolSetting)) {
      previousPool_ = value;
    }
    ASSERT_EQ(setenv(kSetting, "16", 1), 0);
    // Returning a slot to the pool does not synchronize CUDA. Otherwise
    // cudaFreeHost could hide a missing completion wait in the lifetime tests.
    ASSERT_EQ(setenv(kPoolSetting, "1", 1), 0);
  }

  void TearDown() override {
    if (previous_) {
      setenv(kSetting, previous_->c_str(), 1);
    } else {
      unsetenv(kSetting);
    }
    if (previousPool_) {
      setenv(kPoolSetting, previousPool_->c_str(), 1);
    } else {
      unsetenv(kPoolSetting);
    }
    CacheHintRangeStatsTest::TearDown();
  }

  std::shared_ptr<ObservedBufferedDataSource> source(
      const std::string& content,
      std::function<void(size_t)> onRead = [](size_t) {}) {
    return std::make_shared<ObservedBufferedDataSource>(
        std::make_shared<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(content), *pool_),
        std::move(onRead));
  }

  FetchedDeviceByteRanges fetch(
      std::shared_ptr<BufferedInputDataSource> input,
      const std::vector<cudf::io::text::byte_range_info>& ranges,
      rmm::cuda_stream_view stream) {
    return fetchByteRangesAsync(
        std::move(input),
        {ranges.data(), ranges.size()},
        stream,
        cudf::get_current_device_resource_ref());
  }

  static constexpr auto kSetting = "GLUTEN_CUDF_HYBRID_H2D_STAGING_BYTES";
  static constexpr auto kPoolSetting = "GLUTEN_CPP_S3_PINNED_POOL";
  std::optional<std::string> previous_;
  std::optional<std::string> previousPool_;
};

TEST_F(HybridH2dStagingTest, packsSparseRangesAcrossSlotsAndPartialTail) {
  std::string content(4096, '\0');
  for (size_t i = 0; i < content.size(); ++i) {
    content[i] = static_cast<char>(i);
  }
  // A large range crosses both staging slots; other ranges share a copy.
  // Original file gaps are omitted and each device span retains its meaning.
  const std::vector<cudf::io::text::byte_range_info> ranges{
      {7, 5}, {1040, 39}, {1800, 0}, {2160, 9}};
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);
  auto ioStats = std::make_shared<io::IoStatistics>();
  io::ReaderOptions options(pool_.get());
  options.setDataIoStats(ioStats);
  options.setMetadataIoStats(ioStats);
  options.setLoadQuantum(1024);
  auto input = std::make_shared<BufferedInputDataSource>(
      std::make_shared<dwio::common::CachedBufferedInput>(
          readFile,
          dwio::common::MetricsLog::voidLog(),
          StringIdLease(fileIds(), "hybridH2dStaging"),
          cache_.get(),
          nullptr,
          StringIdLease(fileIds(), "hybridH2dStagingGroup"),
          ioStats,
          nullptr,
          nullptr,
          options));
  rmm::cuda_stream stream;
  auto [buffers, spans, completion] = fetch(input, ranges, stream);
  completion.get();
  ASSERT_EQ(buffers.size(), 1);
  ASSERT_EQ(spans.size(), ranges.size());
  std::string actual(53, '\0');
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          buffers.front().data(),
          actual.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(
      actual,
      content.substr(7, 5) + content.substr(1040, 39) +
          content.substr(2160, 9));
  size_t offset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_EQ(spans[i].size(), ranges[i].size());
    EXPECT_EQ(
        spans[i].data(),
        static_cast<const uint8_t*>(buffers.front().data()) + offset);
    offset += ranges[i].size();
  }
  const auto coldReads = readFile->numReads();
  EXPECT_GT(coldReads, 0);
  auto warm = fetch(input, ranges, stream);
  std::get<2>(warm).get();
  EXPECT_EQ(readFile->numReads(), coldReads);
}

TEST_F(HybridH2dStagingTest, handlesEmptyRangesAndShortReads) {
  rmm::cuda_stream stream;
  for (const auto& ranges :
       {std::vector<cudf::io::text::byte_range_info>{},
        std::vector<cudf::io::text::byte_range_info>{{0, 0}, {32, 0}}}) {
    auto [buffers, spans, completion] = fetch(source("abcd"), ranges, stream);
    EXPECT_NO_THROW(completion.get());
    EXPECT_EQ(buffers.front().size(), 0);
    EXPECT_EQ(spans.size(), ranges.size());
  }
  EXPECT_THROW(fetch(source("abcd"), {{2, 3}}, stream), VeloxException);
}

TEST_F(HybridH2dStagingTest, preparesSecondSlotButWaitsBeforeReusingFirst) {
  rmm::cuda_stream stream;
  H2dTestStreamGate gate;
  std::promise<void> secondRead;
  auto secondReady = secondRead.get_future();
  std::atomic<size_t> reads{0};
  auto input = source(std::string(64, 'x'), [&](size_t read) {
    reads = read;
    if (read == 1) {
      gate.enqueue(stream);
    } else if (read == 2) {
      secondRead.set_value();
    }
  });
  auto pending = std::async(
      std::launch::async, [&] { return fetch(input, {{4, 48}}, stream); });
  SCOPE_EXIT {
    gate.release();
    if (pending.valid()) {
      pending.wait();
    }
    stream.synchronize();
  };
  EXPECT_EQ(secondReady.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(pending.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(reads.load(), 2);
  gate.release();
  auto [buffers, spans, completion] = pending.get();
  completion.get();
  EXPECT_EQ(reads.load(), 3);
}

TEST_F(HybridH2dStagingTest, waitsForFinalCopiesOnCompletionOrAbandonment) {
  for (const bool abandon : {false, true}) {
    rmm::cuda_stream stream;
    H2dTestStreamGate gate;
    auto input = source(std::string(64, 'x'), [&](size_t read) {
      if (read == 1) {
        gate.enqueue(stream);
      }
    });
    auto submitting = std::async(
        std::launch::async, [&] { return fetch(input, {{4, 32}}, stream); });
    std::future<void> completing;
    SCOPE_EXIT {
      gate.release();
      if (submitting.valid()) {
        submitting.wait();
      }
      if (completing.valid()) {
        completing.wait();
      }
      stream.synchronize();
    };
    ASSERT_EQ(submitting.wait_for(5s), std::future_status::ready);
    auto [buffers, spans, completion] = submitting.get();
    completing = std::async(
        std::launch::async,
        [abandon, completion = std::move(completion)]() mutable {
          if (abandon) {
            completion = std::future<void>{};
          } else {
            completion.get();
          }
        });
    EXPECT_EQ(completing.wait_for(50ms), std::future_status::timeout);
    gate.release();
    completing.get();
  }
}

TEST_F(HybridH2dStagingTest, drainsSubmittedCopyOnReadFailure) {
  rmm::cuda_stream stream;
  H2dTestStreamGate gate;
  std::promise<void> failedRead;
  auto failureReached = failedRead.get_future();
  auto input = source(std::string(64, 'x'), [&](size_t read) {
    if (read == 1) {
      gate.enqueue(stream);
    } else {
      failedRead.set_value();
      throw std::runtime_error("injected read failure");
    }
  });
  auto pending = std::async(
      std::launch::async, [&] { return fetch(input, {{4, 32}}, stream); });
  SCOPE_EXIT {
    gate.release();
    if (pending.valid()) {
      pending.wait();
    }
    stream.synchronize();
  };
  EXPECT_EQ(failureReached.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(pending.wait_for(50ms), std::future_status::timeout);
  gate.release();
  EXPECT_THROW(pending.get(), std::runtime_error);
}

TEST_F(CacheHintRangeStatsTest, countsPhysicalChunkRelativeKeys) {
  constexpr uint64_t kQuantum = 1024;
  std::string content(4 * kQuantum, 'x');
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);
  auto ioStats = std::make_shared<io::IoStatistics>();
  io::ReaderOptions options(pool_.get());
  options.setDataIoStats(ioStats);
  options.setMetadataIoStats(ioStats);
  options.setLoadQuantum(kQuantum);
  auto& ids = fileIds();
  StringIdLease fileId(ids, "cacheHintRangeStats");
  StringIdLease groupId(ids, "cacheHintRangeStatsGroup");
  auto input = std::make_shared<dwio::common::CachedBufferedInput>(
      readFile,
      dwio::common::MetricsLog::voidLog(),
      std::move(fileId),
      cache_.get(),
      nullptr,
      std::move(groupId),
      ioStats,
      nullptr,
      nullptr,
      options);
  BufferedInputDataSource source(input);

  const auto stats =
      source.canonicalCacheStats({{100, 1000}, {1100, 300}, {2048, 100}});
  EXPECT_EQ(stats.logicalRanges, 3);
  EXPECT_EQ(stats.logicalBytes, 1400);
  EXPECT_EQ(stats.uniqueRanges, 3);
  EXPECT_EQ(stats.uniqueBytes, 1400);
  EXPECT_EQ(stats.overlapBytes, 0);
  EXPECT_EQ(stats.duplicateSuppressedRanges, 0);
  EXPECT_EQ(stats.duplicateSuppressedBytes, 0);

  EXPECT_ANY_THROW(source.canonicalCacheStats({{100, 1000}, {900, 300}}));
}

TEST_F(CacheHintRangeStatsTest, bufferedMetadataTailUsesOneReadAhead) {
  ASSERT_EQ(setenv("GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC", "1", 1), 0);
  ASSERT_EQ(setenv("GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES", "65536", 1), 0);
  SCOPE_EXIT {
    unsetenv("GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC");
    unsetenv("GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES");
  };

  constexpr size_t kFileSize = 1 << 20;
  constexpr size_t kTailSize = 1 << 16;
  std::string content(kFileSize, '\0');
  std::memcpy(content.data(), "PAR1", 4);
  for (size_t i = kFileSize - kTailSize; i < content.size(); ++i) {
    content[i] = static_cast<char>(i % 251);
  }
  auto readFile = std::make_shared<tests::utils::CountingReadFile>(content);
  auto input = std::make_shared<dwio::common::BufferedInput>(readFile, *pool_);
  BufferedInputDataSource source(input);

  std::array<uint8_t, 4> magic{};
  EXPECT_EQ(source.host_read(0, magic.size(), magic.data()), magic.size());
  EXPECT_EQ(std::string_view(reinterpret_cast<char*>(magic.data()), 4), "PAR1");
  EXPECT_EQ(readFile->numReads(), 0);

  std::array<uint8_t, 8> trailer{};
  EXPECT_EQ(
      source.host_read(
          kFileSize - trailer.size(), trailer.size(), trailer.data()),
      trailer.size());
  const auto readsAfterFirstTailAccess = readFile->numReads();
  EXPECT_GT(readsAfterFirstTailAccess, 0);

  std::array<uint8_t, 256> footer{};
  EXPECT_EQ(
      source.host_read(
          kFileSize - kTailSize + 1024, footer.size(), footer.data()),
      footer.size());
  EXPECT_EQ(readFile->numReads(), readsAfterFirstTailAccess);
  EXPECT_EQ(
      std::string_view(reinterpret_cast<char*>(footer.data()), footer.size()),
      std::string_view(content).substr(
          kFileSize - kTailSize + 1024, footer.size()));
}

TEST_F(CacheHintRangeStatsTest, directCachePageH2dAvoidsHostStaging) {
  ASSERT_EQ(setenv("GLUTEN_CUDF_CACHE_H2D_INLINE", "1", 1), 0);
  SCOPE_EXIT {
    unsetenv("GLUTEN_CUDF_CACHE_H2D_INLINE");
  };
  const auto callerThread = std::this_thread::get_id();
  std::thread::id copyThread;
  constexpr size_t kBlockSize = 1024;
  constexpr size_t kOffset = 137;
  constexpr size_t kReadSize = 2500;
  std::string content(4 * kBlockSize, '\0');
  for (size_t i = 0; i < content.size(); ++i) {
    content[i] = static_cast<char>(i % 251);
  }
  folly::CPUThreadPoolExecutor executor(1);
  auto backedUpBytes = std::make_shared<std::atomic<uint64_t>>(0);
  auto streamDestroyed = std::make_shared<std::atomic<bool>>(false);
  auto input = std::make_shared<FakeCachedBufferedInput>(
      content, *pool_, &executor, kBlockSize, backedUpBytes, streamDestroyed);

  std::vector<const void*> copySources;
  std::vector<size_t> copySizes;
  std::vector<std::shared_ptr<void>> retained;
  std::weak_ptr<void> retainedWeak;
  BufferedInputDeviceCopyHooks hooks{
      .copy =
          [&](uint8_t* destination,
              const void* source,
              size_t bytes,
              rmm::cuda_stream_view /*stream*/) {
            copyThread = std::this_thread::get_id();
            copySources.push_back(source);
            copySizes.push_back(bytes);
            std::memcpy(destination, source, bytes);
          },
      .retainUntilComplete =
          [&retained, &retainedWeak](
              std::shared_ptr<void> stream,
              rmm::cuda_stream_view /*cudaStream*/) {
            retainedWeak = stream;
            retained.push_back(std::move(stream));
          }};
  BufferedInputDataSource source(input, std::move(hooks));
  std::vector<uint8_t> fakeDevice(kReadSize);
  const auto statsBefore = directCachePageH2dStats();
  auto read = source.device_read_async(
      kOffset, kReadSize, fakeDevice.data(), rmm::cuda_stream_view{});
  EXPECT_TRUE(copySizes.empty());
  EXPECT_EQ(
      read.wait_for(std::chrono::seconds{0}), std::future_status::deferred);
  EXPECT_EQ(read.get(), kReadSize);
  EXPECT_EQ(copyThread, callerThread);

  EXPECT_EQ(
      std::string_view(
          reinterpret_cast<const char*>(fakeDevice.data()), fakeDevice.size()),
      std::string_view(content).substr(kOffset, kReadSize));
  ASSERT_EQ(copySizes.size(), 3);
  EXPECT_THAT(copySizes, testing::ElementsAre(1024, 1024, 452));
  for (const auto* copySource : copySources) {
    const auto* sourceBytes = static_cast<const char*>(copySource);
    EXPECT_GE(sourceBytes, input->contentBegin());
    EXPECT_LT(sourceBytes, input->contentEnd());
  }
  EXPECT_EQ(backedUpBytes->load(), 572);
  EXPECT_FALSE(streamDestroyed->load(std::memory_order_acquire));
  EXPECT_FALSE(retainedWeak.expired());

  const auto statsAfter = directCachePageH2dStats();
  EXPECT_EQ(statsAfter.copies - statsBefore.copies, 3);
  EXPECT_EQ(statsAfter.bytes - statsBefore.bytes, kReadSize);

  retained.clear();
  EXPECT_TRUE(retainedWeak.expired());
  EXPECT_TRUE(streamDestroyed->load(std::memory_order_acquire));
}

TEST_F(
    CacheHintRangeStatsTest,
    deferredCacheH2dDoesNotHoldCudfSubmissionMutex) {
  ASSERT_EQ(setenv("GLUTEN_CUDF_CACHE_H2D_INLINE", "1", 1), 0);
  SCOPE_EXIT {
    unsetenv("GLUTEN_CUDF_CACHE_H2D_INLINE");
  };

  struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool firstEntered{false};
    bool secondEntered{false};
    bool releaseFirst{false};
  } gate;

  folly::CPUThreadPoolExecutor executor(2);
  auto makeInput = [&]() {
    return std::make_shared<FakeCachedBufferedInput>(
        std::string(4096, 'x'),
        *pool_,
        &executor,
        4096,
        std::make_shared<std::atomic<uint64_t>>(0),
        std::make_shared<std::atomic<bool>>(false));
  };
  auto makeHooks = [&](bool first) {
    return BufferedInputDeviceCopyHooks{
        .copy =
            [&, first](uint8_t*, const void*, size_t, rmm::cuda_stream_view) {
              std::unique_lock lock(gate.mutex);
              (first ? gate.firstEntered : gate.secondEntered) = true;
              gate.cv.notify_all();
              if (first) {
                gate.cv.wait(lock, [&] { return gate.releaseFirst; });
              }
            },
        .retainUntilComplete = [](std::shared_ptr<void>,
                                  rmm::cuda_stream_view) {}};
  };

  BufferedInputDataSource firstSource(makeInput(), makeHooks(true));
  BufferedInputDataSource secondSource(makeInput(), makeHooks(false));
  rmm::cuda_stream firstStream;
  rmm::cuda_stream secondStream;
  auto fetch = [](BufferedInputDataSource& source,
                  rmm::cuda_stream_view stream) {
    std::array<cudf::io::text::byte_range_info, 1> ranges{{{0, 1}}};
    auto [buffers, spans, completion] =
        cudf::io::parquet::fetch_byte_ranges_to_device_async(
            source,
            cudf::host_span<const cudf::io::text::byte_range_info>{
                ranges.data(), ranges.size()},
            stream,
            cudf::get_current_device_resource_ref());
    completion.get();
  };

  auto first = std::async(
      std::launch::async, fetch, std::ref(firstSource), firstStream.view());
  bool firstEntered;
  {
    std::unique_lock lock(gate.mutex);
    firstEntered =
        gate.cv.wait_for(lock, 5s, [&] { return gate.firstEntered; });
  }
  EXPECT_TRUE(firstEntered);
  if (!firstEntered) {
    {
      std::lock_guard lock(gate.mutex);
      gate.releaseFirst = true;
    }
    gate.cv.notify_all();
    first.get();
    return;
  }

  auto second = std::async(
      std::launch::async, fetch, std::ref(secondSource), secondStream.view());
  bool secondEnteredWhileFirstBlocked;
  {
    std::unique_lock lock(gate.mutex);
    secondEnteredWhileFirstBlocked =
        gate.cv.wait_for(lock, 500ms, [&] { return gate.secondEntered; });
    gate.releaseFirst = true;
    gate.cv.notify_all();
  }

  first.get();
  second.get();
  EXPECT_TRUE(secondEnteredWhileFirstBlocked);
}

TEST_F(CacheHintRangeStatsTest, boundedCachePageRegistrationFallsBack) {
  memory::Allocation allocation;
  ASSERT_TRUE(allocator_->allocateNonContiguous(3, allocation));
  ASSERT_GE(allocation.numRuns(), 2);
  const auto bytes = allocation.byteSize();
  std::vector<void*> registered;
  std::vector<void*> unregistered;
  auto registration = makeBoundedCachePageRegistration(
      bytes,
      CachePageHostRegistrationHooks{
          .registerRun =
              [&registered](void* address, size_t /*bytes*/) {
                registered.push_back(address);
                return true;
              },
          .unregisterRun =
              [&unregistered](void* address) {
                unregistered.push_back(address);
              }});
  const auto before = cachePageRegistrationStats();
  auto token = registration.registerBackingRuns(allocation);
  ASSERT_NE(token, nullptr);
  EXPECT_EQ(registered.size(), allocation.numRuns());

  EXPECT_EQ(registration.registerBackingRuns(allocation), nullptr);
  const auto during = cachePageRegistrationStats();
  EXPECT_EQ(during.successes - before.successes, 1);
  EXPECT_EQ(during.failures - before.failures, 1);
  EXPECT_EQ(during.currentBytes - before.currentBytes, bytes);

  token.reset();
  const auto after = cachePageRegistrationStats();
  EXPECT_EQ(unregistered.size(), allocation.numRuns());
  EXPECT_EQ(after.currentBytes, before.currentBytes);
  EXPECT_EQ(after.unregisteredBytes - before.unregisteredBytes, bytes);

  uint64_t registerCalls = 0;
  std::vector<void*> rolledBack;
  auto failingRegistration = makeBoundedCachePageRegistration(
      bytes,
      CachePageHostRegistrationHooks{
          .registerRun = [&registerCalls](
                             void*, size_t) { return ++registerCalls != 2; },
          .unregisterRun =
              [&rolledBack](void* address) { rolledBack.push_back(address); }});
  const auto beforeFailure = cachePageRegistrationStats();
  EXPECT_EQ(failingRegistration.registerBackingRuns(allocation), nullptr);
  const auto afterFailure = cachePageRegistrationStats();
  EXPECT_EQ(afterFailure.failures - beforeFailure.failures, 1);
  EXPECT_EQ(afterFailure.currentBytes, before.currentBytes);
  EXPECT_EQ(rolledBack.size(), 1);
  allocator_->freeNonContiguous(allocation);
}

TEST_F(CacheHintRangeStatsTest, prewarmsProtectedLargestSizeClassBacking) {
  memory::MemoryAllocator::Options options;
  options.capacity = 64ULL << 20;
  auto allocator = std::make_shared<memory::MmapAllocator>(options);
  std::vector<std::pair<void*, size_t>> registered;
  std::vector<void*> unregistered;
  auto registration = makeBoundedCachePageRegistration(
      48ULL << 20,
      CachePageHostRegistrationHooks{
          .registerRun =
              [&registered](void* address, size_t bytes) {
                registered.emplace_back(address, bytes);
                return true;
              },
          .unregisterRun =
              [&unregistered](void* address) {
                unregistered.push_back(address);
              }});
  const auto before = cachePageRegistrationStats();
  auto prewarm = registration.prewarmLargestSizeClass(*allocator, 32ULL << 20);
  ASSERT_NE(prewarm, nullptr);
  const auto afterPrewarm = cachePageRegistrationStats();
  EXPECT_EQ(afterPrewarm.prewarmSuccesses - before.prewarmSuccesses, 1);
  EXPECT_EQ(afterPrewarm.prewarmBytes - before.prewarmBytes, 32ULL << 20);
  EXPECT_EQ(afterPrewarm.currentBytes - before.currentBytes, 32ULL << 20);
  EXPECT_FALSE(registered.empty());

  // The backing is free for AsyncDataCache use, but is already registered.
  memory::Allocation reuse;
  ASSERT_TRUE(allocator->allocateNonContiguous(
      memory::AllocationTraits::numPages(16ULL << 20),
      reuse,
      nullptr,
      allocator->largestSizeClass()));
  const auto registerCalls = registered.size();
  auto entryLifetime = registration.registerBackingRuns(reuse);
  ASSERT_NE(entryLifetime, nullptr);
  EXPECT_EQ(registered.size(), registerCalls);
  const auto covered = cachePageRegistrationStats();
  EXPECT_GT(covered.prewarmCoveredBytes - before.prewarmCoveredBytes, 0);
  entryLifetime.reset();
  EXPECT_TRUE(unregistered.empty());
  allocator->freeNonContiguous(reuse);

  prewarm.reset();
  EXPECT_EQ(unregistered.size(), registered.size());
  const auto after = cachePageRegistrationStats();
  EXPECT_EQ(after.currentBytes, before.currentBytes);
  EXPECT_EQ(after.unregisteredBytes - before.unregisteredBytes, 32ULL << 20);
}

TEST_F(CacheHintRangeStatsTest, boundedCacheRangeRegistrationOwnsLifetime) {
  memory::Allocation allocation;
  ASSERT_TRUE(allocator_->allocateNonContiguous(1, allocation));
  ASSERT_EQ(allocation.numRuns(), 1);
  const auto run = allocation.runAt(0);
  std::vector<std::pair<void*, size_t>> registered;
  std::vector<void*> unregistered;
  auto registration = makeBoundedCachePageRegistration(
      run.numBytes(),
      CachePageHostRegistrationHooks{
          .registerRun =
              [&registered](void* address, size_t bytes) {
                registered.emplace_back(address, bytes);
                return true;
              },
          .unregisterRun =
              [&unregistered](void* address) {
                unregistered.push_back(address);
              }});

  const auto before = cachePageRegistrationStats();
  auto token =
      registration.registerBackingRange(run.data<void>(), run.numBytes());
  ASSERT_NE(token, nullptr);
  ASSERT_THAT(
      registered,
      testing::ElementsAre(
          std::pair<void*, size_t>{run.data<void>(), run.numBytes()}));
  EXPECT_EQ(
      cachePageRegistrationStats().currentBytes - before.currentBytes,
      run.numBytes());

  token.reset();
  EXPECT_THAT(unregistered, testing::ElementsAre(run.data<void>()));
  EXPECT_EQ(cachePageRegistrationStats().currentBytes, before.currentBytes);
  allocator_->freeNonContiguous(allocation);
}

TEST_F(
    CacheHintRangeStatsTest,
    persistentRangeRegistrationSurvivesEntryEviction) {
  memory::MmapAllocator::Options options;
  options.capacity = 16ULL << 20;
  auto allocator = std::make_shared<memory::MmapAllocator>(options);
  std::vector<std::pair<void*, size_t>> registered;
  std::vector<void*> unregistered;
  auto registration = makeBoundedCachePageRegistration(
      4ULL << 20,
      CachePageHostRegistrationHooks{
          .registerRun =
              [&registered](void* address, size_t bytes) {
                registered.emplace_back(address, bytes);
                return true;
              },
          .unregisterRun =
              [&unregistered](void* address) {
                unregistered.push_back(address);
              }});
  constexpr uint64_t kBytes = 1ULL << 20;
  auto* address = allocator->allocateBytes(kBytes);
  ASSERT_NE(address, nullptr);
  const auto before = cachePageRegistrationStats();
  auto first =
      registration.registerPersistentBackingRange(*allocator, address, kBytes);
  ASSERT_NE(first, nullptr);
  ASSERT_THAT(
      registered,
      testing::ElementsAre(std::pair<void*, size_t>{address, kBytes}));

  // Multiple cache entries referring to the same reusable allocator backing
  // share one CUDA registration. Dropping entry tokens does not unregister it.
  auto covered =
      registration.registerPersistentBackingRange(*allocator, address, kBytes);
  ASSERT_NE(covered, nullptr);
  EXPECT_EQ(registered.size(), 1);
  first.reset();
  covered.reset();
  EXPECT_TRUE(unregistered.empty());
  allocator->freeBytes(address, kBytes);

  registration.persistentLifetime.reset();
  EXPECT_THAT(unregistered, testing::ElementsAre(address));
  const auto after = cachePageRegistrationStats();
  EXPECT_EQ(after.currentBytes, before.currentBytes);
  EXPECT_EQ(after.unregisteredBytes - before.unregisteredBytes, kBytes);
}

TEST_F(CacheHintRangeStatsTest, cacheBackingRegistrationTracksEntryLifetime) {
  auto allocator = std::make_shared<memory::MallocAllocator>(
      memory::MemoryAllocator::Options{
          .capacity = 1 << 20, .reservationByteLimit = 0});
  std::atomic<uint64_t> registrations{0};
  std::atomic<uint64_t> releases{0};
  cache::AsyncDataCache::Options options;
  options.registerBackingRuns = [&](const memory::Allocation& allocation) {
    EXPECT_GT(allocation.numRuns(), 0);
    registrations.fetch_add(1, std::memory_order_relaxed);
    return std::shared_ptr<void>(new uint8_t, [&releases](void* value) {
      delete static_cast<uint8_t*>(value);
      releases.fetch_add(1, std::memory_order_relaxed);
    });
  };
  auto cache = cache::AsyncDataCache::create(allocator.get(), nullptr, options);
  StringIdLease file(fileIds(), "cacheBackingRegistrationLifecycle");
  auto pin = cache->findOrCreate(
      cache::RawFileCacheKey{file.id(), 0}, 16 << 10, /*contiguous=*/false);
  pin.checkedEntry()->setExclusiveToShared();
  EXPECT_TRUE(pin.checkedEntry()->hasBackingRegistration());
  EXPECT_EQ(registrations.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(releases.load(std::memory_order_relaxed), 0);

  pin.clear();
  EXPECT_GT(cache->shrink(1), 0);
  EXPECT_EQ(releases.load(std::memory_order_relaxed), 1);
  cache->shutdown();
  EXPECT_EQ(releases.load(std::memory_order_relaxed), 1);
  cache.reset();
  allocator.reset();
}

#ifdef VELOX_ENABLE_S3
class RecordingReadFile final : public ReadFile {
 public:
  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context) const override {
    auto* destination = static_cast<char*>(buffer);
    for (uint64_t index = 0; index < length; ++index) {
      destination[index] = static_cast<char>(offset + index);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    offsets_.push_back(offset);
    destinations_.push_back(destination);
    contextMarkers_.push_back(context.fileOpts.at("marker"));
    return {destination, static_cast<size_t>(length)};
  }

  bool shouldCoalesce() const override {
    return false;
  }

  uint64_t size() const override {
    return 1024;
  }

  uint64_t memoryUsage() const override {
    return sizeof(*this);
  }

  std::string getName() const override {
    return "s3://test-bucket/test-key";
  }

  uint64_t getNaturalReadSize() const override {
    return 8;
  }

  std::vector<uint64_t> offsets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return offsets_;
  }

  std::vector<char*> destinations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return destinations_;
  }

  std::vector<std::string> contextMarkers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return contextMarkers_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::vector<uint64_t> offsets_;
  mutable std::vector<char*> destinations_;
  mutable std::vector<std::string> contextMarkers_;
};

TEST(ExecutorPrefetchTest, nativeScheduledReadWritesScatterBuffersDirectly) {
  ASSERT_EQ(setenv("GLUTEN_CPP_S3_AWS_SDK", "1", 1), 0);
  ASSERT_EQ(unsetenv("GLUTEN_CPP_S3_CRT"), 0);
  auto base = std::make_shared<RecordingReadFile>();
  auto scheduled =
      makeNativeScheduledS3ReadFile(base, "s3://test-bucket/test-key");
  EXPECT_FALSE(prioritizeNativeS3File("s3://test-bucket/test-key"));

  std::array<char, 3> first{};
  std::array<char, 2> second{};
  const std::vector<folly::Range<char*>> buffers{
      {first.data(), first.size()},
      {nullptr, 3},
      {second.data(), second.size()}};
  FileIoContext context;
  context.fileOpts["marker"] = "cache-miss";

  EXPECT_EQ(scheduled->preadv(10, buffers, context), 8);
  EXPECT_THAT(base->offsets(), testing::UnorderedElementsAre(10, 16));
  EXPECT_THAT(
      base->destinations(),
      testing::UnorderedElementsAre(first.data(), second.data()));
  EXPECT_THAT(
      base->contextMarkers(), testing::ElementsAre("cache-miss", "cache-miss"));
  EXPECT_THAT(first, testing::ElementsAre(10, 11, 12));
  EXPECT_THAT(second, testing::ElementsAre(16, 17));
  EXPECT_EQ(scheduled->bytesRead(), first.size() + second.size());
}

TEST(ExecutorPrefetchTest, nativeScatterGroupsOnlySplitAtLogicalGaps) {
  std::array<char, 3> first{};
  std::array<char, 4> second{};
  std::array<char, 2> third{};
  const std::vector<folly::Range<char*>> destinations{
      {first.data(), first.size()},
      {second.data(), second.size()},
      {nullptr, 5},
      {third.data(), third.size()}};

  const auto groups = groupNativeS3ReadDestinations(10, destinations);
  ASSERT_EQ(groups.size(), 2);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, first.size() + second.size());
  ASSERT_EQ(groups[0].destinations.size(), 2);
  EXPECT_EQ(groups[0].destinations[0].data(), first.data());
  EXPECT_EQ(groups[0].destinations[1].data(), second.data());
  EXPECT_EQ(groups[1].offset, 10 + first.size() + second.size() + 5);
  EXPECT_EQ(groups[1].size, third.size());
  ASSERT_EQ(groups[1].destinations.size(), 1);
  EXPECT_EQ(groups[1].destinations[0].data(), third.data());
}

TEST(ExecutorPrefetchTest, nativeS3StaticReadPolicyPreservesOverrides) {
  std::array<char, 8> first{};
  std::array<char, 8> second{};
  const auto policy = chooseNativeS3ReadPolicy(
      {{first.data(), first.size()},
       {nullptr, 3},
       {second.data(), second.size()}},
      /*baseRangeBytes=*/4,
      /*maxGapBytes=*/7,
      /*maxRangeBytes=*/13,
      /*adaptive=*/false);

  EXPECT_EQ(policy.maxGapBytes, 7);
  EXPECT_EQ(policy.maxRangeBytes, 13);
}

TEST(ExecutorPrefetchTest, nativeS3AdaptiveReadPolicyKeepsSingleRangeBounded) {
  std::array<char, 2'000> destination{};
  const auto policy = chooseNativeS3ReadPolicy(
      {{destination.data(), destination.size()}},
      /*baseRangeBytes=*/1'000,
      /*maxGapBytes=*/100,
      /*maxRangeBytes=*/2'000,
      /*adaptive=*/true);

  EXPECT_EQ(policy.maxGapBytes, 0);
  EXPECT_EQ(policy.maxRangeBytes, 1'000);
}

TEST(ExecutorPrefetchTest, nativeS3AdaptiveReadPolicyWidensDenseMultiRange) {
  std::array<char, 1'000> first{};
  std::array<char, 1'000> second{};
  const auto policy = chooseNativeS3ReadPolicy(
      {{nullptr, 500},
       {first.data(), first.size()},
       {nullptr, 50},
       {second.data(), second.size()},
       {nullptr, 500}},
      /*baseRangeBytes=*/1'000,
      /*maxGapBytes=*/100,
      /*maxRangeBytes=*/2'000,
      /*adaptive=*/true);

  // Leading and trailing holes do not count. The 50-byte internal gap is
  // below both the explicit ceiling and the 2,000 / 32 overhead budget.
  EXPECT_EQ(policy.maxGapBytes, 50);
  EXPECT_EQ(policy.maxRangeBytes, 2'000);
}

TEST(ExecutorPrefetchTest, nativeS3AdaptiveReadPolicyRejectsSparseRanges) {
  std::array<char, 1'000> first{};
  std::array<char, 1'000> second{};
  const auto policy = chooseNativeS3ReadPolicy(
      {{first.data(), first.size()},
       {nullptr, 100},
       {second.data(), second.size()}},
      /*baseRangeBytes=*/1'000,
      /*maxGapBytes=*/1'000,
      /*maxRangeBytes=*/2'000,
      /*adaptive=*/true);

  EXPECT_EQ(policy.maxGapBytes, 0);
  EXPECT_EQ(policy.maxRangeBytes, 1'000);
}

TEST(ExecutorPrefetchTest, nativeScatterCoalescesBoundedLogicalGaps) {
  std::array<char, 3> first{};
  std::array<char, 4> second{};
  std::array<char, 2> third{};
  const std::vector<folly::Range<char*>> destinations{
      {first.data(), first.size()},
      {nullptr, 5},
      {second.data(), second.size()},
      {nullptr, 6},
      {third.data(), third.size()}};

  const auto groups = groupNativeS3ReadDestinations(
      10, destinations, /*maxGapBytes=*/5, /*maxRangeBytes=*/16);
  ASSERT_EQ(groups.size(), 2);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, first.size() + 5 + second.size());
  ASSERT_EQ(groups[0].destinations.size(), 3);
  EXPECT_EQ(groups[0].destinations[0].data(), first.data());
  EXPECT_EQ(groups[0].destinations[1].data(), nullptr);
  EXPECT_EQ(groups[0].destinations[1].size(), 5);
  EXPECT_EQ(groups[0].destinations[2].data(), second.data());
  EXPECT_EQ(groups[1].offset, 10 + first.size() + 5 + second.size() + 6);
  EXPECT_EQ(groups[1].size, third.size());
}

TEST(ExecutorPrefetchTest, nativeScatterRespectsMaximumCoalescedRange) {
  std::array<char, 8> first{};
  std::array<char, 8> second{};
  const std::vector<folly::Range<char*>> destinations{
      {first.data(), first.size()},
      {nullptr, 1},
      {second.data(), second.size()}};

  const auto groups = groupNativeS3ReadDestinations(
      10, destinations, /*maxGapBytes=*/1, /*maxRangeBytes=*/16);
  ASSERT_EQ(groups.size(), 2);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, first.size());
  EXPECT_EQ(groups[1].offset, 10 + first.size() + 1);
  EXPECT_EQ(groups[1].size, second.size());
}

TEST(ExecutorPrefetchTest, nativeScatterSplitsOversizedDestination) {
  std::array<char, 10> destination{};
  const std::vector<folly::Range<char*>> destinations{
      {destination.data(), destination.size()}};

  const auto groups = groupNativeS3ReadDestinations(
      10, destinations, /*maxGapBytes=*/0, /*maxRangeBytes=*/4);
  ASSERT_EQ(groups.size(), 3);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, 4);
  ASSERT_EQ(groups[0].destinations.size(), 1);
  EXPECT_EQ(groups[0].destinations[0].data(), destination.data());
  EXPECT_EQ(groups[0].destinations[0].size(), 4);
  EXPECT_EQ(groups[1].offset, 14);
  EXPECT_EQ(groups[1].size, 4);
  ASSERT_EQ(groups[1].destinations.size(), 1);
  EXPECT_EQ(groups[1].destinations[0].data(), destination.data() + 4);
  EXPECT_EQ(groups[1].destinations[0].size(), 4);
  EXPECT_EQ(groups[2].offset, 18);
  EXPECT_EQ(groups[2].size, 2);
  ASSERT_EQ(groups[2].destinations.size(), 1);
  EXPECT_EQ(groups[2].destinations[0].data(), destination.data() + 8);
  EXPECT_EQ(groups[2].destinations[0].size(), 2);
}

TEST(ExecutorPrefetchTest, nativeScatterCanPreserveOversizedDestination) {
  std::array<char, 10> destination{};

  const auto groups = groupNativeS3ReadDestinations(
      10,
      {{destination.data(), destination.size()}},
      /*maxGapBytes=*/0,
      /*maxRangeBytes=*/4,
      /*sliceOversizedRanges=*/false);

  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, destination.size());
  ASSERT_EQ(groups[0].destinations.size(), 1);
  EXPECT_EQ(groups[0].destinations[0].data(), destination.data());
  EXPECT_EQ(groups[0].destinations[0].size(), destination.size());
}

TEST(ExecutorPrefetchTest, nativeScatterStartsOversizeDestinationAfterGap) {
  std::array<char, 3> first{};
  std::array<char, 6> second{};
  const std::vector<folly::Range<char*>> destinations{
      {first.data(), first.size()},
      {nullptr, 1},
      {second.data(), second.size()}};

  const auto groups = groupNativeS3ReadDestinations(
      10, destinations, /*maxGapBytes=*/1, /*maxRangeBytes=*/5);

  ASSERT_EQ(groups.size(), 3);
  EXPECT_EQ(groups[0].offset, 10);
  EXPECT_EQ(groups[0].size, 3);
  ASSERT_EQ(groups[0].destinations.size(), 1);
  EXPECT_EQ(groups[0].destinations[0].data(), first.data());
  EXPECT_EQ(groups[1].offset, 14);
  EXPECT_EQ(groups[1].size, 5);
  EXPECT_EQ(groups[1].destinations[0].data(), second.data());
  EXPECT_EQ(groups[1].destinations[0].size(), 5);
  EXPECT_EQ(groups[2].offset, 19);
  EXPECT_EQ(groups[2].size, 1);
  EXPECT_EQ(groups[2].destinations[0].data(), second.data() + 5);
  EXPECT_EQ(groups[2].destinations[0].size(), 1);
}

TEST(ExecutorPrefetchTest, nativeScatterStreamWritesExactDestinations) {
  std::array<char, 2> first{};
  std::array<char, 4> second{};
  NativeS3ScatterWriteStream stream(
      {{first.data(), first.size()}, {second.data(), second.size()}});

  stream.write("abcdef", 6);

  EXPECT_TRUE(stream.good());
  EXPECT_EQ(stream.bytesWritten(), 6);
  EXPECT_FALSE(stream.overflowed());
  EXPECT_THAT(first, testing::ElementsAre('a', 'b'));
  EXPECT_THAT(second, testing::ElementsAre('c', 'd', 'e', 'f'));
}

TEST(ExecutorPrefetchTest, nativeScatterStreamDiscardsGapBytes) {
  std::array<char, 2> first{};
  std::array<char, 4> second{};
  NativeS3ScatterWriteStream stream(
      {{first.data(), first.size()},
       {nullptr, 5},
       {second.data(), second.size()}});

  stream.write("ab12345cdef", 11);

  EXPECT_TRUE(stream.good());
  EXPECT_EQ(stream.bytesWritten(), 11);
  EXPECT_FALSE(stream.overflowed());
  EXPECT_THAT(first, testing::ElementsAre('a', 'b'));
  EXPECT_THAT(second, testing::ElementsAre('c', 'd', 'e', 'f'));
}
#endif

TEST(ExecutorPrefetchTest, reservesBytesBeforeAllocation) {
  folly::CPUThreadPoolExecutor executor(1);
  auto broker = ExecutorReadBroker::get(&executor, 4, 1);
  auto first = broker->reserve(4);

  auto second =
      std::async(std::launch::async, [&] { return broker->reserve(1); });
  EXPECT_EQ(second.wait_for(50ms), std::future_status::timeout);

  first.reset();
  EXPECT_EQ(second.wait_for(5s), std::future_status::ready);
  auto secondReservation = second.get();
  secondReservation.reset();

  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, splitConcurrencyConfigIsIndependentOfReadThreads) {
  auto connectorConfig = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kPrefetchThreads, "3"}});
  CudfHiveConfig hiveConfig(connectorConfig);

  config::ConfigBase defaultSession(
      std::unordered_map<std::string, std::string>{});
  // Zero/unset preserves the historical min(16, prefetchThreads) behavior.
  EXPECT_EQ(
      hiveConfig.executorSplitPrefetchConcurrencySession(&defaultSession), 3);

  config::ConfigBase explicitSession(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kPrefetchThreadsSession, "64"},
          {CudfHiveConfig::kExecutorSplitPrefetchConcurrencySession, "5"}});
  EXPECT_EQ(hiveConfig.prefetchThreadsSession(&explicitSession), 64);
  EXPECT_EQ(
      hiveConfig.executorSplitPrefetchConcurrencySession(&explicitSession), 5);
}

TEST(ExecutorPrefetchTest, createsOneReadHandlePerPhysicalPath) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 8, 2);
  std::mutex pathsMutex;
  std::vector<std::string> paths;

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-paths",
      "s3://bucket-a/primary.parquet",
      {{"s3://bucket-a/primary.parquet", 1},
       {"s3://bucket-b/coalesced.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-paths",
      [&](const std::string& path, std::optional<std::size_t>) {
        {
          std::lock_guard<std::mutex> lock(pathsMutex);
          paths.push_back(path);
        }
        return PrefetchReadFunction{
            [](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
            }};
      },
      broker,
      1,
      8);

  auto result = ExecutorSplitPrefetch::take(
      &executor, "query-paths", "s3://bucket-a/primary.parquet");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->buffers.size(), 2);
  EXPECT_THAT(
      paths,
      testing::UnorderedElementsAre(
          "s3://bucket-a/primary.parquet", "s3://bucket-b/coalesced.parquet"));

  result.reset();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-paths");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, reusesOneReadSourceAcrossRangesOfFile) {
  constexpr uint64_t kRangeBytes = 16ULL << 20;
  constexpr uint64_t kFileBytes = kRangeBytes + 1;
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, kFileBytes, 2);
  std::atomic<uint32_t> factoryCount{0};
  std::atomic<uint32_t> rangeCount{0};
  std::mutex observationsMutex;
  std::vector<uint32_t> sourceIds;
  std::vector<uint64_t> offsets;

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-multi-range",
      "s3://bucket/multi-range.parquet",
      {{"s3://bucket/multi-range.parquet", kFileBytes}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-multi-range",
      [&](const std::string&, std::optional<std::size_t>) {
        const auto sourceId =
            factoryCount.fetch_add(1, std::memory_order_relaxed) + 1;
        return PrefetchReadFunction{
            [&, sourceId](
                uint64_t offset, uint64_t size, uint8_t* destination) {
              std::memset(destination, static_cast<int>(sourceId), size);
              rangeCount.fetch_add(1, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(observationsMutex);
              sourceIds.push_back(sourceId);
              offsets.push_back(offset);
            }};
      },
      broker,
      1,
      kFileBytes);

  auto result = ExecutorSplitPrefetch::take(
      &executor, "query-multi-range", "s3://bucket/multi-range.parquet");
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->buffers.size(), 1);
  EXPECT_EQ(factoryCount.load(), 1);
  EXPECT_EQ(rangeCount.load(), 2);
  EXPECT_THAT(sourceIds, testing::ElementsAre(1, 1));
  EXPECT_THAT(offsets, testing::UnorderedElementsAre(0, kRangeBytes));
  EXPECT_EQ(result->buffers.front()->data()[0], 1);
  EXPECT_EQ(result->buffers.front()->data()[kFileBytes - 1], 1);

  result.reset();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-multi-range");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, requestedSplitBypassesFullReadyWindow) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 1, 2);
  std::promise<void> firstReadCompleted;
  std::atomic<uint32_t> readCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-demand",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 1}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-demand",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-demand",
      [&](const std::string& path, std::optional<std::size_t>) {
        return PrefetchReadFunction{
            [&, path](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
              ++readCount;
              if (path.ends_with("first.parquet")) {
                firstReadCompleted.set_value();
              }
            }};
      },
      broker,
      2,
      1);

  ASSERT_EQ(
      firstReadCompleted.get_future().wait_for(5s), std::future_status::ready);
  // The first completed result still owns the entire ready window. It must
  // not also retain the broker's full I/O budget: a demand for the second
  // split has to make progress without first consuming the ready result.
  auto second = std::async(std::launch::async, [&] {
    return ExecutorSplitPrefetch::take(
        &executor, "query-demand", "s3://bucket/second.parquet");
  });
  ASSERT_EQ(second.wait_for(5s), std::future_status::ready);
  auto secondResult = second.get();
  ASSERT_NE(secondResult, nullptr);
  secondResult.reset();

  auto firstResult = ExecutorSplitPrefetch::take(
      &executor, "query-demand", "s3://bucket/first.parquet");
  ASSERT_NE(firstResult, nullptr);
  firstResult.reset();
  EXPECT_EQ(readCount.load(), 2);

  ExecutorSplitPrefetch::eraseQuery(&executor, "query-demand");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, oversizedSpeculationWaitsForDemand) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 1, 2);
  std::atomic<uint32_t> firstReadCount{0};
  std::atomic<uint32_t> secondReadCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-oversized-demand",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 2}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-oversized-demand",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 2}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-oversized-demand",
      [&](const std::string& path, std::optional<std::size_t>) {
        return PrefetchReadFunction{
            [&, path](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
              if (path.ends_with("first.parquet")) {
                ++firstReadCount;
              } else {
                ++secondReadCount;
              }
            }};
      },
      broker,
      2,
      1);

  // Neither oversized split may monopolize the broker speculatively.
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(firstReadCount.load(), 0);
  EXPECT_EQ(secondReadCount.load(), 0);

  auto secondResult = ExecutorSplitPrefetch::take(
      &executor, "query-oversized-demand", "s3://bucket/second.parquet");
  ASSERT_NE(secondResult, nullptr);
  secondResult.reset();
  EXPECT_EQ(firstReadCount.load(), 0);
  EXPECT_EQ(secondReadCount.load(), 1);

  auto firstResult = ExecutorSplitPrefetch::take(
      &executor, "query-oversized-demand", "s3://bucket/first.parquet");
  ASSERT_NE(firstResult, nullptr);
  firstResult.reset();
  EXPECT_EQ(firstReadCount.load(), 1);

  ExecutorSplitPrefetch::eraseQuery(&executor, "query-oversized-demand");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, cacheHintDemandBypassesReadyByteWindow) {
  folly::CPUThreadPoolExecutor executor(2);
  std::promise<void> firstCompleted;
  std::atomic<uint32_t> firstCount{0};
  std::atomic<uint32_t> secondCount{0};

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-demand",
      "first",
      1,
      [&] {
        ++firstCount;
        firstCompleted.set_value();
      },
      2,
      1);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-demand",
      "second",
      1,
      [&] { ++secondCount; },
      2,
      1);

  ASSERT_EQ(
      firstCompleted.get_future().wait_for(5s), std::future_status::ready);
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(firstCount.load(), 1);
  EXPECT_EQ(secondCount.load(), 0);

  auto demanded = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor, "query-cache-demand", "second");
  });
  EXPECT_EQ(demanded.wait_for(5s), std::future_status::ready);
  demanded.get();
  EXPECT_EQ(secondCount.load(), 1);

  ExecutorSplitPrefetch::takeCacheHint(
      &executor, "query-cache-demand", "first");
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-demand");
}

TEST(ExecutorPrefetchTest, cacheHintFailureFallsBackToDemand) {
  folly::CPUThreadPoolExecutor executor(1);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-fallback",
      "broken",
      1,
      [] { throw std::runtime_error("expected cache hint failure"); },
      1,
      1);

  EXPECT_NO_THROW(
      ExecutorSplitPrefetch::takeCacheHint(
          &executor, "query-cache-fallback", "broken"));
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-fallback");
}

TEST(ExecutorPrefetchTest, cacheHintAccountsConsumedAndUnusedRanges) {
  folly::CPUThreadPoolExecutor executor(1);
  const auto before = ExecutorSplitPrefetch::cacheHintRangeStatsForTest();
  const CacheHintRangeStats consumed{
      .logicalRanges = 3,
      .logicalBytes = 30,
      .uniqueRanges = 2,
      .uniqueBytes = 20,
      .overlapBytes = 5,
      .duplicateSuppressedRanges = 1,
      .duplicateSuppressedBytes = 10};
  const CacheHintRangeStats unused{
      .logicalRanges = 2,
      .logicalBytes = 25,
      .uniqueRanges = 1,
      .uniqueBytes = 15,
      .duplicateSuppressedRanges = 1,
      .duplicateSuppressedBytes = 10};

  ExecutorSplitPrefetch::registerCacheHint(
      &executor, "query-cache-range-stats", "consumed", consumed, [] {}, 1, 64);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor, "query-cache-range-stats", "unused", unused, [] {}, 1, 64);
  ExecutorSplitPrefetch::takeCacheHint(
      &executor, "query-cache-range-stats", "consumed");
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-range-stats");

  const auto after = ExecutorSplitPrefetch::cacheHintRangeStatsForTest();
  EXPECT_EQ(after.logicalRanges - before.logicalRanges, 5);
  EXPECT_EQ(after.logicalBytes - before.logicalBytes, 55);
  EXPECT_EQ(after.uniqueRanges - before.uniqueRanges, 3);
  EXPECT_EQ(after.uniqueBytes - before.uniqueBytes, 35);
  EXPECT_EQ(after.overlapBytes - before.overlapBytes, 5);
  EXPECT_EQ(
      after.duplicateSuppressedRanges - before.duplicateSuppressedRanges, 2);
  EXPECT_EQ(
      after.duplicateSuppressedBytes - before.duplicateSuppressedBytes, 20);
  EXPECT_EQ(after.consumedRanges - before.consumedRanges, 2);
  EXPECT_EQ(after.consumedBytes - before.consumedBytes, 20);
  EXPECT_EQ(after.unusedRanges - before.unusedRanges, 1);
  EXPECT_EQ(after.unusedBytes - before.unusedBytes, 15);
}

TEST(ExecutorPrefetchTest, cachePrefetchPlanStatsTrackReuseAndBound) {
  const auto before = ExecutorSplitPrefetch::cachePrefetchPlanStatsForTest();
  ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(false);
  ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(true);
  ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(true);
  ExecutorSplitPrefetch::recordCachePrefetchPlanEntries(37);

  const auto after = ExecutorSplitPrefetch::cachePrefetchPlanStatsForTest();
  EXPECT_EQ(after.hits - before.hits, 2);
  EXPECT_EQ(after.misses - before.misses, 1);
  EXPECT_EQ(after.entries, 37);
}

TEST(ExecutorPrefetchTest, stopsBeforeErasingQueryState) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 2, 1);
  std::promise<void> firstReadStarted;
  std::promise<void> unblockFirstRead;
  auto unblock = unblockFirstRead.get_future().share();
  std::atomic<uint32_t> secondReadCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-stop",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 1}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-stop",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-stop",
      [&](const std::string& path, std::optional<std::size_t>) {
        if (path.ends_with("first.parquet")) {
          firstReadStarted.set_value();
          return PrefetchReadFunction{
              [unblock](uint64_t, uint64_t size, uint8_t* destination) {
                unblock.wait();
                std::memset(destination, 0, size);
              }};
        }
        ++secondReadCount;
        return PrefetchReadFunction{
            [](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
            }};
      },
      broker,
      1,
      1);

  ASSERT_EQ(
      firstReadStarted.get_future().wait_for(5s), std::future_status::ready);
  auto waitingTake = std::async(std::launch::async, [&] {
    return ExecutorSplitPrefetch::take(
        &executor, "query-stop", "s3://bucket/second.parquet");
  });
  ASSERT_EQ(waitingTake.wait_for(50ms), std::future_status::timeout);

  auto cleanup = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::eraseQuery(&executor, "query-stop");
  });
  EXPECT_EQ(cleanup.wait_for(50ms), std::future_status::timeout);

  unblockFirstRead.set_value();
  EXPECT_EQ(cleanup.wait_for(5s), std::future_status::ready);
  cleanup.get();
  EXPECT_THROW(waitingTake.get(), std::runtime_error);
  EXPECT_EQ(secondReadCount.load(), 0);

  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, cacheHintSplitPreloadsCompleteReadyFirst) {
  folly::CPUThreadPoolExecutor executor(2);
  std::promise<void> firstRelease;
  std::promise<void> secondRelease;
  auto firstReleaseFuture = firstRelease.get_future().share();
  auto secondReleaseFuture = secondRelease.get_future().share();

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-ready-first",
      "first",
      1,
      [firstReleaseFuture] { firstReleaseFuture.wait(); },
      2,
      2);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-ready-first",
      "second",
      1,
      [secondReleaseFuture] { secondReleaseFuture.wait(); },
      2,
      2);

  const auto before = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  auto first = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor,
        "query-cache-ready-first",
        "first",
        CacheHintWaitMode::kSplitPreload);
  });
  auto second = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor,
        "query-cache-ready-first",
        "second",
        CacheHintWaitMode::kSplitPreload);
  });

  EXPECT_EQ(first.wait_for(50ms), std::future_status::timeout);
  EXPECT_EQ(second.wait_for(50ms), std::future_status::timeout);
  secondRelease.set_value();
  ASSERT_EQ(second.wait_for(5s), std::future_status::ready);
  second.get();
  EXPECT_EQ(first.wait_for(50ms), std::future_status::timeout);
  firstRelease.set_value();
  ASSERT_EQ(first.wait_for(5s), std::future_status::ready);
  first.get();

  const auto after = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  EXPECT_EQ(after.splitPreloadWaits - before.splitPreloadWaits, 2);
  EXPECT_EQ(after.splitPreloadReadyAtTake - before.splitPreloadReadyAtTake, 0);
  EXPECT_GT(after.splitPreloadWaitWallNanos, before.splitPreloadWaitWallNanos);

  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-ready-first");
}

TEST(ExecutorPrefetchTest, cacheHintRangeReadyRequestReleasesAdmissionWindow) {
  folly::CPUThreadPoolExecutor executor(2);
  std::promise<void> firstStarted;
  std::promise<void> unblockFirst;
  auto unblockFirstFuture = unblockFirst.get_future().share();
  std::promise<void> secondCompleted;

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-range-ready",
      "first",
      1,
      [&] {
        firstStarted.set_value();
        unblockFirstFuture.wait();
      },
      2,
      1);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-range-ready",
      "second",
      1,
      [&] { secondCompleted.set_value(); },
      2,
      1);

  ASSERT_EQ(firstStarted.get_future().wait_for(5s), std::future_status::ready);
  const auto before = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  const auto requestStart = std::chrono::steady_clock::now();
  ExecutorSplitPrefetch::requestCacheHint(
      &executor, "query-cache-range-ready", "first");
  EXPECT_LT(
      std::chrono::steady_clock::now() - requestStart, std::chrono::seconds(1));

  // The first load is still active, but finishing its scan lifecycle releases
  // admission bytes so a second file can start immediately.
  ExecutorSplitPrefetch::releaseCacheHint(
      &executor, "query-cache-range-ready", "first");
  ASSERT_EQ(
      secondCompleted.get_future().wait_for(5s), std::future_status::ready);
  ExecutorSplitPrefetch::requestCacheHint(
      &executor, "query-cache-range-ready", "second");
  ExecutorSplitPrefetch::releaseCacheHint(
      &executor, "query-cache-range-ready", "second");

  const auto after = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  EXPECT_EQ(after.rangeReadyRequests - before.rangeReadyRequests, 2);
  EXPECT_EQ(after.rangeReadyReleases - before.rangeReadyReleases, 2);
  EXPECT_EQ(after.scanWaits - before.scanWaits, 0);
  EXPECT_EQ(after.splitPreloadWaits - before.splitPreloadWaits, 0);

  unblockFirst.set_value();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-range-ready");
}

TEST(ExecutorPrefetchTest, cacheHintFirstLoadGroupPrecedesFullCompletion) {
  folly::CPUThreadPoolExecutor executor(1);
  auto firstLoadReady = std::make_shared<CacheHintFirstLoadSignal>();
  std::promise<void> unblockFullLoad;
  auto unblockFullLoadFuture = unblockFullLoad.get_future().share();
  std::atomic<bool> fullLoadCompleted{false};

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-first-load",
      "split",
      CacheHintRangeStats{
          .logicalRanges = 2,
          .logicalBytes = 2,
          .uniqueRanges = 2,
          .uniqueBytes = 2},
      [&] {
        unblockFullLoadFuture.wait();
        fullLoadCompleted.store(true, std::memory_order_relaxed);
      },
      1,
      2,
      firstLoadReady);

  const auto before = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  auto take = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor,
        "query-cache-first-load",
        "split",
        CacheHintWaitMode::kFirstLoadGroup);
  });
  EXPECT_EQ(take.wait_for(50ms), std::future_status::timeout);
  firstLoadReady->signal();
  ASSERT_EQ(take.wait_for(5s), std::future_status::ready);
  take.get();
  EXPECT_FALSE(fullLoadCompleted.load(std::memory_order_relaxed));

  const auto after = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
  EXPECT_EQ(after.firstLoadGroupWaits - before.firstLoadGroupWaits, 1);
  EXPECT_EQ(
      after.firstLoadGroupReadyAtTake - before.firstLoadGroupReadyAtTake, 0);
  EXPECT_GT(
      after.firstLoadGroupWaitWallNanos, before.firstLoadGroupWaitWallNanos);

  unblockFullLoad.set_value();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-cache-first-load");
  EXPECT_TRUE(fullLoadCompleted.load(std::memory_order_relaxed));
}

TEST(ExecutorPrefetchTest, cacheHintFirstLoadFailureDoesNotHang) {
  folly::CPUThreadPoolExecutor executor(1);
  auto firstLoadReady = std::make_shared<CacheHintFirstLoadSignal>();

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-first-load-failure",
      "split",
      CacheHintRangeStats{
          .logicalRanges = 1,
          .logicalBytes = 1,
          .uniqueRanges = 1,
          .uniqueBytes = 1},
      [] { throw std::runtime_error("failure before first load callback"); },
      1,
      1,
      firstLoadReady);

  auto take = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor,
        "query-cache-first-load-failure",
        "split",
        CacheHintWaitMode::kFirstLoadGroup);
  });
  ASSERT_EQ(take.wait_for(5s), std::future_status::ready);
  EXPECT_NO_THROW(take.get());
  ExecutorSplitPrefetch::eraseQuery(
      &executor, "query-cache-first-load-failure");
}

TEST(ExecutorPrefetchTest, cacheHintFirstLoadCancellationDoesNotHang) {
  folly::CPUThreadPoolExecutor executor(1);
  std::promise<void> blockerStarted;
  std::promise<void> unblockBlocker;
  auto unblockBlockerFuture = unblockBlocker.get_future().share();

  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-first-load-cancel",
      "blocker",
      1,
      [&] {
        blockerStarted.set_value();
        unblockBlockerFuture.wait();
      },
      1,
      2);
  ASSERT_EQ(
      blockerStarted.get_future().wait_for(5s), std::future_status::ready);

  auto firstLoadReady = std::make_shared<CacheHintFirstLoadSignal>();
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      "query-cache-first-load-cancel",
      "canceled-before-schedule",
      CacheHintRangeStats{
          .logicalRanges = 1,
          .logicalBytes = 1,
          .uniqueRanges = 1,
          .uniqueBytes = 1},
      [] {},
      1,
      2,
      firstLoadReady);

  auto take = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::takeCacheHint(
        &executor,
        "query-cache-first-load-cancel",
        "canceled-before-schedule",
        CacheHintWaitMode::kFirstLoadGroup);
  });
  EXPECT_EQ(take.wait_for(50ms), std::future_status::timeout);

  auto erase = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::eraseQuery(
        &executor, "query-cache-first-load-cancel");
  });
  ASSERT_EQ(take.wait_for(5s), std::future_status::ready);
  EXPECT_NO_THROW(take.get());

  unblockBlocker.set_value();
  ASSERT_EQ(erase.wait_for(5s), std::future_status::ready);
  erase.get();
}

TEST(ExecutorPrefetchTest, cacheHintQueryRegisteredSplitDecisionIsUniform) {
  folly::CPUThreadPoolExecutor coordinatorExecutor(1);
  folly::CPUThreadPoolExecutor ioExecutor(3);
  const std::string kQuery = "query-cache-registered-split-decision";
  for (uint32_t i = 0; i < 3; ++i) {
    const auto path = fmt::format("s3://bucket/registered-{}.parquet", i);
    ExecutorSplitPrefetch::registerSplit(
        &coordinatorExecutor, kQuery, path, {{path, 1}});
  }

  ExecutorSplitPrefetch::registerCacheHint(
      &ioExecutor, kQuery, "first", 4, [] {}, 3, 32);
  EXPECT_TRUE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(&ioExecutor, kQuery, 3));
  // The query-wide result is frozen; another split cannot choose a different
  // policy even if its caller supplies a different threshold.
  ExecutorSplitPrefetch::registerCacheHint(
      &ioExecutor, kQuery, "second", 4, [] {}, 3, 32);
  EXPECT_TRUE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
          &ioExecutor, kQuery, 100));

  ExecutorSplitPrefetch::eraseQuery(&ioExecutor, kQuery);
  // Query cleanup must evict the immutable fast-path decision. Reusing the
  // same id with a new registered-split population must recompute it.
  ExecutorSplitPrefetch::registerSplit(
      &coordinatorExecutor,
      kQuery,
      "s3://bucket/reused.parquet",
      {{"s3://bucket/reused.parquet", 1}});
  EXPECT_FALSE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(&ioExecutor, kQuery, 3));
  ExecutorSplitPrefetch::eraseQuery(&ioExecutor, kQuery);

  const std::string kSmallQuery = "query-cache-small-registered-split-count";
  for (uint32_t i = 0; i < 2; ++i) {
    const auto path = fmt::format("s3://bucket/small-{}.parquet", i);
    ExecutorSplitPrefetch::registerSplit(
        &coordinatorExecutor, kSmallQuery, path, {{path, 1}});
  }
  ExecutorSplitPrefetch::registerCacheHint(
      &ioExecutor, kSmallQuery, "small", 4, [] {}, 3, 32);
  EXPECT_FALSE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
          &ioExecutor, kSmallQuery, 3));
  ExecutorSplitPrefetch::eraseQuery(&ioExecutor, kSmallQuery);
}

TEST(ExecutorPrefetchTest, cacheHintExpectedSplitCountAvoidsRegistrationWait) {
  folly::CPUThreadPoolExecutor coordinatorExecutor(1);
  folly::CPUThreadPoolExecutor ioExecutor(1);
  const std::string kQuery = "query-cache-expected-split-count";

  // The coordinator knows the complete assignment before entering its
  // per-split registration loop. No registered QueryPrefetchState exists yet.
  ExecutorSplitPrefetch::setExpectedSplitCount(
      &coordinatorExecutor, kQuery, 600, 600);
  EXPECT_TRUE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
          &ioExecutor, kQuery, 600));
  // The first result remains query-wide even if a later caller supplies a
  // different threshold.
  EXPECT_TRUE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
          &ioExecutor, kQuery, 1000));
  ExecutorSplitPrefetch::eraseQuery(&ioExecutor, kQuery);

  ExecutorSplitPrefetch::setExpectedSplitCount(
      &coordinatorExecutor, kQuery, 599, 600);
  EXPECT_FALSE(
      ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
          &ioExecutor, kQuery, 600));
  ExecutorSplitPrefetch::eraseQuery(&ioExecutor, kQuery);
}

TEST(ExecutorPrefetchTest, cacheHintScopesRequestPressureByQuerySize) {
  folly::CPUThreadPoolExecutor executor(3);
  const std::string kLowQuery = "query-cache-low-request-pressure";
  const std::string kHighQuery = "query-cache-high-request-pressure";
  const std::string kWideQuery = "query-cache-wide-request-pressure";
  std::atomic<bool> lowObserved{true};
  std::atomic<bool> highObserved{false};
  std::atomic<bool> wideObserved{false};

  ExecutorSplitPrefetch::setExpectedSplitCount(&executor, kLowQuery, 649, 0, 2);
  ExecutorSplitPrefetch::setExpectedSplitCount(
      &executor, kHighQuery, 670, 0, 2);
  ExecutorSplitPrefetch::setExpectedSplitCount(
      &executor, kWideQuery, 649, 0, 6);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      kLowQuery,
      "low",
      1,
      [&] {
        lowObserved.store(
            nativeS3HighRequestPressureForCurrentThread(),
            std::memory_order_relaxed);
      },
      1,
      1);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      kHighQuery,
      "high",
      1,
      [&] {
        highObserved.store(
            nativeS3HighRequestPressureForCurrentThread(),
            std::memory_order_relaxed);
      },
      1,
      1);
  ExecutorSplitPrefetch::registerCacheHint(
      &executor,
      kWideQuery,
      "wide",
      1,
      [&] {
        wideObserved.store(
            nativeS3HighRequestPressureForCurrentThread(),
            std::memory_order_relaxed);
      },
      1,
      1);

  ExecutorSplitPrefetch::takeCacheHint(&executor, kLowQuery, "low");
  ExecutorSplitPrefetch::takeCacheHint(&executor, kHighQuery, "high");
  ExecutorSplitPrefetch::takeCacheHint(&executor, kWideQuery, "wide");
  EXPECT_FALSE(lowObserved.load(std::memory_order_relaxed));
  EXPECT_TRUE(highObserved.load(std::memory_order_relaxed));
  EXPECT_TRUE(wideObserved.load(std::memory_order_relaxed));
  EXPECT_FALSE(nativeS3HighRequestPressureForCurrentThread());

  ExecutorSplitPrefetch::eraseQuery(&executor, kLowQuery);
  ExecutorSplitPrefetch::eraseQuery(&executor, kHighQuery);
  ExecutorSplitPrefetch::eraseQuery(&executor, kWideQuery);
}

TEST(ExecutorPrefetchTest, requestPressureExpectedSplitBand) {
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(499, 500, 600, 670));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(500, 500, 600, 670));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(600, 500, 600, 670));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(601, 500, 600, 670));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(669, 500, 600, 670));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(670, 500, 600, 670));

  // Same split count, different scan topology: two-table scans retain the
  // base profile while a wider scan graph fills the bounded/large-band gap.
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(600, 500, 600, 670, 2));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(601, 500, 600, 670, 2));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(601, 500, 600, 670, 3));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(649, 500, 600, 670, 2));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(649, 500, 600, 670, 3));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(649, 500, 600, 670, 6));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(669, 500, 600, 670, 2));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(669, 500, 600, 670, 3));
  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(499, 500, 600, 670, 6));

  // The legacy single-threshold configuration gets the same topology-aware
  // gap without requiring two additional environment variables.
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(649, 670, 0, 0, 6));

  EXPECT_FALSE(useHighRequestPressureForExpectedSplits(1000, 0, 0, 0));
  EXPECT_TRUE(useHighRequestPressureForExpectedSplits(1000, 500, 0, 0));
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
