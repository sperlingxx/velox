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

#pragma once

#include <folly/Executor.h>
#include <folly/container/F14Set.h>

#include <atomic>
#include <functional>
#include <mutex>
#include "velox/common/caching/FileGroupStats.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/common/InputStream.h"

namespace facebook::velox::dwio::common {

struct CacheHintDemandStats {
  uint64_t firstHitRanges{0};
  uint64_t firstHitBytes{0};
  uint64_t missRanges{0};
  uint64_t missBytes{0};
  uint64_t remoteDuplicateRanges{0};
  uint64_t remoteDuplicateBytes{0};
  uint64_t sizeMismatchRanges{0};
  uint64_t sizeMismatchBytes{0};
};

CacheHintDemandStats cacheHintDemandStats();

struct CacheHintAsyncLoadStats {
  uint64_t makePinsCalls{0};
  uint64_t makePinsEntries{0};
  uint64_t makePinsWallNanos{0};
  uint64_t makePinsActive{0};
  uint64_t makePinsMaxActive{0};
  uint64_t submitCalls{0};
  uint64_t submittedRequests{0};
  uint64_t submitWallNanos{0};
  uint64_t submitActive{0};
  uint64_t submitMaxActive{0};
  uint64_t waitCalls{0};
  uint64_t waitWallNanos{0};
  uint64_t waitActive{0};
  uint64_t waitMaxActive{0};
};

CacheHintAsyncLoadStats cacheHintAsyncLoadStats();

/// Immutable physical-column boundaries shared by hint and demand streams.
/// Cache keys are load-quantum pieces relative to each physical chunk, not to
/// the file. This makes the identity stable across projections while avoiding
/// file-global head and tail padding.
class CacheRegionPlan {
 public:
  static std::shared_ptr<CacheRegionPlan> create(
      std::vector<velox::common::Region> regions,
      uint64_t fileSize,
      uint64_t loadQuantum);

  const std::vector<velox::common::Region>& cacheRegions() const {
    return cacheRegions_;
  }

  velox::common::Region regionForPosition(
      const velox::common::Region& demandRegion,
      uint64_t absolutePosition) const;

  bool isPlannedKey(const velox::common::Region& region) const;

  void markPrefetchComplete() {
    prefetchComplete_.store(true, std::memory_order_release);
  }

  void recordDemand(
      const velox::common::Region& region,
      bool cacheHit,
      bool sizeMismatch) const;

 private:
  CacheRegionPlan(
      std::vector<velox::common::Region> regions,
      std::vector<velox::common::Region> cacheRegions,
      uint64_t loadQuantum)
      : regions_(std::move(regions)),
        cacheRegions_(std::move(cacheRegions)),
        loadQuantum_(loadQuantum) {}

  const std::vector<velox::common::Region> regions_;
  const std::vector<velox::common::Region> cacheRegions_;
  const uint64_t loadQuantum_;
  std::atomic<bool> prefetchComplete_{false};
  mutable std::mutex demandMutex_;
  mutable folly::F14FastSet<uint64_t> observedDemandKeys_;
};

struct CacheRequest {
  CacheRequest(
      cache::RawFileCacheKey _key,
      uint64_t _size,
      cache::TrackingId _trackingId)
      : key(_key), size(_size), trackingId(_trackingId) {}

  cache::RawFileCacheKey key;
  uint64_t size;
  cache::TrackingId trackingId;
  cache::CachePin pin;
  cache::SsdPin ssdPin;

  /// True if this should be coalesced into a CoalescedLoad with other nearby
  /// requests with a similar load probability. This is false for sparsely
  /// accessed large columns where hitting one piece should not load the
  /// adjacent pieces.
  bool coalesces{true};
  const SeekableInputStream* stream;
};

class CachedBufferedInput : public BufferedInput {
 public:
  CachedBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      const MetricsLogPtr& metricsLog,
      StringIdLease fileNum,
      cache::AsyncDataCache* cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      StringIdLease groupId,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      folly::Executor* executor,
      const io::ReaderOptions& readerOptions,
      folly::F14FastMap<std::string, std::string> fileReadOps = {})
      : BufferedInput(
            std::move(readFile),
            readerOptions.memoryPool(),
            metricsLog,
            ioStatistics.get(),
            ioStats.get(),
            kMaxMergeDistance,
            std::nullopt,
            std::move(fileReadOps),
            readerOptions.cacheable()),
        cache_(cache),
        fileNum_(std::move(fileNum)),
        tracker_(std::move(tracker)),
        groupId_(std::move(groupId)),
        ioStatistics_(std::move(ioStatistics)),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        fileSize_(input_->getLength()),
        options_(readerOptions) {
    VELOX_CHECK_NOT_NULL(cache_, "CachedBufferedInput requires a cache");
    checkLoadQuantum();
  }

  CachedBufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      StringIdLease fileNum,
      cache::AsyncDataCache* cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      StringIdLease groupId,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      folly::Executor* executor,
      const io::ReaderOptions& readerOptions)
      : BufferedInput(std::move(input), readerOptions.memoryPool()),
        cache_(cache),
        fileNum_(std::move(fileNum)),
        tracker_(std::move(tracker)),
        groupId_(std::move(groupId)),
        ioStatistics_(std::move(ioStatistics)),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        fileSize_(input_->getLength()),
        options_(readerOptions) {
    VELOX_CHECK_NOT_NULL(cache_, "CachedBufferedInput requires a cache");
    checkLoadQuantum();
  }

  ~CachedBufferedInput() override {
    for (auto& load : coalescedLoads_) {
      load->cancel();
    }
    if (!options_.cacheable() && !preloadPin_.empty()) {
      preloadPin_.checkedEntry()->makeEvictable();
    }
  }

  std::unique_ptr<SeekableInputStream> enqueue(
      velox::common::Region region,
      const StreamIdentifier* sid) override;

  void preload() override;

  bool preloaded() const override {
    return !preloadPin_.empty();
  }

  void load(const LogType /*unused*/) override;

  bool isBuffered(uint64_t /*unused*/, uint64_t /*unused*/) const override;

  std::unique_ptr<SeekableInputStream>
  read(uint64_t offset, uint64_t length, LogType logType) const override;

  /// Returns an exact demand stream whose cache keys follow 'plan'. The plan
  /// is immutable and owned by the stream, so independent preparation jobs do
  /// not mutate this BufferedInput's shared hint state.
  std::unique_ptr<SeekableInputStream> readWithCacheRegionPlan(
      uint64_t offset,
      uint64_t length,
      LogType logType,
      std::shared_ptr<CacheRegionPlan> plan) const;

  /// Schedules load of 'region' on 'executor_'. Fails silently if no memory or
  /// if shouldPreload() is false.
  bool prefetch(velox::common::Region region);

  /// Loads a group of best-effort prefetch hints and waits until all loads
  /// which were admitted by the cache have completed. Grouping the regions in
  /// one load preserves normal cache coalescing and the underlying ReadFile's
  /// executor-wide I/O admission policy.
  /// If 'inlineLoad' is true, the caller executes each CoalescedLoad instead
  /// of scheduling it back onto 'executor_'. This is useful when the caller is
  /// already running on a dedicated prefetch pool and will wait synchronously.
  void prefetchSync(
      const std::vector<velox::common::Region>& regions,
      bool inlineLoad = false);

  /// Builds cache keys and CoalescedLoad objects for 'regions' and returns a
  /// callable which performs and waits for the physical reads. This allows a
  /// producer to prepare many independent files before executor workers begin
  /// blocking on their first S3 request.
  std::function<void()> preparePrefetch(
      const std::vector<velox::common::Region>& regions,
      bool inlineLoad = true,
      bool preallocatePins = false,
      bool asyncPhysicalGroups = false);

  std::function<void()> preparePrefetch(
      const std::vector<velox::common::Region>& regions,
      bool inlineLoad,
      bool preallocatePins,
      bool asyncPhysicalGroups,
      std::function<void()> firstLoadReady);

  std::vector<velox::common::Region> canonicalizeRegions(
      const std::vector<velox::common::Region>& regions) const;

  std::shared_ptr<CacheRegionPlan> makeCacheRegionPlan(
      const std::vector<velox::common::Region>& regions) const;

  uint64_t cacheFileNum() const {
    return fileNum_.id();
  }

  bool shouldPreload(int32_t numPages = 0) override;

  bool shouldPrefetchStripes() const override {
    return true;
  }

  void setNumStripes(int32_t numStripes) override {
    auto stats = tracker_->fileGroupStats();
    if (stats) {
      stats->recordFile(fileNum_.id(), groupId_.id(), numStripes);
    }
  }

  virtual std::unique_ptr<BufferedInput> clone() const override {
    return std::make_unique<CachedBufferedInput>(
        input_,
        fileNum_,
        cache_,
        tracker_,
        groupId_,
        ioStatistics_,
        ioStats_,
        executor_,
        options_);
  }

  /// Creates a clone that reads through the cache but marks entries as
  /// immediately evictable on destruction (cacheable=false). Use when the
  /// caller has its own caching layer (e.g., MetadataCache) and does not need
  /// AsyncDataCache to retain the raw bytes.
  std::unique_ptr<CachedBufferedInput> cloneNonCacheable() const {
    auto nonCacheableOptions = options_;
    nonCacheableOptions.setCacheable(false);
    return std::make_unique<CachedBufferedInput>(
        input_,
        fileNum_,
        cache_,
        tracker_,
        groupId_,
        ioStatistics_,
        ioStats_,
        executor_,
        nonCacheableOptions);
  }

  cache::AsyncDataCache* cache() const {
    return cache_;
  }

  uint64_t cacheLoadQuantum() const {
    return options_.loadQuantum();
  }

  bool hasCache() const override {
    return true;
  }

  void cacheRegion(uint64_t offset, uint64_t length, std::string_view data)
      override;

  void cacheRegion(
      uint64_t offset,
      uint64_t length,
      const folly::IOBuf& buffer,
      uint64_t bufferOffset) override;

  std::optional<CachedRegion> findCachedRegion(uint64_t offset) const override;

  /// Returns the CoalescedLoad that contains the correlated loads for 'stream'
  /// or nullptr if none. Returns nullptr on all but first call for 'stream'
  /// since the load is to be triggered by the first access.
  std::shared_ptr<cache::CoalescedLoad> coalescedLoad(
      const SeekableInputStream* stream);

  folly::Executor* executor() const override {
    return executor_;
  }

  uint64_t nextFetchSize() const override {
    VELOX_NYI();
  }

  /// Resets the buffered input for reuse across different operations.
  void reset() override;

  const std::vector<std::shared_ptr<cache::CoalescedLoad>>&
  testingCoalescedLoads() const {
    return coalescedLoads_;
  }

  size_t testingStreamToCoalescedLoadSize() const {
    return streamToCoalescedLoad_.rlock()->size();
  }

 private:
  template <bool kSsd>
  std::vector<int32_t> groupRequests(
      const std::vector<CacheRequest*>& requests,
      bool prefetch) const;

  // Makes a CoalescedLoad for 'requests' to be read together, coalescing IO is
  // appropriate. If 'prefetch' is set, schedules the CoalescedLoad on
  // 'executor_'. Links the CoalescedLoad to all CacheInputStreams that it
  // concerns.
  void readRegion(const std::vector<CacheRequest*>& requests, bool prefetch);

  // Read coalesced regions.  Regions are grouped together using `groupEnds'.
  // For example if there are 5 regions, 1 and 2 are coalesced together and 3,
  // 4, 5 are coalesced together, we will have {2, 5} in `groupEnds'.
  void readRegions(
      const std::vector<CacheRequest*>& requests,
      bool prefetch,
      const std::vector<int32_t>& groupEnds);

  template <bool kSsd>
  void makeLoads(std::vector<CacheRequest*> requests[2]);

  // We only support up to 8MB load quantum size on SSD and there is no need for
  // larger SSD read size performance wise.
  void checkLoadQuantum() {
    if (cache_->ssdCache() != nullptr) {
      VELOX_CHECK_LE(
          options_.loadQuantum(),
          1 << cache::SsdRun::kSizeBits,
          "Load quantum exceeded SSD cache entry size limit.");
    }
  }

  cache::AsyncDataCache* const cache_;
  const StringIdLease fileNum_;
  const std::shared_ptr<cache::ScanTracker> tracker_;
  const StringIdLease groupId_;
  const std::shared_ptr<IoStatistics> ioStatistics_;
  const std::shared_ptr<velox::IoStats> ioStats_;
  folly::Executor* const executor_;
  const uint64_t fileSize_;
  const io::ReaderOptions options_;
  folly::Synchronized<std::shared_ptr<CacheRegionPlan>> cacheRegionPlan_;

  // Regions that are candidates for loading.
  std::vector<CacheRequest> requests_;

  // Map from stream to its coalesced load.
  folly::Synchronized<folly::F14FastMap<
      const SeekableInputStream*,
      std::shared_ptr<cache::CoalescedLoad>>>
      streamToCoalescedLoad_;

  // All distinct coalesced loads.
  std::vector<std::shared_ptr<cache::CoalescedLoad>> coalescedLoads_;

  // Set only while prefetchSync() constructs its loads. CachedBufferedInput is
  // used sequentially, so this flag does not need cross-thread synchronization.
  bool inlinePrefetchLoad_{false};

  // Holds the whole-file cache entry alive for the lifetime of this input.
  // Set by preload(), used by CacheInputStream to serve sub-region reads.
  cache::CachePin preloadPin_;
};

} // namespace facebook::velox::dwio::common
