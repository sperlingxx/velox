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

#include "velox/dwio/common/CachedBufferedInput.h"
#include "folly/io/Cursor.h"
#include "velox/common/Casts.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/CacheInputStream.h"

#include <folly/container/F14Set.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>

DECLARE_int32(cache_prefetch_min_pct);

using ::facebook::velox::common::Region;

namespace facebook::velox::dwio::common {

namespace {
std::atomic<uint64_t> hintDemandFirstHitRanges{0};
std::atomic<uint64_t> hintDemandFirstHitBytes{0};
std::atomic<uint64_t> hintDemandMissRanges{0};
std::atomic<uint64_t> hintDemandMissBytes{0};
std::atomic<uint64_t> hintDemandRemoteDuplicateRanges{0};
std::atomic<uint64_t> hintDemandRemoteDuplicateBytes{0};
std::atomic<uint64_t> hintDemandSizeMismatchRanges{0};
std::atomic<uint64_t> hintDemandSizeMismatchBytes{0};

std::atomic<uint64_t> hintAsyncMakePinsCalls{0};
std::atomic<uint64_t> hintAsyncMakePinsEntries{0};
std::atomic<uint64_t> hintAsyncMakePinsWallNanos{0};
std::atomic<uint64_t> hintAsyncMakePinsActive{0};
std::atomic<uint64_t> hintAsyncMakePinsMaxActive{0};
std::atomic<uint64_t> hintAsyncSubmitCalls{0};
std::atomic<uint64_t> hintAsyncSubmittedRequests{0};
std::atomic<uint64_t> hintAsyncSubmitWallNanos{0};
std::atomic<uint64_t> hintAsyncSubmitActive{0};
std::atomic<uint64_t> hintAsyncSubmitMaxActive{0};
std::atomic<uint64_t> hintAsyncWaitCalls{0};
std::atomic<uint64_t> hintAsyncWaitWallNanos{0};
std::atomic<uint64_t> hintAsyncWaitActive{0};
std::atomic<uint64_t> hintAsyncWaitMaxActive{0};

uint32_t hintAsyncMakePinsConcurrency() {
  static const uint32_t value = [] {
    constexpr const char* kEnv = "GLUTEN_CUDF_CACHE_HINT_MAKE_PINS_CONCURRENCY";
    const auto* text = std::getenv(kEnv);
    if (text == nullptr || text[0] == '\0') {
      return 1U;
    }
    VELOX_CHECK_NE(text[0], '-', "{} must be a positive integer", kEnv);
    char* end = nullptr;
    errno = 0;
    const auto parsed = std::strtoul(text, &end, 10);
    VELOX_CHECK(
        errno == 0 && end != text && *end == '\0' && parsed > 0 &&
            parsed <= 1024,
        "{} must be in [1, 1024], got '{}'",
        kEnv,
        text);
    return static_cast<uint32_t>(parsed);
  }();
  return value;
}

// AsyncDataCache entry allocation is internally sharded, but hundreds of
// cache-hint workers entering allocation together cause severe allocator
// backoff and starve physical-read submission. Bound this stage separately
// from the much larger pool that waits for network completion.
class HintAsyncMakePinsGate {
 public:
  void acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(
        lock, [&] { return active_ < hintAsyncMakePinsConcurrency(); });
    ++active_;
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_CHECK_GT(active_, 0);
      --active_;
    }
    condition_.notify_one();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t active_{0};
};

class HintAsyncMakePinsSlot {
 public:
  explicit HintAsyncMakePinsSlot(HintAsyncMakePinsGate& gate) : gate_(gate) {
    gate_.acquire();
  }

  ~HintAsyncMakePinsSlot() {
    gate_.release();
  }

 private:
  HintAsyncMakePinsGate& gate_;
};

HintAsyncMakePinsGate hintAsyncMakePinsGate;

void updateMax(std::atomic<uint64_t>& maximum, uint64_t value) {
  auto current = maximum.load(std::memory_order_relaxed);
  while (current < value &&
         !maximum.compare_exchange_weak(
             current, value, std::memory_order_relaxed)) {
  }
}

class AsyncLoadStageTimer {
 public:
  AsyncLoadStageTimer(
      std::atomic<uint64_t>& calls,
      std::atomic<uint64_t>& wallNanos,
      std::atomic<uint64_t>& active,
      std::atomic<uint64_t>& maxActive)
      : wallNanos_(wallNanos), active_(active) {
    calls.fetch_add(1, std::memory_order_relaxed);
    const auto nowActive = active_.fetch_add(1, std::memory_order_relaxed) + 1;
    updateMax(maxActive, nowActive);
  }

  ~AsyncLoadStageTimer() {
    wallNanos_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count(),
        std::memory_order_relaxed);
    active_.fetch_sub(1, std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>& wallNanos_;
  std::atomic<uint64_t>& active_;
  const std::chrono::steady_clock::time_point start_{
      std::chrono::steady_clock::now()};
};
} // namespace

CacheHintDemandStats cacheHintDemandStats() {
  return {
      hintDemandFirstHitRanges.load(std::memory_order_relaxed),
      hintDemandFirstHitBytes.load(std::memory_order_relaxed),
      hintDemandMissRanges.load(std::memory_order_relaxed),
      hintDemandMissBytes.load(std::memory_order_relaxed),
      hintDemandRemoteDuplicateRanges.load(std::memory_order_relaxed),
      hintDemandRemoteDuplicateBytes.load(std::memory_order_relaxed),
      hintDemandSizeMismatchRanges.load(std::memory_order_relaxed),
      hintDemandSizeMismatchBytes.load(std::memory_order_relaxed)};
}

CacheHintAsyncLoadStats cacheHintAsyncLoadStats() {
  return {
      hintAsyncMakePinsCalls.load(std::memory_order_relaxed),
      hintAsyncMakePinsEntries.load(std::memory_order_relaxed),
      hintAsyncMakePinsWallNanos.load(std::memory_order_relaxed),
      hintAsyncMakePinsActive.load(std::memory_order_relaxed),
      hintAsyncMakePinsMaxActive.load(std::memory_order_relaxed),
      hintAsyncSubmitCalls.load(std::memory_order_relaxed),
      hintAsyncSubmittedRequests.load(std::memory_order_relaxed),
      hintAsyncSubmitWallNanos.load(std::memory_order_relaxed),
      hintAsyncSubmitActive.load(std::memory_order_relaxed),
      hintAsyncSubmitMaxActive.load(std::memory_order_relaxed),
      hintAsyncWaitCalls.load(std::memory_order_relaxed),
      hintAsyncWaitWallNanos.load(std::memory_order_relaxed),
      hintAsyncWaitActive.load(std::memory_order_relaxed),
      hintAsyncWaitMaxActive.load(std::memory_order_relaxed)};
}

std::shared_ptr<CacheRegionPlan> CacheRegionPlan::create(
    std::vector<velox::common::Region> regions,
    uint64_t fileSize,
    uint64_t loadQuantum) {
  VELOX_CHECK_GT(loadQuantum, 0);
  std::erase_if(regions, [](const auto& region) { return region.length == 0; });
  VELOX_CHECK(!regions.empty(), "Cache region plan is empty");
  std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) {
    return std::tie(a.offset, a.length) < std::tie(b.offset, b.length);
  });
  uint64_t previousEnd = 0;
  bool first = true;
  std::vector<velox::common::Region> keys;
  for (const auto& region : regions) {
    VELOX_CHECK_LE(region.offset, fileSize);
    VELOX_CHECK_LE(region.length, fileSize - region.offset);
    VELOX_CHECK(
        first || region.offset >= previousEnd,
        "Overlapping cache plan regions: previous end {}, next offset {}",
        previousEnd,
        region.offset);
    first = false;
    previousEnd = region.offset + region.length;
    for (uint64_t offset = 0; offset < region.length; offset += loadQuantum) {
      keys.push_back(
          {region.offset + offset,
           std::min<uint64_t>(loadQuantum, region.length - offset)});
    }
  }
  return std::shared_ptr<CacheRegionPlan>(
      new CacheRegionPlan(std::move(regions), std::move(keys), loadQuantum));
}

velox::common::Region CacheRegionPlan::regionForPosition(
    const velox::common::Region& demandRegion,
    uint64_t absolutePosition) const {
  VELOX_CHECK_GE(absolutePosition, demandRegion.offset);
  VELOX_CHECK_LT(absolutePosition, demandRegion.offset + demandRegion.length);
  const auto it = std::upper_bound(
      regions_.begin(),
      regions_.end(),
      absolutePosition,
      [](uint64_t position, const auto& region) {
        return position < region.offset;
      });
  if (it != regions_.begin()) {
    const auto& planned = *std::prev(it);
    const auto plannedEnd = planned.offset + planned.length;
    if (absolutePosition < plannedEnd) {
      const auto offset = planned.offset +
          ((absolutePosition - planned.offset) / loadQuantum_) * loadQuantum_;
      return {offset, std::min<uint64_t>(loadQuantum_, plannedEnd - offset)};
    }
  }
  // Metadata and any unplanned demand stay exact. Do not expand these reads to
  // a file-global quantum. Stop at the next planned boundary so a merged
  // regular read can switch to stable planned keys when it reaches one.
  auto end = demandRegion.offset + demandRegion.length;
  if (it != regions_.end()) {
    end = std::min(end, it->offset);
  }
  return {absolutePosition, end - absolutePosition};
}

bool CacheRegionPlan::isPlannedKey(const velox::common::Region& region) const {
  const auto it = std::lower_bound(
      cacheRegions_.begin(),
      cacheRegions_.end(),
      region.offset,
      [](const auto& candidate, uint64_t offset) {
        return candidate.offset < offset;
      });
  return it != cacheRegions_.end() && it->offset == region.offset &&
      it->length == region.length;
}

void CacheRegionPlan::recordDemand(
    const velox::common::Region& region,
    bool cacheHit,
    bool sizeMismatch) const {
  if (!isPlannedKey(region)) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(demandMutex_);
    if (!observedDemandKeys_.insert(region.offset).second) {
      return;
    }
  }
  if (sizeMismatch) {
    hintDemandSizeMismatchRanges.fetch_add(1, std::memory_order_relaxed);
    hintDemandSizeMismatchBytes.fetch_add(
        region.length, std::memory_order_relaxed);
  }
  if (cacheHit && !sizeMismatch) {
    hintDemandFirstHitRanges.fetch_add(1, std::memory_order_relaxed);
    hintDemandFirstHitBytes.fetch_add(region.length, std::memory_order_relaxed);
    return;
  }
  hintDemandMissRanges.fetch_add(1, std::memory_order_relaxed);
  hintDemandMissBytes.fetch_add(region.length, std::memory_order_relaxed);
  if (prefetchComplete_.load(std::memory_order_acquire)) {
    hintDemandRemoteDuplicateRanges.fetch_add(1, std::memory_order_relaxed);
    hintDemandRemoteDuplicateBytes.fetch_add(
        region.length, std::memory_order_relaxed);
  }
}

using cache::CachePin;
using cache::CoalescedLoad;
using cache::RawFileCacheKey;
using cache::ScanTracker;
using cache::SsdFile;
using cache::SsdPin;
using cache::TrackingId;
using memory::MemoryAllocator;

std::unique_ptr<SeekableInputStream> CachedBufferedInput::enqueue(
    Region region,
    const StreamIdentifier* sid = nullptr) {
  if (region.length == 0) {
    return std::make_unique<SeekableArrayInputStream>(
        static_cast<const char*>(nullptr), 0);
  }

  TrackingId id;
  if (sid != nullptr) {
    id = TrackingId(sid->getId());
  }
  VELOX_CHECK_LE(region.offset + region.length, fileSize_);
  if (tracker_ != nullptr) {
    tracker_->recordReference(id, region.length, fileNum_.id(), groupId_.id());
  }
  auto stream = std::make_unique<CacheInputStream>(
      this,
      ioStatistics_.get(),
      region,
      input_,
      fileNum_.id(),
      options_.cacheable(),
      tracker_,
      id,
      groupId_.id(),
      options_.loadQuantum(),
      *cacheRegionPlan_.rlock());
  if (preloaded()) {
    // Data is already in cache. Give the stream its own pin copy so it can
    // outlive this CachedBufferedInput and skip all loading/prefetch logic.
    stream->setPreloadedPin(preloadPin_);
  } else {
    requests_.emplace_back(
        RawFileCacheKey{fileNum_.id(), region.offset}, region.length, id);
    requests_.back().stream = stream.get();
  }
  return stream;
}

bool CachedBufferedInput::isBuffered(uint64_t /*offset*/, uint64_t /*length*/)
    const {
  // When preloaded, the entire file content is already in cache, so any
  // region within the file is considered buffered and can be served without
  // additional I/O.
  return preloaded();
}

bool CachedBufferedInput::shouldPreload(int32_t numPages) {
  // True if after scheduling this for preload, half the capacity would be in a
  // loading but not yet accessed state.
  if (requests_.empty() && (numPages == 0)) {
    return false;
  }
  for (const auto& request : requests_) {
    numPages += memory::AllocationTraits::numPages(
        std::min<int32_t>(request.size, options_.loadQuantum()));
  }
  const auto cachePages = cache_->cachedPages();
  auto* allocator = cache_->allocator();
  const auto maxPages =
      memory::AllocationTraits::numPages(allocator->capacity());
  const auto allocatedPages = allocator->numAllocated();
  if (numPages < maxPages - allocatedPages) {
    // There is free space for the read-ahead.
    return true;
  }
  const auto prefetchPages = cache_->incrementPrefetchPages(0);
  if (numPages + prefetchPages < cachePages / 2) {
    // The planned prefetch plus other prefetches are under half the cache.
    return true;
  }
  return false;
}

namespace {

bool isPrefetchPct(int32_t pct) {
  return pct >= FLAGS_cache_prefetch_min_pct;
}

std::vector<CacheRequest*> makeRequestParts(
    CacheRequest& request,
    const cache::TrackingData& trackingData,
    int32_t loadQuantum,
    std::vector<std::unique_ptr<CacheRequest>>& extraRequests) {
  if (request.size <= loadQuantum) {
    return {&request};
  }

  // Large columns will be part of coalesced reads if the access frequency
  // qualifies for read ahead and if over 80% of the column gets accessed. Large
  // metadata columns (empty no trackingData) always coalesce.
  const bool prefetchOne =
      request.trackingId.id() == StreamIdentifier::sequentialFile().id_;
  const auto readDensity =
      trackingData.readBytes / (1 + trackingData.referencedBytes);
  const auto readPct = 100 * readDensity;
  const bool prefetch = trackingData.referencedBytes > 0 &&
      isPrefetchPct(readPct) && readDensity >= 0.8;
  std::vector<CacheRequest*> parts;
  for (uint64_t offset = 0; offset < request.size; offset += loadQuantum) {
    const int32_t size = std::min<int32_t>(loadQuantum, request.size - offset);
    extraRequests.push_back(
        std::make_unique<CacheRequest>(
            RawFileCacheKey{request.key.fileNum, request.key.offset + offset},
            size,
            request.trackingId));
    parts.push_back(extraRequests.back().get());
    parts.back()->coalesces = prefetch;
    if (prefetchOne) {
      break;
    }
  }
  return parts;
}

template <bool kSsd>
uint64_t getOffset(const CacheRequest& request) {
  if constexpr (kSsd) {
    VELOX_DCHECK(!request.ssdPin.empty());
    return request.ssdPin.run().offset();
  } else {
    return request.key.offset;
  }
}

template <bool kSsd>
bool lessThan(const CacheRequest* left, const CacheRequest* right) {
  auto leftOffset = getOffset<kSsd>(*left);
  auto rightOffset = getOffset<kSsd>(*right);
  return leftOffset < rightOffset ||
      (leftOffset == rightOffset && left->size > right->size);
}

} // namespace

void CachedBufferedInput::preload() {
  VELOX_CHECK(preloadPin_.empty(), "preload() called more than once");
  VELOX_CHECK(requests_.empty(), "preload() must be called before enqueue()");
  cache::RawFileCacheKey key{fileNum_.id(), 0};
  folly::SemiFuture<bool> waitFuture(false);
  do {
    preloadPin_ =
        cache_->findOrCreate(key, fileSize_, /*contiguous=*/false, &waitFuture);
    if (preloadPin_.empty()) {
      uint64_t waitUs{0};
      {
        MicrosecondWallTimer timer(&waitUs);
        std::move(waitFuture).wait();
      }
      ioStatistics_->queryThreadIoLatencyUs().increment(waitUs);
      ioStatistics_->cacheWaitLatencyUs().increment(waitUs);
    }
  } while (preloadPin_.empty());

  auto* entry = preloadPin_.checkedEntry();
  if (!entry->getAndClearFirstUseFlag()) {
    // Already loaded by another concurrent query.
    ioStatistics_->ramHit().increment(fileSize_);
  }
  if (!entry->isExclusive()) {
    // Cache hit — already loaded.
    return;
  }

  entry->setGroupId(groupId_.id());
  entry->setTrackingId(
      cache::TrackingId(StreamIdentifier::sequentialFile().id_));
  auto ranges = entry->dataRanges(fileSize_);
  uint64_t storageReadUs{0};
  {
    MicrosecondWallTimer timer(&storageReadUs);
    input_->read(ranges, 0, LogType::FILE);
  }
  ioStatistics_->read().increment(fileSize_);
  ioStatistics_->incRawBytesRead(fileSize_);
  ioStatistics_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStatistics_->storageReadLatencyUs().increment(storageReadUs);
  ioStatistics_->incTotalScanTimeNs(storageReadUs * 1'000);
  entry->setExclusiveToShared(options_.cacheable());
}

void CachedBufferedInput::load(const LogType /*unused*/) {
  // 'requests_ is cleared on exit.
  auto requests = std::move(requests_);
  cache::SsdFile* ssdFile{nullptr};
  auto* ssdCache = cache_->ssdCache();
  if (ssdCache != nullptr) {
    ssdFile = &ssdCache->file(fileNum_.id());
  }

  // Extra requests made for pre-loadable regions that are larger than
  // 'loadQuantum'.
  std::vector<std::unique_ptr<CacheRequest>> extraRequests;
  std::vector<CacheRequest*> storageLoad[2];
  std::vector<CacheRequest*> ssdLoad[2];
  for (auto& request : requests) {
    cache::TrackingData trackingData;
    const bool prefetchAnyway = request.trackingId.empty() ||
        request.trackingId.id() == StreamIdentifier::sequentialFile().id_;
    if (!prefetchAnyway && (tracker_ != nullptr)) {
      trackingData = tracker_->trackingData(request.trackingId);
    }
    const int loadIndex =
        (prefetchAnyway || isPrefetchPct(adjustedReadPct(trackingData))) ? 1
                                                                         : 0;
    auto parts = makeRequestParts(
        request, trackingData, options_.loadQuantum(), extraRequests);
    for (auto part : parts) {
      if (cache_->exists(part->key)) {
        continue;
      }
      if (ssdFile != nullptr) {
        part->ssdPin = ssdFile->find(part->key);
        if (!part->ssdPin.empty() && part->ssdPin.run().size() < part->size) {
          LOG(WARNING) << "Ignoring SSD shorter than requested: "
                       << part->ssdPin.run().size() << " vs " << part->size;
          part->ssdPin.clear();
        }
        if (!part->ssdPin.empty()) {
          ssdLoad[loadIndex].push_back(part);
          continue;
        }
      }
      storageLoad[loadIndex].push_back(part);
    }
  }

  std::sort(storageLoad[0].begin(), storageLoad[0].end(), lessThan<false>);
  std::sort(storageLoad[1].begin(), storageLoad[1].end(), lessThan<false>);
  std::sort(ssdLoad[0].begin(), ssdLoad[0].end(), lessThan<true>);
  std::sort(ssdLoad[1].begin(), ssdLoad[1].end(), lessThan<true>);
  makeLoads<false>(storageLoad);
  makeLoads<true>(ssdLoad);
}

template <bool kSsd>
void CachedBufferedInput::makeLoads(std::vector<CacheRequest*> requests[2]) {
  std::vector<int32_t> groupEnds[2];
  groupEnds[1] = groupRequests<kSsd>(requests[1], true);
  moveCoalesced(
      requests[1],
      groupEnds[1],
      requests[0],
      [](auto* request) { return getOffset<kSsd>(*request); },
      [](auto* request) { return getOffset<kSsd>(*request) + request->size; });
  groupEnds[0] = groupRequests<kSsd>(requests[0], false);
  readRegions(requests[1], true, groupEnds[1]);
  readRegions(requests[0], false, groupEnds[0]);
}

template <bool kSsd>
std::vector<int32_t> CachedBufferedInput::groupRequests(
    const std::vector<CacheRequest*>& requests,
    bool prefetch) const {
  if (requests.empty() || (requests.size() < 2 && !prefetch)) {
    return {};
  }
  const int32_t maxDistance = kSsd ? 20'000 : options_.maxCoalesceDistance();

  // Combine adjacent short reads.
  int64_t coalescedBytes = 0;
  std::vector<int32_t> ends;
  ends.reserve(requests.size());
  std::vector<char> ranges;
  const auto stats = coalesceIo<CacheRequest*, char>(
      requests,
      maxDistance,
      std::numeric_limits<int32_t>::max(),
      [&](int32_t index) { return getOffset<kSsd>(*requests[index]); },
      [&](int32_t index) {
        const auto size = requests[index]->size;
        coalescedBytes += size;
        return size;
      },
      [&](int32_t index) {
        if (coalescedBytes > options_.maxCoalesceBytes()) {
          coalescedBytes = 0;
          return kNoCoalesce;
        }
        return requests[index]->coalesces ? 1 : kNoCoalesce;
      },
      [&](CacheRequest* /*request*/, std::vector<char>& ranges) {
        ranges.push_back(0);
      },
      [&](int32_t /*gap*/, std::vector<char> /*ranges*/) { /*no op*/ },
      [&](const std::vector<CacheRequest*>& /*requests*/,
          int32_t /*begin*/,
          int32_t end,
          uint64_t /*offset*/,
          const std::vector<char>& /*ranges*/) { ends.push_back(end); });
  ioStatistics_->readGap().merge(stats.gaps);
  ioStatistics_->incDuplicateRead(stats.duplicateRegions, stats.duplicateBytes);
  return ends;
}

namespace {
// Base class for CoalescedLoads for different storage types.
class DwioCoalescedLoadBase : public cache::CoalescedLoad {
 public:
  DwioCoalescedLoadBase(
      cache::AsyncDataCache& cache,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : CoalescedLoad(makeKeys(requests), makeSizes(requests)),
        cache_(cache),
        ioStatistics_(std::move(ioStatistics)),
        ioStats_(std::move(ioStats)),
        groupId_(groupId) {
    requests_.reserve(requests.size());
    for (const auto& request : requests) {
      size_ += request->size;
      requests_.push_back(std::move(*request));
    }
  }

  const std::vector<CacheRequest>& requests() {
    return requests_;
  }

  int64_t size() const override {
    return size_;
  }

  std::string toString() const override {
    int32_t payload = 0;
    VELOX_CHECK(!requests_.empty());

    int32_t total = requests_.back().key.offset + requests_.back().size -
        requests_[0].key.offset;
    for (const auto& request : requests_) {
      payload += request.size;
    }
    return fmt::format(
        "<CoalescedLoad: {} entries, {} total {} extra>",
        requests_.size(),
        succinctBytes(total),
        succinctBytes(total - payload));
  }

  virtual void prepare(bool /*prefetch*/) {}

  virtual std::function<void()> startAsyncLoad(bool /*ssdSavable*/) {
    return {};
  }

 protected:
  void updateStats(const CoalesceIoStats& stats, bool prefetch, bool ssd) {
    if (ioStatistics_ == nullptr) {
      return;
    }
    ioStatistics_->incRawOverreadBytes(stats.extraBytes);
    if (ssd) {
      ioStatistics_->ssdRead().increment(stats.payloadBytes);
    } else {
      ioStatistics_->read().increment(stats.payloadBytes);
    }
    if (prefetch) {
      ioStatistics_->prefetch().increment(stats.payloadBytes);
    }
  }

  static std::vector<RawFileCacheKey> makeKeys(
      std::vector<CacheRequest*>& requests) {
    std::vector<RawFileCacheKey> keys;
    keys.reserve(requests.size());
    for (auto& request : requests) {
      keys.push_back(request->key);
    }
    return keys;
  }

  std::vector<int32_t> makeSizes(std::vector<CacheRequest*> requests) {
    std::vector<int32_t> sizes;
    sizes.reserve(requests.size());
    for (auto& request : requests) {
      sizes.push_back(request->size);
    }
    return sizes;
  }

  cache::AsyncDataCache& cache_;
  std::vector<CacheRequest> requests_;
  std::shared_ptr<IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;
  const uint64_t groupId_;
  int64_t size_{0};
};

// Represents a CoalescedLoad from ReadFile, e.g. disagg disk.
class DwioCoalescedLoad : public DwioCoalescedLoadBase {
 public:
  DwioCoalescedLoad(
      cache::AsyncDataCache& cache,
      std::shared_ptr<ReadFileInputStream> input,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests,
      int32_t maxCoalesceDistance)
      : DwioCoalescedLoadBase(
            cache,
            std::move(ioStatistics),
            std::move(ioStats),
            groupId,
            std::move(requests)),
        input_(std::move(input)),
        maxCoalesceDistance_(maxCoalesceDistance) {}

  bool isSsdLoad() const override {
    return false;
  }

  void prepare(bool prefetch) override {
    VELOX_CHECK(!prepared_, "Coalesced cache load prepared more than once");
    HintAsyncMakePinsSlot slot(hintAsyncMakePinsGate);
    preparedPins_ = makePins(prefetch, /*asyncHint=*/true);
    prepared_ = true;
  }

  std::function<void()> startAsyncLoad(bool ssdSavable) override {
    if (!input_->hasReadAsync()) {
      return {};
    }
    std::vector<CachePin> pins;
    if (prepared_) {
      pins = std::move(preparedPins_);
    } else {
      HintAsyncMakePinsSlot slot(hintAsyncMakePinsGate);
      pins = makePins(/*prefetch=*/false, /*asyncHint=*/true);
    }
    prepared_ = false;
    if (pins.empty()) {
      return [] {};
    }
    struct PendingRead {
      folly::SemiFuture<uint64_t> future;
      uint64_t expectedBytes;
    };
    std::vector<PendingRead> pending;
    CoalesceIoStats stats;
    try {
      AsyncLoadStageTimer submitTimer(
          hintAsyncSubmitCalls,
          hintAsyncSubmitWallNanos,
          hintAsyncSubmitActive,
          hintAsyncSubmitMaxActive);
      stats = cache::readPins(
          pins,
          maxCoalesceDistance_,
          1000,
          [&](int32_t i) { return pins[i].entry()->offset(); },
          [&](const std::vector<CachePin>& /*pins*/,
              int32_t /*begin*/,
              int32_t /*end*/,
              uint64_t offset,
              const std::vector<folly::Range<char*>>& buffers) {
            uint64_t expectedBytes = 0;
            for (const auto& buffer : buffers) {
              expectedBytes += buffer.size();
            }
            pending.push_back(
                {input_->readAsync(buffers, offset, LogType::FILE),
                 expectedBytes});
            hintAsyncSubmittedRequests.fetch_add(1, std::memory_order_relaxed);
          });
    } catch (...) {
      const auto failure = std::current_exception();
      for (auto& read : pending) {
        try {
          std::move(read.future).get();
        } catch (...) {
        }
      }
      std::rethrow_exception(failure);
    }
    auto pendingReads =
        std::make_shared<std::vector<PendingRead>>(std::move(pending));
    return [this,
            pins = std::move(pins),
            pendingReads = std::move(pendingReads),
            stats = std::move(stats),
            ssdSavable]() mutable {
      std::exception_ptr failure;
      {
        AsyncLoadStageTimer waitTimer(
            hintAsyncWaitCalls,
            hintAsyncWaitWallNanos,
            hintAsyncWaitActive,
            hintAsyncWaitMaxActive);
        for (auto& read : *pendingReads) {
          try {
            const auto bytesRead = std::move(read.future).get();
            VELOX_CHECK_EQ(
                bytesRead,
                read.expectedBytes,
                "Async cache read must return exactly the requested bytes");
          } catch (...) {
            if (!failure) {
              failure = std::current_exception();
            }
          }
        }
      }
      if (failure) {
        std::rethrow_exception(failure);
      }
      updateStats(stats, /*prefetch=*/false, /*ssd=*/false);
      for (const auto& pin : pins) {
        pin.checkedEntry()->setExclusiveToShared(ssdSavable);
      }
    };
  }

  std::vector<CachePin> loadData(bool prefetch) override {
    auto pins = prepared_ ? std::move(preparedPins_)
                          : makePins(prefetch, /*asyncHint=*/false);
    prepared_ = false;
    if (pins.empty()) {
      return pins;
    }
    auto stats = cache::readPins(
        pins,
        maxCoalesceDistance_,
        1000,
        [&](int32_t i) { return pins[i].entry()->offset(); },
        [&](const std::vector<CachePin>& /*pins*/,
            int32_t /*begin*/,
            int32_t /*end*/,
            uint64_t offset,
            const std::vector<folly::Range<char*>>& buffers) {
          input_->read(buffers, offset, LogType::FILE);
        });
    updateStats(stats, prefetch, false);
    return pins;
  }

 private:
  std::vector<CachePin> makePins(bool prefetch, bool asyncHint) {
    std::optional<AsyncLoadStageTimer> makePinsTimer;
    if (asyncHint) {
      makePinsTimer.emplace(
          hintAsyncMakePinsCalls,
          hintAsyncMakePinsWallNanos,
          hintAsyncMakePinsActive,
          hintAsyncMakePinsMaxActive);
    }
    std::vector<CachePin> pins;
    pins.reserve(keys_.size());
    cache_.makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t /*index*/, CachePin pin) {
          if (prefetch) {
            pin.checkedEntry()->setPrefetch(true);
          }
          pins.push_back(std::move(pin));
        });
    if (asyncHint) {
      hintAsyncMakePinsEntries.fetch_add(
          pins.size(), std::memory_order_relaxed);
    }
    if (pins.empty()) {
      return pins;
    }
    return pins;
  }

  std::shared_ptr<ReadFileInputStream> input_;
  const int32_t maxCoalesceDistance_;
  std::vector<CachePin> preparedPins_;
  bool prepared_{false};
};

// Represents a CoalescedLoad from local SSD cache.
class SsdLoad : public DwioCoalescedLoadBase {
 public:
  SsdLoad(
      cache::AsyncDataCache& cache,
      std::shared_ptr<IoStatistics> ioStatistics,
      std::shared_ptr<velox::IoStats> ioStats,
      uint64_t groupId,
      std::vector<CacheRequest*> requests)
      : DwioCoalescedLoadBase(
            cache,
            std::move(ioStatistics),
            std::move(ioStats),
            groupId,
            std::move(requests)) {}

  bool isSsdLoad() const override {
    return true;
  }

  std::vector<CachePin> loadData(bool prefetch) override {
    std::vector<SsdPin> ssdPins;
    std::vector<CachePin> pins;
    cache_.makePins(
        keys_,
        [&](int32_t index) { return sizes_[index]; },
        [&](int32_t index, CachePin pin) {
          if (prefetch) {
            pin.checkedEntry()->setPrefetch(true);
          }
          pins.push_back(std::move(pin));
          ssdPins.push_back(std::move(requests_[index].ssdPin));
        });
    if (pins.empty()) {
      return pins;
    }
    assert(!ssdPins.empty()); // for lint.
    const auto stats = ssdPins[0].file()->load(ssdPins, pins);
    updateStats(stats, prefetch, true);
    return pins;
  }
};

} // namespace

void CachedBufferedInput::readRegion(
    const std::vector<CacheRequest*>& requests,
    bool prefetch) {
  if (requests.empty() || (requests.size() == 1 && !prefetch)) {
    return;
  }

  std::shared_ptr<cache::CoalescedLoad> load;
  if (!requests[0]->ssdPin.empty()) {
    load = std::make_shared<SsdLoad>(
        *cache_, ioStatistics_, ioStats_, groupId_.id(), requests);
  } else {
    load = std::make_shared<DwioCoalescedLoad>(
        *cache_,
        input_,
        ioStatistics_,
        ioStats_,
        groupId_.id(),
        requests,
        options_.maxCoalesceDistance());
  }
  coalescedLoads_.push_back(load);
  streamToCoalescedLoad_.withWLock([&](auto& loads) {
    for (auto& request : requests) {
      loads[request->stream] = load;
    }
  });
}

void CachedBufferedInput::readRegions(
    const std::vector<CacheRequest*>& requests,
    bool prefetch,
    const std::vector<int32_t>& groupEnds) {
  if (requests.empty()) {
    VELOX_CHECK(groupEnds.empty());
    return;
  }
  // Record the starting position so that we only submit the loads created by
  // this call. Without this, non-prefetch loads or stale loads from previous
  // cycles could be incorrectly submitted for async prefetching.
  const int32_t startIndex = static_cast<int32_t>(coalescedLoads_.size());
  int32_t requestIdx{0};
  std::vector<CacheRequest*> requestGroup;
  for (auto groupEndIdx : groupEnds) {
    while (requestIdx < groupEndIdx) {
      requestGroup.push_back(requests[requestIdx++]);
    }
    readRegion(requestGroup, prefetch);
    requestGroup.clear();
  }

  if (prefetch && executor_ && !inlinePrefetchLoad_) {
    // Only submit the loads created by this call to the executor.
    for (auto i = startIndex; i < coalescedLoads_.size(); ++i) {
      auto& load = coalescedLoads_[i];
      if (load->state() == CoalescedLoad::State::kPlanned) {
        executor_->add(
            [pendingLoad = load, ssdSavable = options_.cacheable()]() {
              pendingLoad->loadOrFuture(nullptr, ssdSavable);
            });
      }
    }
    // Remove the loads that were complete. There can be done loads if the same
    // CachedBufferedInput has multiple cycles of enqueues and loads.
    std::vector<int32_t> doneIndices;
    for (int32_t i = 0; i < startIndex; ++i) {
      if (coalescedLoads_[i]->state() != CoalescedLoad::State::kPlanned) {
        doneIndices.push_back(i);
      }
    }
    for (int i = 0, j = 0, k = 0; i < coalescedLoads_.size(); ++i) {
      if (j < doneIndices.size() && doneIndices[j] == i) {
        ++j;
      } else {
        coalescedLoads_[k++] = std::move(coalescedLoads_[i]);
      }
    }
    coalescedLoads_.resize(coalescedLoads_.size() - doneIndices.size());
  }
}

std::shared_ptr<cache::CoalescedLoad> CachedBufferedInput::coalescedLoad(
    const SeekableInputStream* stream) {
  return streamToCoalescedLoad_.withWLock(
      [&](auto& loads) -> std::shared_ptr<cache::CoalescedLoad> {
        auto it = loads.find(stream);
        if (it == loads.end()) {
          return nullptr;
        }
        auto load = std::move(it->second);
        auto* dwioLoad = checkedPointerCast<DwioCoalescedLoadBase>(load.get());
        for (auto& request : dwioLoad->requests()) {
          loads.erase(request.stream);
        }
        return load;
      });
}

void CachedBufferedInput::reset() {
  BufferedInput::reset();
  for (auto& load : coalescedLoads_) {
    load->cancel();
  }
  coalescedLoads_.clear();
  streamToCoalescedLoad_.wlock()->clear();
  requests_.clear();
}

std::unique_ptr<SeekableInputStream> CachedBufferedInput::read(
    uint64_t offset,
    uint64_t length,
    LogType logType) const {
  return readWithCacheRegionPlan(
      offset, length, logType, *cacheRegionPlan_.rlock());
}

std::unique_ptr<SeekableInputStream>
CachedBufferedInput::readWithCacheRegionPlan(
    uint64_t offset,
    uint64_t length,
    LogType /*logType*/,
    std::shared_ptr<CacheRegionPlan> plan) const {
  VELOX_CHECK_LE(offset, fileSize_);
  VELOX_CHECK_LE(length, fileSize_ - offset);
  auto stream = std::make_unique<CacheInputStream>(
      const_cast<CachedBufferedInput*>(this),
      ioStatistics_.get(),
      Region{offset, length},
      input_,
      fileNum_.id(),
      options_.cacheable(),
      nullptr,
      TrackingId(),
      0,
      options_.loadQuantum(),
      std::move(plan));
  if (preloaded()) {
    stream->setPreloadedPin(preloadPin_);
  }
  return stream;
}

bool CachedBufferedInput::prefetch(Region region) {
  const int32_t numPages = memory::AllocationTraits::numPages(region.length);
  if (!shouldPreload(numPages)) {
    return false;
  }
  auto stream = enqueue(region, nullptr);
  load(LogType::FILE);
  // Remove the coalesced load made for the stream. It will not be accessed. The
  // cache entry will be accessed.
  coalescedLoad(stream.get());
  return true;
}

void CachedBufferedInput::prefetchSync(
    const std::vector<Region>& regions,
    bool inlineLoad) {
  auto prepared = preparePrefetch(regions, inlineLoad);
  prepared();
}

std::function<void()> CachedBufferedInput::preparePrefetch(
    const std::vector<Region>& regions,
    bool inlineLoad,
    bool preallocatePins,
    bool asyncPhysicalGroups) {
  return preparePrefetch(
      regions,
      inlineLoad,
      preallocatePins,
      asyncPhysicalGroups,
      /*firstLoadReady=*/{});
}

std::function<void()> CachedBufferedInput::preparePrefetch(
    const std::vector<Region>& regions,
    bool inlineLoad,
    bool preallocatePins,
    bool asyncPhysicalGroups,
    std::function<void()> firstLoadReady) {
  VELOX_CHECK(
      inlineLoad || (!preallocatePins && !asyncPhysicalGroups),
      "Cache-hint pin preallocation and asynchronous physical groups require "
      "inline loading");
  auto plan = makeCacheRegionPlan(regions);
  {
    auto locked = cacheRegionPlan_.wlock();
    if (*locked) {
      const auto& existing = (*locked)->cacheRegions();
      const auto& replacement = plan->cacheRegions();
      VELOX_CHECK_EQ(
          existing.size(),
          replacement.size(),
          "Cached input received incompatible cache region plans");
      for (size_t i = 0; i < existing.size(); ++i) {
        VELOX_CHECK(
            existing[i].offset == replacement[i].offset &&
                existing[i].length == replacement[i].length,
            "Cached input received incompatible cache region plans");
      }
    }
    *locked = plan;
  }
  const auto& canonical = plan->cacheRegions();
  std::vector<std::unique_ptr<SeekableInputStream>> streams;
  streams.reserve(canonical.size());
  for (const auto& region : canonical) {
    if (region.length == 0) {
      continue;
    }
    streams.push_back(enqueue(region, nullptr));
  }
  if (streams.empty()) {
    return [plan = std::move(plan),
            firstLoadReady = std::move(firstLoadReady)]() mutable {
      if (firstLoadReady) {
        firstLoadReady();
      }
      plan->markPrefetchComplete();
    };
  }

  VELOX_CHECK(!inlinePrefetchLoad_);
  inlinePrefetchLoad_ = inlineLoad;
  try {
    load(LogType::FILE);
  } catch (...) {
    inlinePrefetchLoad_ = false;
    throw;
  }
  inlinePrefetchLoad_ = false;
  std::vector<std::shared_ptr<cache::CoalescedLoad>> loads;
  loads.reserve(streams.size());
  folly::F14FastSet<cache::CoalescedLoad*> seen;
  for (const auto& stream : streams) {
    auto pending = coalescedLoad(stream.get());
    if (pending && seen.insert(pending.get()).second) {
      loads.push_back(std::move(pending));
    }
  }
  if (preallocatePins) {
    for (const auto& pending : loads) {
      checkedPointerCast<DwioCoalescedLoadBase>(pending.get())
          ->prepare(/*prefetch=*/false);
    }
  }
  const auto cacheable = options_.cacheable();
  return [plan = std::move(plan),
          loads = std::move(loads),
          cacheable,
          asyncPhysicalGroups,
          firstLoadReady = std::move(firstLoadReady)]() mutable {
    bool firstLoadSignaled = false;
    auto signalFirstLoad = [&]() {
      if (!firstLoadSignaled) {
        firstLoadSignaled = true;
        if (firstLoadReady) {
          firstLoadReady();
        }
      }
    };
    if (asyncPhysicalGroups) {
      std::vector<std::function<void()>> completions;
      completions.reserve(loads.size());
      std::exception_ptr failure;
      try {
        for (const auto& pending : loads) {
          auto completion =
              checkedPointerCast<DwioCoalescedLoadBase>(pending.get())
                  ->startAsyncLoad(cacheable);
          if (completion) {
            completions.push_back(std::move(completion));
          } else {
            // Submit all natively asynchronous remote reads before running
            // SSD or non-async fallback loads synchronously.
            completions.push_back([pending, cacheable]() {
              folly::SemiFuture<bool> waitFuture(false);
              if (!pending->loadOrFuture(&waitFuture, cacheable)) {
                std::move(waitFuture).wait();
              }
            });
          }
        }
      } catch (...) {
        failure = std::current_exception();
      }
      bool completedOne = false;
      for (auto& completion : completions) {
        try {
          completion();
          completedOne = true;
        } catch (...) {
          if (!failure) {
            failure = std::current_exception();
          }
        }
        if (completedOne && !firstLoadSignaled) {
          try {
            signalFirstLoad();
          } catch (...) {
            if (!failure) {
              failure = std::current_exception();
            }
          }
        }
      }
      if (completions.empty() && !failure) {
        try {
          signalFirstLoad();
        } catch (...) {
          failure = std::current_exception();
        }
      }
      if (failure) {
        std::rethrow_exception(failure);
      }
      plan->markPrefetchComplete();
      return;
    }
    for (const auto& pending : loads) {
      folly::SemiFuture<bool> waitFuture(false);
      if (!pending->loadOrFuture(&waitFuture, cacheable)) {
        std::move(waitFuture).wait();
      }
      signalFirstLoad();
    }
    signalFirstLoad();
    plan->markPrefetchComplete();
  };
}

std::shared_ptr<CacheRegionPlan> CachedBufferedInput::makeCacheRegionPlan(
    const std::vector<Region>& regions) const {
  return CacheRegionPlan::create(regions, fileSize_, options_.loadQuantum());
}

std::vector<Region> CachedBufferedInput::canonicalizeRegions(
    const std::vector<Region>& regions) const {
  const uint64_t quantum = options_.loadQuantum();
  VELOX_CHECK_GT(quantum, 0);
  std::vector<uint64_t> offsets;
  for (const auto& region : regions) {
    VELOX_CHECK_LE(region.offset, fileSize_);
    VELOX_CHECK_LE(region.length, fileSize_ - region.offset);
    if (region.length == 0) {
      continue;
    }
    auto offset = (region.offset / quantum) * quantum;
    const auto end = region.offset + region.length;
    while (offset < end) {
      offsets.push_back(offset);
      offset += quantum;
    }
  }
  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());

  std::vector<Region> canonical;
  canonical.reserve(offsets.size());
  for (const auto offset : offsets) {
    canonical.push_back(
        {offset, std::min<uint64_t>(quantum, fileSize_ - offset)});
  }
  return canonical;
}

void CachedBufferedInput::cacheRegion(
    uint64_t offset,
    uint64_t length,
    std::string_view data) {
  VELOX_CHECK_EQ(data.size(), length);
  auto iobuf = folly::IOBuf::wrapBufferAsValue(data.data(), data.size());
  cacheRegion(offset, length, iobuf, 0);
}

void CachedBufferedInput::cacheRegion(
    uint64_t offset,
    uint64_t length,
    const folly::IOBuf& buffer,
    uint64_t bufferOffset) {
  auto pin =
      cache_->findOrCreate(RawFileCacheKey{fileNum_.id(), offset}, length);
  // Empty pin means the cache is at capacity and cannot accept new entries.
  // Non-exclusive means another thread already cached this region; skip the
  // duplicate write.
  if (pin.empty() || !pin.checkedEntry()->isExclusive()) {
    return;
  }

  folly::io::Cursor cursor(&buffer);
  cursor.skip(bufferOffset);
  VELOX_CHECK_GE(
      cursor.totalLength(),
      length,
      "IOBuf has {} bytes after offset {}, need {}",
      cursor.totalLength(),
      bufferOffset,
      length);

  auto* entry = pin.checkedEntry();
  if (entry->hasContiguousData()) {
    cursor.pull(entry->contiguousData(), length);
  } else {
    auto& allocation = entry->nonContiguousData();
    uint64_t copyBytes = 0;
    for (int i = 0; i < allocation.numRuns() && copyBytes < length; ++i) {
      const auto run = allocation.runAt(i);
      const uint64_t copySize =
          std::min<uint64_t>(run.numBytes(), length - copyBytes);
      cursor.pull(run.data(), copySize);
      copyBytes += copySize;
    }
    VELOX_CHECK_EQ(copyBytes, length);
  }

  // Clear the first-use flag since this entry is being populated externally
  // (not loaded on-demand). The first findCachedRegion access should count
  // as a cache hit.
  entry->getAndClearFirstUseFlag();
  entry->setExclusiveToShared();
}

std::optional<CachedRegion> CachedBufferedInput::findCachedRegion(
    uint64_t offset) const {
  const cache::RawFileCacheKey key{fileNum_.id(), offset};
  for (;;) {
    folly::SemiFuture<bool> waitFuture(false);
    auto result = cache_->find(key, &waitFuture);
    if (!result.has_value()) {
      return std::nullopt;
    }
    if (!result->empty()) {
      auto* entry = result->checkedEntry();
      if (!entry->getAndClearFirstUseFlag()) {
        ioStatistics_->ramHit().increment(entry->size());
      }
      return CachedRegion{std::move(*result)};
    }
    // Entry is exclusive — wait for it to become shared, then retry.
    uint64_t waitUs{0};
    {
      MicrosecondWallTimer timer(&waitUs);
      std::move(waitFuture)
          .via(&folly::QueuedImmediateExecutor::instance())
          .wait();
    }
    ioStatistics_->queryThreadIoLatencyUs().increment(waitUs);
    ioStatistics_->cacheWaitLatencyUs().increment(waitUs);
  }
}

} // namespace facebook::velox::dwio::common
