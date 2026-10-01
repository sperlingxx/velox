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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/Casts.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/CachedBufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <curl/curl.h>
#include <folly/futures/Future.h>

#ifdef VELOX_ENABLE_S3
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" bool glutenCrtS3RangeReaderAvailable();
extern "C" uint64_t glutenCrtS3ObjectSize(const char* uri);
extern "C" uint64_t glutenCrtS3ReadRanges(
    const char* uri,
    uint8_t* destination,
    const uint64_t* offsets,
    const uint64_t* lengths,
    const uint64_t* destinationOffsets,
    size_t count);

namespace {

thread_local bool nativeS3HighRequestPressure{false};

using facebook::velox::cudf_velox::connector::hive::
    BufferedInputDeviceCopyHooks;
using facebook::velox::cudf_velox::connector::hive::
    CachePageHostRegistrationHooks;
using BufferedInputStream = facebook::velox::dwio::common::SeekableInputStream;

struct DirectCachePageH2dLifetime {
  std::shared_ptr<BufferedInputStream> inputStream;
  std::vector<facebook::velox::cache::CachePin> cachePins;
};

std::atomic<uint64_t> directCachePageH2dCopies{0};
std::atomic<uint64_t> directCachePageH2dBytes{0};
std::atomic<uint64_t> directCachePageH2dPinnedCopies{0};
std::atomic<uint64_t> directCachePageH2dPinnedBytes{0};
std::atomic<uint64_t> cachePageRegistrationAttempts{0};
std::atomic<uint64_t> cachePageRegistrationSuccesses{0};
std::atomic<uint64_t> cachePageRegistrationFailures{0};
std::atomic<uint64_t> cachePageRegistrationBudgetRejectedBytes{0};
std::atomic<uint64_t> cachePageRegisteredRuns{0};
std::atomic<uint64_t> cachePageRegisteredBytes{0};
std::atomic<uint64_t> cachePageCurrentRegisteredBytes{0};
std::atomic<uint64_t> cachePagePeakRegisteredBytes{0};
std::atomic<uint64_t> cachePageUnregisteredRuns{0};
std::atomic<uint64_t> cachePageUnregisteredBytes{0};
std::atomic<uint64_t> cachePagePrewarmAttempts{0};
std::atomic<uint64_t> cachePagePrewarmSuccesses{0};
std::atomic<uint64_t> cachePagePrewarmFailures{0};
std::atomic<uint64_t> cachePagePrewarmRuns{0};
std::atomic<uint64_t> cachePagePrewarmBytes{0};
std::atomic<uint64_t> cachePagePrewarmCoveredRuns{0};
std::atomic<uint64_t> cachePagePrewarmCoveredBytes{0};
std::atomic<uint64_t> cachePageRegistrationWallNanos{0};
std::atomic<uint64_t> cachePagePrewarmWallNanos{0};

class AtomicWallTimer {
 public:
  explicit AtomicWallTimer(std::atomic<uint64_t>& counter)
      : counter_(counter), start_(std::chrono::steady_clock::now()) {}

  ~AtomicWallTimer() {
    counter_.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start_)
                .count()),
        std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>& counter_;
  const std::chrono::steady_clock::time_point start_;
};

class BoundedCachePageRegistrationState
    : public std::enable_shared_from_this<BoundedCachePageRegistrationState> {
 public:
  struct Run {
    void* address;
    size_t bytes;
    bool prewarmed;
  };

  struct PrewarmLifetime {
    std::vector<Run> runs;
    std::shared_ptr<void> mappedPageProtection;
  };

  BoundedCachePageRegistrationState(
      uint64_t maxBytes,
      CachePageHostRegistrationHooks hooks)
      : maxBytes_(maxBytes), hooks_(std::move(hooks)) {}

  std::shared_ptr<void> registerAllocation(
      const facebook::velox::memory::Allocation& allocation) {
    AtomicWallTimer timer(cachePageRegistrationWallNanos);
    cachePageRegistrationAttempts.fetch_add(1, std::memory_order_relaxed);
    if (allocation.empty()) {
      return nullptr;
    }

    std::vector<Run> candidates;
    candidates.reserve(allocation.numRuns());
    for (int32_t i = 0; i < allocation.numRuns(); ++i) {
      const auto run = allocation.runAt(i);
      candidates.push_back(Run{run.data<void>(), run.numBytes(), false});
    }
    return registerRuns(candidates);
  }

  std::shared_ptr<void> registerRange(void* address, uint64_t bytes) {
    AtomicWallTimer timer(cachePageRegistrationWallNanos);
    cachePageRegistrationAttempts.fetch_add(1, std::memory_order_relaxed);
    if (address == nullptr || bytes == 0) {
      return nullptr;
    }
    return registerRuns({Run{address, bytes, false}});
  }

  std::shared_ptr<void> registerPersistentRange(
      facebook::velox::memory::MmapAllocator& allocator,
      void* address,
      uint64_t bytes) {
    AtomicWallTimer timer(cachePageRegistrationWallNanos);
    cachePageRegistrationAttempts.fetch_add(1, std::memory_order_relaxed);
    if (address == nullptr || bytes == 0) {
      return nullptr;
    }
    const auto pages =
        facebook::velox::memory::AllocationTraits::numPages(bytes);
    // Contiguous allocations above the largest size class are individually
    // mmapped and disappear on free. Only retain registration for reusable
    // size-class backing.
    if (pages > allocator.largestSizeClass()) {
      return nullptr;
    }
    const Run run{address, bytes, true};
    std::lock_guard<std::mutex> lock(mutex_);
    if (isCoveredLocked(run)) {
      cachePagePrewarmCoveredRuns.fetch_add(1, std::memory_order_relaxed);
      cachePagePrewarmCoveredBytes.fetch_add(bytes, std::memory_order_relaxed);
      return std::shared_ptr<void>(shared_from_this(), address);
    }
    if (overlapsLocked(run)) {
      cachePageRegistrationFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    if (!reserve(bytes)) {
      cachePageRegistrationBudgetRejectedBytes.fetch_add(
          bytes, std::memory_order_relaxed);
      return nullptr;
    }
    bool registered = false;
    std::shared_ptr<void> protection;
    try {
      registered = hooks_.registerRun(address, bytes);
      if (registered) {
        protection = allocator.protectMappedRange(address, pages);
      }
    } catch (...) {
      registered = false;
    }
    if (!registered || !protection) {
      if (registered) {
        try {
          hooks_.unregisterRun(address);
        } catch (...) {
        }
      }
      releaseReservation(bytes);
      cachePageRegistrationFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    registry_.emplace(reinterpret_cast<uintptr_t>(address), run);
    persistentRuns_.push_back(run);
    persistentProtections_.push_back(std::move(protection));
    cachePageRegistrationSuccesses.fetch_add(1, std::memory_order_relaxed);
    cachePageRegisteredRuns.fetch_add(1, std::memory_order_relaxed);
    cachePageRegisteredBytes.fetch_add(bytes, std::memory_order_relaxed);
    return std::shared_ptr<void>(shared_from_this(), address);
  }

  void releasePersistent() noexcept {
    std::vector<Run> runs;
    std::vector<std::shared_ptr<void>> protections;
    uint64_t bytes = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      runs.swap(persistentRuns_);
      protections.swap(persistentProtections_);
      for (const auto& run : runs) {
        bytes += run.bytes;
      }
      unregisterLocked(runs);
    }
    // CUDA registration must be removed before mapped pages may be advised.
    protections.clear();
    if (bytes != 0) {
      cachePageUnregisteredRuns.fetch_add(
          runs.size(), std::memory_order_relaxed);
      cachePageUnregisteredBytes.fetch_add(bytes, std::memory_order_relaxed);
      releaseReservation(bytes);
    }
  }

  std::shared_ptr<void> prewarmLargestSizeClass(
      facebook::velox::memory::MmapAllocator& allocator,
      uint64_t requestedBytes) {
    AtomicWallTimer timer(cachePagePrewarmWallNanos);
    cachePagePrewarmAttempts.fetch_add(1, std::memory_order_relaxed);
    const auto classPages = allocator.largestSizeClass();
    const auto requestedPages =
        facebook::velox::memory::AllocationTraits::numPages(requestedBytes);
    const auto pages = (requestedPages / classPages) * classPages;
    const uint64_t bytes =
        facebook::velox::memory::AllocationTraits::pageBytes(pages);
    if (pages == 0 || bytes > maxBytes_ || !reserve(bytes)) {
      cachePagePrewarmFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    facebook::velox::memory::Allocation allocation;
    if (!allocator.allocateNonContiguous(
            pages, allocation, nullptr, classPages)) {
      releaseReservation(bytes);
      cachePagePrewarmFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    std::vector<Run> runs;
    runs.reserve(allocation.numRuns());
    for (int32_t i = 0; i < allocation.numRuns(); ++i) {
      const auto run = allocation.runAt(i);
      runs.push_back(Run{run.data<void>(), run.numBytes(), true});
    }

    bool registered = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& run : runs) {
        if (overlapsLocked(run)) {
          registered = false;
          break;
        }
        try {
          if (!hooks_.registerRun(run.address, run.bytes)) {
            registered = false;
            break;
          }
        } catch (...) {
          registered = false;
          break;
        }
        registry_.emplace(reinterpret_cast<uintptr_t>(run.address), run);
      }
      if (!registered) {
        unregisterLocked(runs);
      }
    }
    if (!registered) {
      allocator.freeNonContiguous(allocation);
      releaseReservation(bytes);
      cachePagePrewarmFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }

    std::shared_ptr<void> mappedPageProtection;
    try {
      mappedPageProtection = allocator.protectMappedPages(allocation);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        unregisterLocked(runs);
      }
      allocator.freeNonContiguous(allocation);
      releaseReservation(bytes);
      cachePagePrewarmFailures.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
    allocator.freeNonContiguous(allocation);

    cachePageRegistrationSuccesses.fetch_add(1, std::memory_order_relaxed);
    cachePageRegisteredRuns.fetch_add(runs.size(), std::memory_order_relaxed);
    cachePageRegisteredBytes.fetch_add(bytes, std::memory_order_relaxed);
    cachePagePrewarmSuccesses.fetch_add(1, std::memory_order_relaxed);
    cachePagePrewarmRuns.fetch_add(runs.size(), std::memory_order_relaxed);
    cachePagePrewarmBytes.fetch_add(bytes, std::memory_order_relaxed);
    auto self = shared_from_this();
    return std::shared_ptr<void>(
        new PrewarmLifetime{std::move(runs), std::move(mappedPageProtection)},
        [self, bytes](void* opaque) noexcept {
          std::unique_ptr<PrewarmLifetime> owned(
              static_cast<PrewarmLifetime*>(opaque));
          // Keep allocator backing protected until CUDA registration is gone.
          self->release(owned->runs, bytes);
          owned->mappedPageProtection.reset();
        });
  }

 private:
  std::shared_ptr<void> registerRuns(const std::vector<Run>& candidates) {
    std::vector<Run> uncovered;
    uint64_t uncoveredBytes = 0;
    uint64_t coveredBytes = 0;
    uint64_t coveredRuns = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& candidate : candidates) {
        if (isCoveredLocked(candidate)) {
          ++coveredRuns;
          coveredBytes += candidate.bytes;
        } else if (overlapsLocked(candidate)) {
          cachePageRegistrationFailures.fetch_add(1, std::memory_order_relaxed);
          return nullptr;
        } else {
          uncoveredBytes += candidate.bytes;
          uncovered.push_back(candidate);
        }
      }
    }
    cachePagePrewarmCoveredRuns.fetch_add(
        coveredRuns, std::memory_order_relaxed);
    cachePagePrewarmCoveredBytes.fetch_add(
        coveredBytes, std::memory_order_relaxed);

    if (!reserve(uncoveredBytes)) {
      cachePageRegistrationBudgetRejectedBytes.fetch_add(
          uncoveredBytes, std::memory_order_relaxed);
      return nullptr;
    }

    std::vector<Run> registered;
    registered.reserve(uncovered.size());
    bool success = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& candidate : uncovered) {
        if (overlapsLocked(candidate)) {
          success = false;
          break;
        }
        try {
          if (!hooks_.registerRun(candidate.address, candidate.bytes)) {
            success = false;
            break;
          }
        } catch (...) {
          success = false;
          break;
        }
        registry_.emplace(
            reinterpret_cast<uintptr_t>(candidate.address), candidate);
        registered.push_back(candidate);
      }
      if (!success) {
        unregisterLocked(registered);
      }
    }
    if (!success) {
      cachePageRegistrationFailures.fetch_add(1, std::memory_order_relaxed);
      releaseReservation(uncoveredBytes);
      return nullptr;
    }

    cachePageRegistrationSuccesses.fetch_add(1, std::memory_order_relaxed);
    cachePageRegisteredRuns.fetch_add(
        registered.size(), std::memory_order_relaxed);
    cachePageRegisteredBytes.fetch_add(
        uncoveredBytes, std::memory_order_relaxed);
    auto self = shared_from_this();
    return std::shared_ptr<void>(
        new std::vector<Run>(std::move(registered)),
        [self, uncoveredBytes](void* opaque) {
          std::unique_ptr<std::vector<Run>> runs(
              static_cast<std::vector<Run>*>(opaque));
          self->release(*runs, uncoveredBytes);
        });
  }

  bool isCoveredLocked(const Run& run) const {
    const auto begin = reinterpret_cast<uintptr_t>(run.address);
    const auto end = begin + run.bytes;
    auto it = registry_.upper_bound(begin);
    if (it == registry_.begin()) {
      return false;
    }
    --it;
    if (!it->second.prewarmed || it->first > begin ||
        it->first + it->second.bytes <= begin) {
      return false;
    }
    auto coveredEnd = it->first + it->second.bytes;
    while (coveredEnd < end) {
      ++it;
      if (it == registry_.end() || !it->second.prewarmed ||
          it->first != coveredEnd) {
        return false;
      }
      coveredEnd = it->first + it->second.bytes;
    }
    return true;
  }

  bool overlapsLocked(const Run& run) const {
    const auto begin = reinterpret_cast<uintptr_t>(run.address);
    const auto end = begin + run.bytes;
    auto it = registry_.lower_bound(begin);
    if (it != registry_.end() && it->first < end) {
      return true;
    }
    if (it != registry_.begin()) {
      --it;
      return it->first + it->second.bytes > begin;
    }
    return false;
  }

  bool reserve(uint64_t bytes) {
    if (bytes == 0) {
      return true;
    }
    auto current =
        cachePageCurrentRegisteredBytes.load(std::memory_order_relaxed);
    while (bytes <= maxBytes_ && current <= maxBytes_ - bytes) {
      if (cachePageCurrentRegisteredBytes.compare_exchange_weak(
              current,
              current + bytes,
              std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        auto peak =
            cachePagePeakRegisteredBytes.load(std::memory_order_relaxed);
        while (peak < current + bytes &&
               !cachePagePeakRegisteredBytes.compare_exchange_weak(
                   peak,
                   current + bytes,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
        return true;
      }
    }
    return false;
  }

  void releaseReservation(uint64_t bytes) {
    if (bytes == 0) {
      return;
    }
    const auto previous = cachePageCurrentRegisteredBytes.fetch_sub(
        bytes, std::memory_order_acq_rel);
    VELOX_CHECK_GE(previous, bytes);
  }

  void unregisterLocked(const std::vector<Run>& runs) noexcept {
    for (auto iter = runs.rbegin(); iter != runs.rend(); ++iter) {
      const auto key = reinterpret_cast<uintptr_t>(iter->address);
      const auto found = registry_.find(key);
      if (found != registry_.end()) {
        // A failed prewarm may pass the complete requested run vector after a
        // prefix was registered. Only unregister that prefix.
        try {
          hooks_.unregisterRun(iter->address);
        } catch (...) {
          // A lifetime deleter must not throw. Registry removal still makes a
          // duplicate unregister impossible within this manager.
        }
        registry_.erase(found);
      }
    }
  }

  void release(const std::vector<Run>& runs, uint64_t bytes) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      unregisterLocked(runs);
    }
    cachePageUnregisteredRuns.fetch_add(runs.size(), std::memory_order_relaxed);
    cachePageUnregisteredBytes.fetch_add(bytes, std::memory_order_relaxed);
    releaseReservation(bytes);
  }

  const uint64_t maxBytes_;
  CachePageHostRegistrationHooks hooks_;
  std::mutex mutex_;
  std::map<uintptr_t, Run> registry_;
  std::vector<Run> persistentRuns_;
  std::vector<std::shared_ptr<void>> persistentProtections_;
};

BufferedInputDeviceCopyHooks defaultBufferedInputDeviceCopyHooks() {
  return {
      .copy =
          [](uint8_t* destination,
             const void* source,
             size_t bytes,
             rmm::cuda_stream_view stream) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                destination,
                source,
                bytes,
                cudaMemcpyHostToDevice,
                stream.value()));
          },
      .retainUntilComplete =
          [](std::shared_ptr<void> lifetime, rmm::cuda_stream_view stream) {
            using Completion = std::shared_ptr<void>;
            auto completion = std::make_unique<Completion>(std::move(lifetime));
            const auto status = cudaLaunchHostFunc(
                stream.value(),
                [](void* opaque) { delete static_cast<Completion*>(opaque); },
                completion.get());
            if (status != cudaSuccess) {
              // Some copies might already be queued. Keep their source pages
              // pinned until the stream is quiescent before propagating the
              // callback submission failure. The callback itself intentionally
              // invokes no CUDA API.
              CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
              CUDF_CUDA_TRY(status);
            }
            completion.release();
          }};
}

#ifdef VELOX_ENABLE_S3
kvikio::RemoteHandle makeKvikioS3Handle(
    const std::string& filePath,
    const std::string& accessKeyId,
    const std::string& secretAccessKey,
    const std::string& sessionToken,
    std::optional<std::string> region,
    std::optional<std::string> endpoint,
    std::optional<std::size_t> fileSize) {
  auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
  auto s3Endpoint = std::make_unique<kvikio::S3Endpoint>(
      std::move(bucketAndObject),
      std::move(region),
      accessKeyId,
      secretAccessKey,
      std::move(endpoint),
      sessionToken.empty() ? std::nullopt
                           : std::optional<std::string>{sessionToken});
  if (fileSize.has_value()) {
    return kvikio::RemoteHandle(std::move(s3Endpoint), fileSize.value());
  }
  return kvikio::RemoteHandle(std::move(s3Endpoint));
}
#endif

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

std::size_t envBytesOrZero(const char* name) {
  const auto* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  VELOX_CHECK(
      errno == 0 && end != value && *end == '\0',
      "Invalid byte count {}='{}'",
      name,
      value);
  return static_cast<std::size_t>(parsed);
}

std::size_t envBytesOrDefault(const char* name, std::size_t defaultValue) {
  const auto* value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? defaultValue
                                              : envBytesOrZero(name);
}

// Single-reader staging for the hybrid buffered-input path. Both allocations
// precede the first submission: allocating pinned memory between copies can
// itself synchronize CUDA. Each event protects just its slot's last H2D copy.
class BufferedH2dStaging {
 public:
  BufferedH2dStaging(
      size_t capacity,
      rmm::cuda_stream_view stream,
      facebook::velox::cudf_velox::connector::hive::PackedH2dStats* stats =
          nullptr)
      : stream_(stream), stats_(stats) {
    for (auto& slot : slots_) {
      slot.stream = stream.value();
      slot.buffer = std::make_unique<
          facebook::velox::cudf_velox::connector::hive::PinnedHostBuffer>(
          capacity);
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(
          &slot.done, cudaEventDisableTiming | cudaEventBlockingSync));
    }
    if (stats_) {
      stats_->peakStagingBytes = slots_[0].buffer->allocatedBytes() +
          slots_[1].buffer->allocatedBytes();
    }
  }

  uint8_t* acquire() {
    auto& slot = slots_[next_];
    wait(slot);
    return slot.buffer->data();
  }

  void submit(uint8_t* destination, size_t size) {
    auto& slot = slots_[next_];
    // If submission or event recording throws, the slot destructor drains
    // the stream before releasing a possibly still referenced host buffer.
    slot.pending = true;
    slot.recorded = false;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        destination,
        slot.buffer->data(),
        size,
        cudaMemcpyHostToDevice,
        stream_.value()));
    if (stats_) {
      ++stats_->copies;
      stats_->bytes += size;
      if (!slot.buffer->isPinned()) {
        stats_->pageableBytes += size;
      }
    }
    CUDF_CUDA_TRY(cudaEventRecord(slot.done, stream_.value()));
    slot.recorded = true;
    next_ = (next_ + 1) % slots_.size();
  }

  void finish() {
    for (auto& slot : slots_) {
      wait(slot);
    }
  }

 private:
  struct Slot {
    ~Slot() {
      if (pending) {
        if (recorded) {
          cudaEventSynchronize(done);
        } else {
          cudaStreamSynchronize(stream);
        }
      }
      if (done != nullptr) {
        cudaEventDestroy(done);
      }
    }

    void wait() {
      if (pending) {
        CUDF_CUDA_TRY(cudaEventSynchronize(done));
        pending = false;
      }
    }

    std::unique_ptr<
        facebook::velox::cudf_velox::connector::hive::PinnedHostBuffer>
        buffer;
    cudaEvent_t done{nullptr};
    cudaStream_t stream{nullptr};
    bool pending{false};
    bool recorded{false};
  };

  void wait(Slot& slot) {
    if (!stats_ || !slot.pending) {
      slot.wait();
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    slot.wait();
    stats_->stagingWaitNanos +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
  }

  const rmm::cuda_stream_view stream_;
  facebook::velox::cudf_velox::connector::hive::PackedH2dStats* const stats_;
  std::array<Slot, 2> slots_;
  size_t next_{0};
};

bool envFlagEnabled(const char* name) {
  const auto* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool envFlagValueEnabled(const char* value) {
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0;
}

bool envFlagOrDefault(const char* name, bool defaultValue) {
  const auto* value = std::getenv(name);
  return value == nullptr || value[0] == '\0' ? defaultValue
                                              : envFlagValueEnabled(value);
}

bool adaptiveS3PrefetchEnabled() {
  return envFlagValueEnabled(std::getenv("GLUTEN_CUDF_S3_ADAPTIVE_PREFETCH"));
}

void updateAtomicMax(std::atomic<uint64_t>& maximum, uint64_t value) {
  auto current = maximum.load(std::memory_order_relaxed);
  while (value > current &&
         !maximum.compare_exchange_weak(
             current, value, std::memory_order_relaxed)) {
  }
}

class NativeS3H2dGate {
 public:
  static NativeS3H2dGate& instance() {
    static NativeS3H2dGate gate;
    return gate;
  }

  void acquire() {
    if (capacity_ == 0) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return active_ < capacity_; });
    ++active_;
  }

  void release() {
    if (capacity_ == 0) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_CHECK_GT(active_, 0);
      --active_;
    }
    condition_.notify_one();
  }

  bool enabled() const {
    return capacity_ != 0;
  }

 private:
  NativeS3H2dGate() : capacity_(envBytesOrZero("GLUTEN_CPP_S3_H2D_SLOTS")) {}

  const size_t capacity_;
  size_t active_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
};

class NativeS3H2dPermit {
 public:
  explicit NativeS3H2dPermit(bool enabled)
      : gate_(enabled ? &NativeS3H2dGate::instance() : nullptr) {
    if (gate_ != nullptr && gate_->enabled()) {
      gate_->acquire();
      acquired_ = true;
    }
  }

  ~NativeS3H2dPermit() {
    release();
  }

  NativeS3H2dPermit(const NativeS3H2dPermit&) = delete;
  NativeS3H2dPermit& operator=(const NativeS3H2dPermit&) = delete;

  void release() {
    if (acquired_) {
      gate_->release();
      acquired_ = false;
    }
  }

 private:
  NativeS3H2dGate* gate_;
  bool acquired_{false};
};

#ifdef VELOX_ENABLE_S3
class NativeS3MultiScheduler {
 public:
  struct Auth {
    std::string url;
    std::string awsSigV4;
    std::string userPassword;
    std::string sessionToken;
  };

  static NativeS3MultiScheduler& instance() {
    static NativeS3MultiScheduler scheduler;
    return scheduler;
  }

  std::future<size_t>
  submit(const Auth& auth, size_t offset, size_t size, uint8_t* destination) {
    auto request = std::make_shared<Request>();
    request->auth = auth;
    request->offset = offset;
    request->size = size;
    request->destination = destination;
    auto future = request->promise.get_future();

    auto& shard = *shards_
                      [nextShard_.fetch_add(1, std::memory_order_relaxed) %
                       shards_.size()];
    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      if (size <= metadataPriorityBytes_) {
        shard.priorityQueue.push_back(std::move(request));
      } else {
        shard.queue.push_back(std::move(request));
      }
    }
    shard.condition.notify_one();
    curl_multi_wakeup(shard.multi);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    const auto inflight =
        inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
    while (inflight > peak &&
           !peakInflightRequests_.compare_exchange_weak(
               peak, inflight, std::memory_order_relaxed)) {
    }
    return future;
  }

  size_t rangeBytes() const {
    return rangeBytes_;
  }

  NativeS3MultiScheduler(const NativeS3MultiScheduler&) = delete;
  NativeS3MultiScheduler& operator=(const NativeS3MultiScheduler&) = delete;

 private:
  struct Request {
    Auth auth;
    size_t offset{0};
    size_t size{0};
    uint8_t* destination{nullptr};
    size_t written{0};
    CURL* easy{nullptr};
    curl_slist* headers{nullptr};
    std::string range;
    size_t retryCount{0};
    std::promise<size_t> promise;
  };

  struct Shard {
    CURLM* multi{nullptr};
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<Request>> priorityQueue;
    std::deque<std::shared_ptr<Request>> queue;
    std::multimap<
        std::chrono::steady_clock::time_point,
        std::shared_ptr<Request>>
        retryQueue;
    std::unordered_map<CURL*, std::shared_ptr<Request>> active;
    bool stopping{false};
    std::thread worker;
  };

  static size_t writeCallback(
      char* data,
      size_t elementSize,
      size_t elementCount,
      void* opaque) {
    auto* request = static_cast<Request*>(opaque);
    const auto bytes = elementSize * elementCount;
    if (request->written + bytes > request->size) {
      return CURL_WRITEFUNC_ERROR;
    }
    std::memcpy(request->destination + request->written, data, bytes);
    request->written += bytes;
    return bytes;
  }

  static size_t envOrDefault(const char* name, size_t defaultValue) {
    const auto parsed = envBytesOrZero(name);
    return parsed == 0 ? defaultValue : parsed;
  }

  NativeS3MultiScheduler()
      : concurrencyPerShard_(
            envOrDefault("GLUTEN_CPP_S3_CONCURRENCY_PER_SHARD", 16)),
        rangeBytes_(envOrDefault("GLUTEN_CPP_S3_RANGE_BYTES", 4UL << 20)),
        metadataPriorityBytes_(
            envOrDefault("GLUTEN_CPP_S3_METADATA_PRIORITY_BYTES", 64UL << 10)),
        maxRetries_(envOrDefault("GLUTEN_CPP_S3_MAX_RETRIES", 5)),
        retryBaseDelayMs_(
            envOrDefault("GLUTEN_CPP_S3_RETRY_BASE_DELAY_MS", 50)),
        retryMaxDelayMs_(
            envOrDefault("GLUTEN_CPP_S3_RETRY_MAX_DELAY_MS", 1'000)),
        diagnostics_(envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0),
        started_(std::chrono::steady_clock::now()) {
    const auto shardCount = envOrDefault("GLUTEN_CPP_S3_SHARDS", 8);
    VELOX_CHECK_GT(shardCount, 0);
    VELOX_CHECK_GT(concurrencyPerShard_, 0);
    VELOX_CHECK_GT(rangeBytes_, 0);
    curl_global_init(CURL_GLOBAL_ALL);
    shards_.reserve(shardCount);
    for (size_t index = 0; index < shardCount; ++index) {
      auto shard = std::make_unique<Shard>();
      shard->multi = curl_multi_init();
      VELOX_CHECK_NOT_NULL(shard->multi);
      curl_multi_setopt(
          shard->multi,
          CURLMOPT_MAX_TOTAL_CONNECTIONS,
          static_cast<long>(concurrencyPerShard_));
      curl_multi_setopt(
          shard->multi,
          CURLMOPT_MAX_HOST_CONNECTIONS,
          static_cast<long>(concurrencyPerShard_));
      shards_.push_back(std::move(shard));
    }
    for (auto& shard : shards_) {
      shard->worker =
          std::thread([this, rawShard = shard.get()] { run(*rawShard); });
    }
  }

  ~NativeS3MultiScheduler() {
    for (auto& shard : shards_) {
      {
        std::lock_guard<std::mutex> lock(shard->mutex);
        shard->stopping = true;
      }
      shard->condition.notify_one();
      curl_multi_wakeup(shard->multi);
    }
    for (auto& shard : shards_) {
      if (shard->worker.joinable()) {
        shard->worker.join();
      }
      curl_multi_cleanup(shard->multi);
    }
  }

  void startRequest(Shard& shard, std::shared_ptr<Request> request) {
    request->easy = curl_easy_init();
    VELOX_CHECK_NOT_NULL(request->easy);
    request->written = 0;
    request->range = std::to_string(request->offset) + "-" +
        std::to_string(request->offset + request->size - 1);
    if (!request->auth.sessionToken.empty()) {
      const auto header = "x-amz-security-token: " + request->auth.sessionToken;
      request->headers = curl_slist_append(request->headers, header.c_str());
      VELOX_CHECK_NOT_NULL(request->headers);
    }
    curl_easy_setopt(request->easy, CURLOPT_URL, request->auth.url.c_str());
    curl_easy_setopt(
        request->easy, CURLOPT_AWS_SIGV4, request->auth.awsSigV4.c_str());
    curl_easy_setopt(
        request->easy, CURLOPT_USERPWD, request->auth.userPassword.c_str());
    if (request->headers != nullptr) {
      curl_easy_setopt(request->easy, CURLOPT_HTTPHEADER, request->headers);
    }
    curl_easy_setopt(request->easy, CURLOPT_RANGE, request->range.c_str());
    curl_easy_setopt(request->easy, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(request->easy, CURLOPT_WRITEDATA, request.get());
    curl_easy_setopt(request->easy, CURLOPT_PRIVATE, request.get());
    curl_easy_setopt(request->easy, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(request->easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(request->easy, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(
        request->easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(request->easy, CURLOPT_BUFFERSIZE, 512L * 1024L);
    const auto code = curl_multi_add_handle(shard.multi, request->easy);
    VELOX_CHECK(
        code == CURLM_OK,
        "curl_multi_add_handle failed: {}",
        curl_multi_strerror(code));
    shard.active.emplace(request->easy, std::move(request));
  }

  void fillActive(Shard& shard) {
    while (shard.active.size() < concurrencyPerShard_) {
      std::shared_ptr<Request> request;
      {
        std::lock_guard<std::mutex> lock(shard.mutex);
        const auto now = std::chrono::steady_clock::now();
        if (!shard.retryQueue.empty() &&
            shard.retryQueue.begin()->first <= now) {
          request = std::move(shard.retryQueue.begin()->second);
          shard.retryQueue.erase(shard.retryQueue.begin());
        } else {
          auto* source = !shard.priorityQueue.empty() ? &shard.priorityQueue
                                                      : &shard.queue;
          if (source->empty()) {
            return;
          }
          request = std::move(source->front());
          source->pop_front();
        }
      }
      startRequest(shard, std::move(request));
    }
  }

  static bool isRetryable(
      CURLcode result,
      long responseCode,
      size_t written,
      size_t expected) {
    if (responseCode == 408 || responseCode == 429 || responseCode == 500 ||
        responseCode == 502 || responseCode == 503 || responseCode == 504) {
      return true;
    }
    if (responseCode == 206 && written != expected) {
      return true;
    }
    switch (result) {
      case CURLE_COULDNT_RESOLVE_HOST:
      case CURLE_COULDNT_CONNECT:
      case CURLE_OPERATION_TIMEDOUT:
      case CURLE_SEND_ERROR:
      case CURLE_RECV_ERROR:
      case CURLE_PARTIAL_FILE:
      case CURLE_GOT_NOTHING:
      case CURLE_SSL_CONNECT_ERROR:
        return true;
      default:
        return false;
    }
  }

  std::chrono::milliseconds retryDelay(const Request& request) const {
    const auto exponent = std::min<size_t>(request.retryCount - 1, 8);
    const auto exponential = retryBaseDelayMs_ * (size_t{1} << exponent);
    const auto capped = std::min(exponential, retryMaxDelayMs_);
    const auto jitterRange = std::max<size_t>(retryBaseDelayMs_, 1);
    const auto jitter =
        (request.offset ^ request.size ^ request.retryCount) % jitterRange;
    return std::chrono::milliseconds(
        std::min(capped + jitter, retryMaxDelayMs_));
  }

  void finishRequest(Shard& shard, CURL* easy, CURLcode result) {
    auto active = shard.active.find(easy);
    VELOX_CHECK(active != shard.active.end());
    auto request = std::move(active->second);
    shard.active.erase(active);
    long responseCode = 0;
    curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &responseCode);
    curl_off_t totalTimeUs = 0;
    curl_off_t connectTimeUs = 0;
    curl_off_t startTransferTimeUs = 0;
    curl_easy_getinfo(easy, CURLINFO_TOTAL_TIME_T, &totalTimeUs);
    curl_easy_getinfo(easy, CURLINFO_CONNECT_TIME_T, &connectTimeUs);
    curl_easy_getinfo(
        easy, CURLINFO_STARTTRANSFER_TIME_T, &startTransferTimeUs);
    curl_multi_remove_handle(shard.multi, easy);
    curl_easy_cleanup(easy);
    request->easy = nullptr;
    if (request->headers != nullptr) {
      curl_slist_free_all(request->headers);
      request->headers = nullptr;
    }

    completedAttempts_.fetch_add(1, std::memory_order_relaxed);
    totalTimeUs_.fetch_add(totalTimeUs, std::memory_order_relaxed);
    connectTimeUs_.fetch_add(connectTimeUs, std::memory_order_relaxed);
    startTransferTimeUs_.fetch_add(
        startTransferTimeUs, std::memory_order_relaxed);

    const auto succeeded = result == CURLE_OK && responseCode == 206 &&
        request->written == request->size;
    if (!succeeded && request->retryCount < maxRetries_ &&
        isRetryable(result, responseCode, request->written, request->size)) {
      ++request->retryCount;
      const auto delay = retryDelay(*request);
      retryAttempts_.fetch_add(1, std::memory_order_relaxed);
      if (diagnostics_) {
        LOG(WARNING) << "CPP_S3_MULTI retry=" << request->retryCount
                     << " curl=" << curl_easy_strerror(result)
                     << " http=" << responseCode
                     << " expected=" << request->size
                     << " received=" << request->written
                     << " delayMs=" << delay.count();
      }
      request->written = 0;
      {
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.retryQueue.emplace(
            std::chrono::steady_clock::now() + delay, std::move(request));
      }
      shard.condition.notify_one();
      curl_multi_wakeup(shard.multi);
      return;
    }

    const auto completed =
        completedRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (succeeded) {
      completedBytes_.fetch_add(request->written, std::memory_order_relaxed);
    }
    inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
    if (diagnostics_ && completed % 256 == 0) {
      const auto elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started_)
                               .count();
      const auto bytes = completedBytes_.load(std::memory_order_relaxed);
      const auto requests = completedRequests_.load(std::memory_order_relaxed);
      const auto attempts = completedAttempts_.load(std::memory_order_relaxed);
      LOG(WARNING) << "CPP_S3_MULTI completed=" << requests << " submitted="
                   << submittedRequests_.load(std::memory_order_relaxed)
                   << " attempts=" << attempts << " retries="
                   << retryAttempts_.load(std::memory_order_relaxed)
                   << " inflight="
                   << inflightRequests_.load(std::memory_order_relaxed)
                   << " peakInflight="
                   << peakInflightRequests_.load(std::memory_order_relaxed)
                   << " bytes=" << bytes << " elapsedSeconds=" << elapsed
                   << " payloadGbps="
                   << (elapsed == 0 ? 0.0 : bytes * 8.0 / elapsed / 1e9)
                   << " avgTotalUs="
                   << totalTimeUs_.load(std::memory_order_relaxed) / attempts
                   << " avgConnectUs="
                   << connectTimeUs_.load(std::memory_order_relaxed) / attempts
                   << " avgStartTransferUs="
                   << startTransferTimeUs_.load(std::memory_order_relaxed) /
              attempts;
    }

    if (succeeded) {
      request->promise.set_value(request->written);
      return;
    }
    std::ostringstream message;
    message << "Native S3 curl-multi range failed: curl="
            << curl_easy_strerror(result) << " http=" << responseCode
            << " expected=" << request->size
            << " received=" << request->written;
    request->promise.set_exception(
        std::make_exception_ptr(std::runtime_error(message.str())));
  }

  void failQueued(Shard& shard, const std::string& message) {
    std::deque<std::shared_ptr<Request>> requests;
    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      requests.swap(shard.priorityQueue);
      requests.insert(
          requests.end(),
          std::make_move_iterator(shard.queue.begin()),
          std::make_move_iterator(shard.queue.end()));
      shard.queue.clear();
      for (auto& retry : shard.retryQueue) {
        requests.push_back(std::move(retry.second));
      }
      shard.retryQueue.clear();
    }
    for (auto& request : requests) {
      request->promise.set_exception(
          std::make_exception_ptr(std::runtime_error(message)));
    }
  }

  void run(Shard& shard) {
    try {
      while (true) {
        fillActive(shard);
        {
          std::unique_lock<std::mutex> lock(shard.mutex);
          if (shard.active.empty() && shard.priorityQueue.empty() &&
              shard.queue.empty() && shard.retryQueue.empty()) {
            if (shard.stopping) {
              break;
            }
            shard.condition.wait(lock, [&shard] {
              return shard.stopping || !shard.priorityQueue.empty() ||
                  !shard.queue.empty() || !shard.retryQueue.empty();
            });
            continue;
          }
          if (shard.active.empty() && shard.priorityQueue.empty() &&
              shard.queue.empty() && !shard.retryQueue.empty()) {
            shard.condition.wait_until(lock, shard.retryQueue.begin()->first);
            continue;
          }
        }

        int running = 0;
        auto code = curl_multi_perform(shard.multi, &running);
        VELOX_CHECK(
            code == CURLM_OK,
            "curl_multi_perform failed: {}",
            curl_multi_strerror(code));
        int remaining = 0;
        while (auto* message = curl_multi_info_read(shard.multi, &remaining)) {
          if (message->msg == CURLMSG_DONE) {
            finishRequest(shard, message->easy_handle, message->data.result);
          }
        }
        fillActive(shard);
        if (!shard.active.empty()) {
          int descriptors = 0;
          code = curl_multi_poll(shard.multi, nullptr, 0, 100, &descriptors);
          VELOX_CHECK(
              code == CURLM_OK,
              "curl_multi_poll failed: {}",
              curl_multi_strerror(code));
        }
      }
    } catch (const std::exception& error) {
      failQueued(shard, error.what());
      for (auto& [easy, request] : shard.active) {
        curl_multi_remove_handle(shard.multi, easy);
        curl_easy_cleanup(easy);
        if (request->headers != nullptr) {
          curl_slist_free_all(request->headers);
        }
        request->promise.set_exception(std::current_exception());
      }
      shard.active.clear();
    }
  }

  const size_t concurrencyPerShard_;
  const size_t rangeBytes_;
  const size_t metadataPriorityBytes_;
  const size_t maxRetries_;
  const size_t retryBaseDelayMs_;
  const size_t retryMaxDelayMs_;
  const bool diagnostics_;
  const std::chrono::steady_clock::time_point started_;
  std::atomic<size_t> nextShard_{0};
  std::atomic<uint64_t> submittedRequests_{0};
  std::atomic<uint64_t> submittedBytes_{0};
  std::atomic<uint64_t> completedRequests_{0};
  std::atomic<uint64_t> completedAttempts_{0};
  std::atomic<uint64_t> retryAttempts_{0};
  std::atomic<uint64_t> completedBytes_{0};
  std::atomic<uint64_t> inflightRequests_{0};
  std::atomic<uint64_t> peakInflightRequests_{0};
  std::atomic<uint64_t> totalTimeUs_{0};
  std::atomic<uint64_t> connectTimeUs_{0};
  std::atomic<uint64_t> startTransferTimeUs_{0};
  std::vector<std::unique_ptr<Shard>> shards_;
};

// Executor-global admission and priority queue for the official AWS SDK paths.
// The scheduler owns one S3CrtClient per executor when CRT is selected. The
// classic path uses S3ReadFile instances that share S3FileSystem's S3Client.
// Both paths retain official credentials/retry behavior and write directly
// into the caller-provided pinned destination.
class NativeS3SdkScheduler {
 public:
  static NativeS3SdkScheduler& instance() {
    static NativeS3SdkScheduler scheduler;
    return scheduler;
  }

  std::future<size_t> submit(
      std::shared_ptr<facebook::velox::ReadFile> readFile,
      size_t offset,
      size_t size,
      uint8_t* destination,
      const facebook::velox::FileIoContext& context = {},
      bool directCacheFill = false) {
    VELOX_CHECK_NOT_NULL(readFile);
    markStarted();
    auto request = std::make_shared<Request>();
    request->readFile = std::move(readFile);
    request->context = context;
    request->directCacheFill = directCacheFill;
    request->offset = offset;
    request->size = size;
    request->logicalSize = size;
    request->destination = destination;
    auto future = request->promise.get_future();
    recordDirectCacheFillSubmission(*request);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& queue = size <= metadataPriorityBytes_ ? priorityQueue_ : queue_;
      if (request->highThroughput) {
        ++queuedHighPressureRequests_;
      }
      queue.push_back(std::move(request));
    }
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    condition_.notify_one();
    return future;
  }

  std::future<size_t> submitCrt(
      std::string bucket,
      std::string key,
      size_t offset,
      size_t size,
      uint8_t* destination,
      bool directCacheFill = false) {
    VELOX_CHECK(useCrt_);
    markStarted();
    auto request = std::make_shared<Request>();
    request->bucket = std::move(bucket);
    request->key = std::move(key);
    request->offset = offset;
    request->size = size;
    request->logicalSize = size;
    request->destination = destination;
    request->directCacheFill = directCacheFill;
    request->highThroughput = nativeS3HighRequestPressure;
    auto future = request->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& queue = size <= metadataPriorityBytes_ ? priorityQueue_ : queue_;
      if (request->highThroughput) {
        ++queuedHighPressureRequests_;
      }
      queue.push_back(std::move(request));
    }
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    dispatchCrt();
    return future;
  }

  std::future<size_t> submitCrtScatter(
      std::string bucket,
      std::string key,
      size_t offset,
      std::vector<folly::Range<char*>> destinations) {
    VELOX_CHECK(useCrt_);
    VELOX_CHECK(!destinations.empty());
    markStarted();
    auto request = std::make_shared<Request>();
    request->bucket = std::move(bucket);
    request->key = std::move(key);
    request->offset = offset;
    request->directCacheFill = true;
    request->highThroughput = nativeS3HighRequestPressure;
    for (const auto& destination : destinations) {
      VELOX_CHECK_LE(
          destination.size(),
          std::numeric_limits<size_t>::max() - request->size,
          "Native S3 scatter request size overflow");
      request->size += destination.size();
      if (destination.data() != nullptr) {
        VELOX_CHECK_LE(
            destination.size(),
            std::numeric_limits<size_t>::max() - request->logicalSize,
            "Native S3 scatter logical size overflow");
        request->logicalSize += destination.size();
      }
    }
    request->scatterDestinations = std::move(destinations);
    auto future = request->promise.get_future();
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(request->size, std::memory_order_relaxed);
    directCacheFillScatterBuffers_.fetch_add(
        request->scatterDestinations.size(), std::memory_order_relaxed);
    recordDirectCacheFillSubmission(*request);
    enqueueCrt(std::move(request));
    return future;
  }

  folly::SemiFuture<size_t> submitCrtScatterAsync(
      std::string bucket,
      std::string key,
      size_t offset,
      std::vector<folly::Range<char*>> destinations) {
    VELOX_CHECK(useCrt_);
    VELOX_CHECK(!destinations.empty());
    markStarted();
    auto request = std::make_shared<Request>();
    request->bucket = std::move(bucket);
    request->key = std::move(key);
    request->offset = offset;
    request->directCacheFill = true;
    request->highThroughput = nativeS3HighRequestPressure;
    for (const auto& destination : destinations) {
      VELOX_CHECK_LE(
          destination.size(),
          std::numeric_limits<size_t>::max() - request->size,
          "Native S3 scatter request size overflow");
      request->size += destination.size();
      if (destination.data() != nullptr) {
        VELOX_CHECK_LE(
            destination.size(),
            std::numeric_limits<size_t>::max() - request->logicalSize,
            "Native S3 scatter logical size overflow");
        request->logicalSize += destination.size();
      }
    }
    request->scatterDestinations = std::move(destinations);
    request->asyncPromise = std::make_unique<folly::Promise<size_t>>();
    auto future = request->asyncPromise->getSemiFuture();
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(request->size, std::memory_order_relaxed);
    directCacheFillScatterBuffers_.fetch_add(
        request->scatterDestinations.size(), std::memory_order_relaxed);
    recordDirectCacheFillSubmission(*request);
    enqueueCrt(std::move(request));
    return future;
  }

  size_t rangeBytes() const {
    return rangeBytes_;
  }

  facebook::velox::cudf_velox::connector::hive::NativeS3ReadPolicy readPolicy(
      const std::vector<folly::Range<char*>>& destinations) {
    auto policy =
        facebook::velox::cudf_velox::connector::hive::chooseNativeS3ReadPolicy(
            destinations,
            rangeBytes_,
            rangeCoalesceGapBytes_,
            maxCoalescedRangeBytes_,
            adaptiveRanges_);
    if (adaptiveRanges_) {
      adaptiveRangePolicyCalls_.fetch_add(1, std::memory_order_relaxed);
      if (policy.maxRangeBytes >
          std::min(rangeBytes_, maxCoalescedRangeBytes_)) {
        adaptiveWideRangePolicyCalls_.fetch_add(1, std::memory_order_relaxed);
      }
      if (policy.maxGapBytes > 0) {
        adaptiveGapPolicyCalls_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return policy;
  }

  bool sliceOversizedRanges() const {
    return sliceOversizedRanges_;
  }

  void logSnapshot(std::string_view event, std::string_view id) {
    if (!diagnostics_) {
      return;
    }
    uint64_t priorityQueuedBytes = 0;
    uint64_t queuedBytes = 0;
    size_t priorityQueuedRequests = 0;
    size_t queuedRequests = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      priorityQueuedRequests = priorityQueue_.size();
      queuedRequests = queue_.size();
      for (const auto& request : priorityQueue_) {
        priorityQueuedBytes += request->size;
      }
      for (const auto& request : queue_) {
        queuedBytes += request->size;
      }
    }
    const auto submitted = submittedRequests_.load(std::memory_order_relaxed);
    const auto completed = completedRequests_.load(std::memory_order_relaxed);
    const auto submittedBytes = submittedBytes_.load(std::memory_order_relaxed);
    const auto completedBytes = completedBytes_.load(std::memory_order_relaxed);
    LOG(WARNING)
        << (useCrt_ ? "CPP_S3_CRT_SNAPSHOT" : "CPP_S3_SDK_SNAPSHOT")
        << " event=" << event << (id.empty() ? "" : " id=") << id
        << " submitted=" << submitted << " completed=" << completed
        << " inflight=" << inflightRequests_.load(std::memory_order_relaxed)
        << " queued=" << queuedRequests
        << " priorityQueued=" << priorityQueuedRequests
        << " queuedBytes=" << queuedBytes
        << " priorityQueuedBytes=" << priorityQueuedBytes << " peakInflight="
        << peakInflightRequests_.load(std::memory_order_relaxed)
        << " submittedBytes=" << submittedBytes
        << " completedBytes=" << completedBytes << " avgSubmittedBytes="
        << (submitted == 0 ? 0 : submittedBytes / submitted)
        << " avgCompletedBytes="
        << (completed == 0 ? 0 : completedBytes / completed)
        << " directCacheFillSubmitted="
        << directCacheFillSubmittedRequests_.load(std::memory_order_relaxed)
        << " directCacheFillCompleted="
        << directCacheFillCompletedRequests_.load(std::memory_order_relaxed)
        << " directCacheFillBytes="
        << directCacheFillSubmittedBytes_.load(std::memory_order_relaxed)
        << " directCacheFillLogicalBytes="
        << directCacheFillLogicalBytes_.load(std::memory_order_relaxed)
        << " directCacheFillGapBytes="
        << directCacheFillGapBytes_.load(std::memory_order_relaxed)
        << " directCacheFillScatterBuffers="
        << directCacheFillScatterBuffers_.load(std::memory_order_relaxed)
        << " highThroughputSubmitted="
        << highThroughputSubmittedRequests_.load(std::memory_order_relaxed)
        << " highThroughputCompleted="
        << highThroughputCompletedRequests_.load(std::memory_order_relaxed)
        << " highPressurePeakInflight="
        << highThroughputPeakInflightRequests_.load(std::memory_order_relaxed)
        << " adaptiveRanges=" << adaptiveRanges_ << " adaptiveRangePolicyCalls="
        << adaptiveRangePolicyCalls_.load(std::memory_order_relaxed)
        << " adaptiveWideRangePolicyCalls="
        << adaptiveWideRangePolicyCalls_.load(std::memory_order_relaxed)
        << " adaptiveGapPolicyCalls="
        << adaptiveGapPolicyCalls_.load(std::memory_order_relaxed)
        << " demandKeyPriorityDispatches="
        << demandKeyPriorityDispatches_.load(std::memory_order_relaxed)
        << " completionKeyContinuityDispatches="
        << completionKeyContinuityDispatches_.load(std::memory_order_relaxed)
        << " retries=" << retryAttempts_.load(std::memory_order_relaxed)
        << " totalTimeUs=" << totalTimeUs_.load(std::memory_order_relaxed);
  }

  NativeS3SdkScheduler(const NativeS3SdkScheduler&) = delete;
  NativeS3SdkScheduler& operator=(const NativeS3SdkScheduler&) = delete;

  bool prepareHighThroughputCrtClient() {
    if (!useCrt_ || highThroughputTargetGbps_ <= baseTargetGbps_) {
      return false;
    }
    if (highThroughputCrtClient_.load(std::memory_order_acquire) != nullptr) {
      return true;
    }
    std::lock_guard<std::mutex> lock(highThroughputCrtClientMutex_);
    if (highThroughputCrtClient_.load(std::memory_order_relaxed) == nullptr) {
      auto client =
          makeCrtClient(highThroughputTargetGbps_, highThroughputWindowBytes_);
      highThroughputCrtClient_.store(
          std::move(client), std::memory_order_release);
      LOG(WARNING)
          << "CPP_S3_CRT_ADAPTIVE_PRESSURE_CLIENT_READY baseTargetGbps="
          << baseTargetGbps_ << " highTargetGbps=" << highThroughputTargetGbps_
          << " baseWindowBytes=" << baseWindowBytes_
          << " highWindowBytes=" << highThroughputWindowBytes_;
    }
    return true;
  }

  bool prioritizeFile(const std::string& bucket, const std::string& key) {
    if (!useCrt_ || !demandPriority_) {
      return false;
    }
    bool prioritized = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto identity = fileIdentity(bucket, key);
      if (std::any_of(queue_.begin(), queue_.end(), [&](const auto& queued) {
            return fileIdentity(queued->bucket, queued->key) == identity;
          })) {
        demandPreferredFiles_.insert(identity);
        prioritized = true;
      }
    }
    if (prioritized) {
      dispatchCrt();
    }
    return prioritized;
  }

 private:
  struct Request {
    std::shared_ptr<facebook::velox::ReadFile> readFile;
    facebook::velox::FileIoContext context;
    std::string bucket;
    std::string key;
    size_t offset{0};
    size_t size{0};
    size_t logicalSize{0};
    uint8_t* destination{nullptr};
    std::vector<folly::Range<char*>> scatterDestinations;
    bool directCacheFill{false};
    bool highThroughput{false};
    bool usedHighThroughputClient{false};
    std::promise<size_t> promise;
    std::unique_ptr<folly::Promise<size_t>> asyncPromise;
  };

  static size_t envOrDefault(const char* name, size_t defaultValue) {
    const auto parsed = envBytesOrZero(name);
    return parsed == 0 ? defaultValue : parsed;
  }

  void recordDirectCacheFillSubmission(const Request& request) {
    if (!request.directCacheFill) {
      return;
    }
    directCacheFillSubmittedRequests_.fetch_add(1, std::memory_order_relaxed);
    directCacheFillSubmittedBytes_.fetch_add(
        request.size, std::memory_order_relaxed);
    directCacheFillLogicalBytes_.fetch_add(
        request.logicalSize, std::memory_order_relaxed);
    directCacheFillGapBytes_.fetch_add(
        request.size - request.logicalSize, std::memory_order_relaxed);
  }

  void markStarted() {
    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    int64_t expected = 0;
    startedNs_.compare_exchange_strong(
        expected, nowNs, std::memory_order_relaxed);
  }

  std::shared_ptr<Aws::S3Crt::S3CrtClient> makeCrtClient(
      size_t targetGbps,
      size_t downloadWindowBytes) {
    Aws::S3Crt::ClientConfiguration config;
    if (const auto* region = std::getenv("AWS_REGION");
        region != nullptr && region[0] != '\0') {
      config.region = region;
    } else if (const auto* region = std::getenv("AWS_DEFAULT_REGION");
               region != nullptr && region[0] != '\0') {
      config.region = region;
    }
    config.throughputTargetGbps = static_cast<double>(targetGbps);
    config.partSize = envOrDefault("GLUTEN_CPP_S3_CRT_PART_BYTES", 8UL << 20);
    config.downloadMemoryUsageWindow = downloadWindowBytes;
    config.crtRetryStrategyConfig.crtRetryStrategyType =
        Aws::S3Crt::S3CrtClientConfiguration::CrtRetryStrategyConfig::
            CrtRetryStrategyType::EXPONENTIAL_BACKOFF;
    config.crtRetryStrategyConfig.config.maxRetries =
        envOrDefault("GLUTEN_CPP_S3_CRT_MAX_RETRIES", 5);
    auto credentialsProvider = facebook::velox::filesystems::
        makeSynchronizedCachingCredentialsProvider(
            std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>());
    return std::make_shared<Aws::S3Crt::S3CrtClient>(
        std::move(credentialsProvider), config);
  }

  NativeS3SdkScheduler()
      : useCrt_(envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0),
        concurrency_(envOrDefault("GLUTEN_CPP_S3_SDK_CONCURRENCY", 128)),
        highThroughputConcurrency_(
            std::min(
                concurrency_,
                envOrDefault(
                    "GLUTEN_CPP_S3_CRT_HIGH_MAX_CONCURRENCY",
                    concurrency_))),
        rangeBytes_(envOrDefault("GLUTEN_CPP_S3_SDK_RANGE_BYTES", 4UL << 20)),
        adaptiveRanges_(envFlagOrDefault(
            "GLUTEN_CPP_S3_ADAPTIVE_RANGES",
            adaptiveS3PrefetchEnabled())),
        rangeCoalesceGapBytes_(envBytesOrDefault(
            "GLUTEN_CPP_S3_RANGE_COALESCE_GAP_BYTES",
            adaptiveRanges_ ? 1UL << 20 : 0)),
        maxCoalescedRangeBytes_(envOrDefault(
            "GLUTEN_CPP_S3_MAX_COALESCED_RANGE_BYTES",
            adaptiveRanges_ ? 32UL << 20 : 16UL << 20)),
        sliceOversizedRanges_(
            envOrDefault("GLUTEN_CPP_S3_SLICE_OVERSIZED_RANGES", 1) != 0),
        metadataPriorityBytes_(
            envOrDefault("GLUTEN_CPP_S3_METADATA_PRIORITY_BYTES", 64UL << 10)),
        demandPriority_(envFlagOrDefault(
            "GLUTEN_CPP_S3_DEMAND_PRIORITY",
            adaptiveS3PrefetchEnabled())),
        completionKeyContinuity_(envFlagOrDefault(
            "GLUTEN_CPP_S3_COMPLETION_KEY_CONTINUITY",
            adaptiveS3PrefetchEnabled())),
        completionKeyContinuityMinBytes_(
            envOrDefault("GLUTEN_CPP_S3_COMPLETION_KEY_MIN_BYTES", 1UL << 20)),
        diagnostics_(envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0) {
    VELOX_CHECK_GT(concurrency_, 0);
    VELOX_CHECK_GT(highThroughputConcurrency_, 0);
    VELOX_CHECK_GT(rangeBytes_, 0);
    VELOX_CHECK_GT(maxCoalescedRangeBytes_, 0);
    VELOX_CHECK_GT(completionKeyContinuityMinBytes_, 0);
    if (completionKeyContinuity_) {
      LOG(WARNING) << "CPP_S3_COMPLETION_KEY_CONTINUITY enabled=1 minBytes="
                   << completionKeyContinuityMinBytes_;
    }
    if (demandPriority_) {
      LOG(WARNING) << "CPP_S3_DEMAND_PRIORITY enabled=1";
    }
    if (adaptiveRanges_) {
      LOG(WARNING) << "CPP_S3_ADAPTIVE_RANGES enabled=1 baseBytes="
                   << rangeBytes_ << " maxBytes=" << maxCoalescedRangeBytes_
                   << " maxGapBytes=" << rangeCoalesceGapBytes_
                   << " maxGapOverheadDivisor=32";
    }
    if (useCrt_) {
      baseTargetGbps_ = envOrDefault("GLUTEN_CPP_S3_CRT_TARGET_GBPS", 25);
      baseWindowBytes_ =
          envOrDefault("GLUTEN_CPP_S3_CRT_DOWNLOAD_WINDOW_BYTES", 1UL << 30);
      crtClient_ = makeCrtClient(baseTargetGbps_, baseWindowBytes_);
      highThroughputTargetGbps_ =
          envBytesOrZero("GLUTEN_CPP_S3_CRT_HIGH_TARGET_GBPS");
      if (highThroughputTargetGbps_ > baseTargetGbps_) {
        highThroughputWindowBytes_ = envOrDefault(
            "GLUTEN_CPP_S3_CRT_HIGH_DOWNLOAD_WINDOW_BYTES",
            std::max<size_t>(baseWindowBytes_, 4UL << 30));
        LOG(WARNING)
            << "CPP_S3_CRT_ADAPTIVE_PRESSURE configured=1 lazyClient=1 "
            << "baseTargetGbps=" << baseTargetGbps_
            << " highTargetGbps=" << highThroughputTargetGbps_
            << " baseWindowBytes=" << baseWindowBytes_
            << " highWindowBytes=" << highThroughputWindowBytes_
            << " highMaxConcurrency=" << highThroughputConcurrency_;
      }
    }
    if (!useCrt_) {
      workers_.reserve(concurrency_);
      for (size_t index = 0; index < concurrency_; ++index) {
        workers_.emplace_back([this] { run(); });
      }
    }
  }

  ~NativeS3SdkScheduler() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void enqueueCrt(std::shared_ptr<Request> request) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      enqueueCrtLocked(std::move(request));
    }
    dispatchCrt();
  }

  void enqueueCrtLocked(std::shared_ptr<Request> request) {
    auto& queue =
        request->size <= metadataPriorityBytes_ ? priorityQueue_ : queue_;
    if (request->highThroughput) {
      ++queuedHighPressureRequests_;
    }
    queue.push_back(std::move(request));
  }

  void dispatchCrt() {
    while (true) {
      std::shared_ptr<Request> request;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || (priorityQueue_.empty() && queue_.empty())) {
          return;
        }
        const auto highInflight =
            highThroughputInflightRequests_.load(std::memory_order_relaxed);
        const bool highPressureActive =
            highInflight > 0 || queuedHighPressureRequests_ > 0;
        const auto effectiveConcurrency =
            highPressureActive ? highThroughputConcurrency_ : concurrency_;
        if (inflightRequests_.load(std::memory_order_relaxed) >=
            effectiveConcurrency) {
          return;
        }
        const auto canDispatch = [&](const auto& queued) {
          return !queued->highThroughput ||
              highInflight < highThroughputConcurrency_;
        };
        auto prioritySelected = std::find_if(
            priorityQueue_.begin(), priorityQueue_.end(), canDispatch);
        auto regularSelected =
            std::find_if(queue_.begin(), queue_.end(), canDispatch);
        if (prioritySelected == priorityQueue_.end() &&
            regularSelected == queue_.end()) {
          return;
        }
        auto& source =
            prioritySelected != priorityQueue_.end() ? priorityQueue_ : queue_;
        auto selected = prioritySelected != priorityQueue_.end()
            ? prioritySelected
            : regularSelected;
        bool selectedByDemand = false;
        if (&source == &queue_ && !demandPreferredFiles_.empty()) {
          const auto demanded = std::find_if(
              source.begin(), source.end(), [&](const auto& queued) {
                return canDispatch(queued) &&
                    demandPreferredFiles_.contains(
                        fileIdentity(queued->bucket, queued->key));
              });
          if (demanded != source.end()) {
            selected = demanded;
            selectedByDemand = true;
            const auto selectedIdentity =
                fileIdentity((*selected)->bucket, (*selected)->key);
            const auto anotherQueued = std::find_if(
                std::next(selected), source.end(), [&](const auto& queued) {
                  return fileIdentity(queued->bucket, queued->key) ==
                      selectedIdentity;
                });
            if (anotherQueued == source.end()) {
              demandPreferredFiles_.erase(selectedIdentity);
            }
            demandKeyPriorityDispatches_.fetch_add(
                1, std::memory_order_relaxed);
          }
        }
        if (!selectedByDemand && completionKeyContinuity_ &&
            &source == &queue_ && !completionPreferredKeys_.empty()) {
          const auto preferred = std::find_if(
              source.begin(), source.end(), [&](const auto& queued) {
                return canDispatch(queued) && queued->directCacheFill &&
                    queued->size >= completionKeyContinuityMinBytes_ &&
                    completionPreferredKeys_.contains(queued->key);
              });
          if (preferred != source.end()) {
            selected = preferred;
            completionPreferredKeys_.erase((*selected)->key);
            completionKeyContinuityDispatches_.fetch_add(
                1, std::memory_order_relaxed);
          }
        }
        request = std::move(*selected);
        source.erase(selected);
        if (request->highThroughput) {
          VELOX_CHECK_GT(queuedHighPressureRequests_, 0);
          --queuedHighPressureRequests_;
        }
        const auto inflight =
            inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (request->highThroughput) {
          const auto highThroughputInflight =
              highThroughputInflightRequests_.fetch_add(
                  1, std::memory_order_relaxed) +
              1;
          auto highThroughputPeak = highThroughputPeakInflightRequests_.load(
              std::memory_order_relaxed);
          while (highThroughputInflight > highThroughputPeak &&
                 !highThroughputPeakInflightRequests_.compare_exchange_weak(
                     highThroughputPeak,
                     highThroughputInflight,
                     std::memory_order_relaxed)) {
          }
        }
        auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
        while (inflight > peak &&
               !peakInflightRequests_.compare_exchange_weak(
                   peak, inflight, std::memory_order_relaxed)) {
        }
      }

      auto get = std::make_shared<Aws::S3Crt::Model::GetObjectRequest>();
      get->SetBucket(request->bucket.c_str());
      get->SetKey(request->key.c_str());
      get->SetRange(
          fmt::format(
              "bytes={}-{}",
              request->offset,
              request->offset + request->size - 1)
              .c_str());
      if (request->scatterDestinations.empty()) {
        const auto destination = request->destination;
        const auto size = request->size;
        get->SetResponseStreamFactory([destination, size]() {
          return Aws::New<facebook::velox::filesystems::StringViewStream>(
              "NativeS3CrtScheduler", destination, size);
        });
      } else {
        const auto destinations = request->scatterDestinations;
        get->SetResponseStreamFactory([destinations]() {
          return Aws::New<facebook::velox::cudf_velox::connector::hive::
                              NativeS3ScatterWriteStream>(
              "NativeS3CrtScatter", destinations);
        });
      }
      const auto started = std::chrono::steady_clock::now();
      auto client = crtClient_;
      if (request->highThroughput) {
        const auto highThroughputClient =
            highThroughputCrtClient_.load(std::memory_order_acquire);
        if (highThroughputClient) {
          client = highThroughputClient;
          request->usedHighThroughputClient = true;
          highThroughputSubmittedRequests_.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      client->GetObjectAsync(
          *get,
          [this, request, get, started](
              const Aws::S3Crt::S3CrtClient*,
              const Aws::S3Crt::Model::GetObjectRequest&,
              Aws::S3Crt::Model::GetObjectOutcome outcome,
              const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) {
            bool succeeded = false;
            try {
              if (outcome.IsSuccess()) {
                retryAttempts_.fetch_add(
                    outcome.GetRetryCount(), std::memory_order_relaxed);
              }
              VELOX_CHECK(
                  outcome.IsSuccess(),
                  "Native S3 CRT range failed for s3://{}/{}: {}",
                  request->bucket,
                  request->key,
                  outcome.GetError().GetMessage());
              if (!request->scatterDestinations.empty()) {
                auto* stream =
                    dynamic_cast<facebook::velox::cudf_velox::connector::hive::
                                     NativeS3ScatterWriteStream*>(
                        &outcome.GetResult().GetBody());
                VELOX_CHECK_NOT_NULL(
                    stream, "Native S3 CRT scatter response stream mismatch");
                VELOX_CHECK(
                    !stream->overflowed(),
                    "Native S3 CRT scatter response exceeded destinations");
                VELOX_CHECK_EQ(
                    stream->bytesWritten(),
                    request->size,
                    "Short native S3 CRT scatter response");
              }
              if (request->asyncPromise) {
                request->asyncPromise->setValue(request->size);
              } else {
                request->promise.set_value(request->size);
              }
              succeeded = true;
            } catch (...) {
              if (request->asyncPromise) {
                request->asyncPromise->setException(
                    folly::exception_wrapper(std::current_exception()));
              } else {
                request->promise.set_exception(std::current_exception());
              }
            }
            const auto elapsedUs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            totalTimeUs_.fetch_add(elapsedUs, std::memory_order_relaxed);
            if (succeeded) {
              completedBytes_.fetch_add(
                  request->size, std::memory_order_relaxed);
              if (request->directCacheFill) {
                directCacheFillCompletedRequests_.fetch_add(
                    1, std::memory_order_relaxed);
              }
              if (request->usedHighThroughputClient) {
                highThroughputCompletedRequests_.fetch_add(
                    1, std::memory_order_relaxed);
              }
              if (completionKeyContinuity_ && request->directCacheFill &&
                  request->size >= completionKeyContinuityMinBytes_) {
                std::lock_guard<std::mutex> lock(mutex_);
                completionPreferredKeys_.insert(request->key);
              }
            }
            if (request->highThroughput) {
              highThroughputInflightRequests_.fetch_sub(
                  1, std::memory_order_relaxed);
            }
            inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
            const auto completed =
                completedRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
            logProgress(completed);
            dispatchCrt();
          });
    }
  }

  void logProgress(uint64_t completed) {
    if (!diagnostics_ || completed % 256 != 0) {
      return;
    }
    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    const auto startedNs = startedNs_.load(std::memory_order_relaxed);
    VELOX_CHECK_GT(startedNs, 0);
    const auto elapsed = (nowNs - startedNs) / 1e9;
    const auto bytes = completedBytes_.load(std::memory_order_relaxed);
    LOG(WARNING)
        << (useCrt_ ? "CPP_S3_CRT" : "CPP_S3_SDK") << " completed=" << completed
        << " submitted=" << submittedRequests_.load(std::memory_order_relaxed)
        << " inflight=" << inflightRequests_.load(std::memory_order_relaxed)
        << " peakInflight="
        << peakInflightRequests_.load(std::memory_order_relaxed)
        << " bytes=" << bytes << " directCacheFillSubmitted="
        << directCacheFillSubmittedRequests_.load(std::memory_order_relaxed)
        << " directCacheFillCompleted="
        << directCacheFillCompletedRequests_.load(std::memory_order_relaxed)
        << " directCacheFillBytes="
        << directCacheFillSubmittedBytes_.load(std::memory_order_relaxed)
        << " directCacheFillLogicalBytes="
        << directCacheFillLogicalBytes_.load(std::memory_order_relaxed)
        << " directCacheFillGapBytes="
        << directCacheFillGapBytes_.load(std::memory_order_relaxed)
        << " directCacheFillScatterBuffers="
        << directCacheFillScatterBuffers_.load(std::memory_order_relaxed)
        << " highThroughputSubmitted="
        << highThroughputSubmittedRequests_.load(std::memory_order_relaxed)
        << " highThroughputCompleted="
        << highThroughputCompletedRequests_.load(std::memory_order_relaxed)
        << " highPressurePeakInflight="
        << highThroughputPeakInflightRequests_.load(std::memory_order_relaxed)
        << " demandKeyPriorityDispatches="
        << demandKeyPriorityDispatches_.load(std::memory_order_relaxed)
        << " completionKeyContinuityDispatches="
        << completionKeyContinuityDispatches_.load(std::memory_order_relaxed)
        << " retries=" << retryAttempts_.load(std::memory_order_relaxed)
        << " elapsedSeconds=" << elapsed
        << " payloadGbps=" << (elapsed == 0 ? 0.0 : bytes * 8.0 / elapsed / 1e9)
        << " avgTotalUs="
        << totalTimeUs_.load(std::memory_order_relaxed) / completed;
  }

  void run() {
    while (true) {
      std::shared_ptr<Request> request;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return stopping_ || !priorityQueue_.empty() || !queue_.empty();
        });
        if (stopping_ && priorityQueue_.empty() && queue_.empty()) {
          return;
        }
        auto& source = !priorityQueue_.empty() ? priorityQueue_ : queue_;
        request = std::move(source.front());
        source.pop_front();
      }

      const auto inflight =
          inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
      auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
      while (inflight > peak &&
             !peakInflightRequests_.compare_exchange_weak(
                 peak, inflight, std::memory_order_relaxed)) {
      }
      const auto started = std::chrono::steady_clock::now();
      bool succeeded = false;
      try {
        const auto view = request->readFile->pread(
            request->offset,
            request->size,
            request->destination,
            request->context);
        VELOX_CHECK_EQ(
            view.size(), request->size, "Short AWS SDK S3 range read");
        request->promise.set_value(view.size());
        succeeded = true;
      } catch (...) {
        request->promise.set_exception(std::current_exception());
      }
      const auto elapsedUs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started)
              .count();
      totalTimeUs_.fetch_add(elapsedUs, std::memory_order_relaxed);
      if (succeeded) {
        completedBytes_.fetch_add(request->size, std::memory_order_relaxed);
        if (request->directCacheFill) {
          directCacheFillCompletedRequests_.fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
      const auto completed =
          completedRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
      logProgress(completed);
    }
  }

  const bool useCrt_;
  const size_t concurrency_;
  const size_t highThroughputConcurrency_;
  const size_t rangeBytes_;
  const bool adaptiveRanges_;
  const size_t rangeCoalesceGapBytes_;
  const size_t maxCoalescedRangeBytes_;
  const bool sliceOversizedRanges_;
  const size_t metadataPriorityBytes_;
  const bool demandPriority_;
  const bool completionKeyContinuity_;
  const size_t completionKeyContinuityMinBytes_;
  const bool diagnostics_;
  static std::string fileIdentity(
      const std::string& bucket,
      const std::string& key) {
    return bucket + '\n' + key;
  }
  std::atomic<int64_t> startedNs_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::shared_ptr<Request>> priorityQueue_;
  std::deque<std::shared_ptr<Request>> queue_;
  size_t queuedHighPressureRequests_{0};
  std::unordered_set<std::string> demandPreferredFiles_;
  std::unordered_set<std::string> completionPreferredKeys_;
  bool stopping_{false};
  std::vector<std::thread> workers_;
  std::shared_ptr<Aws::S3Crt::S3CrtClient> crtClient_;
  size_t baseTargetGbps_{0};
  size_t baseWindowBytes_{0};
  size_t highThroughputTargetGbps_{0};
  size_t highThroughputWindowBytes_{0};
  std::mutex highThroughputCrtClientMutex_;
  std::atomic<std::shared_ptr<Aws::S3Crt::S3CrtClient>>
      highThroughputCrtClient_;
  std::atomic<uint64_t> submittedRequests_{0};
  std::atomic<uint64_t> submittedBytes_{0};
  std::atomic<uint64_t> directCacheFillSubmittedRequests_{0};
  std::atomic<uint64_t> directCacheFillCompletedRequests_{0};
  std::atomic<uint64_t> directCacheFillSubmittedBytes_{0};
  std::atomic<uint64_t> directCacheFillLogicalBytes_{0};
  std::atomic<uint64_t> directCacheFillGapBytes_{0};
  std::atomic<uint64_t> directCacheFillScatterBuffers_{0};
  std::atomic<uint64_t> highThroughputSubmittedRequests_{0};
  std::atomic<uint64_t> highThroughputCompletedRequests_{0};
  std::atomic<uint64_t> highThroughputInflightRequests_{0};
  std::atomic<uint64_t> highThroughputPeakInflightRequests_{0};
  std::atomic<uint64_t> adaptiveRangePolicyCalls_{0};
  std::atomic<uint64_t> adaptiveWideRangePolicyCalls_{0};
  std::atomic<uint64_t> adaptiveGapPolicyCalls_{0};
  std::atomic<uint64_t> demandKeyPriorityDispatches_{0};
  std::atomic<uint64_t> completionKeyContinuityDispatches_{0};
  std::atomic<uint64_t> completedRequests_{0};
  std::atomic<uint64_t> completedBytes_{0};
  std::atomic<uint64_t> inflightRequests_{0};
  std::atomic<uint64_t> peakInflightRequests_{0};
  std::atomic<uint64_t> totalTimeUs_{0};
  std::atomic<uint64_t> retryAttempts_{0};
};

class NativeScheduledS3ReadFile final : public facebook::velox::ReadFile {
 public:
  NativeScheduledS3ReadFile(
      std::shared_ptr<facebook::velox::ReadFile> readFile,
      const std::string& filePath)
      : readFile_(std::move(readFile)) {
    VELOX_CHECK_NOT_NULL(readFile_);
    VELOX_CHECK(
        filePath.starts_with("s3://"),
        "Native scheduled S3 ReadFile requires an S3 path: {}",
        filePath);
    auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
    bucket_ = std::move(bucketAndObject.first);
    key_ = std::move(bucketAndObject.second);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const facebook::velox::FileIoContext& context) const override {
    VELOX_CHECK_NOT_NULL(buffer);
    readRanges(
        offset,
        {folly::Range<char*>(
            static_cast<char*>(buffer), static_cast<size_t>(length))},
        context);
    return {static_cast<char*>(buffer), static_cast<size_t>(length)};
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const facebook::velox::FileIoContext& context) const override {
    return readRanges(offset, buffers, context);
  }

  bool hasPreadvAsync() const override {
    return envFlagOrDefault(
               "GLUTEN_CPP_S3_ASYNC_CACHE_READ_GROUPS",
               adaptiveS3PrefetchEnabled()) &&
        envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const facebook::velox::FileIoContext& context) const override {
    if (!hasPreadvAsync()) {
      return facebook::velox::ReadFile::preadvAsync(offset, buffers, context);
    }
    auto& scheduler = NativeS3SdkScheduler::instance();
    const auto policy = scheduler.readPolicy(buffers);
    auto groups = facebook::velox::cudf_velox::connector::hive::
        groupNativeS3ReadDestinations(
            offset,
            buffers,
            policy.maxGapBytes,
            policy.maxRangeBytes,
            scheduler.sliceOversizedRanges());
    std::vector<folly::SemiFuture<size_t>> pending;
    pending.reserve(groups.size());
    uint64_t submittedBytes = 0;
    for (const auto& group : groups) {
      submittedBytes += group.size;
    }
    for (auto& group : groups) {
      pending.push_back(scheduler.submitCrtScatterAsync(
          bucket_, key_, group.offset, std::move(group.destinations)));
    }
    uint64_t logicalBytes = 0;
    for (const auto& buffer : buffers) {
      VELOX_CHECK_LE(
          buffer.size(),
          std::numeric_limits<uint64_t>::max() - logicalBytes,
          "Native scheduled S3 async read size overflow");
      logicalBytes += buffer.size();
    }
    return folly::collectAll(std::move(pending))
        .deferValue([this, submittedBytes, logicalBytes](
                        std::vector<folly::Try<size_t>> results) {
          uint64_t completedBytes = 0;
          for (auto& result : results) {
            completedBytes += std::move(result).value();
          }
          VELOX_CHECK_EQ(
              completedBytes,
              submittedBytes,
              "Short executor-global asynchronous native S3 read");
          bytesRead_.fetch_add(submittedBytes, std::memory_order_relaxed);
          return logicalBytes;
        });
  }

  bool shouldCoalesce() const override {
    return readFile_->shouldCoalesce();
  }

  uint64_t size() const override {
    return readFile_->size();
  }

  uint64_t memoryUsage() const override {
    return sizeof(*this) + bucket_.capacity() + key_.capacity();
  }

  std::string getName() const override {
    return readFile_->getName();
  }

  uint64_t getNaturalReadSize() const override {
    return readFile_->getNaturalReadSize();
  }

 private:
  uint64_t readRanges(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const facebook::velox::FileIoContext& context) const {
    auto& scheduler = NativeS3SdkScheduler::instance();
    const auto useCrt = envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
    std::vector<std::future<size_t>> pending;
    uint64_t logicalOffset = offset;
    uint64_t logicalBytes = 0;
    uint64_t submittedBytes = 0;
    if (useCrt) {
      const auto policy = scheduler.readPolicy(buffers);
      auto groups = facebook::velox::cudf_velox::connector::hive::
          groupNativeS3ReadDestinations(
              offset,
              buffers,
              policy.maxGapBytes,
              policy.maxRangeBytes,
              scheduler.sliceOversizedRanges());
      pending.reserve(groups.size());
      for (auto& group : groups) {
        submittedBytes += group.size;
        pending.emplace_back(scheduler.submitCrtScatter(
            bucket_, key_, group.offset, std::move(group.destinations)));
      }
      for (const auto& buffer : buffers) {
        VELOX_CHECK_LE(
            buffer.size(),
            std::numeric_limits<uint64_t>::max() - logicalBytes,
            "Native scheduled S3 read size overflow");
        logicalBytes += buffer.size();
      }
    } else {
      for (const auto& buffer : buffers) {
        VELOX_CHECK_LE(
            buffer.size(),
            std::numeric_limits<uint64_t>::max() - logicalBytes,
            "Native scheduled S3 read size overflow");
        logicalBytes += buffer.size();
        if (buffer.data() != nullptr) {
          size_t scheduled = 0;
          while (scheduled < buffer.size()) {
            const auto chunkSize =
                std::min(scheduler.rangeBytes(), buffer.size() - scheduled);
            pending.emplace_back(scheduler.submit(
                readFile_,
                logicalOffset + scheduled,
                chunkSize,
                reinterpret_cast<uint8_t*>(buffer.data() + scheduled),
                context,
                /*directCacheFill=*/true));
            submittedBytes += chunkSize;
            scheduled += chunkSize;
          }
        }
        VELOX_CHECK_LE(
            buffer.size(),
            std::numeric_limits<uint64_t>::max() - logicalOffset,
            "Native scheduled S3 read offset overflow");
        logicalOffset += buffer.size();
      }
    }
    uint64_t completedBytes = 0;
    for (auto& future : pending) {
      completedBytes += future.get();
    }
    VELOX_CHECK_EQ(
        completedBytes, submittedBytes, "Short executor-global native S3 read");
    bytesRead_.fetch_add(submittedBytes, std::memory_order_relaxed);
    return logicalBytes;
  }

  const std::shared_ptr<facebook::velox::ReadFile> readFile_;
  std::string bucket_;
  std::string key_;
};
#endif

template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

void prepareNativeS3HighThroughputClient() {
#ifdef VELOX_ENABLE_S3
  NativeS3SdkScheduler::instance().prepareHighThroughputCrtClient();
#endif
}

void runWithNativeS3RequestPressure(
    bool highThroughput,
    const std::function<void()>& producer) {
  VELOX_CHECK(static_cast<bool>(producer));
  const auto previous = nativeS3HighRequestPressure;
  nativeS3HighRequestPressure = highThroughput;
  try {
    producer();
  } catch (...) {
    nativeS3HighRequestPressure = previous;
    throw;
  }
  nativeS3HighRequestPressure = previous;
}

bool nativeS3HighRequestPressureForCurrentThread() {
  return nativeS3HighRequestPressure;
}

DirectCachePageH2dStats directCachePageH2dStats() {
  return {
      directCachePageH2dCopies.load(std::memory_order_relaxed),
      directCachePageH2dBytes.load(std::memory_order_relaxed),
      directCachePageH2dPinnedCopies.load(std::memory_order_relaxed),
      directCachePageH2dPinnedBytes.load(std::memory_order_relaxed)};
}

CachePageRegistrationStats cachePageRegistrationStats() {
  return {
      cachePageRegistrationAttempts.load(std::memory_order_relaxed),
      cachePageRegistrationSuccesses.load(std::memory_order_relaxed),
      cachePageRegistrationFailures.load(std::memory_order_relaxed),
      cachePageRegistrationBudgetRejectedBytes.load(std::memory_order_relaxed),
      cachePageRegisteredRuns.load(std::memory_order_relaxed),
      cachePageRegisteredBytes.load(std::memory_order_relaxed),
      cachePageCurrentRegisteredBytes.load(std::memory_order_relaxed),
      cachePagePeakRegisteredBytes.load(std::memory_order_relaxed),
      cachePageUnregisteredRuns.load(std::memory_order_relaxed),
      cachePageUnregisteredBytes.load(std::memory_order_relaxed),
      cachePagePrewarmAttempts.load(std::memory_order_relaxed),
      cachePagePrewarmSuccesses.load(std::memory_order_relaxed),
      cachePagePrewarmFailures.load(std::memory_order_relaxed),
      cachePagePrewarmRuns.load(std::memory_order_relaxed),
      cachePagePrewarmBytes.load(std::memory_order_relaxed),
      cachePagePrewarmCoveredRuns.load(std::memory_order_relaxed),
      cachePagePrewarmCoveredBytes.load(std::memory_order_relaxed),
      cachePageRegistrationWallNanos.load(std::memory_order_relaxed),
      cachePagePrewarmWallNanos.load(std::memory_order_relaxed)};
}

BoundedCachePageRegistration makeBoundedCachePageRegistration(
    uint64_t maxRegisteredBytes,
    std::optional<CachePageHostRegistrationHooks> hooks) {
  if (maxRegisteredBytes == 0) {
    return {};
  }
  auto effectiveHooks = hooks.value_or(
      CachePageHostRegistrationHooks{
          .registerRun =
              [](void* address, size_t bytes) {
                // Portable makes the registration visible to every CUDA context
                // in this executor. No-device/driver errors are ordinary
                // fallback.
                return cudaHostRegister(
                           address, bytes, cudaHostRegisterPortable) ==
                    cudaSuccess;
              },
          .unregisterRun =
              [](void* address) {
                const auto status = cudaHostUnregister(address);
                if (status != cudaSuccess) {
                  LOG(WARNING) << "cudaHostUnregister failed for cache run: "
                               << cudaGetErrorString(status);
                }
              }});
  VELOX_CHECK(effectiveHooks.registerRun);
  VELOX_CHECK(effectiveHooks.unregisterRun);
  auto manager = std::make_shared<BoundedCachePageRegistrationState>(
      maxRegisteredBytes, std::move(effectiveHooks));
  return {
      .registerBackingRuns =
          [manager](const memory::Allocation& allocation) {
            return manager->registerAllocation(allocation);
          },
      .registerBackingRange =
          [manager](void* address, uint64_t bytes) {
            return manager->registerRange(address, bytes);
          },
      .registerPersistentBackingRange =
          [manager](
              memory::MmapAllocator& allocator, void* address, uint64_t bytes) {
            return manager->registerPersistentRange(allocator, address, bytes);
          },
      .persistentLifetime = std::shared_ptr<void>(
          new uint8_t,
          [manager](void* opaque) noexcept {
            delete static_cast<uint8_t*>(opaque);
            manager->releasePersistent();
          }),
      .prewarmLargestSizeClass =
          [manager](memory::MmapAllocator& allocator, uint64_t bytes) {
            return manager->prewarmLargestSizeClass(allocator, bytes);
          }};
}

#ifdef VELOX_ENABLE_S3
NativeS3ReadPolicy chooseNativeS3ReadPolicy(
    const std::vector<folly::Range<char*>>& destinations,
    uint64_t baseRangeBytes,
    uint64_t maxGapBytes,
    uint64_t maxRangeBytes,
    bool adaptive) {
  VELOX_CHECK_GT(baseRangeBytes, 0);
  VELOX_CHECK_GT(maxRangeBytes, 0);
  if (!adaptive) {
    return {.maxGapBytes = maxGapBytes, .maxRangeBytes = maxRangeBytes};
  }

  const auto boundedBaseRangeBytes = std::min(baseRangeBytes, maxRangeBytes);
  uint64_t logicalBytes{0};
  uint64_t pendingGapBytes{0};
  uint64_t internalGapBytes{0};
  uint64_t largestInternalGapBytes{0};
  uint64_t dataRanges{0};
  bool sawData{false};
  for (const auto& destination : destinations) {
    if (destination.empty()) {
      continue;
    }
    if (destination.data() == nullptr) {
      if (sawData) {
        VELOX_CHECK_LE(
            destination.size(),
            std::numeric_limits<uint64_t>::max() - pendingGapBytes,
            "Native S3 adaptive gap byte count overflow");
        pendingGapBytes += destination.size();
      }
      continue;
    }
    VELOX_CHECK_LE(
        destination.size(),
        std::numeric_limits<uint64_t>::max() - logicalBytes,
        "Native S3 adaptive logical byte count overflow");
    logicalBytes += destination.size();
    if (sawData) {
      VELOX_CHECK_LE(
          pendingGapBytes,
          std::numeric_limits<uint64_t>::max() - internalGapBytes,
          "Native S3 adaptive internal gap byte count overflow");
      internalGapBytes += pendingGapBytes;
      largestInternalGapBytes =
          std::max(largestInternalGapBytes, pendingGapBytes);
    }
    pendingGapBytes = 0;
    sawData = true;
    ++dataRanges;
  }

  NativeS3ReadPolicy policy{
      .maxGapBytes = 0, .maxRangeBytes = boundedBaseRangeBytes};
  if (dataRanges < 2 || logicalBytes < boundedBaseRangeBytes) {
    return policy;
  }

  // A dense multi-range read can amortize request latency without materially
  // increasing bytes transferred. Keep at least 32 useful bytes for every
  // byte of admitted logical gap. Leading/trailing holes are never admitted.
  const auto gapBudgetBytes = logicalBytes / 32;
  if (internalGapBytes > gapBudgetBytes ||
      largestInternalGapBytes > maxGapBytes) {
    return policy;
  }
  policy.maxGapBytes = largestInternalGapBytes;
  policy.maxRangeBytes = maxRangeBytes;
  return policy;
}

std::vector<NativeS3ReadGroup> groupNativeS3ReadDestinations(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& destinations,
    uint64_t maxGapBytes,
    uint64_t maxRangeBytes,
    bool sliceOversizedRanges) {
  VELOX_CHECK_GT(maxRangeBytes, 0);
  std::vector<NativeS3ReadGroup> groups;
  uint64_t logicalOffset = offset;
  for (const auto& destination : destinations) {
    VELOX_CHECK_LE(
        destination.size(),
        std::numeric_limits<uint64_t>::max() - logicalOffset,
        "Native S3 scatter offset overflow");
    if (destination.data() == nullptr) {
      logicalOffset += destination.size();
      continue;
    }
    if (!sliceOversizedRanges) {
      bool startGroup = groups.empty();
      uint64_t gapBytes = 0;
      uint64_t coalescedSize = destination.size();
      if (!startGroup) {
        const auto& group = groups.back();
        VELOX_CHECK_LE(
            group.size,
            std::numeric_limits<uint64_t>::max() - group.offset,
            "Native S3 scatter group end overflow");
        const auto groupEnd = group.offset + group.size;
        VELOX_CHECK_LE(groupEnd, logicalOffset);
        gapBytes = logicalOffset - groupEnd;
        VELOX_CHECK_LE(
            destination.size(),
            std::numeric_limits<uint64_t>::max() - logicalOffset,
            "Native S3 coalesced range end overflow");
        const auto destinationEnd = logicalOffset + destination.size();
        coalescedSize = destinationEnd - group.offset;
        startGroup = gapBytes > maxGapBytes || coalescedSize > maxRangeBytes;
      }
      if (startGroup) {
        groups.push_back(
            NativeS3ReadGroup{
                .offset = logicalOffset, .size = 0, .destinations = {}});
        gapBytes = 0;
        coalescedSize = destination.size();
      }
      auto& group = groups.back();
      if (gapBytes != 0) {
        group.destinations.emplace_back(nullptr, gapBytes);
      }
      group.size = coalescedSize;
      group.destinations.push_back(destination);
      logicalOffset += destination.size();
      continue;
    }
    uint64_t consumed = 0;
    while (consumed < destination.size()) {
      bool startGroup = groups.empty();
      uint64_t gapBytes = 0;
      uint64_t available = maxRangeBytes;
      if (!startGroup) {
        const auto& group = groups.back();
        VELOX_CHECK_LE(
            group.size,
            std::numeric_limits<uint64_t>::max() - group.offset,
            "Native S3 scatter group end overflow");
        const auto groupEnd = group.offset + group.size;
        VELOX_CHECK_LE(groupEnd, logicalOffset);
        gapBytes = logicalOffset - groupEnd;
        startGroup =
            gapBytes > maxGapBytes || gapBytes > maxRangeBytes - group.size;
        if (!startGroup) {
          available = maxRangeBytes - group.size - gapBytes;
          startGroup =
              available == 0 || destination.size() - consumed > available;
        }
      }
      if (startGroup) {
        groups.push_back(
            NativeS3ReadGroup{
                .offset = logicalOffset, .size = 0, .destinations = {}});
        gapBytes = 0;
        available = maxRangeBytes;
      }
      auto& group = groups.back();
      if (gapBytes != 0) {
        group.destinations.emplace_back(nullptr, gapBytes);
        group.size += gapBytes;
      }
      const auto chunkSize =
          std::min<uint64_t>(destination.size() - consumed, available);
      VELOX_CHECK_GT(chunkSize, 0);
      group.destinations.emplace_back(destination.data() + consumed, chunkSize);
      group.size += chunkSize;
      consumed += chunkSize;
      logicalOffset += chunkSize;
    }
  }
  return groups;
}

NativeS3ScatterWriteStreamBuf::NativeS3ScatterWriteStreamBuf(
    std::vector<folly::Range<char*>> destinations)
    : destinations_(std::move(destinations)) {}

size_t NativeS3ScatterWriteStreamBuf::writeBytes(
    const char* source,
    size_t count) {
  size_t written = 0;
  while (written < count && destinationIndex_ < destinations_.size()) {
    auto& destination = destinations_[destinationIndex_];
    const auto available = destination.size() - destinationOffset_;
    const auto copySize = std::min(available, count - written);
    if (copySize != 0) {
      if (destination.data() != nullptr) {
        std::memcpy(
            destination.data() + destinationOffset_,
            source + written,
            copySize);
      }
      destinationOffset_ += copySize;
      written += copySize;
      bytesWritten_ += copySize;
    }
    if (destinationOffset_ == destination.size()) {
      ++destinationIndex_;
      destinationOffset_ = 0;
    }
  }
  if (written != count) {
    overflowed_ = true;
  }
  return written;
}

std::streamsize NativeS3ScatterWriteStreamBuf::xsputn(
    const char* source,
    std::streamsize count) {
  if (count <= 0) {
    return 0;
  }
  return static_cast<std::streamsize>(
      writeBytes(source, static_cast<size_t>(count)));
}

NativeS3ScatterWriteStreamBuf::int_type NativeS3ScatterWriteStreamBuf::overflow(
    int_type value) {
  if (traits_type::eq_int_type(value, traits_type::eof())) {
    return traits_type::not_eof(value);
  }
  const auto character = traits_type::to_char_type(value);
  return writeBytes(&character, 1) == 1 ? value : traits_type::eof();
}

bool crtS3RangeReaderAvailable() {
  return glutenCrtS3RangeReaderAvailable();
}

bool nativeS3ScheduledReadEnabled() {
  return envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0 ||
      envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0;
}

std::shared_ptr<facebook::velox::ReadFile> makeNativeScheduledS3ReadFile(
    std::shared_ptr<facebook::velox::ReadFile> readFile,
    const std::string& filePath) {
  return std::make_shared<NativeScheduledS3ReadFile>(
      std::move(readFile), filePath);
}

CrtS3DataSource::CrtS3DataSource(
    std::string filePath,
    std::optional<std::size_t> fileSize)
    : filePath_(std::move(filePath)),
      fileSize_(
          fileSize.has_value()
              ? fileSize.value()
              : static_cast<size_t>(glutenCrtS3ObjectSize(filePath_.c_str()))),
      metadataReadAheadBytes_(
          envBytesOrZero("GLUTEN_CRT_S3_METADATA_READAHEAD_BYTES")) {
  VELOX_CHECK(
      glutenCrtS3RangeReaderAvailable(),
      "AWS CRT S3 range bridge is not available");
}

size_t CrtS3DataSource::clampedReadSize(size_t offset, size_t requestedSize)
    const {
  if (offset >= fileSize_) {
    return 0;
  }
  return std::min(requestedSize, fileSize_ - offset);
}

size_t CrtS3DataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer> CrtS3DataSource::host_read(
    size_t offset,
    size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (!data.empty()) {
    const auto bytes = host_read(offset, data.size(), data.data());
    VELOX_CHECK_EQ(bytes, data.size(), "Short AWS CRT S3 read");
  }
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
CrtS3DataSource::host_read(size_t offset, size_t requestedSize, uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return 0;
  }
  // This datasource is constructed only for the cuDF Parquet reader. Parquet
  // always starts with the four-byte PAR1 magic, and the real footer magic is
  // still fetched and validated from S3. Avoid one latency-dominated 4-byte
  // GET per file when explicitly enabled.
  if (offset == 0 && readSize == 4 &&
      envBytesOrZero("GLUTEN_CRT_S3_SYNTHESIZE_PARQUET_MAGIC") != 0) {
    static constexpr uint8_t kParquetMagic[] = {'P', 'A', 'R', '1'};
    std::memcpy(dst, kParquetMagic, sizeof(kParquetMagic));
    return sizeof(kParquetMagic);
  }
  const auto tailReadSize = std::min(metadataReadAheadBytes_, fileSize_);
  const auto tailOffset = fileSize_ - tailReadSize;
  if (tailReadSize != 0 && readSize <= tailReadSize && offset >= tailOffset) {
    std::call_once(tailCacheOnce_, [this, tailOffset, tailReadSize]() {
      tailCacheOffset_ = tailOffset;
      tailCache_.resize(tailReadSize);
      const auto bytes =
          readRanges({tailOffset}, {tailReadSize}, tailCache_.data(), {0});
      VELOX_CHECK_EQ(
          bytes, tailReadSize, "Short AWS CRT S3 metadata read-ahead");
    });
    VELOX_CHECK_LE(
        readSize,
        tailCache_.size() - (offset - tailCacheOffset_),
        "AWS CRT S3 metadata read exceeds tail cache");
    std::memcpy(dst, tailCache_.data() + offset - tailCacheOffset_, readSize);
    return readSize;
  }
  return readRanges({offset}, {readSize}, dst, {0});
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CrtS3DataSource::host_read_async(size_t offset, size_t requestedSize) {
  return std::async(std::launch::deferred, [this, offset, requestedSize]() {
    return host_read(offset, requestedSize);
  });
}

std::future<size_t> CrtS3DataSource::host_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  return std::async(
      std::launch::deferred, [this, offset, requestedSize, dst]() {
        return host_read(offset, requestedSize, dst);
      });
}

bool CrtS3DataSource::supports_device_read() const {
  return false;
}

size_t CrtS3DataSource::readRanges(
    const std::vector<size_t>& offsets,
    const std::vector<size_t>& sizes,
    uint8_t* destination,
    const std::vector<size_t>& destinationOffsets) const {
  VELOX_CHECK_EQ(offsets.size(), sizes.size());
  VELOX_CHECK_EQ(sizes.size(), destinationOffsets.size());
  std::vector<uint64_t> rangeOffsets(offsets.begin(), offsets.end());
  std::vector<uint64_t> rangeSizes(sizes.begin(), sizes.end());
  std::vector<uint64_t> outputOffsets(
      destinationOffsets.begin(), destinationOffsets.end());
  return static_cast<size_t>(glutenCrtS3ReadRanges(
      filePath_.c_str(),
      destination,
      rangeOffsets.data(),
      rangeSizes.data(),
      outputOffsets.data(),
      rangeOffsets.size()));
}

KvikioS3DataSource::KvikioS3DataSource(
    const std::string& filePath,
    const std::string& accessKeyId,
    const std::string& secretAccessKey,
    const std::string& sessionToken,
    std::optional<std::string> region,
    std::optional<std::string> endpoint,
    std::shared_ptr<ReadFile> nativeS3ReadFile,
    std::optional<std::size_t> fileSize)
    : handle_(makeKvikioS3Handle(
          filePath,
          accessKeyId,
          secretAccessKey,
          sessionToken,
          region,
          endpoint,
          fileSize)),
      nativeS3ReadFile_(std::move(nativeS3ReadFile)),
      metadataReadAheadBytes_(
          envBytesOrZero("GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES")) {
  VELOX_CHECK_NOT_NULL(nativeS3ReadFile_);
  auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
  nativeS3Bucket_ = bucketAndObject.first;
  nativeS3Key_ = bucketAndObject.second;
  nativeS3Url_ = kvikio::S3Endpoint::url_from_bucket_and_object(
      bucketAndObject.first, bucketAndObject.second, region, endpoint);
  std::string resolvedRegion;
  if (region.has_value()) {
    resolvedRegion = region.value();
  } else if (
      const auto* environmentRegion = std::getenv("AWS_DEFAULT_REGION")) {
    resolvedRegion = environmentRegion;
  }
  VELOX_CHECK(
      !resolvedRegion.empty(), "Native S3 curl-multi requires AWS region");
  nativeAwsSigV4_ = "aws:amz:" + resolvedRegion + ":s3";
  nativeUserPassword_ = accessKeyId + ":" + secretAccessKey;
  nativeSessionToken_ = sessionToken;
}

size_t KvikioS3DataSource::clampedReadSize(size_t offset, size_t requestedSize)
    const {
  if (offset >= size()) {
    return 0;
  }
  return std::min(requestedSize, size() - offset);
}

size_t KvikioS3DataSource::size() const {
  return handle_.nbytes();
}

std::unique_ptr<cudf::io::datasource::buffer> KvikioS3DataSource::host_read(
    size_t offset,
    size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (!data.empty()) {
    handle_.pread(data.data(), data.size(), offset).get();
  }
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t KvikioS3DataSource::host_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return 0;
  }
  if (offset == 0 && readSize == 4 &&
      envBytesOrZero("GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC") != 0) {
    static constexpr uint8_t kParquetMagic[] = {'P', 'A', 'R', '1'};
    std::memcpy(dst, kParquetMagic, sizeof(kParquetMagic));
    return sizeof(kParquetMagic);
  }
  const auto fileSize = size();
  const auto tailReadSize = std::min(metadataReadAheadBytes_, fileSize);
  const auto tailOffset = fileSize - tailReadSize;
  if (tailReadSize != 0 && readSize <= tailReadSize && offset >= tailOffset) {
    std::call_once(tailCacheOnce_, [this, tailOffset, tailReadSize]() {
      tailCacheOffset_ = tailOffset;
      tailCache_.resize(tailReadSize);
      const auto bytes =
          readRanges({tailOffset}, {tailReadSize}, tailCache_.data(), {0});
      VELOX_CHECK_EQ(
          bytes, tailReadSize, "Short native S3 metadata read-ahead");
    });
    VELOX_CHECK_LE(
        readSize,
        tailCache_.size() - (offset - tailCacheOffset_),
        "Native S3 metadata read exceeds tail cache");
    std::memcpy(dst, tailCache_.data() + offset - tailCacheOffset_, readSize);
    return readSize;
  }
  return readRanges({offset}, {readSize}, dst, {0});
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
KvikioS3DataSource::host_read_async(size_t offset, size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (data.empty()) {
    return std::async(
        std::launch::deferred, [data = std::move(data)]() mutable {
          return cudf::io::datasource::buffer::create(std::move(data));
        });
  }
  auto readFuture = handle_.pread(data.data(), data.size(), offset);
  return std::async(
      std::launch::deferred,
      [data = std::move(data), readFuture = std::move(readFuture)]() mutable {
        readFuture.get();
        return cudf::io::datasource::buffer::create(std::move(data));
      });
}

std::future<size_t> KvikioS3DataSource::host_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return std::async(std::launch::deferred, [] { return size_t{0}; });
  }
  auto readFuture = handle_.pread(dst, readSize, offset);
  return std::async(
      std::launch::deferred, [readFuture = std::move(readFuture)]() mutable {
        return readFuture.get();
      });
}

size_t KvikioS3DataSource::readRanges(
    const std::vector<size_t>& offsets,
    const std::vector<size_t>& sizes,
    uint8_t* destination,
    const std::vector<size_t>& destinationOffsets) {
  VELOX_CHECK_EQ(offsets.size(), sizes.size());
  VELOX_CHECK_EQ(sizes.size(), destinationOffsets.size());
  std::vector<std::future<size_t>> pending;
  pending.reserve(offsets.size());
  size_t expectedBytes = 0;
  const auto useCrt = envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
  const auto useAwsSdk =
      !useCrt && envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0;
  const auto useCurlMulti =
      !useCrt && !useAwsSdk && envBytesOrZero("GLUTEN_CPP_S3_CURL_MULTI") != 0;
  auto* sdkScheduler =
      (useCrt || useAwsSdk) ? &NativeS3SdkScheduler::instance() : nullptr;
  auto* scheduler =
      useCurlMulti ? &NativeS3MultiScheduler::instance() : nullptr;
  const NativeS3MultiScheduler::Auth auth{
      nativeS3Url_, nativeAwsSigV4_, nativeUserPassword_, nativeSessionToken_};
  for (size_t index = 0; index < offsets.size(); ++index) {
    const auto readSize = clampedReadSize(offsets[index], sizes[index]);
    VELOX_CHECK_EQ(
        readSize, sizes[index], "Native S3 range extends beyond end of object");
    if (readSize == 0) {
      continue;
    }
    if (sdkScheduler != nullptr) {
      size_t scheduled = 0;
      while (scheduled < readSize) {
        const auto chunkSize =
            std::min(sdkScheduler->rangeBytes(), readSize - scheduled);
        if (useCrt) {
          pending.emplace_back(sdkScheduler->submitCrt(
              nativeS3Bucket_,
              nativeS3Key_,
              offsets[index] + scheduled,
              chunkSize,
              destination + destinationOffsets[index] + scheduled));
        } else {
          pending.emplace_back(sdkScheduler->submit(
              nativeS3ReadFile_,
              offsets[index] + scheduled,
              chunkSize,
              destination + destinationOffsets[index] + scheduled));
        }
        scheduled += chunkSize;
      }
    } else if (scheduler == nullptr) {
      pending.emplace_back(handle_.pread(
          destination + destinationOffsets[index], readSize, offsets[index]));
    } else {
      size_t scheduled = 0;
      while (scheduled < readSize) {
        const auto chunkSize =
            std::min(scheduler->rangeBytes(), readSize - scheduled);
        pending.emplace_back(scheduler->submit(
            auth,
            offsets[index] + scheduled,
            chunkSize,
            destination + destinationOffsets[index] + scheduled));
        scheduled += chunkSize;
      }
    }
    expectedBytes += readSize;
  }
  size_t actualBytes = 0;
  for (auto& read : pending) {
    actualBytes += read.get();
  }
  VELOX_CHECK_EQ(actualBytes, expectedBytes, "Short native S3 range batch");
  return actualBytes;
}

bool KvikioS3DataSource::supports_device_read() const {
  return true;
}

bool KvikioS3DataSource::is_device_read_preferred(
    size_t /* requestedSize */) const {
  return true;
}

std::future<size_t> KvikioS3DataSource::device_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return std::async(std::launch::deferred, [] { return size_t{0}; });
  }
  if (envBytesOrZero("GLUTEN_CPP_S3_REGULAR_DEVICE") != 0 &&
      (envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0 ||
       envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0 ||
       envBytesOrZero("GLUTEN_CPP_S3_CURL_MULTI") != 0)) {
    auto hostBuffer = std::make_shared<PinnedHostBuffer>(readSize);
    const auto useCrt = envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
    const auto useAwsSdk =
        !useCrt && envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0;
    auto* sdkScheduler =
        (useCrt || useAwsSdk) ? &NativeS3SdkScheduler::instance() : nullptr;
    auto* scheduler =
        (useCrt || useAwsSdk) ? nullptr : &NativeS3MultiScheduler::instance();
    const NativeS3MultiScheduler::Auth auth{
        nativeS3Url_,
        nativeAwsSigV4_,
        nativeUserPassword_,
        nativeSessionToken_};
    std::vector<std::future<size_t>> reads;
    const auto rangeBytes = (useCrt || useAwsSdk) ? sdkScheduler->rangeBytes()
                                                  : scheduler->rangeBytes();
    reads.reserve((readSize + rangeBytes - 1) / rangeBytes);
    size_t scheduled = 0;
    while (scheduled < readSize) {
      const auto chunkSize = std::min(rangeBytes, readSize - scheduled);
      if (useCrt) {
        reads.emplace_back(sdkScheduler->submitCrt(
            nativeS3Bucket_,
            nativeS3Key_,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      } else if (useAwsSdk) {
        reads.emplace_back(sdkScheduler->submit(
            nativeS3ReadFile_,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      } else {
        reads.emplace_back(scheduler->submit(
            auth,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      }
      scheduled += chunkSize;
    }
    // cuDF invokes device_read_async for the projected ranges before waiting
    // on their futures. Submission above therefore fills the executor-global
    // curl-multi window immediately; deferred completion avoids one host
    // thread per range batch.
    return std::async(
        std::launch::deferred,
        [reads = std::move(reads),
         hostBuffer = std::move(hostBuffer),
         dst,
         stream,
         readSize]() mutable {
          size_t actual = 0;
          for (auto& read : reads) {
            actual += read.get();
          }
          VELOX_CHECK_EQ(actual, readSize, "Short regular native S3 read");
          if (envBytesOrZero("GLUTEN_CPP_S3_REGULAR_STREAM_ORDERED_H2D") != 0) {
            using Completion = std::pair<
                std::shared_ptr<PinnedHostBuffer>,
                std::unique_ptr<NativeS3H2dPermit>>;
            auto completion = std::make_unique<Completion>(
                std::move(hostBuffer),
                std::make_unique<NativeS3H2dPermit>(/*enabled=*/true));
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                dst,
                completion->first->data(),
                readSize,
                cudaMemcpyHostToDevice,
                stream.value()));
            CUDF_CUDA_TRY(cudaLaunchHostFunc(
                stream.value(),
                [](void* opaque) { delete static_cast<Completion*>(opaque); },
                completion.get()));
            completion.release();
            return actual;
          }
          NativeS3H2dPermit h2dPermit(/*enabled=*/true);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              dst,
              hostBuffer->data(),
              readSize,
              cudaMemcpyHostToDevice,
              stream.value()));
          stream.synchronize();
          h2dPermit.release();
          return actual;
        });
  }
  // KvikIO owns the worker streams used by remote device reads. Keep the H2D
  // copy on those internal streams and use pread()'s completion as the device
  // visibility boundary. The consumer stream is not exposed to KvikIO.
  auto readFuture = handle_.pread(dst, readSize, offset);
  return std::async(
      std::launch::deferred, [readFuture = std::move(readFuture)]() mutable {
        return readFuture.get();
      });
}

size_t KvikioS3DataSource::device_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  return device_read_async(offset, requestedSize, dst, stream).get();
}

std::unique_ptr<cudf::io::datasource::buffer> KvikioS3DataSource::device_read(
    size_t offset,
    size_t requestedSize,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer data(clampedReadSize(offset, requestedSize), stream);
  const auto readSize = device_read(
      offset, requestedSize, static_cast<uint8_t*>(data.data()), stream);
  data.resize(readSize, stream);
  return cudf::io::datasource::buffer::create(std::move(data));
}
#endif

void logNativeS3SchedulerStats(std::string_view event, std::string_view id) {
#ifdef VELOX_ENABLE_S3
  if (nativeS3ScheduledReadEnabled()) {
    NativeS3SdkScheduler::instance().logSnapshot(event, id);
  }
#else
  static_cast<void>(event);
  static_cast<void>(id);
#endif
}

void initializeNativeS3Scheduler() {
#ifdef VELOX_ENABLE_S3
  if (!nativeS3ScheduledReadEnabled()) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto initializedAwsSdk = facebook::velox::filesystems::initializeS3();
  static_cast<void>(NativeS3SdkScheduler::instance());
  const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  LOG(WARNING) << "Native S3 scheduler initialized eagerly in " << elapsedMs
               << " ms; crt=" << (envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0)
               << "; initializedAwsSdk=" << initializedAwsSdk;
#endif
}

bool prioritizeNativeS3File(const std::string& filePath) {
#ifdef VELOX_ENABLE_S3
  if (!nativeS3ScheduledReadEnabled() || !filePath.starts_with("s3://")) {
    return false;
  }
  auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
  return NativeS3SdkScheduler::instance().prioritizeFile(
      bucketAndObject.first, bucketAndObject.second);
#else
  static_cast<void>(filePath);
  return false;
#endif
}

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::optional<BufferedInputDeviceCopyHooks> deviceCopyHooks)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      metadataReadAheadBytes_(
          envBytesOrZero("GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES")),
      deviceCopyHooks_(
          deviceCopyHooks.has_value() ? std::move(deviceCopyHooks.value())
                                      : defaultBufferedInputDeviceCopyHooks()) {
  VELOX_CHECK(deviceCopyHooks_.copy, "Device copy hook is not initialized");
  VELOX_CHECK(
      deviceCopyHooks_.retainUntilComplete,
      "Device copy lifetime hook is not initialized");
}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back([dst, size, sharedStream](
                                    rmm::cuda_stream_view stream) {
    uint64_t copied = 0;
    while (copied < size) {
      const void* buffer = nullptr;
      int32_t available = 0;
      VELOX_CHECK(
          sharedStream->Next(&buffer, &available),
          "BufferedInput stream ended after {} of {} bytes",
          copied,
          size);
      VELOX_CHECK_GT(available, 0);
      const auto bytes = std::min<uint64_t>(available, size - copied);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          dst + copied, buffer, bytes, cudaMemcpyHostToDevice, stream.value()));
      copied += bytes;
      if (bytes < static_cast<uint64_t>(available)) {
        sharedStream->BackUp(available - bytes);
      }
    }
  });
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);
  // The cache load above is already complete and pendingDeviceLoads_ belongs
  // exclusively to this data source. Avoid serializing cache-to-device copies
  // from independent scan drivers on the process-global remote-IO mutex.
  for (auto& deviceLoad : pendingDeviceLoads_) {
    deviceLoad(stream);
  }
}

bool BufferedInputDataSource::prefetchToCache(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges) {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  if (cached == nullptr) {
    return false;
  }
  std::vector<facebook::velox::common::Region> regions;
  regions.reserve(byteRanges.size());
  for (const auto& range : byteRanges) {
    regions.push_back(
        {static_cast<uint64_t>(range.offset()),
         static_cast<uint64_t>(range.size())});
  }
  const auto inlineLoad = envFlagEnabled("GLUTEN_CUDF_CACHE_HINT_INLINE_LOAD");
  if (inlineLoad) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
      LOG(WARNING) << "CUDF_CACHE_HINT_INLINE_LOAD enabled=1";
    }
  }
  cached->prefetchSync(regions, inlineLoad);
  return true;
}

std::function<void()> BufferedInputDataSource::prepareCachePrefetch(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges) {
  return prepareCachePrefetch(byteRanges, /*firstLoadReady=*/{});
}

std::function<void()> BufferedInputDataSource::prepareCachePrefetch(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges,
    std::function<void()> firstLoadReady) {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  if (cached == nullptr) {
    return {};
  }
  std::vector<facebook::velox::common::Region> regions;
  regions.reserve(byteRanges.size());
  for (const auto& range : byteRanges) {
    regions.push_back(
        {static_cast<uint64_t>(range.offset()),
         static_cast<uint64_t>(range.size())});
  }
  const auto asyncPhysicalGroups =
      envFlagEnabled("GLUTEN_CPP_S3_ASYNC_CACHE_READ_GROUPS");
  const auto preallocatePins = !asyncPhysicalGroups ||
      envFlagEnabled("GLUTEN_CUDF_CACHE_HINT_PREALLOCATE_PINS");
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true, std::memory_order_relaxed)) {
    LOG(WARNING) << "CUDF_CACHE_HINT_PREPARE_LOADS enabled=1 preallocatePins="
                 << preallocatePins
                 << " asyncPhysicalGroups=" << asyncPhysicalGroups;
  }
  return cached->preparePrefetch(
      regions,
      /*inlineLoad=*/true,
      preallocatePins,
      asyncPhysicalGroups,
      std::move(firstLoadReady));
}

uint64_t BufferedInputDataSource::canonicalCacheBytes(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges) const {
  return canonicalCacheStats(byteRanges).uniqueBytes;
}

CacheHintRangeStats BufferedInputDataSource::canonicalCacheStats(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges) const {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  if (cached == nullptr) {
    return {};
  }
  std::vector<facebook::velox::common::Region> regions;
  regions.reserve(byteRanges.size());
  CacheHintRangeStats stats;
  for (const auto& range : byteRanges) {
    VELOX_CHECK_GE(range.offset(), 0);
    VELOX_CHECK_GE(range.size(), 0);
    if (range.size() == 0) {
      continue;
    }
    regions.push_back(
        {static_cast<uint64_t>(range.offset()),
         static_cast<uint64_t>(range.size())});
    ++stats.logicalRanges;
    stats.logicalBytes += static_cast<uint64_t>(range.size());
  }
  if (regions.empty()) {
    return stats;
  }

  // CacheRegionPlan validates that physical chunks do not overlap. This is a
  // correctness fallback: an unexpected layout disables the best-effort hint
  // instead of creating ambiguous offset-only cache identities.
  const auto plan = cached->makeCacheRegionPlan(regions);
  const auto& unique = plan->cacheRegions();
  stats.uniqueRanges = unique.size();
  for (const auto& region : unique) {
    stats.uniqueBytes += region.length;
  }
  return stats;
}

std::optional<uint64_t> BufferedInputDataSource::cacheFileNum() const {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  return cached == nullptr ? std::nullopt
                           : std::optional<uint64_t>(cached->cacheFileNum());
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readBuffered(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  return readBuffered(offset, readSize, dst);
}

size_t BufferedInputDataSource::readBuffered(
    size_t offset,
    size_t readSize,
    uint8_t* dst) {
  if (readSize == 0) {
    return 0;
  }
  VELOX_CHECK_NOT_NULL(dst);
  if (offset == 0 && readSize == 4 &&
      envBytesOrZero("GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC") != 0) {
    static constexpr uint8_t kParquetMagic[] = {'P', 'A', 'R', '1'};
    std::memcpy(dst, kParquetMagic, sizeof(kParquetMagic));
    return sizeof(kParquetMagic);
  }
  const auto tailReadSize = std::min(metadataReadAheadBytes_, fileSize_);
  const auto tailOffset = fileSize_ - tailReadSize;
  if (tailReadSize != 0 && readSize <= tailReadSize && offset >= tailOffset) {
    std::call_once(tailCacheOnce_, [this, tailOffset, tailReadSize]() {
      tailCacheOffset_ = tailOffset;
      tailCache_.resize(tailReadSize);
      readContiguous(tailOffset, tailReadSize, tailCache_.data());
    });
    VELOX_CHECK_LE(
        readSize,
        tailCache_.size() - (offset - tailCacheOffset_),
        "Buffered S3 metadata read exceeds tail cache");
    std::memcpy(dst, tailCache_.data() + offset - tailCacheOffset_, readSize);
    return readSize;
  }
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  const auto readSize =
      offset >= fileSize_ ? 0 : std::min(size, fileSize_ - offset);
  if (readSize == 0) {
    return std::async(std::launch::deferred, []() { return size_t{0}; });
  }
  if (input_->hasCache()) {
    auto input = input_;
    auto hooks = deviceCopyHooks_;
    auto* executor = input_->executor();
    const bool retainCachePins =
        dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
            input_.get()) != nullptr;
    auto copyFromCache = [input = std::move(input),
                          hooks = std::move(hooks),
                          offset,
                          readSize,
                          dst,
                          stream,
                          retainCachePins]() mutable {
      using facebook::velox::dwio::common::LogType;
      auto ownedStream = input->read(offset, readSize, LogType::FILE);
      VELOX_CHECK_NOT_NULL(ownedStream, "read() returned null stream");
      std::shared_ptr sharedStream(std::move(ownedStream));
      auto lifetime = std::make_shared<DirectCachePageH2dLifetime>();
      lifetime->inputStream = sharedStream;
      auto* cacheStream =
          dynamic_cast<facebook::velox::dwio::common::CacheInputStream*>(
              sharedStream.get());
      VELOX_CHECK(
          !retainCachePins || cacheStream != nullptr,
          "Cached input returned a non-cache stream");
      size_t copied = 0;
      uint64_t copies = 0;
      uint64_t pinnedCopies = 0;
      uint64_t pinnedBytes = 0;
      try {
        while (copied < readSize) {
          const void* source = nullptr;
          int32_t available = 0;
          VELOX_CHECK(
              sharedStream->Next(&source, &available),
              "BufferedInput stream ended after {} of {} bytes",
              copied,
              readSize);
          VELOX_CHECK_GT(available, 0);
          const auto bytes = std::min<size_t>(
              static_cast<size_t>(available), readSize - copied);
          if (cacheStream != nullptr) {
            lifetime->cachePins.push_back(cacheStream->retainCurrentCachePin());
            if (cacheStream->currentCachePageHasBackingRegistration()) {
              ++pinnedCopies;
              pinnedBytes += bytes;
            }
          }
          hooks.copy(dst + copied, source, bytes, stream);
          ++copies;
          copied += bytes;
          if (bytes < static_cast<size_t>(available)) {
            sharedStream->BackUp(
                static_cast<int32_t>(static_cast<size_t>(available) - bytes));
          }
        }
      } catch (...) {
        const auto error = std::current_exception();
        if (copies > 0) {
          hooks.retainUntilComplete(lifetime, stream);
        }
        std::rethrow_exception(error);
      }
      hooks.retainUntilComplete(lifetime, stream);
      directCachePageH2dCopies.fetch_add(copies, std::memory_order_relaxed);
      directCachePageH2dBytes.fetch_add(copied, std::memory_order_relaxed);
      directCachePageH2dPinnedCopies.fetch_add(
          pinnedCopies, std::memory_order_relaxed);
      directCachePageH2dPinnedBytes.fetch_add(
          pinnedBytes, std::memory_order_relaxed);
      return copied;
    };
    if (envFlagEnabled("GLUTEN_CUDF_CACHE_H2D_INLINE")) {
      // cuDF may invoke this callback under a process-wide submission mutex.
      // Run the copy when cuDF waits on the future, after leaving that mutex.
      return std::async(std::launch::deferred, std::move(copyFromCache));
    }
    auto future = folly::via(executor).thenValue(
        [copyFromCache = std::move(copyFromCache)](auto&&) mutable {
          return copyFromCache();
        });
    return toStdFuture(std::move(future));
  }
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, readSize, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, readSize);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyHostToDevice,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::shared_ptr<PreparedHostByteRanges> prepareByteRangesToHost(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges) {
#ifdef VELOX_ENABLE_S3
  using PackedHostRead = std::function<size_t(
      const std::vector<size_t>&,
      const std::vector<size_t>&,
      uint8_t*,
      const std::vector<size_t>&)>;
  PackedHostRead packedHostRead;
  const char* pageableHostBufferEnv = nullptr;
  bool nativeCppHostBatch = false;
  if (auto* crtSource = dynamic_cast<CrtS3DataSource*>(dataSource.get())) {
    packedHostRead = [crtSource](
                         const std::vector<size_t>& offsets,
                         const std::vector<size_t>& sizes,
                         uint8_t* destination,
                         const std::vector<size_t>& destinationOffsets) {
      return crtSource->readRanges(
          offsets, sizes, destination, destinationOffsets);
    };
    pageableHostBufferEnv = "GLUTEN_CRT_S3_PAGEABLE_HOST_BUFFER";
  } else if (envBytesOrZero("GLUTEN_CPP_S3_HOST_BATCH") != 0) {
    if (auto* nativeSource =
            dynamic_cast<KvikioS3DataSource*>(dataSource.get())) {
      packedHostRead = [nativeSource](
                           const std::vector<size_t>& offsets,
                           const std::vector<size_t>& sizes,
                           uint8_t* destination,
                           const std::vector<size_t>& destinationOffsets) {
        return nativeSource->readRanges(
            offsets, sizes, destination, destinationOffsets);
      };
      pageableHostBufferEnv = "GLUTEN_CPP_S3_PAGEABLE_HOST_BUFFER";
      nativeCppHostBatch = true;
    }
  }
  if (!packedHostRead) {
    return nullptr;
  }

  auto prepared = std::make_shared<PreparedHostByteRanges>();
  prepared->hostSourceOffsets.resize(byteRanges.size());
  prepared->coalesceGapBytes =
      envBytesOrZero("GLUTEN_CRT_S3_RANGE_COALESCE_GAP_BYTES");
  prepared->nativeCppHostBatch = nativeCppHostBatch;
  prepared->totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        return acc + static_cast<size_t>(byteRange.size());
      });

  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<size_t> destinationOffsets;
  size_t hostBufferSize = 0;
  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    const auto destinationOffset = hostBufferSize;
    prepared->hostSourceOffsets[chunk] = destinationOffset;
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      const size_t nextSize = byteRanges[nextChunk].size();
      VELOX_CHECK_LE(
          ioSize,
          std::numeric_limits<size_t>::max() - ioOffset,
          "S3 coalesced range end overflow");
      const auto ioEnd = ioOffset + ioSize;
      if (nextOffset < ioEnd ||
          nextOffset - ioEnd > prepared->coalesceGapBytes) {
        break;
      }
      VELOX_CHECK_LE(
          nextSize,
          std::numeric_limits<size_t>::max() - nextOffset,
          "S3 next range end overflow");
      const auto nextEnd = nextOffset + nextSize;
      ioSize = nextEnd - ioOffset;
      prepared->hostSourceOffsets[nextChunk] =
          destinationOffset + nextOffset - ioOffset;
      ++nextChunk;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinationOffsets.push_back(destinationOffset);
      VELOX_CHECK_LE(
          ioSize,
          std::numeric_limits<size_t>::max() - hostBufferSize,
          "S3 host buffer size overflow");
      hostBufferSize += ioSize;
    }
    chunk = nextChunk;
  }

  VELOX_CHECK_NOT_NULL(pageableHostBufferEnv);
  prepared->hostBuffer = std::make_shared<PinnedHostBuffer>(
      hostBufferSize, nullptr, envBytesOrZero(pageableHostBufferEnv) == 0);
  const auto bytes = packedHostRead(
      ioOffsets, ioSizes, prepared->hostBuffer->data(), destinationOffsets);
  const auto expected =
      std::accumulate(ioSizes.begin(), ioSizes.end(), size_t{0});
  VELOX_CHECK_EQ(bytes, expected, "Short packed S3 range batch");
  return prepared;
#else
  return nullptr;
#endif
}

FetchedDeviceByteRanges copyPreparedByteRangesToDevice(
    std::shared_ptr<PreparedHostByteRanges> prepared,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto totalStart = std::chrono::steady_clock::now();
  constexpr auto kBufferPaddingMultiple = 8;
  VELOX_CHECK_NOT_NULL(prepared);
  VELOX_CHECK_NOT_NULL(prepared->hostBuffer);
  VELOX_CHECK_EQ(prepared->hostSourceOffsets.size(), byteRanges.size());

  NativeS3H2dPermit h2dPermit(prepared->nativeCppHostBatch);
  const auto allocationBytes = cudf::util::round_up_safe<size_t>(
      prepared->totalSize, kBufferPaddingMultiple);

  std::vector<rmm::device_buffer> columnChunkBuffers;
  columnChunkBuffers.emplace_back(allocationBytes, stream, mr);
  const auto allocationDone = std::chrono::steady_clock::now();
  auto* bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::vector<cudf::device_span<const uint8_t>> columnChunkData;
  columnChunkData.reserve(byteRanges.size());
  size_t deviceOffset = 0;
  for (size_t chunk = 0; chunk < byteRanges.size(); ++chunk) {
    const auto chunkSize = static_cast<size_t>(byteRanges[chunk].size());
    columnChunkData.emplace_back(bufferData + deviceOffset, chunkSize);
    if (prepared->coalesceGapBytes != 0 && chunkSize != 0) {
      std::memmove(
          prepared->hostBuffer->data() + deviceOffset,
          prepared->hostBuffer->data() + prepared->hostSourceOffsets[chunk],
          chunkSize);
    }
    deviceOffset += chunkSize;
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      bufferData,
      prepared->hostBuffer->data(),
      prepared->totalSize,
      cudaMemcpyHostToDevice,
      stream.value()));
  stream.synchronize();
  const auto copyDone = std::chrono::steady_clock::now();
  // Return pooled pinned memory before publishing the device spans. Timing
  // this separately catches CUDA-wide synchronization in host unregister.
  prepared.reset();
  const auto recycleDone = std::chrono::steady_clock::now();
  h2dPermit.release();
  static std::atomic<uint64_t> completed{0};
  static std::atomic<uint64_t> bytes{0};
  static std::atomic<uint64_t> allocationNanos{0};
  static std::atomic<uint64_t> copyNanos{0};
  static std::atomic<uint64_t> recycleNanos{0};
  static std::atomic<uint64_t> totalNanos{0};
  const auto count = completed.fetch_add(1, std::memory_order_relaxed) + 1;
  bytes.fetch_add(allocationBytes, std::memory_order_relaxed);
  allocationNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          allocationDone - totalStart)
          .count(),
      std::memory_order_relaxed);
  copyNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          copyDone - allocationDone)
          .count(),
      std::memory_order_relaxed);
  recycleNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          recycleDone - copyDone)
          .count(),
      std::memory_order_relaxed);
  totalNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          recycleDone - totalStart)
          .count(),
      std::memory_order_relaxed);
  if (envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0 && count % 64 == 0) {
    LOG(WARNING) << "CPP_S3_H2D completed=" << count
                 << " bytes=" << bytes.load(std::memory_order_relaxed)
                 << " avgAllocationMs="
                 << allocationNanos.load(std::memory_order_relaxed) / count /
            1e6 << " avgCopySyncMs="
                 << copyNanos.load(std::memory_order_relaxed) / count / 1e6
                 << " avgRecycleMs="
                 << recycleNanos.load(std::memory_order_relaxed) / count / 1e6
                 << " avgTotalMs="
                 << totalNanos.load(std::memory_order_relaxed) / count / 1e6;
  }
  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(std::launch::deferred, [] {})};
}

size_t hybridH2dStagingBytes() {
  return envBytesOrZero("GLUTEN_CUDF_HYBRID_H2D_STAGING_BYTES");
}

PackedDeviceByteRanges fetchPackedByteRangesToDevice(
    cudf::host_span<const std::reference_wrapper<cudf::io::datasource>> sources,
    cudf::host_span<const cudf::io::text::byte_range_info> ranges,
    cudf::host_span<const cudf::size_type> sourceIndices,
    size_t stagingBytes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  constexpr size_t kBufferPaddingMultiple = 8;
  VELOX_CHECK_GT(stagingBytes, 0);
  VELOX_CHECK_EQ(ranges.size(), sourceIndices.size());
  size_t totalSize = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    VELOX_CHECK_GE(sourceIndices[i], 0);
    VELOX_CHECK_LT(static_cast<size_t>(sourceIndices[i]), sources.size());
    const auto& range = ranges[i];
    VELOX_CHECK_GE(range.offset(), 0);
    VELOX_CHECK_GE(range.size(), 0);
    const auto offset = static_cast<size_t>(range.offset());
    const auto size = static_cast<size_t>(range.size());
    const auto fileSize = sources[sourceIndices[i]].get().size();
    VELOX_CHECK_LE(offset, fileSize);
    VELOX_CHECK_LE(size, fileSize - offset);
    VELOX_CHECK_LE(size, std::numeric_limits<size_t>::max() - totalSize);
    totalSize += size;
  }

  // Declare the device owner before staging: exception unwinding must drain
  // outstanding copies before freeing either the host or device allocation.
  PackedDeviceByteRanges result{
      rmm::device_buffer(
          cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
          stream,
          mr),
      {},
      {}};
  auto* device = static_cast<uint8_t*>(result.buffer.data());
  result.spans.reserve(ranges.size());
  size_t packedOffset = 0;
  for (const auto& range : ranges) {
    result.spans.emplace_back(
        device == nullptr ? nullptr : device + packedOffset, range.size());
    packedOffset += range.size();
  }
  if (totalSize == 0) {
    return result;
  }

  const auto capacity = std::min(stagingBytes, totalSize);
  BufferedH2dStaging staging(capacity, stream, &result.stats);
  uint8_t* host = nullptr;
  size_t staged = 0;
  size_t deviceOffset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto offset = static_cast<size_t>(ranges[i].offset());
    auto remaining = static_cast<size_t>(ranges[i].size());
    while (remaining != 0) {
      if (staged == 0) {
        host = staging.acquire();
      }
      const auto bytes = std::min(remaining, capacity - staged);
      const auto start = std::chrono::steady_clock::now();
      VELOX_CHECK_EQ(
          sources[sourceIndices[i]].get().host_read(
              offset, bytes, host + staged),
          bytes,
          "Short prepared packed host read");
      result.stats.gatherNanos +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      offset += bytes;
      remaining -= bytes;
      staged += bytes;
      if (staged == capacity) {
        staging.submit(device + deviceOffset, staged);
        deviceOffset += staged;
        staged = 0;
      }
    }
  }
  if (staged != 0) {
    staging.submit(device + deviceOffset, staged);
  }
  staging.finish();
  return result;
}

FetchedDeviceByteRanges fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        VELOX_CHECK_LE(
            byteRange.size(),
            std::numeric_limits<size_t>::max() - acc,
            "Projected byte range total overflow");
        return acc + byteRange.size();
      });

  if (auto prepared = prepareByteRangesToHost(dataSource, byteRanges)) {
    return copyPreparedByteRangesToDevice(
        std::move(prepared), byteRanges, stream, mr);
  }

#ifdef VELOX_ENABLE_S3
  using PackedHostRead = std::function<size_t(
      const std::vector<size_t>&,
      const std::vector<size_t>&,
      uint8_t*,
      const std::vector<size_t>&)>;
  PackedHostRead packedHostRead;
  const char* pageableHostBufferEnv = nullptr;
  bool nativeCppHostBatch = false;
  if (auto* crtSource = dynamic_cast<CrtS3DataSource*>(dataSource.get())) {
    packedHostRead = [crtSource](
                         const std::vector<size_t>& offsets,
                         const std::vector<size_t>& sizes,
                         uint8_t* destination,
                         const std::vector<size_t>& destinationOffsets) {
      return crtSource->readRanges(
          offsets, sizes, destination, destinationOffsets);
    };
    pageableHostBufferEnv = "GLUTEN_CRT_S3_PAGEABLE_HOST_BUFFER";
  } else if (envBytesOrZero("GLUTEN_CPP_S3_HOST_BATCH") != 0) {
    if (auto* nativeSource =
            dynamic_cast<KvikioS3DataSource*>(dataSource.get())) {
      packedHostRead = [nativeSource](
                           const std::vector<size_t>& offsets,
                           const std::vector<size_t>& sizes,
                           uint8_t* destination,
                           const std::vector<size_t>& destinationOffsets) {
        return nativeSource->readRanges(
            offsets, sizes, destination, destinationOffsets);
      };
      pageableHostBufferEnv = "GLUTEN_CPP_S3_PAGEABLE_HOST_BUFFER";
      nativeCppHostBatch = true;
    }
  }

  // Remote range sources read directly into their final packed host offsets.
  // Do not reserve the matching device buffer until S3 has completed: at high
  // scan parallelism, holding a device allocation per in-flight remote file
  // exhausts RMM before any file can reach decode. One H2D copy publishes the
  // complete projected batch after all native range futures finish.
  if (packedHostRead) {
    std::vector<size_t> ioOffsets;
    std::vector<size_t> ioSizes;
    std::vector<size_t> destinationOffsets;
    std::vector<size_t> hostSourceOffsets(byteRanges.size());
    const auto coalesceGapBytes =
        envBytesOrZero("GLUTEN_CRT_S3_RANGE_COALESCE_GAP_BYTES");
    size_t hostBufferSize = 0;
    for (size_t chunk = 0; chunk < byteRanges.size();) {
      const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
      auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
      const auto destinationOffset = hostBufferSize;
      hostSourceOffsets[chunk] = destinationOffset;
      size_t nextChunk = chunk + 1;
      while (nextChunk < byteRanges.size()) {
        const size_t nextOffset = byteRanges[nextChunk].offset();
        const size_t nextSize = byteRanges[nextChunk].size();
        VELOX_CHECK_LE(
            ioSize,
            std::numeric_limits<size_t>::max() - ioOffset,
            "CRT S3 coalesced range end overflow");
        const auto ioEnd = ioOffset + ioSize;
        if (nextOffset < ioEnd || nextOffset - ioEnd > coalesceGapBytes) {
          break;
        }
        VELOX_CHECK_LE(
            nextSize,
            std::numeric_limits<size_t>::max() - nextOffset,
            "CRT S3 next range end overflow");
        const auto nextEnd = nextOffset + nextSize;
        ioSize = nextEnd - ioOffset;
        hostSourceOffsets[nextChunk] =
            destinationOffset + nextOffset - ioOffset;
        ++nextChunk;
      }
      if (ioSize != 0) {
        ioOffsets.push_back(ioOffset);
        ioSizes.push_back(ioSize);
        destinationOffsets.push_back(destinationOffset);
        VELOX_CHECK_LE(
            ioSize,
            std::numeric_limits<size_t>::max() - hostBufferSize,
            "CRT S3 host buffer size overflow");
        hostBufferSize += ioSize;
      }
      chunk = nextChunk;
    }

    VELOX_CHECK_NOT_NULL(pageableHostBufferEnv);
    const auto preferPinned = envBytesOrZero(pageableHostBufferEnv) == 0;
    auto hostBuffer = std::make_shared<PinnedHostBuffer>(
        hostBufferSize, nullptr, preferPinned);
    const auto bytes = packedHostRead(
        ioOffsets, ioSizes, hostBuffer->data(), destinationOffsets);
    const auto expected =
        std::accumulate(ioSizes.begin(), ioSizes.end(), size_t{0});
    VELOX_CHECK_EQ(bytes, expected, "Short AWS CRT S3 range batch");

    // Keep remote host reads independent from GPU backpressure. A large Velox
    // IO pool can therefore keep curl-multi full, while only a small number of
    // completed host batches allocate device memory and copy concurrently.
    // The permit ends immediately after H2D synchronization; it is not held
    // through parquet decode or the lifetime of the returned device buffer.
    NativeS3H2dPermit h2dPermit(nativeCppHostBatch);
    const auto allocationBytes =
        cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple);

    std::vector<rmm::device_buffer> columnChunkBuffers;
    columnChunkBuffers.emplace_back(allocationBytes, stream, mr);
    auto* bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
    std::vector<cudf::device_span<const uint8_t>> columnChunkData;
    columnChunkData.reserve(byteRanges.size());
    size_t deviceOffset = 0;
    for (size_t chunk = 0; chunk < byteRanges.size(); ++chunk) {
      const auto chunkSize = static_cast<size_t>(byteRanges[chunk].size());
      columnChunkData.emplace_back(bufferData + deviceOffset, chunkSize);
      if (coalesceGapBytes != 0 && chunkSize != 0) {
        // Coalesced S3 responses include the gaps between projected column
        // chunks. Compact the selected chunks in place so the host layout
        // matches the packed device layout. Source offsets are never before
        // their packed destinations, so forward memmove is safe and avoids
        // allocating a second host buffer.
        std::memmove(
            hostBuffer->data() + deviceOffset,
            hostBuffer->data() + hostSourceOffsets[chunk],
            chunkSize);
      }
      deviceOffset += chunkSize;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        bufferData,
        hostBuffer->data(),
        totalSize,
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();
    h2dPermit.release();
    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, [] {})};
  }
#endif

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData == nullptr ? nullptr : bufferData + acc,
            static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    const auto stagingBytes = hybridH2dStagingBytes();
    if (stagingBytes != 0) {
      if (totalSize == 0) {
        return {
            std::move(columnChunkBuffers),
            std::move(columnChunkData),
            std::async(std::launch::deferred, [] {})};
      }
      // Preserve batch cache loading before gathering individual ranges;
      // otherwise a cold cache would turn packing into serial remote reads.
      bufferedInput->prefetchToCache({byteRanges.begin(), byteRanges.end()});
      const auto capacity = std::min(stagingBytes, totalSize);
      auto staging = std::make_unique<BufferedH2dStaging>(capacity, stream);
      uint8_t* host = nullptr;
      size_t staged = 0;
      size_t deviceOffset = 0;
      for (const auto& range : byteRanges) {
        auto offset = static_cast<size_t>(range.offset());
        auto remaining = static_cast<size_t>(range.size());
        while (remaining != 0) {
          if (staged == 0) {
            host = staging->acquire();
          }
          const auto bytes = std::min(remaining, capacity - staged);
          VELOX_CHECK_EQ(
              bufferedInput->host_read(offset, bytes, host + staged),
              bytes,
              "Short hybrid buffered-input read");
          offset += bytes;
          remaining -= bytes;
          staged += bytes;
          if (staged == capacity) {
            staging->submit(bufferData + deviceOffset, staged);
            deviceOffset += staged;
            staged = 0;
          }
        }
      }
      if (staged != 0) {
        staging->submit(bufferData + deviceOffset, staged);
      }
      // The caller can do independent host work before get(). The existing
      // hybrid reader gets this future before decode. Destruction without
      // get(), including an I/O exception above, also drains pending copies.
      auto completion = std::async(
          std::launch::deferred, [staging = std::move(staging)]() mutable {
            staging->finish();
            staging.reset();
          });
      return {
          std::move(columnChunkBuffers),
          std::move(columnChunkData),
          std::move(completion)};
    }
    auto iter =
        cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
    std::for_each(
        iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
          const auto& byteRange = cuda::std::get<0>(tuple);
          const auto& destination = cuda::std::get<1>(tuple);
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              const_cast<uint8_t*>(destination.data()));
        });

    // load buffered input data source
    auto syncFunction = [](std::shared_ptr<cudf::io::datasource> dataSource,
                           rmm::cuda_stream_view stream) {
      auto buffer =
          checkedPointerCast<BufferedInputDataSource>(dataSource.get());
      buffer->load(stream);
    };

    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, syncFunction, dataSource, stream)};
  }

  // KvikIO dataSource: Impl borrowed from `fetch_byte_ranges_to_device_async()`
  // in `parquet_io_utils.cpp` in cuDF.
  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<uint8_t*> destinations;

  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      if (nextOffset != ioOffset + ioSize) {
        break;
      }
      ioSize += byteRanges[nextChunk].size();
      nextChunk++;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinations.push_back(
          const_cast<uint8_t*>(columnChunkData[chunk].data()));
    }
    chunk = nextChunk;
  }
  VELOX_CHECK_EQ(
      ioOffsets.size(),
      ioSizes.size(),
      "Number of IO offsets and sizes must be equal");
  VELOX_CHECK_EQ(
      ioSizes.size(),
      destinations.size(),
      "Number of IO sizes and destinations must be equal");

  auto iter = cuda::make_zip_iterator(
      ioOffsets.begin(), ioSizes.begin(), destinations.begin());

  std::vector<std::future<size_t>> deviceReadTasks;
  std::vector<std::future<size_t>> hostReadTasks;
  deviceReadTasks.reserve(ioOffsets.size());
  hostReadTasks.reserve(ioOffsets.size());

  // KvikIO's remote read uses internal worker streams. Synchronize preceding
  // consumer work before scheduling the batch; completion of each pread()
  // includes its H2D copy.
  stream.synchronize();

  {
    std::lock_guard<std::mutex> lock(ioBatchMutex());

    std::for_each(iter, iter + ioOffsets.size(), [&](const auto& tuple) {
      const auto ioOffset = cuda::std::get<0>(tuple);
      const auto ioSize = cuda::std::get<1>(tuple);
      const auto dest = cuda::std::get<2>(tuple);

      if (dataSource->supports_device_read() and
          dataSource->is_device_read_preferred(ioSize)) {
        deviceReadTasks.emplace_back(
            dataSource->device_read_async(ioOffset, ioSize, dest, stream));
      } else {
        // TODO(mh): We can't yet guarantee (without a safe thread pool) that
        // all `cudaMemcpyAsync`s will be launched by the time we release the
        // mutex. That said, this is a rare usecase as host-buffer data should
        // prefer using a `BufferedInputDataSource` datasource.
        hostReadTasks.emplace_back(
            std::async(
                std::launch::async,
                [dataSource, ioOffset, ioSize, dest, stream]() {
                  auto hostBuffer = dataSource->host_read(ioOffset, ioSize);
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                      dest,
                      hostBuffer->data(),
                      hostBuffer->size(),
                      cudaMemcpyHostToDevice,
                      stream.value()));
                  return ioSize;
                }));
      }
    });
  }

  auto syncFunction = [](decltype(hostReadTasks)&& hostReadTasks,
                         decltype(deviceReadTasks)&& deviceReadTasks) {
    for (auto& task : hostReadTasks) {
      task.get();
    }
    for (auto& task : deviceReadTasks) {
      task.get();
    }
  };

  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(
          std::launch::deferred,
          std::move(syncFunction),
          std::move(hostReadTasks),
          std::move(deviceReadTasks))};
}

} // namespace facebook::velox::cudf_velox::connector::hive
