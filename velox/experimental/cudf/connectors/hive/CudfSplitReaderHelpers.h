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

#include "velox/experimental/cudf/connectors/hive/CudfScanPreparation.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"
#include "velox/experimental/cudf/connectors/hive/PinnedHostBuffer.h"

#include "velox/common/file/File.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#ifdef VELOX_ENABLE_S3
#include <kvikio/remote_handle.hpp>
#endif

#include <algorithm>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <streambuf>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

struct DirectCachePageH2dStats {
  uint64_t copies{0};
  uint64_t bytes{0};
  uint64_t pinnedCopies{0};
  uint64_t pinnedBytes{0};
};

DirectCachePageH2dStats directCachePageH2dStats();

struct CachePageRegistrationStats {
  uint64_t attempts{0};
  uint64_t successes{0};
  uint64_t failures{0};
  uint64_t budgetRejectedBytes{0};
  uint64_t registeredRuns{0};
  uint64_t registeredBytes{0};
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t unregisteredRuns{0};
  uint64_t unregisteredBytes{0};
  uint64_t prewarmAttempts{0};
  uint64_t prewarmSuccesses{0};
  uint64_t prewarmFailures{0};
  uint64_t prewarmRuns{0};
  uint64_t prewarmBytes{0};
  uint64_t prewarmCoveredRuns{0};
  uint64_t prewarmCoveredBytes{0};
  uint64_t registrationWallNanos{0};
  uint64_t prewarmWallNanos{0};
};

CachePageRegistrationStats cachePageRegistrationStats();

struct CachePageHostRegistrationHooks {
  std::function<bool(void*, size_t)> registerRun;
  std::function<void(void*)> unregisterRun;
};

struct BoundedCachePageRegistration {
  std::function<std::shared_ptr<void>(const memory::Allocation&)>
      registerBackingRuns;
  // Registers one page-aligned contiguous cache allocation. 'bytes' is the
  // physical allocation size rather than the logical cache-entry length.
  std::function<std::shared_ptr<void>(void*, uint64_t)> registerBackingRange;
  // Permanently registers size-class backing on first use. The registration
  // follows the physical allocator page instead of a cache-entry lifetime, so
  // eviction and reuse do not churn cudaHostRegister/cudaHostUnregister.
  std::function<std::shared_ptr<void>(memory::MmapAllocator&, void*, uint64_t)>
      registerPersistentBackingRange;
  // Releases persistent registrations and mapped-page protections. Destroy
  // this after AsyncDataCache shutdown and before its MmapAllocator.
  std::shared_ptr<void> persistentLifetime;
  // Pre-registers largest-size-class backing and returns its RAII lifetime.
  // The lifetime must be released before the allocator is destroyed.
  std::function<std::shared_ptr<void>(memory::MmapAllocator&, uint64_t)>
      prewarmLargestSizeClass;
};

BoundedCachePageRegistration makeBoundedCachePageRegistration(
    uint64_t maxRegisteredBytes,
    std::optional<CachePageHostRegistrationHooks> hooks = std::nullopt);

struct BufferedInputDeviceCopyHooks {
  std::function<void(uint8_t*, const void*, size_t, rmm::cuda_stream_view)>
      copy;
  std::function<void(std::shared_ptr<void>, rmm::cuda_stream_view)>
      retainUntilComplete;
};

constexpr std::size_t kDefaultMultiFileChunkReadLimit = 256UL << 20;

inline std::size_t multiFileChunkReadLimit(
    std::size_t configuredLimit,
    std::size_t batchTarget) {
  if (configuredLimit > 0) {
    return configuredLimit;
  }
  return std::max(batchTarget, kDefaultMultiFileChunkReadLimit);
}

// ---------------- Internal helper ----------------
// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  explicit BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
      std::optional<BufferedInputDeviceCopyHooks> deviceCopyHooks =
          std::nullopt);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  // Use the enqueue API from dwio::common::BufferedInput.
  // Pass a device buffer to copy to after load.
  void enqueueForDevice(uint64_t offset, uint64_t size, uint8_t* dst);

  // loads and copies to device.
  void load(rmm::cuda_stream_view stream);

  /// Populates AsyncDataCache for projected ranges without allocating device
  /// memory. Returns false when this data source is not cache-backed.
  bool prefetchToCache(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges);

  /// Prepares cache/coalesced-load state without issuing physical reads and
  /// returns the load phase. An empty function means the input is not
  /// cache-backed.
  std::function<void()> prepareCachePrefetch(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges);

  std::function<void()> prepareCachePrefetch(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges,
      std::function<void()> firstLoadReady);

  uint64_t canonicalCacheBytes(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges) const;

  CacheHintRangeStats canonicalCacheStats(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges) const;

  std::optional<uint64_t> cacheFileNum() const;

  /// Returns the conservative cache allocation charge for these ranges
  /// without reading their payload. Returns nullopt for a non-cache input.
  std::optional<uint64_t> hostRangeReservationBytes(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges) const;

  /// Reads every requested cache range and transfers ownership of the backing
  /// cache entries into one admitted lease. No device work is started.
  std::unique_ptr<HostRangeLease> prepareHostRangeLease(
      const std::vector<cudf::io::text::byte_range_info>& byteRanges,
      HostRangeAdmission& admission,
      const folly::CancellationToken& cancellationToken = {});

 private:
  size_t readBuffered(size_t offset, size_t size, uint8_t* dst);

  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  const size_t metadataReadAheadBytes_;
  const BufferedInputDeviceCopyHooks deviceCopyHooks_;
  std::once_flag tailCacheOnce_;
  size_t tailCacheOffset_{0};
  std::vector<uint8_t> tailCache_;
  std::vector<std::function<void(rmm::cuda_stream_view stream)>>
      pendingDeviceLoads_;
};

/// KvikIO remote S3 source with credentials resolved by Velox's AWS provider
/// chain. KvikIO remote handles do not consume Velox credential providers.
#ifdef VELOX_ENABLE_S3
class KvikioS3DataSource final : public cudf::io::datasource {
 public:
  KvikioS3DataSource(
      const std::string& filePath,
      const std::string& accessKeyId,
      const std::string& secretAccessKey,
      const std::string& sessionToken,
      std::optional<std::string> region,
      std::optional<std::string> endpoint,
      std::shared_ptr<facebook::velox::ReadFile> nativeS3ReadFile,
      std::optional<std::size_t> fileSize = std::nullopt);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst)
      override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  size_t device_read(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  std::unique_ptr<datasource::buffer> device_read(
      size_t offset,
      size_t size,
      rmm::cuda_stream_view stream) override;

  size_t readRanges(
      const std::vector<size_t>& offsets,
      const std::vector<size_t>& sizes,
      uint8_t* destination,
      const std::vector<size_t>& destinationOffsets);

 private:
  size_t clampedReadSize(size_t offset, size_t requestedSize) const;

  kvikio::RemoteHandle handle_;
  std::string nativeS3Bucket_;
  std::string nativeS3Key_;
  std::string nativeS3Url_;
  std::string nativeAwsSigV4_;
  std::string nativeUserPassword_;
  std::string nativeSessionToken_;
  std::shared_ptr<facebook::velox::ReadFile> nativeS3ReadFile_;
  const size_t metadataReadAheadBytes_;
  std::once_flag tailCacheOnce_;
  size_t tailCacheOffset_{0};
  std::vector<uint8_t> tailCache_;
};

/// S3 source backed by the executor-local AWS SDK v2 CRT bridge. Reads land
/// in host memory; the hybrid reader batches all requested Parquet ranges
/// before issuing one host-to-device copy.
class CrtS3DataSource final : public cudf::io::datasource {
 public:
  CrtS3DataSource(
      std::string filePath,
      std::optional<std::size_t> fileSize = std::nullopt);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst)
      override;

  [[nodiscard]] bool supports_device_read() const override;

  size_t readRanges(
      const std::vector<size_t>& offsets,
      const std::vector<size_t>& sizes,
      uint8_t* destination,
      const std::vector<size_t>& destinationOffsets) const;

 private:
  size_t clampedReadSize(size_t offset, size_t requestedSize) const;

  std::string filePath_;
  size_t fileSize_;
  const size_t metadataReadAheadBytes_;
  std::once_flag tailCacheOnce_;
  size_t tailCacheOffset_{0};
  std::vector<uint8_t> tailCache_;
};

bool crtS3RangeReaderAvailable();

/// Returns true when the executor-global native AWS SDK scheduler is selected
/// for S3 reads.
bool nativeS3ScheduledReadEnabled();

struct NativeS3ReadGroup {
  uint64_t offset{0};
  uint64_t size{0};
  std::vector<folly::Range<char*>> destinations;
};

struct NativeS3ReadPolicy {
  uint64_t maxGapBytes{0};
  uint64_t maxRangeBytes{0};
};

/// Chooses a bounded physical request shape from the current cache-read
/// destinations. In adaptive mode, a single/sparse range keeps the base
/// request bound, while dense multi-range reads may use the larger configured
/// bound. Logical gaps are admitted only when every gap is below the
/// configured ceiling and their aggregate overhead is at most 1/32 of the
/// useful bytes. This keeps the policy query-agnostic and prevents request
/// reduction from turning into unbounded read amplification.
NativeS3ReadPolicy chooseNativeS3ReadPolicy(
    const std::vector<folly::Range<char*>>& destinations,
    uint64_t baseRangeBytes,
    uint64_t maxGapBytes,
    uint64_t maxRangeBytes,
    bool adaptive);

/// Groups caller-owned destinations into bounded file ranges. Null
/// destinations represent logical gaps. Gaps up to 'maxGapBytes' are retained
/// as discard spans when the resulting request does not exceed
/// 'maxRangeBytes'. When 'sliceOversizedRanges' is true, non-null destinations
/// larger than the remaining range capacity are sliced, so 'maxRangeBytes' is
/// a hard physical GET bound. When false, an individual destination may exceed
/// that bound, preserving the original range-plan request shape. Zero gap
/// bytes preserves exact contiguous-only grouping.
std::vector<NativeS3ReadGroup> groupNativeS3ReadDestinations(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& destinations,
    uint64_t maxGapBytes = 0,
    uint64_t maxRangeBytes = std::numeric_limits<uint64_t>::max(),
    bool sliceOversizedRanges = true);

class NativeS3ScatterWriteStreamBuf : public std::streambuf {
 public:
  explicit NativeS3ScatterWriteStreamBuf(
      std::vector<folly::Range<char*>> destinations);

  uint64_t bytesWritten() const {
    return bytesWritten_;
  }

  bool overflowed() const {
    return overflowed_;
  }

 protected:
  std::streamsize xsputn(const char* source, std::streamsize count) override;
  int_type overflow(int_type value) override;

 private:
  size_t writeBytes(const char* source, size_t count);

  std::vector<folly::Range<char*>> destinations_;
  size_t destinationIndex_{0};
  size_t destinationOffset_{0};
  uint64_t bytesWritten_{0};
  bool overflowed_{false};
};

/// AWS response stream that scatters a single Range GET body directly across
/// caller-owned cache page destinations. Null destinations discard coalesced
/// gap bytes without allocating a bounce buffer.
class NativeS3ScatterWriteStream : private NativeS3ScatterWriteStreamBuf,
                                   public std::iostream {
 public:
  explicit NativeS3ScatterWriteStream(
      std::vector<folly::Range<char*>> destinations)
      : NativeS3ScatterWriteStreamBuf(std::move(destinations)),
        std::iostream(static_cast<NativeS3ScatterWriteStreamBuf*>(this)) {}

  uint64_t bytesWritten() const {
    return NativeS3ScatterWriteStreamBuf::bytesWritten();
  }

  bool overflowed() const {
    return NativeS3ScatterWriteStreamBuf::overflowed();
  }
};

/// Wraps an S3 ReadFile so caller-owned destinations, including
/// AsyncDataCache pages, are filled directly by the executor-global native
/// AWS SDK scheduler. The wrapper retains 'readFile' and preserves its file
/// identity and metadata.
std::shared_ptr<facebook::velox::ReadFile> makeNativeScheduledS3ReadFile(
    std::shared_ptr<facebook::velox::ReadFile> readFile,
    const std::string& filePath);
#endif

/// Starts construction of the optional high-throughput CRT client. This is
/// intended to run asynchronously once query-level request pressure is known,
/// before cache-load producers begin issuing their bounded range requests.
/// This is a no-op when S3 support is disabled.
void prepareNativeS3HighThroughputClient();

/// Runs a cache-load producer with a request-pressure hint scoped to the
/// current thread. Native S3 requests submitted by the producer may use the
/// optional high-throughput CRT client; unrelated and concurrent producers
/// retain the base client. The hint does not alter cache readiness semantics.
/// When S3 support is disabled the producer still runs.
void runWithNativeS3RequestPressure(
    bool highThroughput,
    const std::function<void()>& producer);

/// Exposed for focused tests of thread-scoped pressure propagation.
bool nativeS3HighRequestPressureForCurrentThread();

/// Emits an executor-global scheduler snapshot when native S3 diagnostics are
/// enabled. Counters are cumulative, so the first query-exit snapshot is the
/// exact cold-query physical request total. This is a no-op when S3 support
/// or the native scheduler is disabled.
void logNativeS3SchedulerStats(
    std::string_view event,
    std::string_view id = {});

/// Constructs the executor-global native AWS SDK/CRT scheduler and client.
/// Call during executor initialization to keep one-time client setup out of
/// the first measured scan. This performs no object reads. It is a no-op when
/// S3 support or the native scheduler is disabled.
void initializeNativeS3Scheduler();

/// Raises queued native S3 ranges for 'filePath' ahead of speculative data
/// ranges without shrinking the executor-global request window. Returns true
/// when at least one queued range was marked. This is a no-op when demand
/// priority or the native CRT scheduler is disabled, or for non-S3 paths.
bool prioritizeNativeS3File(const std::string& filePath);

/// Projected remote ranges fetched into one final-layout host buffer. This is
/// deliberately separate from the device allocation so split preload threads
/// can keep S3 busy without speculatively consuming GPU memory or streams.
struct PreparedHostByteRanges {
  std::shared_ptr<PinnedHostBuffer> hostBuffer;
  std::vector<size_t> hostSourceOffsets;
  size_t totalSize{0};
  size_t coalesceGapBytes{0};
  bool nativeCppHostBatch{false};
};

/**
 * @brief Hybrid scan reader state
 *
 * This struct is used to store the column chunk data for the hybrid scan reader
 * and a once flag to ensure the setup is only done once.
 */
struct HybridScanState {
  HybridScanState() : isHybridScanSetup_(std::make_unique<std::once_flag>()) {}

  std::vector<rmm::device_buffer> columnChunkBuffers_;
  std::vector<cudf::device_span<const uint8_t>> columnChunkData_;
  std::shared_ptr<PreparedHostByteRanges> preparedHostByteRanges_;
  std::optional<std::vector<cudf::size_type>> rowGroupIndices_;
  std::optional<std::vector<cudf::io::text::byte_range_info>>
      columnChunkByteRanges_;
  std::unique_ptr<std::once_flag> isHybridScanSetup_;
};

using FetchedDeviceByteRanges = std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>;

/// Fetch supported packed remote ranges into host memory only. Returns null
/// for data sources that use the existing direct/device paths.
std::shared_ptr<PreparedHostByteRanges> prepareByteRangesToHost(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges);

/// Allocate and copy a previously prepared host batch to device memory.
FetchedDeviceByteRanges copyPreparedByteRangesToDevice(
    std::shared_ptr<PreparedHostByteRanges> prepared,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/**
 * @brief Fetches a list of byte ranges from a host buffer into device buffers
 *
 * @param dataSource Input datasource
 * @param byteRanges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return A tuple containing the device buffers, the device spans of the
 * fetched data, and a future to wait on the read tasks. Keep the device buffers
 * alive until completion. With GLUTEN_CUDF_HYBRID_H2D_STAGING_BYTES enabled,
 * buffered inputs pack into two host slots; events protect slot reuse and the
 * future drains the remaining H2D copies before decode.
 */
FetchedDeviceByteRanges fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::connector::hive
