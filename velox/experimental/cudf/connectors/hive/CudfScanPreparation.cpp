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
#include "velox/experimental/cudf/connectors/hive/CudfScanPreparation.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Allocation.h"
#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/common/CachedBufferedInput.h"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <folly/OperationCancelled.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

void throwIfCancelled(const folly::CancellationToken& token) {
  if (token.isCancellationRequested()) {
    throw folly::OperationCancelled{};
  }
}

uint64_t checkedAdd(uint64_t total, uint64_t value, const char* label) {
  VELOX_CHECK_LE(
      value,
      std::numeric_limits<uint64_t>::max() - total,
      "{} overflow",
      label);
  return total + value;
}

uint64_t cacheEntryReservationBytes(
    const dwio::common::CachedBufferedInput& input,
    uint64_t logicalBytes) {
  auto* cache = input.cache();
  VELOX_CHECK_NOT_NULL(cache);
  auto* allocator = cache->allocator();
  VELOX_CHECK_NOT_NULL(allocator);
  const auto requestedPages = memory::AllocationTraits::numPages(logicalBytes);
  const auto& sizeClasses = allocator->sizeClasses();
  VELOX_CHECK(!sizeClasses.empty());
  const auto largest = sizeClasses.back();
  auto reservedPages = (requestedPages / largest) * largest;
  const auto remainder = requestedPages % largest;
  if (remainder > 0) {
    const auto it =
        std::lower_bound(sizeClasses.begin(), sizeClasses.end(), remainder);
    VELOX_CHECK(it != sizeClasses.end());
    reservedPages += *it;
  }
  auto bytes = memory::AllocationTraits::pageBytes(reservedPages);
  // dataCapacity() also includes the empty std::string's inline capacity for
  // allocation-backed cache entries.
  return checkedAdd(
      bytes, std::string{}.capacity(), "Host range cache entry bytes");
}

uint64_t cachePlanReservationBytes(
    const dwio::common::CachedBufferedInput& input,
    const dwio::common::CacheRegionPlan& plan,
    uint64_t fileSize) {
  if (input.preloaded()) {
    return cacheEntryReservationBytes(input, fileSize);
  }
  uint64_t reservationBytes = 0;
  for (const auto& region : plan.cacheRegions()) {
    VELOX_CHECK_LE(region.offset, fileSize);
    const auto maximumEntryBytes =
        std::min(input.cacheLoadQuantum(), fileSize - region.offset);
    reservationBytes = checkedAdd(
        reservationBytes,
        cacheEntryReservationBytes(input, maximumEntryBytes),
        "Host range reservation bytes");
  }
  return reservationBytes;
}

} // namespace

struct HostRangeAdmission::State {
  State(uint64_t maxBytes, uint32_t maxUnits)
      : maxBytes(maxBytes), maxUnits(maxUnits) {}

  std::mutex mutex;
  std::condition_variable condition;
  const uint64_t maxBytes;
  const uint32_t maxUnits;
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint32_t currentUnits{0};
  uint32_t peakUnits{0};
  uint32_t waiters{0};
};

HostRangeAdmission::Reservation::Reservation(
    std::shared_ptr<State> state,
    uint64_t bytes,
    uint32_t units)
    : state_(std::move(state)), bytes_(bytes), units_(units) {}

HostRangeAdmission::Reservation::Reservation(Reservation&& other) noexcept
    : state_(std::move(other.state_)),
      bytes_(std::exchange(other.bytes_, 0)),
      units_(std::exchange(other.units_, 0)) {}

HostRangeAdmission::Reservation& HostRangeAdmission::Reservation::operator=(
    Reservation&& other) noexcept {
  if (this != &other) {
    reset();
    state_ = std::move(other.state_);
    bytes_ = std::exchange(other.bytes_, 0);
    units_ = std::exchange(other.units_, 0);
  }
  return *this;
}

HostRangeAdmission::Reservation::~Reservation() {
  reset();
}

void HostRangeAdmission::Reservation::reset() {
  if (!state_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    VELOX_CHECK_GE(state_->currentBytes, bytes_);
    VELOX_CHECK_GE(state_->currentUnits, units_);
    state_->currentBytes -= bytes_;
    state_->currentUnits -= units_;
  }
  state_->condition.notify_all();
  state_.reset();
  bytes_ = 0;
  units_ = 0;
}

HostRangeAdmission::HostRangeAdmission(uint64_t maxBytes, uint32_t maxUnits)
    : state_(std::make_shared<State>(maxBytes, maxUnits)) {
  VELOX_CHECK_GT(maxBytes, 0, "Host range byte limit must be positive");
  VELOX_CHECK_GT(maxUnits, 0, "Host range unit limit must be positive");
}

HostRangeAdmission::Reservation HostRangeAdmission::reserve(
    uint64_t bytes,
    uint32_t units,
    const folly::CancellationToken& cancellationToken) {
  VELOX_CHECK_GT(bytes, 0, "Host range reservation must contain bytes");
  VELOX_CHECK_GT(units, 0, "Host range reservation must contain units");
  VELOX_CHECK_LE(bytes, state_->maxBytes, "Host range exceeds byte limit");
  VELOX_CHECK_LE(units, state_->maxUnits, "Host range exceeds unit limit");
  throwIfCancelled(cancellationToken);

  std::weak_ptr<State> weakState = state_;
  std::unique_ptr<folly::CancellationCallback> cancellationCallback;
  if (cancellationToken.canBeCancelled()) {
    cancellationCallback = std::make_unique<folly::CancellationCallback>(
        cancellationToken, [weakState]() noexcept {
          if (auto state = weakState.lock()) {
            state->condition.notify_all();
          }
        });
  }

  std::unique_lock<std::mutex> lock(state_->mutex);
  const auto fits = [&]() {
    return bytes <= state_->maxBytes - state_->currentBytes &&
        units <= state_->maxUnits - state_->currentUnits;
  };
  if (!fits()) {
    ++state_->waiters;
    state_->condition.wait(lock, [&]() {
      return fits() || cancellationToken.isCancellationRequested();
    });
    VELOX_CHECK_GT(state_->waiters, 0);
    --state_->waiters;
  }
  if (cancellationToken.isCancellationRequested()) {
    throw folly::OperationCancelled{};
  }
  state_->currentBytes += bytes;
  state_->currentUnits += units;
  state_->peakBytes = std::max(state_->peakBytes, state_->currentBytes);
  state_->peakUnits = std::max(state_->peakUnits, state_->currentUnits);
  return Reservation{state_, bytes, units};
}

bool HostRangeAdmission::canEverReserve(uint64_t bytes, uint32_t units) const {
  return bytes > 0 && units > 0 && bytes <= state_->maxBytes &&
      units <= state_->maxUnits;
}

HostRangeAdmission::Snapshot HostRangeAdmission::snapshot() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return {
      state_->currentBytes,
      state_->peakBytes,
      state_->currentUnits,
      state_->peakUnits,
      state_->waiters};
}

struct HostRangeLease::Impl {
  // Keep the admission charge until every pin has been released.
  HostRangeAdmission::Reservation reservation;
  std::vector<cache::CachePin> pins;
  std::vector<LeasedHostRange> ranges;
  uint64_t logicalBytes{0};
  uint64_t retainedBytes{0};
};

HostRangeLease::HostRangeLease() : impl_(std::make_unique<Impl>()) {}

HostRangeLease::HostRangeLease(HostRangeLease&&) noexcept = default;

HostRangeLease& HostRangeLease::operator=(HostRangeLease&&) noexcept = default;

HostRangeLease::~HostRangeLease() = default;

const std::vector<LeasedHostRange>& HostRangeLease::ranges() const {
  return impl_->ranges;
}

uint64_t HostRangeLease::logicalBytes() const {
  return impl_->logicalBytes;
}

uint64_t HostRangeLease::retainedBytes() const {
  return impl_->retainedBytes;
}

uint64_t HostRangeLease::reservedBytes() const {
  return impl_->reservation.bytes();
}

void HostRangeLease::copyRange(
    size_t rangeIndex,
    uint8_t* destination,
    size_t size) const {
  VELOX_CHECK_LT(rangeIndex, impl_->ranges.size());
  VELOX_CHECK_NOT_NULL(destination);
  const auto& range = impl_->ranges[rangeIndex];
  VELOX_CHECK_EQ(size, static_cast<size_t>(range.requested.size()));
  size_t copied = 0;
  for (const auto& fragment : range.fragments) {
    VELOX_CHECK_NOT_NULL(fragment.data);
    VELOX_CHECK_LE(fragment.size, size - copied);
    std::memcpy(destination + copied, fragment.data, fragment.size);
    copied += fragment.size;
  }
  VELOX_CHECK_EQ(copied, size, "Leased range has incomplete coverage");
}

void HostRangeLease::copyFileRange(
    uint64_t offset,
    uint8_t* destination,
    size_t size) const {
  if (size == 0) {
    return;
  }
  VELOX_CHECK_NOT_NULL(destination);
  VELOX_CHECK_LE(
      static_cast<uint64_t>(size),
      std::numeric_limits<uint64_t>::max() - offset,
      "Host lease range overflow");
  const auto end = offset + size;
  auto cursor = offset;
  while (cursor < end) {
    const HostRangeFragment* covering = nullptr;
    for (const auto& range : impl_->ranges) {
      for (const auto& fragment : range.fragments) {
        const auto fragmentEnd = checkedAdd(
            fragment.fileOffset, fragment.size, "Host lease fragment end");
        if (fragment.fileOffset <= cursor && cursor < fragmentEnd) {
          covering = &fragment;
          break;
        }
      }
      if (covering != nullptr) {
        break;
      }
    }
    VELOX_CHECK_NOT_NULL(
        covering, "Host lease does not cover file offset {}", cursor);
    const auto fragmentOffset = cursor - covering->fileOffset;
    const auto available = covering->size - fragmentOffset;
    const auto bytes = std::min<uint64_t>(available, end - cursor);
    std::memcpy(
        destination + (cursor - offset),
        covering->data + fragmentOffset,
        bytes);
    cursor += bytes;
  }
}

HostRangeLeaseDataSource::HostRangeLeaseDataSource(
    std::shared_ptr<const HostRangeLease> lease,
    size_t fileSize)
    : lease_(std::move(lease)), fileSize_(fileSize) {
  VELOX_CHECK_NOT_NULL(lease_);
  VELOX_CHECK_GT(fileSize_, 0);
}

size_t HostRangeLeaseDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
HostRangeLeaseDataSource::host_read(size_t offset, size_t requestedSize) {
  VELOX_CHECK_LE(offset, fileSize_, "Prepared read starts past EOF");
  const auto size = std::min(requestedSize, fileSize_ - offset);
  std::vector<uint8_t> data(size);
  if (size > 0) {
    lease_->copyFileRange(offset, data.data(), size);
  }
  hostReadCalls_.fetch_add(1, std::memory_order_relaxed);
  hostReadBytes_.fetch_add(size, std::memory_order_relaxed);
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t HostRangeLeaseDataSource::host_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* destination) {
  VELOX_CHECK_LE(offset, fileSize_, "Prepared read starts past EOF");
  const auto size = std::min(requestedSize, fileSize_ - offset);
  if (size > 0) {
    VELOX_CHECK_NOT_NULL(destination);
    lease_->copyFileRange(offset, destination, size);
  }
  hostReadCalls_.fetch_add(1, std::memory_order_relaxed);
  hostReadBytes_.fetch_add(size, std::memory_order_relaxed);
  return size;
}

uint64_t HostRangeLeaseDataSource::hostReadCalls() const {
  return hostReadCalls_.load(std::memory_order_relaxed);
}

uint64_t HostRangeLeaseDataSource::hostReadBytes() const {
  return hostReadBytes_.load(std::memory_order_relaxed);
}

std::optional<uint64_t> BufferedInputDataSource::hostRangeReservationBytes(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges) const {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  if (cached == nullptr) {
    return std::nullopt;
  }
  VELOX_CHECK(!byteRanges.empty(), "Host range reservation is empty");
  std::vector<facebook::velox::common::Region> regions;
  regions.reserve(byteRanges.size());
  for (const auto& range : byteRanges) {
    VELOX_CHECK_GE(range.offset(), 0, "Negative host range offset");
    VELOX_CHECK_GT(range.size(), 0, "Empty host range is not supported");
    const auto offset = static_cast<uint64_t>(range.offset());
    const auto size = static_cast<uint64_t>(range.size());
    VELOX_CHECK_LE(offset, fileSize_, "Host range starts past EOF");
    VELOX_CHECK_LE(size, fileSize_ - offset, "Host range ends past EOF");
    regions.emplace_back(offset, size);
  }
  auto cachePlan = cached->makeCacheRegionPlan(regions);
  return cachePlanReservationBytes(*cached, *cachePlan, fileSize_);
}

std::unique_ptr<HostRangeLease> BufferedInputDataSource::prepareHostRangeLease(
    const std::vector<cudf::io::text::byte_range_info>& byteRanges,
    HostRangeAdmission& admission,
    const folly::CancellationToken& cancellationToken) {
  auto* cached =
      dynamic_cast<facebook::velox::dwio::common::CachedBufferedInput*>(
          input_.get());
  VELOX_CHECK_NOT_NULL(
      cached, "Strong host readiness requires a cache-backed input");
  VELOX_CHECK(!byteRanges.empty(), "Host range lease is empty");

  std::vector<facebook::velox::common::Region> regions;
  regions.reserve(byteRanges.size());
  uint64_t logicalBytes = 0;
  for (const auto& range : byteRanges) {
    VELOX_CHECK_GE(range.offset(), 0, "Negative host range offset");
    VELOX_CHECK_GT(range.size(), 0, "Empty host range is not supported");
    const auto offset = static_cast<uint64_t>(range.offset());
    const auto size = static_cast<uint64_t>(range.size());
    VELOX_CHECK_LE(offset, fileSize_, "Host range starts past EOF");
    VELOX_CHECK_LE(size, fileSize_ - offset, "Host range ends past EOF");
    logicalBytes = checkedAdd(logicalBytes, size, "Host range logical bytes");
    regions.emplace_back(offset, size);
  }

  throwIfCancelled(cancellationToken);
  auto cachePlan = cached->makeCacheRegionPlan(regions);
  const auto reservationBytes =
      cachePlanReservationBytes(*cached, *cachePlan, fileSize_);
  if (!admission.canEverReserve(reservationBytes, 1)) {
    return nullptr;
  }
  auto reservation = admission.reserve(reservationBytes, 1, cancellationToken);

  auto lease = std::unique_ptr<HostRangeLease>(new HostRangeLease());
  lease->impl_->reservation = std::move(reservation);
  lease->impl_->logicalBytes = logicalBytes;
  lease->impl_->ranges.reserve(byteRanges.size());
  std::unordered_set<cache::AsyncDataCacheEntry*> retainedEntries;

  for (const auto& requested : byteRanges) {
    throwIfCancelled(cancellationToken);
    const auto offset = static_cast<uint64_t>(requested.offset());
    const auto size = static_cast<uint64_t>(requested.size());
    auto stream = cached->readWithCacheRegionPlan(
        offset, size, dwio::common::LogType::FILE, cachePlan);
    auto* cacheStream =
        dynamic_cast<dwio::common::CacheInputStream*>(stream.get());
    VELOX_CHECK_NOT_NULL(
        cacheStream, "Cached input returned a non-cache stream");

    LeasedHostRange leasedRange;
    leasedRange.requested = requested;
    uint64_t copied = 0;
    while (copied < size) {
      throwIfCancelled(cancellationToken);
      const void* data = nullptr;
      int32_t available = 0;
      VELOX_CHECK(
          stream->Next(&data, &available),
          "Host range ended after {} of {} bytes at file offset {}",
          copied,
          size,
          offset);
      throwIfCancelled(cancellationToken);
      VELOX_CHECK_GT(available, 0, "Host range returned an empty fragment");
      const auto bytes = std::min<uint64_t>(available, size - copied);
      auto pin = cacheStream->retainCurrentCachePin();
      auto* entry = pin.checkedEntry();
      if (retainedEntries.insert(entry).second) {
        const auto capacity = static_cast<uint64_t>(entry->dataCapacity());
        lease->impl_->retainedBytes = checkedAdd(
            lease->impl_->retainedBytes, capacity, "Host range retained bytes");
        VELOX_CHECK_LE(
            lease->impl_->retainedBytes,
            lease->impl_->reservation.bytes(),
            "Cache allocation exceeded the conservative host reservation");
        lease->impl_->pins.push_back(std::move(pin));
      }
      leasedRange.fragments.push_back(
          {offset + copied, static_cast<const uint8_t*>(data), bytes});
      copied += bytes;
      if (bytes < static_cast<uint64_t>(available)) {
        stream->BackUp(static_cast<int32_t>(available - bytes));
      }
    }
    VELOX_CHECK_EQ(copied, size, "Host range has incomplete coverage");
    lease->impl_->ranges.push_back(std::move(leasedRange));
  }
  throwIfCancelled(cancellationToken);
  return lease;
}

PreparedParquetFilePlan planParquetFileRanges(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    const cudf::io::parquet_reader_options& options,
    rmm::cuda_stream_view stream,
    const folly::CancellationToken& cancellationToken) {
  VELOX_CHECK_NOT_NULL(dataSource);
  throwIfCancelled(cancellationToken);

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource.get()));
  auto metadatas = cudf::io::read_parquet_footers(sources);
  throwIfCancelled(cancellationToken);
  VELOX_CHECK_EQ(
      metadatas.size(), 1, "Expected one Parquet footer during preparation");

  PreparedParquetFilePlan result;
  result.metadata = std::move(metadatas.front());
  result.totalRowGroups = result.metadata.row_groups.size();
  cudf::io::parquet::experimental::hybrid_scan_reader planner{
      result.metadata, options};
  auto selected = planner.all_row_groups(options);
  if (options.get_skip_bytes() > 0 || options.get_num_bytes().has_value()) {
    selected = planner.filter_row_groups_with_byte_range(selected, options);
  }
  if (options.get_filter().has_value()) {
    selected = planner.filter_row_groups_with_stats(selected, options, stream);
  }

  result.rowGroups.reserve(selected.size());
  for (const auto rowGroup : selected) {
    throwIfCancelled(cancellationToken);
    std::vector<cudf::size_type> oneRowGroup{rowGroup};
    PreparedParquetRowGroupPlan plan;
    plan.rowGroup = rowGroup;
    plan.inputRows = planner.total_rows_in_row_groups(oneRowGroup);
    plan.ranges = planner.all_column_chunks_byte_ranges(oneRowGroup, options);
    VELOX_CHECK(
        !plan.ranges.empty(),
        "Projected row group {} produced no byte ranges",
        rowGroup);
    for (const auto& range : plan.ranges) {
      VELOX_CHECK_GE(range.offset(), 0, "Negative planned range offset");
      VELOX_CHECK_GT(range.size(), 0, "Empty planned range");
      const auto offset = static_cast<uint64_t>(range.offset());
      const auto size = static_cast<uint64_t>(range.size());
      VELOX_CHECK_LE(
          offset, dataSource->size(), "Planned range starts past EOF");
      VELOX_CHECK_LE(
          size, dataSource->size() - offset, "Planned range ends past EOF");
      plan.logicalBytes =
          checkedAdd(plan.logicalBytes, size, "Planned logical bytes");
    }
    result.rowGroups.push_back(std::move(plan));
  }
  throwIfCancelled(cancellationToken);
  return result;
}

} // namespace facebook::velox::cudf_velox::connector::hive
