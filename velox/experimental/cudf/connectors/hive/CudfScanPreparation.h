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

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <folly/CancellationToken.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Atomic admission for host-ready inputs. A reservation owns both its byte
/// charge and unit slot until the reservation is destroyed.
class HostRangeAdmission {
 private:
  struct State;

 public:
  struct Snapshot {
    uint64_t currentBytes{0};
    uint64_t peakBytes{0};
    uint32_t currentUnits{0};
    uint32_t peakUnits{0};
    uint32_t waiters{0};
  };

  class Reservation {
   public:
    Reservation() = default;
    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;
    ~Reservation();

    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    uint64_t bytes() const {
      return bytes_;
    }

    uint32_t units() const {
      return units_;
    }

    void reset();

   private:
    friend class HostRangeAdmission;

    Reservation(std::shared_ptr<State> state, uint64_t bytes, uint32_t units);

    std::shared_ptr<State> state_;
    uint64_t bytes_{0};
    uint32_t units_{0};
  };

  HostRangeAdmission(uint64_t maxBytes, uint32_t maxUnits);

  HostRangeAdmission(const HostRangeAdmission&) = delete;
  HostRangeAdmission& operator=(const HostRangeAdmission&) = delete;

  Reservation reserve(
      uint64_t bytes,
      uint32_t units,
      const folly::CancellationToken& cancellationToken = {});

  /// Returns whether a reservation can fit when the admission window is
  /// otherwise empty. Callers use this to choose fallback before payload I/O.
  bool canEverReserve(uint64_t bytes, uint32_t units) const;

  Snapshot snapshot() const;

 private:
  std::shared_ptr<State> state_;
};

struct HostRangeFragment {
  uint64_t fileOffset{0};
  const uint8_t* data{nullptr};
  size_t size{0};
};

struct LeasedHostRange {
  cudf::io::text::byte_range_info requested;
  std::vector<HostRangeFragment> fragments;
};

/// Owns every cache pin backing a set of exact logical file ranges. Fragment
/// pointers remain valid until this object is destroyed.
class HostRangeLease {
 public:
  HostRangeLease(HostRangeLease&&) noexcept;
  HostRangeLease& operator=(HostRangeLease&&) noexcept;
  ~HostRangeLease();

  HostRangeLease(const HostRangeLease&) = delete;
  HostRangeLease& operator=(const HostRangeLease&) = delete;

  const std::vector<LeasedHostRange>& ranges() const;

  uint64_t logicalBytes() const;

  uint64_t retainedBytes() const;

  uint64_t reservedBytes() const;

  void copyRange(size_t rangeIndex, uint8_t* destination, size_t size) const;

  /// Copies an arbitrary covered file range. Throws if any requested byte is
  /// not owned by this lease.
  void copyFileRange(uint64_t offset, uint8_t* destination, size_t size) const;

 private:
  friend class BufferedInputDataSource;

  HostRangeLease();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// A cuDF datasource that can read only bytes retained by a HostRangeLease.
/// It never falls back to the underlying file or cache input.
class HostRangeLeaseDataSource final : public cudf::io::datasource {
 public:
  HostRangeLeaseDataSource(
      std::shared_ptr<const HostRangeLease> lease,
      size_t fileSize);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* destination) override;

  uint64_t hostReadCalls() const;

  uint64_t hostReadBytes() const;

 private:
  const std::shared_ptr<const HostRangeLease> lease_;
  const size_t fileSize_;
  std::atomic<uint64_t> hostReadCalls_{0};
  std::atomic<uint64_t> hostReadBytes_{0};
};

struct PreparedParquetRowGroupPlan {
  cudf::size_type rowGroup{0};
  uint64_t inputRows{0};
  uint64_t logicalBytes{0};
  std::vector<cudf::io::text::byte_range_info> ranges;
};

/// Footer metadata and exact projected ranges partitioned by independently
/// schedulable row group. No decode reader survives this planning call.
struct PreparedParquetFilePlan {
  cudf::io::parquet::FileMetaData metadata;
  uint64_t totalRowGroups{0};
  std::vector<PreparedParquetRowGroupPlan> rowGroups;
};

struct PreparedHostReadyDataSource {
  std::shared_ptr<cudf::io::datasource> dataSource;
  std::shared_ptr<const cudf::io::parquet::FileMetaData> metadata;
};

PreparedParquetFilePlan planParquetFileRanges(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    const cudf::io::parquet_reader_options& options,
    rmm::cuda_stream_view stream,
    const folly::CancellationToken& cancellationToken = {});

} // namespace facebook::velox::cudf_velox::connector::hive
