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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace facebook::velox::cudf_velox {

// Spark-compatible bloom filter binary format (big-endian on GPU):
//   bytes 0-3:  version   (int32, big-endian, must be 1)
//   bytes 4-7:  numHashes (int32, big-endian)
//   bytes 8-11: numLongs  (int32, big-endian)
//   bytes 12+:  bit array (numLongs * 8 bytes)
//
// Algorithm and format adapted from spark-rapids-jni (Apache 2.0 License).
// Uses Spark's MurmurHash3 two-hash scheme for bit indexing.

struct SparkBloomFilterHeader {
  int32_t version;
  int32_t numHashes;
  int32_t numLongs;
};
constexpr int kSparkBloomFilterHeaderSize = sizeof(SparkBloomFilterHeader);
constexpr int kSparkBloomFilterVersion = 1;

// Format detection: Spark format starts with big-endian int32 version=1
// (byte 0 = 0x00), Velox format starts with int8 version=1 (byte 0 = 0x01).
constexpr uint8_t kVeloxBloomFormatMarker = 0x01;
constexpr uint8_t kSparkBloomFormatMarker = 0x00;

int computeNumHashes(int64_t expectedNumItems, int64_t numBits);

// Create an empty bloom filter on GPU. Returns a device buffer containing
// the packed header + zero-initialized bit array.
std::unique_ptr<rmm::device_buffer> bloomFilterCreate(
    int numHashes,
    int numLongs,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr =
        rmm::mr::get_current_device_resource_ref());

// Insert int64 values into a bloom filter (in-place).
void bloomFilterPut(
    rmm::device_buffer& bloomFilter,
    cudf::column_view const& input,
    rmm::cuda_stream_view stream = cudf::get_default_stream());

// Probe a bloom filter with int64 values. Returns a BOOL8 column.
std::unique_ptr<cudf::column> bloomFilterProbe(
    cudf::column_view const& input,
    cudf::device_span<uint8_t const> bloomFilter,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr =
        rmm::mr::get_current_device_resource_ref());

// Merge multiple bloom filters (stored as a strings column where each row
// is a serialized bloom filter). Returns a device buffer with the merged
// bloom filter.
std::unique_ptr<rmm::device_buffer> bloomFilterMerge(
    cudf::column_view const& bloomFiltersColumn,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr =
        rmm::mr::get_current_device_resource_ref());

// Create a single-row strings column from a device bloom filter buffer.
std::unique_ptr<cudf::column> bloomFilterToStringsColumn(
    rmm::device_buffer const& bloomFilter,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr =
        rmm::mr::get_current_device_resource_ref());

} // namespace facebook::velox::cudf_velox
