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

// GPU bloom filter using Spark-compatible format.
// Hash algorithm and bit layout adapted from NVIDIA spark-rapids-jni
// (Apache 2.0 License): src/main/cpp/src/bloom_filter.cu
// MurmurHash3 adapted from: src/main/cpp/src/hash/murmur_hash.cuh

#include "BloomFilterKernels.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

#include <byteswap.h>
#include <cmath>

namespace facebook::velox::cudf_velox {

namespace {

// ---------------------------------------------------------------------------
// MurmurHash3_32 for int64 (Spark-compatible)
// ---------------------------------------------------------------------------

__device__ inline uint32_t rotateLeft(uint32_t x, uint32_t r) {
  return __funnelshift_l(x, x, r);
}

constexpr uint32_t kC1 = 0xcc9e2d51;
constexpr uint32_t kC2 = 0x1b873593;
constexpr uint32_t kC3 = 0xe6546b64;

__device__ inline int32_t murmurHash3_32(int64_t key, uint32_t seed) {
  auto data = reinterpret_cast<const uint8_t*>(&key);
  uint32_t h = seed;

  // Process two 4-byte blocks
  for (int i = 0; i < 2; i++) {
    uint32_t k1 = data[i * 4] | (data[i * 4 + 1] << 8) |
        (data[i * 4 + 2] << 16) | (data[i * 4 + 3] << 24);
    k1 *= kC1;
    k1 = rotateLeft(k1, 15);
    k1 *= kC2;
    h ^= k1;
    h = rotateLeft(h, 13);
    h = h * 5 + kC3;
  }

  // Finalize
  h ^= 8; // len = sizeof(int64_t) = 8
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return static_cast<int32_t>(h);
}

// ---------------------------------------------------------------------------
// Bloom filter bit indexing (Spark big-endian layout)
// ---------------------------------------------------------------------------

using bloom_hash_t = int32_t;

__device__ inline auto getHashMask(
    bloom_hash_t h,
    int32_t bloomFilterBits) {
  // https://github.com/apache/spark/blob/master/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L94
  auto const index =
      (h < 0 ? ~h : h) % static_cast<bloom_hash_t>(bloomFilterBits);
  // Big-endian word/byte swizzle to match Spark's serialized layout
  auto const wordIndex = cudf::word_index(index) ^ 0x1;
  auto const bitIndex = cudf::intra_word_index(index) ^ 0x18;
  return cuda::std::make_pair(wordIndex, uint32_t(1u << bitIndex));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template <bool nullable>
CUDF_KERNEL void bloomFilterPutKernel(
    cudf::bitmask_type* bloomFilter,
    int32_t bloomFilterBits,
    cudf::column_device_view input,
    int32_t numHashes) {
  auto const tid = static_cast<cudf::size_type>(
      threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x));
  if (tid >= input.size())
    return;

  if constexpr (nullable) {
    if (!input.is_valid(tid))
      return;
  }

  auto const el = input.element<int64_t>(tid);
  bloom_hash_t const h1 = murmurHash3_32(el, 0);
  bloom_hash_t const h2 = murmurHash3_32(el, h1);

  for (int idx = 1; idx <= numHashes; idx++) {
    bloom_hash_t combined = h1 + (idx * h2);
    auto const [wordIdx, mask] = getHashMask(combined, bloomFilterBits);
    atomicOr(bloomFilter + wordIdx, mask);
  }
}

struct BloomProbeFunctor {
  cudf::bitmask_type const* filter;
  int32_t bloomFilterBits;
  int32_t numHashes;

  __device__ bool operator()(int64_t input) const {
    bloom_hash_t const h1 = murmurHash3_32(input, 0);
    bloom_hash_t const h2 = murmurHash3_32(input, h1);

    for (int idx = 1; idx <= numHashes; idx++) {
      bloom_hash_t combined = h1 + (idx * h2);
      auto const [wordIdx, mask] = getHashMask(combined, bloomFilterBits);
      if (!(filter[wordIdx] & mask))
        return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Header pack/unpack (big-endian swizzle)
// ---------------------------------------------------------------------------

SparkBloomFilterHeader byteSwapHeader(SparkBloomFilterHeader const& h) {
  return {
      static_cast<int32_t>(bswap_32(static_cast<uint32_t>(h.version))),
      static_cast<int32_t>(bswap_32(static_cast<uint32_t>(h.numHashes))),
      static_cast<int32_t>(bswap_32(static_cast<uint32_t>(h.numLongs)))};
}

void packHeader(
    uint8_t* dst,
    SparkBloomFilterHeader const& header,
    rmm::cuda_stream_view stream) {
  SparkBloomFilterHeader swizzled = byteSwapHeader(header);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst,
      &swizzled,
      kSparkBloomFilterHeaderSize,
      cudaMemcpyHostToDevice,
      stream.value()));
}

SparkBloomFilterHeader unpackHeader(
    uint8_t const* src,
    rmm::cuda_stream_view stream) {
  SparkBloomFilterHeader swizzled;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &swizzled,
      src,
      kSparkBloomFilterHeaderSize,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return byteSwapHeader(swizzled);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int computeNumHashes(int64_t expectedNumItems, int64_t numBits) {
  if (expectedNumItems <= 0)
    return 1;
  return std::max(
      1, static_cast<int>(std::round(
             static_cast<double>(numBits) / expectedNumItems * std::log(2))));
}

std::unique_ptr<rmm::device_buffer> bloomFilterCreate(
    int numHashes,
    int numLongs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const bloomBytes = static_cast<size_t>(numLongs) * 8;
  auto const totalSize = kSparkBloomFilterHeaderSize + bloomBytes;

  auto buf = std::make_unique<rmm::device_buffer>(totalSize, stream, mr);

  SparkBloomFilterHeader header{kSparkBloomFilterVersion, numHashes, numLongs};
  packHeader(static_cast<uint8_t*>(buf->data()), header, stream);

  CUDF_CUDA_TRY(cudaMemsetAsync(
      static_cast<uint8_t*>(buf->data()) + kSparkBloomFilterHeaderSize,
      0,
      bloomBytes,
      stream.value()));

  return buf;
}

void bloomFilterPut(
    rmm::device_buffer& bloomFilter,
    cudf::column_view const& input,
    rmm::cuda_stream_view stream) {
  auto header = unpackHeader(
      static_cast<uint8_t const*>(bloomFilter.data()), stream);
  CUDF_EXPECTS(
      header.version == kSparkBloomFilterVersion,
      "Unexpected bloom filter version");

  auto const bloomFilterBits = header.numLongs * 64;
  auto* bits = reinterpret_cast<cudf::bitmask_type*>(
      static_cast<uint8_t*>(bloomFilter.data()) + kSparkBloomFilterHeaderSize);

  constexpr int blockSize = 256;
  auto const gridSize = (input.size() + blockSize - 1) / blockSize;
  auto dInput = cudf::column_device_view::create(input, stream);

  if (input.has_nulls()) {
    bloomFilterPutKernel<true>
        <<<gridSize, blockSize, 0, stream.value()>>>(
            bits, bloomFilterBits, *dInput, header.numHashes);
  } else {
    bloomFilterPutKernel<false>
        <<<gridSize, blockSize, 0, stream.value()>>>(
            bits, bloomFilterBits, *dInput, header.numHashes);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

std::unique_ptr<cudf::column> bloomFilterProbe(
    cudf::column_view const& input,
    cudf::device_span<uint8_t const> bloomFilter,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto header =
      unpackHeader(const_cast<uint8_t*>(bloomFilter.data()), stream);
  CUDF_EXPECTS(
      header.version == kSparkBloomFilterVersion,
      "Unexpected bloom filter version");

  auto const bloomFilterBits = header.numLongs * 64;
  auto const* bits = reinterpret_cast<cudf::bitmask_type const*>(
      bloomFilter.data() + kSparkBloomFilterHeaderSize);

  auto out = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8},
      input.size(),
      cudf::copy_bitmask(input, stream, mr),
      input.null_count(),
      stream,
      mr);

  thrust::transform(
      rmm::exec_policy(stream),
      input.begin<int64_t>(),
      input.end<int64_t>(),
      out->mutable_view().begin<bool>(),
      BloomProbeFunctor{bits, bloomFilterBits, header.numHashes});

  return out;
}

std::unique_ptr<rmm::device_buffer> bloomFilterMerge(
    cudf::column_view const& bloomFiltersColumn,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const numFilters = bloomFiltersColumn.size();
  CUDF_EXPECTS(numFilters > 0, "Need at least one bloom filter to merge");

  cudf::strings_column_view scv(bloomFiltersColumn);
  auto const offsetsView = scv.offsets();
  auto const charsPtr =
      reinterpret_cast<uint8_t const*>(scv.chars_begin(stream));

  // Read offsets to host to find the first filter's size
  std::vector<int32_t> hostOffsets(numFilters + 1);
  auto const offsetsType = offsetsView.type().id();
  if (offsetsType == cudf::type_id::INT64) {
    std::vector<int64_t> offsets64(numFilters + 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsets64.data(),
        offsetsView.data<int64_t>(),
        (numFilters + 1) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    for (size_t i = 0; i <= static_cast<size_t>(numFilters); i++) {
      hostOffsets[i] = static_cast<int32_t>(offsets64[i]);
    }
  } else {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostOffsets.data(),
        offsetsView.data<int32_t>(),
        (numFilters + 1) * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
  }

  auto const filterSize = hostOffsets[1] - hostOffsets[0];
  CUDF_EXPECTS(
      filterSize >= kSparkBloomFilterHeaderSize,
      "Bloom filter too small");

  auto header =
      unpackHeader(charsPtr + hostOffsets[0], stream);
  CUDF_EXPECTS(
      header.version == kSparkBloomFilterVersion,
      "Unexpected bloom filter version in merge");

  auto const bloomBytes = static_cast<size_t>(header.numLongs) * 8;
  auto const totalSize = kSparkBloomFilterHeaderSize + bloomBytes;
  CUDF_EXPECTS(
      static_cast<size_t>(filterSize) == totalSize,
      "Bloom filter size mismatch");

  // Allocate output and pack header
  auto output = std::make_unique<rmm::device_buffer>(totalSize, stream, mr);
  packHeader(static_cast<uint8_t*>(output->data()), header, stream);

  auto const numWords = header.numLongs * 2; // 32-bit words
  auto* dst = reinterpret_cast<cudf::bitmask_type*>(
      static_cast<uint8_t*>(output->data()) + kSparkBloomFilterHeaderSize);

  auto const* srcBase = charsPtr + hostOffsets[0] + kSparkBloomFilterHeaderSize;
  auto const stride = filterSize;

  // Bitwise-OR all bloom filters together
  thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(numWords),
      dst,
      cuda::proclaim_return_type<cudf::bitmask_type>(
          [srcBase,
           numBuffers = numFilters,
           stride] __device__(int32_t wordIndex) {
            auto const* base =
                reinterpret_cast<cudf::bitmask_type const*>(srcBase);
            cudf::bitmask_type result = base[wordIndex];
            for (int i = 1; i < numBuffers; i++) {
              auto const* filter = reinterpret_cast<cudf::bitmask_type const*>(
                  reinterpret_cast<uint8_t const*>(srcBase) + i * stride);
              result |= filter[wordIndex];
            }
            return result;
          }));

  return output;
}

std::unique_ptr<cudf::column> bloomFilterToStringsColumn(
    rmm::device_buffer const& bloomFilter,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const totalSize = static_cast<cudf::size_type>(bloomFilter.size());

  // Build offsets: [0, totalSize]
  int32_t hostOffsets[2] = {0, totalSize};
  rmm::device_buffer offsetsBuf(2 * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetsBuf.data(),
      hostOffsets,
      2 * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));

  // Copy bloom filter data to chars buffer
  rmm::device_buffer charsBuf(bloomFilter.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      charsBuf.data(),
      bloomFilter.data(),
      bloomFilter.size(),
      cudaMemcpyDeviceToDevice,
      stream.value()));

  stream.synchronize();

  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      2,
      std::move(offsetsBuf),
      rmm::device_buffer{},
      0);

  return cudf::make_strings_column(
      1,
      std::move(offsetsColumn),
      std::move(charsBuf),
      0,
      rmm::device_buffer{});
}

} // namespace facebook::velox::cudf_velox
