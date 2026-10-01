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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace detail {

class PinnedHostBufferPool {
 public:
  static PinnedHostBufferPool& instance() {
    static PinnedHostBufferPool pool;
    return pool;
  }

  std::pair<uint8_t*, size_t> acquire(size_t requested) {
    const auto capacity = sizeClass(requested);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& buffers = buffers_[capacity];
      if (!buffers.empty()) {
        auto* data = buffers.back();
        buffers.pop_back();
        cachedBytes_ -= capacity;
        return {data, capacity};
      }
    }
    void* allocation = nullptr;
    if (cudaHostAlloc(&allocation, capacity, cudaHostAllocPortable) !=
        cudaSuccess) {
      cudaGetLastError();
      return {nullptr, 0};
    }
    return {static_cast<uint8_t*>(allocation), capacity};
  }

  void release(uint8_t* data, size_t capacity) {
    bool cache = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (capacity <= maxCachedBytes_ &&
          cachedBytes_ <= maxCachedBytes_ - capacity) {
        buffers_[capacity].push_back(data);
        cachedBytes_ += capacity;
        cache = true;
      }
    }
    if (!cache) {
      cudaFreeHost(data);
    }
  }

  ~PinnedHostBufferPool() {
    for (auto& [capacity, buffers] : buffers_) {
      for (auto* data : buffers) {
        cudaFreeHost(data);
      }
    }
  }

 private:
  static size_t sizeClass(size_t requested) {
    constexpr size_t kMinimumClass = 64UL << 10;
    auto capacity = kMinimumClass;
    while (capacity < requested &&
           capacity <= std::numeric_limits<size_t>::max() / 2) {
      capacity *= 2;
    }
    return capacity < requested ? requested : capacity;
  }

  static size_t maxCachedBytes() {
    constexpr size_t kDefault = 4ULL << 30;
    const auto* value = std::getenv("GLUTEN_CPP_S3_PINNED_POOL_MAX_BYTES");
    if (value == nullptr || value[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<size_t>(parsed)
                                        : kDefault;
  }

  PinnedHostBufferPool() : maxCachedBytes_(maxCachedBytes()) {}

  std::mutex mutex_;
  std::unordered_map<size_t, std::vector<uint8_t*>> buffers_;
  size_t cachedBytes_{0};
  const size_t maxCachedBytes_;
};

inline bool usePinnedHostBufferPool() {
  const auto* value = std::getenv("GLUTEN_CPP_S3_PINNED_POOL");
  return value != nullptr && std::string_view(value) == "1";
}

} // namespace detail

/// Host buffer used by the S3 prefetch path. It first uses CUDA portable
/// pinned memory and falls back to pageable memory without changing
/// correctness.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(
      size_t size,
      std::shared_ptr<void> lifetime = nullptr,
      bool preferPinned = true)
      : lifetime_(std::move(lifetime)), size_(size) {
    if (size_ == 0) {
      return;
    }
    if (preferPinned) {
      if (detail::usePinnedHostBufferPool()) {
        auto [allocation, capacity] =
            detail::PinnedHostBufferPool::instance().acquire(size_);
        if (allocation != nullptr) {
          data_ = allocation;
          capacity_ = capacity;
          pinned_ = true;
          pooled_ = true;
          return;
        }
      }
      void* allocation = nullptr;
      const auto result =
          cudaHostAlloc(&allocation, size_, cudaHostAllocPortable);
      if (result == cudaSuccess) {
        data_ = static_cast<uint8_t*>(allocation);
        pinned_ = true;
        return;
      }
      // Clear the CUDA runtime's per-thread error before falling back. This
      // path must remain usable when the pinned-memory budget is exhausted.
      cudaGetLastError();
    }
    data_ = static_cast<uint8_t*>(std::malloc(size_));
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
  }

  ~PinnedHostBuffer() {
    if (data_ == nullptr) {
      return;
    }
    if (pinned_) {
      if (pooled_) {
        detail::PinnedHostBufferPool::instance().release(data_, capacity_);
        return;
      }
      // Direct CUDA allocation keeps the allocation/deallocation pair stable
      // even if cuDF swaps its process-wide pinned memory resource while a
      // reader is alive.
      cudaFreeHost(data_);
    } else {
      std::free(data_);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  uint8_t* data() {
    return data_;
  }

  const uint8_t* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

  bool isPinned() const {
    return pinned_;
  }

  size_t allocatedBytes() const {
    return pooled_ ? capacity_ : size_;
  }

 private:
  // Keep any executor-wide byte reservation alive until after the allocation
  // is released by the destructor body.
  std::shared_ptr<void> lifetime_;
  uint8_t* data_{nullptr};
  size_t size_{0};
  size_t capacity_{0};
  bool pinned_{false};
  bool pooled_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
