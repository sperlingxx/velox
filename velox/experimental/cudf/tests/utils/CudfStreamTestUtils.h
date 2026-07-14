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

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::test {

struct RecordingAsyncResourceState {
  cudaStream_t lastDeallocationStream{nullptr};
  std::size_t deallocationCount{0};
};

class RecordingAsyncDeviceResource {
  struct PendingDeallocation {
    void* ptr;
    std::size_t bytes;
    std::size_t alignment;
  };

  struct Storage {
    explicit Storage(bool defer) : deferDeallocations(defer) {}

    ~Storage() {
      releaseDeferred();
    }

    void releaseDeferred() noexcept {
      if (pendingDeallocations.empty()) {
        return;
      }
      cudaDeviceSynchronize();
      for (const auto& pending : pendingDeallocations) {
        asyncUpstream.deallocate(
            cuda::stream_ref{cudaStreamPerThread},
            pending.ptr,
            pending.bytes,
            pending.alignment);
      }
      cudaDeviceSynchronize();
      pendingDeallocations.clear();
    }

    RecordingAsyncResourceState state;
    rmm::mr::cuda_async_memory_resource asyncUpstream;
    bool deferDeallocations;
    std::vector<PendingDeallocation> pendingDeallocations;
  };

 public:
  explicit RecordingAsyncDeviceResource(bool deferDeallocations = false)
      : storage_(std::make_shared<Storage>(deferDeallocations)) {}

  void* allocate(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return storage_->asyncUpstream.allocate(stream, bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    storage_->state.lastDeallocationStream = stream.get();
    ++storage_->state.deallocationCount;
    if (storage_->deferDeallocations) {
      try {
        storage_->pendingDeallocations.push_back({ptr, bytes, alignment});
      } catch (...) {
        cudaDeviceSynchronize();
        storage_->asyncUpstream.deallocate(stream, ptr, bytes, alignment);
      }
      return;
    }
    storage_->asyncUpstream.deallocate(stream, ptr, bytes, alignment);
  }

  void* allocate_sync(
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return storage_->asyncUpstream.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(
      void* ptr,
      std::size_t bytes,
      std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    storage_->asyncUpstream.deallocate_sync(ptr, bytes, alignment);
  }

  void reset() {
    storage_->state.lastDeallocationStream = nullptr;
    storage_->state.deallocationCount = 0;
  }

  std::size_t deallocationCount() const {
    return storage_->state.deallocationCount;
  }

  cudaStream_t lastDeallocationStream() const {
    return storage_->state.lastDeallocationStream;
  }

  void releaseDeferred() noexcept {
    storage_->releaseDeferred();
  }

  bool operator==(const RecordingAsyncDeviceResource& other) const noexcept {
    return storage_ == other.storage_;
  }

 private:
  std::shared_ptr<Storage> storage_;
};

inline void get_property(
    const RecordingAsyncDeviceResource&,
    cuda::mr::device_accessible) noexcept {}

} // namespace facebook::velox::cudf_velox::test
