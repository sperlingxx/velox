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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {

namespace {
std::mutex asyncMemoryPoolsMutex;
std::vector<cudaMemPool_t> asyncMemoryPools;
// registerCudf creates the main resource before its optional output resource.
// Remember which device has seen that first resource so an async output-only
// pool is never mistaken for the primary cuDF allocation pool.
std::unordered_set<int> primaryMemoryResourceDevices;
std::unordered_map<int, cudaMemPool_t> primaryAsyncMemoryPools;

std::mutex deviceMemoryAdmissionMutex;
std::unordered_map<int, std::size_t> deviceMemoryAdmissionBytes;

struct MemoryResourceRegistration {
  int device{-1};
  bool primary{false};
};

MemoryResourceRegistration beginMemoryResourceRegistration() {
  MemoryResourceRegistration registration;
  if (cudaGetDevice(&registration.device) != cudaSuccess) {
    return registration;
  }
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  registration.primary =
      primaryMemoryResourceDevices.insert(registration.device).second;
  return registration;
}

void registerAsyncMemoryPool(
    const MemoryResourceRegistration& registration,
    cudaMemPool_t pool) {
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  asyncMemoryPools.push_back(pool);
  if (registration.primary && registration.device >= 0) {
    primaryAsyncMemoryPools[registration.device] = pool;
  }
}

void releaseDeviceMemoryAdmission(int device, std::size_t bytes) noexcept {
  std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
  const auto it = deviceMemoryAdmissionBytes.find(device);
  if (it == deviceMemoryAdmissionBytes.end() || it->second < bytes) {
    LOG(ERROR) << "Invalid device-memory admission release device=" << device
               << " bytes=" << bytes;
    return;
  }
  it->second -= bytes;
  if (it->second == 0) {
    deviceMemoryAdmissionBytes.erase(it);
  }
}
} // namespace

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  const auto registration = beginMemoryResourceRegistration();
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    // cuda_async_memory_resource otherwise defaults its release threshold to
    // UINT64_MAX. After a transient large join/partition workspace, the CUDA
    // pool then retains almost the entire device even when RMM's live bytes
    // have fallen. UCX receive buffers use a separate cudaMalloc resource and
    // cannot reuse those cached blocks. Apply memoryPercent as the async
    // pool's retention threshold so synchronization returns high-watermark
    // slack while preserving a large cache for steady-state operators.
    auto resource = rmm::mr::cuda_async_memory_resource(
        std::nullopt, rmm::percent_of_free_device_memory(percent));
    registerAsyncMemoryPool(registration, resource.pool_handle());
    return resource;
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

bool trimAsyncMemoryPoolsAtQueryEnd() {
  const auto* value = std::getenv("GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }

  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    LOG(ERROR) << "Ignoring invalid GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES='"
               << value << "'";
    return false;
  }
  return trimAsyncMemoryPoolsAtQueryEnd(static_cast<std::size_t>(parsed));
}

bool trimAsyncMemoryPoolsAtQueryEnd(std::size_t bytesToKeep) {
  std::vector<cudaMemPool_t> pools;
  {
    std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
    pools = asyncMemoryPools;
  }
  if (pools.empty()) {
    return false;
  }

  std::size_t freeBefore = 0;
  std::size_t totalBytes = 0;
  cudaMemGetInfo(&freeBefore, &totalBytes);
  const auto syncStatus = cudaDeviceSynchronize();
  if (syncStatus != cudaSuccess) {
    LOG(ERROR) << "CUDF_ASYNC_QUERY_END_TRIM cudaDeviceSynchronize failed: "
               << cudaGetErrorString(syncStatus);
    return false;
  }

  for (const auto pool : pools) {
    const auto trimStatus = cudaMemPoolTrimTo(pool, bytesToKeep);
    if (trimStatus != cudaSuccess) {
      LOG(ERROR) << "CUDF_ASYNC_QUERY_END_TRIM cudaMemPoolTrimTo failed: "
                 << cudaGetErrorString(trimStatus)
                 << " bytesToKeep=" << bytesToKeep;
      return false;
    }
  }

  std::size_t freeAfter = 0;
  cudaMemGetInfo(&freeAfter, &totalBytes);
  LOG(INFO) << "CUDF_ASYNC_QUERY_END_TRIM pools=" << pools.size()
            << " bytesToKeep=" << bytesToKeep << " freeBefore=" << freeBefore
            << " freeAfter=" << freeAfter << " released="
            << (freeAfter >= freeBefore ? freeAfter - freeBefore : 0);
  return true;
}

void clearAsyncMemoryPoolHandles() {
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  asyncMemoryPools.clear();
  primaryAsyncMemoryPools.clear();
  primaryMemoryResourceDevices.clear();
}

std::size_t DeviceAllocationHeadroom::reusablePoolBytes() const noexcept {
  if (!asyncPoolValid || poolReservedBytes <= poolUsedBytes) {
    return 0;
  }
  return poolReservedBytes - poolUsedBytes;
}

std::size_t DeviceAllocationHeadroom::allocatableBytes() const noexcept {
  if (!cudaValid) {
    return 0;
  }
  const auto reusable = reusablePoolBytes();
  if (freeBytes > std::numeric_limits<std::size_t>::max() - reusable) {
    return std::numeric_limits<std::size_t>::max();
  }
  return freeBytes + reusable;
}

DeviceAllocationHeadroom captureDeviceAllocationHeadroom() {
  DeviceAllocationHeadroom headroom;
  if (cudaGetDevice(&headroom.device) != cudaSuccess) {
    return headroom;
  }

  headroom.cudaValid =
      cudaMemGetInfo(&headroom.freeBytes, &headroom.totalBytes) == cudaSuccess;
  if (!headroom.cudaValid) {
    headroom.freeBytes = 0;
    headroom.totalBytes = 0;
    return headroom;
  }

  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  const auto poolIt = primaryAsyncMemoryPools.find(headroom.device);
  if (poolIt == primaryAsyncMemoryPools.end()) {
    return headroom;
  }

  std::uint64_t reservedBytes = 0;
  std::uint64_t usedBytes = 0;
  const auto reservedStatus = cudaMemPoolGetAttribute(
      poolIt->second, cudaMemPoolAttrReservedMemCurrent, &reservedBytes);
  const auto usedStatus = cudaMemPoolGetAttribute(
      poolIt->second, cudaMemPoolAttrUsedMemCurrent, &usedBytes);
  if (reservedStatus != cudaSuccess || usedStatus != cudaSuccess) {
    return headroom;
  }

  headroom.asyncPoolValid = true;
  headroom.poolReservedBytes = static_cast<std::size_t>(reservedBytes);
  headroom.poolUsedBytes = static_cast<std::size_t>(usedBytes);
  return headroom;
}

DeviceMemoryAdmissionReservation::DeviceMemoryAdmissionReservation(
    int device,
    std::size_t bytes) noexcept
    : device_{device}, bytes_{bytes}, active_{true} {}

DeviceMemoryAdmissionReservation::~DeviceMemoryAdmissionReservation() {
  release();
}

DeviceMemoryAdmissionReservation::DeviceMemoryAdmissionReservation(
    DeviceMemoryAdmissionReservation&& other) noexcept
    : device_{other.device_}, bytes_{other.bytes_}, active_{other.active_} {
  other.device_ = -1;
  other.bytes_ = 0;
  other.active_ = false;
}

DeviceMemoryAdmissionReservation& DeviceMemoryAdmissionReservation::operator=(
    DeviceMemoryAdmissionReservation&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  device_ = other.device_;
  bytes_ = other.bytes_;
  active_ = other.active_;
  other.device_ = -1;
  other.bytes_ = 0;
  other.active_ = false;
  return *this;
}

void DeviceMemoryAdmissionReservation::release() noexcept {
  if (!active_) {
    return;
  }
  releaseDeviceMemoryAdmission(device_, bytes_);
  device_ = -1;
  bytes_ = 0;
  active_ = false;
}

std::optional<DeviceMemoryAdmissionReservation> tryAcquireDeviceMemoryAdmission(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes) {
  if (device < 0 || bytes > capacityBytes) {
    return std::nullopt;
  }

  {
    std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
    const auto it = deviceMemoryAdmissionBytes.find(device);
    const auto reserved =
        it == deviceMemoryAdmissionBytes.end() ? 0 : it->second;
    if (reserved > capacityBytes - bytes) {
      return std::nullopt;
    }
    if (it == deviceMemoryAdmissionBytes.end()) {
      deviceMemoryAdmissionBytes.emplace(device, bytes);
    } else {
      it->second += bytes;
    }
  }
  return DeviceMemoryAdmissionReservation{device, bytes};
}

std::size_t deviceMemoryAdmissionReservedBytes(int device) {
  if (device < 0) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
  const auto it = deviceMemoryAdmissionBytes.find(device);
  return it == deviceMemoryAdmissionBytes.end() ? 0 : it->second;
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;
std::optional<rmm::mr::statistics_resource_adaptor> statistics_mr_;
std::optional<rmm::mr::statistics_resource_adaptor> output_statistics_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

bool deviceMemoryDiagnosticsEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS");
    if (value == nullptr) {
      return false;
    }
    std::string normalized{value};
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !normalized.empty() && normalized != "0" && normalized != "false" &&
        normalized != "off" && normalized != "no";
  }();
  return enabled;
}

DeviceMemorySnapshot captureDeviceMemorySnapshot() {
  DeviceMemorySnapshot snapshot;
  snapshot.enabled = deviceMemoryDiagnosticsEnabled();
  if (!snapshot.enabled) {
    return snapshot;
  }

  if (cudaGetDevice(&snapshot.device) == cudaSuccess) {
    snapshot.cudaValid =
        cudaMemGetInfo(&snapshot.freeBytes, &snapshot.totalBytes) ==
        cudaSuccess;
    if (snapshot.cudaValid) {
      snapshot.usedBytes = snapshot.totalBytes - snapshot.freeBytes;
    }
  }

  const auto addStatistics = [&](const auto& statistics) {
    if (!statistics.has_value()) {
      return;
    }
    const auto bytes = statistics->get_bytes_counter();
    const auto allocations = statistics->get_allocations_counter();
    snapshot.rmmCurrentBytes += bytes.value;
    snapshot.rmmPeakBytes += bytes.peak;
    snapshot.rmmTotalBytes += bytes.total;
    snapshot.rmmCurrentAllocations += allocations.value;
    snapshot.rmmPeakAllocations += allocations.peak;
    snapshot.rmmTotalAllocations += allocations.total;
  };
  addStatistics(statistics_mr_);
  addStatistics(output_statistics_mr_);
  return snapshot;
}

void logDeviceMemorySnapshot(
    const std::string& label,
    const DeviceMemorySnapshot& snapshot) {
  if (!snapshot.enabled) {
    return;
  }
  const auto* visibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
  const auto* executorId = std::getenv("SPARK_EXECUTOR_ID");
  LOG(WARNING) << "CUDF_DEVICE_MEMORY"
               << " label={" << label << "}"
               << " pid=" << static_cast<int64_t>(::getpid())
               << " thread=" << std::this_thread::get_id()
               << " executorId=" << (executorId == nullptr ? "" : executorId)
               << " cudaVisibleDevices="
               << (visibleDevices == nullptr ? "" : visibleDevices)
               << " device=" << snapshot.device
               << " cudaValid=" << snapshot.cudaValid
               << " freeBytes=" << snapshot.freeBytes
               << " totalBytes=" << snapshot.totalBytes
               << " usedBytes=" << snapshot.usedBytes
               << " rmmCurrentBytes=" << snapshot.rmmCurrentBytes
               << " rmmPeakBytes=" << snapshot.rmmPeakBytes
               << " rmmTotalBytes=" << snapshot.rmmTotalBytes
               << " rmmCurrentAllocations=" << snapshot.rmmCurrentAllocations
               << " rmmPeakAllocations=" << snapshot.rmmPeakAllocations
               << " rmmTotalAllocations=" << snapshot.rmmTotalAllocations;
}

CudaAllocationTraceScope::CudaAllocationTraceScope(const std::string& label) {
  using Push = void (*)(const char*);
  static auto push = reinterpret_cast<Push>(
      dlsym(RTLD_DEFAULT, "cuda_alloc_trace_push_context"));
  if (push != nullptr) {
    push(label.c_str());
    active_ = true;
  }
}

CudaAllocationTraceScope::~CudaAllocationTraceScope() {
  if (!active_) {
    return;
  }
  using Pop = void (*)();
  static auto pop = reinterpret_cast<Pop>(
      dlsym(RTLD_DEFAULT, "cuda_alloc_trace_pop_context"));
  if (pop != nullptr) {
    pop();
  }
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
