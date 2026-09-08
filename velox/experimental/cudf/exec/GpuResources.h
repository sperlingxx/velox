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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

struct DeviceMemorySnapshot {
  bool enabled{false};
  bool cudaValid{false};
  int device{-1};
  std::size_t freeBytes{0};
  std::size_t totalBytes{0};
  std::size_t usedBytes{0};
  int64_t rmmCurrentBytes{0};
  int64_t rmmPeakBytes{0};
  int64_t rmmTotalBytes{0};
  int64_t rmmCurrentAllocations{0};
  int64_t rmmPeakAllocations{0};
  int64_t rmmTotalAllocations{0};
};

/// A cheap, always-available view of memory that allocations through the
/// current device's primary cuDF resource may be able to use. When the primary
/// resource is not cudaMallocAsync, or its pool attributes cannot be read,
/// allocatableBytes() conservatively reports cudaMemGetInfo's free bytes only.
struct DeviceAllocationHeadroom {
  bool cudaValid{false};
  bool asyncPoolValid{false};
  int device{-1};
  std::size_t freeBytes{0};
  std::size_t totalBytes{0};
  std::size_t poolReservedBytes{0};
  std::size_t poolUsedBytes{0};

  [[nodiscard]] std::size_t reusablePoolBytes() const noexcept;
  [[nodiscard]] std::size_t allocatableBytes() const noexcept;
};

/// Process-wide, per-device admission reservation. This does not allocate GPU
/// memory; it prevents cooperating operators from admitting work against the
/// same headroom snapshot. The reservation is released on destruction.
class DeviceMemoryAdmissionReservation {
 public:
  DeviceMemoryAdmissionReservation() = default;
  ~DeviceMemoryAdmissionReservation();

  DeviceMemoryAdmissionReservation(
      DeviceMemoryAdmissionReservation&& other) noexcept;
  DeviceMemoryAdmissionReservation& operator=(
      DeviceMemoryAdmissionReservation&& other) noexcept;

  DeviceMemoryAdmissionReservation(const DeviceMemoryAdmissionReservation&) =
      delete;
  DeviceMemoryAdmissionReservation& operator=(
      const DeviceMemoryAdmissionReservation&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return active_;
  }

  [[nodiscard]] int device() const noexcept {
    return device_;
  }

  [[nodiscard]] std::size_t reservedBytes() const noexcept {
    return bytes_;
  }

  void release() noexcept;

 private:
  friend std::optional<DeviceMemoryAdmissionReservation>
  tryAcquireDeviceMemoryAdmission(
      int device,
      std::size_t bytes,
      std::size_t capacityBytes);

  DeviceMemoryAdmissionReservation(int device, std::size_t bytes) noexcept;

  int device_{-1};
  std::size_t bytes_{0};
  bool active_{false};
};

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;
extern std::optional<rmm::mr::statistics_resource_adaptor> statistics_mr_;
extern std::optional<rmm::mr::statistics_resource_adaptor>
    output_statistics_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for pool or
 * arena resources, or the retained-memory release threshold for async.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// Wraps a cuDF resource with global statistics and per-operator live
/// allocation attribution. Enabled only by the diagnostic execution path.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
wrapDeviceMemoryResourceForDiagnostics(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    bool outputResource);

/// Re-attributes a live device allocation from the context it was allocated
/// under to 'newContext', so the OOM dump names the operator that currently
/// holds the memory rather than the one that created it. Live bytes and the
/// live allocation count move; the old context keeps its peak, and the new
/// context may raise its own.
///
/// Returns false when diagnostics are disabled or 'pointer' was not allocated
/// through a cuDF resource. Callers must invoke this only from the thread that
/// takes ownership, and only once per transfer, or live bytes will be counted
/// against more than one holder.
[[nodiscard]] bool reattributeDeviceAllocation(
    void* pointer,
    const std::string& newContext);

/// Live device memory billed to one allocation context.
struct DeviceAllocationContextStats {
  std::string context;
  std::size_t currentBytes{0};
  std::size_t peakBytes{0};
  std::size_t currentAllocations{0};
};

/// Returns live per-context attribution across the primary and output cuDF
/// resources, including zero-byte contexts. Empty when diagnostics are
/// disabled. The currentBytes sum equals the RMM live bytes of the same
/// resources, so this is the programmatic form of the CUDF_DEVICE_OOM owner
/// list. Copies under the attribution mutex; not for per-batch use.
[[nodiscard]] std::vector<DeviceAllocationContextStats>
captureDeviceAllocationAttribution();

/// Returns the allocation context that CudaAllocationTraceScope has
/// established on the calling thread, or "unattributed" when tracing is off.
[[nodiscard]] std::string currentDeviceAllocationContext();

/// Drops the diagnostic attribution resources. Must be called before the
/// statistics resources they wrap are destroyed, otherwise they are left
/// holding a reference to a destroyed upstream.
void clearDeviceMemoryAttributionResources();

/// If GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES is set, synchronizes the device
/// and releases unused cudaMallocAsync pool memory down to that retained-byte
/// target. Returns true when a trim was requested and completed successfully.
[[nodiscard]] bool trimAsyncMemoryPoolsAtQueryEnd();

/// Explicit query-scoped variant. Unlike the no-argument environment
/// fallback, a zero value is a valid request to release every unused block.
[[nodiscard]] bool trimAsyncMemoryPoolsAtQueryEnd(std::size_t bytesToKeep);

/// Drops native pool handles after their owning RMM resources are destroyed.
void clearAsyncMemoryPoolHandles();

/// Captures current-device free memory and, when the primary cuDF resource is
/// cudaMallocAsync, that pool's currently reusable reserved memory. This API is
/// independent of diagnostic logging and returns a conservative partial
/// snapshot when pool attributes are unavailable.
[[nodiscard]] DeviceAllocationHeadroom captureDeviceAllocationHeadroom();

/// Atomically acquires a cooperative admission reservation for a device when
/// existing reservations plus 'bytes' do not exceed 'capacityBytes'. Returns
/// nullopt for invalid devices or insufficient capacity.
[[nodiscard]] std::optional<DeviceMemoryAdmissionReservation>
tryAcquireDeviceMemoryAdmission(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes);

/// Returns bytes currently reserved by cooperative admission clients on a
/// device. This is admission accounting, not live RMM allocation usage.
[[nodiscard]] std::size_t deviceMemoryAdmissionReservedBytes(int device);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Enables low-overhead RMM statistics and operator-level CUDA memory
/// diagnostics when GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS is set to a true
/// value in the executor environment.
[[nodiscard]] bool deviceMemoryDiagnosticsEnabled();

/// Captures both CUDA device-wide usage and allocations made through the
/// cuDF RMM resource. Unlike cudaMemGetInfo-only diagnostics, the snapshot
/// always includes the current CUDA device id.
[[nodiscard]] DeviceMemorySnapshot captureDeviceMemorySnapshot();

/// Emits a structured CUDF_DEVICE_MEMORY log record for later correlation
/// with the operator node and method that triggered the sample.
void logDeviceMemorySnapshot(
    const std::string& label,
    const DeviceMemorySnapshot& snapshot);

inline void logDeviceMemorySnapshot(const std::string& label) {
  if (deviceMemoryDiagnosticsEnabled()) {
    logDeviceMemorySnapshot(label, captureDeviceMemorySnapshot());
  }
}

/// Returns true when an allocation context is worth establishing, i.e. when
/// device memory diagnostics are on or the optional LD_PRELOAD tracer is
/// loaded. Cached after the first call.
[[nodiscard]] bool deviceAllocationTracingEnabled();

/// Associates device allocations made on the current thread with the operator
/// and method that requested them. The context is recorded by the diagnostic
/// attribution resource and, when the optional LD_PRELOAD tracer is loaded,
/// pushed to it as well.
///
/// Inert when deviceAllocationTracingEnabled() is false: nothing is formatted,
/// copied, or stored. Prefer the format-string constructor on hot paths so the
/// label is built only when it will be used; use the std::string constructor
/// for labels cached outside the hot path.
class CudaAllocationTraceScope {
 public:
  explicit CudaAllocationTraceScope(const std::string& label);

  /// Formats 'label' only when tracing is enabled. Takes at least one
  /// argument so a single-argument call is unambiguously the constructor
  /// above.
  template <typename Arg, typename... Args>
  CudaAllocationTraceScope(
      fmt::format_string<Arg, Args...> label,
      Arg&& arg,
      Args&&... args) {
    if (!deviceAllocationTracingEnabled()) {
      return;
    }
    enter(
        fmt::format(
            label, std::forward<Arg>(arg), std::forward<Args>(args)...));
  }

  ~CudaAllocationTraceScope();

  CudaAllocationTraceScope(const CudaAllocationTraceScope&) = delete;
  CudaAllocationTraceScope& operator=(const CudaAllocationTraceScope&) = delete;

 private:
  void enter(const std::string& label);

  bool entered_{false};
  bool active_{false};
  std::string previousContext_;
};

} // namespace facebook::velox::cudf_velox
