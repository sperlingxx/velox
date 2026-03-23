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

// Separate translation unit for pinned-memory Arrow interop.
// Includes nanoarrow (cudf's private dependency) — must NOT include
// Apache Arrow C++ headers in this file to avoid struct redefinition.

#include "velox/experimental/cudf/exec/PinnedArrowInterop.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/CudfConfig.h"

#include <glog/logging.h>

#include <cudf/contiguous_split.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/hpp/exception.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>

namespace facebook::velox::cudf_velox {

namespace {

struct DtoHStats {
  std::atomic<uint64_t> packedCalls{0};
  std::atomic<uint64_t> packedBytes{0};
  std::atomic<uint64_t> packedCols{0};
  std::atomic<uint64_t> packedRows{0};
  std::atomic<uint64_t> fallbackCalls{0};
  std::atomic<uint64_t> fallbackBuffers{0};
  std::atomic<uint64_t> fallbackBytes{0};
  std::atomic<uint64_t> emptyCalls{0};

  static DtoHStats& instance() {
    static DtoHStats s;
    return s;
  }

  void dump(const char* event) {
    fprintf(
        stderr,
        "[DtoH] %s: packed=%lu/%luMB (%lu cols, %lu rows) "
        "fallback=%lu/%lu bufs/%luMB empty=%lu\n",
        event,
        (unsigned long)packedCalls.load(std::memory_order_relaxed),
        (unsigned long)(packedBytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long)packedCols.load(std::memory_order_relaxed),
        (unsigned long)packedRows.load(std::memory_order_relaxed),
        (unsigned long)fallbackCalls.load(std::memory_order_relaxed),
        (unsigned long)fallbackBuffers.load(std::memory_order_relaxed),
        (unsigned long)(fallbackBytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long)emptyCalls.load(std::memory_order_relaxed));
  }
};

int countBuffersRecursive(ArrowArray* array) {
  int cnt = 0;
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    auto* buf = ArrowArrayBuffer(array, i);
    if (buf->data != nullptr && buf->size_bytes > 0)
      ++cnt;
  }
  for (int64_t i = 0; i < array->n_children; ++i)
    cnt += countBuffersRecursive(array->children[i]);
  if (array->dictionary)
    cnt += countBuffersRecursive(array->dictionary);
  return cnt;
}

/// Recursively copy device buffers in a nanoarrow ArrowArray to pinned host.
/// (Legacy per-buffer path — kept as fallback.)
void copyDeviceBuffersToPinnedHost(
    ArrowArray* array,
    rmm::cuda_stream_view stream) {
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    auto* buf = ArrowArrayBuffer(array, i);
    if (buf->data == nullptr || buf->size_bytes <= 0) {
      continue;
    }

    auto* pinned = new PinnedHostBuffer(buf->size_bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        pinned->data(),
        buf->data,
        buf->size_bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));

    buf->data = pinned->data();
    buf->allocator.reallocate = nullptr;
    buf->allocator.free = [](ArrowBufferAllocator* alloc,
                             uint8_t* /*ptr*/,
                             int64_t /*size*/) {
      delete static_cast<PinnedHostBuffer*>(alloc->private_data);
    };
    buf->allocator.private_data = pinned;
    array->buffers[i] = pinned->data();
  }

  for (int64_t i = 0; i < array->n_children; ++i) {
    copyDeviceBuffersToPinnedHost(array->children[i], stream);
  }
  if (array->dictionary) {
    copyDeviceBuffersToPinnedHost(array->dictionary, stream);
  }
}

/// Recursively rebase ArrowArray buffer pointers from a device address range
/// into a contiguous pinned host buffer.  Each nanoarrow ArrowBuffer gets a
/// shared_ptr ref to keep the host buffer alive until all consumers release.
/// Returns false if any buffer pointer falls outside the packed device range
/// (caller should fall back to per-buffer D2H).
bool rebaseArrowBuffersToHost(
    ArrowArray* array,
    const uint8_t* devBase,
    size_t devSize,
    const std::shared_ptr<PinnedHostBuffer>& hostBuf) {
  uint8_t* hostBase = hostBuf->data();

  for (int64_t i = 0; i < array->n_buffers; ++i) {
    auto* buf = ArrowArrayBuffer(array, i);
    if (buf->data == nullptr || buf->size_bytes <= 0) {
      continue;
    }

    auto* devPtr = static_cast<const uint8_t*>(buf->data);
    ptrdiff_t offset = devPtr - devBase;
    if (offset < 0 ||
        static_cast<size_t>(offset + buf->size_bytes) > devSize) {
      return false;
    }

    buf->data = hostBase + offset;
    array->buffers[i] = hostBase + offset;

    auto* ref = new std::shared_ptr<PinnedHostBuffer>(hostBuf);
    buf->allocator.reallocate = nullptr;
    buf->allocator.free = [](ArrowBufferAllocator* alloc,
                             uint8_t* /*ptr*/,
                             int64_t /*size*/) {
      delete static_cast<std::shared_ptr<PinnedHostBuffer>*>(
          alloc->private_data);
    };
    buf->allocator.private_data = ref;
  }

  for (int64_t i = 0; i < array->n_children; ++i) {
    if (!rebaseArrowBuffersToHost(
            array->children[i], devBase, devSize, hostBuf)) {
      return false;
    }
  }
  if (array->dictionary) {
    if (!rebaseArrowBuffersToHost(
            array->dictionary, devBase, devSize, hostBuf)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Pack metadata layout (mirrors cudf internal serialized_column).
// ---------------------------------------------------------------------------
struct SerializedColumn {
  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type null_count;
  int64_t data_offset;
  int64_t null_mask_offset;
  cudf::size_type num_children;
  int pad;
};

/// Attach a shared_ptr<PinnedHostBuffer> ref to a nanoarrow ArrowBuffer so the
/// host buffer stays alive as long as any consumer holds a reference.
void attachHostBufToArrowBuffer(
    ArrowBuffer* buf,
    const std::shared_ptr<PinnedHostBuffer>& hostBuf,
    uint8_t* ptr,
    int64_t sizeBytes) {
  buf->data = ptr;
  buf->size_bytes = sizeBytes;
  auto* ref = new std::shared_ptr<PinnedHostBuffer>(hostBuf);
  buf->allocator.reallocate = nullptr;
  buf->allocator.free = [](ArrowBufferAllocator* alloc,
                           uint8_t* /*ptr*/,
                           int64_t /*size*/) {
    delete static_cast<std::shared_ptr<PinnedHostBuffer>*>(
        alloc->private_data);
  };
  buf->allocator.private_data = ref;
}

/// Recursively build a nanoarrow ArrowArray from packed host buffer + metadata.
/// `idx` is advanced past the current column and all its children.
void buildArrowColumnFromPacked(
    const SerializedColumn* meta,
    size_t& idx,
    uint8_t* hostBase,
    const std::shared_ptr<PinnedHostBuffer>& hostBuf,
    ArrowArray* out) {
  auto& col = meta[idx++];

  auto typeId = col.type.id();
  bool isString = (typeId == cudf::type_id::STRING);

  if (isString) {
    // String column: cudf stores offsets child + chars data.
    // Arrow layout: buf[0]=validity, buf[1]=offsets, buf[2]=chars

    // Peek at the offsets child to determine offset width.
    CUDF_EXPECTS(
        col.num_children >= 1, "String column must have offsets child");
    auto& offsetsCol = meta[idx]; // peek, don't advance yet
    bool largeOffsets = (offsetsCol.type.id() == cudf::type_id::INT64);

    auto arrowStringType =
        largeOffsets ? NANOARROW_TYPE_LARGE_STRING : NANOARROW_TYPE_STRING;
    NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(out, arrowStringType));
    out->length = col.size;
    out->null_count = col.null_count;

    // validity
    if (col.null_mask_offset != -1) {
      auto* buf = ArrowArrayBuffer(out, 0);
      auto maskBytes =
          static_cast<int64_t>(cudf::bitmask_allocation_size_bytes(col.size));
      attachHostBufToArrowBuffer(
          buf, hostBuf, hostBase + col.null_mask_offset, maskBytes);
      out->buffers[0] = buf->data;
    }

    // Consume the offsets child in metadata.
    idx++; // advance past offsetsCol
    for (int c = 0; c < offsetsCol.num_children; ++c) {
      idx++;
    }

    auto offsetElemSize =
        largeOffsets ? sizeof(int64_t) : sizeof(int32_t);
    auto offsetsBytes =
        static_cast<int64_t>(col.size + 1) * offsetElemSize;
    auto* offsetsBuf = ArrowArrayBuffer(out, 1);
    if (offsetsCol.data_offset != -1) {
      attachHostBufToArrowBuffer(
          offsetsBuf,
          hostBuf,
          hostBase + offsetsCol.data_offset,
          offsetsBytes);
    }
    out->buffers[1] = offsetsBuf->data;

    // chars data: read chars_size from host-side offsets (last element).
    // This avoids the D2H that cudf::strings_column_view::chars_size() does.
    int64_t charsSize = 0;
    if (col.size > 0 && offsetsCol.data_offset != -1) {
      auto* offsetsPtr = hostBase + offsetsCol.data_offset;
      if (largeOffsets) {
        charsSize = reinterpret_cast<const int64_t*>(offsetsPtr)[col.size];
      } else {
        charsSize = reinterpret_cast<const int32_t*>(offsetsPtr)[col.size];
      }
    }

    auto* charsBuf = ArrowArrayBuffer(out, 2);
    if (col.data_offset != -1 && charsSize > 0) {
      attachHostBufToArrowBuffer(
          charsBuf, hostBuf, hostBase + col.data_offset, charsSize);
    }
    out->buffers[2] = charsBuf->data;

    // Skip remaining children beyond offsets (chars is stored as parent data).
    for (int c = 1; c < col.num_children; ++c) {
      auto& childCol = meta[idx++];
      for (int gc = 0; gc < childCol.num_children; ++gc) {
        idx++;
      }
    }
  } else {
    // Fixed-width or other column
    auto arrowType = NANOARROW_TYPE_UNINITIALIZED;
    int64_t elemSize = 0;
    switch (typeId) {
      case cudf::type_id::INT8:
        arrowType = NANOARROW_TYPE_INT8;
        elemSize = 1;
        break;
      case cudf::type_id::INT16:
        arrowType = NANOARROW_TYPE_INT16;
        elemSize = 2;
        break;
      case cudf::type_id::INT32:
        arrowType = NANOARROW_TYPE_INT32;
        elemSize = 4;
        break;
      case cudf::type_id::INT64:
        arrowType = NANOARROW_TYPE_INT64;
        elemSize = 8;
        break;
      case cudf::type_id::UINT8:
        arrowType = NANOARROW_TYPE_UINT8;
        elemSize = 1;
        break;
      case cudf::type_id::UINT16:
        arrowType = NANOARROW_TYPE_UINT16;
        elemSize = 2;
        break;
      case cudf::type_id::UINT32:
        arrowType = NANOARROW_TYPE_UINT32;
        elemSize = 4;
        break;
      case cudf::type_id::UINT64:
        arrowType = NANOARROW_TYPE_UINT64;
        elemSize = 8;
        break;
      case cudf::type_id::FLOAT32:
        arrowType = NANOARROW_TYPE_FLOAT;
        elemSize = 4;
        break;
      case cudf::type_id::FLOAT64:
        arrowType = NANOARROW_TYPE_DOUBLE;
        elemSize = 8;
        break;
      case cudf::type_id::BOOL8:
        arrowType = NANOARROW_TYPE_BOOL;
        elemSize = 0;
        break;
      case cudf::type_id::TIMESTAMP_DAYS:
        arrowType = NANOARROW_TYPE_INT32;
        elemSize = 4;
        break;
      case cudf::type_id::TIMESTAMP_SECONDS:
      case cudf::type_id::TIMESTAMP_MILLISECONDS:
      case cudf::type_id::TIMESTAMP_MICROSECONDS:
      case cudf::type_id::TIMESTAMP_NANOSECONDS:
      case cudf::type_id::DURATION_SECONDS:
      case cudf::type_id::DURATION_MILLISECONDS:
      case cudf::type_id::DURATION_MICROSECONDS:
      case cudf::type_id::DURATION_NANOSECONDS:
        arrowType = NANOARROW_TYPE_INT64;
        elemSize = 8;
        break;
      case cudf::type_id::DECIMAL32:
        arrowType = NANOARROW_TYPE_INT32;
        elemSize = 4;
        break;
      case cudf::type_id::DECIMAL64:
        arrowType = NANOARROW_TYPE_INT64;
        elemSize = 8;
        break;
      case cudf::type_id::DECIMAL128:
        arrowType = NANOARROW_TYPE_DECIMAL128;
        elemSize = 16;
        break;
      case cudf::type_id::STRUCT:
        arrowType = NANOARROW_TYPE_STRUCT;
        elemSize = 0;
        break;
      default:
        arrowType = NANOARROW_TYPE_BINARY;
        elemSize = 0;
        break;
    }

    NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(out, arrowType));
    out->length = col.size;
    out->null_count = col.null_count;

    // validity
    if (col.null_mask_offset != -1) {
      auto* buf = ArrowArrayBuffer(out, 0);
      auto maskBytes =
          static_cast<int64_t>(cudf::bitmask_allocation_size_bytes(col.size));
      attachHostBufToArrowBuffer(
          buf, hostBuf, hostBase + col.null_mask_offset, maskBytes);
      out->buffers[0] = buf->data;
    }

    // data
    if (col.data_offset != -1 && elemSize > 0) {
      auto* buf = ArrowArrayBuffer(out, 1);
      auto dataBytes = static_cast<int64_t>(col.size) * elemSize;
      attachHostBufToArrowBuffer(
          buf, hostBuf, hostBase + col.data_offset, dataBytes);
      out->buffers[1] = buf->data;
    } else if (col.data_offset != -1 && arrowType == NANOARROW_TYPE_BOOL) {
      // cudf BOOL8 stores 1 byte per element; Arrow BOOL stores 1 bit
      // per element. Convert byte-per-value to bit-packed bitmask.
      auto numElements = static_cast<int64_t>(col.size);
      auto bitmaskBytes = (numElements + 7) / 8;
      auto* srcBytes = hostBase + col.data_offset;

      auto* buf = ArrowArrayBuffer(out, 1);
      NANOARROW_THROW_NOT_OK(ArrowBufferResize(buf, bitmaskBytes, 0));
      auto* dst = reinterpret_cast<uint8_t*>(buf->data);
      memset(dst, 0, bitmaskBytes);
      for (int64_t i = 0; i < numElements; ++i) {
        if (srcBytes[i]) {
          dst[i / 8] |= static_cast<uint8_t>(1 << (i % 8));
        }
      }
    }

    // Recurse into children (e.g. for STRUCT or LIST types)
    if (col.num_children > 0) {
      NANOARROW_THROW_NOT_OK(
          ArrowArrayAllocateChildren(out, col.num_children));
      for (int c = 0; c < col.num_children; ++c) {
        buildArrowColumnFromPacked(
            meta, idx, hostBase, hostBuf, out->children[c]);
      }
    }
  }
}

/// Build a complete ArrowArray (struct-of-columns) from packed host buffer.
void buildArrowFromPackedHost(
    const std::vector<uint8_t>& metadata,
    const std::shared_ptr<PinnedHostBuffer>& hostBuf,
    int numColumns,
    int64_t numRows,
    ArrowArray* out) {
  auto* meta = reinterpret_cast<const SerializedColumn*>(metadata.data());
  // First entry is a stub: size == number of top-level columns
  size_t idx = 1;

  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromType(out, NANOARROW_TYPE_STRUCT));
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(out, numColumns));
  out->length = numRows;
  out->null_count = 0;

  for (int c = 0; c < numColumns; ++c) {
    buildArrowColumnFromPacked(
        meta, idx, hostBuf->data(), hostBuf, out->children[c]);
  }

  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuilding(
      out, NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));
}

} // namespace

cudf::unique_device_array_t pinnedToArrowHost(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream) {
  if (table.num_rows() == 0 || table.num_columns() == 0) {
    DtoHStats::instance().emptyCalls.fetch_add(1, std::memory_order_relaxed);
    auto devArray = cudf::to_arrow_device(table, stream);
    devArray->device_type = ARROW_DEVICE_CPU;
    return devArray;
  }

  const bool usePacked = CudfConfig::getInstance().packedDtoH;

  if (!usePacked) {
    auto devArray = cudf::to_arrow_device(table, stream);
    int bufs = countBuffersRecursive(&devArray->array);
    auto& stats = DtoHStats::instance();
    stats.fallbackCalls.fetch_add(1, std::memory_order_relaxed);
    stats.fallbackBuffers.fetch_add(bufs, std::memory_order_relaxed);
    copyDeviceBuffersToPinnedHost(&devArray->array, stream);
    stream.synchronize();
    auto calls = stats.fallbackCalls.load(std::memory_order_relaxed);
    if ((calls & (calls - 1)) == 0 || (calls % 500) == 0) {
      stats.dump("legacy");
    }
    devArray->device_type = ARROW_DEVICE_CPU;
    return devArray;
  }

  // --- Packed path: cudf::pack + 1 big D2H + build Arrow on host ---
  // Falls back to the legacy per-buffer path on any failure (e.g. pinned pool
  // exhaustion, CUDA errors) to avoid poisoning the CUDA context.
  try {
    auto packed = cudf::pack(table, stream);

    auto* devBase = static_cast<const uint8_t*>(packed.gpu_data->data());
    size_t devSize = packed.gpu_data->size();

    auto hostBuf = std::make_shared<PinnedHostBuffer>(devSize);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostBuf->data(),
        devBase,
        devSize,
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();

    auto* raw = new ArrowDeviceArray;
    std::memset(raw, 0, sizeof(ArrowDeviceArray));
    cudf::unique_device_array_t result(
        raw, [](ArrowDeviceArray* arr) {
          if (arr->array.release != nullptr) {
            ArrowArrayRelease(&arr->array);
          }
          delete arr;
        });
    result->device_id = 0;
    result->device_type = ARROW_DEVICE_CPU;
    result->sync_event = nullptr;

    buildArrowFromPackedHost(
        *packed.metadata,
        hostBuf,
        table.num_columns(),
        table.num_rows(),
        &result->array);

    auto& stats = DtoHStats::instance();
    auto calls =
        stats.packedCalls.fetch_add(1, std::memory_order_relaxed) + 1;
    stats.packedBytes.fetch_add(devSize, std::memory_order_relaxed);
    stats.packedCols.fetch_add(
        table.num_columns(), std::memory_order_relaxed);
    stats.packedRows.fetch_add(table.num_rows(), std::memory_order_relaxed);
    if ((calls & (calls - 1)) == 0 || (calls % 500) == 0) {
      stats.dump("packed-v2");
    }

    return result;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Packed D2H failed (" << e.what()
                 << "), falling back to legacy per-buffer D2H. "
                 << "Consider increasing "
                 << "spark.gluten.sql.columnar.backend.velox.cudf.pinnedPoolSize";
    // Clear any sticky CUDA error so the legacy path has a chance to succeed.
    cudaGetLastError();
  }

  // Legacy per-buffer fallback (smaller individual allocations).
  auto devArray = cudf::to_arrow_device(table, stream);
  int bufs = countBuffersRecursive(&devArray->array);
  auto& stats = DtoHStats::instance();
  stats.fallbackCalls.fetch_add(1, std::memory_order_relaxed);
  stats.fallbackBuffers.fetch_add(bufs, std::memory_order_relaxed);
  copyDeviceBuffersToPinnedHost(&devArray->array, stream);
  stream.synchronize();
  auto calls = stats.fallbackCalls.load(std::memory_order_relaxed);
  if ((calls & (calls - 1)) == 0 || (calls % 500) == 0) {
    stats.dump("packed-fallback-to-legacy");
  }
  devArray->device_type = ARROW_DEVICE_CPU;
  return devArray;
}

} // namespace facebook::velox::cudf_velox
