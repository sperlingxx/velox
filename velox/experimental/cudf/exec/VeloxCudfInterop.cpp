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

#include "velox/experimental/cudf/exec/PinnedArrowInterop.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

#include <cstdlib>
#include <cstring>

#include <atomic>


namespace facebook::velox::cudf_velox {

namespace {
struct PinnedTransferStats {
  std::atomic<uint64_t> htodCalls{0};
  std::atomic<uint64_t> htodBytes{0};
  std::atomic<uint64_t> dtohCalls{0};
  std::atomic<uint64_t> dtohBytes{0};
};
PinnedTransferStats& pinnedStats() {
  static PinnedTransferStats stats;
  return stats;
}
} // namespace

cudf::type_id veloxToCudfTypeId(const TypePtr& type) {
  // Legacy helper retained for compatibility. Note: returning cudf::type_id
  // discards decimal scale; prefer veloxToCudfDataType when scale matters.
  return veloxToCudfDataType(type).id();
}

cudf::data_type veloxToCudfDataType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::data_type{cudf::type_id::BOOL8};
    case TypeKind::TINYINT:
      return cudf::data_type{cudf::type_id::INT8};
    case TypeKind::SMALLINT:
      return cudf::data_type{cudf::type_id::INT16};
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
      }
      return cudf::data_type{cudf::type_id::INT32};
    case TypeKind::BIGINT:
      // BIGINT is used for both INT64 and DECIMAL64
      if (type->isDecimal()) {
        auto const decimalType =
            std::dynamic_pointer_cast<const ShortDecimalType>(type);
        VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
        auto const cudfScale = numeric::scale_type{-decimalType->scale()};
        return cudf::data_type{cudf::type_id::DECIMAL64, cudfScale};
      }
      return cudf::data_type{cudf::type_id::INT64};
    case TypeKind::HUGEINT: {
      // HUGEINT is used only for DECIMAL128
      // per facebookincubator/velox PR 4434 (May 2, 2023)
      // although see commented-out HUGEINT -> DURATION_DAYS below
      VELOX_CHECK(
          type->isDecimal(), "HUGEINT should only be used for DECIMAL128");
      auto const decimalType =
          std::dynamic_pointer_cast<const LongDecimalType>(type);
      VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
      auto const cudfScale = numeric::scale_type{-decimalType->scale()};
      return cudf::data_type{cudf::type_id::DECIMAL128, cudfScale};
    }
    case TypeKind::REAL:
      return cudf::data_type{cudf::type_id::FLOAT32};
    case TypeKind::DOUBLE:
      return cudf::data_type{cudf::type_id::FLOAT64};
    case TypeKind::VARCHAR:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::VARBINARY:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::TIMESTAMP:
      return cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    case TypeKind::ARRAY:
      return cudf::data_type{cudf::type_id::LIST};
    case TypeKind::ROW:
      return cudf::data_type{cudf::type_id::STRUCT};
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      break;
  }
  CUDF_FAIL(
      "Unsupported Velox type: " +
      std::string(TypeKindName::toName(type->kind())));
  return cudf::data_type{cudf::type_id::EMPTY};
}

namespace with_arrow {

namespace {

/// Compute the byte size of the i-th buffer in a C Data Interface ArrowArray,
/// given the Arrow format string from ArrowSchema.
/// `buffersOnHost` must be true when buffers are readable from CPU (HtoD path).
int64_t arrowBufferSize(
    const ArrowArray* array,
    const char* format,
    int bufIdx,
    bool buffersOnHost) {
  const int64_t len = array->length + array->offset;

  // Buffer 0: validity bitmap for all types.
  if (bufIdx == 0) {
    if (array->null_count == 0 && array->buffers[0] == nullptr) {
      return 0;
    }
    return (len + 7) / 8;
  }

  switch (format[0]) {
    case 'n':
      return 0; // null type
    case 'b':
      return (bufIdx == 1) ? (len + 7) / 8 : 0; // bool (bit-packed)
    case 'c':
    case 'C':
      return (bufIdx == 1) ? len : 0; // int8/uint8
    case 's':
    case 'S':
      return (bufIdx == 1) ? len * 2 : 0; // int16/uint16
    case 'i':
    case 'I':
      return (bufIdx == 1) ? len * 4 : 0; // int32/uint32
    case 'l':
    case 'L':
      return (bufIdx == 1) ? len * 8 : 0; // int64/uint64
    case 'e':
      return (bufIdx == 1) ? len * 2 : 0; // float16
    case 'f':
      return (bufIdx == 1) ? len * 4 : 0; // float32
    case 'g':
      return (bufIdx == 1) ? len * 8 : 0; // float64
    case 'u': // utf8
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int32_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int32_t*>(array->buffers[1])[len];
      }
      return 0;
    case 'U': // large utf8
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int64_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int64_t*>(array->buffers[1])[len];
      }
      return 0;
    case 'z': // binary
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int32_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int32_t*>(array->buffers[1])[len];
      }
      return 0;
    case 't': // temporal
      if (format[1] == 'd' && format[2] == 'D')
        return (bufIdx == 1) ? len * 4 : 0; // date32
      if (format[1] == 'd' && format[2] == 'm')
        return (bufIdx == 1) ? len * 8 : 0; // date64
      return (bufIdx == 1) ? len * 8 : 0; // timestamp variants
    case '+': // nested
      if (format[1] == 's')
        return 0; // struct: only validity (buf 0)
      if (format[1] == 'l') // list
        return (bufIdx == 1) ? static_cast<int64_t>(len + 1) * sizeof(int32_t)
                             : 0;
      if (format[1] == 'L') // large list
        return (bufIdx == 1) ? static_cast<int64_t>(len + 1) * sizeof(int64_t)
                             : 0;
      return 0;
    case 'd': // decimal128
      return (bufIdx == 1) ? len * 16 : 0;
    default:
      return 0;
  }
}

/// Recursively copy pageable ArrowArray buffers to pinned host memory.
/// The ArrowArray buffer pointers are replaced in-place.  The PinnedHostBuffers
/// are kept alive via `pinnedBufs` — caller must not clear it until
/// cudf::from_arrow + stream sync is done.
void pinArrowBuffersRecursive(
    ArrowArray* array,
    ArrowSchema* schema,
    std::vector<std::shared_ptr<PinnedHostBuffer>>& pinnedBufs) {
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    if (array->buffers[i] == nullptr) {
      continue;
    }
    int64_t size = arrowBufferSize(array, schema->format, i, /*buffersOnHost=*/true);
    if (size <= 0) {
      continue;
    }
    auto pinned = std::make_shared<PinnedHostBuffer>(size);
    std::memcpy(pinned->data(), array->buffers[i], size);
    array->buffers[i] = pinned->data();
    pinnedBufs.push_back(std::move(pinned));
  }
  for (int64_t i = 0; i < array->n_children; ++i) {
    pinArrowBuffersRecursive(
        array->children[i], schema->children[i], pinnedBufs);
  }
  if (array->dictionary && schema->dictionary) {
    pinArrowBuffersRecursive(
        array->dictionary, schema->dictionary, pinnedBufs);
  }
}

/// Result of an async HtoD transfer. The caller must call sync() or let the
/// owning batch function handle synchronization before the pinned buffers and
/// Arrow resources are released.
struct AsyncHtoD {
  std::unique_ptr<cudf::table> table;
  std::vector<std::shared_ptr<PinnedHostBuffer>> pinnedBufs;
  ArrowArray arrowArray{};
  ArrowSchema arrowSchema{};

  AsyncHtoD() = default;
  AsyncHtoD(const AsyncHtoD&) = delete;
  AsyncHtoD& operator=(const AsyncHtoD&) = delete;
  AsyncHtoD(AsyncHtoD&& o) noexcept
      : table(std::move(o.table)),
        pinnedBufs(std::move(o.pinnedBufs)),
        arrowArray(o.arrowArray),
        arrowSchema(o.arrowSchema) {
    o.arrowArray.release = nullptr;
    o.arrowSchema.release = nullptr;
  }

  // Prevent double-exception crash: if bad_alloc is in flight and a
  // destructor here throws (e.g. CUDA error freeing device memory),
  // std::terminate would be called.  Swallow secondary exceptions.
  ~AsyncHtoD() noexcept {
    try { releaseArrow(); } catch (...) {}
    try { pinnedBufs.clear(); } catch (...) {}
    try { table.reset(); } catch (...) {}
  }

  void releaseArrow() {
    pinnedBufs.clear();
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
    if (arrowSchema.release) {
      arrowSchema.release(&arrowSchema);
    }
  }
};

/// Reconcile a RowVector's declared RowType with its actual children's types.
/// After shuffle deserialization, a RowVector may have children whose physical
/// types differ from the declared type (e.g. decimal BIGINT vs HUGEINT when
/// the GPU produced DECIMAL128 but the Spark schema expects smaller precision).
/// BaseVector::create + copy requires matching typeKinds, so we reconstruct
/// the RowType from the actual children when a mismatch is detected.
facebook::velox::TypePtr reconcileRowType(
    const facebook::velox::RowVectorPtr& rv) {
  auto declaredType = rv->type();
  if (declaredType->kind() != facebook::velox::TypeKind::ROW) {
    return declaredType;
  }
  auto rowType =
      std::dynamic_pointer_cast<const facebook::velox::RowType>(declaredType);
  bool needsFix = false;
  for (size_t i = 0; i < rowType->size(); ++i) {
    auto child = rv->childAt(i);
    if (child && child->typeKind() != rowType->childAt(i)->kind()) {
      needsFix = true;
      break;
    }
  }
  if (!needsFix) {
    return declaredType;
  }
  std::vector<std::string> names;
  std::vector<facebook::velox::TypePtr> types;
  names.reserve(rowType->size());
  types.reserve(rowType->size());
  for (size_t i = 0; i < rowType->size(); ++i) {
    names.push_back(rowType->nameOf(i));
    auto child = rv->childAt(i);
    types.push_back(child ? child->type() : rowType->childAt(i));
  }
  return facebook::velox::ROW(std::move(names), std::move(types));
}

/// Issues the from_arrow transfer on `stream` but does NOT synchronize.
/// The returned AsyncHtoD keeps pinned buffers alive; caller must sync
/// the stream before destroying it.
AsyncHtoD toCudfTableNoSync(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  // Flatten nested encodings (e.g. dictionary-of-dictionary) that the Arrow
  // bridge cannot export.  BaseVector::copy always produces flat output.
  // Use reconciled type so that the target matches actual children's physical
  // types (handles decimal BIGINT/HUGEINT mismatches from shuffle).
  auto targetType = reconcileRowType(veloxTable);
  auto flat = facebook::velox::BaseVector::create<facebook::velox::RowVector>(
      targetType, veloxTable->size(), pool);
  flat->copy(veloxTable.get(), 0, 0, veloxTable->size());

  AsyncHtoD result;
  ArrowOptions arrowOptions{
      .flattenDictionary = true,
      .flattenConstant = true,
      .exportVarbinaryAsString = true};
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(flat),
      result.arrowArray,
      pool,
      arrowOptions);
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(flat),
      result.arrowSchema,
      arrowOptions);

  pinArrowBuffersRecursive(
      &result.arrowArray, &result.arrowSchema, result.pinnedBufs);

  uint64_t pinnedBytes = 0;
  for (const auto& b : result.pinnedBufs) {
    pinnedBytes += b->size();
  }
  auto& stats = pinnedStats();
  auto callNum = stats.htodCalls.fetch_add(1, std::memory_order_relaxed) + 1;
  stats.htodBytes.fetch_add(pinnedBytes, std::memory_order_relaxed);
  if ((callNum & (callNum - 1)) == 0) {
    fprintf(
        stderr,
        "[PinnedDMA] HtoD: calls=%lu thisBytes=%lu totalBytes=%lu\n",
        (unsigned long)callNum,
        (unsigned long)pinnedBytes,
        (unsigned long)stats.htodBytes.load(std::memory_order_relaxed));
  }

  result.table =
      cudf::from_arrow(&result.arrowSchema, &result.arrowArray, stream);
  return result;
}

} // anonymous namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  auto async = toCudfTableNoSync(veloxTable, pool, stream);
  stream.synchronize();
  async.releaseArrow();
  return std::move(async.table);
}

std::unique_ptr<cudf::table> toCudfTableBatched(
    const std::vector<facebook::velox::RowVectorPtr>& batches,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  if (batches.empty()) {
    return nullptr;
  }
  if (batches.size() == 1) {
    return toCudfTable(batches[0], pool, stream);
  }

  // Issue all from_arrow calls on the same stream without syncing.
  std::vector<AsyncHtoD> pending;
  pending.reserve(batches.size());
  for (const auto& batch : batches) {
    pending.push_back(toCudfTableNoSync(batch, pool, stream));
  }

  // Build table views for GPU-side concatenation.
  // No sync needed here: concatenation is enqueued on the same stream as
  // from_arrow transfers, so CUDA stream ordering guarantees the data is
  // available. Pinned host buffers stay alive until releaseArrow() below.
  std::vector<cudf::table_view> views;
  views.reserve(pending.size());
  for (auto& p : pending) {
    views.push_back(p.table->view());
  }

  auto concatenated = cudf::concatenate(views, stream);

  // Synchronize before releasing source tables: cudf::concatenate is async
  // and reads from the source table device buffers. Without this sync, the
  // source tables' GPU memory could be freed (when `pending` goes out of
  // scope) while the concatenation kernel is still reading from them.
  stream.synchronize();

  for (auto& p : pending) {
    p.releaseArrow();
  }

  return concatenated;
}

namespace {

void setArrowSchemaFormat(ArrowSchema* schema, const char* format) {
  if (!schema) {
    return;
  }
  if (schema->format != nullptr) {
    std::free(const_cast<char*>(schema->format));
    schema->format = nullptr;
  }
  if (format != nullptr) {
    const size_t size = std::strlen(format) + 1;
    auto* buffer = static_cast<char*>(std::malloc(size));
    VELOX_CHECK_NOT_NULL(buffer);
    std::memcpy(buffer, format, size);
    schema->format = buffer;
  }
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    const RowTypePtr* expectedType,
    rmm::cuda_stream_view stream) {
  auto arrowDeviceArray = pinnedToArrowHost(table, stream);
  ArrowArray arrayCopy = arrowDeviceArray->array;
  arrowDeviceArray->array.release = nullptr;

  auto& stats = pinnedStats();
  auto callNum = stats.dtohCalls.fetch_add(1, std::memory_order_relaxed) + 1;
  int64_t rows = arrayCopy.length;
  if ((callNum & (callNum - 1)) == 0) {
    LOG(WARNING) << "Pinned DtoH: calls=" << callNum << " rows=" << rows;
  }

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  ArrowSchema schemaCopy = *arrowSchema;
  arrowSchema->release = nullptr;

  if (expectedType) {
    auto applyExpectedArrowFormat =
        [&](auto&& self, ArrowSchema* schema, const TypePtr& type) -> void {
      if (!schema || !schema->format) {
        return;
      }
      switch (type->kind()) {
        case TypeKind::ROW: {
          if (schema->n_children != static_cast<int64_t>(type->size())) {
            return;
          }
          for (size_t i = 0; i < type->size(); ++i) {
            self(self, schema->children[i], type->childAt(i));
          }
          return;
        }
        case TypeKind::ARRAY: {
          if (schema->n_children < 1) {
            return;
          }
          self(self, schema->children[0], type->childAt(0));
          return;
        }
        case TypeKind::MAP: {
          if (schema->n_children < 1) {
            return;
          }
          auto* entry = schema->children[0];
          if (!entry || entry->n_children < 2) {
            return;
          }
          self(self, entry->children[0], type->childAt(0));
          self(self, entry->children[1], type->childAt(1));
          return;
        }
        case TypeKind::VARBINARY: {
          setArrowSchemaFormat(schema, "z");
          return;
        }
        default:
          return;
      }
    };
    applyExpectedArrowFormat(
        applyExpectedArrowFormat, &schemaCopy, *expectedType);
  }

  auto veloxTable = importFromArrowAsOwner(schemaCopy, arrayCopy, pool);
  auto castedPtr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(castedPtr);
  return castedPtr;
}

template <typename Iterator>
std::vector<cudf::column_metadata>
getMetadata(Iterator begin, Iterator end, const std::string& namePrefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(namePrefix + std::to_string(i)));
    metadata.back().children_meta = getMetadata(
        c->child_begin(), c->child_end(), namePrefix + std::to_string(i));
    i++;
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, nullptr, stream);
}

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    const facebook::velox::RowTypePtr& expectedType,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref /*mr*/) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, &expectedType, stream);
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return toVeloxColumn(table, pool, metadata, nullptr, stream);
}

} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
