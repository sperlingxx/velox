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

#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <glog/logging.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>

#include <common/base/Exceptions.h>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <limits>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

namespace {
/// \brief Makes a cuda resource
[[nodiscard]] auto makeCudaMr() {
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

/// \brief Makes a pool<cuda> resource
[[nodiscard]] auto makePoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes an async resource
[[nodiscard]] auto makeAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

/// \brief Makes a managed resource
[[nodiscard]] auto makeManagedMr() {
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

/// \brief Makes a prefetch<managed> resource
[[nodiscard]] auto makePrefetchManagedMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedMr());
}

/// \brief Makes an arena<cuda> resource
[[nodiscard]] auto makeArenaMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(
      makeCudaMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a pool<managed> resource
[[nodiscard]] auto makeManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      makeManagedMr(), rmm::percent_of_free_device_memory(percent));
}

/// \brief Makes a prefetch<pool<managed>> resource
[[nodiscard]] auto makePrefetchManagedPoolMr(int percent) {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedPoolMr(percent));
}

/// \brief Makes a managed_async resource (experimental, requires CUDA 13+)
[[nodiscard]] auto makeManagedAsyncMr() {
  return std::make_shared<rmm::mr::cuda_async_managed_memory_resource>();
}

/// \brief Makes a prefetch<managed_async> resource (experimental, requires CUDA
/// 13+)
[[nodiscard]] auto makePrefetchManagedAsyncMr() {
  return rmm::mr::make_owning_wrapper<rmm::mr::prefetch_resource_adaptor>(
      makeManagedAsyncMr());
}

void enablePrefetching() {
  cudf::prefetch::enable();
}

} // namespace

std::shared_ptr<rmm::mr::device_memory_resource> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda")
    return makeCudaMr();
  if (mode == "pool")
    return makePoolMr(percent);
  if (mode == "async")
    return makeAsyncMr();
  if (mode == "arena")
    return makeArenaMr(percent);
  if (mode == "managed")
    return makeManagedMr();
  if (mode == "managed_pool")
    return makeManagedPoolMr(percent);
  if (mode == "managed_async")
    return makeManagedAsyncMr();
  if (mode == "prefetch_managed") {
    enablePrefetching();
    return makePrefetchManagedMr();
  }
  if (mode == "prefetch_managed_pool") {
    enablePrefetching();
    return makePrefetchManagedPoolMr(percent);
  }
  if (mode == "prefetch_managed_async") {
    enablePrefetching();
    return makePrefetchManagedAsyncMr();
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

namespace {
uint64_t estimateColumnViewBytes(cudf::column_view const& col) {
  uint64_t bytes = 0;
  if (col.size() > 0 && cudf::is_fixed_width(col.type())) {
    bytes += static_cast<uint64_t>(cudf::size_of(col.type())) * col.size();
  }
  if (col.nullable()) {
    bytes += cudf::bitmask_allocation_size_bytes(col.size());
  }
  for (int i = 0; i < col.num_children(); ++i) {
    bytes += estimateColumnViewBytes(col.child(i));
  }
  return bytes;
}
} // namespace

uint64_t estimateTableBytes(std::unique_ptr<cudf::table>& table) {
  if (!table || table->num_columns() == 0 || table->num_rows() == 0) {
    return 0;
  }
  uint64_t totalBytes = 0;
  auto view = table->view();
  for (int i = 0; i < view.num_columns(); ++i) {
    totalBytes += estimateColumnViewBytes(view.column(i));
  }
  return totalBytes;
}

namespace {
void alignDecimalColumnsForConcat(
    std::vector<cudf::table_view>& tableViews,
    std::vector<std::unique_ptr<cudf::table>>& castStorage,
    rmm::cuda_stream_view stream);
} // namespace

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });

  std::vector<std::unique_ptr<cudf::table>> castStorage;
  alignDecimalColumnsForConcat(tableViews, castStorage, stream);

  return cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::table> makeEmptyTable(TypePtr const& inputType) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    if (auto const& childType = inputType->childAt(i);
        childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType);
      auto structColumn = std::make_unique<cudf::column>(
          cudf::data_type(cudf::type_id::STRUCT),
          0,
          rmm::device_buffer(),
          rmm::device_buffer(),
          0,
          tbl->release());
      emptyColumns.push_back(std::move(structColumn));
    } else {
      auto emptyColumn = cudf::make_empty_column(
          cudf_velox::veloxToCudfDataType(inputType->childAt(i)));
      emptyColumns.push_back(std::move(emptyColumn));
    }
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

namespace {

// Align decimal column types across tables so cudf::concatenate succeeds.
// When batches have mismatched decimal types (e.g., DECIMAL64 vs DECIMAL128
// for the same column), cast all to the widest type. Stores cast columns
// in 'castStorage' to keep them alive, and updates 'tableViews' in place.
void alignDecimalColumnsForConcat(
    std::vector<cudf::table_view>& tableViews,
    std::vector<std::unique_ptr<cudf::table>>& castStorage,
    rmm::cuda_stream_view stream) {
  if (tableViews.size() <= 1) return;
  auto numCols = tableViews[0].num_columns();
  if (numCols == 0) return;

  // For each column, find the widest fixed_point type across all tables.
  std::vector<cudf::data_type> targetTypes(numCols);
  bool needsCast = false;
  for (cudf::size_type c = 0; c < numCols; ++c) {
    targetTypes[c] = tableViews[0].column(c).type();
    for (size_t t = 1; t < tableViews.size(); ++t) {
      auto colType = tableViews[t].column(c).type();
      if (colType == targetTypes[c]) continue;
      if (!cudf::is_fixed_point(colType) ||
          !cudf::is_fixed_point(targetTypes[c])) continue;
      needsCast = true;
      // Pick wider type_id (DECIMAL128 > DECIMAL64 > DECIMAL32) and
      // finer scale (more negative = more fractional digits).
      auto widerId = std::max(targetTypes[c].id(), colType.id());
      auto finerScale = std::min(targetTypes[c].scale(), colType.scale());
      targetTypes[c] = cudf::data_type{widerId, finerScale};
    }
  }
  if (!needsCast) return;

  auto mr = cudf::get_current_device_resource_ref();
  for (size_t t = 0; t < tableViews.size(); ++t) {
    bool tableNeedsCast = false;
    for (cudf::size_type c = 0; c < numCols; ++c) {
      if (cudf::is_fixed_point(tableViews[t].column(c).type()) &&
          tableViews[t].column(c).type() != targetTypes[c]) {
        tableNeedsCast = true;
        break;
      }
    }
    if (!tableNeedsCast) continue;

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(numCols);
    for (cudf::size_type c = 0; c < numCols; ++c) {
      auto colView = tableViews[t].column(c);
      if (cudf::is_fixed_point(colView.type()) &&
          colView.type() != targetTypes[c]) {
        cols.push_back(cudf::cast(colView, targetTypes[c], stream, mr));
      } else {
        cols.push_back(std::make_unique<cudf::column>(colView, stream, mr));
      }
    }
    auto newTable = std::make_unique<cudf::table>(std::move(cols));
    tableViews[t] = newTable->view();
    castStorage.push_back(std::move(newTable));
  }
}

} // namespace

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream) {
  if (tables.size() == 0) {
    LOG(INFO) << "[DIAG] getConcatenatedTable: 0 tables, returning empty";
    return makeEmptyTable(tableType);
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  size_t totalRows = 0;
  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
    totalRows += table->size();
  }
  LOG(INFO) << "[DIAG] getConcatenatedTable: nTables=" << tables.size()
            << " totalRows=" << totalRows
            << " cols=" << (tableViews.empty() ? 0 : tableViews[0].num_columns());

  cudf::detail::join_streams(inputStreams, stream);

  if (tables.size() == 1 && inputStreams[0] == stream) {
    return tables[0]->release();
  }

  std::vector<std::unique_ptr<cudf::table>> castStorage;
  alignDecimalColumnsForConcat(tableViews, castStorage, stream);

  auto output = cudf::concatenate(
      tableViews, stream, cudf::get_current_device_resource_ref());
  stream.synchronize();
  return output;
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType));
    return concatTables;
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  if (tables.size() == 1 && inputStreams[0] == stream) {
    concatTables.push_back(tables[0]->release());
    return concatTables;
  }

  std::vector<std::unique_ptr<cudf::table>> castStorage;
  alignDecimalColumnsForConcat(tableViews, castStorage, stream);

  // Estimate bytes per row from the first table's schema.
  size_t bytesPerRow = 0;
  {
    auto tv0 = tableViews[0];
    for (cudf::size_type c = 0; c < tv0.num_columns(); ++c) {
      auto dt = tv0.column(c).type();
      if (cudf::is_fixed_width(dt)) {
        bytesPerRow += cudf::size_of(dt);
      } else {
        bytesPerRow += 32; // estimate for variable-width types
      }
    }
    bytesPerRow = std::max(bytesPerRow, size_t(1));
  }

  // Limit batches by both row count and byte size. Query free GPU memory
  // and cap each batch to ~40% of free memory to avoid oversized allocations
  // that trigger cudaErrorInvalidValue.
  size_t maxBatchBytes = std::numeric_limits<size_t>::max();
  {
    size_t freeMem = 0, totalMem = 0;
    if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess && freeMem > 0) {
      maxBatchBytes = freeMem * 40 / 100;
    }
  }

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  auto const maxRows =
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
  size_t startpos = 0;
  size_t runningRows = 0;
  size_t runningBytes = 0;
  for (size_t i = 0; i < tableViews.size(); ++i) {
    auto const numRows = static_cast<size_t>(tableViews[i].num_rows());
    auto const numBytes = numRows * bytesPerRow;
    // If adding this table would exceed either limit, flush current batch
    // [startpos, i).
    if (runningRows > 0 &&
        (runningRows + numRows > maxRows ||
         runningBytes + numBytes > maxBatchBytes)) {
      outputTables.push_back(
          cudf::concatenate(
              std::vector<cudf::table_view>(
                  tableViews.begin() + startpos, tableViews.begin() + i),
              stream,
              cudf::get_current_device_resource_ref()));
      startpos = i;
      runningRows = 0;
      runningBytes = 0;
    }
    runningRows += numRows;
    runningBytes += numBytes;
  }
  // Flush the final batch [startpos, end).
  if (startpos < tableViews.size()) {
    outputTables.push_back(
        cudf::concatenate(
            std::vector<cudf::table_view>(
                tableViews.begin() + startpos, tableViews.end()),
            stream,
            cudf::get_current_device_resource_ref()));
  }
  stream.synchronize();
  return outputTables;
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  cudaEventCreateWithFlags(&ev, flags);
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(rmm::cuda_stream_view stream) const {
  cudaEventRecord(event_, stream.value());
  return *this;
}

const CudaEvent& CudaEvent::waitOn(rmm::cuda_stream_view stream) const {
  cudaStreamWaitEvent(stream.value(), event_, 0);
  return *this;
}

} // namespace facebook::velox::cudf_velox
