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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
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

#include <cstdlib>
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <limits>
#include <memory>
#include <string_view>
#include <unistd.h>

namespace facebook::velox::cudf_velox {

void checkCudaOperationError(
    rmm::cuda_stream_view stream,
    const char* context) {
  auto peekErr = cudaPeekAtLastError();
  if (peekErr != cudaSuccess) {
    auto errStr = cudaGetErrorString(peekErr);
    cudaGetLastError(); // clear the sticky error
    VELOX_FAIL(
        "CUDA error detected before {} : {} (code {}). "
        "A prior GPU operation produced an asynchronous error.",
        context,
        errStr,
        static_cast<int>(peekErr));
  }
  auto syncErr = cudaStreamSynchronize(stream.value());
  if (syncErr != cudaSuccess) {
    auto errStr = cudaGetErrorString(syncErr);
    cudaGetLastError();
    VELOX_FAIL(
        "CUDA stream sync error at {} : {} (code {}). "
        "A GPU kernel launched on this stream failed.",
        context,
        errStr,
        static_cast<int>(syncErr));
  }
}

namespace {

struct OldSignalHandlers {
  struct sigaction oldSigsegv;
  struct sigaction oldSigabrt;
  struct sigaction oldSigbus;
};

static OldSignalHandlers oldHandlers;

void fatalSignalHandler(int sig, siginfo_t* info, void* ucontext) {
  const char* sigName = "UNKNOWN";
  if (sig == SIGSEGV) sigName = "SIGSEGV";
  else if (sig == SIGABRT) sigName = "SIGABRT";
  else if (sig == SIGBUS) sigName = "SIGBUS";

  char buf[256];
  int len = snprintf(
      buf, sizeof(buf),
      "\n=== Fatal signal %s (%d) in Velox/cuDF native code ===\n"
      "Fault address: %p\n",
      sigName, sig, info ? info->si_addr : nullptr);
  write(STDERR_FILENO, buf, len);

  void* frames[128];
  int nframes = backtrace(frames, 128);
  backtrace_symbols_fd(frames, nframes, STDERR_FILENO);

  const char* footer = "=== End fatal signal trace ===\n";
  write(STDERR_FILENO, footer, strlen(footer));

  // Restore original handler and re-raise so the JVM / default handler runs.
  struct sigaction* old = nullptr;
  if (sig == SIGSEGV) old = &oldHandlers.oldSigsegv;
  else if (sig == SIGABRT) old = &oldHandlers.oldSigabrt;
  else if (sig == SIGBUS) old = &oldHandlers.oldSigbus;

  if (old) {
    sigaction(sig, old, nullptr);
  } else {
    signal(sig, SIG_DFL);
  }
  raise(sig);
}

} // anonymous namespace

void installFatalSignalHandler() {
  static bool installed = false;
  if (installed) return;
  installed = true;

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = fatalSignalHandler;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigemptyset(&sa.sa_mask);

  sigaction(SIGSEGV, &sa, &oldHandlers.oldSigsegv);
  sigaction(SIGABRT, &sa, &oldHandlers.oldSigabrt);
  sigaction(SIGBUS, &sa, &oldHandlers.oldSigbus);
}

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

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
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

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType);
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
    // Zero-copy: safe because buffers are on `stream`, so any later
    // cudaFreeAsync is ordered with work on the same stream.
    return tables[0]->release();
  }

  // For multiple tables, or a single table whose buffers live on a different
  // stream, materialize on `stream`.  This ensures the returned table's
  // device_buffers are allocated (and later freed) on `stream`, avoiding
  // cross-stream free-vs-read races when the caller uses the table on
  // `stream` and the buffers would otherwise be freed on the original stream.
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
    // Zero-copy: safe because buffers are on `stream`.
    concatTables.push_back(tables[0]->release());
    return concatTables;
  }

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  auto const maxRows =
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
  size_t startpos = 0;
  size_t runningRows = 0;
  for (size_t i = 0; i < tableViews.size(); ++i) {
    auto const numRows = static_cast<size_t>(tableViews[i].num_rows());
    // If adding this table would exceed the limit, flush current batch
    // [startpos, i).
    if (runningRows > 0 && runningRows + numRows > maxRows) {
      outputTables.push_back(
          cudf::concatenate(
              std::vector<cudf::table_view>(
                  tableViews.begin() + startpos, tableViews.begin() + i),
              stream,
              cudf::get_current_device_resource_ref()));
      startpos = i;
      runningRows = 0;
    }
    runningRows += numRows;
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
