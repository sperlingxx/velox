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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/OperatorUtils.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/merge.hpp>
#include <cudf/reduction.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>

namespace facebook::velox::cudf_velox {
namespace {

// A sorted run is deliberately much larger than an exchange/read batch.  On a
// 32 GiB device this leaves enough room for concatenate + stable sort while
// avoiding the thousands of tiny spill files produced by hash bucketing.
constexpr uint64_t kWindowSortedRunBytes = 3ULL << 30;
constexpr uint64_t kWindowMergeChunkBytes = 256ULL << 20;
std::atomic<uint64_t> windowSpillDirectorySequence{0};

bool isSupportedScalarWindowType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      return true;
    default:
      return false;
  }
}

bool isFieldAccessExpr(const core::TypedExprPtr& expr) {
  return std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr) !=
      nullptr;
}

bool isUnboundedPrecedingToCurrentRowFrame(
    const core::WindowNode::Frame& frame) {
  return frame.startType == core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kCurrentRow &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isFullPartitionRowsFrame(const core::WindowNode::Frame& frame) {
  return frame.type == core::WindowNode::WindowType::kRows &&
      frame.startType == core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kUnboundedFollowing &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isSupportedRankLikeFunction(const core::WindowNode::Function& function) {
  const auto& name = function.functionCall->name();
  if ((name != "row_number" && name != "rank") || function.ignoreNulls ||
      !function.functionCall->inputs().empty() ||
      !isUnboundedPrecedingToCurrentRowFrame(function.frame)) {
    return false;
  }

  const auto resultKind = function.functionCall->type()->kind();
  return resultKind == TypeKind::INTEGER || resultKind == TypeKind::BIGINT;
}

bool isSupportedSumInputType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
      return true;
    default:
      return false;
  }
}

bool isSupportedFullPartitionSumFunction(
    const core::WindowNode::Function& function) {
  if (function.functionCall->name() != "sum" || function.ignoreNulls ||
      function.functionCall->inputs().size() != 1 ||
      !isFullPartitionRowsFrame(function.frame)) {
    return false;
  }

  const auto& input = function.functionCall->inputs()[0];
  return isFieldAccessExpr(input) && isSupportedSumInputType(input->type());
}

bool isSupportedRunningSumFunction(const core::WindowNode::Function& function) {
  if (function.functionCall->name() != "sum" || function.ignoreNulls ||
      function.functionCall->inputs().size() != 1 ||
      function.frame.type != core::WindowNode::WindowType::kRows ||
      !isUnboundedPrecedingToCurrentRowFrame(function.frame)) {
    return false;
  }
  const auto& input = function.functionCall->inputs()[0];
  return isFieldAccessExpr(input) && isSupportedSumInputType(input->type());
}

bool isSupportedFirstValueFunction(const core::WindowNode::Function& function) {
  const auto& name = function.functionCall->name();
  if ((name != "first" && name != "first_value") || function.ignoreNulls ||
      function.functionCall->inputs().size() != 1 ||
      !isUnboundedPrecedingToCurrentRowFrame(function.frame)) {
    return false;
  }

  const auto& input = function.functionCall->inputs()[0];
  return isFieldAccessExpr(input) && isSupportedScalarWindowType(input->type());
}

std::vector<cudf::column_view> selectColumns(
    cudf::table_view const& table,
    const std::vector<cudf::size_type>& channels) {
  std::vector<cudf::column_view> columns;
  columns.reserve(channels.size());
  for (const auto channel : channels) {
    columns.push_back(table.column(channel));
  }
  return columns;
}

std::unique_ptr<cudf::table> copyTableSlice(
    cudf::table_view input,
    cudf::size_type begin,
    cudf::size_type end,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LE(begin, end);
  auto slices = cudf::slice(input, {begin, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);
  return std::make_unique<cudf::table>(slices.front(), stream, mr);
}

cudf::size_type firstSearchPosition(
    cudf::column_view positions,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(positions.size(), 1);
  cudf::size_type result{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &result,
      positions.data<cudf::size_type>(),
      sizeof(result),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return result;
}

} // namespace

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node) {
  if (node == nullptr || node->windowFunctions().empty()) {
    return false;
  }

  bool hasRowNumber = false;
  bool hasFullPartitionSum = false;
  bool hasRunningSum = false;
  bool hasFirstValue = false;
  for (const auto& function : node->windowFunctions()) {
    if (isSupportedRankLikeFunction(function)) {
      hasRowNumber = true;
    } else if (isSupportedFullPartitionSumFunction(function)) {
      hasFullPartitionSum = true;
    } else if (isSupportedRunningSumFunction(function)) {
      hasRunningSum = true;
    } else if (isSupportedFirstValueFunction(function)) {
      hasFirstValue = true;
    } else {
      return false;
    }
  }

  if ((hasRowNumber || hasFirstValue || hasRunningSum) &&
      node->sortingKeys().empty()) {
    return false;
  }

  if (hasFirstValue && node->partitionKeys().empty()) {
    return false;
  }

  if ((hasRowNumber || hasFirstValue || hasRunningSum) && hasFullPartitionSum) {
    return false;
  }

  for (const auto& key : node->partitionKeys()) {
    if (!isFieldAccessExpr(key) || !isSupportedScalarWindowType(key->type())) {
      return false;
    }
  }

  for (const auto& key : node->sortingKeys()) {
    if (!isFieldAccessExpr(key) || !isSupportedScalarWindowType(key->type())) {
      return false;
    }
  }

  return true;
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          windowNode->outputType(),
          windowNode->id(),
          "CudfWindow",
          nvtx3::rgb{123, 104, 238},
          NvtxMethodFlag::kAll,
          std::nullopt,
          windowNode),
      windowNode_(windowNode),
      inputType_(windowNode->sources()[0]->outputType()),
      rankLikeStreaming_(
          windowNode->inputsSorted() &&
          !windowNode->partitionKeys().empty() &&
          !windowNode->sortingKeys().empty() &&
          std::all_of(
              windowNode->windowFunctions().begin(),
              windowNode->windowFunctions().end(),
              isSupportedRankLikeFunction)),
      stateStream_(cudfGlobalStreamPool().get_stream()) {
  for (const auto& key : windowNode->partitionKeys()) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel, "Window partition key must be a column");
    partitionKeyChannels_.push_back(static_cast<cudf::size_type>(channel));
  }

  const auto& sortingKeys = windowNode->sortingKeys();
  const auto& sortingOrders = windowNode->sortingOrders();
  for (size_t i = 0; i < sortingKeys.size(); ++i) {
    const auto channel = exec::exprToChannel(sortingKeys[i].get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel, "Window sorting key must be a column");
    sortKeyChannels_.push_back(static_cast<cudf::size_type>(channel));

    const auto& order = sortingOrders[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    sortNullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfWindow::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

  if (rankLikeStreaming_) {
    VELOX_CHECK_NULL(
        pendingOutput_,
        "CudfWindow received sorted input before prior output was drained");
    const auto inputStream = cudfInput->stream();
    if (inputStream.value() != stateStream_.value()) {
      std::vector<rmm::cuda_stream_view> inputStreams{inputStream};
      cudf::detail::join_streams(inputStreams, stateStream_);
    }
    VELOX_CHECK(
        cudfInput->rebindStream(stateStream_),
        "CudfWindow cannot rebind sorted input to its state stream");

    auto inputTable = cudfInput->release();
    auto output = computeOutputTable(
        std::move(inputTable), stateStream_, get_output_mr(), true);
    if (output->num_rows() > 0) {
      pendingOutput_ = std::make_shared<CudfVector>(
          pool(),
          outputType_,
          output->num_rows(),
          std::move(output),
          stateStream_);
    }
    return;
  }

  bufferedBytes_ += cudfInput->estimateFlatSize();
  inputs_.push_back(std::move(cudfInput));

  if (deviceMemoryDiagnosticsEnabled() &&
      bufferedBytes_ >= nextDiagnosticBufferedBytes_) {
    logDeviceMemorySnapshot(
        fmt::format(
            "operator=CudfWindow node={} state=buffering bufferedBytes={} "
            "bufferedInputs={} spilled={}",
            windowNode_->id(),
            bufferedBytes_,
            inputs_.size(),
            spilled_));
    while (nextDiagnosticBufferedBytes_ <= bufferedBytes_) {
      nextDiagnosticBufferedBytes_ += 512ULL << 20;
    }
  }

  // Build independently sorted runs.  After noMoreInput() these runs are read
  // in chunks and order-merged, matching Velox CPU SortWindowBuild's external
  // sort architecture.
  if (!windowNode_->inputsSorted() && !partitionKeyChannels_.empty() &&
      bufferedBytes_ >= kWindowSortedRunBytes) {
    spillSortedRun();
  }
}

void CudfWindow::doNoMoreInput() {
  Operator::noMoreInput();
  if (rankLikeStreaming_) {
    finished_ = true;
    return;
  }

  if (spilled_ && !inputs_.empty()) {
    spillSortedRun();
  }
  if (spilled_) {
    initializeSortedRunReaders();
  }
  if (!spilled_ && inputs_.empty()) {
    finished_ = true;
  }
}

RowVectorPtr CudfWindow::doGetOutput() {
  if (pendingOutput_ != nullptr) {
    return std::exchange(pendingOutput_, nullptr);
  }

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (spilled_) {
    auto result = computeNextSortedOutput();
    if (result != nullptr) {
      return result;
    }
    finished_ = true;
    cleanupSpillFiles();
    return nullptr;
  }

  if (inputs_.empty()) {
    finished_ = true;
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  auto input = getConcatenatedTable(std::move(inputs_), inputType_, stream, mr);
  inputs_.clear();
  bufferedBytes_ = 0;

  auto output = computeOutputTable(std::move(input), stream, mr);
  finished_ = true;
  if (output->num_rows() == 0) {
    return nullptr;
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, output->num_rows(), std::move(output), stream);
}

void CudfWindow::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }
  VELOX_CHECK(
      !partitionKeyChannels_.empty(),
      "CudfWindow spill requires partition keys");

  namespace fs = std::filesystem;
  if (!spilled_) {
    const auto sequence = windowSpillDirectorySequence.fetch_add(1);
    spillDirectory_ = (fs::temp_directory_path() /
                       fmt::format(
                           "velox-cudf-window-spill-{}-{}",
                           static_cast<int64_t>(::getpid()),
                           sequence))
                          .string();
    fs::create_directories(spillDirectory_);
    spilled_ = true;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfWindow node={} state=sortRun.concatenate.begin "
          "bufferedBytes={} bufferedInputs={}",
          windowNode_->id(),
          bufferedBytes_,
          inputs_.size()));
  auto input =
      getConcatenatedTable(std::exchange(inputs_, {}), inputType_, stream, mr);
  bufferedBytes_ = 0;

  std::vector<cudf::size_type> sortChannels = partitionKeyChannels_;
  sortChannels.insert(
      sortChannels.end(), sortKeyChannels_.begin(), sortKeyChannels_.end());
  std::vector<cudf::order> orders(
      partitionKeyChannels_.size(), cudf::order::ASCENDING);
  orders.insert(orders.end(), sortOrders_.begin(), sortOrders_.end());
  std::vector<cudf::null_order> nullOrders(
      partitionKeyChannels_.size(), cudf::null_order::BEFORE);
  nullOrders.insert(
      nullOrders.end(), sortNullOrders_.begin(), sortNullOrders_.end());

  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfWindow node={} state=sortRun.sort.begin rows={}",
          windowNode_->id(),
          input->num_rows()));
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(sortChannels),
      orders,
      nullOrders,
      stream,
      mr);
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfWindow node={} state=sortRun.sort.end rows={}",
          windowNode_->id(),
          input->num_rows()));

  auto path = fmt::format(
      "{}/run-{:06}.parquet", spillDirectory_, spillFileSequence_++);
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{path}, sorted->view())
                     .build();
  cudf::io::write_parquet(options, stream);
  sortedRuns_.push_back({std::move(path), nullptr});
  ::malloc_trim(0);
}

void CudfWindow::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        kWindowMergeChunkBytes, 0, options, stream, mr);
  }
  readersInitialized_ = true;
}

std::unique_ptr<cudf::table> CudfWindow::mergeNextSortedBatch(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finalBatch) {
  finalBatch = false;
  while (!mergeFinished_) {
    std::vector<std::unique_ptr<cudf::table>> chunks;
    std::vector<cudf::table_view> mergeViews;
    std::vector<cudf::table_view> boundaryRows;
    chunks.reserve(sortedRuns_.size());
    mergeViews.reserve(sortedRuns_.size() + (mergeCarry_ ? 1 : 0));
    boundaryRows.reserve(sortedRuns_.size());

    if (mergeCarry_ && mergeCarry_->num_rows() > 0) {
      mergeViews.push_back(mergeCarry_->view());
    }

    for (auto& run : sortedRuns_) {
      if (!run.reader || !run.reader->has_next()) {
        continue;
      }
      auto chunk = run.reader->read_chunk();
      if (chunk.tbl->num_rows() == 0) {
        continue;
      }
      chunks.push_back(std::move(chunk.tbl));
      mergeViews.push_back(chunks.back()->view());
      // Only a run with unread rows can constrain the globally safe prefix.
      // Rows in its next chunk are >= the final row of this chunk.
      if (run.reader->has_next()) {
        auto last = cudf::slice(
            chunks.back()->view(),
            {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
            stream);
        boundaryRows.push_back(last.front());
      }
    }

    if (mergeViews.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return std::exchange(mergeCarry_, nullptr);
    }

    std::vector<cudf::size_type> sortChannels = partitionKeyChannels_;
    sortChannels.insert(
        sortChannels.end(), sortKeyChannels_.begin(), sortKeyChannels_.end());
    std::vector<cudf::order> orders(
        partitionKeyChannels_.size(), cudf::order::ASCENDING);
    orders.insert(orders.end(), sortOrders_.begin(), sortOrders_.end());
    std::vector<cudf::null_order> nullOrders(
        partitionKeyChannels_.size(), cudf::null_order::BEFORE);
    nullOrders.insert(
        nullOrders.end(), sortNullOrders_.begin(), sortNullOrders_.end());

    std::unique_ptr<cudf::table> merged;
    if (mergeViews.size() == 1) {
      merged = std::make_unique<cudf::table>(mergeViews.front(), stream, mr);
    } else {
      merged =
          cudf::merge(mergeViews, sortChannels, orders, nullOrders, stream, mr);
    }
    mergeCarry_.reset();

    if (boundaryRows.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return merged;
    }

    auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
    auto sortedBoundaries = cudf::sort_by_key(
        boundaryCandidates->view(),
        boundaryCandidates->view().select(sortChannels),
        orders,
        nullOrders,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
    // Equal boundary keys are safe to emit: future chunks cannot contain a
    // smaller key and cross-run ordering of peers is not stable/observable.
    auto positions = cudf::upper_bound(
        merged->view().select(sortChannels),
        boundary.front().select(sortChannels),
        orders,
        nullOrders,
        stream,
        mr);
    const auto safeEnd = firstSearchPosition(positions->view(), stream);
    mergeCarry_ =
        copyTableSlice(merged->view(), safeEnd, merged->num_rows(), stream, mr);
    if (safeEnd > 0) {
      return copyTableSlice(merged->view(), 0, safeEnd, stream, mr);
    }
  }
  return nullptr;
}

std::unique_ptr<cudf::table> CudfWindow::takeCompletePartitions(
    std::unique_ptr<cudf::table> sorted,
    bool finalBatch,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!sorted || sorted->num_rows() == 0) {
    return nullptr;
  }
  if (partitionCarry_ && partitionCarry_->num_rows() > 0) {
    std::vector<cudf::table_view> pieces{
        partitionCarry_->view(), sorted->view()};
    sorted = cudf::concatenate(pieces, stream, mr);
    partitionCarry_.reset();
  }
  if (finalBatch || partitionKeyChannels_.empty()) {
    return sorted;
  }

  auto partitionColumns = sorted->view().select(partitionKeyChannels_);
  auto lastPartition = cudf::slice(
      partitionColumns, {sorted->num_rows() - 1, sorted->num_rows()}, stream);
  std::vector<cudf::order> orders(
      partitionKeyChannels_.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> nullOrders(
      partitionKeyChannels_.size(), cudf::null_order::BEFORE);
  auto positions = cudf::lower_bound(
      partitionColumns, lastPartition.front(), orders, nullOrders, stream, mr);
  const auto completeEnd = firstSearchPosition(positions->view(), stream);
  partitionCarry_ = copyTableSlice(
      sorted->view(), completeEnd, sorted->num_rows(), stream, mr);
  if (completeEnd == 0) {
    return nullptr;
  }
  return copyTableSlice(sorted->view(), 0, completeEnd, stream, mr);
}

CudfVectorPtr CudfWindow::computeNextSortedOutput() {
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  while (!mergeFinished_ || mergeCarry_ || partitionCarry_) {
    bool finalBatch = false;
    auto sorted = mergeNextSortedBatch(stream, mr, finalBatch);
    if (finalBatch && partitionCarry_) {
      if (sorted && sorted->num_rows() > 0) {
        std::vector<cudf::table_view> pieces{
            partitionCarry_->view(), sorted->view()};
        sorted = cudf::concatenate(pieces, stream, mr);
      } else {
        sorted = std::exchange(partitionCarry_, nullptr);
      }
      partitionCarry_.reset();
    } else {
      sorted =
          takeCompletePartitions(std::move(sorted), finalBatch, stream, mr);
    }
    if (!sorted || sorted->num_rows() == 0) {
      if (finalBatch) {
        return nullptr;
      }
      continue;
    }
    auto output = computeOutputTable(std::move(sorted), stream, mr, true);
    return std::make_shared<CudfVector>(
        pool(), outputType_, output->num_rows(), std::move(output), stream);
  }
  return nullptr;
}

void CudfWindow::cleanupSpillFiles() {
  sortedRuns_.clear();
  mergeCarry_.reset();
  partitionCarry_.reset();
  if (spillDirectory_.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove_all(spillDirectory_, error);
  spillDirectory_.clear();
  ::malloc_trim(0);
}

void CudfWindow::doClose() {
  if (rankLikeStreaming_) {
    // close() can run while a task exception is unwinding. A poisoned CUDA
    // context must not replace that original failure, but owners are always
    // destroyed between best-effort pre/post drains.
    try {
      stateStream_.synchronize();
    } catch (const std::exception& error) {
      LOG(WARNING) << "CudfWindow pre-destruction cleanup failed: "
                   << error.what();
    }
    pendingOutput_.reset();
    previousPartitionKey_.reset();
    previousOrderKey_.reset();
    try {
      // Destruction above enqueues stream-ordered frees for async RMM.
      stateStream_.synchronize();
    } catch (const std::exception& error) {
      LOG(WARNING) << "CudfWindow post-destruction cleanup failed: "
                   << error.what();
    }
  }
  inputs_.clear();
  cleanupSpillFiles();
  Operator::close();
}

cudf::size_type CudfWindow::continuingPrefixSize(
    cudf::table_view sortedInput,
    const std::vector<cudf::size_type>& channels,
    const std::unique_ptr<cudf::table>& previousKey,
    const std::vector<cudf::order>& orders,
    const std::vector<cudf::null_order>& nullOrders,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (!previousKey || sortedInput.num_rows() == 0) {
    return 0;
  }
  VELOX_CHECK_EQ(channels.size(), orders.size());
  VELOX_CHECK_EQ(channels.size(), nullOrders.size());
  auto positions = cudf::upper_bound(
      cudf::table_view(selectColumns(sortedInput, channels)),
      previousKey->view(),
      orders,
      nullOrders,
      stream,
      mr);
  return firstSearchPosition(positions->view(), stream);
}

std::unique_ptr<cudf::column> CudfWindow::fixRankLikeColumn(
    std::unique_ptr<cudf::column> localResult,
    std::string_view functionName,
    cudf::table_view sortedInput,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (!hasRankLikeState_ || sortedInput.num_rows() == 0) {
    return localResult;
  }

  const std::vector<cudf::order> partitionOrders(
      partitionKeyChannels_.size(), cudf::order::ASCENDING);
  const std::vector<cudf::null_order> partitionNullOrders(
      partitionKeyChannels_.size(), cudf::null_order::BEFORE);
  const auto samePartitionEnd = continuingPrefixSize(
      sortedInput,
      partitionKeyChannels_,
      previousPartitionKey_,
      partitionOrders,
      partitionNullOrders,
      stream,
      mr);
  if (samePartitionEnd == 0) {
    return localResult;
  }

  auto addOffset =
      [&](cudf::column_view values,
          int64_t offset) -> std::unique_ptr<cudf::column> {
    switch (values.type().id()) {
      case cudf::type_id::INT32: {
        cudf::numeric_scalar<int32_t> scalar(
            static_cast<int32_t>(offset), true, stream, mr);
        return cudf::binary_operation(
            values,
            scalar,
            cudf::binary_operator::ADD,
            values.type(),
            stream,
            mr);
      }
      case cudf::type_id::INT64: {
        cudf::numeric_scalar<int64_t> scalar(offset, true, stream, mr);
        return cudf::binary_operation(
            values,
            scalar,
            cudf::binary_operator::ADD,
            values.type(),
            stream,
            mr);
      }
      default:
        VELOX_FAIL(
            "Unsupported rank output type {}",
            static_cast<int>(values.type().id()));
    }
  };
  auto makeConstant =
      [&](int64_t value,
          cudf::size_type rows) -> std::unique_ptr<cudf::column> {
    switch (localResult->type().id()) {
      case cudf::type_id::INT32: {
        cudf::numeric_scalar<int32_t> scalar(
            static_cast<int32_t>(value), true, stream, mr);
        return cudf::make_column_from_scalar(scalar, rows, stream, mr);
      }
      case cudf::type_id::INT64: {
        cudf::numeric_scalar<int64_t> scalar(value, true, stream, mr);
        return cudf::make_column_from_scalar(scalar, rows, stream, mr);
      }
      default:
        VELOX_FAIL(
            "Unsupported rank output type {}",
            static_cast<int>(localResult->type().id()));
    }
  };

  std::vector<std::unique_ptr<cudf::column>> ownedSegments;
  std::vector<cudf::column_view> segments;
  auto appendUnchanged = [&](cudf::size_type begin, cudf::size_type end) {
    if (begin == end) {
      return;
    }
    auto slice = cudf::slice(localResult->view(), {begin, end}, stream);
    segments.push_back(slice.front());
  };
  auto appendOffset =
      [&](cudf::size_type begin, cudf::size_type end, int64_t offset) {
    if (begin == end) {
      return;
    }
    auto slice = cudf::slice(localResult->view(), {begin, end}, stream);
    ownedSegments.push_back(addOffset(slice.front(), offset));
    segments.push_back(ownedSegments.back()->view());
  };

  cudf::size_type fixedEnd{0};
  if (functionName == "rank") {
    auto continuingPartition =
        cudf::slice(sortedInput, {0, samePartitionEnd}, stream);
    const auto sameOrderEnd = continuingPrefixSize(
        continuingPartition.front(),
        sortKeyChannels_,
        previousOrderKey_,
        sortOrders_,
        sortNullOrders_,
        stream,
        mr);
    if (sameOrderEnd > 0) {
      ownedSegments.push_back(makeConstant(previousRank_, sameOrderEnd));
      segments.push_back(ownedSegments.back()->view());
    }
    fixedEnd = sameOrderEnd;
  }
  appendOffset(
      fixedEnd,
      samePartitionEnd,
      previousPartitionRows_);
  appendUnchanged(samePartitionEnd, localResult->size());
  VELOX_CHECK(!segments.empty());
  return segments.size() == 1
      ? std::make_unique<cudf::column>(segments.front(), stream, mr)
      : cudf::concatenate(segments, stream, mr);
}

void CudfWindow::updateRankLikeState(
    cudf::table_view sortedInput,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GT(sortedInput.num_rows(), 0);
  const auto numRows = sortedInput.num_rows();
  const std::vector<cudf::order> partitionOrders(
      partitionKeyChannels_.size(), cudf::order::ASCENDING);
  const std::vector<cudf::null_order> partitionNullOrders(
      partitionKeyChannels_.size(), cudf::null_order::BEFORE);
  const auto continuedRows = hasRankLikeState_
      ? continuingPrefixSize(
            sortedInput,
            partitionKeyChannels_,
            previousPartitionKey_,
            partitionOrders,
            partitionNullOrders,
            stream,
            mr)
      : 0;
  const bool continuedWholeBatch = continuedRows == numRows;

  auto partitionColumns =
      cudf::table_view(selectColumns(sortedInput, partitionKeyChannels_));
  auto lastPartition =
      cudf::slice(partitionColumns, {numRows - 1, numRows}, stream);
  auto partitionPositions = cudf::lower_bound(
      partitionColumns,
      lastPartition.front(),
      partitionOrders,
      partitionNullOrders,
      stream,
      mr);
  const auto trailingPartitionBegin =
      firstSearchPosition(partitionPositions->view(), stream);

  auto trailingPartition =
      cudf::slice(sortedInput, {trailingPartitionBegin, numRows}, stream);
  auto trailingOrderColumns = cudf::table_view(
      selectColumns(trailingPartition.front(), sortKeyChannels_));
  auto lastOrder = cudf::slice(
      trailingOrderColumns,
      {trailingOrderColumns.num_rows() - 1, trailingOrderColumns.num_rows()},
      stream);
  auto orderPositions = cudf::lower_bound(
      trailingOrderColumns,
      lastOrder.front(),
      sortOrders_,
      sortNullOrders_,
      stream,
      mr);
  const auto lastPeerBegin =
      firstSearchPosition(orderPositions->view(), stream);
  const bool lastPeerContinued =
      continuedWholeBatch &&
      continuingPrefixSize(
          sortedInput,
          sortKeyChannels_,
          previousOrderKey_,
          sortOrders_,
          sortNullOrders_,
          stream,
          mr) == numRows;

  const auto priorRows =
      continuedWholeBatch ? previousPartitionRows_ : 0;
  if (!lastPeerContinued) {
    previousRank_ = priorRows + lastPeerBegin + 1;
  }
  previousPartitionRows_ =
      priorRows + numRows - trailingPartitionBegin;
  previousPartitionKey_ =
      std::make_unique<cudf::table>(lastPartition.front(), stream, mr);
  previousOrderKey_ =
      std::make_unique<cudf::table>(lastOrder.front(), stream, mr);
  hasRankLikeState_ = true;
}

std::unique_ptr<cudf::column> CudfWindow::computeRowNumberColumn(
    cudf::table_view const& sortedInput,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto valuesColumn = sortedInput.column(sortKeyChannels_[0]);
  const auto order = sortOrders_[0];
  const auto nullOrder = sortNullOrders_[0];

  std::unique_ptr<cudf::column> rowNumber;
  if (partitionKeyChannels_.empty()) {
    auto aggregation = cudf::make_rank_aggregation<cudf::scan_aggregation>(
        cudf::rank_method::FIRST, order, cudf::null_policy::INCLUDE, nullOrder);
    rowNumber = cudf::scan(
        valuesColumn,
        *aggregation,
        cudf::scan_type::INCLUSIVE,
        cudf::null_policy::INCLUDE,
        stream,
        mr);
  } else {
    auto partitionColumns = selectColumns(sortedInput, partitionKeyChannels_);
    std::vector<cudf::groupby::scan_request> requests(1);
    requests[0].values = valuesColumn;
    requests[0].aggregations.push_back(
        cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::FIRST,
            order,
            cudf::null_policy::INCLUDE,
            nullOrder));

    cudf::groupby::groupby grouper(
        cudf::table_view(partitionColumns),
        cudf::null_policy::INCLUDE,
        cudf::sorted::YES,
        std::vector<cudf::order>(
            partitionKeyChannels_.size(), cudf::order::ASCENDING),
        std::vector<cudf::null_order>(
            partitionKeyChannels_.size(), cudf::null_order::BEFORE));

    auto scanResult = grouper.scan(requests, stream, mr);
    VELOX_CHECK_EQ(scanResult.second.size(), 1);
    VELOX_CHECK_EQ(scanResult.second[0].results.size(), 1);
    rowNumber = std::move(scanResult.second[0].results[0]);
  }

  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (rowNumber->type() != expectedCudfType) {
    rowNumber = cudf::cast(rowNumber->view(), expectedCudfType, stream, mr);
  }
  return rowNumber;
}

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto rowNumber =
      computeRowNumberColumn(sortedInput, expectedType, stream, mr);

  std::vector<cudf::size_type> rankKeyChannels = partitionKeyChannels_;
  rankKeyChannels.insert(
      rankKeyChannels.end(), sortKeyChannels_.begin(), sortKeyChannels_.end());
  VELOX_CHECK(!rankKeyChannels.empty(), "Window rank requires sort keys");

  auto rankKeyColumns = selectColumns(sortedInput, rankKeyChannels);
  cudf::groupby::groupby grouper(
      cudf::table_view(rankKeyColumns), cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = rowNumber->view();
  requests[0].aggregations.push_back(
      cudf::make_min_aggregation<cudf::groupby_aggregation>());

  auto [groupKeys, results] = grouper.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 1);
  auto minRankByPeerGroup = std::move(results[0].results[0]);

  cudf::hash_join lookup(
      groupKeys->view(),
      cudf::nullable_join::YES,
      cudf::null_equality::EQUAL,
      0.5,
      stream);

  auto [leftJoinIndices, rightJoinIndices] = lookup.left_join(
      cudf::table_view(rankKeyColumns),
      static_cast<std::size_t>(sortedInput.num_rows()),
      stream,
      mr);

  auto leftIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

  auto gatheredRanks = cudf::gather(
      cudf::table_view{{minRankByPeerGroup->view()}},
      rightIndicesCol,
      cudf::out_of_bounds_policy::NULLIFY,
      stream,
      mr);
  auto gatheredColumns = gatheredRanks->release();
  VELOX_CHECK_EQ(gatheredColumns.size(), 1);

  auto target = cudf::allocate_like(
      gatheredColumns[0]->view(),
      sortedInput.num_rows(),
      cudf::mask_allocation_policy::RETAIN,
      stream,
      mr);
  auto scattered = cudf::scatter(
      cudf::table_view{{gatheredColumns[0]->view()}},
      leftIndicesCol,
      cudf::table_view{{target->view()}},
      stream,
      mr);
  return std::move(scattered->release()[0]);
}

std::unique_ptr<cudf::column> CudfWindow::computeFullPartitionSumColumn(
    cudf::table_view const& input,
    const core::WindowNode::Function& function,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!partitionKeyChannels_.empty());

  const auto valueChannel =
      exec::exprToChannel(function.functionCall->inputs()[0].get(), inputType_);
  VELOX_CHECK(
      valueChannel != kConstantChannel, "Window sum input must be a column");

  auto partitionColumns = selectColumns(input, partitionKeyChannels_);

  cudf::groupby::groupby grouper(
      cudf::table_view(partitionColumns), cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = input.column(valueChannel);
  requests[0].aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  auto [groupKeys, results] = grouper.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 1);

  auto sumByGroup = std::move(results[0].results[0]);
  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (sumByGroup->type() != expectedCudfType) {
    sumByGroup = cudf::cast(sumByGroup->view(), expectedCudfType, stream, mr);
  }

  cudf::hash_join lookup(
      groupKeys->view(),
      cudf::nullable_join::YES,
      cudf::null_equality::EQUAL,
      0.5,
      stream);

  auto [leftJoinIndices, rightJoinIndices] = lookup.left_join(
      cudf::table_view(partitionColumns),
      static_cast<std::size_t>(input.num_rows()),
      stream,
      mr);

  auto leftIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

  auto gatheredSums = cudf::gather(
      cudf::table_view{{sumByGroup->view()}},
      rightIndicesCol,
      cudf::out_of_bounds_policy::NULLIFY,
      stream,
      mr);
  auto gatheredColumns = gatheredSums->release();
  VELOX_CHECK_EQ(gatheredColumns.size(), 1);

  auto target = cudf::allocate_like(
      gatheredColumns[0]->view(),
      input.num_rows(),
      cudf::mask_allocation_policy::RETAIN,
      stream,
      mr);
  auto scattered = cudf::scatter(
      cudf::table_view{{gatheredColumns[0]->view()}},
      leftIndicesCol,
      cudf::table_view{{target->view()}},
      stream,
      mr);
  return std::move(scattered->release()[0]);
}

std::unique_ptr<cudf::column> CudfWindow::computeRunningPartitionSumColumn(
    cudf::table_view const& sortedInput,
    const core::WindowNode::Function& function,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!partitionKeyChannels_.empty());
  const auto valueChannel =
      exec::exprToChannel(function.functionCall->inputs()[0].get(), inputType_);
  VELOX_CHECK_NE(valueChannel, kConstantChannel);

  auto partitionColumns = selectColumns(sortedInput, partitionKeyChannels_);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = sortedInput.column(valueChannel);
  requests[0].aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>());

  cudf::groupby::groupby grouper(
      cudf::table_view(partitionColumns),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyChannels_.size(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyChannels_.size(), cudf::null_order::BEFORE));
  auto scanResult = grouper.scan(requests, stream, mr);
  VELOX_CHECK_EQ(scanResult.second.size(), 1);
  VELOX_CHECK_EQ(scanResult.second[0].results.size(), 1);
  auto result = std::move(scanResult.second[0].results[0]);
  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (result->type() != expectedCudfType) {
    result = cudf::cast(result->view(), expectedCudfType, stream, mr);
  }
  return result;
}

std::unique_ptr<cudf::column> CudfWindow::computePartitionFirstColumn(
    cudf::table_view const& sortedInput,
    const core::WindowNode::Function& function,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!partitionKeyChannels_.empty());

  const auto valueChannel =
      exec::exprToChannel(function.functionCall->inputs()[0].get(), inputType_);
  VELOX_CHECK(
      valueChannel != kConstantChannel, "Window first input must be a column");

  auto partitionColumns = selectColumns(sortedInput, partitionKeyChannels_);

  cudf::groupby::groupby grouper(
      cudf::table_view(partitionColumns),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyChannels_.size(), cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyChannels_.size(), cudf::null_order::BEFORE));

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = sortedInput.column(valueChannel);
  requests[0].aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(
          0, cudf::null_policy::INCLUDE));

  auto [groupKeys, results] = grouper.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 1);

  auto firstByGroup = std::move(results[0].results[0]);
  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (firstByGroup->type() != expectedCudfType) {
    firstByGroup =
        cudf::cast(firstByGroup->view(), expectedCudfType, stream, mr);
  }

  cudf::hash_join lookup(
      groupKeys->view(),
      cudf::nullable_join::YES,
      cudf::null_equality::EQUAL,
      0.5,
      stream);

  auto [leftJoinIndices, rightJoinIndices] = lookup.left_join(
      cudf::table_view(partitionColumns),
      static_cast<std::size_t>(sortedInput.num_rows()),
      stream,
      mr);

  auto leftIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

  auto gatheredFirstValues = cudf::gather(
      cudf::table_view{{firstByGroup->view()}},
      rightIndicesCol,
      cudf::out_of_bounds_policy::NULLIFY,
      stream,
      mr);
  auto gatheredColumns = gatheredFirstValues->release();
  VELOX_CHECK_EQ(gatheredColumns.size(), 1);

  auto target = cudf::allocate_like(
      gatheredColumns[0]->view(),
      sortedInput.num_rows(),
      cudf::mask_allocation_policy::RETAIN,
      stream,
      mr);
  auto scattered = cudf::scatter(
      cudf::table_view{{gatheredColumns[0]->view()}},
      leftIndicesCol,
      cudf::table_view{{target->view()}},
      stream,
      mr);
  return std::move(scattered->release()[0]);
}

std::unique_ptr<cudf::table> CudfWindow::computeOutputTable(
    std::unique_ptr<cudf::table> input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool inputAlreadySorted) {
  const auto inputSize = inputType_->size();
  const auto numFunctions = windowNode_->windowFunctions().size();

  if (input->num_rows() == 0) {
    auto columns = input->release();
    for (size_t i = 0; i < numFunctions; ++i) {
      columns.push_back(
          cudf::make_empty_column(
              veloxToCudfDataType(outputType_->childAt(inputSize + i))));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  const auto inputView = input->view();
  const auto hasFullPartitionSum = std::all_of(
      windowNode_->windowFunctions().begin(),
      windowNode_->windowFunctions().end(),
      isSupportedFullPartitionSumFunction);
  if (hasFullPartitionSum) {
    std::vector<std::unique_ptr<cudf::column>> resultColumns;
    resultColumns.reserve(numFunctions);
    for (size_t i = 0; i < numFunctions; ++i) {
      resultColumns.push_back(computeFullPartitionSumColumn(
          inputView,
          windowNode_->windowFunctions()[i],
          outputType_->childAt(inputSize + i),
          stream,
          mr));
    }

    auto columns = input->release();
    for (auto& column : resultColumns) {
      columns.push_back(std::move(column));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  std::vector<cudf::column_view> sortKeyViews;
  std::vector<cudf::order> columnOrders;
  std::vector<cudf::null_order> nullOrders;

  for (const auto channel : partitionKeyChannels_) {
    sortKeyViews.push_back(inputView.column(channel));
    columnOrders.push_back(cudf::order::ASCENDING);
    nullOrders.push_back(cudf::null_order::BEFORE);
  }
  for (size_t i = 0; i < sortKeyChannels_.size(); ++i) {
    sortKeyViews.push_back(inputView.column(sortKeyChannels_[i]));
    columnOrders.push_back(sortOrders_[i]);
    nullOrders.push_back(sortNullOrders_[i]);
  }

  std::unique_ptr<cudf::column> sortedOrder;
  std::unique_ptr<cudf::table> sortedTable;
  cudf::table_view sortedView;

  if (inputAlreadySorted || windowNode_->inputsSorted()) {
    sortedView = inputView;
  } else {
    sortedOrder = cudf::stable_sorted_order(
        cudf::table_view(sortKeyViews), columnOrders, nullOrders, stream, mr);
    sortedTable = cudf::gather(
        inputView,
        sortedOrder->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    sortedView = sortedTable->view();
  }

  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(numFunctions);
  for (size_t i = 0; i < numFunctions; ++i) {
    const auto& function = windowNode_->windowFunctions()[i];
    const auto& functionName = function.functionCall->name();
    std::unique_ptr<cudf::column> result;
    if (functionName == "rank") {
      result = computeRankColumn(
          sortedView, outputType_->childAt(inputSize + i), stream, mr);
    } else if (functionName == "first" || functionName == "first_value") {
      result = computePartitionFirstColumn(
          sortedView,
          function,
          outputType_->childAt(inputSize + i),
          stream,
          mr);
    } else if (functionName == "sum") {
      result = computeRunningPartitionSumColumn(
          sortedView,
          function,
          outputType_->childAt(inputSize + i),
          stream,
          mr);
    } else {
      result = computeRowNumberColumn(
          sortedView, outputType_->childAt(inputSize + i), stream, mr);
    }
    if (rankLikeStreaming_) {
      result = fixRankLikeColumn(
          std::move(result),
          functionName,
          sortedView,
          stream,
          mr);
    }
    resultColumns.push_back(std::move(result));
  }

  if (rankLikeStreaming_) {
    updateRankLikeState(sortedView, stream, mr);
  }

  if (sortedOrder) {
    for (auto& column : resultColumns) {
      auto target = cudf::allocate_like(
          column->view(),
          column->size(),
          cudf::mask_allocation_policy::RETAIN,
          stream,
          mr);
      auto scattered = cudf::scatter(
          cudf::table_view{{column->view()}},
          sortedOrder->view(),
          cudf::table_view{{target->view()}},
          stream,
          mr);
      column = std::move(scattered->release()[0]);
    }
  }

  auto columns = input->release();
  for (auto& column : resultColumns) {
    columns.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace facebook::velox::cudf_velox
