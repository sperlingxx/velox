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

#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <glog/logging.h>

namespace {

using namespace facebook::velox;

bool isSupportedWindowType(const TypePtr& type) {
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
    case TypeKind::ROW:
      for (auto i = 0; i < type->size(); ++i) {
        if (!isSupportedWindowType(type->childAt(i))) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

bool isFieldAccessExpr(const core::TypedExprPtr& expr) {
  return std::dynamic_pointer_cast<
             const core::FieldAccessTypedExpr>(expr) != nullptr;
}

bool isConstantExpr(const core::TypedExprPtr& expr) {
  return std::dynamic_pointer_cast<
             const core::ConstantTypedExpr>(expr) != nullptr;
}

int64_t extractConstantBound(const core::TypedExprPtr& expr) {
  auto constExpr = std::dynamic_pointer_cast<
      const core::ConstantTypedExpr>(expr);
  VELOX_CHECK_NOT_NULL(
      constExpr, "Frame bound must be a constant expression");

  if (constExpr->hasValueVector()) {
    auto vec = constExpr->valueVector();
    VELOX_CHECK(!vec->isNullAt(0));
    auto kind = vec->type()->kind();
    if (kind == TypeKind::INTEGER) {
      return vec->as<SimpleVector<int32_t>>()->valueAt(0);
    }
    return vec->as<SimpleVector<int64_t>>()->valueAt(0);
  }

  auto kind = constExpr->type()->kind();
  switch (kind) {
    case TypeKind::INTEGER:
      return constExpr->value().value<TypeKind::INTEGER>();
    case TypeKind::BIGINT:
      return constExpr->value().value<TypeKind::BIGINT>();
    default:
      VELOX_FAIL(
          "Unsupported frame bound type: {}",
          constExpr->type()->toString());
  }
}

std::optional<cudf_velox::WindowFunctionKind> parseWindowFunctionKind(
    const std::string& name) {
  using Kind = cudf_velox::WindowFunctionKind;
  if (name == "row_number") return Kind::kRowNumber;
  if (name == "rank") return Kind::kRank;
  if (name == "dense_rank") return Kind::kDenseRank;
  if (name == "sum") return Kind::kSum;
  if (name == "min") return Kind::kMin;
  if (name == "max") return Kind::kMax;
  if (name == "count") return Kind::kCount;
  if (name == "avg") return Kind::kAvg;
  return std::nullopt;
}

bool isDefaultRankFrame(const core::WindowNode::Frame& frame) {
  // Rank-like functions ignore the frame, but the SQL default
  // is UNBOUNDED PRECEDING to CURRENT ROW (type can be ROWS
  // or RANGE depending on planner).
  return frame.startType ==
      core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kCurrentRow &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isSupportedFrame(const core::WindowNode::Frame& frame) {
  using WT = core::WindowNode::WindowType;
  using BT = core::WindowNode::BoundType;

  // RANGE with both unbounded is equivalent to ROWS unbounded
  // and can be handled by grouped_rolling_window.
  if (frame.type == WT::kRange) {
    return frame.startType == BT::kUnboundedPreceding &&
        frame.endType == BT::kUnboundedFollowing;
  }

  if (frame.type != WT::kRows) {
    return false;
  }
  auto startOk = frame.startType == BT::kUnboundedPreceding ||
      frame.startType == BT::kCurrentRow ||
      (frame.startType == BT::kPreceding &&
       frame.startValue &&
       isConstantExpr(frame.startValue));
  auto endOk = frame.endType == BT::kUnboundedFollowing ||
      frame.endType == BT::kCurrentRow ||
      (frame.endType == BT::kFollowing &&
       frame.endValue &&
       isConstantExpr(frame.endValue));
  return startOk && endOk;
}

bool isSupportedRankLikeFunction(
    const core::WindowNode::Function& function) {
  auto kind = parseWindowFunctionKind(
      function.functionCall->name());
  if (!kind.has_value()) {
    return false;
  }
  if (!cudf_velox::CudfWindow::isRankLike(*kind)) {
    return false;
  }
  if (function.ignoreNulls) {
    return false;
  }
  if (!function.functionCall->inputs().empty()) {
    return false;
  }
  if (!isDefaultRankFrame(function.frame)) {
    return false;
  }
  auto resultKind = function.functionCall->type()->kind();
  return resultKind == TypeKind::INTEGER ||
      resultKind == TypeKind::BIGINT;
}

bool isSupportedAggregateFunction(
    const core::WindowNode::Function& function) {
  auto kind = parseWindowFunctionKind(
      function.functionCall->name());
  if (!kind.has_value() ||
      cudf_velox::CudfWindow::isRankLike(*kind)) {
    return false;
  }
  if (function.ignoreNulls) {
    return false;
  }
  if (!isSupportedFrame(function.frame)) {
    return false;
  }
  auto const& inputs = function.functionCall->inputs();
  if (*kind == cudf_velox::WindowFunctionKind::kCount) {
    if (inputs.size() > 1) return false;
    if (inputs.size() == 1 && !isFieldAccessExpr(inputs[0])) {
      return false;
    }
  } else {
    if (inputs.size() != 1 || !isFieldAccessExpr(inputs[0])) {
      return false;
    }
  }
  return true;
}

std::vector<cudf::column_view> selectColumns(
    cudf::table_view const& table,
    std::vector<cudf::size_type> const& channels) {
  std::vector<cudf::column_view> columns;
  columns.reserve(channels.size());
  for (auto channel : channels) {
    columns.push_back(table.column(channel));
  }
  return columns;
}

} // namespace

namespace facebook::velox::cudf_velox {

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node) {
  if (!node || !isSupportedWindowType(node->inputType()) ||
      !isSupportedWindowType(node->outputType())) {
    return false;
  }

  auto const& functions = node->windowFunctions();
  if (functions.empty()) {
    return false;
  }

  bool hasRankLike = false;
  for (auto const& func : functions) {
    if (!isSupportedRankLikeFunction(func) &&
        !isSupportedAggregateFunction(func)) {
      return false;
    }
    if (isSupportedRankLikeFunction(func)) {
      hasRankLike = true;
    }
  }

  // Rank-like functions require at least one sort key.
  if (hasRankLike && node->sortingKeys().empty()) {
    return false;
  }

  for (auto const& key : node->sortingKeys()) {
    if (!isFieldAccessExpr(key) ||
        !isSupportedWindowType(key->type())) {
      return false;
    }
  }

  for (auto const& key : node->partitionKeys()) {
    if (!isFieldAccessExpr(key) ||
        !isSupportedWindowType(key->type())) {
      return false;
    }
  }

  return true;
}

// static
bool CudfWindow::isRankLike(WindowFunctionKind kind) {
  return kind == WindowFunctionKind::kRowNumber ||
      kind == WindowFunctionKind::kRank ||
      kind == WindowFunctionKind::kDenseRank;
}

// static
cudf::rank_method CudfWindow::toRankMethod(
    WindowFunctionKind kind) {
  switch (kind) {
    case WindowFunctionKind::kRowNumber:
      return cudf::rank_method::FIRST;
    case WindowFunctionKind::kRank:
      return cudf::rank_method::MIN;
    case WindowFunctionKind::kDenseRank:
      return cudf::rank_method::DENSE;
    default:
      VELOX_UNREACHABLE(
          "toRankMethod called on non-rank function");
  }
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : exec::Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "CudfWindow"),
      NvtxHelper(
          nvtx3::rgb{123, 104, 238}, // MediumSlateBlue
          operatorId,
          fmt::format("[{}]", windowNode->id())),
      windowNode_(windowNode),
      inputType_(windowNode->sources()[0]->outputType()),
      isFlushByPartition_(
          windowNode->inputsSorted() &&
          !windowNode->partitionKeys().empty()) {
  std::vector<std::string> partitionKeyNames;
  std::vector<TypePtr> partitionKeyTypes;

  // Parse partition key channels.
  partitionKeyChannels_.reserve(
      windowNode->partitionKeys().size());
  partitionKeyOrders_.reserve(windowNode->partitionKeys().size());
  partitionKeyNullOrders_.reserve(windowNode->partitionKeys().size());
  for (auto const& key : windowNode->partitionKeys()) {
    auto ch = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK_NE(
        ch,
        kConstantChannel,
        "Window partition key must be a column");
    partitionKeyChannels_.push_back(
        static_cast<cudf::size_type>(ch));
    partitionKeyOrders_.push_back(cudf::order::ASCENDING);
    partitionKeyNullOrders_.push_back(cudf::null_order::BEFORE);
    partitionKeyNames.push_back(fmt::format("pk{}", ch));
    partitionKeyTypes.push_back(inputType_->childAt(ch));
  }
  if (!partitionKeyTypes.empty()) {
    partitionKeyType_ =
        facebook::velox::ROW(std::move(partitionKeyNames), std::move(partitionKeyTypes));
  }

  // Parse sort key channels and orders.
  sortKeyChannels_.reserve(windowNode->sortingKeys().size());
  sortOrders_.reserve(windowNode->sortingOrders().size());
  sortNullOrders_.reserve(windowNode->sortingOrders().size());
  for (size_t i = 0; i < windowNode->sortingKeys().size(); ++i) {
    auto ch = exec::exprToChannel(
        windowNode->sortingKeys()[i].get(), inputType_);
    VELOX_CHECK_NE(
        ch,
        kConstantChannel,
        "Window sorting key must be a column");
    sortKeyChannels_.push_back(
        static_cast<cudf::size_type>(ch));
    auto const& order = windowNode->sortingOrders()[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING
                            : cudf::order::DESCENDING);
    sortNullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }

  // Parse window function specs.
  for (auto const& func : windowNode->windowFunctions()) {
    WindowFunctionSpec spec;
    auto kind = parseWindowFunctionKind(
        func.functionCall->name());
    VELOX_CHECK(
        kind.has_value(),
        "Unsupported window function: {}",
        func.functionCall->name());
    spec.kind = *kind;
    spec.frame = func.frame;
    spec.countNullPolicy = cudf::null_policy::EXCLUDE;

    if (isRankLike(spec.kind)) {
      spec.inputChannel = -1;
    } else if (
        spec.kind == WindowFunctionKind::kCount &&
        func.functionCall->inputs().empty()) {
      // count(*): use first column as dummy input, count all
      spec.inputChannel = 0;
      spec.countNullPolicy = cudf::null_policy::INCLUDE;
    } else {
      auto ch = exec::exprToChannel(
          func.functionCall->inputs()[0].get(), inputType_);
      VELOX_CHECK_NE(
          ch,
          kConstantChannel,
          "Window function input must be a column");
      spec.inputChannel = static_cast<cudf::size_type>(ch);
    }
    functionSpecs_.push_back(std::move(spec));
  }
}

void CudfWindow::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() == 0) {
    return;
  }
  auto cudfInput =
      std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  GpuGuard gpuGuard;
  auto stream = cudfGlobalStreamPool().get_stream();
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{cudfInput->stream()}, stream);

  cacheNextBatch(std::move(cudfInput), stream);
  refreshFinished();
}

void CudfWindow::noMoreInput() {
  exec::Operator::noMoreInput();

  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  auto stream = cudfGlobalStreamPool().get_stream();
  if (isFlushByPartition_) {
    if (!cachedInputs_.empty()) {
      VLOG(1) << "CudfWindow planNodeId=" << windowNode_->id()
              << " flushing remaining cached input at noMoreInput: batches="
              << cachedInputs_.size() << " rows=" << cachedRows_
              << " bytes=" << cachedBytes_;
    }
    cachedRows_ = 0;
    cachedBytes_ = 0;
  }
  flushCachedInput(stream, false);
  refreshFinished();
}

RowVectorPtr CudfWindow::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (outputQueue_.empty()) {
    refreshFinished();
    return nullptr;
  }
  auto output = outputQueue_.front();
  outputQueue_.pop_front();
  refreshFinished();
  return output;
}

void CudfWindow::close() {
  exec::Operator::close();
  cachedInputs_.clear();
  outputQueue_.clear();
  cachedRows_ = 0;
  cachedBytes_ = 0;
  isFinished_ = true;
}

CudfVectorPtr CudfWindow::materializeTableView(
    cudf::table_view view,
    const RowTypePtr& type,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  return wrapOwnedTable(
      std::make_unique<cudf::table>(view, stream, mr), type, stream);
}

CudfVectorPtr CudfWindow::wrapOwnedTable(
    std::unique_ptr<cudf::table> table,
    const RowTypePtr& type,
    rmm::cuda_stream_view stream) const {
  auto size = static_cast<vector_size_t>(table->num_rows());
  return std::make_shared<CudfVector>(
      pool(), type, size, std::move(table), stream);
}

cudf::table_view CudfWindow::extractPartitionKeyRow(
    cudf::table_view input,
    cudf::size_type row,
    rmm::cuda_stream_view stream) const {
  auto partitionKeys = cudf::table_view(selectColumns(input, partitionKeyChannels_));
  return cudf::slice(partitionKeys, {row, row + 1}, stream).front();
}

bool CudfWindow::isSamePartitionKey(
    cudf::table_view lhs,
    cudf::table_view rhs,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::table_view> tableViews{lhs, rhs};
  auto combined = cudf::concatenate(tableViews, stream, mr);
  return cudf::distinct_count(
             combined->view(), cudf::null_equality::EQUAL, stream) == 1;
}

cudf::size_type CudfWindow::trailingPartitionStartRow(
    cudf::table_view input,
    rmm::cuda_stream_view stream) const {
  VELOX_CHECK_GT(input.num_rows(), 0);
  VELOX_CHECK(
      !partitionKeyChannels_.empty(),
      "Trailing partition split requires partition keys");

  auto mr = cudf::get_current_device_resource_ref();
  auto partitionTable = cudf::table_view(selectColumns(input, partitionKeyChannels_));
  auto lastKey = extractPartitionKeyRow(input, input.num_rows() - 1, stream);
  auto lowerBound = cudf::lower_bound(
      partitionTable,
      lastKey,
      partitionKeyOrders_,
      partitionKeyNullOrders_,
      stream,
      mr);
  VELOX_CHECK_EQ(lowerBound->size(), 1);
  auto lowerBoundScalar = cudf::get_element(lowerBound->view(), 0, stream, mr);
  VELOX_CHECK(lowerBoundScalar->is_valid(stream));
  return static_cast<cudf::numeric_scalar<cudf::size_type>*>(lowerBoundScalar.get())
      ->value(stream);
}

bool CudfWindow::batchTargetReached(
    vector_size_t rows,
    uint64_t bytes) const {
  const auto& config = CudfConfig::getInstance();
  if (config.gpuTargetBatchBytes > 0) {
    return bytes > static_cast<uint64_t>(config.gpuTargetBatchBytes);
  }
  if (config.gpuTargetBatchRows == 0) {
    return true;
  }
  return rows > static_cast<vector_size_t>(config.gpuTargetBatchRows);
}

void CudfWindow::enqueueOutputTable(
    std::unique_ptr<cudf::table> outputTable,
    rmm::cuda_stream_view stream) {
  if (!outputTable || outputTable->num_rows() == 0) {
    return;
  }
  outputQueue_.push_back(wrapOwnedTable(std::move(outputTable), outputType_, stream));
}

void CudfWindow::cacheNextBatch(
    CudfVectorPtr input,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(input);
  cachedInputs_.push_back(std::move(input));

  if (!isFlushByPartition_) {
    return;
  }

  cachedRows_ += cachedInputs_.back()->size();
  cachedBytes_ += cachedInputs_.back()->estimateFlatSize();
  if (batchTargetReached(cachedRows_, cachedBytes_)) {
    const auto& config = CudfConfig::getInstance();
    VLOG(1) << "CudfWindow planNodeId=" << windowNode_->id()
            << " triggering partition-aware flush attempt: batches="
            << cachedInputs_.size() << " rows=" << cachedRows_
            << " bytes=" << cachedBytes_
            << " targetRows=" << config.gpuTargetBatchRows
            << " targetBytes=" << config.gpuTargetBatchBytes;
    flushCachedInput(stream, true);
  }
}

void CudfWindow::flushCachedInput(
    rmm::cuda_stream_view stream,
    bool isPartitionAware) {
  if (cachedInputs_.empty()) {
    return;
  }

  auto cachedInputs = std::move(cachedInputs_);
  cachedInputs_.clear();
  cachedRows_ = 0;
  cachedBytes_ = 0;

  std::vector<CudfVectorPtr> nextCachedInputs;
  auto inputTable = isPartitionAware
      ? preparePartitionAwareFlush(
            std::move(cachedInputs), stream, nextCachedInputs)
      : getConcatenatedTable(cachedInputs, inputType_, stream);
  cachedInputs.clear();

  if (inputTable && inputTable->num_rows() > 0) {
    auto outputTable = computeOutputTable(std::move(inputTable), stream);
    enqueueOutputTable(std::move(outputTable), stream);
  }

  for (auto& cachedInput : nextCachedInputs) {
    cachedRows_ += cachedInput->size();
    cachedBytes_ += cachedInput->estimateFlatSize();
    cachedInputs_.push_back(std::move(cachedInput));
  }
}

std::unique_ptr<cudf::table> CudfWindow::preparePartitionAwareFlush(
    std::vector<CudfVectorPtr> cachedInputs,
    rmm::cuda_stream_view stream,
    std::vector<CudfVectorPtr>& nextCachedInputs) {
  // Start from the optimistic assumption that everything except the last cached
  // batch is flushable. The primary goal is to keep this flush close to the
  // batch-size threshold that triggered in cacheNextBatch(). Once batch-size
  // control forces a split, we also have to verify that the split point does
  // not land in the middle of a partition.
  auto mr = cudf::get_current_device_resource_ref();
  auto splitTrailingPartition =
      [&](std::unique_ptr<cudf::table> inputTable) -> std::unique_ptr<cudf::table> {
    auto processView = inputTable->view();
    auto trailingStart = trailingPartitionStartRow(processView, stream);
    if (trailingStart == 0) {
      // The whole flush candidate belongs to that trailing partition, so keep it
      // all cached and emit nothing this round.
      VLOG(1) << "CudfWindow planNodeId=" << windowNode_->id()
              << " deferring flush because the current candidate belongs to a "
                 "single trailing partition: rows="
              << processView.num_rows() << " batches="
              << cachedInputs.size() << " heldBackTailBatches="
              << nextCachedInputs.size();
      nextCachedInputs.insert(
          nextCachedInputs.begin(),
          wrapOwnedTable(std::move(inputTable), inputType_, stream));
      return nullptr;
    }

    // Split the candidate into a flushable prefix and a trailing partition tail.
    auto slices = cudf::split(processView, {trailingStart}, stream);
    auto processSlice = slices[0];
    auto trailingSlice = slices[1];
    VLOG(1) << "CudfWindow planNodeId=" << windowNode_->id()
            << " splitting partition-aware flush candidate at row "
            << trailingStart << ": flushRows=" << processSlice.num_rows()
            << " cachedTailRows=" << trailingSlice.num_rows();
    auto trailingInput =
        materializeTableView(trailingSlice, inputType_, stream);
    nextCachedInputs.insert(nextCachedInputs.begin(), std::move(trailingInput));
    auto outputTable = std::make_unique<cudf::table>(processSlice, stream, mr);
    inputTable.reset();
    return outputTable;
  };

  std::vector<CudfVectorPtr> processInputs;
  if (cachedInputs.size() == 1) {
    // With only one cached batch, we still need to honor partition integrity.
    // If this batch contains only one partition, keep pulling more input until
    // a partition boundary shows up, even if that means going past the nominal
    // batch-size threshold.
    processInputs = std::move(cachedInputs);
  } else {
    auto lastInput = std::move(cachedInputs.back());
    cachedInputs.pop_back();
    processInputs = std::move(cachedInputs);
    nextCachedInputs.push_back(std::move(lastInput));
  }

  auto inputTable = getConcatenatedTable(processInputs, inputType_, stream);
  processInputs.clear();
  if (inputTable->num_rows() == 0) {
    return nullptr;
  }

  if (nextCachedInputs.empty()) {
    if (noMoreInput_) {
      return inputTable;
    }

    // No tail batch has been held back yet, but the current candidate may still
    // end in the middle of a partition and need to keep that trailing partition cached.
    return splitTrailingPartition(std::move(inputTable));
  }

  auto processView = inputTable->view();
  // Batch-size control picked a tentative flush boundary. Compare the last
  // partition key in that flush candidate with the first key in the still-
  // cached tail. If they differ, the batch-size split already lands on a
  // partition boundary and the whole candidate table is safe to emit.
  // The cached tail may have been produced on a different stream.
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{nextCachedInputs.front()->stream()},
      stream);
  auto lastProcessKey = extractPartitionKeyRow(
      processView, processView.num_rows() - 1, stream);
  auto firstCachedKey =
      extractPartitionKeyRow(nextCachedInputs.front()->getTableView(), 0, stream);

  if (!isSamePartitionKey(lastProcessKey, firstCachedKey, stream)) {
    return inputTable;
  }

  // Batch-size control landed in the middle of a partition. Peel that trailing
  // partition back out so the next flush still computes the full partition in
  // one shot.
  return splitTrailingPartition(std::move(inputTable));
}

void CudfWindow::refreshFinished() {
  isFinished_ =
      noMoreInput_ && outputQueue_.empty() && cachedInputs_.empty();
}

std::unique_ptr<cudf::rolling_aggregation>
CudfWindow::makeRollingAggregation(
    const WindowFunctionSpec& spec) const {
  switch (spec.kind) {
    case WindowFunctionKind::kSum:
      return cudf::make_sum_aggregation<
          cudf::rolling_aggregation>();
    case WindowFunctionKind::kMin:
      return cudf::make_min_aggregation<
          cudf::rolling_aggregation>();
    case WindowFunctionKind::kMax:
      return cudf::make_max_aggregation<
          cudf::rolling_aggregation>();
    case WindowFunctionKind::kCount:
      return cudf::make_count_aggregation<
          cudf::rolling_aggregation>(spec.countNullPolicy);
    case WindowFunctionKind::kAvg:
      return cudf::make_mean_aggregation<
          cudf::rolling_aggregation>();
    default:
      VELOX_UNREACHABLE(
          "makeRollingAggregation for non-aggregate function");
  }
}

std::pair<cudf::window_bounds, cudf::window_bounds>
CudfWindow::toWindowBounds(
    const core::WindowNode::Frame& frame) const {
  using BT = core::WindowNode::BoundType;

  cudf::window_bounds preceding = [&]() {
    switch (frame.startType) {
      case BT::kUnboundedPreceding:
        return cudf::window_bounds::unbounded();
      case BT::kCurrentRow:
        return cudf::window_bounds::get(1);
      case BT::kPreceding: {
        auto val = extractConstantBound(frame.startValue);
        return cudf::window_bounds::get(
            static_cast<cudf::size_type>(val + 1));
      }
      default:
        VELOX_UNREACHABLE(
            "Unsupported frame start type");
    }
  }();

  cudf::window_bounds following = [&]() {
    switch (frame.endType) {
      case BT::kUnboundedFollowing:
        return cudf::window_bounds::unbounded();
      case BT::kCurrentRow:
        return cudf::window_bounds::get(0);
      case BT::kFollowing: {
        auto val = extractConstantBound(frame.endValue);
        return cudf::window_bounds::get(
            static_cast<cudf::size_type>(val));
      }
      default:
        VELOX_UNREACHABLE(
            "Unsupported frame end type");
    }
  }();

  return {preceding, following};
}

cudf::column_view CudfWindow::multiSortKeyStructView(
    cudf::table_view const& sortedInput) const {
  VELOX_CHECK_GE(
      sortKeyChannels_.size(),
      2,
      "multiSortKeyStructView requires >= 2 sort keys");
  sortKeyStructChildren_.clear();
  sortKeyStructChildren_.reserve(sortKeyChannels_.size());
  for (auto ch : sortKeyChannels_) {
    sortKeyStructChildren_.push_back(sortedInput.column(ch));
  }
  // Zero-copy struct column_view: cudf's row comparator treats children
  // as a composite key, comparing element-wise for equality.
  return cudf::column_view(
      cudf::data_type{cudf::type_id::STRUCT},
      sortedInput.num_rows(),
      nullptr,
      nullptr,
      0,
      0,
      sortKeyStructChildren_);
}

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    WindowFunctionKind kind,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  auto method = toRankMethod(kind);

  // Build the "values" column for rank tie detection.
  // - row_number (FIRST): ties don't matter, use any single column.
  // - rank/dense_rank with 1 sort key: use that key directly.
  // - rank/dense_rank with N sort keys: build a zero-copy struct
  //   column_view over all sort keys so cudf's row equality comparator
  //   detects ties across the full composite key.
  cudf::column_view valuesCol = [&]() -> cudf::column_view {
    if (sortKeyChannels_.empty()) {
      return sortedInput.column(0);
    }
    if (sortKeyChannels_.size() == 1 ||
        kind == WindowFunctionKind::kRowNumber) {
      return sortedInput.column(sortKeyChannels_[0]);
    }
    // Multi-key struct: cudf's rank_scan and group_rank_scan both use
    // row::equality::self_comparator which handles STRUCT via nested
    // column comparison (see rank_scan.cu and group_rank_scan.cu).
    return multiSortKeyStructView(sortedInput);
  }();

  auto colOrder = sortKeyChannels_.empty()
      ? cudf::order::ASCENDING
      : sortOrders_[0];
  auto nullOrd = sortKeyChannels_.empty()
      ? cudf::null_order::BEFORE
      : sortNullOrders_[0];

  if (partitionKeyChannels_.empty()) {
    auto agg = cudf::make_rank_aggregation<
        cudf::scan_aggregation>(
        method,
        colOrder,
        cudf::null_policy::INCLUDE,
        nullOrd);
    return cudf::scan(
        valuesCol,
        *agg,
        cudf::scan_type::INCLUSIVE,
        cudf::null_policy::INCLUDE,
        stream,
        mr);
  }

  auto partCols =
      selectColumns(sortedInput, partitionKeyChannels_);
  std::vector<cudf::groupby::scan_request> requests(1);
  requests[0].values = valuesCol;
  requests[0].aggregations.push_back(
      cudf::make_rank_aggregation<
          cudf::groupby_scan_aggregation>(
          method,
          colOrder,
          cudf::null_policy::INCLUDE,
          nullOrd));

  cudf::groupby::groupby grouper(
      cudf::table_view(partCols),
      cudf::null_policy::INCLUDE,
      cudf::sorted::YES,
      std::vector<cudf::order>(
          partitionKeyChannels_.size(),
          cudf::order::ASCENDING),
      std::vector<cudf::null_order>(
          partitionKeyChannels_.size(),
          cudf::null_order::BEFORE));

  auto scanResult = grouper.scan(requests, stream, mr);
  auto& aggResults = scanResult.second;
  VELOX_CHECK_EQ(aggResults.size(), 1);
  VELOX_CHECK_EQ(aggResults[0].results.size(), 1);
  return std::move(aggResults[0].results[0]);
}

std::unique_ptr<cudf::column>
CudfWindow::computeAggregateColumn(
    cudf::table_view const& sortedInput,
    const WindowFunctionSpec& spec,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  auto agg = makeRollingAggregation(spec);
  auto [preceding, following] = toWindowBounds(spec.frame);
  auto inputCol = sortedInput.column(spec.inputChannel);

  if (partitionKeyChannels_.empty()) {
    // Non-grouped rolling window.
    cudf::size_type prec = preceding.is_unbounded()
        ? sortedInput.num_rows()
        : preceding.value();
    cudf::size_type foll = following.is_unbounded()
        ? sortedInput.num_rows()
        : following.value();
    return cudf::rolling_window(
        inputCol, prec, foll, 1, *agg, stream, mr);
  }

  auto partCols =
      selectColumns(sortedInput, partitionKeyChannels_);
  return cudf::grouped_rolling_window(
      cudf::table_view(partCols),
      inputCol,
      preceding,
      following,
      1,
      *agg,
      stream,
      mr);
}

std::unique_ptr<cudf::table> CudfWindow::computeOutputTable(
    std::unique_ptr<cudf::table> input,
    rmm::cuda_stream_view stream) const {
  auto mr = cudf::get_current_device_resource_ref();
  auto numFunctions = functionSpecs_.size();

  // Empty input: append empty result columns.
  if (input->num_rows() == 0) {
    auto columns = input->release();
    for (size_t i = 0; i < numFunctions; ++i) {
      auto expectedType = veloxToCudfDataType(
          outputType_->childAt(inputType_->size() + i));
      columns.push_back(
          cudf::make_empty_column(expectedType));
    }
    return std::make_unique<cudf::table>(
        std::move(columns));
  }

  auto inputView = input->view();

  // Build combined sort key views: partition keys (asc,
  // nulls-before) followed by sort keys (user-specified).
  std::vector<cudf::column_view> sortKeyViews;
  std::vector<cudf::order> colOrders;
  std::vector<cudf::null_order> colNullOrders;

  for (auto ch : partitionKeyChannels_) {
    sortKeyViews.push_back(inputView.column(ch));
    colOrders.push_back(cudf::order::ASCENDING);
    colNullOrders.push_back(cudf::null_order::BEFORE);
  }
  for (size_t i = 0; i < sortKeyChannels_.size(); ++i) {
    sortKeyViews.push_back(
        inputView.column(sortKeyChannels_[i]));
    colOrders.push_back(sortOrders_[i]);
    colNullOrders.push_back(sortNullOrders_[i]);
  }

  // Sort if needed.
  std::unique_ptr<cudf::column> sortedOrder;
  cudf::table_view sortedView;
  std::unique_ptr<cudf::table> sortedTable;

  if (windowNode_->inputsSorted() || sortKeyViews.empty()) {
    sortedView = inputView;
  } else {
    sortedOrder = cudf::stable_sorted_order(
        cudf::table_view(sortKeyViews),
        colOrders,
        colNullOrders,
        stream,
        mr);
    sortedTable = cudf::gather(
        inputView,
        sortedOrder->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream,
        mr);
    sortedView = sortedTable->view();
  }

  // Compute each window function on the sorted data.
  std::vector<std::unique_ptr<cudf::column>> resultCols;
  resultCols.reserve(numFunctions);

  for (size_t i = 0; i < numFunctions; ++i) {
    auto const& spec = functionSpecs_[i];
    std::unique_ptr<cudf::column> col;

    if (isRankLike(spec.kind)) {
      col = computeRankColumn(
          sortedView, spec.kind, stream);
    } else {
      col = computeAggregateColumn(
          sortedView, spec, stream);
    }

    // Cast to expected output type if needed.
    auto expectedType = veloxToCudfDataType(
        outputType_->childAt(inputType_->size() + i));
    if (col->type() != expectedType) {
      col = cudf::cast(
          col->view(), expectedType, stream, mr);
    }

    resultCols.push_back(std::move(col));
  }

  // The result columns no longer depend on the temporary sorted table.
  sortedTable.reset();

  // Scatter results back to original order if we sorted.
  if (sortedOrder) {
    for (auto& col : resultCols) {
      auto target = cudf::allocate_like(
          col->view(),
          col->size(),
          cudf::mask_allocation_policy::RETAIN,
          stream,
          mr);
      auto scattered = cudf::scatter(
          cudf::table_view{{col->view()}},
          sortedOrder->view(),
          cudf::table_view{{target->view()}},
          stream,
          mr);
      col = std::move(scattered->release()[0]);
    }
    sortedOrder.reset();
  }

  // Assemble output: all input columns + result columns.
  auto columns = input->release();
  for (auto& col : resultCols) {
    columns.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace facebook::velox::cudf_velox
