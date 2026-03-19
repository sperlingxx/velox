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

#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <velox/cudf/kernels/TopNRowNumber.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {

::velox::cudf_kernels::RankFunction toKernelRank(
    core::TopNRowNumberNode::RankFunction f) {
  switch (f) {
    case core::TopNRowNumberNode::RankFunction::kRowNumber:
      return ::velox::cudf_kernels::RankFunction::kRowNumber;
    case core::TopNRowNumberNode::RankFunction::kRank:
      return ::velox::cudf_kernels::RankFunction::kRank;
    case core::TopNRowNumberNode::RankFunction::kDenseRank:
      return ::velox::cudf_kernels::RankFunction::kDenseRank;
  }
  VELOX_UNREACHABLE();
}

} // namespace

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : exec::Operator(
          driverCtx,
          node->outputType(),
          operatorId,
          node->id(),
          "CudfTopNRowNumber"),
      NvtxHelper(
          nvtx3::rgb{255, 165, 0}, // Orange
          operatorId,
          fmt::format("[{}]", node->id())),
      limit_(node->limit()),
      generateRowNumber_(node->generateRowNumber()),
      rankFunction_(node->rankFunction()),
      planNode_(node) {
  auto inputType = node->sources()[0]->outputType();

  for (const auto& key : node->partitionKeys()) {
    partitionKeyIndices_.push_back(
        exec::exprToChannel(key.get(), inputType));
  }

  // Build sort column indices covering partition + sorting keys for the
  // initial stable sort.  Partition keys come first so that rows sharing the
  // same partition are contiguous after sorting (required by groupby scan).
  std::vector<cudf::size_type> allSortCols;
  std::vector<cudf::order> allOrders;
  std::vector<cudf::null_order> allNullOrders;

  for (auto idx : partitionKeyIndices_) {
    allSortCols.push_back(idx);
    allOrders.push_back(cudf::order::ASCENDING);
    allNullOrders.push_back(cudf::null_order::BEFORE);
  }

  for (size_t i = 0; i < node->sortingKeys().size(); ++i) {
    auto channel =
        exec::exprToChannel(node->sortingKeys()[i].get(), inputType);
    allSortCols.push_back(channel);

    auto const& order = node->sortingOrders()[i];
    auto cudfOrder = order.isAscending() ? cudf::order::ASCENDING
                                         : cudf::order::DESCENDING;
    auto cudfNull = (order.isNullsFirst() ^ !order.isAscending())
        ? cudf::null_order::BEFORE
        : cudf::null_order::AFTER;

    allOrders.push_back(cudfOrder);
    allNullOrders.push_back(cudfNull);
  }

  // Combined partition + sorting specification for use in getOutput.
  sortKeyIndices_ = std::move(allSortCols);
  sortOrders_ = std::move(allOrders);
  nullOrders_ = std::move(allNullOrders);
}

void CudfTopNRowNumber::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  if (limit_ == 0 || input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputBatches_.push_back(cudfInput);
}

RowVectorPtr CudfTopNRowNumber::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (inputBatches_.empty()) {
    finished_ = true;
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Concatenate all input batches.
  std::vector<cudf::table_view> views;
  std::vector<rmm::cuda_stream_view> inputStreams;
  for (auto& batch : inputBatches_) {
    views.push_back(batch->getTableView());
    inputStreams.push_back(batch->stream());
  }
  cudf::detail::join_streams(inputStreams, stream);

  std::unique_ptr<cudf::table> concatenated;
  if (views.size() == 1) {
    concatenated = std::make_unique<cudf::table>(views[0], stream, mr);
  } else {
    concatenated = cudf::concatenate(views, stream, mr);
  }

  // Sort by partition keys + sorting keys.
  auto sortKeysView = concatenated->view().select(sortKeyIndices_);
  auto sortedIndices = cudf::stable_sorted_order(
      sortKeysView, sortOrders_, nullOrders_, stream, mr);
  auto sorted = cudf::gather(
      concatenated->view(), sortedIndices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  // Run the GPU topN kernel.
  auto [filtered, rankCol] = ::velox::cudf_kernels::topNRowNumber(
      sorted->view(),
      partitionKeyIndices_,
      toKernelRank(rankFunction_),
      limit_,
      generateRowNumber_,
      stream,
      mr);

  // Build the output table.  If generateRowNumber_ is true, append the rank
  // column to the data columns.
  std::unique_ptr<cudf::table> result;
  if (generateRowNumber_ && rankCol) {
    auto cols = filtered->release();
    cols.push_back(std::move(rankCol));
    result = std::make_unique<cudf::table>(std::move(cols));
  } else {
    result = std::move(filtered);
  }

  auto const numRows = result->num_rows();
  stream.synchronize();

  inputBatches_.clear();
  finished_ = true;

  return std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      outputType_,
      numRows,
      std::move(result),
      stream);
}

void CudfTopNRowNumber::noMoreInput() {
  Operator::noMoreInput();
  if (inputBatches_.empty()) {
    finished_ = true;
  }
}

bool CudfTopNRowNumber::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
