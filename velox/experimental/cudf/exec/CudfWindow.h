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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/type/Type.h"

#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

/// GPU-accelerated Window operator using cuDF.
///
/// Partitioned row_number/rank windows with an inputsSorted contract are
/// evaluated and emitted one input batch at a time while retaining only the
/// partition/order boundary state. Other windows store incoming GPU batches in
/// addInput(). Small inputs are concatenated and sorted when noMoreInput() is
/// called. Large, unsorted, partitioned inputs are written as sorted runs and
/// merged into complete-partition batches before window evaluation.
///
/// inputsSorted fast path: when WindowNode::inputsSorted() is true, this
/// operator skips stable_sorted_order and the full-table gather (see
/// WindowNode::inputsSorted() for the ordering contract). The flag is taken
/// from the plan as-is; Velox does not infer it here. Connectors / optimizers
/// must only set it when a Sort or ordered exchange actually guarantees
/// globally sorted input across concatenated batches with partition keys
/// ASCENDING / nulls-first, followed by the window ORDER BY keys (matching
/// sortOrders_/nullOrders_). Rank grouping with partition keys also assumes
/// that partition-key ordering when constructing the groupby grouper.
///
/// Memory: the buffered sorted path peaks at roughly concat output plus gather
/// copy plus window result columns. The narrow rank-like streaming path is
/// bounded by one input/output batch and constant-size boundary state.
///
/// Rank-like functions (row_number, rank, dense_rank) use
/// cudf::groupby::scan with cudf::make_rank_aggregation.
/// Aggregate windows and lag/lead use cudf::grouped_rolling_window.
class CudfWindow : public CudfOperatorBase {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  /// Returns true if every window function and frame in the plan node is
  /// supported by CudfWindow. On failure, @p reason is populated with a
  /// human-readable explanation when a non-null pointer is provided.
  static bool canRunOnGPU(const core::WindowNode& windowNode);

  static bool canRunOnGPU(
      const core::WindowNode& windowNode,
      std::string* reason);

  /// Returns true if the window function is supported by CudfWindow.
  static bool isSupportedWindowFunction(
      const std::string& baseName,
      size_t numArgs);

  bool needsInput() const override {
    return !noMoreInput_ && pendingOutput_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doNoMoreInput() override;

  void doClose() override;

 private:
  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
  };

  void spillSortedRun();
  void initializeSortedRunReaders();
  std::unique_ptr<cudf::table> mergeNextSortedBatch(bool& finalBatch);
  std::unique_ptr<cudf::table> takeCompletePartitions(
      std::unique_ptr<cudf::table> sorted,
      bool finalBatch);
  RowVectorPtr computeNextSortedOutput();
  RowVectorPtr computeOutput();
  void cleanupSpillFiles() noexcept;

  std::unique_ptr<cudf::column> computeStreamingRowNumberColumn(
      const cudf::table_view& sortedInput,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeStreamingRankColumn(
      const cudf::table_view& sortedInput,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> fixRankLikeColumn(
      std::unique_ptr<cudf::column> localResult,
      std::string_view functionName,
      const cudf::table_view& sortedInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  void updateRankLikeState(
      const cudf::table_view& sortedInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  cudf::size_type continuingPrefixSize(
      const cudf::table_view& sortedInput,
      const std::vector<cudf::size_type>& indices,
      const std::unique_ptr<cudf::table>& previousKey,
      const std::vector<cudf::order>& orders,
      const std::vector<cudf::null_order>& nullOrders,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute row_number/rank/dense_rank via cudf::groupby::scan or cudf::scan.
  void computeRankColumnsBatch(
      const cudf::table_view& sortedInput,
      const std::vector<std::pair<size_t, std::string>>& pendingRanks,
      cudf::groupby::groupby* rankGrouper,
      std::vector<std::unique_ptr<cudf::column>>& windowResultCols,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeLeadLagColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute first_value or last_value via cudf rolling window APIs.
  std::unique_ptr<cudf::column> computeNthValueColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      bool isFullPartition,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Compute aggregate window functions (sum, min, max, count, avg)
  // with frame bounds from the WindowNode.
  std::unique_ptr<cudf::column> computeAggregateColumn(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      const std::string& baseName,
      bool isCountStar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  // Dispatch ROWS window frames to grouped_rolling_window. RANGE frames are
  // handled separately by the batched grouped_range_rolling_window path.
  std::unique_ptr<cudf::column> invokeGroupedRollingWindow(
      const cudf::table_view& partKeys,
      cudf::column_view inputCol,
      const core::WindowNode::Function& func,
      std::unique_ptr<cudf::rolling_aggregation> agg,
      bool isFullPartition,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputRowType_;
  const bool rankLikeStreaming_;
  const rmm::cuda_stream_view stateStream_;

  std::vector<cudf::size_type> partitionKeyIndices_;
  std::vector<cudf::size_type> sortKeyIndices_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> nullOrders_;

  std::vector<CudfVectorPtr> inputBatches_;
  RowVectorPtr pendingOutput_;
  const uint64_t sortedRunBytes_;
  uint64_t bufferedBytes_{0};
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> mergeCarry_;
  std::unique_ptr<cudf::table> partitionCarry_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};

  // Sorted and concatenated input data, prepared in doNoMoreInput().
  std::unique_ptr<cudf::table> sortedData_;
  cudf::size_type logicalRowCount_{0};
  rmm::cuda_stream_view stream_{};
  bool streamAcquired_{false};

  bool hasRankLikeState_{false};
  std::unique_ptr<cudf::table> previousPartitionKey_;
  std::unique_ptr<cudf::table> previousOrderKey_;
  int64_t previousPartitionRows_{0};
  int64_t previousRank_{0};

  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
