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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>

#include <deque>

namespace facebook::velox::cudf_velox {

enum class WindowFunctionKind {
  kRowNumber,
  kRank,
  kDenseRank,
  kSum,
  kMin,
  kMax,
  kCount,
  kAvg,
};

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node);

/// GPU implementation of the Window operator using cuDF.
/// Supports rank-like functions (row_number, rank, dense_rank)
/// via cudf::groupby::scan, and aggregate window functions
/// (sum, min, max, count, avg) via cudf::grouped_rolling_window.
class CudfWindow : public exec::Operator, public NvtxHelper {
 public:
  /// Builds a cuDF-backed window operator for one Velox WindowNode.
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty();
  }

  /// Accepts one GPU batch from upstream and routes it to the sorted or
  /// blocking path.
  void addInput(RowVectorPtr input) override;

  /// Drains any cached input into final output batches once upstream is done.
  void noMoreInput() override;

  /// Returns one ready output batch at a time.
  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    refreshFinished();
    return isFinished_;
  }

  /// Releases cached input and output state.
  void close() override;

  /// Returns true for rank-like functions that do not consume a value column.
  static bool isRankLike(WindowFunctionKind kind);

 private:
  struct WindowFunctionSpec {
    WindowFunctionKind kind;
    // Column channel for aggregate input. -1 for rank-like,
    // 0 for count(*) (values column is unused but must be valid).
    cudf::size_type inputChannel;
    core::WindowNode::Frame frame;
    cudf::null_policy countNullPolicy;
  };

  /// Maps a parsed window function kind to the cuDF rank method enum.
  static cudf::rank_method toRankMethod(WindowFunctionKind kind);

  /// Creates the cuDF rolling aggregation object for one aggregate window
  /// function.
  std::unique_ptr<cudf::rolling_aggregation>
  makeRollingAggregation(const WindowFunctionSpec& spec) const;

  /// Converts a Velox frame definition into cuDF window bounds.
  std::pair<cudf::window_bounds, cudf::window_bounds>
  toWindowBounds(const core::WindowNode::Frame& frame) const;

  /// Builds a temporary struct view for multi-column ORDER BY rank operations.
  cudf::column_view multiSortKeyStructView(
      cudf::table_view const& sortedInput) const;

  /// Computes one rank-like output column on already sorted input.
  std::unique_ptr<cudf::column> computeRankColumn(
      cudf::table_view const& sortedInput,
      WindowFunctionKind kind,
      rmm::cuda_stream_view stream) const;

  /// Computes one aggregate window output column on already sorted input.
  std::unique_ptr<cudf::column> computeAggregateColumn(
      cudf::table_view const& sortedInput,
      const WindowFunctionSpec& spec,
      rmm::cuda_stream_view stream) const;

  /// Runs the full window computation for one input table and appends the
  /// resulting window columns.
  std::unique_ptr<cudf::table> computeOutputTable(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream) const;

  /// Materializes a table_view into an owning CudfVector on the target stream.
  CudfVectorPtr materializeTableView(
      cudf::table_view view,
      const RowTypePtr& type,
      rmm::cuda_stream_view stream) const;

  /// Wraps an owning cuDF table into a CudfVector without copying the GPU data.
  CudfVectorPtr wrapOwnedTable(
      std::unique_ptr<cudf::table> table,
      const RowTypePtr& type,
      rmm::cuda_stream_view stream) const;

  /// Extracts one partition-key row from the input table as a 1-row table_view
  /// so boundary partitions can be compared without materializing an owning table.
  cudf::table_view extractPartitionKeyRow(
      cudf::table_view input,
      cudf::size_type row,
      rmm::cuda_stream_view stream) const;

  /// Returns true if two 1-row partition-key table views belong to the same
  /// logical partition.
  bool isSamePartitionKey(
      cudf::table_view lhs,
      cudf::table_view rhs,
      rmm::cuda_stream_view stream) const;

  /// Finds the start row of the trailing partition in a sorted table.
  cudf::size_type trailingPartitionStartRow(
      cudf::table_view input,
      rmm::cuda_stream_view stream) const;

  /// Checks whether the buffered input has exceeded the configured GPU batch
  /// target.
  bool batchTargetReached(vector_size_t rows, uint64_t bytes) const;

  /// Wraps one computed cuDF table into the output queue if it is non-empty.
  void enqueueOutputTable(
      std::unique_ptr<cudf::table> outputTable,
      rmm::cuda_stream_view stream);

  /// Caches one input batch. In the partition-aware path, the size target is
  /// only a trigger to attempt a flush; partition completeness still wins, so
  /// whole partitions may stay cached past the nominal threshold.
  void cacheNextBatch(
      CudfVectorPtr input,
      rmm::cuda_stream_view stream);

  /// Flushes cached input into output. In the partition-aware path, keeps the
  /// trailing boundary partition cached for the next batch.
  void flushCachedInput(
      rmm::cuda_stream_view stream,
      bool isPartitionAware);

  /// Builds the next flushable input table for the partition-aware path.
  /// If the current flush candidate does not yet end on a partition boundary,
  /// the trailing partition rows stay in `nextCachedInputs` for the next round.
  std::unique_ptr<cudf::table> preparePartitionAwareFlush(
      std::vector<CudfVectorPtr> cachedInputs,
      rmm::cuda_stream_view stream,
      std::vector<CudfVectorPtr>& nextCachedInputs);

  /// Refreshes the finished flag from cached input and queued output state.
  void refreshFinished();

  std::shared_ptr<const core::WindowNode> windowNode_;
  RowTypePtr inputType_;
  RowTypePtr partitionKeyType_;
  const bool isFlushByPartition_;

  /// Shared cache for both modes: unsorted input accumulates here until the
  /// final blocking flush, while sorted input uses it as the rolling cache for
  /// prefix flushing and boundary-partition carry-over.
  std::vector<CudfVectorPtr> cachedInputs_;

  /// Approximate size of the currently cached sorted input. Unsorted mode does
  /// not consult these counters when deciding when to flush.
  vector_size_t cachedRows_{0};
  uint64_t cachedBytes_{0};

  /// Output batches ready to be returned from getOutput().
  std::deque<CudfVectorPtr> outputQueue_;
  bool isFinished_{false};

  std::vector<cudf::size_type> partitionKeyChannels_;
  std::vector<cudf::order> partitionKeyOrders_;
  std::vector<cudf::null_order> partitionKeyNullOrders_;
  std::vector<cudf::size_type> sortKeyChannels_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> sortNullOrders_;
  std::vector<WindowFunctionSpec> functionSpecs_;

  // Scratch storage for the struct column_view children returned by
  // multiSortKeyStructView. Mutable because computeRankColumn is const.
  mutable std::vector<cudf::column_view> sortKeyStructChildren_;
};

} // namespace facebook::velox::cudf_velox
