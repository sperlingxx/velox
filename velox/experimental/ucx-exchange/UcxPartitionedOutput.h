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

#include <optional>

#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

/// This is the cudf equivalent of the PartitionedOutput operator for cudf.
/// Instead of serializing and segmenting the partitioned data into an
/// OutputBuffer, the UcxPartitionedOutput operator transfers entire
/// cudf::packed_columns corresponding to CudfVectors to other workers.
class UcxPartitionedOutput : public exec::Operator,
                             public cudf_velox::NvtxHelper {
 public:
  // Default minimum rows to accumulate before flushing. Matches HTTP
  // PartitionedOutput's ~10,000 row target. Overridable via
  // QueryConfig::kUcxPartitionedOutputBatchRows.
  static constexpr int64_t kDefaultTargetRowsPerChunk = 10'000;

  UcxPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. The action is to further process
  /// unprocessed input. If all input has been processed, 'this' is in
  /// a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  /// Do not accept another input while a large input is being partitioned a
  /// window at a time. The driver calls getOutput() to resume that work.
  bool needsInput() const override {
    return !noMoreInput_ && !hasActiveFlush();
  }

  /// Moves the shared output queue's backpressure future to the Driver.
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  /// Finished after all input windows have been enqueued and EOS published.
  bool isFinished() override;

 private:
  std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
  sharedQueueManager();

  // Heuristic method to derive the partition keys from the PartitionNode
  // specification.
  void initPartitionKeys(
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  // Partitions the cudf table view using the partition keys and a hash
  // function using the given stream.
  void hashPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Computes a Spark-compatible INT32 partition id from serialized range
  // boundaries, then routes rows by that explicit id.
  void rangePartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the cudf table view into equal sizes. This is used when
  // RoundRobin partitioning is requested but round robin on a
  // row-by-row basis is not meaningful for UCX exchange.
  void equalPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the table along the given offsets and enqueues each offset
  // to the corresponding partition, i.e. first split to the partition 0,
  // second split to partition 1 etc.
  void splitAndEnqueue(
      cudf::table_view tableView,
      std::vector<cudf::size_type> offsets,
      rmm::cuda_stream_view stream);

  const std::weak_ptr<UcxOutputQueueManager> queueManager_;
  std::vector<column_index_t> partitionKeyIndices_;
  std::string rangeBoundsJson_;
  std::unique_ptr<cudf::table> rangeBoundaries_;
  std::vector<cudf::order> rangeOrders_;
  std::vector<cudf::null_order> rangeNullOrders_;
  const size_t numPartitions_;

  const int pipelineId_;
  const int driverId_;
  /// Partial aggregation may need a large transient allocation to produce its
  /// next batch. Do not retain one of its oversized outputs across a queue
  /// wait; scan/join producers keep normal mid-source backpressure.
  const bool sourceNeedsOwnerBoundaryBackpressure_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;

  bool finished_{false};
  std::string spec_;

  // Used for switching columns when column order differs between input and
  // output.
  std::vector<uint32_t> remap_;

  /// Concatenates pending inputs and partitions/enqueues the merged result.
  /// Each invocation advances at most one hash-partition residency window (or
  /// one SINGLE chunk), allowing the Driver to observe queue backpressure in
  /// between windows.
  void flushPending();

  /// Moves pending inputs into resumable flush state. For multiple inputs,
  /// owns the concatenated table; for one input, activeInputs_ owns the vector
  /// referenced by the active table view.
  void preparePendingFlush();

  /// Processes and enqueues one resumable work unit from the active flush.
  void advanceActiveFlush();

  /// Checks task-wide output queue pressure immediately after a work unit.
  void updateBackpressure();

  bool hasActiveFlush() const;
  cudf::table_view activeTableView();
  void clearActiveFlush();

  /// Accumulated CudfVectors awaiting flush.
  std::vector<cudf_velox::CudfVectorPtr> pendingInputs_;
  /// Total rows across pendingInputs_.
  int64_t pendingRows_{0};
  /// Source GPU bytes across pendingInputs_, captured before owners are freed.
  uint64_t pendingFlatBytes_{0};

  /// Ownership and cursor for a flush that yielded between partition windows.
  std::vector<cudf_velox::CudfVectorPtr> activeInputs_;
  std::unique_ptr<cudf::table> activeMergedTable_;
  std::optional<rmm::cuda_stream_view> activeStream_;
  uint64_t activeSourceFlatBytes_{0};
  cudf::size_type activeNextRow_{0};
  cudf::size_type activeRowsPerWindow_{0};
  /// A source larger than the task queue cap must not remain pinned while the
  /// producer is blocked. Such an oversized, multi-window source is drained
  /// before observing backpressure so its owner can be released promptly.
  bool activeDrainBeforeBackpressure_{false};
  /// Task-wide queue byte cap. Normal window sizing uses device headroom;
  /// this supplies only the emergency schema-width fallback under pressure.
  const uint64_t maxOutputBufferSize_;
  /// Configured row threshold for flushing (from QueryConfig).
  const int64_t targetRowsPerChunk_;
  /// Configured byte threshold for flushing and destination chunking.
  const uint64_t targetBytesPerChunk_;
  /// Optional libcudf hash_partition safety limit. Zero disables slicing.
  const int64_t hashPartitionInputBatchRows_;
  /// Source-row window for recombining safely sliced hash output.
  const int64_t hashPartitionWindowRows_;
};

} // namespace facebook::velox::ucx_exchange
