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

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

namespace test {
class CudfTopNRowNumberTestHelper;
}

/// GPU TopNRowNumber for limit=1 rank-like windows.
///
/// row_number keeps the single first row in each partition. rank and dense_rank
/// keep every row in the first peer group in each partition.
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    return !noMoreInput_ && passthroughOutputs_.empty();
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return finished_ && passthroughOutputs_.empty();
  }

  static bool shouldReplace(
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void spillSortedRun();
  void compactSortedRunsForMerge();
  void initializeSortedRunReaders();
  struct PausedMergeStats {
    uint64_t sourceChunks{0};
    uint64_t sourceRows{0};
    uint64_t sourceBytes{0};
    uint64_t outputBatches{0};
    uint64_t outputRows{0};
    uint64_t outputBytes{0};
    uint64_t maxResidentRows{0};
    uint64_t maxResidentBytes{0};
    uint64_t maxOutputBytes{0};
    uint64_t maxActiveRuns{0};
  };
  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
    // A paused reader owns at most one bounded chunk. chunkOffset identifies
    // the not-yet-consumed suffix without copying it into a growing carry.
    std::unique_ptr<cudf::table> chunk;
    cudf::size_type chunkOffset{0};
    uint64_t chunkBytes{0};
  };

  class CompactionPermit {
   public:
    explicit CompactionPermit(size_t concurrency);
    ~CompactionPermit();

    CompactionPermit(const CompactionPermit&) = delete;
    CompactionPermit& operator=(const CompactionPermit&) = delete;

    bool ready() const;
    bool hasWaitFuture() const;
    ContinueFuture takeWaitFuture();

   private:
    static size_t testingActivePermits();

    struct State;
    std::unique_ptr<State> state_;

    friend class test::CudfTopNRowNumberTestHelper;
  };

  /// Overrides production memory limits for deterministic spill/merge tests.
  /// All values must be positive. Tests must restore the defaults afterwards.
  static void testingSetMemoryLimits(
      uint64_t candidateRunBytes,
      uint64_t mergeChunkBytes,
      uint64_t outputChunkBytes,
      cudf::size_type maxOutputRows);
  static void testingResetMemoryLimits();

  bool loadPausedChunk(
      SortedRun& run,
      rmm::cuda_stream_view stream,
      PausedMergeStats& stats);
  std::unique_ptr<cudf::table> mergeNextPausedBatch(
      std::vector<SortedRun*>& runs,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finished,
      PausedMergeStats& stats);
  std::unique_ptr<cudf::table> mergeNextSortedBatch(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finalBatch);
  std::unique_ptr<cudf::table> reduceSortedBatchToTopOne(
      std::unique_ptr<cudf::table> sorted,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void updateSortedPartitionState(
      cudf::table_view sorted,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::unique_ptr<cudf::table> appendGeneratedRank(
      std::unique_ptr<cudf::table> table,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;
  uint64_t measureTableBytes(
      std::unique_ptr<cudf::table>& table,
      const TypePtr& type,
      rmm::cuda_stream_view stream);
  void setPendingOutput(std::unique_ptr<cudf::table> output);
  CudfVectorPtr takePendingOutput();
  CudfVectorPtr computeNextSortedOutput();
  void cleanupSpillFiles();
  void cleanupSpillStateAfterFailure(std::string_view context) noexcept;
  void prepareSpilledOutput();

  CudfVectorPtr computeLimitOneRowNumber(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeLimitOneRankLike(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::unique_ptr<cudf::table> reduceToCandidates(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  const int32_t limit_;
  const core::TopNRowNumberNode::RankFunction rankFunction_;
  const bool generateRowNumber_;
  const RowTypePtr inputType_;
  const core::PlanNodeId diagnosticNodeId_;
  // All stateful work stays on one stream. candidates_, Parquet readers, and
  // merge/output state outlive a single Operator call, so changing streams
  // between calls would lose their producer dependency.
  const rmm::cuda_stream_view stateStream_;

  std::vector<cudf::size_type> partitionKeys_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::size_type> allKeyIndices_;
  std::vector<cudf::order> columnOrders_;
  std::vector<cudf::null_order> nullOrders_;

  // A boolean partition key whose name starts with this marker makes Top-N
  // conditional: false rows are known singleton/pass-through partitions and
  // can be emitted immediately; only true rows enter rank state.  This keeps
  // one input scan while avoiding state for high-volume unaffected rows.
  std::optional<cudf::size_type> passthroughKey_;
  std::deque<CudfVectorPtr> passthroughOutputs_;

  // Declared before GPU merge state so normal member destruction releases all
  // reader/output owners before returning the process-wide admission slot.
  std::unique_ptr<CompactionPermit> compactionPermit_;
  std::vector<CudfVectorPtr> inputs_;
  // Incremental per-partition Top-1 state. Every input batch is reduced first,
  // then merged with this already-reduced state and reduced again. For
  // row_number this is at most one row per partition; rank/dense_rank retain
  // only ties for the current best sort key. Device residency therefore
  // depends on candidate cardinality, not total input rows.
  std::unique_ptr<cudf::table> candidates_;
  uint64_t bufferedBytes_{0};
  uint64_t nextDiagnosticBufferedBytes_{512ULL << 20};
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  // The merge is globally sorted. Only one row of cross-batch state is needed:
  // the current partition and its first sort peer. Whole partitions are never
  // retained in device memory.
  bool hasCurrentPartition_{false};
  std::unique_ptr<cudf::table> currentPartitionKey_;
  std::unique_ptr<cudf::table> currentPeerKey_;
  std::unique_ptr<cudf::table> pendingOutput_;
  cudf::size_type pendingOutputOffset_{0};
  uint64_t pendingOutputBytes_{0};
  PausedMergeStats outputMergeStats_;
  bool outputMergeEndLogged_{false};
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};

  friend class test::CudfTopNRowNumberTestHelper;
};

} // namespace facebook::velox::cudf_velox
