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

#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"

#include <cudf/groupby.hpp>

#include <optional>

namespace facebook::velox::cudf_velox {

namespace test {
class CudfGroupbyTestHelper;
}

struct GroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;

  virtual void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) = 0;

  // Releases columns materialized solely to back views in the most recently
  // built request. This is normally handled when the next request replaces
  // the state, but an unsupported streaming probe must release it before
  // switching to levelled aggregation.
  virtual size_t releaseRequestState() {
    return 0;
  }

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) = 0;

  virtual ~GroupbyAggregator() = default;

 protected:
  GroupbyAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : step(step),
        inputIndex(inputIndex),
        constant(constant),
        resultType(resultType) {}
};

// Factory functions for creating groupby aggregators from plan nodes.
std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    std::optional<core::AggregationNode::Step> forcedStep = std::nullopt);

// Groupby-specific validation
bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

class CudfGroupby : public CudfOperatorBase {
 public:
  CudfGroupby(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::AggregationNode> const& aggregationNode);

  void initialize() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doNoMoreInput() override;

  void doClose() override;

 private:
  enum class FinalAggregationMode {
    kUndecided,
    kStreaming,
    kLevelled,
  };

  struct FinalAggregationRun {
    CudfVectorPtr data;
    uint64_t representedRows;
  };

  struct IntermediateAggregationRun {
    CudfVectorPtr data;
    uint64_t representedRows;
  };

  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  CudfVectorPtr releaseAndResetBufferedResult();

  void prepareInputForStateStream(const CudfVectorPtr& input);

  void computePartialGroupbyStreaming(CudfVectorPtr tbl);
  void computeFinalGroupbyStreaming(CudfVectorPtr tbl);
  void computeSingleGroupbyStreaming(CudfVectorPtr tbl);

  void addIntermediateAggregationRun(IntermediateAggregationRun run);
  IntermediateAggregationRun mergeIntermediateAggregationRuns(
      IntermediateAggregationRun left,
      IntermediateAggregationRun right,
      size_t outputLevel,
      bool finalizing);
  CudfVectorPtr drainIntermediateAggregationRuns();

  void addFinalAggregationRun(FinalAggregationRun run);
  FinalAggregationRun mergeFinalAggregationRuns(
      FinalAggregationRun left,
      FinalAggregationRun right,
      size_t outputLevel,
      bool finalizing);
  CudfVectorPtr drainFinalAggregationRuns();
  CudfVectorPtr finalizeStreamingFinalAggregation();

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;
  std::vector<column_index_t> aggregationInputChannels_;
  std::vector<CudfExpressionPtr> precomputedInputEvaluators_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  const core::PlanNodeId diagnosticNodeId_;
  // Aggregation state outlives an individual addInput() call. Keep all state
  // production, compaction, and finalization on one stream so dependencies are
  // not lost when successive exchange pages arrive on different streams.
  const rmm::cuda_stream_view stateStream_;
  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> intermediateAggregators_;
  // Used for kSingle streaming: partial-step aggregators (raw -> intermediate)
  // and final-step aggregators (intermediate -> final).
  std::vector<std::unique_ptr<GroupbyAggregator>> partialAggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> finalAggregators_;

  const bool isPartialOutput_;
  const bool isSingleStep_;
  // Companion aggregate names encode the Spark plan step. Internal streaming
  // compaction overrides that suffix with the intermediate step.
  bool streamingEnabled_{true};
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<CudfVectorPtr> inputs_;
  TypePtr inputType_;
  RowTypePtr bufferedResultType_;
  CudfVectorPtr bufferedResult_;

  // PARTIAL and SINGLE aggregation produce one compacted state per input
  // page. Keep those states in logarithmic size levels and merge only peers
  // of comparable weight. This avoids regrouping the complete accumulated
  // state for every page while retaining the configured partial-memory flush
  // boundary.
  std::vector<std::optional<IntermediateAggregationRun>> intermediateRunLevels_;
  uint64_t intermediateBufferedBytes_{0};
  uint64_t intermediateInputRunCount_{0};
  uint64_t intermediateRunMergeCount_{0};

  // Supported FINAL aggregates use cuDF's persistent hash state directly
  // across exchange pages. The choice is made from the first page and is never
  // changed after state has been accumulated. Unsupported aggregate kinds keep
  // using the all-GPU levelled-run implementation below.
  FinalAggregationMode finalAggregationMode_{FinalAggregationMode::kUndecided};
  std::unique_ptr<cudf::groupby::streaming_groupby> finalStreamingGroupby_;
  std::vector<size_t> finalStreamingRequestAggregationCounts_;
  cudf::size_type finalStreamingMaxDistinctKeys_{0};
  uint64_t finalStreamingBatchCount_{0};

  // One independently aggregated FINAL run per binary size level. A new run is
  // merged only with a peer at the same level, preventing the previous
  // state-plus-every-batch quadratic re-aggregation pattern.
  std::vector<std::optional<FinalAggregationRun>> finalRunLevels_;
  uint64_t finalInputRunCount_{0};
  uint64_t finalRunMergeCount_{0};

  friend class test::CudfGroupbyTestHelper;
};

} // namespace facebook::velox::cudf_velox
