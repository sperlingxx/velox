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

#include "velox/exec/Operator.h"
#include "velox/expression/FunctionSignature.h"

#include <cudf/groupby.hpp>

#include <unordered_map>

namespace facebook::velox::cudf_velox {

// Forward declaration
struct CudfFunctionSpec;

class CudfHashAggregation : public exec::Operator, public NvtxHelper {
 public:
  struct Aggregator {
    core::AggregationNode::Step step;
    bool is_global;
    cudf::aggregation::Kind kind;
    uint32_t inputIndex;
    VectorPtr constant;
    TypePtr resultType;

    virtual void addGroupbyRequest(
        cudf::table_view const& tbl,
        std::vector<cudf::groupby::aggregation_request>& requests) = 0;

    virtual std::unique_ptr<cudf::column> doReduce(
        cudf::table_view const& input,
        TypePtr const& outputType,
        rmm::cuda_stream_view stream) = 0;

    virtual std::unique_ptr<cudf::column> makeOutputColumn(
        std::vector<cudf::groupby::aggregation_result>& results,
        rmm::cuda_stream_view stream) = 0;

   protected:
    Aggregator(
        core::AggregationNode::Step step,
        cudf::aggregation::Kind kind,
        uint32_t inputIndex,
        VectorPtr constant,
        bool isGlobal,
        const TypePtr& _resultType)
        : step(step),
          is_global(isGlobal),
          kind(kind),
          inputIndex(inputIndex),
          constant(constant),
          resultType(_resultType) {}
  };

  CudfHashAggregation(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::AggregationNode> const& aggregationNode);

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  // Setups the projections for accessing grouping keys stored in grouping
  // set.
  // For 'groupingKeyInputChannels', the index is the key column index from
  // the grouping set, and the value is the key column channel from the input.
  // For 'outputChannelProjections', the index is the key column channel from
  // the output, and the value is the key column index from the grouping set.
  void setupGroupingKeyChannelProjections(
      std::vector<column_index_t>& groupingKeyInputChannels,
      std::vector<column_index_t>& groupingKeyOutputChannels) const;

  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<Aggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream);
  CudfVectorPtr doGlobalAggregation(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream);
  CudfVectorPtr getDistinctKeys(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      rmm::cuda_stream_view stream);

  CudfVectorPtr releaseAndResetPartialOutput();

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;
  std::vector<column_index_t> aggregationInputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  std::vector<std::unique_ptr<Aggregator>> aggregators_;
  std::vector<std::unique_ptr<Aggregator>> intermediateAggregators_;
  // Used for kSingle streaming: partial-step aggregators (raw -> intermediate)
  // and final-step aggregators (intermediate -> final).
  std::vector<std::unique_ptr<Aggregator>> partialAggregators_;
  std::vector<std::unique_ptr<Aggregator>> finalAggregators_;

  // Partial aggregation is the first phase of aggregation. e.g. count(*) when
  // in partial phase will do a count_agg but in the final phase will do a sum
  // of the previous calculated counts
  const bool isPartialOutput_;
  // Global means it's an aggregation without groupby. Like cudf::reduce
  const bool isGlobal_;
  // Distinct means it's a count distinct on the groupby keys, without any
  // aggregations
  const bool isDistinct_;
  // Single means it's a single step aggregation (partial + final combined).
  const bool isSingleStep_;
  // Streaming aggregation is enabled when (a) the aggregate is grouped (not
  // global) AND (b) either no companion aggregates are present OR the step
  // is kPartial with all-`_partial` companions (the latter carve-out lets
  // Spark MPP partial-stage groupbys stream; see CudfHashAggregation::initialize
  // for the gating logic and the canStreamPartialCompanions flag).
  bool streamingEnabled_{true};

  // Maximum memory usage for partial aggregation.
  const int64_t maxPartialAggregationMemoryUsage_;
  // Number of rows received in the input so far.
  int64_t numInputRows_ = 0;

  bool finished_ = false;

  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<cudf_velox::CudfVectorPtr> inputs_;

  TypePtr inputType_;

  // This is for partial aggregation to keep reducing the amount of memory it
  // has to hold on to.
  void computePartialGroupbyStreaming(CudfVectorPtr tbl);

  void computePartialDistinctStreaming(CudfVectorPtr tbl);

  void computeFinalGroupbyStreaming(CudfVectorPtr tbl);

  void computeSingleGroupbyStreaming(CudfVectorPtr tbl);

  CudfVectorPtr bufferedResult_;
  RowTypePtr bufferedResultType_;

  // Partial-streaming convergence detection. After each cross-batch
  // concat+merge in computePartialGroupbyStreaming, compare the current
  // bufferedResult_ row count against the cumulative pre-aggregated input
  // row count.
  //
  // Threshold kPartialBypassThreshold = 0.75: derived from "if 2 batches
  // pre-aggregated to S rows each fail to combine below 1.5*S after merge,
  // future batches won't help" -- i.e. group-key ranges are effectively
  // disjoint and continued merging is wasted D2D work. 0.75 = 1.5/2 leaves
  // some margin around the threshold for noise. Anything > 0.75 is treated
  // as "not converging".
  //
  // If the ratio exceeds 0.75, switch to bypass mode: subsequent batches'
  // pre-aggregated results are flushed downstream directly without merging
  // into bufferedResult_. Avoids the O(N^2) D2D cost of repeatedly
  // concatenating with a growing buffered state that never compacts.
  // Typical trip cases: high-cardinality group keys whose per-batch key
  // ranges are disjoint (e.g., Q18 lineitem-by-l_orderkey, ratio = 1.0 on
  // first concat).
  //
  // partialBypassMode_ is sticky: once set, the operator stays in bypass
  // for the rest of its lifetime (no transition back to compacting mode).
  bool partialBypassMode_{false};
  int64_t partialCumulativeInputRows_{0};
};

// Step-aware aggregation function registry
// Map of function name -> Map of step -> signatures
using StepAwareAggregationRegistry = std::unordered_map<
    std::string,
    std::unordered_map<
        core::AggregationNode::Step,
        std::vector<exec::FunctionSignaturePtr>>>;

// Get the step-aware aggregation registry
StepAwareAggregationRegistry& getStepAwareAggregationRegistry();

// Register aggregation function signatures for a specific step
bool registerAggregationFunctionForStep(
    const std::string& name,
    core::AggregationNode::Step step,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true);

// Register step-aware builtin aggregation functions
bool registerStepAwareBuiltinAggregationFunctions(const std::string& prefix);

// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

bool canBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

// Utility functions
core::TypedExprPtr expandFieldReference(
    const core::TypedExprPtr& expr,
    const core::PlanNode* sourceNode);

bool canGroupingKeysBeEvaluatedByCudf(
    const std::vector<core::FieldAccessTypedExprPtr>& groupingKeys,
    const core::PlanNode* sourceNode,
    core::QueryCtx* queryCtx);

bool matchTypedCallAgainstSignatures(
    const core::CallTypedExpr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs);

} // namespace facebook::velox::cudf_velox
