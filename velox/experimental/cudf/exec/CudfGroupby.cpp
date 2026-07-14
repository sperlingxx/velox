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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>

#include <limits>

namespace {

using namespace facebook::velox;
using cudf_velox::castDecimal64InputToDecimal128;
using cudf_velox::CountInputKind;
using cudf_velox::finalizeDecimalAverage;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;
using cudf_velox::ResolvedAggregateInfo;
using cudf_velox::serializeDecimalPartialOrIntermediateState;
using cudf_velox::validateIntermediateColumnType;

constexpr const char* kFinalAggregationInputRuns =
    "cudfFinalAggregationInputRuns";
constexpr const char* kFinalAggregationRunMerges =
    "cudfFinalAggregationRunMerges";
constexpr const char* kFinalAggregationFinalizeMerges =
    "cudfFinalAggregationFinalizeMerges";
constexpr const char* kFinalAggregationMergeRows =
    "cudfFinalAggregationMergeRows";
constexpr const char* kFinalAggregationMergeBytes =
    "cudfFinalAggregationMergeBytes";
constexpr const char* kFinalAggregationMaxLevel =
    "cudfFinalAggregationMaxLevel";
constexpr const char* kFinalStreamingBatches = "cudfFinalStreamingBatches";
constexpr const char* kFinalStreamingInputRows =
    "cudfFinalStreamingInputRows";
constexpr const char* kFinalStreamingDistinctKeys =
    "cudfFinalStreamingDistinctKeys";
constexpr const char* kFinalStreamingOutputRows =
    "cudfFinalStreamingOutputRows";
constexpr const char* kFinalStreamingFallbackProbeColumnsReleased =
    "cudfFinalStreamingFallbackProbeColumnsReleased";

size_t aggregationRunLevel(uint64_t representedRows) {
  VELOX_CHECK_GT(representedRows, 0);
  size_t level = 0;
  while (representedRows >>= 1) {
    ++level;
  }
  return level;
}

uint64_t addRepresentedRows(uint64_t left, uint64_t right) {
  VELOX_CHECK_LE(left, std::numeric_limits<uint64_t>::max() - right);
  return left + right;
}

#define DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Name, name, KIND)                \
  struct Groupby##Name##Aggregator : GroupbyAggregator {                  \
    Groupby##Name##Aggregator(                                            \
        core::AggregationNode::Step step,                                 \
        uint32_t inputIndex,                                              \
        VectorPtr constant,                                               \
        const TypePtr& resultType)                                        \
        : GroupbyAggregator(step, inputIndex, constant, resultType) {}    \
                                                                          \
    void addGroupbyRequest(                                               \
        cudf::table_view const& tbl,                                      \
        std::vector<cudf::groupby::aggregation_request>& requests,        \
        rmm::cuda_stream_view stream) override {                          \
      auto& request = requests.emplace_back();                            \
      output_idx = requests.size() - 1;                                   \
      if (constant != nullptr) {                                          \
        auto scalar = cudf_velox::makeScalarFromConstantVector(constant); \
        constant_input = cudf::make_column_from_scalar(                   \
            *scalar, tbl.num_rows(), stream, get_temp_mr());              \
        request.values = constant_input->view();                          \
      } else {                                                            \
        request.values = tbl.column(inputIndex);                          \
      }                                                                   \
      request.aggregations.push_back(                                     \
          cudf::make_##name##_aggregation<cudf::groupby_aggregation>());  \
    }                                                                     \
                                                                          \
    size_t releaseRequestState() override {                               \
      const size_t released = constant_input == nullptr ? 0 : 1;          \
      constant_input.reset();                                             \
      return released;                                                    \
    }                                                                     \
                                                                          \
    std::unique_ptr<cudf::column> makeOutputColumn(                       \
        std::vector<cudf::groupby::aggregation_result>& results,          \
        rmm::cuda_stream_view stream,                                     \
        rmm::device_async_resource_ref mr) override {                     \
      auto col = std::move(results[output_idx].results[0]);               \
      const auto cudfType = cudf_velox::veloxToCudfDataType(resultType);  \
      if (col->type() != cudfType) {                                      \
        col = cudf::cast(*col, cudfType, stream, mr);                     \
      }                                                                   \
      return col;                                                         \
    }                                                                     \
                                                                          \
   private:                                                               \
    uint32_t output_idx;                                                  \
    std::unique_ptr<cudf::column> constant_input;                         \
  };

DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Sum, sum, SUM)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Min, min, MIN)
DEFINE_SIMPLE_GROUPBY_AGGREGATOR(Max, max, MAX)

// Decimal SUM and AVG aggregators are separate implementations, as they need to
// handle the VARBINARY encoded intermediate state for streaming aggregation.
// Due to the packing and unpacking of that intermediate state, and the special
// handling required for the decimal divide, we cannot just use the existing
// cudf::make_mean_aggregation class. Also, unlike other aggregators, these
// classes hold state (the decoded intermediate sum and count columns and
// associated indices) in order to guarantee a lifetime constraint between
// aggregation steps.

void addDecimalSumCountRequestsAfterDecode(
    cudf::column_view encodedColumn,
    int32_t scale,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(encodedColumn, scale, stream);
  decodedSum.swap(sumAndCount.sum);
  decodedCount.swap(sumAndCount.count);

  sumIdx = requests.size();
  auto& sumRequest = requests.emplace_back();
  sumRequest.values = decodedSum->view();
  sumRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  countIdx = requests.size();
  auto& countRequest = requests.emplace_back();
  countRequest.values = decodedCount->view();
  countRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

// Decodes serialized state and adds sum + count groupby requests, used by the
// intermediate step (both SUM and AVG) and the final AVG step. resultType is
// DECIMAL for final AVG and DECIMAL or VARBINARY for intermediate; VARBINARY
// carries no scale, so decode at scale 0.
void addDecimalDecodedSumCountRequests(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = resultType->isDecimal()
      ? getDecimalPrecisionScale(*resultType).second
      : 0;
  addDecimalSumCountRequestsAfterDecode(
      tbl.column(inputIndex),
      scale,
      requests,
      stream,
      sumIdx,
      countIdx,
      decodedSum,
      decodedCount);
}

void addDecimalFinalSumOnlyRequest(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& decodedSum) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = getDecimalPrecisionScale(*resultType).second;
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  auto sumAndCount = cudf_velox::deserializeDecimalSumState(
      tbl.column(inputIndex), scale, stream);
  decodedSum.swap(sumAndCount.sum);
  request.values = decodedSum->view();
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

void addDecimalRawPartialSingleSumRequest(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    std::vector<cudf::groupby::aggregation_request>& requests,
    bool includeCountAggregation,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& castedInput) {
  auto inputView = castDecimal64InputToDecimal128(
      tbl.column(inputIndex), castedInput, stream);
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  request.values = inputView;
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (includeCountAggregation) {
    request.aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE));
  }
}

struct GroupbyDecimalSumAggregator : GroupbyAggregator {
  GroupbyDecimalSumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    if (step == core::AggregationNode::Step::kIntermediate) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else if (step == core::AggregationNode::Step::kFinal) {
      addDecimalFinalSumOnlyRequest(
          tbl, inputIndex, resultType, requests, stream, sumIdx_, decodedSum_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          tbl,
          inputIndex,
          requests,
          step == core::AggregationNode::Step::kPartial,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  size_t releaseRequestState() override {
    const size_t released = static_cast<size_t>(decodedSum_ != nullptr) +
        static_cast<size_t>(decodedCount_ != nullptr) +
        static_cast<size_t>(castedInput_ != nullptr);
    decodedSum_.reset();
    decodedCount_.reset();
    castedInput_.reset();
    return released;
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    auto const cudfResType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfResType) {
      col = cudf::cast(*col, cudfResType, stream, mr);
    }
    return col;
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyDecimalAvgAggregator : GroupbyAggregator {
  GroupbyDecimalAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    if (step == core::AggregationNode::Step::kIntermediate ||
        step == core::AggregationNode::Step::kFinal) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          tbl,
          inputIndex,
          requests,
          step == core::AggregationNode::Step::kPartial ||
              step == core::AggregationNode::Step::kSingle,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  size_t releaseRequestState() override {
    const size_t released = static_cast<size_t>(decodedSum_ != nullptr) +
        static_cast<size_t>(decodedCount_ != nullptr) +
        static_cast<size_t>(castedInput_ != nullptr);
    decodedSum_.reset();
    decodedCount_.reset();
    castedInput_.reset();
    return released;
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kSingle) {
      auto count = std::move(results[sumIdx_].results[1]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kFinal) {
      auto count = std::move(results[countIdx_].results[0]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    // All four aggregation steps are handled above.
    VELOX_UNREACHABLE();
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyCountAggregator : GroupbyAggregator {
  GroupbyCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      CountInputKind inputKind,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, nullptr, resultType),
        inputKind_(inputKind) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    auto& request = requests.emplace_back();
    outputIndex_ = requests.size() - 1;
    // kCountAll and kNullConstant both submit a count-all-rows request;
    // kNullConstant overrides the result with zeros in makeOutputColumn.
    const bool countAll = (inputKind_ != CountInputKind::kColumn);
    // For raw input, count(*) can use any column (column 0) since we just
    // need a row count. For non-raw input (intermediate/final in streaming),
    // the input is partial results where column 0 is the grouping key;
    // we must use inputIndex to access the partial count column.
    request.values =
        tbl.column((countAll && exec::isRawInput(step)) ? 0 : inputIndex);
    std::unique_ptr<cudf::groupby_aggregation> aggRequest =
        exec::isRawInput(step)
        ? cudf::make_count_aggregation<cudf::groupby_aggregation>(
              countAll ? cudf::null_policy::INCLUDE
                       : cudf::null_policy::EXCLUDE)
        : cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    request.aggregations.push_back(std::move(aggRequest));
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[outputIndex_].results[0]);
    if (inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(zero, col->size(), stream, mr);
    }
    // cudf produces int32 for count but velox expects int64.
    const auto cudfOutputType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, mr);
    }
    return col;
  }

 private:
  CountInputKind inputKind_;
  uint32_t outputIndex_;
};

struct GroupbyMeanAggregator : GroupbyAggregator {
  GroupbyMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto& request = requests.emplace_back();
        meanIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        break;
      }
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal: {
        // In intermediate and final aggregation, the previously computed sum
        // and count are in the child columns of the input column.
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex).child(0);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto& request2 = requests.emplace_back();
        countIdx_ = requests.size() - 1;
        request2.values = tbl.column(inputIndex).child(1);
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        request2.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[meanIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[sumIdx_].results[1]);

        auto const size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kIntermediate: {
        // The difference between intermediate and partial is in where the
        // sum and count are coming from. In partial, since the input column is
        // the same, the sum and count are in the same agg result. In
        // intermediate, the input columns are different (it's the child
        // columns of the input column) and so the sum and count are in
        // different agg results.
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);

        auto size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            cudf_velox::veloxToCudfDataType(resultType),
            stream,
            mr);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  uint32_t meanIdx_;
  uint32_t sumIdx_;
  uint32_t countIdx_;
};

struct GroupbyStddevSampAggregator : GroupbyAggregator {
  GroupbyStddevSampAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    auto& request = requests.emplace_back();
    outputIdx_ = requests.size() - 1;
    request.values = tbl.column(inputIndex);

    switch (step) {
      case core::AggregationNode::Step::kSingle:
        // Use cuDF's built-in std aggregation with ddof=1 (sample stddev)
        request.aggregations.push_back(
            cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
        break;
      case core::AggregationNode::Step::kPartial:
        // Compute count, mean, m2 from raw values
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        break;
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal:
        // Input is struct(count, mean, m2) - use MERGE_M2 to merge
        request.aggregations.push_back(
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        break;
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[outputIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto count = std::move(results[outputIdx_].results[0]);
        auto mean = std::move(results[outputIdx_].results[1]);
        auto m2 = std::move(results[outputIdx_].results[2]);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
      }
      case core::AggregationNode::Step::kIntermediate: {
        auto merged = std::move(results[outputIdx_].results[0]);

        // Check if types already match expected output - avoid copies if so
        const auto& outputType = asRowType(resultType);
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfMeanType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        auto const cudfM2Type =
            cudf_velox::veloxToCudfDataType(outputType->childAt(2));

        auto mergedView = merged->view();
        bool typesMatch = mergedView.child(0).type() == cudfCountType &&
            mergedView.child(1).type() == cudfMeanType &&
            mergedView.child(2).type() == cudfM2Type;

        if (typesMatch) {
          // Types match - return merged directly to avoid device copies
          return merged;
        }

        // Types don't match - need to copy and cast (use output_mr since
        // these become part of the output)
        auto count =
            std::make_unique<cudf::column>(mergedView.child(0), stream, mr);
        auto mean =
            std::make_unique<cudf::column>(mergedView.child(1), stream, mr);
        auto m2 =
            std::make_unique<cudf::column>(mergedView.child(2), stream, mr);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
      }
      case core::AggregationNode::Step::kFinal: {
        // MERGE_M2 returns struct(count, mean, m2)
        // Compute sqrt(m2 / (count - 1)) with NULL where count < 2
        auto merged = std::move(results[outputIdx_].results[0]);
        auto mergedView = merged->view();
        auto countView = mergedView.child(0);
        auto m2View = mergedView.child(2);

        // count - 1 (binary_operation handles type promotion)
        cudf::numeric_scalar<double> one(1.0, true, stream, get_temp_mr());
        auto countMinus1 = cudf::binary_operation(
            countView,
            one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // m2 / (count - 1)
        auto variance = cudf::binary_operation(
            m2View,
            *countMinus1,
            cudf::binary_operator::DIV,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // sqrt(variance)
        auto stddev = cudf::unary_operation(
            *variance, cudf::unary_operator::SQRT, stream, get_temp_mr());

        // count >= 2
        cudf::numeric_scalar<int64_t> two(2, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            countView,
            two,
            cudf::binary_operator::GREATER_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());

        // Apply mask: where count < 2, result is NULL
        cudf::numeric_scalar<double> nullDouble(
            0.0, false, stream, get_temp_mr());
        return cudf::copy_if_else(*stddev, nullDouble, *validMask, stream, mr);
      }
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

 private:
  // Build a struct column with (count, mean, m2), casting to expected types.
  std::unique_ptr<cudf::column> makeM2StructColumn(
      std::unique_ptr<cudf::column> count,
      std::unique_ptr<cudf::column> mean,
      std::unique_ptr<cudf::column> m2,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    const auto& outputType = asRowType(resultType);
    auto const cudfCountType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(0));
    auto const cudfMeanType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(1));
    auto const cudfM2Type =
        cudf_velox::veloxToCudfDataType(outputType->childAt(2));
    if (count->type() != cudfCountType) {
      count = cudf::cast(*count, cudfCountType, stream, mr);
    }
    if (mean->type() != cudfMeanType) {
      mean = cudf::cast(*mean, cudfMeanType, stream, mr);
    }
    if (m2->type() != cudfM2Type) {
      m2 = cudf::cast(*m2, cudfM2Type, stream, mr);
    }

    auto const size = count->size();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(count));
    children.push_back(std::move(mean));
    children.push_back(std::move(m2));

    return std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::STRUCT),
        size,
        rmm::device_buffer{},
        rmm::device_buffer{},
        0,
        std::move(children));
  }

  uint32_t outputIdx_;
};

struct GroupbyCollectSetAggregator : GroupbyAggregator {
  GroupbyCollectSetAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(step, inputIndex, constant, resultType) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) override {
    VELOX_CHECK(
        constant == nullptr,
        "GroupbyCollectSetAggregator does not support constant input");
    auto& request = requests.emplace_back();
    outputIdx_ = requests.size() - 1;
    request.values = tbl.column(inputIndex);
    if (exec::isRawInput(step)) {
      request.aggregations.push_back(
          cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
              cudf::null_policy::EXCLUDE));
      return;
    }
    request.aggregations.push_back(
        cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>());
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view /*stream*/,
      rmm::device_async_resource_ref /*mr*/) override {
    return std::move(results[outputIdx_].results[0]);
  }

 private:
  uint32_t outputIdx_{0};
};

std::unique_ptr<GroupbyAggregator> createGroupbyAggregator(
    const ResolvedAggregateInfo& p) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalSumAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
    return std::make_unique<GroupbySumAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<GroupbyCountAggregator>(
        p.companionStep, p.inputIndex, *p.countInputKind, p.resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<GroupbyMinAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<GroupbyMaxAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalAvgAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
    return std::make_unique<GroupbyMeanAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "stddev_samp", 0) == 0) {
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "stddev", 0) == 0) {
    // stddev is an alias for stddev_samp
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "collect_set", 0) == 0) {
    return std::make_unique<GroupbyCollectSetAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    std::optional<core::AggregationNode::Step> forcedStep) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, outputType, constants);

  if (forcedStep.has_value()) {
    const auto numKeys = aggregationNode.groupingKeys().size();
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].companionStep = *forcedStep;
      params[i].resultType = outputType->childAt(numKeys + i);
    }
  }

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    aggregators.push_back(createGroupbyAggregator(p));
  }
  return aggregators;
}

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getGroupbyAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canGroupbyAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    // `mask` is NOT supported (in testing do not appear to be be applied and
    // return incorrect results )
    if (aggregate.mask) {
      return false;
    }

    if (isCountFunctionName(aggregate.call->name())) {
      continue;
    }

    // Check input expressions can be evaluated by cuDF, expand the input first.
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      std::vector<core::TypedExprPtr> exprs = {expandedInput};
      if (!canBeEvaluatedByCudf(exprs, queryCtx)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode.groupingKeys(), sourceNode, queryCtx)) {
    return false;
  }

  return true;
}

CudfGroupby::CudfGroupby(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfGroupby"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      diagnosticNodeId_(aggregationNode->id()),
      stateStream_(cudfGlobalStreamPool().get_stream()),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

void CudfGroupby::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      *aggregationNode_, groupingKeyInputChannels_, groupingKeyOutputChannels_);

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  const auto inputRowSchema = asRowType(inputType_);
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_,
      *operatorCtx_,
      inputRowSchema,
      groupingKeyInputChannels_);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  precomputedInputEvaluators_ = createAggregationInputEvaluators(
      aggregationInput.precomputedInputs, *operatorCtx_, inputRowSchema);
  aggregators_ = toGroupbyAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants);
  // The old blanket companion-aggregate guard forced every input batch into
  // inputs_ until noMoreInput(). Large MPP final aggregates consequently
  // retained several GiB of materialized exchange pages per
  // executor. Companion suffixes describe the external Spark plan step; for
  // streaming compaction we explicitly force the internal intermediate step
  // below, so these aggregates can be compacted incrementally as well.
  streamingEnabled_ = true;

  if (deviceMemoryDiagnosticsEnabled()) {
    for (const auto& aggregate : aggregationNode_->aggregates()) {
      LOG(WARNING) << "CUDF_GROUPBY_AGGREGATE node=" << diagnosticNodeId_
                   << " planStep="
                   << core::AggregationNode::toName(aggregationNode_->step())
                   << " function=" << aggregate.call->name();
    }
  }

  // Make aggregators for intermediate step when streaming is enabled.
  if (streamingEnabled_) {
    const bool isFinalOrSingle =
        aggregationNode_->step() == core::AggregationNode::Step::kFinal ||
        aggregationNode_->step() == core::AggregationNode::Step::kSingle;
    bufferedResultType_ = isFinalOrSingle
        ? getBufferedResultType(*aggregationNode_)
        : outputType_;

    std::vector<VectorPtr> nullConstants(numAggregates_);
    intermediateAggregators_ = toGroupbyAggregators(
        *aggregationNode_,
        core::AggregationNode::Step::kIntermediate,
        bufferedResultType_,
        nullConstants,
        core::AggregationNode::Step::kIntermediate);

    if (isSingleStep_) {
      partialAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kPartial,
          bufferedResultType_,
          aggregationInput.constants,
          core::AggregationNode::Step::kPartial);
      finalAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kFinal,
          outputType_,
          nullConstants,
          core::AggregationNode::Step::kFinal);
    }
  }

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfGroupby::computePartialGroupbyStreaming(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto preparedInput = prepareAggregationInput(
      tbl->getTableView(),
      static_cast<cudf::size_type>(tbl->size()),
      precomputedInputEvaluators_,
      inputTableStream,
      get_temp_mr());
  auto permutedInputView = preparedInput.tableView.select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      bufferedResultType_,
      inputTableStream,
      get_output_mr());

  // If we already have partial output, concatenate the new results with it.
  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(std::exchange(bufferedResult_, nullptr));
    tablesToConcat.push_back(std::move(groupbyOnInput));
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_temp_mr());

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());
    bufferedResult_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    bufferedResult_ = groupbyOnInput;
  }
}

void CudfGroupby::computeFinalGroupbyStreaming(CudfVectorPtr tbl) {
  VELOX_CHECK(tbl->stream().value() == stateStream_.value());
  const auto representedRows = static_cast<uint64_t>(tbl->size());
  auto preparedInput = prepareAggregationInput(
      tbl->getTableView(),
      static_cast<cudf::size_type>(tbl->size()),
      precomputedInputEvaluators_,
      stateStream_,
      get_temp_mr());
  auto permutedInputView = preparedInput.tableView.select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());

  const auto addLevelledRun = [&]() {
    // Reduce this page independently. Combining it with the entire prior state
    // here made high-cardinality FINAL aggregation quadratic in the number of
    // exchange pages. Binary levels only combine similarly sized runs.
    auto run = doGroupByAggregation(
        permutedInputView,
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        stateStream_,
        get_output_mr());
    if (run) {
      addFinalAggregationRun(
          FinalAggregationRun{std::move(run), representedRows});
    }
  };

  if (finalAggregationMode_ == FinalAggregationMode::kUndecided) {
    const auto configuredMaxDistinctKeys =
        CudfConfig::getInstance().groupbyStreamingMaxDistinctKeys;
    VELOX_CHECK_GE(configuredMaxDistinctKeys, 0);
    finalStreamingMaxDistinctKeys_ =
        static_cast<cudf::size_type>(configuredMaxDistinctKeys);
    if (finalStreamingMaxDistinctKeys_ == 0) {
      finalAggregationMode_ = FinalAggregationMode::kLevelled;
      LOG(INFO) << "CUDF_GROUPBY_STREAMING node=" << diagnosticNodeId_
                << " state=disabled maxDistinctKeys=0 mode=levelled_gpu";
    }
  }

  if (finalAggregationMode_ == FinalAggregationMode::kLevelled) {
    addLevelledRun();
    return;
  }

  // Build the real FINAL requests against this page. Their value views may be
  // top-level columns, struct children, or temporary decoded columns owned by
  // the aggregators. Pack those views next to the key views so
  // streaming_groupby can address every value by a stable column index for
  // this call.
  std::vector<cudf::groupby::aggregation_request> regularRequests;
  for (auto& aggregator : aggregators_) {
    aggregator->addGroupbyRequest(
        permutedInputView, regularRequests, stateStream_);
  }

  std::vector<cudf::column_view> packedColumns;
  packedColumns.reserve(
      groupingKeyOutputChannels_.size() + regularRequests.size());
  std::vector<cudf::size_type> keyIndices;
  keyIndices.reserve(groupingKeyOutputChannels_.size());
  for (const auto key : groupingKeyOutputChannels_) {
    VELOX_CHECK_LT(
        packedColumns.size(),
        static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()));
    keyIndices.push_back(static_cast<cudf::size_type>(packedColumns.size()));
    packedColumns.push_back(permutedInputView.column(key));
  }

  bool allRequestsSupported = !regularRequests.empty();
  std::vector<size_t> requestAggregationCounts;
  requestAggregationCounts.reserve(regularRequests.size());
  std::vector<cudf::groupby::streaming_aggregation_request>
      streamingRequests;
  for (auto& request : regularRequests) {
    VELOX_CHECK_LT(
        packedColumns.size(),
        static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto valueIndex =
        static_cast<cudf::size_type>(packedColumns.size());
    packedColumns.push_back(request.values);
    requestAggregationCounts.push_back(request.aggregations.size());
    for (auto& aggregation : request.aggregations) {
      VELOX_CHECK_NOT_NULL(aggregation);
      allRequestsSupported &= cudf::groupby::is_streaming_groupby_supported(
          request.values.type(), aggregation->kind);
      streamingRequests.push_back(
          cudf::groupby::streaming_aggregation_request{
              valueIndex, std::move(aggregation)});
    }
  }

  if (finalAggregationMode_ == FinalAggregationMode::kUndecided) {
    if (!allRequestsSupported) {
      finalAggregationMode_ = FinalAggregationMode::kLevelled;
      // The support probe above may have decoded page-sized intermediate
      // columns (for example decimal SUM/AVG) into aggregator-owned request
      // state. None of these views or aggregations are used by the levelled
      // path. Drop all view holders first, then release the owning columns
      // before the first levelled run is materialized.
      streamingRequests.clear();
      regularRequests.clear();
      packedColumns.clear();
      size_t releasedColumns = 0;
      for (auto& aggregator : aggregators_) {
        releasedColumns += aggregator->releaseRequestState();
      }
      if (releasedColumns > 0) {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            kFinalStreamingFallbackProbeColumnsReleased,
            RuntimeCounter(static_cast<int64_t>(releasedColumns)));
      }
      addLevelledRun();
      return;
    }
    finalStreamingRequestAggregationCounts_ = requestAggregationCounts;
    finalStreamingGroupby_ =
        std::make_unique<cudf::groupby::streaming_groupby>(
            keyIndices,
            streamingRequests,
            finalStreamingMaxDistinctKeys_,
            ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                            : cudf::null_policy::INCLUDE);
    finalAggregationMode_ = FinalAggregationMode::kStreaming;
    LOG(INFO) << "CUDF_GROUPBY_STREAMING node=" << diagnosticNodeId_
              << " state=selected maxDistinctKeys="
              << finalStreamingMaxDistinctKeys_ << " keys="
              << keyIndices.size() << " requests=" << streamingRequests.size();
  } else {
    VELOX_CHECK(
        finalAggregationMode_ == FinalAggregationMode::kStreaming);
    VELOX_CHECK(
        allRequestsSupported,
        "CudfGroupby FINAL request support changed after streaming state "
        "was initialized; refusing an unsafe mid-stream fallback");
    VELOX_CHECK(
        requestAggregationCounts == finalStreamingRequestAggregationCounts_,
        "CudfGroupby FINAL request shape changed after streaming state "
        "was initialized");
  }

  VELOX_CHECK_NOT_NULL(finalStreamingGroupby_);
  const auto batchSize = permutedInputView.num_rows();
  VELOX_USER_CHECK_LE(
      batchSize,
      finalStreamingMaxDistinctKeys_,
      "CudfGroupby streaming batch size ({}) exceeds configured capacity "
      "{} ({})",
      batchSize,
      CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys,
      finalStreamingMaxDistinctKeys_);
  VELOX_USER_CHECK_LE(
      static_cast<int64_t>(finalStreamingMaxDistinctKeys_) +
          static_cast<int64_t>(batchSize),
      static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "CudfGroupby streaming capacity ({}) plus batch size ({}) exceeds "
      "cuDF size_type max ({})",
      finalStreamingMaxDistinctKeys_,
      batchSize,
      std::numeric_limits<cudf::size_type>::max());
  finalStreamingGroupby_->aggregate(
      cudf::table_view{packedColumns}, stateStream_);
  ++finalStreamingBatchCount_;
  const auto distinctKeys = finalStreamingGroupby_->distinct_keys();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(kFinalStreamingBatches, RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        kFinalStreamingInputRows,
        RuntimeCounter(static_cast<int64_t>(representedRows)));
  }
  if (deviceMemoryDiagnosticsEnabled() &&
      (finalStreamingBatchCount_ == 1 ||
       finalStreamingBatchCount_ % 64 == 0)) {
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfGroupby node={} state=final_streaming.aggregate "
        "batch={} inputRows={} distinctKeys={} maxDistinctKeys={}",
        diagnosticNodeId_,
        finalStreamingBatchCount_,
        representedRows,
        distinctKeys,
        finalStreamingMaxDistinctKeys_));
  }
}

void CudfGroupby::addFinalAggregationRun(FinalAggregationRun run) {
  VELOX_CHECK_NOT_NULL(run.data);
  VELOX_CHECK_GT(run.representedRows, 0);
  ++finalInputRunCount_;

  auto level = aggregationRunLevel(run.representedRows);
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        kFinalAggregationInputRuns, RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        kFinalAggregationMaxLevel,
        RuntimeCounter(static_cast<int64_t>(level)));
  }

  for (;;) {
    if (finalRunLevels_.size() <= level) {
      finalRunLevels_.resize(level + 1);
    }
    if (!finalRunLevels_[level].has_value()) {
      finalRunLevels_[level] = std::move(run);
      return;
    }

    auto peer = std::move(*finalRunLevels_[level]);
    finalRunLevels_[level].reset();
    const auto representedRows =
        addRepresentedRows(peer.representedRows, run.representedRows);
    auto outputLevel = aggregationRunLevel(representedRows);
    // Both inputs occupied the same logarithmic interval, so their combined
    // weight must advance. Keep this explicit to prevent a malformed weight
    // from repeatedly colliding at the same level.
    outputLevel = std::max(outputLevel, level + 1);
    run = mergeFinalAggregationRuns(
        std::move(peer), std::move(run), outputLevel, false);
    level = outputLevel;
  }
}

CudfGroupby::FinalAggregationRun CudfGroupby::mergeFinalAggregationRuns(
    FinalAggregationRun left,
    FinalAggregationRun right,
    size_t outputLevel,
    bool finalizing) {
  VELOX_CHECK_NOT_NULL(left.data);
  VELOX_CHECK_NOT_NULL(right.data);
  VELOX_CHECK(left.data->stream().value() == stateStream_.value());
  VELOX_CHECK(right.data->stream().value() == stateStream_.value());

  const auto inputRows = static_cast<int64_t>(left.data->size()) +
      static_cast<int64_t>(right.data->size());
  const auto inputBytes = left.data->estimateFlatSize() +
      right.data->estimateFlatSize();
  VELOX_CHECK_LE(
      inputBytes,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  const auto representedRows =
      addRepresentedRows(left.representedRows, right.representedRows);

  std::vector<CudfVectorPtr> inputs;
  inputs.reserve(2);
  inputs.push_back(std::move(left.data));
  inputs.push_back(std::move(right.data));
  auto concatenated = getConcatenatedTable(
      std::move(inputs), bufferedResultType_, stateStream_, get_temp_mr());
  auto output = doGroupByAggregation(
      concatenated->view(),
      groupingKeyOutputChannels_,
      intermediateAggregators_,
      bufferedResultType_,
      stateStream_,
      get_output_mr());
  VELOX_CHECK_NOT_NULL(output);

  ++finalRunMergeCount_;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        kFinalAggregationRunMerges, RuntimeCounter(1));
    if (finalizing) {
      lockedStats->addRuntimeStat(
          kFinalAggregationFinalizeMerges, RuntimeCounter(1));
    }
    lockedStats->addRuntimeStat(
        kFinalAggregationMergeRows, RuntimeCounter(inputRows));
    lockedStats->addRuntimeStat(
        kFinalAggregationMergeBytes,
        RuntimeCounter(static_cast<int64_t>(inputBytes)));
    lockedStats->addRuntimeStat(
        kFinalAggregationMaxLevel,
        RuntimeCounter(static_cast<int64_t>(outputLevel)));
  }

  if (deviceMemoryDiagnosticsEnabled() &&
      (finalizing || finalRunMergeCount_ == 1 ||
       finalRunMergeCount_ % 64 == 0)) {
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfGroupby node={} state=final_run.merge phase={} "
        "level={} merge={} inputRows={} inputBytes={} outputRows={} "
        "outputBytes={} representedRows={}",
        diagnosticNodeId_,
        finalizing ? "finalize" : "online",
        outputLevel,
        finalRunMergeCount_,
        inputRows,
        inputBytes,
        output->size(),
        output->estimateFlatSize(),
        representedRows));
  }

  return FinalAggregationRun{std::move(output), representedRows};
}

CudfVectorPtr CudfGroupby::drainFinalAggregationRuns() {
  size_t retainedRuns = 0;
  uint64_t retainedRows = 0;
  uint64_t retainedBytes = 0;
  for (const auto& level : finalRunLevels_) {
    if (level.has_value()) {
      ++retainedRuns;
      retainedRows += static_cast<uint64_t>(level->data->size());
      retainedBytes += level->data->estimateFlatSize();
    }
  }
  if (deviceMemoryDiagnosticsEnabled()) {
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfGroupby node={} state=final_runs.drain.begin "
        "inputRuns={} retainedRuns={} retainedRows={} retainedBytes={} "
        "onlineMerges={}",
        diagnosticNodeId_,
        finalInputRunCount_,
        retainedRuns,
        retainedRows,
        retainedBytes,
        finalRunMergeCount_));
  }

  std::optional<FinalAggregationRun> carry;
  for (auto& level : finalRunLevels_) {
    if (!level.has_value()) {
      continue;
    }
    auto run = std::move(*level);
    level.reset();
    if (!carry.has_value()) {
      carry = std::move(run);
      continue;
    }
    const auto representedRows =
        addRepresentedRows(carry->representedRows, run.representedRows);
    const auto outputLevel = aggregationRunLevel(representedRows);
    carry = mergeFinalAggregationRuns(
        std::move(*carry), std::move(run), outputLevel, true);
  }
  finalRunLevels_.clear();

  if (!carry.has_value()) {
    return nullptr;
  }
  return std::move(carry->data);
}

CudfVectorPtr CudfGroupby::finalizeStreamingFinalAggregation() {
  VELOX_CHECK(finalAggregationMode_ == FinalAggregationMode::kStreaming);
  VELOX_CHECK_NOT_NULL(finalStreamingGroupby_);

  const auto distinctKeys = finalStreamingGroupby_->distinct_keys();
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfGroupby node={} state=final_streaming.finalize.begin "
      "batches={} distinctKeys={} maxDistinctKeys={}",
      diagnosticNodeId_,
      finalStreamingBatchCount_,
      distinctKeys,
      finalStreamingMaxDistinctKeys_));

  auto [groupKeys, flatResults] =
      finalStreamingGroupby_->finalize(stateStream_, get_output_mr());
  VELOX_CHECK_NOT_NULL(groupKeys);
  VELOX_CHECK_EQ(groupKeys->num_rows(), distinctKeys);

  // streaming_groupby returns one aggregation_result per flattened request.
  // Restore the regular request layout expected by each GroupbyAggregator so
  // its existing output conversion (including casts and compound results) is
  // reused unchanged.
  std::vector<cudf::groupby::aggregation_result> regularResults;
  regularResults.reserve(finalStreamingRequestAggregationCounts_.size());
  size_t flatResultIndex = 0;
  for (const auto aggregationCount :
       finalStreamingRequestAggregationCounts_) {
    auto& regularResult = regularResults.emplace_back();
    regularResult.results.reserve(aggregationCount);
    for (size_t i = 0; i < aggregationCount; ++i) {
      VELOX_CHECK_LT(flatResultIndex, flatResults.size());
      auto& flatResult = flatResults[flatResultIndex++];
      VELOX_CHECK_EQ(flatResult.results.size(), 1);
      regularResult.results.push_back(std::move(flatResult.results.front()));
    }
  }
  VELOX_CHECK_EQ(flatResultIndex, flatResults.size());

  auto resultColumns = groupKeys->release();
  for (auto& aggregator : aggregators_) {
    resultColumns.push_back(aggregator->makeOutputColumn(
        regularResults, stateStream_, get_output_mr()));
  }
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));
  const auto outputRows = resultTable->num_rows();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        kFinalStreamingDistinctKeys,
        RuntimeCounter(static_cast<int64_t>(distinctKeys)));
    lockedStats->addRuntimeStat(
        kFinalStreamingOutputRows, RuntimeCounter(outputRows));
  }

  // finalize and makeOutputColumn may enqueue work that reads persistent state.
  // Complete it before destroying that state. Both state and output use the
  // configured async resources; no pool resource or release threshold is used.
  stateStream_.synchronize();
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfGroupby node={} state=final_streaming.finalize.end "
      "batches={} distinctKeys={} outputRows={}",
      diagnosticNodeId_,
      finalStreamingBatchCount_,
      distinctKeys,
      outputRows));
  finalStreamingGroupby_.reset();

  if (outputRows == 0) {
    return nullptr;
  }
  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, outputRows, std::move(resultTable), stateStream_);
}

void CudfGroupby::computeSingleGroupbyStreaming(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto preparedInput = prepareAggregationInput(
      tbl->getTableView(),
      static_cast<cudf::size_type>(tbl->size()),
      precomputedInputEvaluators_,
      inputTableStream,
      get_temp_mr());
  auto permutedInputView = preparedInput.tableView.select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      partialAggregators_,
      bufferedResultType_,
      inputTableStream,
      get_output_mr());

  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(std::exchange(bufferedResult_, nullptr));
    tablesToConcat.push_back(std::move(groupbyOnInput));
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_temp_mr());

    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());
    bufferedResult_ = compactedOutput;
  } else {
    bufferedResult_ = groupbyOnInput;
  }
}

void CudfGroupby::prepareInputForStateStream(const CudfVectorPtr& input) {
  const auto inputStream = input->stream();
  if (inputStream.value() != stateStream_.value()) {
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputStream}, stateStream_);
  }
  // Rebind before taking a view: rebindStream may rebuild an owned table, and
  // a packed buffer can retain a different deallocation stream even when the
  // vector's logical stream already equals stateStream_.
  VELOX_CHECK(
      input->rebindStream(stateStream_),
      "CudfGroupby cannot rebind its input to the state stream");
}

void CudfGroupby::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  input.reset();

  if (!isPartialOutput_ && !isSingleStep_) {
    prepareInputForStateStream(cudfInput);
  }

  if (streamingEnabled_) {
    if (isPartialOutput_) {
      computePartialGroupbyStreaming(std::move(cudfInput));
      return;
    } else if (isSingleStep_) {
      computeSingleGroupbyStreaming(std::move(cudfInput));
      return;
    } else {
      computeFinalGroupbyStreaming(std::move(cudfInput));
      return;
    }
  }

  // Handle non-streaming cases.
  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfGroupby::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());

  // TODO: All other args to groupby are related to sort groupby. We don't
  // support optimizations related to it yet.
  cudf::groupby::groupby groupByOwner(
      groupbyKeyView,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& aggregator : aggregators) {
    aggregator->addGroupbyRequest(tableView, requests, stream);
  }

  auto [groupKeys, results] = groupByOwner.aggregate(requests, stream, mr);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  // first fill the grouping keys
  auto groupKeysColumns = groupKeys->release();
  resultColumns.insert(
      resultColumns.begin(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  // then fill the aggregation results
  for (auto& aggregator : aggregators) {
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream, mr));
  }

  // make a cudf table out of columns
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));

  auto numRows = resultTable->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::releaseAndResetBufferedResult() {
  auto numOutputRows = bufferedResult_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushRowCount),
        RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushTimes), RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kPartialAggregationPct),
        RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving bufferedResult_ to the caller because we want it to be null
  // after this call.
  return std::move(bufferedResult_);
}

RowVectorPtr CudfGroupby::doGetOutput() {
  // Handle partial streaming groupby.
  if (isPartialOutput_ && streamingEnabled_) {
    if (bufferedResult_ &&
        bufferedResult_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      return releaseAndResetBufferedResult();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!bufferedResult_ && finished_) {
      return nullptr;
    }
    return releaseAndResetBufferedResult();
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  // Streaming finalization: single step uses finalAggregators_ to convert
  // intermediate results to final output; final step uses aggregators_.
  // At this point isPartialOutput_ is false (handled above) and noMoreInput_
  // is true (guarded by the check above).
  if (streamingEnabled_) {
    if (!isPartialOutput_ && !isSingleStep_ &&
        finalAggregationMode_ == FinalAggregationMode::kStreaming) {
      finished_ = true;
      return finalizeStreamingFinalAggregation();
    }
    if (!isPartialOutput_ && !isSingleStep_) {
      VELOX_CHECK_NULL(bufferedResult_);
      bufferedResult_ = drainFinalAggregationRuns();
    }
    finished_ = true;
    if (!bufferedResult_) {
      return nullptr;
    }
    auto& aggs = isSingleStep_ ? finalAggregators_ : aggregators_;
    auto stream = bufferedResult_->stream();
    logDeviceMemorySnapshot(
        fmt::format(
            "operator=CudfGroupby node={} state=finalize.begin bufferedRows={} "
            "bufferedBytes={}",
            diagnosticNodeId_,
            bufferedResult_->size(),
            bufferedResult_->estimateFlatSize()));
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        aggs,
        outputType_,
        stream,
        get_output_mr());
    stream.synchronize();
    logDeviceMemorySnapshot(
        fmt::format(
            "operator=CudfGroupby node={} state=finalize.end outputRows={} "
            "bufferedBytes={}",
            diagnosticNodeId_,
            result == nullptr ? 0 : result->size(),
            bufferedResult_->estimateFlatSize()));
    bufferedResult_.reset();
    return result;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), inputType_, stream, get_temp_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto preparedInput = prepareAggregationInput(
      tbl->view(),
      tbl->num_rows(),
      precomputedInputEvaluators_,
      stream,
      get_temp_mr());
  auto permutedInputView = preparedInput.tableView.select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  return doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      outputType_,
      stream,
      get_output_mr());
}

void CudfGroupby::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

void CudfGroupby::doClose() {
  // close() may run while a failed task unwinds. First attempt to drain all
  // kernels that can still reference persistent streaming state, while
  // preserving the original task failure if the CUDA context is poisoned.
  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfGroupby state stream cleanup failed: "
                 << error.what();
  }
  finalStreamingGroupby_.reset();
  finalStreamingRequestAggregationCounts_.clear();
  finalRunLevels_.clear();
  bufferedResult_.reset();
  inputs_.clear();
  Operator::close();
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
