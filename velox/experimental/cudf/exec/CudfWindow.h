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
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override;

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

  static cudf::rank_method toRankMethod(WindowFunctionKind kind);

  std::unique_ptr<cudf::rolling_aggregation>
  makeRollingAggregation(const WindowFunctionSpec& spec) const;

  std::pair<cudf::window_bounds, cudf::window_bounds>
  toWindowBounds(const core::WindowNode::Frame& frame) const;

  std::unique_ptr<cudf::column> computeRankColumn(
      cudf::table_view const& sortedInput,
      WindowFunctionKind kind,
      rmm::cuda_stream_view stream) const;

  std::unique_ptr<cudf::column> computeAggregateColumn(
      cudf::table_view const& sortedInput,
      const WindowFunctionSpec& spec,
      rmm::cuda_stream_view stream) const;

  std::unique_ptr<cudf::table> computeOutputTable(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream) const;

  std::shared_ptr<const core::WindowNode> windowNode_;
  RowTypePtr inputType_;
  std::vector<CudfVectorPtr> inputs_;
  CudfVectorPtr output_;
  bool finished_{false};

  std::vector<cudf::size_type> partitionKeyChannels_;
  std::vector<cudf::size_type> sortKeyChannels_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> sortNullOrders_;
  std::vector<WindowFunctionSpec> functionSpecs_;
};

} // namespace facebook::velox::cudf_velox
