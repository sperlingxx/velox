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
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h"
#include "velox/type/TypeUtil.h"

#include <fmt/format.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

// ---- Bridge ----

void CudfNestedLoopJoinBridge::setData(
    std::optional<data_type> data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!data_.has_value());
    data_ = std::move(data);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<CudfNestedLoopJoinBridge::data_type>
CudfNestedLoopJoinBridge::dataOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (data_.has_value()) {
    return data_;
  }
  promises_.emplace_back(
      "CudfNestedLoopJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

void CudfNestedLoopJoinBridge::setBuildStream(
    rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view>
CudfNestedLoopJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

// ---- Build ----

CudfNestedLoopJoinBuild::CudfNestedLoopJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          "CudfNestedLoopJoinBuild"),
      NvtxHelper(
          nvtx3::rgb{100, 149, 237}, // Cornflower Blue
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(std::move(joinNode)) {}

void CudfNestedLoopJoinBuild::addInput(RowVectorPtr input) {
  if (input && input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

bool CudfNestedLoopJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfNestedLoopJoinBuild::getOutput() {
  return nullptr;
}

void CudfNestedLoopJoinBuild::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build =
        dynamic_cast<CudfNestedLoopJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  std::vector<std::shared_ptr<cudf::table>> buildTables;
  if (!inputs_.empty()) {
    std::vector<cudf::table_view> views;
    views.reserve(inputs_.size());
    for (auto& cudfVec : inputs_) {
      views.push_back(cudfVec->getTableView());
    }
    auto concatenated =
        cudf::concatenate(views, stream);
    buildTables.push_back(std::move(concatenated));
  } else {
    auto buildType = joinNode_->sources()[1]->outputType();
    std::vector<std::unique_ptr<cudf::column>> emptyCols;
    for (size_t i = 0; i < buildType->size(); ++i) {
      auto cudfType =
          veloxToCudfDataType(buildType->childAt(i));
      auto scalar = cudf::make_default_constructed_scalar(
          cudfType, stream);
      emptyCols.push_back(cudf::make_column_from_scalar(
          *scalar,
          0,
          stream,
          cudf::get_current_device_resource_ref()));
    }
    buildTables.push_back(
        std::make_shared<cudf::table>(std::move(emptyCols)));
  }

  auto bridge = operatorCtx_->task()
                    ->getCustomJoinBridge(
                        operatorCtx_->driverCtx()->splitGroupId,
                        planNodeId());
  auto nlBridge =
      std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(bridge);
  VELOX_CHECK_NOT_NULL(nlBridge);
  nlBridge->setBuildStream(stream);
  nlBridge->setData(std::move(buildTables));

  inputs_.clear();

  for (auto& promise : promises) {
    promise.setValue();
  }
}

exec::BlockingReason CudfNestedLoopJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfNestedLoopJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

// ---- Probe ----

CudfNestedLoopJoinProbe::CudfNestedLoopJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "CudfNestedLoopJoinProbe"),
      NvtxHelper(
          nvtx3::rgb{46, 139, 87}, // Sea Green
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(std::move(joinNode)),
      joinType_(joinNode_->joinType()) {
  probeType_ = joinNode_->sources()[0]->outputType();
  buildType_ = joinNode_->sources()[1]->outputType();

  auto outputType = joinNode_->outputType();
  auto numOutputColumns = outputType->size();
  if (core::isLeftSemiProjectJoin(joinType_)) {
    VELOX_CHECK_GE(numOutputColumns, 1);
    matchColumnOutputIndex_ =
        static_cast<int32_t>(numOutputColumns - 1);
    VELOX_CHECK_EQ(
        outputType->childAt(matchColumnOutputIndex_), BOOLEAN());
    --numOutputColumns;
  }

  for (size_t i = 0; i < numOutputColumns; ++i) {
    auto const& name = outputType->nameOf(i);
    auto channel = probeType_->getChildIdxIfExists(name);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType_->getChildIdxIfExists(name);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL(
        "NLJ output column not found: {}",
        outputType->childAt(i));
  }
}

void CudfNestedLoopJoinProbe::initialize() {
  Operator::initialize();
  if (!joinNode_->joinCondition()) {
    return;
  }
  exec::ExprSet exprs(
      {joinNode_->joinCondition()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);

  std::vector<velox::RowTypePtr> types{probeType_, buildType_};
  filterEvaluator_ = createCudfExpression(
      exprs.exprs()[0],
      facebook::velox::type::concatRowTypes(types));

  std::vector<PrecomputeInstruction> leftPrecompute;
  std::vector<PrecomputeInstruction> rightPrecompute;
  createAstTree(
      exprs.exprs()[0],
      tree_,
      scalars_,
      probeType_,
      buildType_,
      leftPrecompute,
      rightPrecompute);
}

bool CudfNestedLoopJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr &&
      buildData_.has_value();
}

void CudfNestedLoopJoinProbe::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

void CudfNestedLoopJoinProbe::noMoreInput() {
  Operator::noMoreInput();
  if (!core::isRightJoin(joinType_) &&
      !core::isFullJoin(joinType_)) {
    return;
  }
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(),
          operatorCtx_->driver(),
          &future_,
          promises,
          peers)) {
    return;
  }

  if (core::isRightJoin(joinType_) ||
      core::isFullJoin(joinType_)) {
    isLastDriver_ = true;
    if (buildData_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();
      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto* probe = dynamic_cast<CudfNestedLoopJoinProbe*>(
            peer->findOperator(planNodeId()));
        if (probe == nullptr) {
          continue;
        }
        for (size_t i = 0; i < rightMatchedFlags_.size(); ++i) {
          if (i < probe->rightMatchedFlags_.size() &&
              probe->rightMatchedFlags_[i]) {
            auto updated = cudf::binary_operation(
                rightMatchedFlags_[i]->view(),
                probe->rightMatchedFlags_[i]->view(),
                cudf::binary_operator::BITWISE_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream,
                cudf::get_current_device_resource_ref());
            rightMatchedFlags_[i] = std::move(updated);
          }
        }
      }
      stream.synchronize();
    }
  }

  for (auto& promise : promises) {
    promise.setValue();
  }
}

void CudfNestedLoopJoinProbe::close() {
  Operator::close();
  buildData_.reset();
  rightMatchedFlags_.clear();
}

exec::BlockingReason CudfNestedLoopJoinProbe::isBlocked(
    ContinueFuture* future) {
  if ((core::isRightJoin(joinType_) ||
       core::isFullJoin(joinType_)) &&
      buildData_.has_value()) {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (buildData_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto bridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto nlBridge =
      std::dynamic_pointer_cast<CudfNestedLoopJoinBridge>(bridge);
  VELOX_CHECK_NOT_NULL(nlBridge);

  auto data = nlBridge->dataOrFuture(future);
  if (!data.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  buildData_ = std::move(data);
  buildStream_ = nlBridge->getBuildStream();

  if (core::isRightJoin(joinType_) ||
      core::isFullJoin(joinType_)) {
    auto stream = cudfGlobalStreamPool().get_stream();
    rightMatchedFlags_.clear();
    for (auto& tbl : buildData_.value()) {
      auto n = tbl->num_rows();
      auto falseScalar =
          cudf::numeric_scalar<bool>(false, true, stream);
      rightMatchedFlags_.push_back(
          cudf::make_column_from_scalar(
              falseScalar,
              n,
              stream,
              cudf::get_current_device_resource_ref()));
    }
    stream.synchronize();
  }

  return exec::BlockingReason::kNotBlocked;
}

bool CudfNestedLoopJoinProbe::isFinished() {
  if (finished_) {
    return true;
  }
  if (!noMoreInput_ || input_ != nullptr) {
    return false;
  }
  // For right/full join the last driver must still emit
  // unmatched build rows.
  if (isLastDriver_ &&
      (core::isRightJoin(joinType_) ||
       core::isFullJoin(joinType_))) {
    return false;
  }
  return true;
}

// Cross join + optional filter -> gathered output table
std::unique_ptr<cudf::table>
CudfNestedLoopJoinProbe::crossJoinAndFilter(
    cudf::table_view leftView,
    cudf::table_view rightView,
    rmm::cuda_stream_view stream) {
  auto mr = cudf::get_current_device_resource_ref();

  if (leftView.num_rows() == 0 || rightView.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> emptyCols(
        outputType_->size());
    for (size_t li = 0; li < leftColumnOutputIndices_.size();
         ++li) {
      auto outIdx = leftColumnOutputIndices_[li];
      auto srcCol =
          leftView.column(leftColumnIndicesToGather_[li]);
      auto scalar = cudf::make_default_constructed_scalar(
          srcCol.type(), stream, mr);
      emptyCols[outIdx] = cudf::make_column_from_scalar(
          *scalar, 0, stream, mr);
    }
    for (size_t ri = 0; ri < rightColumnOutputIndices_.size();
         ++ri) {
      auto outIdx = rightColumnOutputIndices_[ri];
      auto srcCol =
          rightView.column(rightColumnIndicesToGather_[ri]);
      auto scalar = cudf::make_default_constructed_scalar(
          srcCol.type(), stream, mr);
      emptyCols[outIdx] = cudf::make_column_from_scalar(
          *scalar, 0, stream, mr);
    }
    return std::make_unique<cudf::table>(std::move(emptyCols));
  }

  if (!joinNode_->joinCondition()) {
    // Pure cross join
    auto crossResult = cudf::cross_join(leftView, rightView,
                                         stream, mr);
    auto allCols = crossResult->release();
    auto leftCols = leftView.num_columns();
    std::vector<std::unique_ptr<cudf::column>> outCols(
        outputType_->size());
    for (size_t li = 0; li < leftColumnOutputIndices_.size();
         ++li) {
      outCols[leftColumnOutputIndices_[li]] =
          std::move(allCols[leftColumnIndicesToGather_[li]]);
    }
    for (size_t ri = 0; ri < rightColumnOutputIndices_.size();
         ++ri) {
      outCols[rightColumnOutputIndices_[ri]] = std::move(
          allCols[leftCols + rightColumnIndicesToGather_[ri]]);
    }
    stream.synchronize();
    return std::make_unique<cudf::table>(std::move(outCols));
  }

  // Conditional inner join: returns index pairs
  auto [leftIndices, rightIndices] =
      cudf::conditional_inner_join(
          leftView, rightView, tree_.back(), {}, stream, mr);

  auto leftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*leftIndices};
  auto rightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*rightIndices};
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

  auto leftInput =
      leftView.select(leftColumnIndicesToGather_);
  auto rightInput =
      rightView.select(rightColumnIndicesToGather_);

  static constexpr auto oob = cudf::out_of_bounds_policy::NULLIFY;
  auto leftGathered =
      cudf::gather(leftInput, leftIndicesCol, oob, stream);
  auto rightGathered =
      cudf::gather(rightInput, rightIndicesCol, oob, stream);

  auto leftCols = leftGathered->release();
  auto rightCols = rightGathered->release();
  std::vector<std::unique_ptr<cudf::column>> outCols(
      outputType_->size());
  for (size_t li = 0; li < leftColumnOutputIndices_.size();
       ++li) {
    outCols[leftColumnOutputIndices_[li]] =
        std::move(leftCols[li]);
  }
  for (size_t ri = 0; ri < rightColumnOutputIndices_.size();
       ++ri) {
    outCols[rightColumnOutputIndices_[ri]] =
        std::move(rightCols[ri]);
  }

  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(outCols));
}

std::vector<std::unique_ptr<cudf::table>>
CudfNestedLoopJoinProbe::innerJoin(
    cudf::table_view leftView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> results;
  for (auto& buildTbl : buildData_.value()) {
    results.push_back(
        crossJoinAndFilter(leftView, buildTbl->view(), stream));
  }
  return results;
}

std::vector<std::unique_ptr<cudf::table>>
CudfNestedLoopJoinProbe::leftJoin(
    cudf::table_view leftView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> results;
  auto mr = cudf::get_current_device_resource_ref();

  for (auto& buildTbl : buildData_.value()) {
    auto rightView = buildTbl->view();

    if (!joinNode_->joinCondition()) {
      if (rightView.num_rows() == 0) {
        // Left join with empty build: all left rows, null build
        std::vector<std::unique_ptr<cudf::column>> outCols(
            outputType_->size());
        auto n = leftView.num_rows();
        for (size_t li = 0;
             li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          outCols[outIdx] = std::make_unique<cudf::column>(
              leftView.column(leftColumnIndicesToGather_[li]),
              stream, mr);
        }
        for (size_t ri = 0;
             ri < rightColumnOutputIndices_.size(); ++ri) {
          auto outIdx = rightColumnOutputIndices_[ri];
          auto srcCol = rightView.column(
              rightColumnIndicesToGather_[ri]);
          auto scalar = cudf::make_default_constructed_scalar(
              srcCol.type(), stream, mr);
          outCols[outIdx] = cudf::make_column_from_scalar(
              *scalar, n, stream, mr);
        }
        results.push_back(
            std::make_unique<cudf::table>(std::move(outCols)));
      } else {
        results.push_back(crossJoinAndFilter(
            leftView, rightView, stream));
      }
      continue;
    }

    auto [leftIndices, rightIndices] =
        cudf::conditional_left_join(
            leftView, rightView, tree_.back(), {}, stream, mr);

    auto leftSpan =
        cudf::device_span<cudf::size_type const>{*leftIndices};
    auto rightSpan =
        cudf::device_span<cudf::size_type const>{*rightIndices};
    auto leftCol = cudf::column_view{leftSpan};
    auto rightCol = cudf::column_view{rightSpan};

    auto leftInput =
        leftView.select(leftColumnIndicesToGather_);
    auto rightInput =
        rightView.select(rightColumnIndicesToGather_);

    static constexpr auto oob =
        cudf::out_of_bounds_policy::NULLIFY;
    auto leftGathered =
        cudf::gather(leftInput, leftCol, oob, stream);
    auto rightGathered =
        cudf::gather(rightInput, rightCol, oob, stream);

    auto lCols = leftGathered->release();
    auto rCols = rightGathered->release();
    std::vector<std::unique_ptr<cudf::column>> outCols(
        outputType_->size());
    for (size_t li = 0;
         li < leftColumnOutputIndices_.size(); ++li) {
      outCols[leftColumnOutputIndices_[li]] =
          std::move(lCols[li]);
    }
    for (size_t ri = 0;
         ri < rightColumnOutputIndices_.size(); ++ri) {
      outCols[rightColumnOutputIndices_[ri]] =
          std::move(rCols[ri]);
    }
    stream.synchronize();
    results.push_back(
        std::make_unique<cudf::table>(std::move(outCols)));
  }
  return results;
}

std::vector<std::unique_ptr<cudf::table>>
CudfNestedLoopJoinProbe::rightJoin(
    cudf::table_view leftView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> results;
  auto mr = cudf::get_current_device_resource_ref();
  auto& buildTables = buildData_.value();

  for (size_t i = 0; i < buildTables.size(); ++i) {
    auto rightView = buildTables[i]->view();
    if (leftView.num_rows() == 0 || rightView.num_rows() == 0) {
      continue;
    }

    if (!joinNode_->joinCondition()) {
      results.push_back(crossJoinAndFilter(
          leftView, rightView, stream));
      // Pure cross join: every build row matched
      auto trueScalar =
          cudf::numeric_scalar<bool>(true, true, stream);
      rightMatchedFlags_[i] = cudf::make_column_from_scalar(
          trueScalar,
          rightView.num_rows(),
          stream,
          mr);
      continue;
    }

    // Conditional inner join, then track matched right rows
    auto [leftIndices, rightIndices] =
        cudf::conditional_inner_join(
            leftView, rightView, tree_.back(), {}, stream, mr);

    auto leftSpan =
        cudf::device_span<cudf::size_type const>{*leftIndices};
    auto rightSpan =
        cudf::device_span<cudf::size_type const>{*rightIndices};
    auto leftCol = cudf::column_view{leftSpan};
    auto rightCol = cudf::column_view{rightSpan};

    // Update matched flags for right rows
    if (rightIndices->size() > 0) {
      auto n = rightView.num_rows();
      auto rowSeq = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(
              0, true, stream),
          cudf::numeric_scalar<cudf::size_type>(
              1, true, stream),
          stream,
          mr);
      auto matched =
          cudf::contains(rightCol, rowSeq->view(), stream);
      auto updated = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matched->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
      rightMatchedFlags_[i] = std::move(updated);
    }

    auto leftInput =
        leftView.select(leftColumnIndicesToGather_);
    auto rightInput =
        rightView.select(rightColumnIndicesToGather_);
    static constexpr auto oob =
        cudf::out_of_bounds_policy::NULLIFY;
    auto leftGathered =
        cudf::gather(leftInput, leftCol, oob, stream);
    auto rightGathered =
        cudf::gather(rightInput, rightCol, oob, stream);

    auto lCols = leftGathered->release();
    auto rCols = rightGathered->release();
    std::vector<std::unique_ptr<cudf::column>> outCols(
        outputType_->size());
    for (size_t li = 0;
         li < leftColumnOutputIndices_.size(); ++li) {
      outCols[leftColumnOutputIndices_[li]] =
          std::move(lCols[li]);
    }
    for (size_t ri = 0;
         ri < rightColumnOutputIndices_.size(); ++ri) {
      outCols[rightColumnOutputIndices_[ri]] =
          std::move(rCols[ri]);
    }
    stream.synchronize();
    results.push_back(
        std::make_unique<cudf::table>(std::move(outCols)));
  }
  return results;
}

std::vector<std::unique_ptr<cudf::table>>
CudfNestedLoopJoinProbe::fullJoin(
    cudf::table_view leftView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> results;
  auto mr = cudf::get_current_device_resource_ref();
  auto& buildTables = buildData_.value();

  for (size_t i = 0; i < buildTables.size(); ++i) {
    auto rightView = buildTables[i]->view();

    if (!joinNode_->joinCondition()) {
      if (leftView.num_rows() == 0 &&
          rightView.num_rows() == 0) {
        continue;
      }
      if (leftView.num_rows() > 0 &&
          rightView.num_rows() > 0) {
        results.push_back(crossJoinAndFilter(
            leftView, rightView, stream));
        auto trueScalar =
            cudf::numeric_scalar<bool>(true, true, stream);
        rightMatchedFlags_[i] = cudf::make_column_from_scalar(
            trueScalar, rightView.num_rows(), stream, mr);
        continue;
      }
    }

    if (leftView.num_rows() == 0 || rightView.num_rows() == 0) {
      if (leftView.num_rows() > 0) {
        // Left rows with null build columns
        std::vector<std::unique_ptr<cudf::column>> outCols(
            outputType_->size());
        auto n = leftView.num_rows();
        for (size_t li = 0;
             li < leftColumnOutputIndices_.size(); ++li) {
          auto idx = leftColumnOutputIndices_[li];
          outCols[idx] = std::make_unique<cudf::column>(
              leftView.column(leftColumnIndicesToGather_[li]),
              stream, mr);
        }
        for (size_t ri = 0;
             ri < rightColumnOutputIndices_.size(); ++ri) {
          auto idx = rightColumnOutputIndices_[ri];
          auto srcCol = rightView.column(
              rightColumnIndicesToGather_[ri]);
          auto scalar = cudf::make_default_constructed_scalar(
              srcCol.type(), stream, mr);
          outCols[idx] = cudf::make_column_from_scalar(
              *scalar, n, stream, mr);
        }
        results.push_back(
            std::make_unique<cudf::table>(std::move(outCols)));
      }
      continue;
    }

    auto [leftIndices, rightIndices] =
        cudf::conditional_left_join(
            leftView, rightView, tree_.back(), {}, stream, mr);

    // Track matched right rows
    if (rightIndices->size() > 0) {
      auto n = rightView.num_rows();
      auto rowSeq = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(
              0, true, stream),
          cudf::numeric_scalar<cudf::size_type>(
              1, true, stream),
          stream,
          mr);
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{
              *rightIndices}};
      auto matched =
          cudf::contains(rightIdxCol, rowSeq->view(), stream);
      auto updated = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matched->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
      rightMatchedFlags_[i] = std::move(updated);
    }

    auto leftSpan =
        cudf::device_span<cudf::size_type const>{*leftIndices};
    auto rightSpan =
        cudf::device_span<cudf::size_type const>{*rightIndices};
    auto leftCol = cudf::column_view{leftSpan};
    auto rightCol = cudf::column_view{rightSpan};

    auto leftInput =
        leftView.select(leftColumnIndicesToGather_);
    auto rightInput =
        rightView.select(rightColumnIndicesToGather_);

    static constexpr auto oob =
        cudf::out_of_bounds_policy::NULLIFY;
    auto leftGathered =
        cudf::gather(leftInput, leftCol, oob, stream);
    auto rightGathered =
        cudf::gather(rightInput, rightCol, oob, stream);

    auto lCols = leftGathered->release();
    auto rCols = rightGathered->release();
    std::vector<std::unique_ptr<cudf::column>> outCols(
        outputType_->size());
    for (size_t li = 0;
         li < leftColumnOutputIndices_.size(); ++li) {
      outCols[leftColumnOutputIndices_[li]] =
          std::move(lCols[li]);
    }
    for (size_t ri = 0;
         ri < rightColumnOutputIndices_.size(); ++ri) {
      outCols[rightColumnOutputIndices_[ri]] =
          std::move(rCols[ri]);
    }
    stream.synchronize();
    results.push_back(
        std::make_unique<cudf::table>(std::move(outCols)));
  }
  return results;
}

std::vector<std::unique_ptr<cudf::table>>
CudfNestedLoopJoinProbe::leftSemiProjectJoin(
    cudf::table_view leftView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> results;
  auto mr = cudf::get_current_device_resource_ref();
  auto leftN = leftView.num_rows();

  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream);
  auto matchFlags = cudf::make_column_from_scalar(
      falseScalar, leftN, stream, mr);

  auto rowIndices = cudf::sequence(
      leftN,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream),
      stream,
      mr);

  for (auto& buildTbl : buildData_.value()) {
    auto rightView = buildTbl->view();
    if (rightView.num_rows() == 0) {
      continue;
    }

    std::unique_ptr<rmm::device_uvector<cudf::size_type>>
        leftIndices;
    if (joinNode_->joinCondition()) {
      leftIndices = cudf::conditional_left_semi_join(
          leftView, rightView, tree_.back(), {}, stream, mr);
    } else {
      // No condition: every left row matches if build non-empty
      auto trueScalar =
          cudf::numeric_scalar<bool>(true, true, stream);
      matchFlags = cudf::make_column_from_scalar(
          trueScalar, leftN, stream, mr);
      continue;
    }

    if (leftIndices->size() > 0) {
      auto leftIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{
              *leftIndices}};
      auto matched = cudf::contains(
          leftIdxCol, rowIndices->view(), stream);
      auto updated = cudf::binary_operation(
          matchFlags->view(),
          matched->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
      matchFlags = std::move(updated);
    }
  }

  auto leftInput =
      leftView.select(leftColumnIndicesToGather_);
  std::vector<std::unique_ptr<cudf::column>> outCols(
      outputType_->size());
  for (size_t j = 0; j < leftColumnOutputIndices_.size(); ++j) {
    outCols[leftColumnOutputIndices_[j]] =
        std::make_unique<cudf::column>(
            leftInput.column(j), stream, mr);
  }
  outCols[matchColumnOutputIndex_] = std::move(matchFlags);

  stream.synchronize();
  results.push_back(
      std::make_unique<cudf::table>(std::move(outCols)));
  return results;
}

RowVectorPtr CudfNestedLoopJoinProbe::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  if (finished_ || !buildData_.has_value()) {
    return nullptr;
  }

  if (!input_) {
    // Emit unmatched build rows for right/full joins
    if ((core::isRightJoin(joinType_) ||
         core::isFullJoin(joinType_)) &&
        noMoreInput_ && !finished_ && isLastDriver_) {
      auto& buildTables = buildData_.value();
      auto stream = cudfGlobalStreamPool().get_stream();
      auto mr = cudf::get_current_device_resource_ref();
      std::vector<std::unique_ptr<cudf::table>> toConcat;

      for (size_t i = 0; i < buildTables.size(); ++i) {
        auto& buildTbl = buildTables[i];
        auto n = buildTbl->num_rows();
        if (n == 0) {
          continue;
        }
        auto& flags = rightMatchedFlags_[i];
        auto boolMask = cudf::unary_operation(
            flags->view(), cudf::unary_operator::NOT, stream);

        auto unmatchedCount = cudf::reduce(
            boolMask->view(),
            *cudf::make_sum_aggregation<
                cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::INT32},
            stream);
        auto m = static_cast<cudf::numeric_scalar<int32_t>*>(
                     unmatchedCount.get())
                     ->value(stream);
        if (m == 0) {
          continue;
        }

        std::vector<std::unique_ptr<cudf::column>> outCols(
            outputType_->size());
        for (size_t li = 0;
             li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          auto probeChannel = leftColumnIndicesToGather_[li];
          auto cudfType =
              veloxToCudfDataType(probeType_->childAt(probeChannel));
          auto nullScalar =
              cudf::make_default_constructed_scalar(cudfType);
          outCols[outIdx] = cudf::make_column_from_scalar(
              *nullScalar, m, stream, mr);
        }
        if (!rightColumnIndicesToGather_.empty()) {
          auto rightInput = buildTbl->view().select(
              rightColumnIndicesToGather_);
          auto unmatchedRight = cudf::apply_boolean_mask(
              rightInput, boolMask->view(), stream);
          auto rightCols = unmatchedRight->release();
          for (size_t ri = 0;
               ri < rightColumnOutputIndices_.size(); ++ri) {
            outCols[rightColumnOutputIndices_[ri]] =
                std::move(rightCols[ri]);
          }
        }
        toConcat.push_back(
            std::make_unique<cudf::table>(std::move(outCols)));
      }

      if (!toConcat.empty()) {
        auto out =
            concatenateTables(std::move(toConcat), stream);
        finished_ = true;
        auto size = out->num_rows();
        if (out->num_columns() == 0 || size == 0) {
          return nullptr;
        }
        return std::make_shared<CudfVector>(
            pool(), outputType_, size, std::move(out), stream);
      }
      finished_ = true;
    }
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  auto leftView = cudfInput->getTableView();

  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  switch (joinType_) {
    case core::JoinType::kInner:
      cudfOutputs = innerJoin(leftView, stream);
      break;
    case core::JoinType::kLeft:
      cudfOutputs = leftJoin(leftView, stream);
      break;
    case core::JoinType::kRight:
      cudfOutputs = rightJoin(leftView, stream);
      break;
    case core::JoinType::kFull:
      cudfOutputs = fullJoin(leftView, stream);
      break;
    case core::JoinType::kLeftSemiProject:
      cudfOutputs = leftSemiProjectJoin(leftView, stream);
      break;
    default:
      VELOX_FAIL(
          "Unsupported NLJ type: {}", static_cast<int>(joinType_));
  }

  cudfInput.reset();
  input_.reset();
  finished_ = noMoreInput_ &&
      !core::isRightJoin(joinType_) &&
      !core::isFullJoin(joinType_);

  auto cudfOutput =
      concatenateTables(std::move(cudfOutputs), stream);
  auto size = cudfOutput->num_rows();
  if (cudfOutput->num_columns() == 0 || size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(cudfOutput), stream);
}

// ---- Translator ----

std::unique_ptr<exec::Operator>
CudfNestedLoopJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto nlNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
              node)) {
    return std::make_unique<CudfNestedLoopJoinProbe>(
        id, ctx, nlNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge>
CudfNestedLoopJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (auto nlNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
              node)) {
    return std::make_unique<CudfNestedLoopJoinBridge>();
  }
  return nullptr;
}

exec::OperatorSupplier
CudfNestedLoopJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (auto nlNode =
          std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
              node)) {
    return [nlNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfNestedLoopJoinBuild>(
          operatorId, ctx, nlNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
