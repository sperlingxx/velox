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
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/DecimalUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h" // NOLINT(misc-unused-headers)
#include "velox/type/TypeUtil.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime_api.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

namespace {

static constexpr int kOomMaxRetries = 3;

void recoverGpuMemory() {
  cudaDeviceSynchronize();
  cudaGetLastError();
}

bool isCudaRelatedError(const std::exception& e) {
  if (dynamic_cast<const std::bad_alloc*>(&e) != nullptr) {
    return true;
  }
  if (dynamic_cast<const rmm::cuda_error*>(&e) != nullptr) {
    return true;
  }
  std::string what = e.what();
  return what.find("cudaError") != std::string::npos ||
      what.find("CUDA error") != std::string::npos ||
      what.find("out_of_memory") != std::string::npos;
}

/// Creates extended table view by appending precomputed columns
cudf::table_view createExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }

  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());

  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& col : precomputedColumns) {
    allViews.push_back(asView(col));
  }

  return cudf::table_view(allViews);
}

} // namespace

void CudfHashJoinProbe::close() {
  Operator::close();
  filterEvaluator_.reset();
  scalars_.clear();
  tree_ = {};
  accumulatedProbeInputs_.clear();
  accumulatedProbeRows_ = 0;
  accumulatedProbeBytes_ = 0;
}

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject) {
  if (CudfConfig::getInstance().debugEnabled) {
    if (hashObject.has_value()) {
      const auto& [tables, joins] = hashObject.value();
      VLOG(1) << "Calling CudfHashJoinBridge::setHashTable with tables="
              << tables.size() << ", hashObjects=" << joins.size();
      for (size_t i = 0; i < tables.size(); ++i) {
        VLOG(2) << "  setHashTable table[" << i
                << "]=" << static_cast<const void*>(tables[i].get());
      }
      for (size_t i = 0; i < joins.size(); ++i) {
        VLOG(2) << "  setHashTable hashObject[" << i
                << "]=" << static_cast<const void*>(joins[i].get());
      }
    } else {
      VLOG(1) << "Calling CudfHashJoinBridge::setHashTable with nullopt";
    }
  }
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(
        !hashObject_.has_value(),
        "CudfHashJoinBridge already has a hash table");
    hashObject_ = std::move(hashObject);
    promises = std::move(promises_);
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "CudfHashJoinBridge::setHashTable stored hash table; waiters="
              << promises.size();
    }
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBridge::setHashTable notifying waiters";
  }
  notify(std::move(promises));
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBridge::setHashTable completed";
  }
}

std::optional<CudfHashJoinBridge::hash_type> CudfHashJoinBridge::hashOrFuture(
    ContinueFuture* future) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture";
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return hashObject_;
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture constructing promise";
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture";
  }
  *future = promises_.back().getSemiFuture();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt";
  }
  return std::nullopt;
}

void CudfHashJoinBridge::setBuildStream(rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view> CudfHashJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

CudfHashJoinBuild::CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : exec::Operator(
          driverCtx,
          nullptr, // joinNode->sources(),
          operatorId,
          joinNode->id(),
          "CudfHashJoinBuild"),
      NvtxHelper(
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "CudfHashJoinBuild constructor";
  }
}

void CudfHashJoinBuild::addInput(RowVectorPtr input) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBuild::addInput";
  }
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

bool CudfHashJoinBuild::needsInput() const {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBuild::needsInput";
  }
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::getOutput() {
  return nullptr;
}

void CudfHashJoinBuild::noMoreInput() {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBuild::noMoreInput";
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild: build batches";
    VLOG(1) << "Build batches count: " << inputs_.size();
    if (inputs_.empty()) {
      VLOG(1) << "Build input batches are empty; continuing with empty build table";
    } else {
      VELOX_CHECK_NOT_NULL(inputs_[0]);
      VLOG(1) << "Build batches number of columns: "
              << inputs_[0]->getTableView().num_columns();
    }
    for (size_t i = 0; i < inputs_.size(); i++) {
      VELOX_CHECK_NOT_NULL(inputs_[i]);
      VLOG(1) << "Build batch " << i
              << ": number of rows: " << inputs_[i]->getTableView().num_rows();
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  // OOM retry for build-side concatenation: when many tasks share a GPU,
  // deferred frees may not have completed. cudaDeviceSynchronize forces
  // all pending frees, making memory available for this allocation.
  std::vector<std::unique_ptr<cudf::table>> tbls;
  for (int attempt = 0;; ++attempt) {
    try {
      tbls = getConcatenatedTableBatched(
          inputs_, joinNode_->sources()[1]->outputType(), stream);
      break;
    } catch (const std::bad_alloc& e) {
      if (attempt >= kOomMaxRetries) {
        throw;
      }
      LOG(WARNING)
          << "CudfHashJoinBuild OOM during concatenation for planNode "
          << planNodeId() << " (attempt " << (attempt + 1)
          << "): " << e.what() << ". Recovering GPU memory and retrying.";
      recoverGpuMemory();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100 * (1 << attempt)));
    }
  }
  inputs_.clear();

  for (auto const& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
  }
  VELOX_CHECK(
      !tbls.empty(),
      "Expected at least one build table after concatenation. planNodeId: {}",
      planNodeId());
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Build table batches count: " << tbls.size();
    VLOG(1) << "Build table number of columns: " << tbls[0]->num_columns();
    for (size_t i = 0; i < tbls.size(); i++) {
      VLOG(1) << "Build table " << i
              << ": number of rows: " << tbls[i]->num_rows();
    }
  }

  auto rightKeys = joinNode_->rightKeys();

  auto buildKeyIndices = std::vector<cudf::size_type>(rightKeys.size());
  auto buildType = joinNode_->sources()[1]->outputType();
  for (size_t i = 0; i < buildKeyIndices.size(); i++) {
    buildKeyIndices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  {
    int64_t totalNullKeyRows = 0;
    for (const auto& tbl : tbls) {
      auto n = tbl->num_rows();
      if (n == 0) {
        continue;
      }
      auto nonNull =
          cudf::drop_nulls(*tbl, buildKeyIndices, stream);
      totalNullKeyRows += n - nonNull->num_rows();
    }
    auto lockedStats = stats_.wlock();
    lockedStats->numNullKeys += totalNullKeyRows;
  }

  // Hash table construction is deferred to the probe side (getOutput).
  // With N concurrent tasks, building hash tables here causes all N sets
  // to persist in GPU memory simultaneously (GpuGuard only limits
  // concurrent execution, not memory residency). Deferring to probe
  // ensures at most GpuGuard-max hash table sets exist at any time.
  std::vector<std::shared_ptr<cudf::hash_join>> hashObjects(
      tbls.size(), nullptr);

  std::vector<std::shared_ptr<cudf::table>> shared_tbls;
  for (auto& tbl : tbls) {
    shared_tbls.push_back(std::move(tbl));
  }
  VELOX_CHECK_EQ(
      shared_tbls.size(),
      hashObjects.size(),
      "Mismatched build table/hash object counts. planNodeId: {}, splitGroupId: {}",
      planNodeId(),
      operatorCtx_->driverCtx()->splitGroupId);
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Prepared hash join payload: shared_tbls=" << shared_tbls.size()
            << ", hashObjects=" << hashObjects.size();
    for (size_t i = 0; i < shared_tbls.size(); ++i) {
      VLOG(2) << "  shared_tbls[" << i
              << "]=" << static_cast<const void*>(shared_tbls[i].get());
    }
    for (size_t i = 0; i < hashObjects.size(); ++i) {
      VLOG(2) << "  hashObjects[" << i
              << "]=" << static_cast<const void*>(hashObjects[i].get());
    }
  }
  // set hash table to CudfHashJoinBridge
  const auto splitGroupId = operatorCtx_->driverCtx()->splitGroupId;
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild bridge lookup: planNodeId=" << planNodeId()
            << ", splitGroupId=" << splitGroupId
            << ", joinBridge=" << static_cast<const void*>(joinBridge.get())
            << ", cudfHashJoinBridge="
            << static_cast<const void*>(cudfHashJoinBridge.get());
  }
  VELOX_CHECK_NOT_NULL(
      joinBridge,
      "Expected JoinBridge for CudfHashJoinBuild. planNodeId: {}, splitGroupId: {}",
      planNodeId(),
      splitGroupId);
  VELOX_CHECK_NOT_NULL(
      cudfHashJoinBridge,
      "Expected CudfHashJoinBridge for CudfHashJoinBuild. planNodeId: {}, splitGroupId: {}, joinBridge: {}",
      planNodeId(),
      splitGroupId,
      static_cast<const void*>(joinBridge.get()));

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild setting build stream: planNodeId="
            << planNodeId() << ", splitGroupId=" << splitGroupId;
  }
  cudfHashJoinBridge->setBuildStream(stream);
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild setBuildStream completed: planNodeId="
            << planNodeId() << ", splitGroupId=" << splitGroupId;
    VLOG(1) << "CudfHashJoinBuild setting hash table: planNodeId="
            << planNodeId() << ", splitGroupId=" << splitGroupId;
  }
  cudfHashJoinBridge->setHashTable(
      std::make_optional(
          std::make_pair(std::move(shared_tbls), std::move(hashObjects))));
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild setHashTable completed: planNodeId="
            << planNodeId() << ", splitGroupId=" << splitGroupId;
  }
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

CudfHashJoinProbe::CudfHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : exec::Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "CudfHashJoinProbe"),
      NvtxHelper(
          nvtx3::rgb{0, 128, 128}, // Teal
          operatorId,
          fmt::format("[{}]", joinNode->id())),
      joinNode_(joinNode),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "CudfHashJoinProbe constructor";
  }
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < probeType_->names().size(); i++) {
      VLOG(1) << "Left column " << i << ": " << probeType_->names()[i];
    }

    for (int i = 0; i < buildType_->names().size(); i++) {
      VLOG(1) << "Right column " << i << ": " << buildType_->names()[i];
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      VLOG(1) << "Left key " << i << ": " << leftKeys[i]->name() << " "
              << leftKeys[i]->type()->kind();
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      VLOG(1) << "Right key " << i << ": " << rightKeys[i]->name() << " "
              << rightKeys[i]->type()->kind();
    }
  }

  auto const probeTableNumColumns = probeType_->size();
  leftKeyIndices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < leftKeyIndices_.size(); i++) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType_->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(leftKeyIndices_[i], probeTableNumColumns);
  }
  auto const buildTableNumColumns = buildType_->size();
  rightKeyIndices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < rightKeyIndices_.size(); i++) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType_->getChildIdx(rightKeys[i]->name()));
    VELOX_CHECK_LT(rightKeyIndices_[i], buildTableNumColumns);
  }

  auto outputType = joinNode_->outputType();
  auto numOutputColumns = outputType->names().size();
  if (joinNode_->isLeftSemiProjectJoin()) {
    VELOX_CHECK_GE(numOutputColumns, 1);
    matchColumnOutputIndex_ =
        static_cast<int32_t>(numOutputColumns - 1);
    VELOX_CHECK_EQ(
        outputType->childAt(matchColumnOutputIndex_),
        BOOLEAN());
    --numOutputColumns;
  }

  leftColumnIndicesToGather_ = std::vector<cudf::size_type>();
  rightColumnIndicesToGather_ = std::vector<cudf::size_type>();
  leftColumnOutputIndices_ = std::vector<size_t>();
  rightColumnOutputIndices_ = std::vector<size_t>();
  for (size_t i = 0; i < numOutputColumns; i++) {
    auto const outputName = outputType->names()[i];
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "Output column " << i << ": " << outputName;
    }
    auto channel = probeType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input",
        outputType->children()[i]);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < leftColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Left index to gather " << i << ": "
              << leftColumnIndicesToGather_[i];
    }

    for (int i = 0; i < rightColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Right index to gather " << i << ": "
              << rightColumnIndicesToGather_[i];
    }
  }

  // Setup filter in case it exists
  if (joinNode_->filter()) {
    // simplify expression
    exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
    VELOX_CHECK_EQ(exprs.exprs().size(), 1);
    useAstFilter_ = CudfConfig::getInstance().astExpressionEnabled &&
        !containsDecimalType(exprs.exprs()[0]);
  }
}

void CudfHashJoinProbe::initialize() {
  Operator::initialize();

  if (!joinNode_->filter()) {
    return;
  }

  exec::ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
  VELOX_CHECK_EQ(exprs.exprs().size(), 1);

  // Create a reusable evaluator for the filter column. This is expensive to
  // build, and the expression + input schema are stable for the lifetime of
  // the operator instance.
  std::vector<velox::RowTypePtr> filterRowTypes{probeType_, buildType_};
  filterEvaluator_ = createCudfExpression(
      exprs.exprs()[0],
      facebook::velox::type::concatRowTypes(filterRowTypes));

  // We don't need to get tables that contain conditional comparison columns
  // We'll pass the entire table. The ast will handle finding the required
  // columns. This is required because we build the ast with whole row schema
  // and the column locations in that schema translate to column locations
  // in whole tables

  if (useAstFilter_) {
    try {
      if (joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) {
        createAstTree(
            exprs.exprs()[0],
            tree_,
            scalars_,
            buildType_,
            probeType_,
            rightPrecomputeInstructions_,
            leftPrecomputeInstructions_);
      } else {
        createAstTree(
            exprs.exprs()[0],
            tree_,
            scalars_,
            probeType_,
            buildType_,
            leftPrecomputeInstructions_,
            rightPrecomputeInstructions_);
      }
    } catch (const VeloxException& e) {
      LOG(WARNING)
          << "CudfHashJoinProbe: AST tree creation failed for filter '"
          << joinNode_->filter()->toString()
          << "', disabling AST filter: " << e.what();
      useAstFilter_ = false;
      tree_ = {};
      scalars_.clear();
      leftPrecomputeInstructions_.clear();
      rightPrecomputeInstructions_.clear();
    }
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinProbe::needsInput";
  }
  if (joinNode_->isRightSemiFilterJoin()) {
    return !noMoreInput_;
  }
  auto targetBytes = CudfConfig::getInstance().gpuTargetBatchBytes;
  auto minRows = CudfConfig::getInstance().gpuTargetBatchRows;
  bool belowThreshold;
  if (accumulatedProbeInputs_.empty()) {
    belowThreshold = true;
  } else if (targetBytes > 0 && minRows > 0) {
    belowThreshold =
        accumulatedProbeBytes_ < targetBytes &&
        accumulatedProbeRows_ < minRows;
  } else if (targetBytes > 0) {
    belowThreshold = accumulatedProbeBytes_ < targetBytes;
  } else if (minRows > 0) {
    belowThreshold = accumulatedProbeRows_ < minRows;
  } else {
    belowThreshold = false;
  }
  return !noMoreInput_ && !finished_ && input_ == nullptr && belowThreshold;
}

void CudfHashJoinProbe::addInput(RowVectorPtr input) {
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  if (joinNode_->isRightSemiFilterJoin()) {
    if (input->size() > 0) {
      inputs_.push_back(std::move(cudfInput));
    }
    return;
  }

  if (input->size() > 0) {
    accumulatedProbeRows_ += input->size();
    accumulatedProbeBytes_ += cudfInput->estimateFlatSize();
    accumulatedProbeInputs_.push_back(std::move(cudfInput));
  }
}

void CudfHashJoinProbe::noMoreInput() {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinProbe::noMoreInput";
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  Operator::noMoreInput();
  if (!joinNode_->isRightJoin() && !joinNode_->isRightSemiFilterJoin() &&
      !joinNode_->isFullJoin()) {
    return;
  }
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    isLastDriver_ = true;
    if (hashObject_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();
      // Keep old flag columns alive until after sync: the binary_operation
      // on `stream` reads from them asynchronously, but their memory lives
      // on a different stream. Destroying them before the sync would free
      // the buffers via cudaFreeAsync on the old stream, racing with the
      // reads on `stream`.
      std::vector<std::unique_ptr<cudf::column>> oldFlags;
      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(planNodeId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        if (probe == nullptr) {
          continue;
        }
        for (size_t p = 0; p < rightMatchedFlags_.size(); ++p) {
          auto or_result = cudf::binary_operation(
              rightMatchedFlags_[p]->view(),
              probe->rightMatchedFlags_[p]->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              cudf::get_current_device_resource_ref());
          oldFlags.push_back(std::move(rightMatchedFlags_[p]));
          rightMatchedFlags_[p] = std::move(or_result);
        }
      }
      stream.synchronize();
    }
    return;
  }

  // Handling RightSemiFilterJoin
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
    VELOX_CHECK_NOT_NULL(probe);
    inputs_.insert(inputs_.end(), probe->inputs_.begin(), probe->inputs_.end());
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto tbl = getConcatenatedTable(
      inputs_, joinNode_->sources()[1]->outputType(), stream);

  VELOX_CHECK_NOT_NULL(tbl);

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << tbl->num_columns();
    VLOG(1) << "Probe table number of rows: " << tbl->num_rows();
  }

  // Store the concatenated table in input_
  input_ = std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      joinNode_->outputType(),
      tbl->num_rows(),
      std::move(tbl),
      stream);

  inputs_.clear();
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::unfilteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> joinedCols;
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
  auto leftResult = cudf::gather(leftInput, leftIndicesCol, oobPolicy, stream);
  auto rightResult =
      cudf::gather(rightInput, rightIndicesCol, oobPolicy, stream);

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Left result number of columns: " << leftResult->num_columns();
    VLOG(1) << "Right result number of columns: " << rightResult->num_columns();
  }

  auto leftCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.resize(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    joinedCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }
  if (buildStream_.has_value()) {
    // Ensure deallocation of build table happens after probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(joinedCols));
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::filteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    std::function<std::vector<std::unique_ptr<cudf::column>>(
        std::vector<std::unique_ptr<cudf::column>>&&,
        cudf::column_view)> func,
    rmm::cuda_stream_view stream) {
  auto leftResult =
      cudf::gather(leftTableView, leftIndicesCol, oobPolicy, stream);
  auto rightResult =
      cudf::gather(rightTableView, rightIndicesCol, oobPolicy, stream);
  auto leftColsSize = leftResult->num_columns();
  auto rightColsSize = rightResult->num_columns();

  std::vector<std::unique_ptr<cudf::column>> joinedCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.insert(
      joinedCols.end(),
      std::make_move_iterator(rightCols.begin()),
      std::make_move_iterator(rightCols.end()));

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before filteredOutput()");
  std::vector<cudf::column_view> joinedColViews;
  joinedColViews.reserve(joinedCols.size());
  for (const auto& col : joinedCols) {
    joinedColViews.push_back(col->view());
  }
  auto filterColumns = filterEvaluator_->eval(
      joinedColViews, stream, cudf::get_current_device_resource_ref());
  auto filterColumn = asView(filterColumns);

  joinedCols = func(std::move(joinedCols), filterColumn);

  auto filteredjoinedCols =
      std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[leftColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColumnIndicesToGather_[i]]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[rightColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
  }
  joinedCols = std::move(filteredjoinedCols);
  if (buildStream_.has_value()) {
    // Ensure any deallocation of join indices is ordered wrt probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(joinedCols));
}

std::unique_ptr<cudf::table> CudfHashJoinProbe::filteredOutputIndices(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    cudf::table_view extendedLeftView,
    cudf::table_view extendedRightView,
    cudf::join_kind joinKind,
    rmm::cuda_stream_view stream) {
  // Use extended views (with precomputed columns) for filter evaluation
  auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
      cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesCol,
          rightIndicesCol,
          tree_.back(),
          joinKind,
          stream);

  auto filteredLeftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
  auto filteredRightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
  auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
  auto filteredRightIndicesCol = cudf::column_view{filteredRightIndicesSpan};
  // Use original views (without precomputed columns) for gathering output
  return unfilteredOutput(
      leftTableView,
      filteredLeftIndicesCol,
      rightTableView,
      filteredRightIndicesCol,
      stream);
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::innerJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && useAstFilter_ &&
      !leftPrecomputeInstructions_.empty()) {
    try {
      auto leftColumnViews = tableViewToColumnViews(leftTableView);
      leftPrecomputed = precomputeSubexpressions(
          leftColumnViews,
          leftPrecomputeInstructions_,
          scalars_,
          probeType_,
          stream);
      extendedLeftView =
          createExtendedTableView(leftTableView, leftPrecomputed);
    } catch (const VeloxException& e) {
      LOG(WARNING)
          << "CudfHashJoinProbe::innerJoin: left precompute failed, "
          << "disabling AST filter for planNode " << joinNode_->id()
          << ": " << e.what();
      useAstFilter_ = false;
      leftPrecomputed.clear();
      extendedLeftView = leftTableView;
    }
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && useAstFilter_ &&
         !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // left = probe, right = build
    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      // Make build stream wait for probe tables to become valid
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }

    std::pair<
        std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
        std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
        joinResult;
    try {
      joinResult = hb->inner_join(
          leftTableView.select(leftKeyIndices_),
          std::nullopt,
          buildStream_.has_value() ? buildStream_.value() : stream);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      if (isCudaRelatedError(e)) {
        throw;
      }
      VELOX_FAIL(
          "GPU inner_join failed (probe={} rows, build={} rows, "
          "planNode={}): {}",
          leftTableView.num_rows(),
          rightTableView.num_rows(),
          joinNode_->id(),
          e.what());
    }
    auto& [leftJoinIndices, rightJoinIndices] = joinResult;

    if (buildStream_.has_value()) {
      // Make probe stream wait for join completion before using indices
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }

    auto joinOutputRows =
        static_cast<int64_t>(leftJoinIndices->size());
    auto outputCols = static_cast<int64_t>(outputType_->size());
    VELOX_CHECK_LE(
        joinOutputRows,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
        "Inner join output ({} rows) exceeds cudf::size_type limit. "
        "Probe={} rows, build={} rows, planNode={}. "
        "Consider increasing shuffle partitions to reduce data skew.",
        joinOutputRows,
        leftTableView.num_rows(),
        rightTableView.num_rows(),
        joinNode_->id());

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    try {
      if (joinNode_->filter()) {
        if (useAstFilter_) {
          cudfOutputs.push_back(filteredOutputIndices(
              leftTableView,
              leftIndicesCol,
              rightTableView,
              rightIndicesCol,
              extendedLeftView,
              extendedRightView,
              cudf::join_kind::INNER_JOIN,
              stream));
        } else {
          auto filterFunc =
              [stream](
                  std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                  cudf::column_view filterColumn) {
                auto filterTable =
                    std::make_unique<cudf::table>(std::move(joinedCols));
                auto filteredTable = cudf::apply_boolean_mask(
                    *filterTable, filterColumn, stream, cudf::get_current_device_resource_ref());
                return filteredTable->release();
              };
          cudfOutputs.push_back(filteredOutput(
              leftTableView,
              leftIndicesCol,
              rightTableView,
              rightIndicesCol,
              filterFunc,
              stream));
        }
      } else {
        cudfOutputs.push_back(unfilteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            stream));
      }
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      if (isCudaRelatedError(e)) {
        throw;
      }
      VELOX_FAIL(
          "GPU join gather/filter failed (joinOutput={} rows, "
          "probe={} rows, build={} rows, planNode={}): {}",
          joinOutputRows,
          leftTableView.num_rows(),
          rightTableView.num_rows(),
          joinNode_->id(),
          e.what());
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::leftJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && useAstFilter_ &&
      !leftPrecomputeInstructions_.empty()) {
    try {
      auto leftColumnViews = tableViewToColumnViews(leftTableView);
      leftPrecomputed = precomputeSubexpressions(
          leftColumnViews,
          leftPrecomputeInstructions_,
          scalars_,
          probeType_,
          stream);
      extendedLeftView =
          createExtendedTableView(leftTableView, leftPrecomputed);
    } catch (const VeloxException& e) {
      LOG(WARNING)
          << "CudfHashJoinProbe::leftJoin: left precompute failed, "
          << "disabling AST filter for planNode " << joinNode_->id()
          << ": " << e.what();
      useAstFilter_ = false;
      leftPrecomputed.clear();
      extendedLeftView = leftTableView;
    }
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && useAstFilter_ &&
         !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->left_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      if (useAstFilter_) {
        cudfOutputs.push_back(filteredOutputIndices(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            extendedLeftView,
            extendedRightView,
            cudf::join_kind::LEFT_JOIN,
            stream));
      } else {
        auto filterFunc =
            [stream](
                std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                cudf::column_view filterColumn) {
              auto filterTable =
                  std::make_unique<cudf::table>(std::move(joinedCols));
              auto filteredTable = cudf::apply_boolean_mask(
                  *filterTable, filterColumn, stream, cudf::get_current_device_resource_ref());
              return filteredTable->release();
            };
        cudfOutputs.push_back(filteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            filterFunc,
            stream));
      }
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::rightJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }
    // cudf::scatter is async: it enqueues a device memcpy of the old flags
    // (the target) plus a thrust::scatter kernel onto `stream`, then returns
    // immediately. The old rightMatchedFlags_[i] column must stay alive until
    if (!joinNode_->filter() && rightTableView.num_rows() > 0) {
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream),
          stream,
          cudf::get_current_device_resource_ref());

      auto matchedInBatch = cudf::contains(rightIdxCol, rowIndices->view());

      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          cudf::get_current_device_resource_ref());
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto numBuildRows = rightTableView.num_rows();
      auto filterFunc =
          [&rightMatchedFlags, rightIndicesSpan, numBuildRows, stream](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            // apply the filter
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable =
                cudf::apply_boolean_mask(*filterTable, filterColumn, stream);
            joinedCols = filteredTable->release();

            // For streaming right join, after applying filter, we record
            // matched right indices filter rightJoinIndices with the same mask
            // to update matched flags
            auto rightIdxCol = cudf::column_view{rightIndicesSpan};
            auto filteredIdxTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{rightIdxCol}},
                filterColumn,
                stream);
            auto filteredCols = filteredIdxTable->release();
            auto filteredRightIdxCol = std::move(filteredCols[0]);

            if (numBuildRows > 0) {
              auto rowIndices = cudf::sequence(
                  numBuildRows,
                  cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                  cudf::numeric_scalar<cudf::size_type>(1, true, stream),
                  stream,
                  cudf::get_current_device_resource_ref());

              auto matchedInBatch = cudf::contains(
                  filteredRightIdxCol->view(), rowIndices->view());

              auto updatedFlags = cudf::binary_operation(
                  rightMatchedFlags->view(),
                  matchedInBatch->view(),
                  cudf::binary_operator::BITWISE_OR,
                  cudf::data_type{cudf::type_id::BOOL8},
                  stream,
                  cudf::get_current_device_resource_ref());
              rightMatchedFlags = std::move(updatedFlags);
            }
            return std::move(joinedCols);
          };
      cudfOutputs.push_back(filteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          filterFunc,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::fullJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
    }
    // Use left_join to get all probe rows (matched + unmatched).
    // Track matched build rows in rightMatchedFlags_ for last driver to emit
    // unmatched build rows at the end.
    auto [leftJoinIndices, rightJoinIndices] = hb->left_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        buildStream_.has_value() ? buildStream_.value() : stream);
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(buildStream_.value()).waitOn(stream);
    }
    if (!joinNode_->filter() && rightTableView.num_rows() > 0) {
      auto rightIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream),
          stream,
          cudf::get_current_device_resource_ref());

      auto matchedInBatch = cudf::contains(rightIdxCol, rowIndices->view());

      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          cudf::get_current_device_resource_ref());
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    if (joinNode_->filter()) {
      // Use filter_join_indices with LEFT_JOIN to get proper full join probe
      // semantics: all probe rows are kept, build columns are NULL when filter
      // fails or no match.
      auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
          cudf::filter_join_indices(
              leftTableView,
              rightTableView,
              leftIndicesCol,
              rightIndicesCol,
              tree_.back(),
              cudf::join_kind::LEFT_JOIN,
              stream);

      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto filteredRightIndicesSpan =
          cudf::device_span<cudf::size_type const>{*filteredRightJoinIndices};
      auto filteredRightIdxCol = cudf::column_view{filteredRightIndicesSpan};

      if (rightTableView.num_rows() > 0) {
        auto n = rightTableView.num_rows();
        auto rowIndices = cudf::sequence(
            n,
            cudf::numeric_scalar<cudf::size_type>(0, true, stream),
            cudf::numeric_scalar<cudf::size_type>(1, true, stream),
            stream,
            cudf::get_current_device_resource_ref());

        auto matchedInBatch =
            cudf::contains(filteredRightIdxCol, rowIndices->view());

        auto updatedFlags = cudf::binary_operation(
            rightMatchedFlags->view(),
            matchedInBatch->view(),
            cudf::binary_operator::BITWISE_OR,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            cudf::get_current_device_resource_ref());
        rightMatchedFlags = std::move(updatedFlags);
      }

      // Build output using filtered indices
      auto filteredLeftIndicesSpan =
          cudf::device_span<cudf::size_type const>{*filteredLeftJoinIndices};
      auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
      auto filteredRightIndicesCol =
          cudf::column_view{filteredRightIndicesSpan};
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          filteredLeftIndicesCol,
          rightTableView,
          filteredRightIndicesCol,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::leftSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;

    if (joinNode_->filter()) {
      if (!useAstFilter_) {
        VELOX_NYI("Join filter requires AST for semi joins");
      }
      leftJoinIndices = cudf::mixed_left_semi_join(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftTableView,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_),
          stream,
          cudf::get_current_device_resource_ref());
    }

    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::empty_like(leftIndicesCol);

    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        leftIndicesCol,
        rightTableView,
        rightIndicesCol->view(),
        stream));
  }
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>>
CudfHashJoinProbe::leftSemiProjectJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  auto& rightTables = hashObject_.value().first;
  auto leftNumRows = leftTableView.num_rows();

  if (leftNumRows == 0) {
    return cudfOutputs;
  }

  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream);
  auto matchFlags = cudf::make_column_from_scalar(
      falseScalar,
      leftNumRows,
      stream,
      cudf::get_current_device_resource_ref());

  auto rowIndices = cudf::sequence(
      leftNumRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream),
      stream,
      cudf::get_current_device_resource_ref());

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>
        leftJoinIndices;

    if (joinNode_->filter()) {
      leftJoinIndices = cudf::mixed_left_semi_join(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftTableView,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_),
          stream,
          cudf::get_current_device_resource_ref());
    }

    if (leftJoinIndices->size() > 0) {
      auto leftIdxCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{
              *leftJoinIndices}};
      auto matchedInBatch =
          cudf::contains(leftIdxCol, rowIndices->view());
      auto updated = cudf::binary_operation(
          matchFlags->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          cudf::get_current_device_resource_ref());
      matchFlags = std::move(updated);
    }
  }

  if (joinNode_->isNullAware()) {
    auto mr = cudf::get_current_device_resource_ref();
    bool buildIsEmpty = true;
    bool buildHasNullKeys = false;
    for (const auto& rt : rightTables) {
      if (rt->view().num_rows() > 0) {
        buildIsEmpty = false;
        if (cudf::has_nulls(
                rt->view().select(rightKeyIndices_))) {
          buildHasNullKeys = true;
        }
      }
    }

    // x IN (empty set) = false for any x, including null.
    if (!buildIsEmpty) {
      auto trueScalar =
          cudf::numeric_scalar<bool>(true, true, stream);
      auto probeKeyNotNull = cudf::make_column_from_scalar(
          trueScalar, leftNumRows, stream, mr);
      for (auto ki : leftKeyIndices_) {
        auto keyCol = leftTableView.column(ki);
        if (keyCol.has_nulls()) {
          auto isValid =
              cudf::is_valid(keyCol, stream, mr);
          probeKeyNotNull = cudf::binary_operation(
              probeKeyNotNull->view(),
              isValid->view(),
              cudf::binary_operator::BITWISE_AND,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              mr);
        }
      }

      std::unique_ptr<cudf::column> validMask;
      if (buildHasNullKeys) {
        validMask = cudf::binary_operation(
            probeKeyNotNull->view(),
            matchFlags->view(),
            cudf::binary_operator::BITWISE_AND,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr);
      } else {
        validMask = std::move(probeKeyNotNull);
      }

      auto nullScalar =
          cudf::numeric_scalar<bool>(false, false, stream);
      matchFlags = cudf::copy_if_else(
          matchFlags->view(),
          nullScalar,
          validMask->view(),
          stream,
          mr);
    }
  }

  auto leftInput =
      leftTableView.select(leftColumnIndicesToGather_);
  std::vector<std::unique_ptr<cudf::column>> outCols(
      outputType_->size());
  for (size_t j = 0; j < leftColumnOutputIndices_.size(); ++j) {
    auto outIdx = leftColumnOutputIndices_[j];
    outCols[outIdx] = std::make_unique<cudf::column>(
        leftInput.column(j),
        stream,
        cudf::get_current_device_resource_ref());
  }
  outCols[matchColumnOutputIndex_] = std::move(matchFlags);

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  cudfOutputs.push_back(
      std::make_unique<cudf::table>(std::move(outCols)));
  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>>
CudfHashJoinProbe::rightSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  VELOX_CHECK(
      !rightTables.empty(),
      "rightTables is empty in rightSemiFilterJoin, planNode={}",
      joinNode_->id());
  auto rightTableView = rightTables[0]->view();

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for rightSemiFilterJoin");

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightJoinIndices;
  if (joinNode_->filter()) {
    if (!useAstFilter_) {
      VELOX_NYI("Join filter requires AST for semi joins");
    }
    rightJoinIndices = cudf::mixed_left_semi_join(
        rightTableView.select(rightKeyIndices_),
        leftTableView.select(leftKeyIndices_),
        rightTableView,
        leftTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        cudf::get_current_device_resource_ref());
  } else {
    cudf::filtered_join filter_join(
        leftTableView.select(leftKeyIndices_),
        cudf::null_equality::UNEQUAL,
        cudf::set_as_build_table::RIGHT,
        stream);
    rightJoinIndices = filter_join.semi_join(
        rightTableView.select(rightKeyIndices_),
        stream,
        cudf::get_current_device_resource_ref());
  }

  auto rightIndicesSpan =
      cudf::device_span<cudf::size_type const>{*rightJoinIndices};
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
  auto leftIndicesCol = cudf::empty_like(rightIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol->view(),
      rightTableView,
      rightIndicesCol,
      stream));

  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>> CudfHashJoinProbe::antiJoin(
    cudf::table_view leftTableViewParam,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  auto& rightTables = hashObject_.value().first;
  auto mr = cudf::get_current_device_resource_ref();

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for antiJoin");

  auto rightTableView = rightTables[0]->view();

  if (joinNode_->isNullAware() && joinNode_->filter()) {
    return nullAwareAntiJoinWithFilter(
        leftTableViewParam, rightTableView, stream);
  }

  std::unique_ptr<cudf::table> modifiedLeftTable;
  cudf::table_view leftTableView = leftTableViewParam;

  if (joinNode_->isNullAware() && !joinNode_->filter()) {
    auto const leftTableHasNulls =
        cudf::has_nulls(
            leftTableViewParam.select(leftKeyIndices_));
    auto const rightTableHasNulls =
        cudf::has_nulls(
            rightTableView.select(rightKeyIndices_));
    if (rightTables[0]->num_rows() > 0 &&
        !rightTableHasNulls && leftTableHasNulls) {
      modifiedLeftTable = cudf::drop_nulls(
          leftTableViewParam, leftKeyIndices_, stream);
      leftTableView = modifiedLeftTable->view();
    }
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>>
      leftJoinIndices;
  if (joinNode_->filter()) {
    if (!useAstFilter_) {
      VELOX_NYI("Join filter requires AST for anti joins");
    }
    leftJoinIndices = cudf::mixed_left_anti_join(
        leftTableView.select(leftKeyIndices_),
        rightTableView.select(rightKeyIndices_),
        leftTableView,
        rightTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        mr);
  } else {
    auto const rightTableHasNulls =
        cudf::has_nulls(
            rightTableView.select(rightKeyIndices_));
    if (joinNode_->isNullAware() && rightTableHasNulls) {
      leftJoinIndices =
          std::make_unique<rmm::device_uvector<cudf::size_type>>(
              0, stream, mr);
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          cudf::set_as_build_table::RIGHT,
          stream);
      leftJoinIndices = filter_join.anti_join(
          leftTableView.select(leftKeyIndices_),
          stream,
          mr);
    }
  }

  auto leftIndicesSpan =
      cudf::device_span<cudf::size_type const>{*leftJoinIndices};
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::empty_like(leftIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol,
      rightTableView,
      rightIndicesCol->view(),
      stream));

  return cudfOutputs;
}

std::vector<std::unique_ptr<cudf::table>>
CudfHashJoinProbe::nullAwareAntiJoinWithFilter(
    cudf::table_view leftTableViewParam,
    cudf::table_view rightTableView,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  auto mr = cudf::get_current_device_resource_ref();

  auto leftNonNullTable =
      cudf::drop_nulls(leftTableViewParam, leftKeyIndices_, stream);
  auto leftNonNull = leftNonNullTable->view();
  auto leftN = leftNonNull.num_rows();

  auto buildHasNullKeys =
      cudf::has_nulls(rightTableView.select(rightKeyIndices_));

  std::unique_ptr<cudf::column> rowIndices;
  std::unique_ptr<cudf::column> candidateMask;
  if (leftN > 0) {
    rowIndices = cudf::sequence(
        leftN,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream),
        stream,
        mr);

    candidateMask = cudf::make_column_from_scalar(
        cudf::numeric_scalar<bool>(true, true, stream),
        leftN,
        stream,
        mr);
  }

  // Exclude left rows matching non-null-key build rows via
  // key equality + filter. With UNEQUAL null equality, null
  // build keys are invisible.
  if (leftN > 0 && rightTableView.num_rows() > 0) {
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>
        antiIndices;
    if (buildHasNullKeys) {
      // Extract non-null-key build rows
      auto rightNonNull = cudf::drop_nulls(
          rightTableView, rightKeyIndices_, stream);
      if (rightNonNull->num_rows() > 0) {
        antiIndices = cudf::mixed_left_anti_join(
            leftNonNull.select(leftKeyIndices_),
            rightNonNull->view().select(rightKeyIndices_),
            leftNonNull,
            rightNonNull->view(),
            tree_.back(),
            cudf::null_equality::UNEQUAL,
            stream,
            mr);
      }
    } else {
      antiIndices = cudf::mixed_left_anti_join(
          leftNonNull.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftNonNull,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          mr);
    }
    if (antiIndices && antiIndices->size() > 0) {
      auto antiCol = cudf::column_view{
          cudf::device_span<cudf::size_type const>{
              *antiIndices}};
      auto inAnti =
          cudf::contains(antiCol, rowIndices->view());
      candidateMask = cudf::binary_operation(
          candidateMask->view(),
          inAnti->view(),
          cudf::binary_operator::BITWISE_AND,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
    } else if (!antiIndices || antiIndices->size() == 0) {
      if (!buildHasNullKeys ||
          cudf::drop_nulls(
              rightTableView, rightKeyIndices_, stream)
                  ->num_rows() > 0) {
        // Anti-join returned nothing: all left rows matched.
        // Unless no non-null build rows existed.
        auto falseScalar =
            cudf::numeric_scalar<bool>(false, true, stream);
        candidateMask = cudf::make_column_from_scalar(
            falseScalar, leftN, stream, mr);
      }
    }
  }

  // Exclude left rows matching any null-key build row via
  // filter. Use dummy constant keys so every pair passes key
  // equality, letting the filter decide.
  if (leftN > 0 && buildHasNullKeys &&
      rightTableView.num_rows() > 0) {
    // Build per-row "any key is null" mask for right table
    auto rightN = rightTableView.num_rows();
    auto anyKeyNull = cudf::make_column_from_scalar(
        cudf::numeric_scalar<bool>(false, true, stream),
        rightN,
        stream,
        mr);
    for (auto keyIdx : rightKeyIndices_) {
      auto col = rightTableView.column(keyIdx);
      if (!col.has_nulls()) {
        continue;
      }
      auto nullScalar =
          cudf::make_default_constructed_scalar(
              col.type(), stream, mr);
      auto isNull = cudf::binary_operation(
          col,
          *nullScalar,
          cudf::binary_operator::NULL_EQUALS,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
      anyKeyNull = cudf::binary_operation(
          anyKeyNull->view(),
          isNull->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
    }

    auto rightNullKeys = cudf::apply_boolean_mask(
        rightTableView, anyKeyNull->view(), stream);
    auto nullKeyN = rightNullKeys->num_rows();

    if (nullKeyN > 0) {
      // Dummy constant keys: every (left, rightNull) pair
      // passes key equality so the filter is evaluated on all
      // cross-product pairs.
      auto zeroScalar = cudf::numeric_scalar<cudf::size_type>(
          0, true, stream);
      auto leftDummyKey = cudf::make_column_from_scalar(
          zeroScalar, leftN, stream, mr);
      auto rightDummyKey = cudf::make_column_from_scalar(
          zeroScalar, nullKeyN, stream, mr);

      auto semiIndices = cudf::mixed_left_semi_join(
          cudf::table_view({leftDummyKey->view()}),
          cudf::table_view({rightDummyKey->view()}),
          leftNonNull,
          rightNullKeys->view(),
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          mr);

      if (semiIndices->size() > 0) {
        auto semiCol = cudf::column_view{
            cudf::device_span<cudf::size_type const>{
                *semiIndices}};
        auto inSemi =
            cudf::contains(semiCol, rowIndices->view());
        auto notInSemi = cudf::unary_operation(
            inSemi->view(),
            cudf::unary_operator::NOT,
            stream);
        candidateMask = cudf::binary_operation(
            candidateMask->view(),
            notInSemi->view(),
            cudf::binary_operator::BITWISE_AND,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            mr);
      }
    }
  }

  // Collect surviving non-null-keyed rows
  std::unique_ptr<cudf::table> nonNullResult;
  if (leftN > 0) {
    auto leftInput =
        leftNonNull.select(leftColumnIndicesToGather_);
    nonNullResult = cudf::apply_boolean_mask(
        leftInput, candidateMask->view(), stream);
  }

  // Handle null-keyed probe rows.
  // null NOT IN (empty) = true  -> keep
  // null NOT IN (non-empty where filter passes) = null -> exclude
  // null NOT IN (non-empty where NO filter passes) = true -> keep
  std::unique_ptr<cudf::table> nullKeyResult;
  auto totalLeft = leftTableViewParam.num_rows();
  if (totalLeft > leftN) {
    // Extract rows where any key column is null
    auto trueScalar =
        cudf::numeric_scalar<bool>(true, true, stream);
    auto hasNullKey = cudf::make_column_from_scalar(
        cudf::numeric_scalar<bool>(false, true, stream),
        totalLeft, stream, mr);
    for (auto ki : leftKeyIndices_) {
      auto col = leftTableViewParam.column(ki);
      if (col.has_nulls()) {
        auto isNull = cudf::is_null(col, stream, mr);
        hasNullKey = cudf::binary_operation(
            hasNullKey->view(),
            isNull->view(),
            cudf::binary_operator::BITWISE_OR,
            cudf::data_type{cudf::type_id::BOOL8},
            stream, mr);
      }
    }
    auto leftNullKeyTable = cudf::apply_boolean_mask(
        leftTableViewParam, hasNullKey->view(), stream);
    auto nullN = leftNullKeyTable->num_rows();

    if (nullN > 0 && rightTableView.num_rows() == 0) {
      // Build is empty: all null-keyed rows survive
      nullKeyResult = std::make_unique<cudf::table>(
          leftNullKeyTable->view().select(
              leftColumnIndicesToGather_),
          stream, mr);
    } else if (nullN > 0 && rightTableView.num_rows() > 0) {
      // Cross-join anti: for each null-keyed probe row,
      // check if ANY build row passes the filter.
      auto buildN = rightTableView.num_rows();
      auto zeroScalar = cudf::numeric_scalar<cudf::size_type>(
          0, true, stream);
      auto dummyL = cudf::make_column_from_scalar(
          zeroScalar, nullN, stream, mr);
      auto dummyR = cudf::make_column_from_scalar(
          zeroScalar, buildN, stream, mr);

      auto antiIdx = cudf::mixed_left_anti_join(
          cudf::table_view({dummyL->view()}),
          cudf::table_view({dummyR->view()}),
          leftNullKeyTable->view(),
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream, mr);

      if (antiIdx->size() > 0) {
        auto idxCol = cudf::column_view{
            cudf::device_span<cudf::size_type const>{
                *antiIdx}};
        nullKeyResult = cudf::gather(
            leftNullKeyTable->view().select(
                leftColumnIndicesToGather_),
            idxCol, cudf::out_of_bounds_policy::DONT_CHECK,
            stream, mr);
      }
    }
  }

  // Concatenate non-null and null-key results
  std::vector<cudf::table_view> toConcat;
  if (nonNullResult && nonNullResult->num_rows() > 0) {
    toConcat.push_back(nonNullResult->view());
  }
  if (nullKeyResult && nullKeyResult->num_rows() > 0) {
    toConcat.push_back(nullKeyResult->view());
  }

  if (toConcat.empty()) {
    // Return an empty table with correct schema
    std::vector<std::unique_ptr<cudf::column>> emptyCols;
    for (size_t j = 0; j < leftColumnOutputIndices_.size(); ++j) {
      auto srcIdx = leftColumnIndicesToGather_[j];
      auto type = leftTableViewParam.column(srcIdx).type();
      emptyCols.push_back(
          cudf::make_empty_column(type));
    }
    std::vector<std::unique_ptr<cudf::column>> outCols(
        outputType_->size());
    for (size_t j = 0; j < leftColumnOutputIndices_.size(); ++j) {
      outCols[leftColumnOutputIndices_[j]] =
          std::move(emptyCols[j]);
    }
    if (buildStream_.has_value()) {
      cudaEvent_->recordFrom(stream).waitOn(
          buildStream_.value());
    }
    stream.synchronize();
    cudfOutputs.push_back(
        std::make_unique<cudf::table>(std::move(outCols)));
    return cudfOutputs;
  }

  std::unique_ptr<cudf::table> finalResult;
  if (toConcat.size() == 1) {
    finalResult = std::make_unique<cudf::table>(
        toConcat[0], stream, mr);
  } else {
    finalResult =
        cudf::concatenate(toConcat, stream, mr);
  }

  auto resultCols = finalResult->release();
  std::vector<std::unique_ptr<cudf::column>> outCols(
      outputType_->size());
  for (size_t j = 0; j < leftColumnOutputIndices_.size(); ++j) {
    outCols[leftColumnOutputIndices_[j]] =
        std::move(resultCols[j]);
  }

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  cudfOutputs.push_back(
      std::make_unique<cudf::table>(std::move(outCols)));
  return cudfOutputs;
}

RowVectorPtr CudfHashJoinProbe::getOutput() {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinProbe::getOutput";
  }
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  if (finished_ or !hashObject_.has_value()) {
    return nullptr;
  }

  if (!pendingJoinOutputs_.empty()) {
    auto tbl = std::move(pendingJoinOutputs_.back());
    pendingJoinOutputs_.pop_back();
    auto stream = cudfGlobalStreamPool().get_stream();
    if (pendingJoinOutputs_.empty()) {
      finished_ = noMoreInput_ && !joinNode_->isRightJoin() &&
          !joinNode_->isFullJoin();
    }
    auto const size = tbl->num_rows();
    if (tbl->num_columns() == 0 || size == 0) {
      return nullptr;
    }
    return std::make_shared<CudfVector>(
        pool(), outputType_, size, std::move(tbl), stream);
  }

  // Materialize accumulated probe inputs into input_ when the byte/row
  // threshold is reached or no more input is expected. This coalesces many
  // small GPU batches into one large batch, dramatically reducing
  // kernel-launch and stream-synchronize overhead.
  if (!joinNode_->isRightSemiFilterJoin() &&
      !accumulatedProbeInputs_.empty() && input_ == nullptr) {
    auto targetBytes = CudfConfig::getInstance().gpuTargetBatchBytes;
    auto minRows = CudfConfig::getInstance().gpuTargetBatchRows;
    bool thresholdReached;
    if (targetBytes > 0 && minRows > 0) {
      thresholdReached =
          accumulatedProbeBytes_ >= targetBytes ||
          accumulatedProbeRows_ >= minRows;
    } else if (targetBytes > 0) {
      thresholdReached = accumulatedProbeBytes_ >= targetBytes;
    } else if (minRows > 0) {
      thresholdReached = accumulatedProbeRows_ >= minRows;
    } else {
      thresholdReached = !accumulatedProbeInputs_.empty();
    }
    if (thresholdReached || noMoreInput_) {
      if (accumulatedProbeInputs_.size() == 1) {
        input_ = std::move(accumulatedProbeInputs_[0]);
      } else {
        auto stream = accumulatedProbeInputs_[0]->stream();
        auto probeType = joinNode_->sources()[0]->outputType();
        auto tbl =
            getConcatenatedTable(accumulatedProbeInputs_, probeType, stream);
        input_ = std::make_shared<CudfVector>(
            operatorCtx_->pool(),
            probeType,
            tbl->num_rows(),
            std::move(tbl),
            stream);
      }
      accumulatedProbeInputs_.clear();
      accumulatedProbeRows_ = 0;
      accumulatedProbeBytes_ = 0;
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            "numCoalescedBatches", RuntimeCounter(1));
      }
    }
  }

  if (!input_) {
    // If no more input, emit unmatched-right rows if needed.
    if ((joinNode_->isRightJoin() || joinNode_->isFullJoin()) && noMoreInput_ &&
        !finished_ && isLastDriver_) {
      auto& rightTables = hashObject_.value().first;
      auto stream = cudfGlobalStreamPool().get_stream();
      std::vector<std::unique_ptr<cudf::table>> toConcat;
      for (size_t i = 0; i < rightTables.size(); ++i) {
        auto& rightTable = rightTables[i];
        auto n = rightTable->num_rows();
        if (n == 0) {
          continue;
        }
        auto& flags = rightMatchedFlags_[i];
        // Build a boolean mask: unmatched = NOT(flags)
        auto boolMask = cudf::unary_operation(
            flags->view(), cudf::unary_operator::NOT, stream);

        // Count unmatched rows by summing the boolean mask
        auto unmatchedCountScalar = cudf::reduce(
            boolMask->view(),
            *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::INT32},
            stream);
        auto m = static_cast<cudf::numeric_scalar<int32_t>*>(
                     unmatchedCountScalar.get())
                     ->value(stream);
        if (m == 0) {
          continue;
        }

        // Build left null columns
        std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
        // Left side nulls (types derive from probe schema at the matching
        // channel indices)
        for (size_t li = 0; li < leftColumnOutputIndices_.size(); ++li) {
          auto outIdx = leftColumnOutputIndices_[li];
          auto probeChannel = leftColumnIndicesToGather_[li];
          auto leftCudfDataType =
              veloxToCudfDataType(probeType_->childAt(probeChannel));
          auto nullScalar = cudf::make_default_constructed_scalar(
              leftCudfDataType, stream, cudf::get_current_device_resource_ref());
          outCols[outIdx] = cudf::make_column_from_scalar(
              *nullScalar, m, stream, cudf::get_current_device_resource_ref());
        }
        // Right side - gather unmatched build columns if any
        if (!rightColumnIndicesToGather_.empty()) {
          auto rightInput =
              rightTable->view().select(rightColumnIndicesToGather_);
          auto unmatchedRight =
              cudf::apply_boolean_mask(rightInput, boolMask->view(), stream);
          auto rightCols = unmatchedRight->release();
          for (size_t ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
            auto outIdx = rightColumnOutputIndices_[ri];
            outCols[outIdx] = std::move(rightCols[ri]);
          }
        }
        toConcat.push_back(std::make_unique<cudf::table>(std::move(outCols)));
      }
      // TODO (dm): We build multiple right chunks only when they are too large
      // to fit in cudf::size_type. In case of a right join which doesn't have a
      // lot of matches we'll get outCols of similar size. This concatenation
      // will overflow. Try emitting result of one right chunk at a time.
      if (!toConcat.empty()) {
        auto out = concatenateTables(std::move(toConcat), stream);
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
  // Use getTableView() to avoid expensive materialization for packed_table.
  // cudfInput is staying alive until the table view is no longer needed.
  auto leftTableView = cudfInput->getTableView();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << leftTableView.num_columns();
    VLOG(1) << "Probe table number of rows: " << leftTableView.num_rows();
  }

  if (leftTableView.num_rows() == 0) {
    cudfInput.reset();
    input_.reset();
    finished_ =
        noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();
    return nullptr;
  }

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  for (auto i = 0; i < rightTables.size(); i++) {
    auto& rightTable = rightTables[i];
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(rightTable);
    if (CudfConfig::getInstance().debugEnabled) {
      if (rightTable != nullptr)
        VLOG(2) << "right_table is not nullptr " << rightTable.get()
                << " hasValue(" << hashObject_.has_value() << ")\n";
      if (hb != nullptr)
        VLOG(2) << "hb is not nullptr " << hb.get() << " hasValue("
                << hashObject_.has_value() << ")\n";
    }
  }

  {
    auto n = leftTableView.num_rows();
    if (n > 0) {
      auto nonNull =
          cudf::drop_nulls(leftTableView, leftKeyIndices_, stream);
      auto nullKeyRows =
          static_cast<int64_t>(n - nonNull->num_rows());
      if (nullKeyRows > 0) {
        auto lockedStats = stats_.wlock();
        lockedStats->numNullKeys += nullKeyRows;
      }
    }
  }

  // Build hash tables on-demand (transient). Construction is deferred from
  // the build phase to here so that at most GpuGuard-max hash table sets
  // exist at any instant, rather than one per Spark task.
  bool const needHashJoin =
      joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
      joinNode_->isRightJoin() || joinNode_->isFullJoin();
  if (needHashJoin) {
    auto& rightTables = hashObject_.value().first;
    auto& hbs = hashObject_.value().second;
    for (size_t i = 0; i < rightTables.size(); ++i) {
      if (!hbs[i]) {
        for (int attempt = 0;; ++attempt) {
          try {
            hbs[i] = std::make_shared<cudf::hash_join>(
                rightTables[i]->view().select(rightKeyIndices_),
                cudf::null_equality::UNEQUAL,
                stream);
            break;
          } catch (const std::bad_alloc& e) {
            if (attempt >= kOomMaxRetries) {
              throw;
            }
            LOG(WARNING)
                << "CudfHashJoinProbe OOM building hash table for planNode "
                << joinNode_->id() << " batch " << i << " (attempt "
                << (attempt + 1) << "): " << e.what()
                << ". Recovering GPU memory and retrying.";
            recoverGpuMemory();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(100 * (1 << attempt)));
          }
        }
      }
    }
  }

  auto executeJoin = [&](cudf::table_view probeView)
      -> std::vector<std::unique_ptr<cudf::table>> {
    switch (joinNode_->joinType()) {
      case core::JoinType::kInner:
        return innerJoin(probeView, stream);
      case core::JoinType::kLeft:
        return leftJoin(probeView, stream);
      case core::JoinType::kRight:
        return rightJoin(probeView, stream);
      case core::JoinType::kLeftSemiFilter:
        return leftSemiFilterJoin(probeView, stream);
      case core::JoinType::kLeftSemiProject:
        return leftSemiProjectJoin(probeView, stream);
      case core::JoinType::kRightSemiFilter:
        return rightSemiFilterJoin(probeView, stream);
      case core::JoinType::kAnti:
        return antiJoin(probeView, stream);
      case core::JoinType::kFull:
        return fullJoin(probeView, stream);
      default:
        VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
    }
  };

  bool const canSplitProbe =
      joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
      joinNode_->isLeftSemiFilterJoin() ||
      joinNode_->isLeftSemiProjectJoin() || joinNode_->isAntiJoin();
  static constexpr cudf::size_type kMinSplitRows = 1024;

  std::vector<std::unique_ptr<cudf::table>> cudfOutputs;
  std::vector<cudf::table_view> probeSlices;

  // Proactive probe splitting: when GPU memory is under pressure from
  // concurrent tasks, pre-split large probes to bound peak per-join
  // allocation and prevent cudaErrorIllegalAddress from memory corruption.
  if (canSplitProbe && leftTableView.num_rows() > kMinSplitRows) {
    size_t freeMem = 0, totalMem = 0;
    if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess && totalMem > 0) {
      size_t outputRowBytes =
          std::max(size_t(16), static_cast<size_t>(outputType_->size()) * 8);
      // Conservative: assume 20x amplification per probe row (joins can be
      // many-to-many). Each output row costs indices (8B) + data.
      size_t costPerProbeRow = (8 + outputRowBytes) * 20;
      // Allow each join call to use at most 25% of free GPU memory.
      size_t maxAlloc = freeMem / 4;
      auto maxRows = static_cast<cudf::size_type>(std::min(
          static_cast<size_t>(leftTableView.num_rows()),
          std::max(
              static_cast<size_t>(kMinSplitRows),
              maxAlloc / std::max(costPerProbeRow, size_t(1)))));
      if (maxRows < leftTableView.num_rows()) {
        LOG(INFO) << "Proactive probe split for planNode " << joinNode_->id()
                  << ": " << leftTableView.num_rows() << " rows -> chunks of "
                  << maxRows << " (freeMem=" << (freeMem >> 20) << "MB"
                  << ", totalMem=" << (totalMem >> 20) << "MB)";
        std::vector<cudf::size_type> splitIndices;
        for (cudf::size_type i = maxRows; i < leftTableView.num_rows();
             i += maxRows) {
          splitIndices.push_back(i);
        }
        auto chunks = cudf::split(leftTableView, splitIndices, stream);
        for (int j = static_cast<int>(chunks.size()) - 1; j >= 0; --j) {
          probeSlices.push_back(chunks[j]);
        }
      }
    }
  }
  if (probeSlices.empty()) {
    probeSlices.push_back(leftTableView);
  }

  while (!probeSlices.empty()) {
    auto slice = probeSlices.back();
    probeSlices.pop_back();

    try {
      auto results = executeJoin(slice);
      for (auto& r : results) {
        cudfOutputs.push_back(std::move(r));
      }
    } catch (const std::exception& e) {
      if (!isCudaRelatedError(e)) {
        throw;
      }

      // Force all pending GPU frees across all streams to complete.
      // With many concurrent tasks, cudaFreeAsync defers actual deallocation;
      // cudaDeviceSynchronize forces those frees, recovering memory.
      recoverGpuMemory();

      // If we can't split further, retry with backoff (other tasks may
      // complete and free memory during the wait).
      if (!canSplitProbe || slice.num_rows() <= kMinSplitRows) {
        bool retried = false;
        for (int attempt = 0; attempt < kOomMaxRetries; ++attempt) {
          LOG(WARNING)
              << "GPU join OOM with " << slice.num_rows()
              << " probe rows for planNode " << joinNode_->id()
              << " (retry " << (attempt + 1) << "/" << kOomMaxRetries
              << "): " << e.what();
          std::this_thread::sleep_for(
              std::chrono::milliseconds(100 * (1 << attempt)));
          recoverGpuMemory();
          try {
            auto results = executeJoin(slice);
            for (auto& r : results) {
              cudfOutputs.push_back(std::move(r));
            }
            retried = true;
            break;
          } catch (const std::exception& retryErr) {
            if (!isCudaRelatedError(retryErr)) {
              throw;
            }
            recoverGpuMemory();
          }
        }
        if (!retried) {
          VELOX_FAIL(
              "GPU join error: {} (probe={} rows, planNode={}). "
              "Consider reducing "
              "spark.gluten.sql.columnar.backend.velox.cudf.concurrentGpuTasks "
              "or increasing spark.sql.shuffle.partitions: {}",
              joinNode_->joinType(),
              slice.num_rows(),
              joinNode_->id(),
              e.what());
        }
        continue;
      }

      LOG(WARNING)
          << "GPU join CUDA error with " << slice.num_rows()
          << " probe rows for planNode " << joinNode_->id()
          << ": " << e.what()
          << ". Splitting probe in half and retrying.";

      try {
        auto half = static_cast<cudf::size_type>(slice.num_rows() / 2);
        auto splits = cudf::split(slice, {half}, stream);
        probeSlices.push_back(splits[1]);
        probeSlices.push_back(splits[0]);
      } catch (const std::exception& splitErr) {
        VELOX_FAIL(
            "GPU join error (device likely corrupted, split also failed): "
            "original: {}. split: {}. planNode={}",
            e.what(),
            splitErr.what(),
            joinNode_->id());
      }
    }
  }

  // Release transient hash tables immediately after probing to free GPU
  // memory for other tasks. They'll be rebuilt on the next getOutput() call.
  if (needHashJoin) {
    auto& hbs = hashObject_.value().second;
    for (auto& hb : hbs) {
      hb.reset();
    }
  }

  // Release input CudfVector to free GPU memory before creating output.
  // This reduces peak memory from (input + output) to max(input, output).
  // cudfInput must be released first since input_.reset() only decrements
  // the refcount while cudfInput still holds a reference.
  cudfInput.reset();
  input_.reset();

  // Remove empty tables before deciding how to return.
  cudfOutputs.erase(
      std::remove_if(
          cudfOutputs.begin(),
          cudfOutputs.end(),
          [](const std::unique_ptr<cudf::table>& t) {
            return !t || t->num_rows() == 0 || t->num_columns() == 0;
          }),
      cudfOutputs.end());

  if (cudfOutputs.empty()) {
    finished_ =
        noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();
    return nullptr;
  }

  if (cudfOutputs.size() == 1) {
    // Single result — return directly without concatenation.
    finished_ =
        noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();
    auto tbl = std::move(cudfOutputs[0]);
    return std::make_shared<CudfVector>(
        pool(), outputType_, tbl->num_rows(), std::move(tbl), stream);
  }

  // Multiple split results (from OOM probe splitting).
  // Instead of concatenating (which doubles peak GPU memory), store them
  // and return one per getOutput() call — analogous to Spark RAPIDS
  // JoinGatherer batched output pattern.
  pendingJoinOutputs_ = std::move(cudfOutputs);
  // Reverse so that pop_back returns results in order.
  std::reverse(pendingJoinOutputs_.begin(), pendingJoinOutputs_.end());

  auto tbl = std::move(pendingJoinOutputs_.back());
  pendingJoinOutputs_.pop_back();
  if (pendingJoinOutputs_.empty()) {
    finished_ =
        noMoreInput_ && !joinNode_->isRightJoin() && !joinNode_->isFullJoin();
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, tbl->num_rows(), std::move(tbl), stream);
}

bool CudfHashJoinProbe::skipProbeOnEmptyBuild() const {
  auto const joinType = joinNode_->joinType();
  return isInnerJoin(joinType) || isLeftSemiFilterJoin(joinType) ||
      isRightJoin(joinType) || isRightSemiFilterJoin(joinType) ||
      isRightSemiProjectJoin(joinType);
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      hashObject_.has_value()) {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  const auto splitGroupId = operatorCtx_->driverCtx()->splitGroupId;
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      splitGroupId, planNodeId());
  auto cudfJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinProbe bridge lookup: planNodeId=" << planNodeId()
            << ", splitGroupId=" << splitGroupId
            << ", joinBridge=" << static_cast<const void*>(joinBridge.get())
            << ", cudfJoinBridge="
            << static_cast<const void*>(cudfJoinBridge.get());
  }
  VELOX_CHECK_NOT_NULL(
      joinBridge,
      "Expected JoinBridge for CudfHashJoinProbe. planNodeId: {}, splitGroupId: {}",
      planNodeId(),
      splitGroupId);
  VELOX_CHECK_NOT_NULL(
      cudfJoinBridge,
      "Expected CudfHashJoinBridge for CudfHashJoinProbe. planNodeId: {}, splitGroupId: {}, joinBridge: {}",
      planNodeId(),
      splitGroupId,
      static_cast<const void*>(joinBridge.get()));
  VELOX_CHECK_NOT_NULL(future);
  auto hashObject = cudfJoinBridge->hashOrFuture(future);

  if (!hashObject.has_value()) {
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(2) << "CudfHashJoinProbe is blocked, waiting for join build";
    }
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  hashObject_ = std::move(hashObject);
  buildStream_ = cudfJoinBridge->getBuildStream();

  // Lazy initialize matched flags only when build side is done
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin()) {
    auto& rightTablesInit = hashObject_.value().first;
    rightMatchedFlags_.clear();
    rightMatchedFlags_.reserve(rightTablesInit.size());
    auto initStream = cudfGlobalStreamPool().get_stream();
    for (auto& rt : rightTablesInit) {
      auto n = rt->num_rows();
      if (n == 0) {
        rightMatchedFlags_.push_back(
            cudf::make_empty_column(cudf::data_type(cudf::type_id::BOOL8)));
        continue;
      }
      auto false_scalar = cudf::numeric_scalar<bool>(false, true, initStream);
      auto flags_col = cudf::make_column_from_scalar(
          false_scalar, n, initStream, cudf::get_current_device_resource_ref());
      rightMatchedFlags_.push_back(std::move(flags_col));
    }
    initStream.synchronize();
  }

  // Precompute right table columns if filter exists (once when build is done)
  if (joinNode_->filter() && useAstFilter_ &&
      !rightPrecomputeInstructions_.empty()) {
    try {
      auto& rightTablesInit = hashObject_.value().first;
      cachedRightPrecomputed_.clear();
      cachedExtendedRightViews_.clear();
      cachedRightPrecomputed_.reserve(rightTablesInit.size());
      cachedExtendedRightViews_.reserve(rightTablesInit.size());

      auto initStream = cudfGlobalStreamPool().get_stream();
      for (auto& rt : rightTablesInit) {
        auto rightTableView = rt->view();
        if (rightTableView.num_rows() == 0) {
          cachedRightPrecomputed_.emplace_back();
          cachedExtendedRightViews_.push_back(rightTableView);
          continue;
        }
        auto rightColumnViews = tableViewToColumnViews(rightTableView);
        auto rightPrecomputed = precomputeSubexpressions(
            rightColumnViews,
            rightPrecomputeInstructions_,
            scalars_,
            buildType_,
            initStream);
        auto extendedView =
            createExtendedTableView(rightTableView, rightPrecomputed);
        cachedRightPrecomputed_.push_back(std::move(rightPrecomputed));
        cachedExtendedRightViews_.push_back(extendedView);
      }
      initStream.synchronize();
    } catch (const VeloxException& e) {
      LOG(WARNING)
          << "CudfHashJoinProbe: right-side precompute failed, "
          << "disabling AST filter for planNode " << joinNode_->id()
          << ": " << e.what();
      useAstFilter_ = false;
      cachedRightPrecomputed_.clear();
      cachedExtendedRightViews_.clear();
    }
  }

  auto& rightTables = hashObject_.value().first;
  if (!rightTables.empty() && rightTables[0]->num_rows() == 0) {
    if (skipProbeOnEmptyBuild()) {
      if (operatorCtx_->driverCtx()
              ->queryConfig()
              .hashProbeFinishEarlyOnEmptyBuild()) {
        noMoreInput();
      } else {
        skipInput_ = true;
      }
    }
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isFullJoin()) &&
      future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  bool isFinished;
  if ((joinNode_->isRightJoin() || joinNode_->isFullJoin()) &&
      isLastDriver_) {
    // The last driver must wait until finished_ is set after emitting
    // unmatched build rows.
    isFinished = finished_;
  } else {
    isFinished = finished_ ||
        (noMoreInput_ && input_ == nullptr &&
         accumulatedProbeInputs_.empty() &&
         pendingJoinOutputs_.empty());
  }

  if (isFinished) {
    hashObject_.reset();
  }
  return isFinished;
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperator";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toJoinBridge";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    auto joinBridge = std::make_unique<CudfHashJoinBridge>();
    return joinBridge;
  }
  return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
