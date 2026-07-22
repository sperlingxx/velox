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
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

uint64_t concatThresholdFromEnv(const char* name, uint64_t fallback) {
  if (const char* value = std::getenv(name)) {
    try {
      return std::stoull(value);
    } catch (...) {
    }
  }
  return fallback;
}

uint64_t concatThreshold(
    exec::DriverCtx* driverCtx,
    const char* configKey,
    const char* envName,
    uint64_t fallback) {
  // Prefer the per-query value.  The process-wide CudfConfig is initialized
  // before registration, but a long-lived executor can subsequently execute
  // queries with different batch targets.
  const auto configured =
      driverCtx->queryConfig().get<uint64_t>(configKey, fallback);
  return concatThresholdFromEnv(envName, configured);
}

RowTypePtr getConcatOutputType(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  VELOX_CHECK_EQ(
      planNode->sources().size(),
      1,
      "CudfBatchConcat expects a single-source plan node");
  return planNode->sources()[0]->outputType();
}

size_t checkedConcatTargetRows(int32_t targetRows) {
  VELOX_CHECK_GT(
      targetRows, 0, "CudfBatchConcat target row count must be positive");
  return static_cast<size_t>(targetRows);
}

int32_t queryConcatTargetRows(exec::DriverCtx* driverCtx) {
  const auto targetRows = concatThreshold(
      driverCtx,
      CudfConfig::kCudfBatchSizeMinThreshold,
      "GLUTEN_CUDF_BATCH_SIZE_MIN_THRESHOLD_ROWS",
      CudfConfig::getInstance().batchSizeMinThreshold);
  VELOX_CHECK_LE(
      targetRows,
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
      "CudfBatchConcat target row count exceeds INT32_MAX");
  return static_cast<int32_t>(targetRows);
}

std::string getAggregationStep(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  const auto aggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  return aggregation ? fmt::format("{}", aggregation->step()) : "UNKNOWN";
}

bool logConcatConfig() {
  const auto* value = std::getenv("GLUTEN_CUDF_BATCH_CONCAT_LOG_CONFIG");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

} // namespace

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          planNode,
          getConcatOutputType(planNode)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          planNode,
          std::move(outputType),
          queryConcatTargetRows(driverCtx)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType,
    int32_t targetRows)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          std::move(planNode),
          std::move(outputType),
          targetRows,
          concatThreshold(
              driverCtx,
              CudfConfig::kCudfBatchSizeMinThresholdBytes,
              "GLUTEN_CUDF_BATCH_SIZE_MIN_THRESHOLD_BYTES",
              CudfConfig::getInstance().batchSizeMinThresholdBytes)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType,
    int32_t targetRows,
    uint64_t targetBytes)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          std::move(outputType),
          planNode->id(),
          "CudfBatchConcat",
          nvtx3::rgb{211, 211, 211}, /* LightGrey */
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      driverCtx_(driverCtx),
      aggregationStep_(getAggregationStep(planNode)),
      targetRows_(checkedConcatTargetRows(targetRows)),
      targetBytes_(targetBytes) {
  if (logConcatConfig()) {
    LOG(WARNING) << "CudfBatchConcat configured targetRows=" << targetRows_
                 << ", targetBytes=" << targetBytes_;
  }
}

bool CudfBatchConcat::needsInput() const {
  return !noMoreInput_ && outputQueue_.empty() &&
      currentNumRows_ < targetRows_ &&
      (targetBytes_ == 0 || currentNumBytes_ < targetBytes_);
}

void CudfBatchConcat::doAddInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  if (cudfVector->size() == 0) {
    return;
  }

  // Push input cudf table to buffer
  ++inputBatches_;
  totalInputRows_ += cudfVector->size();
  totalInputBytes_ += cudfVector->estimateFlatSize();
  currentNumRows_ += cudfVector->size();
  currentNumBytes_ += cudfVector->estimateFlatSize();
  buffer_.push_back(std::move(cudfVector));

  // Enforce the bound here as well as in needsInput(). Source pipelines may
  // have already scheduled another input before the driver observes the
  // updated needsInput() result. Keeping a ready output in outputQueue_ makes
  // the backpressure explicit and prevents scan batches from accumulating up
  // to device capacity. Avoid a redundant D2D concatenate for one large input.
  if (currentNumRows_ >= targetRows_ ||
      (targetBytes_ != 0 && currentNumBytes_ >= targetBytes_)) {
    if (buffer_.size() == 1) {
      outputQueue_.push(std::move(buffer_.front()));
      buffer_.clear();
    } else {
      const auto outputStream = buffer_.front()->stream();
      auto outputVectors = getConcatenatedCudfVectorsBatched(
          pool(),
          std::exchange(buffer_, {}),
          outputType_,
          outputStream,
          get_output_mr());
      for (auto& output : outputVectors) {
        outputQueue_.push(std::move(output));
      }
    }
    currentNumRows_ = 0;
    currentNumBytes_ = 0;
  }
}

RowVectorPtr CudfBatchConcat::doGetOutput() {
  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto output = std::move(outputQueue_.front());
    outputQueue_.pop();
    ++outputBatches_;
    return output;
  }

  // Merge tables if there are enough rows
  if (!buffer_.empty() &&
      (currentNumRows_ >= targetRows_ ||
       (targetBytes_ != 0 && currentNumBytes_ >= targetBytes_) ||
       noMoreInput_)) {
    // Preserve the zero-copy Exchange fast path when a single received batch
    // already satisfies the target (or is the final tail batch). CudfVector
    // carries its producing stream, so downstream operators can consume it
    // directly without rebinding allocation ownership to another stream.
    if (buffer_.size() == 1) {
      auto output = std::move(buffer_.front());
      buffer_.clear();
      currentNumRows_ = 0;
      currentNumBytes_ = 0;
      ++outputBatches_;
      return output;
    }
    // Use stream from existing buffer vectors
    const auto outputStream = buffer_[0]->stream();
    auto outputVectors = getConcatenatedCudfVectorsBatched(
        pool(),
        std::exchange(buffer_, {}),
        outputType_,
        outputStream,
        get_output_mr());

    currentNumRows_ = 0;
    currentNumBytes_ = 0;
    VELOX_CHECK_GT(outputVectors.size(), 0);

    for (auto it = outputVectors.begin(); it + 1 != outputVectors.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // If last table is a smaller batch and we still expect more input and keep
    // it in buffer.
    auto& last = outputVectors.back();
    auto rowCount = last->size();

    const auto lastBytes = last->estimateFlatSize();
    if (!noMoreInput_ && rowCount < targetRows_ &&
        (targetBytes_ == 0 || lastBytes < targetBytes_)) {
      currentNumRows_ = rowCount;
      currentNumBytes_ = lastBytes;
      buffer_.push_back(std::move(last));
    } else {
      outputQueue_.push(std::move(last));
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto output = std::move(outputQueue_.front());
      outputQueue_.pop();
      ++outputBatches_;
      return output;
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  const bool finished = noMoreInput_ && buffer_.empty() && outputQueue_.empty();
  if (finished && !summaryLogged_ && logConcatConfig()) {
    summaryLogged_ = true;
    LOG(WARNING) << "CudfBatchConcat summary planNode=" << planNodeId()
                 << ", step=" << aggregationStep_
                 << ", inputBatches=" << inputBatches_
                 << ", outputBatches=" << outputBatches_
                 << ", inputRows=" << totalInputRows_
                 << ", inputBytes=" << totalInputBytes_;
  }
  return finished;
}

} // namespace facebook::velox::cudf_velox
