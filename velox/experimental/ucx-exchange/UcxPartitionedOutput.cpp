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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include <algorithm>
#include <cstdlib>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
int64_t targetRowsPerUcxChunk(const core::QueryConfig& queryConfig) {
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_ROWS")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed > 0) {
        return parsed;
      }
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchRows();
}
} // namespace

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const RowTypePtr& inputType,
    const RowTypePtr& outputType,
    std::vector<uint32_t>& remap) {
  remap.clear();
  remap.reserve(outputType->size());
  std::unordered_map<std::string, size_t> nextOccurrence;
  for (uint32_t out = 0; out < outputType->size(); ++out) {
    const auto& name = outputType->nameOf(out);
    std::vector<uint32_t> matches;
    for (uint32_t in = 0; in < inputType->size(); ++in) {
      if (inputType->nameOf(in) == name &&
          inputType->childAt(in)->equivalent(*outputType->childAt(out))) {
        matches.push_back(in);
      }
    }
    VELOX_CHECK(
        !matches.empty(),
        "UCX output field {}:{} has no name-and-type match in input {}",
        name,
        outputType->childAt(out)->toString(),
        inputType->toString());
    auto& occurrence = nextOccurrence[name];
    const auto selected = matches[std::min(occurrence, matches.size() - 1)];
    ++occurrence;
    remap.push_back(selected);
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(UcxOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      targetRowsPerChunk_(targetRowsPerUcxChunk(ctx->queryConfig())) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(planNode->inputType(), planNode->outputType(), remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  CudaAllocationTraceScope allocationTrace(
      fmt::format("UcxPartitionedOutput task={} method=addInput", taskId()));
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");

  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingInputs_.push_back(std::move(cudfVector));

  if (targetRowsPerChunk_ <= 0 || pendingRows_ >= targetRowsPerChunk_) {
    flushPending();
  }
}

void UcxPartitionedOutput::flushPending() {
  CudaAllocationTraceScope allocationTrace(
      fmt::format(
          "UcxPartitionedOutput task={} method=flushPending", taskId()));
  if (pendingInputs_.empty()) {
    return;
  }

  try {
    cudf::table_view tableView;
    rmm::cuda_stream_view stream = pendingInputs_.back()->stream();
    // Keeps the merged table alive while tableView references it.
    std::unique_ptr<cudf::table> mergedTable;

    if (pendingInputs_.size() == 1) {
      // Fast path: use the single input's view directly (no GPU alloc).
      auto& cv = pendingInputs_[0];
      stream = cv->stream();
      tableView = remap_.empty()
          ? cv->getTableView()
          : cv->getTableView().select(remap_.begin(), remap_.end());
    } else {
      // Collect (remapped) table views.
      std::vector<cudf::table_view> views;
      std::vector<rmm::cuda_stream_view> inputStreams;
      views.reserve(pendingInputs_.size());
      inputStreams.reserve(pendingInputs_.size());
      for (auto& v : pendingInputs_) {
        inputStreams.push_back(v->stream());
        views.push_back(
            remap_.empty()
                ? v->getTableView()
                : v->getTableView().select(remap_.begin(), remap_.end()));
      }

      cudf::detail::join_streams(inputStreams, stream);
      mergedTable = cudf::concatenate(
          views, stream, cudf::get_current_device_resource_ref());

      orderCudfVectorDeallocationsAfterStream(
          pendingInputs_, inputStreams, stream);

      // Free input GPU memory before partitioning (peak = 2x -> 1x).
      pendingInputs_.clear();

      tableView = mergedTable->view();
    }

    // Partition + enqueue (identical to previous addInput logic).
    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
        hashPartition(tableView, stream);
      } else {
        equalPartition(tableView, stream);
      }
    } else {
      const auto tableRows = tableView.num_rows();
      const auto rowsPerChunk = std::max<cudf::size_type>(
          1,
          targetRowsPerChunk_ > 0
              ? std::min<cudf::size_type>(
                    tableRows,
                    static_cast<cudf::size_type>(targetRowsPerChunk_))
              : tableRows);
      // SINGLE/gather exchanges must obey the same chunk bound as HASH and
      // RANGE.  Packing the whole input here bypassed splitAndEnqueue and let
      // a global sort/gather create tens-of-GiB host-staging transfers.
      for (cudf::size_type start = 0; start < tableRows;
           start += rowsPerChunk) {
        const auto end =
            std::min<cudf::size_type>(tableRows, start + rowsPerChunk);
        auto slicedTables = cudf::slice(tableView, {start, end});
        VELOX_CHECK_EQ(slicedTables.size(), 1);
        auto packedCols = cudf::pack(
            slicedTables[0], stream, cudf::get_current_device_resource_ref());
        stream.synchronize();
        auto packedColsPtr = std::make_unique<cudf::packed_columns>(
            std::move(packedCols.metadata), std::move(packedCols.gpu_data));
        queueManager->enqueue(
            this->taskId(),
            0,
            std::move(packedColsPtr),
            slicedTables[0].num_rows());
      }
    }

    // Check backpressure after enqueue.
    auto blocked = queueManager->checkBlocked(this->taskId(), &future_);
    if (blocked) {
      VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
              << " is blocked, can no longer write to output!";
    }
    blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                              : exec::BlockingReason::kNotBlocked;

    pendingInputs_.clear();
    pendingRows_ = 0;

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingRows_ = 0;
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  if (noMoreInput_) {
    flushPending(); // drain any remaining buffered inputs
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  // Erase first element since it's always 0 and we don't need it.
  partitionOffsets.erase(partitionOffsets.begin());
  partitionOffsets.pop_back();

  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  splitAndEnqueue(tableView, offsets, stream);
}

void UcxPartitionedOutput::splitAndEnqueue(
    cudf::table_view tableView,
    std::vector<cudf::size_type> offsets,
    rmm::cuda_stream_view stream) {
  auto contiguousTables = cudf::contiguous_split(
      tableView, offsets, stream, cudf::get_current_device_resource_ref());

  // Synchronize the stream to ensure CUDA operations complete before enqueuing.
  // UCXX/UCX is not stream-aware, so without syncing, data could be sent before
  // the GPU kernels have finished writing to the buffers.
  stream.synchronize();

  VELOX_CHECK_EQ(
      offsets.size() + 1, numPartitions_, "mismatch in numPartitions_");
  auto queueManager = sharedQueueManager();
  for (int i = 0; i < numPartitions_; ++i) {
    auto const& partitionTable = contiguousTables[i];
    const auto partitionRows = partitionTable.table.num_rows();
    if (partitionRows == 0) {
      // Skip empty partitions.
      continue;
    }

    const bool rowChunkingNeeded =
        targetRowsPerChunk_ > 0 && partitionRows > targetRowsPerChunk_;
    if (rowChunkingNeeded) {
      cudf::size_type rowsPerChunk = std::min<cudf::size_type>(
          partitionRows, static_cast<cudf::size_type>(targetRowsPerChunk_));
      VLOG(2) << "UcxPartitionedOutput chunking task=" << taskId()
              << " destination=" << i << " rows=" << partitionRows
              << " rowsPerChunk=" << rowsPerChunk
              << " targetRowsPerChunk=" << targetRowsPerChunk_;
      for (cudf::size_type start = 0; start < partitionRows;
           start += rowsPerChunk) {
        const auto end =
            std::min<cudf::size_type>(partitionRows, start + rowsPerChunk);
        auto slicedTables = cudf::slice(partitionTable.table, {start, end});
        VELOX_CHECK_EQ(slicedTables.size(), 1);
        auto packedCols = cudf::pack(slicedTables[0], stream);
        stream.synchronize();
        auto packedColsPtr = std::make_unique<cudf::packed_columns>(
            std::move(packedCols.metadata), std::move(packedCols.gpu_data));
        queueManager->enqueue(
            this->taskId(),
            i,
            std::move(packedColsPtr),
            slicedTables[0].num_rows());
      }
      continue;
    }

    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(contiguousTables[i].data.metadata),
        std::move(contiguousTables[i].data.gpu_data));

    // enqueue partition data on Ucx Output Buffer
    queueManager->enqueue(
        this->taskId(),
        i,
        std::move(packedColsPtr),
        partitionTable.table.num_rows());
  }
}

} // namespace facebook::velox::ucx_exchange
