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
#include <limits>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
// libcudf's hash-partition exclusive scan can exceed the CUDA kernel launch
// resource limit on very large tables before memory pressure is visible. Keep
// an engine-level call-size ceiling even when the caller does not provide one;
// explicit per-query limits remain authoritative and may be smaller.
constexpr int64_t kDefaultMaxRowsPerHashPartitionCall = 128'000'000;

// Admission is cooperative and can lose a race to another output driver.
// When no reservation can be obtained, cap this driver's estimated transient
// hash-partition peak so concurrent unadmitted fallbacks remain bounded.
constexpr uint64_t kMaxUnadmittedHashPartitionPeakBytes = uint64_t{1} << 30;

bool containsStructColumn(const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::STRUCT) {
    return true;
  }
  for (cudf::size_type child = 0; child < column.num_children(); ++child) {
    if (containsStructColumn(column.child(child))) {
      return true;
    }
  }
  return false;
}

bool containsStructColumn(const cudf::table_view& table) {
  for (cudf::size_type column = 0; column < table.num_columns(); ++column) {
    if (containsStructColumn(table.column(column))) {
      return true;
    }
  }
  return false;
}

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

void normalizePartitionOffsets(
    std::vector<cudf::size_type>& offsets,
    size_t numPartitions) {
  VELOX_CHECK(
      offsets.size() == numPartitions || offsets.size() == numPartitions + 1,
      "Unexpected libcudf partition offset count {} for {} partitions",
      offsets.size(),
      numPartitions);
  VELOX_CHECK_EQ(offsets.front(), 0);
  offsets.erase(offsets.begin());
  if (offsets.size() == numPartitions) {
    offsets.pop_back();
  }
}

int64_t positiveEnvironmentOverride(const char* name) {
  if (const char* value = std::getenv(name)) {
    try {
      const auto parsed = std::stoll(value);
      if (parsed > 0) {
        return static_cast<int64_t>(parsed);
      }
    } catch (...) {
    }
  }
  return 0;
}

int64_t maxRowsPerHashPartitionCall(const core::QueryConfig& queryConfig) {
  const auto environmentRows =
      positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_INPUT_BATCH_ROWS");
  if (environmentRows > 0) {
    return environmentRows;
  }
  const auto configuredRows = queryConfig.ucxHashPartitionInputBatchRows();
  return configuredRows > 0 ? configuredRows
                            : kDefaultMaxRowsPerHashPartitionCall;
}

int64_t maxRowsPerHashPartitionWindow(const core::QueryConfig& queryConfig) {
  const auto environmentRows =
      positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_WINDOW_ROWS");
  return environmentRows > 0 ? environmentRows
                             : queryConfig.ucxHashPartitionWindowRows();
}

bool partialAggregationBehindProjects(
    std::shared_ptr<const core::PlanNode> source) {
  while (source) {
    if (const auto aggregation =
            std::dynamic_pointer_cast<const core::AggregationNode>(source)) {
      return aggregation->step() == core::AggregationNode::Step::kPartial ||
          aggregation->step() == core::AggregationNode::Step::kIntermediate;
    }
    if (!std::dynamic_pointer_cast<const core::ProjectNode>(source) ||
        source->sources().size() != 1) {
      return false;
    }
    source = source->sources().front();
  }
  return false;
}
uint64_t targetBytesPerUcxChunk(const core::QueryConfig& queryConfig) {
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_BYTES")) {
    try {
      return std::stoull(value);
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchBytes();
}

cudf::size_type rowsPerUcxChunk(
    cudf::size_type rows,
    uint64_t bytes,
    int64_t targetRows,
    uint64_t targetBytes) {
  auto rowsPerChunk = rows;
  if (targetRows > 0) {
    rowsPerChunk = std::min<cudf::size_type>(rowsPerChunk, targetRows);
  }
  if (targetBytes > 0 && bytes > targetBytes && rows > 0) {
    const auto byteLimitedRows = std::max<uint64_t>(
        1, static_cast<uint64_t>(rows) * targetBytes / bytes);
    rowsPerChunk = std::min<cudf::size_type>(
        rowsPerChunk,
        static_cast<cudf::size_type>(std::min<uint64_t>(
            byteLimitedRows, std::numeric_limits<cudf::size_type>::max())));
  }
  return std::max<cudf::size_type>(1, rowsPerChunk);
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
      sourceNeedsOwnerBoundaryBackpressure_(
          partialAggregationBehindProjects(planNode->sources().front())),
      maxOutputBufferSize_(ctx->queryConfig().maxOutputBufferSize()),
      targetRowsPerChunk_(targetRowsPerUcxChunk(ctx->queryConfig())),
      targetBytesPerChunk_(targetBytesPerUcxChunk(ctx->queryConfig())),
      hashPartitionInputBatchRows_(
          maxRowsPerHashPartitionCall(ctx->queryConfig())),
      hashPartitionWindowRows_(
          maxRowsPerHashPartitionWindow(ctx->queryConfig())) {
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
  VELOX_CHECK(!hasActiveFlush(), "addInput while a flush is still active");

  const auto inputFlatBytes = input->estimateFlatSize();
  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(inputFlatBytes, input->size());
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingFlatBytes_ += inputFlatBytes;
  pendingInputs_.push_back(std::move(cudfVector));

  if ((targetRowsPerChunk_ <= 0 && targetBytesPerChunk_ == 0) ||
      (targetRowsPerChunk_ > 0 && pendingRows_ >= targetRowsPerChunk_) ||
      (targetBytesPerChunk_ > 0 && pendingFlatBytes_ >= targetBytesPerChunk_)) {
    flushPending();
  }
}

void UcxPartitionedOutput::flushPending() {
  CudaAllocationTraceScope allocationTrace(
      fmt::format(
          "UcxPartitionedOutput task={} method=flushPending", taskId()));
  if (!hasActiveFlush() && pendingInputs_.empty()) {
    return;
  }

  try {
    if (!hasActiveFlush()) {
      preparePendingFlush();
    }
    do {
      advanceActiveFlush();
    } while (hasActiveFlush() && activeDrainBeforeBackpressure_ &&
             blockingReason_ == exec::BlockingReason::kNotBlocked);

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingRows_ = 0;
    pendingFlatBytes_ = 0;
    clearActiveFlush();
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

void UcxPartitionedOutput::preparePendingFlush() {
  VELOX_CHECK(!hasActiveFlush());
  VELOX_CHECK(!pendingInputs_.empty());

  activeInputs_ = std::move(pendingInputs_);
  activeSourceFlatBytes_ = pendingFlatBytes_;
  pendingInputs_.clear();
  pendingRows_ = 0;
  pendingFlatBytes_ = 0;

  auto stream = activeInputs_.back()->stream();
  if (activeInputs_.size() > 1) {
    std::vector<cudf::table_view> views;
    std::vector<rmm::cuda_stream_view> inputStreams;
    views.reserve(activeInputs_.size());
    inputStreams.reserve(activeInputs_.size());
    for (auto& input : activeInputs_) {
      inputStreams.push_back(input->stream());
      views.push_back(
          remap_.empty()
              ? input->getTableView()
              : input->getTableView().select(remap_.begin(), remap_.end()));
    }

    cudf::detail::join_streams(inputStreams, stream);
    activeMergedTable_ = cudf::concatenate(
        views, stream, cudf::get_current_device_resource_ref());
    orderCudfVectorDeallocationsAfterStream(
        activeInputs_, inputStreams, stream);
    // The concatenated table is now the source owner. Releasing the input
    // vectors here preserves the old 2x -> 1x peak-memory behavior.
    activeInputs_.clear();
  }

  activeStream_ = stream;
  activeNextRow_ = 0;
  const auto tableRows = activeTableView().num_rows();
  if (numPartitions_ > 1 && rangeBoundsJson_.empty() &&
      (partitionKeyIndices_.size() > 0 || spec_ == "gather") &&
      hashPartitionInputBatchRows_ > 0) {
    const auto configuredWindowRows = hashPartitionWindowRows_ > 0
        ? hashPartitionWindowRows_
        : (targetRowsPerChunk_ > 0 ? targetRowsPerChunk_
                                   : hashPartitionInputBatchRows_);
    const auto configuredRows = static_cast<uint64_t>(
        std::max<int64_t>(hashPartitionInputBatchRows_, configuredWindowRows));
    // Source residency and destination message size are different bounds.
    // A hash window is distributed across all destinations, so callers with
    // sufficient receive credit may explicitly use a larger source window
    // without increasing the per-destination chunk limit below.
    const auto configuredWindowBytes =
        positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_WINDOW_BYTES");
    const auto sourceWindowBytes = configuredWindowBytes > 0
        ? static_cast<uint64_t>(configuredWindowBytes)
        : targetBytesPerChunk_;
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        static_cast<int64_t>(std::min<uint64_t>(
            configuredRows,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))),
        sourceWindowBytes);
  } else if (numPartitions_ == 1) {
    // SINGLE/gather exchanges yield after each output-sized chunk as well.
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
  } else {
    // Bound HASH/RANGE residency by the output byte target even when no
    // explicit hash-call row limit is configured.
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
  }

  activeDrainBeforeBackpressure_ = sourceNeedsOwnerBoundaryBackpressure_ &&
      maxOutputBufferSize_ > 0 &&
      activeSourceFlatBytes_ > maxOutputBufferSize_ &&
      activeRowsPerWindow_ < tableRows;
  if (activeDrainBeforeBackpressure_) {
    VLOG(2) << "UcxPartitionedOutput will drain oversized source before "
               "backpressure task="
            << taskId() << " sourceRows=" << tableRows
            << " sourceFlatBytes=" << activeSourceFlatBytes_
            << " rowsPerWindow=" << activeRowsPerWindow_
            << " maxOutputBufferBytes=" << maxOutputBufferSize_;
  }
}

bool UcxPartitionedOutput::hasActiveFlush() const {
  return activeMergedTable_ != nullptr || !activeInputs_.empty();
}

cudf::table_view UcxPartitionedOutput::activeTableView() {
  VELOX_CHECK(hasActiveFlush());
  if (activeMergedTable_) {
    return activeMergedTable_->view();
  }
  VELOX_CHECK_EQ(activeInputs_.size(), 1);
  auto tableView = activeInputs_.front()->getTableView();
  return remap_.empty() ? tableView
                        : tableView.select(remap_.begin(), remap_.end());
}

void UcxPartitionedOutput::clearActiveFlush() {
  activeInputs_.clear();
  activeMergedTable_.reset();
  activeStream_.reset();
  activeSourceFlatBytes_ = 0;
  activeNextRow_ = 0;
  activeRowsPerWindow_ = 0;
  activeDrainBeforeBackpressure_ = false;
}

void UcxPartitionedOutput::updateBackpressure() {
  // The queue is shared by every output driver in this task. Checking after
  // each residency window bounds overshoot to at most one window per driver,
  // instead of allowing one addInput() to enqueue every remaining window.
  // P0 deliberately checks after enqueue: exact packed bytes are only known
  // then, and pre-reserving a window larger than maxSize would deadlock unless
  // the credit protocol also grew a special oversized-window grant.
  auto blocked = sharedQueueManager()->checkBlocked(this->taskId(), &future_);
  if (blocked) {
    VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
            << " is blocked after output window";
  }
  blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                            : exec::BlockingReason::kNotBlocked;
}

void UcxPartitionedOutput::advanceActiveFlush() {
  VELOX_CHECK(hasActiveFlush());
  VELOX_CHECK(activeStream_.has_value());
  VELOX_CHECK_EQ(blockingReason_, exec::BlockingReason::kNotBlocked);

  auto tableView = activeTableView();
  const auto tableRows = tableView.num_rows();
  if (activeNextRow_ >= tableRows) {
    clearActiveFlush();
    return;
  }

  auto stream = *activeStream_;
  auto rowsThisWindow = std::min<cudf::size_type>(
      activeRowsPerWindow_, tableRows - activeNextRow_);
  std::optional<cudf_velox::DeviceMemoryAdmissionReservation> memoryAdmission;

  if (numPartitions_ > 1 && rangeBoundsJson_.empty() &&
      (partitionKeyIndices_.size() > 0 || spec_ == "gather") &&
      hashPartitionInputBatchRows_ > 0 && rowsThisWindow > 0 &&
      activeSourceFlatBytes_ > 0 && tableRows > 0) {
    const auto sourceRows = static_cast<uint64_t>(tableRows);
    const auto averageRowBytes = activeSourceFlatBytes_ / sourceRows +
        static_cast<uint64_t>(activeSourceFlatBytes_ % sourceRows != 0);
    const auto candidateRows = static_cast<uint64_t>(rowsThisWindow);
    const auto estimatePeakBytes = [&](uint64_t rows) {
      const auto sourceBytes = averageRowBytes * rows;
      const auto scratchBytes = rows * uint64_t{12} +
          static_cast<uint64_t>(numPartitions_) * uint64_t{4096};
      // activeTableView() already owns the source and headroom excludes that
      // live allocation. Admission reserves only the new partitioned output,
      // hash scratch, and allocator headroom.
      const auto workingBytes = sourceBytes + scratchBytes;
      return workingBytes + workingBytes / uint64_t{4};
    };
    const auto candidatePeakBytes = estimatePeakBytes(candidateRows);
    // The unadmitted fallback below already proves that at most 1 GiB of new
    // hash workspace is safe. Avoid a global CUDA/RMM headroom snapshot for
    // windows inside that bound; sampling every normal batch serializes the
    // producer pipeline and caused a measurable exchange regression.
    if (candidatePeakBytes <= kMaxUnadmittedHashPartitionPeakBytes) {
      VLOG(2) << "UcxPartitionedOutput bounded hash fast path task=" << taskId()
              << " sourceRows=" << tableRows
              << " sourceFlatBytes=" << activeSourceFlatBytes_
              << " selectedRows=" << candidateRows
              << " estimatedPeakBytes=" << candidatePeakBytes;
    } else {
      const auto fallbackRows = maxOutputBufferSize_ == 0
          ? candidateRows
          : std::max<uint64_t>(
                1,
                std::min(
                    candidateRows, maxOutputBufferSize_ / averageRowBytes));
      const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
      const auto allocatableBytes = headroom.allocatableBytes();
      const auto reserveBytes = headroom.totalBytes == 0
          ? uint64_t{1} << 30
          : std::max<uint64_t>(
                uint64_t{1} << 30,
                static_cast<uint64_t>(headroom.totalBytes) / uint64_t{50});
      const auto admissionCapacity = allocatableBytes > reserveBytes
          ? allocatableBytes - reserveBytes
          : allocatableBytes / uint64_t{2};
      const auto alreadyReserved =
          cudf_velox::deviceMemoryAdmissionReservedBytes(headroom.device);
      const auto availableCapacity = admissionCapacity > alreadyReserved
          ? admissionCapacity - alreadyReserved
          : uint64_t{0};
      uint64_t selectedRows = candidateRows;
      if (!headroom.cudaValid || candidatePeakBytes > availableCapacity) {
        const auto pressureRows = candidatePeakBytes == 0
            ? candidateRows
            : std::max<uint64_t>(
                  1,
                  static_cast<uint64_t>(
                      static_cast<long double>(candidateRows) *
                      static_cast<long double>(availableCapacity) /
                      static_cast<long double>(candidatePeakBytes)));
        // The 1GiB queue-derived bound is no longer the normal workspace cap.
        // It is retained as the proven emergency floor when a snapshot is
        // unavailable or severely pressured; Q10 passed 10/10 at this size.
        selectedRows =
            std::min(candidateRows, std::max(fallbackRows, pressureRows));
      }

      auto selectedPeakBytes = estimatePeakBytes(selectedRows);
      memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
          headroom.device, selectedPeakBytes, admissionCapacity);
      bool admissionRetried = false;
      if (!memoryAdmission && fallbackRows < selectedRows) {
        // The initial size used a non-atomic reservation snapshot. Another
        // driver may have acquired capacity before this atomic attempt, so
        // shrink to the proven queue-sized fallback and retry the reservation.
        selectedRows = fallbackRows;
        selectedPeakBytes = estimatePeakBytes(selectedRows);
        memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
            headroom.device, selectedPeakBytes, admissionCapacity);
        admissionRetried = true;
      }
      if (!memoryAdmission &&
          selectedPeakBytes > kMaxUnadmittedHashPartitionPeakBytes) {
        // Admission can still be unavailable when every byte is temporarily
        // reserved, or when CUDA headroom could not be sampled. A driver must
        // not then execute the original large window unaccounted. Find the
        // largest row count whose estimated peak is at most the explicit 1 GiB
        // fallback bound, and give that smaller window one final admission try.
        uint64_t lower = 1;
        uint64_t upper = selectedRows;
        if (estimatePeakBytes(lower) <= kMaxUnadmittedHashPartitionPeakBytes) {
          while (lower < upper) {
            const auto middle = lower + (upper - lower + 1) / 2;
            if (estimatePeakBytes(middle) <=
                kMaxUnadmittedHashPartitionPeakBytes) {
              lower = middle;
            } else {
              upper = middle - 1;
            }
          }
        }
        selectedRows = lower;
        selectedPeakBytes = estimatePeakBytes(selectedRows);
        memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
            headroom.device, selectedPeakBytes, admissionCapacity);
        admissionRetried = true;
      }
      const bool unadmittedFallback = !memoryAdmission;
      rowsThisWindow = static_cast<cudf::size_type>(selectedRows);
      VLOG(2) << "UcxPartitionedOutput pressure-aware hash window task="
              << taskId() << " sourceRows=" << tableRows
              << " sourceFlatBytes=" << activeSourceFlatBytes_
              << " averageRowBytes=" << averageRowBytes
              << " candidateRows=" << candidateRows
              << " selectedRows=" << selectedRows
              << " estimatedPeakBytes=" << selectedPeakBytes
              << " cudaFreeBytes=" << headroom.freeBytes
              << " poolReusableBytes=" << headroom.reusablePoolBytes()
              << " admissionCapacityBytes=" << admissionCapacity
              << " alreadyReservedBytes=" << alreadyReserved
              << " admissionRetried=" << admissionRetried
              << " admitted=" << memoryAdmission.has_value()
              << " unadmittedFallback=" << unadmittedFallback;
    }
  }

  const auto end = activeNextRow_ + rowsThisWindow;
  auto slices = cudf::slice(tableView, {activeNextRow_, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);

  auto partitionInput = slices[0];
  std::unique_ptr<cudf::table> materializedPartitionInput;
  if (numPartitions_ > 1 && containsStructColumn(partitionInput)) {
    // libcudf partition requires STRUCT children to align with their sliced
    // parent. Materialize this bounded window to normalize nested offsets.
    materializedPartitionInput = std::make_unique<cudf::table>(
        partitionInput, stream, cudf::get_current_device_resource_ref());
    partitionInput = materializedPartitionInput->view();
  }

  if (numPartitions_ > 1) {
    if (!rangeBoundsJson_.empty()) {
      rangePartition(partitionInput, stream);
    } else if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
      // hashPartition() may internally split this residency window into safe
      // libcudf call-size chunks, but it cannot cross into the next window.
      hashPartition(partitionInput, stream);
    } else {
      equalPartition(partitionInput, stream);
    }
  } else {
    auto packedCols =
        cudf::pack(slices[0], stream, cudf::get_current_device_resource_ref());
    stream.synchronize();
    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(packedCols.metadata), std::move(packedCols.gpu_data));
    sharedQueueManager()->enqueue(
        this->taskId(), 0, std::move(packedColsPtr), slices[0].num_rows());
  }

  activeNextRow_ = end;
  const bool drainBeforeBackpressure = activeDrainBeforeBackpressure_;
  if (activeNextRow_ == tableRows) {
    // Every enqueue above synchronizes its stream before publication, so the
    // source owner can be released even if the queue check below blocks.
    clearActiveFlush();
  }
  if (drainBeforeBackpressure && hasActiveFlush()) {
    // Retaining an oversized source behind a queue future can starve its
    // upstream operator of the memory needed to produce the next batch. Keep
    // draining this one source; consumers may concurrently retire published
    // buffers. Backpressure is checked after the source owner is released.
    return;
  }
  updateBackpressure();
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
  // Driver calls isBlocked() before getOutput(). Keep this guard for direct
  // test drivers and to ensure an outstanding future is never overwritten.
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    return nullptr;
  }
  if (hasActiveFlush() || (noMoreInput_ && !pendingInputs_.empty())) {
    flushPending();
  }
  // A final work unit may have completed but filled the queue. Defer EOS until
  // its future has been moved by isBlocked() and resumed by the Driver.
  if (noMoreInput_ && !hasActiveFlush() && pendingInputs_.empty() &&
      blockingReason_ == exec::BlockingReason::kNotBlocked) {
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

  if (auto* rangeFunctionSpec = dynamic_cast<const RangePartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec())) {
    partitionKeyIndices_ = rangeFunctionSpec->keyChannels();
    rangeBoundsJson_ = rangeFunctionSpec->boundsJson();
    VELOX_CHECK(
        !partitionKeyIndices_.empty() && !rangeBoundsJson_.empty(),
        "RANGE_PID requires both keys and Spark boundaries");
    return;
  }

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
  const auto maxRows = hashPartitionInputBatchRows_;
  if (maxRows > 0 && tableView.num_rows() > maxRows) {
    VLOG(2) << "UcxPartitionedOutput pre-slicing hash input task=" << taskId()
            << " rows=" << tableView.num_rows()
            << " maxRowsPerCall=" << maxRows;

    struct PartitionedChunk {
      std::unique_ptr<cudf::table> table;
      std::vector<cudf::size_type> offsets;
    };
    struct PackedPartition {
      int32_t rows;
      std::unique_ptr<cudf::packed_columns> data;
    };
    auto queueManager = sharedQueueManager();

    // Bound the temporary residency of the safe hash path.  Keeping every
    // 500K-row partitioned chunk for the full input, followed by all 32
    // recombined destinations and all packed buffers, amplified a wide Q10
    // customer batch by several times before output-buffer backpressure could
    // run.  Work on one output-sized row window and one destination at a time
    // instead.  The source vector remains alive, but all hash, concatenate,
    // and pack temporaries for a window are released before the next window.
    const auto configuredWindowRows = hashPartitionWindowRows_ > 0
        ? hashPartitionWindowRows_
        : (targetRowsPerChunk_ > 0 ? targetRowsPerChunk_ : maxRows);
    const auto rowsPerWindow = static_cast<cudf::size_type>(
        std::max<int64_t>(maxRows, configuredWindowRows));

    // When the safety call size and residency window are identical, every
    // window contains exactly one partitioned table.  Use contiguous_split's
    // single bulk operation instead of packing 32 destinations one at a time.
    // Besides avoiding needless concatenate bookkeeping, this changes Q10
    // from one stream synchronization per destination to one per 8M-row
    // window while preserving the same hard peak-memory bound.  Q21 uses a
    // larger recombination window than its 500K call size and therefore keeps
    // the multi-chunk path below.
    if (rowsPerWindow == maxRows) {
      std::vector<cudf::size_type> partitionKeyIndices;
      partitionKeyIndices.reserve(partitionKeyIndices_.size());
      for (const auto& idx : partitionKeyIndices_) {
        partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
      }
      for (cudf::size_type start = 0; start < tableView.num_rows();
           start += rowsPerWindow) {
        const auto end = std::min<cudf::size_type>(
            tableView.num_rows(), start + rowsPerWindow);
        auto slices = cudf::slice(tableView, {start, end}, stream);
        VELOX_CHECK_EQ(slices.size(), 1);
        auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
            slices[0],
            partitionKeyIndices,
            numPartitions_,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream);
        VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
        VELOX_CHECK_EQ(partitionOffsets.front(), 0);
        partitionOffsets.erase(partitionOffsets.begin());
        partitionOffsets.pop_back();
        splitAndEnqueue(
            partitionedTable->view(), std::move(partitionOffsets), stream);
      }
      return;
    }

    for (cudf::size_type windowStart = 0; windowStart < tableView.num_rows();
         windowStart += rowsPerWindow) {
      const auto windowEnd = std::min<cudf::size_type>(
          tableView.num_rows(), windowStart + rowsPerWindow);
      std::vector<PartitionedChunk> chunks;
      chunks.reserve((windowEnd - windowStart + maxRows - 1) / maxRows);

      for (cudf::size_type start = windowStart; start < windowEnd;) {
        const auto end = std::min<cudf::size_type>(
            windowEnd, start + static_cast<cudf::size_type>(maxRows));
        auto slices = cudf::slice(tableView, {start, end}, stream);
        VELOX_CHECK_EQ(slices.size(), 1);

        std::vector<cudf::size_type> partitionKeyIndices;
        partitionKeyIndices.reserve(partitionKeyIndices_.size());
        for (const auto& idx : partitionKeyIndices_) {
          partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
        }
        auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
            slices[0],
            partitionKeyIndices,
            numPartitions_,
            cudf::hash_id::HASH_MURMUR3,
            cudf::DEFAULT_HASH_SEED,
            stream);
        VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
        VELOX_CHECK_EQ(partitionOffsets.front(), 0);
        chunks.push_back(
            {std::move(partitionedTable), std::move(partitionOffsets)});
        start = end;
      }

      for (int destination = 0; destination < numPartitions_; ++destination) {
        std::vector<cudf::table_view> destinationViews;
        destinationViews.reserve(chunks.size());
        for (const auto& chunk : chunks) {
          const auto begin = chunk.offsets[destination];
          const auto end = chunk.offsets[destination + 1];
          if (begin == end) {
            continue;
          }
          auto slices = cudf::slice(chunk.table->view(), {begin, end}, stream);
          VELOX_CHECK_EQ(slices.size(), 1);
          destinationViews.push_back(slices[0]);
        }
        if (destinationViews.empty()) {
          continue;
        }

        std::unique_ptr<cudf::table> combinedOwner;
        cudf::table_view destinationView;
        if (destinationViews.size() == 1) {
          destinationView = destinationViews.front();
        } else {
          combinedOwner = cudf::concatenate(
              destinationViews,
              stream,
              cudf::get_current_device_resource_ref());
          destinationView = combinedOwner->view();
        }

        std::vector<PackedPartition> packedPartitions;
        const auto rowsPerMessage = std::max<cudf::size_type>(
            1,
            targetRowsPerChunk_ > 0
                ? std::min<cudf::size_type>(
                      destinationView.num_rows(),
                      static_cast<cudf::size_type>(targetRowsPerChunk_))
                : destinationView.num_rows());
        for (cudf::size_type begin = 0; begin < destinationView.num_rows();
             begin += rowsPerMessage) {
          const auto end = std::min<cudf::size_type>(
              destinationView.num_rows(), begin + rowsPerMessage);
          auto slices = cudf::slice(destinationView, {begin, end}, stream);
          VELOX_CHECK_EQ(slices.size(), 1);
          auto packed = cudf::pack(
              slices[0], stream, cudf::get_current_device_resource_ref());
          packedPartitions.push_back(
              {slices[0].num_rows(),
               std::make_unique<cudf::packed_columns>(
                   std::move(packed.metadata), std::move(packed.gpu_data))});
        }

        // UCXX/UCX is not stream-aware.  Publish only completed buffers, then
        // release this destination's concatenate/pack temporaries immediately.
        stream.synchronize();
        for (auto& packed : packedPartitions) {
          queueManager->enqueue(
              this->taskId(), destination, std::move(packed.data), packed.rows);
        }
      }
    }
    return;
  }

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

void UcxPartitionedOutput::rangePartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(!rangeBoundsJson_.empty(), "RANGE_PID descriptor is missing");
  VELOX_CHECK(!partitionKeyIndices_.empty(), "RANGE_PID keys are missing");

  if (!rangeBoundaries_) {
    auto boundaryVector = buildRangeBoundaryVector(
        rangeBoundsJson_,
        outputType_,
        partitionKeyIndices_,
        pool(),
        rangeOrders_,
        rangeNullOrders_);
    rangeBoundaries_ = cudf_velox::with_arrow::toCudfTable(
        boundaryVector,
        pool(),
        stream,
        cudf::get_current_device_resource_ref());
    VELOX_CHECK_LT(
        rangeBoundaries_->num_rows(),
        numPartitions_,
        "RANGE_PID boundary count must be smaller than requested partitions");
  }

  std::vector<cudf::size_type> rangeKeyIndices;
  rangeKeyIndices.reserve(partitionKeyIndices_.size());
  for (const auto index : partitionKeyIndices_) {
    rangeKeyIndices.push_back(static_cast<cudf::size_type>(index));
  }
  const auto keyTable = tableView.select(rangeKeyIndices);
  auto partitionIds = cudf::lower_bound(
      rangeBoundaries_->view(),
      keyTable,
      rangeOrders_,
      rangeNullOrders_,
      stream,
      cudf::get_current_device_resource_ref());
  VELOX_CHECK(
      partitionIds->size() == tableView.num_rows(),
      "RANGE_PID must produce exactly one id per input row");
  VELOX_CHECK(
      partitionIds->type().id() == cudf::type_id::INT32,
      "RANGE_PID must produce an INT32 partition map");

  // libcudf::partition groups by the explicit INT32 map. No hash function is
  // involved; the returned table is routed directly to destination queues.
  auto [partitionedTable, partitionOffsets] = cudf::partition(
      tableView,
      partitionIds->view(),
      numPartitions_,
      stream,
      cudf::get_current_device_resource_ref());
  normalizePartitionOffsets(partitionOffsets, numPartitions_);
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

    const auto partitionBytes = partitionTable.data.gpu_data->size();
    const auto rowsPerChunk = rowsPerUcxChunk(
        partitionRows,
        partitionBytes,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
    if (rowsPerChunk < partitionRows) {
      VLOG(2) << "UcxPartitionedOutput chunking task=" << taskId()
              << " destination=" << i << " rows=" << partitionRows
              << " bytes=" << partitionBytes << " rowsPerChunk=" << rowsPerChunk
              << " targetRowsPerChunk=" << targetRowsPerChunk_
              << " targetBytesPerChunk=" << targetBytesPerChunk_;
      for (cudf::size_type start = 0; start < partitionRows;
           start += rowsPerChunk) {
        const auto end =
            std::min<cudf::size_type>(partitionRows, start + rowsPerChunk);
        auto slicedTables = cudf::slice(
            partitionTable.table,
            {static_cast<cudf::size_type>(start),
             static_cast<cudf::size_type>(end)},
            stream);
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
