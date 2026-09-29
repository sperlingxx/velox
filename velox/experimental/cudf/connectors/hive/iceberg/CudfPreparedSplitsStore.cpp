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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfPreparedSplitsStore.h"

#include <algorithm>
#include <tuple>
#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {
namespace {

uint64_t selectedBytes(const CudfPreparedIcebergFile& file) {
  return file.preparedBytes == 0 ? file.fileSize : file.preparedBytes;
}

} // namespace

CudfPreparedSplitSource::CudfPreparedSplitSource(
    std::weak_ptr<exec::Task> task,
    core::PlanNodeId planNodeId,
    folly::Executor* executor,
    std::shared_ptr<HostRangeAdmission> admission,
    CudfPreparationMode mode)
    : task_(std::move(task)),
      planNodeId_(std::move(planNodeId)),
      executor_(executor),
      admission_(std::move(admission)),
      mode_(mode) {
  VELOX_CHECK_NOT_NULL(executor_);
}

bool CudfPreparedSplitSource::addDescriptor(
    uint64_t descriptorOrdinal,
    std::shared_ptr<velox_iceberg::HiveIcebergSplit> descriptor) {
  auto task = task_.lock();
  VELOX_CHECK_NOT_NULL(task);
  VELOX_CHECK_NOT_NULL(
      admission_, "Descriptor preparation requires host admission");
  auto request = std::make_shared<CudfFilePreparationSplit>(
      descriptorOrdinal,
      std::move(descriptor),
      admission_,
      cancellationSource_.getToken(),
      mode_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!descriptorsSealed_, "Cannot add a descriptor after sealing");
    VELOX_CHECK(!finished_);
    if (stopped_) {
      return false;
    }
    pending_.push_back(std::move(request));
    originalInputs_.fetch_add(1, std::memory_order_relaxed);
  }
  notifyTask();
  return true;
}

void CudfPreparedSplitSource::sealDescriptors() {
  bool closeNow = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || descriptorsSealed_) {
      return;
    }
    descriptorsSealed_ = true;
    finished_ = pending_.empty() && inFlight_.empty();
    closeNow = finished_ && ready_.empty();
  }
  if (closeNow) {
    scheduleClosure();
  } else {
    notifyTask();
  }
}

bool CudfPreparedSplitSource::publishPrepared(CudfPreparedIcebergFile file) {
  const auto payloadBytes = selectedBytes(file);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!finished_, "Cannot publish after source completion");
    if (stopped_) {
      return false;
    }
    VELOX_CHECK_NOT_NULL(file.descriptor);
    VELOX_CHECK_NOT_NULL(file.dataSource);
    ready_.emplace_back(std::move(file));
    preparedCount_.fetch_add(1, std::memory_order_relaxed);
    preparedPayloadBytes_.fetch_add(payloadBytes, std::memory_order_relaxed);
  }
  notifyTask();
  return true;
}

bool CudfPreparedSplitSource::publishFallback(
    std::shared_ptr<::facebook::velox::connector::ConnectorSplit> split,
    CudfPreparationFallbackReason reason) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!finished_, "Cannot publish after source completion");
    if (stopped_) {
      return false;
    }
    VELOX_CHECK_NOT_NULL(split);
    ready_.emplace_back(std::move(split));
    originalInputs_.fetch_add(1, std::memory_order_relaxed);
    fallbackCount_.fetch_add(1, std::memory_order_relaxed);
    switch (reason) {
      case CudfPreparationFallbackReason::kIneligible:
        fallbackIneligible_.fetch_add(1, std::memory_order_relaxed);
        break;
      case CudfPreparationFallbackReason::kPreparationUnavailable:
        fallbackPreparationUnavailable_.fetch_add(1, std::memory_order_relaxed);
        break;
      case CudfPreparationFallbackReason::kPreloadDisabled:
        fallbackPreloadDisabled_.fetch_add(1, std::memory_order_relaxed);
        break;
    }
  }
  notifyTask();
  return true;
}

void CudfPreparedSplitSource::finish() {
  bool closeNow = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || finished_) {
      return;
    }
    finished_ = true;
    closeNow = ready_.empty();
  }
  if (closeNow) {
    scheduleClosure();
  } else {
    notifyTask();
  }
}

void CudfPreparedSplitSource::fail(std::exception_ptr error) {
  VELOX_CHECK_NOT_NULL(error);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || error_) {
      return;
    }
    error_ = std::move(error);
    preparationFailures_.fetch_add(1, std::memory_order_relaxed);
  }
  notifyTask();
}

void CudfPreparedSplitSource::stop() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
      return;
    }
    stopped_ = true;
    pending_.clear();
    inFlight_.clear();
    ready_.clear();
  }
  cancellationSource_.requestCancellation();
  logSummary("stopped");
}

CudfPreparedSplitSource::Metrics CudfPreparedSplitSource::metrics() const {
  return {
      originalInputs_.load(std::memory_order_relaxed),
      plannedFiles_.load(std::memory_order_relaxed),
      totalRowGroups_.load(std::memory_order_relaxed),
      selectedRowGroups_.load(std::memory_order_relaxed),
      prunedRowGroups_.load(std::memory_order_relaxed),
      preparedCount_.load(std::memory_order_relaxed),
      preparedPayloadBytes_.load(std::memory_order_relaxed),
      issuedSplits_.load(std::memory_order_relaxed),
      preparedBatches_.load(std::memory_order_relaxed),
      issuedPreparedUnits_.load(std::memory_order_relaxed),
      issuedPreparedSources_.load(std::memory_order_relaxed),
      fallbackIneligible_.load(std::memory_order_relaxed),
      fallbackPreparationUnavailable_.load(std::memory_order_relaxed),
      fallbackPreloadDisabled_.load(std::memory_order_relaxed),
      preparationFailures_.load(std::memory_order_relaxed)};
}

void CudfPreparedSplitSource::logSummary(std::string_view terminal) {
  bool expected = false;
  if (!summaryLogged_.compare_exchange_strong(
          expected, true, std::memory_order_relaxed)) {
    return;
  }
  const auto values = metrics();
  const auto admission = admission_ == nullptr ? HostRangeAdmission::Snapshot{}
                                               : admission_->snapshot();
  LOG(INFO) << "cuDF ready scan summary planNode=" << planNodeId_ << " mode="
            << (mode_ == CudfPreparationMode::kRowGroupHybrid
                    ? "row_group_hybrid"
                    : "file_regular")
            << " terminal=" << terminal
            << " originalInputs=" << values.originalInputs
            << " plannedFiles=" << values.plannedFiles
            << " totalRowGroups=" << values.totalRowGroups
            << " selectedRowGroups=" << values.selectedRowGroups
            << " prunedRowGroups=" << values.prunedRowGroups
            << " preparedUnits=" << values.preparedUnits
            << " preparedPayloadBytes=" << values.preparedPayloadBytes
            << " issuedSplits=" << values.issuedSplits
            << " preparedBatches=" << values.preparedBatches
            << " issuedPreparedUnits=" << values.issuedPreparedUnits
            << " issuedPreparedSources=" << values.issuedPreparedSources
            << " fallbackIneligible=" << values.fallbackIneligible
            << " fallbackPreparationUnavailable="
            << values.fallbackPreparationUnavailable
            << " fallbackPreloadDisabled=" << values.fallbackPreloadDisabled
            << " preparationFailures=" << values.preparationFailures
            << " hostReservedPeakBytes=" << admission.peakBytes
            << " hostPeakUnits=" << admission.peakUnits
            << " unexpectedBackingReads=0";
}

void CudfPreparedSplitSource::notifyTask() {
  if (auto task = task_.lock()) {
    task->splitPreloadFinished(exec::kUngroupedGroupId, planNodeId_);
  }
}

void CudfPreparedSplitSource::scheduleClosure() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_ || closureScheduled_) {
      return;
    }
    closureScheduled_ = true;
  }
  logSummary("exhausted");
  executor_->add([task = task_, planNodeId = planNodeId_]() {
    if (auto strongTask = task.lock()) {
      if (strongTask->state() == exec::TaskState::kRunning) {
        strongTask->noMoreSplits(planNodeId);
      }
    }
  });
}

CudfPreparedSplitsStore::CudfPreparedSplitsStore(
    std::shared_ptr<CudfPreparedSplitSource> source,
    uint32_t maxFiles,
    uint64_t targetBytes)
    : SplitsStore(false),
      source_(std::move(source)),
      maxFiles_(maxFiles),
      targetBytes_(targetBytes) {
  VELOX_CHECK_NOT_NULL(source_);
  VELOX_CHECK_GT(maxFiles_, 0);
  VELOX_CHECK_GT(targetBytes_, 0);
}

int CudfPreparedSplitsStore::preloadSplits(
    int maxPreloadSplits,
    const exec::ConnectorSplitPreloadFunc& preload) {
  std::lock_guard<std::mutex> lock(source_->mutex_);
  if (source_->stopped_) {
    return 0;
  }
  harvestCompletedLocked();

  if (maxPreloadSplits <= 0 || !preload) {
    // A generated source cannot wait forever for a disabled TableScan preload
    // facility. Preserve correctness by issuing the original singleton
    // descriptors through this store.
    while (!source_->pending_.empty()) {
      auto request = std::move(source_->pending_.front());
      source_->pending_.pop_front();
      source_->ready_.emplace_back(request->descriptor());
      source_->fallbackCount_.fetch_add(1, std::memory_order_relaxed);
      source_->fallbackPreloadDisabled_.fetch_add(1, std::memory_order_relaxed);
    }
  } else {
    VELOX_CHECK_NOT_NULL(preloadingSplits_);
    while (!source_->pending_.empty() &&
           source_->inFlight_.size() < static_cast<size_t>(maxPreloadSplits)) {
      auto request = std::move(source_->pending_.front());
      source_->pending_.pop_front();
      preload(request);
      preloadingSplits_->insert(request);
      source_->inFlight_.push_back(std::move(request));
    }
  }

  if (source_->descriptorsSealed_ && source_->pending_.empty() &&
      source_->inFlight_.empty()) {
    source_->finished_ = true;
  }
  return maxPreloadSplits;
}

void CudfPreparedSplitsStore::harvestCompletedLocked() {
  for (auto it = source_->inFlight_.begin(); it != source_->inFlight_.end();) {
    const auto& request = *it;
    if (request->dataSource == nullptr || !request->dataSource->hasValue()) {
      ++it;
      continue;
    }
    std::unique_ptr<::facebook::velox::connector::DataSource>
        preparationDataSource;
    try {
      preparationDataSource = request->dataSource->move();
    } catch (...) {
      if (!source_->error_) {
        source_->error_ = std::current_exception();
        source_->preparationFailures_.fetch_add(1, std::memory_order_relaxed);
      }
      preloadingSplits_->erase(request);
      it = source_->inFlight_.erase(it);
      continue;
    }
    VELOX_CHECK_NOT_NULL(
        preparationDataSource,
        "File preparation completed without a DataSource result");
    preloadingSplits_->erase(request);
    if (request->isRowGroupPlanRequest()) {
      auto plan = request->result()->takePlan();
      if (!plan.has_value()) {
        source_->ready_.emplace_back(request->descriptor());
        source_->fallbackCount_.fetch_add(1, std::memory_order_relaxed);
        source_->fallbackPreparationUnavailable_.fetch_add(
            1, std::memory_order_relaxed);
      } else {
        source_->plannedFiles_.fetch_add(1, std::memory_order_relaxed);
        source_->totalRowGroups_.fetch_add(
            plan->totalRowGroups, std::memory_order_relaxed);
        source_->selectedRowGroups_.fetch_add(
            plan->rowGroups.size(), std::memory_order_relaxed);
        source_->prunedRowGroups_.fetch_add(
            plan->totalRowGroups - plan->rowGroups.size(),
            std::memory_order_relaxed);
        auto metadata = std::make_shared<const cudf::io::parquet::FileMetaData>(
            std::move(plan->metadata));
        for (auto& rowGroup : plan->rowGroups) {
          source_->pending_.push_back(
              std::make_shared<CudfFilePreparationSplit>(
                  request->descriptorOrdinal(),
                  request->descriptor(),
                  source_->admission_,
                  source_->cancellationSource_.getToken(),
                  CudfPreparationMode::kRowGroupHybrid,
                  metadata,
                  std::move(rowGroup)));
        }
      }
      it = source_->inFlight_.erase(it);
      continue;
    }
    auto prepared = request->result()->takeFile();
    if (prepared.has_value()) {
      const auto payloadBytes = selectedBytes(*prepared);
      source_->ready_.emplace_back(std::move(*prepared));
      source_->preparedCount_.fetch_add(1, std::memory_order_relaxed);
      source_->preparedPayloadBytes_.fetch_add(
          payloadBytes, std::memory_order_relaxed);
    } else {
      source_->ready_.emplace_back(request->descriptor());
      source_->fallbackCount_.fetch_add(1, std::memory_order_relaxed);
      source_->fallbackPreparationUnavailable_.fetch_add(
          1, std::memory_order_relaxed);
    }
    it = source_->inFlight_.erase(it);
  }
}

void CudfPreparedSplitsStore::requestBarrier(
    uint32_t numDrivers,
    std::vector<ContinuePromise>& promises) {
  addSplit(exec::Split::createBarrier(numDrivers), promises);
}

bool CudfPreparedSplitsStore::nextSplit(
    std::optional<uint32_t> driverId,
    int maxPreloadSplits,
    const exec::ConnectorSplitPreloadFunc& preload,
    exec::Split& split,
    ContinueFuture& future) {
  preloadSplits(maxPreloadSplits, preload);
  bool scheduleClosure = false;
  {
    std::lock_guard<std::mutex> lock(source_->mutex_);
    if (source_->error_) {
      std::rethrow_exception(source_->error_);
    }
    if (source_->stopped_) {
      return true;
    }
    if (!source_->ready_.empty()) {
      auto first = std::move(source_->ready_.front());
      source_->ready_.pop_front();
      if (auto* fallback = std::get_if<
              std::shared_ptr<::facebook::velox::connector::ConnectorSplit>>(
              &first)) {
        split = exec::Split(std::move(*fallback));
      } else {
        std::vector<CudfPreparedIcebergFile> files;
        files.reserve(source_->ready_.size() + 1);
        files.push_back(std::move(std::get<CudfPreparedIcebergFile>(first)));
        uint64_t batchBytes = selectedBytes(files.front());
        std::unordered_set<uint64_t> descriptorOrdinals{
            files.front().descriptorOrdinal};
        for (auto it = source_->ready_.begin(); it != source_->ready_.end();) {
          auto* candidate = std::get_if<CudfPreparedIcebergFile>(&*it);
          const bool newSource = candidate != nullptr &&
              !descriptorOrdinals.contains(candidate->descriptorOrdinal);
          if (candidate == nullptr ||
              (newSource && descriptorOrdinals.size() >= maxFiles_) ||
              !compatible(files.front(), *candidate) ||
              selectedBytes(*candidate) >
                  targetBytes_ - std::min(targetBytes_, batchBytes)) {
            ++it;
            continue;
          }
          batchBytes += selectedBytes(*candidate);
          descriptorOrdinals.insert(candidate->descriptorOrdinal);
          files.push_back(std::move(*candidate));
          it = source_->ready_.erase(it);
        }
        std::sort(
            files.begin(),
            files.end(),
            [](const auto& left, const auto& right) {
              return std::tie(left.descriptorOrdinal, left.rowGroup) <
                  std::tie(right.descriptorOrdinal, right.rowGroup);
            });
        source_->preparedBatches_.fetch_add(1, std::memory_order_relaxed);
        source_->issuedPreparedUnits_.fetch_add(
            files.size(), std::memory_order_relaxed);
        source_->issuedPreparedSources_.fetch_add(
            descriptorOrdinals.size(), std::memory_order_relaxed);
        split = exec::Split(
            std::make_shared<CudfPreparedIcebergSplit>(std::move(files)));
      }
      accountIssuedSplit(split);
      source_->issuedSplits_.fetch_add(1, std::memory_order_relaxed);
      scheduleClosure = source_->finished_ && source_->ready_.empty();
    } else if (source_->finished_) {
      scheduleClosure = true;
    } else {
      future = makeFuture();
      return false;
    }
  }

  if (split.hasConnectorSplit()) {
    if (scheduleClosure) {
      source_->scheduleClosure();
    }
    return true;
  }
  if (tryGetBarrier(driverId, split)) {
    if (scheduleClosure) {
      source_->scheduleClosure();
    }
    return true;
  }
  if (scheduleClosure) {
    source_->scheduleClosure();
  }
  if (noMoreSplits_) {
    return true;
  }
  future = makeFuture();
  return false;
}

bool CudfPreparedSplitsStore::allSplitsConsumed() const {
  std::lock_guard<std::mutex> lock(source_->mutex_);
  return noMoreSplits_ && source_->ready_.empty();
}

void CudfPreparedSplitsStore::cancelPendingWork() noexcept {
  source_->stop();
}

bool CudfPreparedSplitsStore::compatible(
    const CudfPreparedIcebergFile& left,
    const CudfPreparedIcebergFile& right) {
  return left.descriptor->connectorId == right.descriptor->connectorId &&
      left.descriptor->fileFormat == right.descriptor->fileFormat &&
      left.rowGroup.has_value() == right.rowGroup.has_value() &&
      left.descriptor->partitionKeys == right.descriptor->partitionKeys &&
      left.descriptor->infoColumns == right.descriptor->infoColumns &&
      ((left.metadata == nullptr && right.metadata == nullptr) ||
       (left.metadata != nullptr && right.metadata != nullptr &&
        left.metadata->schema == right.metadata->schema)) &&
      left.descriptor->deleteFiles.empty() &&
      right.descriptor->deleteFiles.empty();
}

void CudfPreparedSplitsStore::accountIssuedSplit(const exec::Split& split) {
  VELOX_CHECK_NOT_NULL(taskStats_);
  VELOX_CHECK(split.hasConnectorSplit());
  ++taskStats_->numTotalSplits;
  ++taskStats_->numRunningSplits;
  ++taskStats_->numRunningTableScanSplits;
  taskStats_->runningTableScanSplitWeights += split.connectorSplit->splitWeight;
  taskStats_->lastSplitStartTimeMs = getCurrentTimeMs();
  if (taskStats_->firstSplitStartTimeMs == 0) {
    taskStats_->firstSplitStartTimeMs = taskStats_->lastSplitStartTimeMs;
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
