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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfPreparedIcebergSplit.h"

#include "velox/exec/Task.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <string_view>
#include <variant>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

class CudfPreparedSplitsStore;

enum class CudfPreparationFallbackReason {
  kIneligible,
  kPreparationUnavailable,
  kPreloadDisabled,
};

/// Cross-thread publication state for one Task scan node. Producers publish
/// only complete file inputs or original fallback splits.
class CudfPreparedSplitSource final {
 public:
  struct Metrics {
    uint64_t originalInputs{0};
    uint64_t plannedFiles{0};
    uint64_t totalRowGroups{0};
    uint64_t selectedRowGroups{0};
    uint64_t prunedRowGroups{0};
    uint64_t preparedUnits{0};
    uint64_t preparedPayloadBytes{0};
    uint64_t issuedSplits{0};
    uint64_t preparedBatches{0};
    uint64_t issuedPreparedUnits{0};
    uint64_t issuedPreparedSources{0};
    uint64_t fallbackIneligible{0};
    uint64_t fallbackPreparationUnavailable{0};
    uint64_t fallbackPreloadDisabled{0};
    uint64_t preparationFailures{0};
  };

  CudfPreparedSplitSource(
      std::weak_ptr<exec::Task> task,
      core::PlanNodeId planNodeId,
      folly::Executor* executor,
      std::shared_ptr<HostRangeAdmission> admission = nullptr,
      CudfPreparationMode mode = CudfPreparationMode::kFileRegular);

  bool addDescriptor(
      uint64_t descriptorOrdinal,
      std::shared_ptr<velox_iceberg::HiveIcebergSplit> descriptor);

  void sealDescriptors();

  bool publishPrepared(CudfPreparedIcebergFile file);

  bool publishFallback(
      std::shared_ptr<::facebook::velox::connector::ConnectorSplit> split,
      CudfPreparationFallbackReason reason =
          CudfPreparationFallbackReason::kIneligible);

  void finish();

  void fail(std::exception_ptr error);

  void stop() noexcept;

  uint64_t preparedCount() const {
    return preparedCount_.load(std::memory_order_relaxed);
  }

  uint64_t fallbackCount() const {
    return fallbackCount_.load(std::memory_order_relaxed);
  }

  Metrics metrics() const;

 private:
  friend class CudfPreparedSplitsStore;

  using ReadyItem = std::variant<
      CudfPreparedIcebergFile,
      std::shared_ptr<::facebook::velox::connector::ConnectorSplit>>;

  void notifyTask();
  void scheduleClosure();
  void logSummary(std::string_view terminal);

  std::weak_ptr<exec::Task> task_;
  const core::PlanNodeId planNodeId_;
  folly::Executor* const executor_;
  const std::shared_ptr<HostRangeAdmission> admission_;
  const CudfPreparationMode mode_;
  folly::CancellationSource cancellationSource_;
  std::mutex mutex_;
  std::deque<std::shared_ptr<CudfFilePreparationSplit>> pending_;
  std::deque<std::shared_ptr<CudfFilePreparationSplit>> inFlight_;
  std::deque<ReadyItem> ready_;
  std::exception_ptr error_;
  bool descriptorsSealed_{false};
  bool finished_{false};
  bool stopped_{false};
  bool closureScheduled_{false};
  std::atomic_uint64_t preparedCount_{0};
  std::atomic_uint64_t fallbackCount_{0};
  std::atomic_uint64_t originalInputs_{0};
  std::atomic_uint64_t plannedFiles_{0};
  std::atomic_uint64_t totalRowGroups_{0};
  std::atomic_uint64_t selectedRowGroups_{0};
  std::atomic_uint64_t prunedRowGroups_{0};
  std::atomic_uint64_t preparedPayloadBytes_{0};
  std::atomic_uint64_t issuedSplits_{0};
  std::atomic_uint64_t preparedBatches_{0};
  std::atomic_uint64_t issuedPreparedUnits_{0};
  std::atomic_uint64_t issuedPreparedSources_{0};
  std::atomic_uint64_t fallbackIneligible_{0};
  std::atomic_uint64_t fallbackPreparationUnavailable_{0};
  std::atomic_uint64_t fallbackPreloadDisabled_{0};
  std::atomic_uint64_t preparationFailures_{0};
  std::atomic_bool summaryLogged_{false};
};

/// Task-facing selector for ready, compatible complete files. nextSplit() is
/// called under the Task mutex and performs no I/O or reader construction.
class CudfPreparedSplitsStore final : public exec::SplitsStore {
 public:
  CudfPreparedSplitsStore(
      std::shared_ptr<CudfPreparedSplitSource> source,
      uint32_t maxFiles,
      uint64_t targetBytes);

  void requestBarrier(
      uint32_t numDrivers,
      std::vector<ContinuePromise>& promises) override;

  bool nextSplit(
      std::optional<uint32_t> driverId,
      int maxPreloadSplits,
      const exec::ConnectorSplitPreloadFunc& preload,
      exec::Split& split,
      ContinueFuture& future) override;

  bool allSplitsConsumed() const override;

  int preloadSplits(
      int maxPreloadSplits,
      const exec::ConnectorSplitPreloadFunc& preload) override;

  bool generatesSplits() const override {
    return true;
  }

  void cancelPendingWork() noexcept override;

 private:
  static bool compatible(
      const CudfPreparedIcebergFile& left,
      const CudfPreparedIcebergFile& right);

  void accountIssuedSplit(const exec::Split& split);

  void harvestCompletedLocked();

  const std::shared_ptr<CudfPreparedSplitSource> source_;
  const uint32_t maxFiles_;
  const uint64_t targetBytes_;
  bool preloadFallbackNext_{true};
  bool preferPrepared_{true};
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
