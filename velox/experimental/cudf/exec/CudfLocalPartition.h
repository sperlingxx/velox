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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/LocalPartition.h"
#include "velox/exec/Operator.h"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

enum class PartitionFunctionType {
  kHash,
  kRange,
  kRoundRobin,
  kRoundRobinRow,
};

class CudfLocalPartition : public CudfOperatorBase {
 public:
  CudfLocalPartition(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::LocalPartitionNode>& planNode);

  std::string toString() const override {
    return fmt::format("CudfLocalPartition({})", numPartitions_);
  }

  void recordOutputStats(RowVectorPtr& input);

  /// Always true but the caller will check isBlocked before adding input, hence
  /// the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  static bool shouldReplace(
      const std::shared_ptr<const core::LocalPartitionNode>& planNode);

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override {
    return nullptr;
  }

  void doNoMoreInput() override;

 private:
  void enqueuePartition(int partitionIndex, const CudfVectorPtr& cudfVector);
  void flushVectorPool();

 protected:
  const std::vector<std::shared_ptr<exec::LocalExchangeQueue>> queues_;
  const size_t numPartitions_;
  PartitionFunctionType partitionFunctionType_;
  size_t counter_{0};

  // A local keyed-FINAL repartition must not reuse the remote exchange seed.
  // If local partition count divides remote partition count (Q17: 4 vs 32),
  // identical hashing sends the peer's entire remote bucket to one local lane.
  uint32_t hashSeed_;

  std::vector<exec::BlockingReason> blockingReasons_;
  std::vector<ContinueFuture> futures_;

  std::vector<column_index_t> partitionKeyIndices_;
  std::string rangeBoundsJson_;
  std::unique_ptr<cudf::table> rangeBoundaries_;
  std::vector<cudf::order> rangeOrders_;
  std::vector<cudf::null_order> rangeNullOrders_;
};

} // namespace facebook::velox::cudf_velox
