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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <queue>
#include <string>

namespace facebook::velox::cudf_velox {

class CudfBatchConcat : public CudfOperatorBase {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode);

  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode,
      RowTypePtr outputType);

  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode,
      RowTypePtr outputType,
      int32_t targetRows);

  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode,
      RowTypePtr outputType,
      int32_t targetRows,
      uint64_t targetBytes);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  exec::DriverCtx* const driverCtx_;
  const std::string aggregationStep_;
  std::vector<CudfVectorPtr> buffer_;
  std::queue<CudfVectorPtr> outputQueue_;
  uint64_t totalInputRows_{0};
  uint64_t totalInputBytes_{0};
  uint64_t inputBatches_{0};
  uint64_t outputBatches_{0};
  size_t currentNumRows_{0};
  uint64_t currentNumBytes_{0};
  const size_t targetRows_{0};
  const uint64_t targetBytes_{0};
  bool summaryLogged_{false};
};

} // namespace facebook::velox::cudf_velox
