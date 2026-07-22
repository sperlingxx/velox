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

#include "velox/core/PlanNode.h"

#include <cudf/types.hpp>

namespace facebook::velox::ucx_exchange {

/**
 * An explicit range-partition contract for GPU output and local exchange.
 *
 * The JSON payload contains Spark RangePartitioner boundaries plus ascending
 * and null-order flags. UcxPartitionedOutput and CudfLocalPartition evaluate
 * those boundaries into an INT32 partition-id column before routing. The
 * ordinary Velox CPU partition path is intentionally unsupported: using it
 * would make a RANGE exchange silently change semantics when the GPU adapter
 * is absent.
 */
class RangePartitionFunctionSpec final : public core::PartitionFunctionSpec {
 public:
  RangePartitionFunctionSpec(
      RowTypePtr inputType,
      std::vector<column_index_t> keyChannels,
      std::string boundsJson)
      : inputType_(std::move(inputType)),
        keyChannels_(std::move(keyChannels)),
        boundsJson_(std::move(boundsJson)) {
    VELOX_CHECK(!keyChannels_.empty(), "RANGE_PID requires sort-key channels");
    VELOX_CHECK(
        !boundsJson_.empty(), "RANGE_PID requires serialized boundaries");
  }

  std::unique_ptr<core::PartitionFunction> create(
      int numPartitions,
      bool localExchange) const override;

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static core::PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& obj,
      void* context);

  const RowTypePtr& inputType() const {
    return inputType_;
  }

  const std::vector<column_index_t>& keyChannels() const {
    return keyChannels_;
  }

  const std::string& boundsJson() const {
    return boundsJson_;
  }

 private:
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  const std::string boundsJson_;
};

/**
 * Materializes Spark's serialized RANGE boundaries as a Velox row vector and
 * returns the matching libcudf sort/null ordering. Shared by UCX
 * PartitionedOutput and single-task CudfLocalPartition so both paths apply
 * exactly the same boundary semantics.
 */
RowVectorPtr buildRangeBoundaryVector(
    const std::string& boundsJson,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    memory::MemoryPool* pool,
    std::vector<cudf::order>& orders,
    std::vector<cudf::null_order>& nullOrders);

} // namespace facebook::velox::ucx_exchange
