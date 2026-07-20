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

#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

namespace test {
class CudfOrderByTestHelper;
}

class CudfOrderBy : public CudfOperatorBase {
 public:
  CudfOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::OrderByNode>& orderByNode);

  /// Returns true when all output columns can round-trip through the external
  /// Parquet runs and every sorting key has supported cuDF ordering semantics.
  static bool isSupported(
      const RowTypePtr& outputType,
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys);

  static bool isSupported(
      const std::shared_ptr<const core::OrderByNode>& orderByNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  struct MergeStats {
    uint64_t sourceChunks{0};
    uint64_t sourceRows{0};
    uint64_t sourceBytes{0};
    uint64_t outputBatches{0};
    uint64_t outputRows{0};
    uint64_t outputBytes{0};
    uint64_t maxResidentRows{0};
    uint64_t maxResidentBytes{0};
    uint64_t maxOutputBytes{0};
    uint64_t maxActiveRuns{0};
  };

  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
    // A paused reader owns at most one bounded chunk. chunkOffset identifies
    // the unconsumed suffix without copying it into a growing carry table.
    std::unique_ptr<cudf::table> chunk;
    cudf::size_type chunkOffset{0};
    uint64_t chunkBytes{0};
  };

  /// Overrides production limits for deterministic spill/merge tests.
  static void testingSetMemoryLimits(
      uint64_t sortedRunBytes,
      uint64_t mergeChunkBytes,
      uint64_t outputChunkBytes,
      cudf::size_type maxOutputRows,
      size_t mergeFanIn = 2);
  static void testingResetMemoryLimits();
  static uint64_t testingMaxActiveRuns();
  static uint64_t testingSourceChunks();
  static uint64_t testingMergeOutputBatches();
  static uint64_t testingEmittedChunks();
  static uint64_t testingSpillCleanups();

  void spillSortedRun();
  void compactSortedRunsForMerge();
  void initializeSortedRunReaders();
  void prepareSpilledOutput();
  uint64_t measureTableBytes(
      std::unique_ptr<cudf::table>& table,
      rmm::cuda_stream_view stream);
  bool loadPausedChunk(
      SortedRun& run,
      rmm::cuda_stream_view stream,
      MergeStats& stats);
  std::unique_ptr<cudf::table> mergeNextPausedBatch(
      std::vector<SortedRun*>& runs,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finished,
      MergeStats& stats);
  std::unique_ptr<cudf::table> mergeNextSortedBatch();
  void setPendingOutput(std::unique_ptr<cudf::table> output);
  CudfVectorPtr takePendingOutput();
  void cleanupSpillFiles();
  void cleanupSpillStateAfterFailure(std::string_view context) noexcept;

  std::shared_ptr<const core::OrderByNode> orderByNode_;
  // Inputs, readers, merge cursors and output slices all outlive one Operator
  // call. Keep their work and destruction ordered on one persistent stream.
  const rmm::cuda_stream_view stateStream_;
  std::vector<CudfVectorPtr> inputs_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::order> columnOrder_;
  std::vector<cudf::null_order> nullOrder_;
  const uint64_t sortedRunBytes_;
  const size_t mergeFanIn_;
  const uint64_t outputChunkBytes_;
  const cudf::size_type maxOutputRows_;
  uint64_t bufferedBytes_{0};
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> pendingOutput_;
  cudf::size_type pendingOutputOffset_{0};
  uint64_t pendingOutputBytes_{0};
  MergeStats outputMergeStats_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};

  friend class test::CudfOrderByTestHelper;
};

} // namespace facebook::velox::cudf_velox
