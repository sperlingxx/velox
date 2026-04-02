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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class ScopedCudfBatchConfig {
 public:
  ScopedCudfBatchConfig(int32_t rows, int64_t bytes)
      : config_(cudf_velox::CudfConfig::getInstance()),
        savedRows_(config_.gpuTargetBatchRows),
        savedBytes_(config_.gpuTargetBatchBytes) {
    config_.gpuTargetBatchRows = rows;
    config_.gpuTargetBatchBytes = bytes;
  }

  ~ScopedCudfBatchConfig() {
    config_.gpuTargetBatchRows = savedRows_;
    config_.gpuTargetBatchBytes = savedBytes_;
  }

 private:
  cudf_velox::CudfConfig& config_;
  int32_t savedRows_;
  int64_t savedBytes_;
};

class CudfWindowTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    window::prestosql::registerAllWindowFunctions();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  bool wasCudfWindowUsed(const std::shared_ptr<Task>& task) const {
    for (const auto& pipelineStats : task->taskStats().pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfWindow") {
          return true;
        }
      }
    }
    return false;
  }

  uint64_t cudfWindowOutputVectors(const std::shared_ptr<Task>& task) const {
    uint64_t outputVectors = 0;
    for (const auto& pipelineStats : task->taskStats().pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfWindow") {
          outputVectors += operatorStats.outputVectors;
        }
      }
    }
    return outputVectors;
  }

  RowVectorPtr copyResultsWithSingleRowGpuBatches(
      const core::PlanNodePtr& plan,
      std::shared_ptr<Task>& task) {
    return copyResultsWithGpuBatchSizeRows(plan, 1, task);
  }

  RowVectorPtr copyResultsWithGpuBatchSizeRows(
      const core::PlanNodePtr& plan,
      int32_t gpuBatchSizeRows,
      std::shared_ptr<Task>& task) {
    return AssertQueryBuilder(duckDbQueryRunner_)
        .plan(plan)
        .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, gpuBatchSizeRows)
        .copyResults(pool(), task);
  }

  std::vector<RowVectorPtr> copyResultBatchesWithSingleRowGpuBatches(
      const core::PlanNodePtr& plan) {
    return copyResultBatchesWithGpuBatchSizeRows(plan, 1);
  }

  std::vector<RowVectorPtr> copyResultBatchesWithGpuBatchSizeRows(
      const core::PlanNodePtr& plan,
      int32_t gpuBatchSizeRows) {
    return AssertQueryBuilder(duckDbQueryRunner_)
        .plan(plan)
        .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, gpuBatchSizeRows)
        .copyResultBatches(pool());
  }

  void assertBatchSizes(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<vector_size_t>& expectedSizes) const {
    ASSERT_EQ(expectedSizes.size(), batches.size());
    for (auto i = 0; i < expectedSizes.size(); ++i) {
      EXPECT_EQ(expectedSizes[i], batches[i]->size());
    }
  }

  RowVectorPtr makeExpectedWindowResult(
      std::vector<int32_t> partitionKeys,
      std::vector<int32_t> sortingKeys,
      std::vector<int64_t> values,
      std::vector<int64_t> rowNumbers) {
    return makeRowVector(
        {"p", "s", "v", "rn"},
        {makeFlatVector<int32_t>(std::move(partitionKeys)),
         makeFlatVector<int32_t>(std::move(sortingKeys)),
         makeFlatVector<int64_t>(std::move(values)),
         makeFlatVector<int64_t>(std::move(rowNumbers))});
  }

  RowVectorPtr makeExpectedRunningSumResult(
      std::vector<int32_t> partitionKeys,
      std::vector<int32_t> sortingKeys,
      std::vector<int64_t> values,
      std::vector<int64_t> runningSums) {
    return makeRowVector(
        {"p", "s", "v", "sum_v"},
        {makeFlatVector<int32_t>(std::move(partitionKeys)),
         makeFlatVector<int32_t>(std::move(sortingKeys)),
         makeFlatVector<int64_t>(std::move(values)),
         makeFlatVector<int64_t>(std::move(runningSums))});
  }
};

TEST_F(CudfWindowTest, sortedStreamingCoalescesCompletePartitions) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 5, /*bytes*/ 0);

  // Small adjacent partitions should be coalesced into one output batch while
  // preserving full partition boundaries.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1, 1}),
           makeFlatVector<int32_t>({1, 2, 3, 4}),
           makeFlatVector<int64_t>({10, 20, 30, 40})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 2}),
           makeFlatVector<int32_t>({5, 6, 1}),
           makeFlatVector<int64_t>({50, 60, 70})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 3, 3, 4}),
           makeFlatVector<int32_t>({2, 1, 2, 1}),
           makeFlatVector<int64_t>({80, 90, 100, 110})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithSingleRowGpuBatches(plan, task);
  auto expected = makeExpectedWindowResult(
      {1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4},
      {1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 1},
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110},
      {1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 1});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches = copyResultBatchesWithSingleRowGpuBatches(plan);
  assertBatchSizes(resultBatches, {6, 5});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(CudfWindowTest, sortedStreamingCachesTrailingPartition) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 4, /*bytes*/ 0);

  // When the threshold is crossed in the middle of a partition, that trailing
  // partition must stay cached for the next output batch.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1}),
           makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int64_t>({10, 20})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int32_t>({3, 1}),
           makeFlatVector<int64_t>({30, 40})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 3}),
           makeFlatVector<int32_t>({2, 1}),
           makeFlatVector<int64_t>({50, 60})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithSingleRowGpuBatches(plan, task);
  auto expected = makeExpectedWindowResult(
      {1, 1, 1, 2, 2, 3},
      {1, 2, 3, 1, 2, 1},
      {10, 20, 30, 40, 50, 60},
      {1, 2, 3, 1, 2, 1});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches = copyResultBatchesWithSingleRowGpuBatches(plan);
  assertBatchSizes(resultBatches, {3, 3});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(CudfWindowTest, sortedStreamingKeepsOversizedPartitionIntact) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 4, /*bytes*/ 0);

  // A single partition may exceed the target batch size, but it still has to
  // be emitted intact rather than split across outputs.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1}),
           makeFlatVector<int32_t>({1, 2, 3}),
           makeFlatVector<int64_t>({10, 20, 30})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1}),
           makeFlatVector<int32_t>({4, 5, 6}),
           makeFlatVector<int64_t>({40, 50, 60})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 2}),
           makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int64_t>({70, 80})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithSingleRowGpuBatches(plan, task);
  auto expected = makeExpectedWindowResult(
      {1, 1, 1, 1, 1, 1, 2, 2},
      {1, 2, 3, 4, 5, 6, 1, 2},
      {10, 20, 30, 40, 50, 60, 70, 80},
      {1, 2, 3, 4, 5, 6, 1, 2});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches = copyResultBatchesWithSingleRowGpuBatches(plan);
  assertBatchSizes(resultBatches, {6, 2});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(
    CudfWindowTest,
    sortedStreamingWaitsForPartitionBoundaryAcrossBatches) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 3, /*bytes*/ 0);

  // If the first oversized cached batch contains only one partition, the
  // partition-aware path must keep pulling more input until that partition
  // ends, even though the nominal row target was already exceeded.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1, 1}),
           makeFlatVector<int32_t>({1, 2, 3, 4}),
           makeFlatVector<int64_t>({10, 20, 30, 40})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1, 1}),
           makeFlatVector<int32_t>({5, 6, 7, 8}),
           makeFlatVector<int64_t>({50, 60, 70, 80})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 2}),
           makeFlatVector<int32_t>({1, 2}),
           makeFlatVector<int64_t>({90, 100})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithGpuBatchSizeRows(plan, /*gpuBatchSizeRows*/ 4, task);
  auto expected = makeExpectedWindowResult(
      {1, 1, 1, 1, 1, 1, 1, 1, 2, 2},
      {1, 2, 3, 4, 5, 6, 7, 8, 1, 2},
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
      {1, 2, 3, 4, 5, 6, 7, 8, 1, 2});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches =
      copyResultBatchesWithGpuBatchSizeRows(plan, /*gpuBatchSizeRows*/ 4);
  assertBatchSizes(resultBatches, {8, 2});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(
    CudfWindowTest,
    sortedStreamingSplitsSingleCachedBatchAtInternalPartitionBoundary) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 3, /*bytes*/ 0);

  // If one oversized cached batch already contains multiple partitions, emit
  // the flushable prefix immediately and cache only the trailing partition.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
           makeFlatVector<int32_t>({1, 2, 1, 2, 1, 2}),
           makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithGpuBatchSizeRows(plan, /*gpuBatchSizeRows*/ 4, task);
  auto expected = makeExpectedWindowResult(
      {1, 1, 2, 2, 3, 3},
      {1, 2, 1, 2, 1, 2},
      {10, 20, 30, 40, 50, 60},
      {1, 2, 1, 2, 1, 2});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches =
      copyResultBatchesWithGpuBatchSizeRows(plan, /*gpuBatchSizeRows*/ 4);
  assertBatchSizes(resultBatches, {4, 2});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(CudfWindowTest, sortedStreamingRunningSumUsesAggregatePath) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 5, /*bytes*/ 0);

  // Aggregate windows should keep the same partition-aware flush behavior while
  // producing correct running sums.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 1, 1}),
           makeFlatVector<int32_t>({1, 2, 3, 4}),
           makeFlatVector<int64_t>({10, 20, 30, 40})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 1, 2}),
           makeFlatVector<int32_t>({5, 6, 1}),
           makeFlatVector<int64_t>({50, 60, 70})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 3, 3, 4}),
           makeFlatVector<int32_t>({2, 1, 2, 1}),
           makeFlatVector<int64_t>({80, 90, 100, 110})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .streamingWindow(
                      {"sum(v) over (partition by p order by s rows between "
                       "unbounded preceding and current row) as sum_v"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithSingleRowGpuBatches(plan, task);
  auto expected = makeExpectedRunningSumResult(
      {1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4},
      {1, 2, 3, 4, 5, 6, 1, 2, 1, 2, 1},
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110},
      {10, 30, 60, 100, 150, 210, 70, 150, 90, 190, 110});
  facebook::velox::test::assertEqualVectors(expected, result);

  auto resultBatches = copyResultBatchesWithSingleRowGpuBatches(plan);
  assertBatchSizes(resultBatches, {6, 5});

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(2, cudfWindowOutputVectors(task));
}

TEST_F(CudfWindowTest, unsortedInputKeepsBlockingBehavior) {
  ScopedCudfBatchConfig scopedConfig(/*rows*/ 2, /*bytes*/ 0);

  // Unsorted input should keep the legacy blocking behavior and emit one
  // output batch after full materialization.
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({2, 1, 3}),
           makeFlatVector<int32_t>({2, 2, 1}),
           makeFlatVector<int64_t>({80, 20, 100})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({1, 2, 1}),
           makeFlatVector<int32_t>({1, 1, 3}),
           makeFlatVector<int64_t>({10, 70, 30})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int32_t>({4, 3, 1}),
           makeFlatVector<int32_t>({1, 2, 4}),
           makeFlatVector<int64_t>({110, 110, 40})})};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .window({"row_number() over (partition by p order by s) as rn"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = copyResultsWithSingleRowGpuBatches(plan, task);
  auto expected = makeExpectedWindowResult(
      {2, 1, 3, 1, 2, 1, 4, 3, 1},
      {2, 2, 1, 1, 1, 3, 1, 2, 4},
      {80, 20, 100, 10, 70, 30, 110, 110, 40},
      {2, 2, 1, 1, 1, 3, 1, 2, 4});
  facebook::velox::test::assertEqualVectors(expected, result);

  ASSERT_TRUE(wasCudfWindowUsed(task));
  EXPECT_EQ(1, cudfWindowOutputVectors(task));
}

} // namespace
