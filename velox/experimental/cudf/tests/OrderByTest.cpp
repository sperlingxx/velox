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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <fmt/format.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

namespace facebook::velox::cudf_velox::test {

class CudfOrderByTestHelper {
 public:
  static void setMemoryLimits(
      uint64_t sortedRunBytes,
      uint64_t mergeChunkBytes,
      uint64_t outputChunkBytes,
      cudf::size_type maxOutputRows) {
    CudfOrderBy::testingSetMemoryLimits(
        sortedRunBytes, mergeChunkBytes, outputChunkBytes, maxOutputRows);
  }

  static void resetMemoryLimits() {
    CudfOrderBy::testingResetMemoryLimits();
  }

  static uint64_t maxActiveRuns() {
    return CudfOrderBy::testingMaxActiveRuns();
  }

  static uint64_t sourceChunks() {
    return CudfOrderBy::testingSourceChunks();
  }

  static uint64_t mergeOutputBatches() {
    return CudfOrderBy::testingMergeOutputBatches();
  }

  static uint64_t emittedChunks() {
    return CudfOrderBy::testingEmittedChunks();
  }

  static uint64_t spillCleanups() {
    return CudfOrderBy::testingSpillCleanups();
  }
};

} // namespace facebook::velox::cudf_velox::test

namespace {

using OrderByTestHelper =
    facebook::velox::cudf_velox::test::CudfOrderByTestHelper;

class OrderByTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
    rng_.seed(123);
    timestampUnit_ = cudf_velox::CudfConfig::getInstance().timestampUnit;

    rowType_ = ROW(
        {{"c0", INTEGER()},
         {"c1", INTEGER()},
         {"c2", VARCHAR()},
         {"c3", VARCHAR()}});
  }

  void TearDown() override {
    OrderByTestHelper::resetMemoryLimits();
    cudf_velox::CudfConfig::getInstance().timestampUnit = timestampUnit_;
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} NULLS LAST", key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format("SELECT * FROM tmp ORDER BY {} DESC NULLS FIRST", key),
        {keyIndex});
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key,
      const std::string& filter) {
    core::PlanNodeId orderById;
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);
    auto plan = PlanBuilder()
                    .values(input)
                    .filter(filter)
                    .orderBy({fmt::format("{} ASC NULLS LAST", key)}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} NULLS LAST", filter, key),
        {keyIndex});

    plan = PlanBuilder()
               .values(input)
               .filter(filter)
               .orderBy({fmt::format("{} DESC NULLS FIRST", key)}, false)
               .capturePlanNodeId(orderById)
               .planNode();
    runTest(
        plan,
        orderById,
        fmt::format(
            "SELECT * FROM tmp WHERE {} ORDER BY {} DESC NULLS FIRST",
            filter,
            key),
        {keyIndex});
  }

  void testTwoKeys(
      const std::vector<RowVectorPtr>& input,
      const std::string& key1,
      const std::string& key2) {
    auto& rowType = input[0]->type()->asRow();
    auto keyIndices = {rowType.getChildIdx(key1), rowType.getChildIdx(key2)};

    std::vector<core::SortOrder> sortOrders = {
        core::kAscNullsLast, core::kDescNullsFirst};
    std::vector<std::string> sortOrderSqls = {"NULLS LAST", "DESC NULLS FIRST"};

    for (int i = 0; i < sortOrders.size(); i++) {
      for (int j = 0; j < sortOrders.size(); j++) {
        core::PlanNodeId orderById;
        auto plan = PlanBuilder()
                        .values(input)
                        .orderBy(
                            {fmt::format("{} {}", key1, sortOrderSqls[i]),
                             fmt::format("{} {}", key2, sortOrderSqls[j])},
                            false)
                        .capturePlanNodeId(orderById)
                        .planNode();
        runTest(
            plan,
            orderById,
            fmt::format(
                "SELECT * FROM tmp ORDER BY {} {}, {} {}",
                key1,
                sortOrderSqls[i],
                key2,
                sortOrderSqls[j]),
            keyIndices);
      }
    }
  }

  void runTest(
      core::PlanNodePtr planNode,
      const core::PlanNodeId& orderById,
      const std::string& duckDbSql,
      const std::vector<uint32_t>& sortingKeys) {
    {
      SCOPED_TRACE("run without spilling");
      assertQueryOrdered(planNode, duckDbSql, sortingKeys);
    }
  }

  std::vector<RowVectorPtr> makeVectors(
      const RowTypePtr& rowType,
      int32_t numVectors,
      int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          facebook::velox::test::BatchMaker::createBatch(
              rowType, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  folly::Random::DefaultGenerator rng_;
  RowTypePtr rowType_;
  cudf::type_id timestampUnit_;
};

TEST_F(OrderByTest, externalSpillSchemaEligibility) {
  const auto timestampOrderBy =
      std::dynamic_pointer_cast<const core::OrderByNode>(
          PlanBuilder()
              .tableScan(ROW({"key", "payload"}, {TIMESTAMP(), INTEGER()}))
              .orderBy({"key ASC NULLS LAST"}, false)
              .planNode());
  ASSERT_NE(timestampOrderBy, nullptr);

  auto& config = cudf_velox::CudfConfig::getInstance();
  config.timestampUnit = cudf::type_id::TIMESTAMP_SECONDS;
  EXPECT_FALSE(cudf_velox::CudfOrderBy::isSupported(timestampOrderBy));

  for (const auto unit : {cudf::type_id::TIMESTAMP_MILLISECONDS,
                          cudf::type_id::TIMESTAMP_MICROSECONDS,
                          cudf::type_id::TIMESTAMP_NANOSECONDS}) {
    config.timestampUnit = unit;
    EXPECT_TRUE(cudf_velox::CudfOrderBy::isSupported(timestampOrderBy));
  }

  const auto safeNestedPayload =
      std::dynamic_pointer_cast<const core::OrderByNode>(
          PlanBuilder()
              .tableScan(ROW(
                  {"key", "payload"},
                  {INTEGER(), ARRAY(ROW({BIGINT(), VARCHAR()}))}))
              .orderBy({"key ASC NULLS LAST"}, false)
              .planNode());
  ASSERT_NE(safeNestedPayload, nullptr);
  EXPECT_TRUE(cudf_velox::CudfOrderBy::isSupported(safeNestedPayload));

  const auto unsupportedBinaryPayload =
      std::dynamic_pointer_cast<const core::OrderByNode>(
          PlanBuilder()
              .tableScan(
                  ROW({"key", "payload"}, {INTEGER(), VARBINARY()}))
              .orderBy({"key ASC NULLS LAST"}, false)
              .planNode());
  ASSERT_NE(unsupportedBinaryPayload, nullptr);
  EXPECT_FALSE(
      cudf_velox::CudfOrderBy::isSupported(unsupportedBinaryPayload));
}

TEST_F(OrderByTest, externalSpillConstructorDefense) {
  const auto timestampOrderBy =
      std::dynamic_pointer_cast<const core::OrderByNode>(
          PlanBuilder()
              .tableScan(ROW({"key", "payload"}, {INTEGER(), TIMESTAMP()}))
              .orderBy({"key ASC NULLS LAST"}, false)
              .planNode());
  ASSERT_NE(timestampOrderBy, nullptr);

  const auto makeFakeTask = [&](const std::string& taskId) {
    core::PlanFragment fakePlanFragment;
    fakePlanFragment.planNode = std::make_shared<core::ValuesNode>(
        core::PlanNodeId{"0"}, std::vector<RowVectorPtr>{});
    return Task::create(
        taskId,
        std::move(fakePlanFragment),
        0,
        core::QueryCtx::create(executor_.get()),
        Task::ExecutionMode::kParallel);
  };

  auto& config = cudf_velox::CudfConfig::getInstance();
  config.timestampUnit = cudf::type_id::TIMESTAMP_SECONDS;
  {
    auto fakeTask = makeFakeTask("CudfOrderByTest.unsupported");
    DriverCtx driverCtx(fakeTask, 0, 0, 0, 0);
    VELOX_ASSERT_THROW(
        cudf_velox::CudfOrderBy(1, &driverCtx, timestampOrderBy),
        "unsupported external-spill schema");
  }

  config.timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
  {
    auto fakeTask = makeFakeTask("CudfOrderByTest.supported");
    DriverCtx driverCtx(fakeTask, 0, 0, 0, 0);
    EXPECT_NO_THROW({
      cudf_velox::CudfOrderBy orderBy(1, &driverCtx, timestampOrderBy);
      orderBy.close();
    });
  }
}

TEST_F(OrderByTest, selectiveFilter) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 3; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  // c0 values are unique across batches
  testSingleKey(vectors, "c0", "c0 % 333 = 0");

  // c1 values are unique only within a batch
  testSingleKey(vectors, "c1", "c1 % 333 = 0");
}

TEST_F(OrderByTest, singleKey) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");

  // parser doesn't support "is not null" expression, hence, using c0 % 2 >= 0
  testSingleKey(vectors, "c0", "c0 % 2 >= 0");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 DESC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan, orderById, "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST", {0});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 ASC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(plan, orderById, "SELECT * FROM tmp ORDER BY c0 NULLS FIRST", {0});
}

TEST_F(OrderByTest, multipleKeys) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    // c0: half of rows are null, a quarter is 0 and remaining quarter is 1
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [](vector_size_t row) { return row % 4; }, nullEvery(2, 1));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [](vector_size_t row) { return row; }, nullEvery(7));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1");

  core::PlanNodeId orderById;
  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy({"c0 ASC NULLS FIRST", "c1 ASC NULLS LAST"}, false)
                  .capturePlanNodeId(orderById)
                  .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 NULLS FIRST, c1 NULLS LAST",
      {0, 1});

  plan = PlanBuilder()
             .values(vectors)
             .orderBy({"c0 DESC NULLS LAST", "c1 DESC NULLS FIRST"}, false)
             .capturePlanNodeId(orderById)
             .planNode();
  runTest(
      plan,
      orderById,
      "SELECT * FROM tmp ORDER BY c0 DESC NULLS LAST, c1 DESC NULLS FIRST",
      {0, 1});
}

TEST_F(OrderByTest, multiBatchResult) {
  vector_size_t batchSize = 5000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 10; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c1, c1, c1, c1}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0");
}

TEST_F(OrderByTest, varfields) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    auto c2 = makeFlatVector<StringView>(
        batchSize,
        [](vector_size_t row) {
          return StringView::makeInline(std::to_string(row));
        },
        nullEvery(17));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c2");
}

TEST_F(OrderByTest, boundedExternalSortManyRuns) {
  constexpr int32_t kNumRuns = 23;
  constexpr vector_size_t kRowsPerRun = 97;
  constexpr cudf::size_type kMaxOutputRows = 17;
  // One input vector per run plus tiny reader/output chunks deterministically
  // exercises multiple binary compaction levels and final paused readers.
  OrderByTestHelper::setMemoryLimits(1, 4096, 4096, kMaxOutputRows);

  std::vector<RowVectorPtr> vectors;
  vectors.reserve(kNumRuns);
  for (int32_t run = 0; run < kNumRuns; ++run) {
    auto c0 = makeFlatVector<int64_t>(
        kRowsPerRun,
        [run](vector_size_t row) { return (row * 7 + run * 3) % 17; },
        nullEvery(23, run % 23));
    auto c1 = makeFlatVector<int64_t>(
        kRowsPerRun,
        [run](vector_size_t row) { return (row * 5 + run * 11) % 19; },
        nullEvery(29, (run * 2) % 29));
    // A unique final key makes the expected global order deterministic even
    // though SQL does not define insertion order for complete sort-key ties.
    auto c2 = makeFlatVector<int64_t>(
        kRowsPerRun,
        [run](vector_size_t row) { return run * kRowsPerRun + row; });
    auto payload = makeFlatVector<std::string>(
        kRowsPerRun, [run](vector_size_t row) {
          return fmt::format("run={};row={};", run, row) +
              std::string(512, static_cast<char>('a' + run % 26));
        });
    vectors.push_back(makeRowVector({c0, c1, c2, payload}));
  }
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .orderBy(
                      {"c0 ASC NULLS FIRST",
                       "c1 DESC NULLS LAST",
                       "c2 ASC NULLS LAST"},
                      false)
                  .planNode();
  assertQueryOrdered(
      plan,
      "SELECT * FROM tmp ORDER BY c0 ASC NULLS FIRST, "
      "c1 DESC NULLS LAST, c2 ASC NULLS LAST",
      {0, 1, 2});

  EXPECT_EQ(OrderByTestHelper::maxActiveRuns(), 2);
  EXPECT_GE(OrderByTestHelper::sourceChunks(), kNumRuns);
  EXPECT_GT(OrderByTestHelper::mergeOutputBatches(), 0);
  EXPECT_GE(
      OrderByTestHelper::emittedChunks(),
      (kNumRuns * kRowsPerRun + kMaxOutputRows - 1) / kMaxOutputRows);
  EXPECT_EQ(OrderByTestHelper::spillCleanups(), 1);
}

/// Verifies output batch rows of OrderBy
TEST_F(OrderByTest, outputBatchRows) {
  struct {
    int numRowsPerBatch;
    int preferredOutBatchBytes;
    int maxOutBatchRows;
    int expectedOutputVectors;

    // TODO: add output size check with spilling enabled
    std::string debugString() const {
      return fmt::format(
          "numRowsPerBatch:{}, preferredOutBatchBytes:{}, maxOutBatchRows:{}, expectedOutputVectors:{}",
          numRowsPerBatch,
          preferredOutBatchBytes,
          maxOutBatchRows,
          expectedOutputVectors);
    }
  } testSettings[] = {
      {1024, 1, 100, 1024},
      // estimated size per row is ~2092, set preferredOutBatchBytes to 20920,
      // so each batch has 10 rows, so it would return 100 batches
      {1000, 20920, 100, 100},
      // same as above, but maxOutBatchRows is 1, so it would return 1000
      // batches
      {1000, 20920, 1, 1000}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    const vector_size_t batchSize = testData.numRowsPerBatch;
    std::vector<RowVectorPtr> rowVectors;
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(5));
    auto c1 = makeFlatVector<double>(
        batchSize, [&](vector_size_t row) { return row; }, nullEvery(11));
    std::vector<VectorPtr> vectors;
    vectors.push_back(c0);
    for (int i = 0; i < 256; ++i) {
      vectors.push_back(c1);
    }
    rowVectors.push_back(makeRowVector(vectors));
    createDuckDbTable(rowVectors);

    core::PlanNodeId orderById;
    auto plan = PlanBuilder()
                    .values(rowVectors)
                    .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                    .capturePlanNodeId(orderById)
                    .planNode();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    queryCtx->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes,
          std::to_string(testData.preferredOutBatchBytes)},
         {core::QueryConfig::kMaxOutputBatchRows,
          std::to_string(testData.maxOutBatchRows)},
         {facebook::velox::cudf_velox::CudfToVelox::kPassthroughMode,
          "false"}});
    CursorParameters params;
    params.planNode = plan;
    params.queryCtx = queryCtx;
    auto task = assertQueryOrdered(
        params, "SELECT * FROM tmp ORDER BY c0 ASC NULLS LAST", {0});

    EXPECT_EQ(
        testData.expectedOutputVectors,
        toPlanStats(task->taskStats())
            .at(orderById)
            .operatorStats.at("CudfToVelox")
            ->outputVectors);
  }
}

} // namespace
