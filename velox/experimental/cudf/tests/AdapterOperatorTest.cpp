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
#include "velox/experimental/cudf/exec/CudfValues.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/functions/sparksql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class AdapterOperatorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    aggregate::prestosql::registerAllAggregateFunctions();
    functions::prestosql::registerAllScalarFunctions();
    window::prestosql::registerAllWindowFunctions();
    functions::window::sparksql::registerWindowFunctions("");
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  bool wasCudfWindowUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfWindow") {
          return true;
        }
      }
    }
    return false;
  }
};

TEST_F(AdapterOperatorTest, adapterStatsMergedIntoPlanNode) {
  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3, 4, 5})});

  core::PlanNodeId projNodeId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"c0 * 2 as x"})
                  .capturePlanNodeId(projNodeId)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto stats = toPlanStats(task->taskStats());
  auto& projStats = stats.at(projNodeId);

  EXPECT_TRUE(projStats.isMultiOperatorTypeNode());
  EXPECT_TRUE(projStats.operatorStats.count("CudfToVelox"));
}

TEST_F(AdapterOperatorTest, fullPartitionWindowSumUsesCudfWindow) {
  auto data = makeRowVector(
      {"k0", "k1", "v"},
      {makeFlatVector<int64_t>({1, 1, 1, 2, 2, 3}),
       makeFlatVector<std::string>({"us", "us", "ca", "us", "us", "us"}),
       makeNullableFlatVector<int64_t>({10, 20, 5, 7, std::nullopt, 1})});
  createDuckDbTable({data});

  auto parsedPlan =
      PlanBuilder()
          .values({data})
          .window({"sum(v) over (partition by k0, k1 rows between "
                   "unbounded preceding and unbounded following) as s"})
          .planNode();
  auto parsedWindow =
      std::dynamic_pointer_cast<const core::WindowNode>(parsedPlan);
  ASSERT_NE(parsedWindow, nullptr);
  auto windowFunctions = parsedWindow->windowFunctions();
  ASSERT_EQ(windowFunctions.size(), 1);
  // DuckDB's parser cannot retain ROWS vs RANGE when both frame bounds are
  // UNBOUNDED. Rebuild the typed node with the frame required by this test.
  windowFunctions.front().frame.type = core::WindowNode::WindowType::kRows;
  auto plan = core::WindowNode::Builder(*parsedWindow)
                  .windowFunctions(std::move(windowFunctions))
                  .build();
  ASSERT_EQ(
      plan->windowFunctions().front().frame.type,
      core::WindowNode::WindowType::kRows);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT k0, k1, v, sum(v) over (partition by k0, k1 "
                      "rows between unbounded preceding and unbounded "
                      "following) FROM tmp");

  EXPECT_TRUE(wasCudfWindowUsed(task));
}

TEST_F(AdapterOperatorTest, streamingRowNumberCrossesInputBatches) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"k", "o", "payload"},
          {makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt, 1, 1}),
           makeNullableFlatVector<int64_t>({std::nullopt, 0, 0, 1}),
           makeFlatVector<int64_t>({0, 1, 2, 3})}),
      makeRowVector(
          {"k", "o", "payload"},
          {makeNullableFlatVector<int64_t>({1, 1, 2, 2}),
           makeNullableFlatVector<int64_t>({2, 3, std::nullopt, 0}),
           makeFlatVector<int64_t>({4, 5, 6, 7})})};
  createDuckDbTable(data);

  auto plan =
      PlanBuilder()
          .values(data)
          .streamingWindow(
              {"row_number() over (partition by k order by o asc nulls first) "
               "as rn"})
          .planNode();

  auto task = assertQueryOrdered(
      plan,
      "SELECT k, o, payload, row_number() over (partition by k order by o "
      "asc nulls first) FROM tmp ORDER BY k ASC NULLS FIRST, "
      "o ASC NULLS FIRST",
      {0, 1});
  EXPECT_TRUE(wasCudfWindowUsed(task));
}

TEST_F(AdapterOperatorTest, streamingRankFixesContinuedTieAndLaterPeer) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"k", "o", "payload"},
          {makeFlatVector<int64_t>({1, 1, 1, 1}),
           makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt, 1, 1}),
           makeFlatVector<int64_t>({0, 1, 2, 3})}),
      makeRowVector(
          {"k", "o", "payload"},
          {makeFlatVector<int64_t>({1, 1, 1, 2}),
           makeNullableFlatVector<int64_t>({1, 2, 2, std::nullopt}),
           makeFlatVector<int64_t>({4, 5, 6, 7})}),
      makeRowVector(
          {"k", "o", "payload"},
          {makeFlatVector<int64_t>({2, 2}),
           makeNullableFlatVector<int64_t>({std::nullopt, 0}),
           makeFlatVector<int64_t>({8, 9})})};
  createDuckDbTable(data);

  auto plan =
      PlanBuilder()
          .values(data)
          .streamingWindow(
              {"row_number() over (partition by k order by o asc nulls first) "
               "as rn",
               "rank() over (partition by k order by o asc nulls first) "
               "as rnk"})
          .planNode();

  auto task = assertQueryOrdered(
      plan,
      "SELECT k, o, payload, "
      "row_number() over (partition by k order by o asc nulls first), "
      "rank() over (partition by k order by o asc nulls first) "
      "FROM tmp ORDER BY k ASC NULLS FIRST, o ASC NULLS FIRST",
      {0, 1});
  EXPECT_TRUE(wasCudfWindowUsed(task));
}

TEST_F(AdapterOperatorTest, streamingRankProducesOutputBeforeNoMoreInput) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"k", "o"},
          {makeFlatVector<int64_t>({1, 1}),
           makeFlatVector<int64_t>({0, 1})}),
      makeRowVector(
          {"k", "o"},
          {makeFlatVector<int64_t>({1, 1}),
           makeFlatVector<int64_t>({2, 3})})};

  auto plan =
      PlanBuilder()
          .values(data)
          .streamingWindow(
              {"rank() over (partition by k order by o) as rnk"})
          .planNode();
  auto windowNode =
      std::dynamic_pointer_cast<const core::WindowNode>(plan);
  ASSERT_NE(windowNode, nullptr);
  auto valuesNode = std::dynamic_pointer_cast<const core::ValuesNode>(
      windowNode->sources()[0]);
  ASSERT_NE(valuesNode, nullptr);

  auto task = Task::create(
      "streaming-rank-produces-output-before-no-more-input",
      core::PlanFragment{plan},
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driver = Driver::testingCreate(std::make_unique<DriverCtx>(
      task, 0, 0, kUngroupedGroupId, 0));
  cudf_velox::CudfValues values(0, driver->driverCtx(), valuesNode);
  cudf_velox::CudfWindow window(1, driver->driverCtx(), windowNode);
  values.initialize();
  window.initialize();

  for (size_t i = 0; i < data.size(); ++i) {
    auto input = values.getOutput();
    ASSERT_NE(input, nullptr);
    ASSERT_TRUE(window.needsInput());
    window.addInput(std::move(input));

    // This is intentionally before noMoreInput(). The legacy implementation
    // kept needsInput() true and returned nullptr here until all input arrived.
    EXPECT_FALSE(window.needsInput());
    EXPECT_FALSE(window.isFinished());
    {
      auto output = window.getOutput();
      ASSERT_NE(output, nullptr);
      EXPECT_EQ(output->size(), data[i]->size());
    }
    EXPECT_TRUE(window.needsInput());
    EXPECT_FALSE(window.isFinished());
  }

  EXPECT_EQ(values.getOutput(), nullptr);
  window.noMoreInput();
  EXPECT_TRUE(window.isFinished());
  EXPECT_EQ(window.getOutput(), nullptr);
  window.close();
  values.close();
}

TEST_F(AdapterOperatorTest, streamingRankThreeBatchDescendingNullTie) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"k", "a", "b", "payload"},
          {makeFlatVector<int64_t>({1, 1}),
           makeNullableFlatVector<int64_t>({10, 9}),
           makeNullableFlatVector<int64_t>({5, std::nullopt}),
           makeFlatVector<int64_t>({0, 1})}),
      makeRowVector(
          {"k", "a", "b", "payload"},
          {makeFlatVector<int64_t>({1}),
           makeNullableFlatVector<int64_t>({9}),
           makeNullableFlatVector<int64_t>({std::nullopt}),
           makeFlatVector<int64_t>({2})}),
      makeRowVector(
          {"k", "a", "b", "payload"},
          {makeFlatVector<int64_t>({1, 1, 1, 1, 1, 1, 1}),
           makeNullableFlatVector<int64_t>(
               {9, 9, 9, 9, 8, std::nullopt, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {std::nullopt, 7, 7, 3, std::nullopt, std::nullopt, 4}),
           makeFlatVector<int64_t>({3, 4, 5, 6, 7, 8, 9})})};

  auto plan =
      PlanBuilder()
          .values(data)
          .streamingWindow(
              {"row_number() over (partition by k order by a desc nulls last, "
               "b desc nulls first) as rn",
               "rank() over (partition by k order by a desc nulls last, "
               "b desc nulls first) as rnk"})
          .planNode();

  auto expected = makeRowVector(
      {"k", "a", "b", "payload", "rn", "rnk"},
      {makeFlatVector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
       makeNullableFlatVector<int64_t>(
           {10, 9, 9, 9, 9, 9, 9, 8, std::nullopt, std::nullopt}),
       makeNullableFlatVector<int64_t>(
           {5,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            7,
            7,
            3,
            std::nullopt,
            std::nullopt,
            4}),
       makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
       makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
       makeFlatVector<int32_t>({1, 2, 2, 2, 5, 5, 7, 8, 9, 10})});

  auto task = AssertQueryBuilder(plan).maxDrivers(1).assertResults(expected);
  EXPECT_TRUE(wasCudfWindowUsed(task));
}

TEST_F(AdapterOperatorTest, streamingRankDoesNotRetainGiantPartition) {
  constexpr int32_t kBatches = 32;
  constexpr vector_size_t kRowsPerBatch = 2048;
  std::vector<RowVectorPtr> data;
  data.reserve(kBatches);
  for (int32_t batch = 0; batch < kBatches; ++batch) {
    data.push_back(makeRowVector(
        {"k", "o"},
        {makeFlatVector<int64_t>(
             kRowsPerBatch, [](vector_size_t) { return 7; }),
         makeFlatVector<int64_t>(
             kRowsPerBatch, [batch](vector_size_t row) {
               return static_cast<int64_t>(batch) * kRowsPerBatch + row;
             })}));
  }
  createDuckDbTable(data);

  auto plan =
      PlanBuilder()
          .values(data)
          .streamingWindow(
              {"rank() over (partition by k order by o) as rnk"})
          .planNode();
  auto task = assertQueryOrdered(
      plan,
      "SELECT k, o, rank() over (partition by k order by o) "
      "FROM tmp ORDER BY k, o",
      {0, 1});
  EXPECT_TRUE(wasCudfWindowUsed(task));
}

TEST_F(AdapterOperatorTest, orderedFirstValueUsesCudfWindow) {
  auto data = makeRowVector(
      {"k", "v", "o0", "o1"},
      {makeFlatVector<int64_t>({1, 1, 1, 2, 2, 2}),
       makeNullableFlatVector<int64_t>({10, 20, 40, 7, std::nullopt, 9}),
       makeNullableFlatVector<int64_t>({5, 7, 9, 1, 3, 2}),
       makeNullableFlatVector<int64_t>({100, 50, 10, 1, 1, 1})});
  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .window({"first_value(v) over (partition by k order by o0 desc "
                   "nulls last, o1 desc nulls last) as first_v"})
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT k, v, o0, o1, first_value(v) over (partition by "
                      "k order by o0 desc nulls last, o1 desc nulls last) "
                      "FROM tmp");

  EXPECT_TRUE(wasCudfWindowUsed(task));
}
