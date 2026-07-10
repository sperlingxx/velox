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
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <stdexcept>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class TopNTest : public OperatorTestBase {
 public:
  void SetUp() override {
    OperatorTestBase::SetUp();
    cudf_velox::registerCudf();
  }
  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

 protected:
  static std::vector<std::string> getSortOrderSqls() {
    return {"NULLS LAST", "NULLS FIRST", "DESC NULLS FIRST", "DESC NULLS LAST"};
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key,
      int32_t limit) {
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);

    auto sortOrderSqls = getSortOrderSqls();

    for (const auto& sortOrderSql : sortOrderSqls) {
      auto sql = fmt::format("{} {}", key, sortOrderSql);

      auto plan =
          PlanBuilder().values(input).topN({sql}, limit, false).planNode();

      assertQueryOrdered(
          plan,
          fmt::format("SELECT * FROM tmp ORDER BY {} LIMIT {}", sql, limit),
          {keyIndex});
    }
  }

  void testSingleKey(
      const std::vector<RowVectorPtr>& input,
      const std::string& key,
      const std::string& filter) {
    auto keyIndex = input[0]->type()->asRow().getChildIdx(key);

    auto sortOrderSqls = getSortOrderSqls();

    for (const auto& sortOrderSql : sortOrderSqls) {
      auto sql = fmt::format("{} {}", key, sortOrderSql);

      auto plan = PlanBuilder()
                      .values(input)
                      .filter(filter)
                      .topN({sql}, 10, false)
                      .planNode();

      assertQueryOrdered(
          plan,
          fmt::format(
              "SELECT * FROM tmp WHERE {} ORDER BY {} LIMIT 10", filter, sql),
          {keyIndex});
    }
  }

  void testTwoKeys(
      const std::vector<RowVectorPtr>& input,
      const std::string& key1,
      const std::string& key2,
      int32_t limit) {
    auto& rowType = input[0]->type()->asRow();
    auto keyIndices = {rowType.getChildIdx(key1), rowType.getChildIdx(key2)};

    auto sortOrderSqls = getSortOrderSqls();

    for (const auto& sortOrderSql1 : sortOrderSqls) {
      for (const auto& sortOrderSql2 : sortOrderSqls) {
        auto sql1 = fmt::format("{} {}", key1, sortOrderSql1);
        auto sql2 = fmt::format("{} {}", key2, sortOrderSql2);

        auto plan = PlanBuilder()
                        .values(input)
                        .topN({sql1, sql2}, limit, false)
                        .planNode();

        assertQueryOrdered(
            plan,
            fmt::format(
                "SELECT * FROM tmp ORDER BY {}, {} LIMIT {}",
                sql1,
                sql2,
                limit),
            keyIndices);
      }
    }
  }
};

TEST_F(TopNTest, selectiveFilter) {
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

TEST_F(TopNTest, singleKey) {
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

  // DESC NULLS LAST and ASC NULLS FIRST will return all rows where c0 is null;
  // There are 400 rows where c0 is null. Use limit greater than 400 to make the
  // query deterministic.
  testSingleKey(vectors, "c0", 410);

  // parser doesn't support "is not null" expression, hence, using c0 % 2 >= 0
  testSingleKey(vectors, "c0", "c0 % 2 >= 0");
}

TEST_F(TopNTest, multipleKeys) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [](vector_size_t row) { return row % 4; }, nullEvery(31, 1));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [](vector_size_t row) { return row; }, nullEvery(17));
    auto c2 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1", 200);
}

TEST_F(TopNTest, compaction) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [&](vector_size_t row) { return batchSize * i + row; },
        nullEvery(31));
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; }, nullEvery(11));
    vectors.push_back(makeRowVector({c0, c1, c1, c1, c1, c1}));
  }
  createDuckDbTable(vectors);

  // Make sure LIMIT is greater than number of rows with nulls to avoid
  // non-deterministic results.
  testSingleKey(vectors, "c0", 500);

  testSingleKey(vectors, "c0", 900);
}

TEST_F(TopNTest, varchar) {
  vector_size_t batchSize = 1'000;
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
        nullEvery(31));
    auto c3 = makeFlatVector<std::string>(batchSize, [](vector_size_t row) {
      return "non inline string " + std::to_string(row);
    });
    vectors.push_back(makeRowVector({c0, c1, c2, c3}));
  }
  createDuckDbTable(vectors);

  // Make sure LIMIT is greater than number of rows with nulls to avoid
  // non-deterministic results.
  testSingleKey(vectors, "c2", 200);
}

TEST_F(TopNTest, multiBatch) {
  vector_size_t batchSize = 1'000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return batchSize * i + row; });
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; });
    auto c2 = makeFlatVector<StringView>(batchSize, [](vector_size_t row) {
      return StringView::makeInline(std::to_string(row));
    });
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0", 1'500);
  testSingleKey(vectors, "c2", 2'500);
}

TEST_F(TopNTest, numericTopNSynchronization) {
  constexpr vector_size_t batchSize = 132000;
  constexpr int32_t numBatches = 6;
  constexpr int32_t topN = 1024;
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(numBatches);

  for (int32_t batch = 0; batch < numBatches; ++batch) {
    auto id = makeFlatVector<int64_t>(batchSize, [&](vector_size_t row) {
      return static_cast<int64_t>(batch) * batchSize + row;
    });
    auto key = makeFlatVector<double>(batchSize, [&](vector_size_t row) {
      // Repeat a small key space to force tie-breaking on c0.
      auto bucket =
          static_cast<int64_t>((row + batch * 13) % 10007); // A custom hash.
      return static_cast<double>(bucket);
    });
    auto payload = makeFlatVector<int64_t>(batchSize, [&](vector_size_t row) {
      return static_cast<int64_t>(row ^ (batch << 10));
    });
    vectors.push_back(makeRowVector(std::vector<VectorPtr>{id, key, payload}));
  }

  createDuckDbTable(vectors);

  auto plan =
      PlanBuilder()
          .values(vectors)
          .topN({"c1 DESC NULLS LAST", "c0 ASC NULLS LAST"}, topN, false)
          .planNode();

  // Force configuration to maximize chance of replicating a missing stream
  // sync.
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, batchSize)
      .config(cudf_velox::CudfConfig::kCudfTopNBatchSize, 1)
      .assertResults(
          fmt::format(
              "SELECT c0, c1, c2 FROM tmp ORDER BY c1 DESC, c0 LIMIT {}",
              topN));
}

TEST_F(TopNTest, empty) {
  vector_size_t batchSize = 1'000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize, [&](vector_size_t row) { return batchSize * i + row; });
    auto c1 = makeFlatVector<double>(
        batchSize, [](vector_size_t row) { return row * 0.1; });
    auto c2 = makeFlatVector<StringView>(batchSize, [](vector_size_t row) {
      return StringView::makeInline(std::to_string(row));
    });
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testSingleKey(vectors, "c0", "c0 < 0");
}

TEST_F(TopNTest, lowCardinality) {
  vector_size_t size = 1'000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 5; ++i) {
    auto c0 = makeFlatVector<int64_t>(size, [](auto /*row*/) { return 0; });
    auto c1 = makeFlatVector<int32_t>(size, [](auto /*row*/) { return 10; });
    auto c2 = makeFlatVector<double>(size, [](auto /*row*/) { return 12.5; });
    vectors.push_back(makeRowVector({c0, c1, c2}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1", 200);
}

// All columns are sorting keys.
TEST_F(TopNTest, onlyKeys) {
  vector_size_t batchSize = 1000;
  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < 2; ++i) {
    auto c0 = makeFlatVector<int64_t>(
        batchSize,
        [](vector_size_t row) { return row % 20; },
        nullEvery(31, 1));
    auto c1 = makeFlatVector<int32_t>(
        batchSize, [](vector_size_t row) { return row; }, nullEvery(17));
    vectors.push_back(makeRowVector({c0, c1}));
  }
  createDuckDbTable(vectors);

  testTwoKeys(vectors, "c0", "c1", 200);
}

TEST_F(TopNTest, planNodeValidation) {
  auto data = makeRowVector(
      ROW({"a", "b"},
          {
              BIGINT(),
              BIGINT(),
          }),
      10);
  auto plan = [&](const std::vector<std::string>& sortingKeys,
                  int32_t count = 10) {
    PlanBuilder().values({data}).topN(sortingKeys, count, false).planNode();
  };

  VELOX_ASSERT_THROW(plan({}), "TopN must specify sorting keys");
  VELOX_ASSERT_THROW(
      plan({"a"}, 0),
      "TopN must specify greater than zero number of rows to keep");
  VELOX_ASSERT_THROW(
      plan({"a", "b", "a"}),
      "TopN must specify unique sorting keys. Found duplicate key: a");
}

namespace facebook::velox::cudf_velox::test {

class CudfTopNRowNumberTestHelper {
 public:
  using CompactionPermit = CudfTopNRowNumber::CompactionPermit;

  static void setMemoryLimits(
      uint64_t candidateRunBytes,
      uint64_t mergeChunkBytes,
      uint64_t outputChunkBytes,
      cudf::size_type maxOutputRows) {
    CudfTopNRowNumber::testingSetMemoryLimits(
        candidateRunBytes,
        mergeChunkBytes,
        outputChunkBytes,
        maxOutputRows);
  }

  static void resetMemoryLimits() {
    CudfTopNRowNumber::testingResetMemoryLimits();
  }

  static size_t activeCompactionPermits() {
    return CompactionPermit::testingActivePermits();
  }
};

} // namespace facebook::velox::cudf_velox::test

namespace facebook::velox::cudf_velox {
namespace {

using TopNTestHelper = test::CudfTopNRowNumberTestHelper;

class TopNRowNumberTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    registerCudf();
    // Force one spill run per input vector and many small reader/output chunks.
    TopNTestHelper::setMemoryLimits(1, 512, 4096, 31);
  }

  void TearDown() override {
    TopNTestHelper::resetMemoryLimits();
    unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> makeRankInputs(int32_t numRuns) {
    constexpr vector_size_t kRowsPerRun = 97;
    std::vector<RowVectorPtr> inputs;
    inputs.reserve(numRuns);
    for (int32_t run = 0; run < numRuns; ++run) {
      auto partition = makeFlatVector<int64_t>(
          kRowsPerRun,
          [run](vector_size_t row) { return (row + run * 3) % 11; },
          nullEvery(17, run % 17));
      auto orderKey = makeFlatVector<int64_t>(
          kRowsPerRun,
          [run](vector_size_t row) {
            // Small key space creates peers spanning input runs and chunks.
            return (row * 5 + run * 7) % 13;
          },
          nullEvery(19, (run * 2) % 19));
      auto payload = makeFlatVector<int64_t>(
          kRowsPerRun,
          [run](vector_size_t row) {
            return static_cast<int64_t>(run) * kRowsPerRun + row;
          });
      inputs.push_back(makeRowVector({partition, orderKey, payload}));
    }
    return inputs;
  }
};

class TopNRowNumberRunTest
    : public TopNRowNumberTest,
      public testing::WithParamInterface<int32_t> {};

TEST_P(TopNRowNumberRunTest, pausedMergeRunsAndNullOrdering) {
  auto inputs = makeRankInputs(GetParam());
  createDuckDbTable(inputs);

  for (const auto& order : {"DESC NULLS LAST", "ASC NULLS FIRST"}) {
    auto plan = exec::test::PlanBuilder()
                    .values(inputs)
                    .topNRank("rank", {"c0"}, {fmt::format("c1 {}", order)}, 1, true)
                    .planNode();
    assertQuery(
        plan,
        fmt::format(
            "SELECT c0, c1, c2, row_number FROM ("
            "SELECT c0, c1, c2, rank() OVER (PARTITION BY c0 ORDER BY c1 {}) "
            "AS row_number FROM tmp) WHERE row_number = 1",
            order));
  }
}

INSTANTIATE_TEST_SUITE_P(
    TwoThreeFiveRuns,
    TopNRowNumberRunTest,
    testing::Values(2, 3, 5),
    [](const testing::TestParamInfo<int32_t>& info) {
      return fmt::format("{}Runs", info.param);
    });

TEST_F(TopNRowNumberTest, globalPartitionAndGiantPeerStreamInChunks) {
  constexpr int32_t kRuns = 5;
  constexpr vector_size_t kRowsPerRun = 257;
  std::vector<RowVectorPtr> inputs;
  for (int32_t run = 0; run < kRuns; ++run) {
    auto partition = makeFlatVector<int64_t>(
        kRowsPerRun, [](vector_size_t /*row*/) { return 0; });
    auto orderKey = makeFlatVector<int64_t>(
        kRowsPerRun, [](vector_size_t /*row*/) { return 7; });
    auto payload = makeFlatVector<int64_t>(
        kRowsPerRun,
        [run](vector_size_t row) {
          return static_cast<int64_t>(run) * kRowsPerRun + row;
        });
    inputs.push_back(makeRowVector({partition, orderKey, payload}));
  }
  createDuckDbTable(inputs);

  auto partitioned = exec::test::PlanBuilder()
                         .values(inputs)
                         .topNRank(
                             "dense_rank",
                             {"c0"},
                             {"c1 DESC NULLS LAST"},
                             1,
                             true)
                         .planNode();
  assertQuery(
      partitioned,
      "SELECT c0, c1, c2, row_number FROM ("
      "SELECT c0, c1, c2, dense_rank() OVER (PARTITION BY c0 ORDER BY c1 DESC "
      "NULLS LAST) AS row_number FROM tmp) WHERE row_number = 1");

  // Empty partition keys are one global partition. This used to select a
  // zero-column cuDF table and slice it with non-zero row indices.
  auto global = exec::test::PlanBuilder()
                    .values(inputs)
                    .topNRank(
                        "rank", {}, {"c1 DESC NULLS LAST"}, 1, true)
                    .planNode();
  assertQuery(
      global,
      "SELECT c0, c1, c2, row_number FROM ("
      "SELECT c0, c1, c2, rank() OVER (ORDER BY c1 DESC NULLS LAST) AS "
      "row_number FROM tmp) WHERE row_number = 1");
}

TEST_F(TopNRowNumberTest, giantPartitionDropsWorseContinuation) {
  constexpr int32_t kRuns = 5;
  constexpr vector_size_t kRowsPerRun = 257;
  std::vector<RowVectorPtr> inputs;
  for (int32_t run = 0; run < kRuns; ++run) {
    auto partition = makeFlatVector<int64_t>(
        kRowsPerRun, [](vector_size_t /*row*/) { return 0; });
    auto orderKey = makeFlatVector<int64_t>(
        kRowsPerRun, [run](vector_size_t /*row*/) { return run == 0 ? 10 : 9; });
    auto payload = makeFlatVector<int64_t>(
        kRowsPerRun,
        [run](vector_size_t row) {
          return static_cast<int64_t>(run) * kRowsPerRun + row;
        });
    inputs.push_back(makeRowVector({partition, orderKey, payload}));
  }
  createDuckDbTable(inputs);

  auto plan = exec::test::PlanBuilder()
                  .values(inputs)
                  .topNRank(
                      "rank", {"c0"}, {"c1 DESC NULLS LAST"}, 1, true)
                  .planNode();
  assertQuery(
      plan,
      "SELECT c0, c1, c2, row_number FROM ("
      "SELECT c0, c1, c2, rank() OVER (PARTITION BY c0 ORDER BY c1 DESC NULLS "
      "LAST) AS row_number FROM tmp) WHERE row_number = 1");
}

TEST_F(TopNRowNumberTest, rowNumberManyPartitionsAndSpanningContinuation) {
  constexpr int32_t kRuns = 5;
  constexpr vector_size_t kRowsPerRun = 257;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kRuns);
  for (int32_t run = 0; run < kRuns; ++run) {
    auto partition = makeFlatVector<int64_t>(
        kRowsPerRun, [run](vector_size_t row) {
          // One partition crosses every run; all others are unique so the
          // candidate reduction is intentionally close to a no-op (Job8-like).
          return row == 0
              ? -1
              : static_cast<int64_t>(run) * kRowsPerRun + row;
        });
    auto orderKey = makeFlatVector<int64_t>(
        kRowsPerRun, [run](vector_size_t row) {
          if (row == 0) {
            // The first run owns the winner. Later chunks for partition -1 are
            // strictly worse and must be dropped by cross-batch row_number
            // state instead of emitted again.
            return static_cast<int64_t>(run == 0 ? 100 : 90);
          }
          return static_cast<int64_t>((row * 17 + run * 11) % 101);
        });
    auto payload = makeFlatVector<std::string>(
        kRowsPerRun, [run](vector_size_t row) {
          // Wider-than-test-reader-limit rows force the shared partition's five
          // candidates across paused reader/output chunks.
          return fmt::format("run={};row={};", run, row) +
              std::string(2048, static_cast<char>('a' + run));
        });
    inputs.push_back(makeRowVector({partition, orderKey, payload}));
  }
  createDuckDbTable(inputs);

  auto plan = exec::test::PlanBuilder()
                  .values(inputs)
                  .topNRowNumber(
                      {"c0"}, {"c1 DESC NULLS LAST"}, 1, true)
                  .planNode();
  assertQuery(
      plan,
      "SELECT c0, c1, c2, row_number FROM ("
      "SELECT c0, c1, c2, row_number() OVER (PARTITION BY c0 ORDER BY c1 DESC "
      "NULLS LAST) AS row_number FROM tmp) WHERE row_number = 1");
}

} // namespace

TEST(TopNCompactionAdmissionTest, exceptionRelease) {
  using Helper = test::CudfTopNRowNumberTestHelper;
  using Permit = Helper::CompactionPermit;

  EXPECT_EQ(Helper::activeCompactionPermits(), 0);
  bool caught{false};
  try {
    Permit permit(1);
    EXPECT_TRUE(permit.ready());
    EXPECT_FALSE(permit.hasWaitFuture());
    EXPECT_EQ(Helper::activeCompactionPermits(), 1);
    throw std::runtime_error("injected compaction failure");
  } catch (const std::runtime_error&) {
    caught = true;
  }
  EXPECT_TRUE(caught);
  EXPECT_EQ(Helper::activeCompactionPermits(), 0);
  {
    Permit nextPermit(1);
    EXPECT_EQ(Helper::activeCompactionPermits(), 1);
  }
  EXPECT_EQ(Helper::activeCompactionPermits(), 0);
}

TEST(TopNCompactionAdmissionTest, fifoAndCancellation) {
  using Helper = test::CudfTopNRowNumberTestHelper;
  using Permit = Helper::CompactionPermit;

  EXPECT_EQ(Helper::activeCompactionPermits(), 0);

  // Zero preserves the documented unlimited mode and never consumes a slot.
  {
    Permit unlimited(0);
    EXPECT_TRUE(unlimited.ready());
    EXPECT_FALSE(unlimited.hasWaitFuture());
    EXPECT_EQ(Helper::activeCompactionPermits(), 0);
  }

  auto first = std::make_unique<Permit>(1);
  auto second = std::make_unique<Permit>(1);
  auto third = std::make_unique<Permit>(1);
  ASSERT_TRUE(first->ready());
  ASSERT_FALSE(second->ready());
  ASSERT_FALSE(third->ready());
  ASSERT_TRUE(second->hasWaitFuture());
  ASSERT_TRUE(third->hasWaitFuture());
  auto secondFuture = second->takeWaitFuture();
  auto thirdFuture = third->takeWaitFuture();
  EXPECT_FALSE(secondFuture.isReady());
  EXPECT_FALSE(thirdFuture.isReady());

  // Releasing the holder grants exactly the oldest request.
  first.reset();
  EXPECT_TRUE(secondFuture.isReady());
  EXPECT_FALSE(thirdFuture.isReady());
  EXPECT_TRUE(second->ready());
  EXPECT_FALSE(third->ready());
  EXPECT_EQ(Helper::activeCompactionPermits(), 1);

  // Cancelling the granted request releases its slot and wakes the next FIFO
  // waiter. A queued cancellation also realizes its future without consuming a
  // slot, which is what lets task cancellation remove a parked driver.
  second.reset();
  EXPECT_TRUE(thirdFuture.isReady());
  EXPECT_TRUE(third->ready());
  EXPECT_EQ(Helper::activeCompactionPermits(), 1);

  auto cancelled = std::make_unique<Permit>(1);
  ASSERT_FALSE(cancelled->ready());
  auto cancelledFuture = cancelled->takeWaitFuture();
  EXPECT_FALSE(cancelledFuture.isReady());
  cancelled.reset();
  EXPECT_TRUE(cancelledFuture.isReady());
  EXPECT_EQ(Helper::activeCompactionPermits(), 1);

  third.reset();
  EXPECT_EQ(Helper::activeCompactionPermits(), 0);
}

} // namespace facebook::velox::cudf_velox
