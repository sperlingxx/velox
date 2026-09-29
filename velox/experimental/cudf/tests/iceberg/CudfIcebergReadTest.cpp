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

/// Basic end-to-end read tests for the cudf Iceberg connector.

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfPreparedSplitsStore.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/iceberg/CudfDeletionVectorTestUtils.h"
#include "velox/experimental/cudf/tests/iceberg/CudfIcebergTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/Base64.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"

#include <cudf/io/parquet.hpp>

#include <folly/Random.h>
#include <folly/ScopeGuard.h>
#include <folly/String.h>
#include <folly/lang/Bits.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector::hive::iceberg;
using namespace facebook::velox::cudf_velox::iceberg::test;
using facebook::velox::common::testutil::TempFilePath;
using facebook::velox::connector::hive::HiveColumnHandle;

namespace facebook::velox::cudf_velox::exec::test {

namespace {

class CountingBufferedInputBuilder final
    : public ::facebook::velox::connector::hive::BufferedInputBuilder {
 public:
  explicit CountingBufferedInputBuilder(
      std::shared_ptr<::facebook::velox::connector::hive::BufferedInputBuilder>
          delegate)
      : delegate_(std::move(delegate)) {}

  std::unique_ptr<dwio::common::BufferedInput> create(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOptions,
      const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<io::IoStatistics> ioStatistics,
      std::shared_ptr<IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {})
      override {
    createCount_.fetch_add(1, std::memory_order_relaxed);
    return delegate_->create(
        fileHandle,
        readerOptions,
        connectorQueryCtx,
        std::move(ioStatistics),
        std::move(ioStats),
        executor,
        fileReadOps);
  }

  uint64_t createCount() const {
    return createCount_.load(std::memory_order_relaxed);
  }

 private:
  std::shared_ptr<::facebook::velox::connector::hive::BufferedInputBuilder>
      delegate_;
  std::atomic_uint64_t createCount_{0};
};

class OwnedFileDataSource final : public cudf::io::datasource {
 public:
  explicit OwnedFileDataSource(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    VELOX_CHECK(input, "Failed to open prepared test file {}", path);
    const auto end = input.tellg();
    VELOX_CHECK(
        end != std::streampos{-1},
        "Failed to size prepared test file {}",
        path);
    bytes_.resize(static_cast<size_t>(end));
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(bytes_.data()),
        static_cast<std::streamsize>(bytes_.size()));
    VELOX_CHECK(input, "Failed to read prepared test file {}", path);
  }

  size_t size() const override {
    return bytes_.size();
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override {
    VELOX_CHECK_LE(offset, bytes_.size());
    const auto readSize = std::min(size, bytes_.size() - offset);
    std::vector<uint8_t> result(readSize);
    if (readSize > 0) {
      std::memcpy(result.data(), bytes_.data() + offset, readSize);
    }
    return buffer::create(std::move(result));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* destination) override {
    VELOX_CHECK_LE(offset, bytes_.size());
    const auto readSize = std::min(size, bytes_.size() - offset);
    if (readSize > 0) {
      VELOX_CHECK_NOT_NULL(destination);
      std::memcpy(destination, bytes_.data() + offset, readSize);
    }
    return readSize;
  }

 private:
  std::vector<uint8_t> bytes_;
};

} // namespace

class CudfIcebergReadTest : public CudfIcebergTestBase {
 protected:
  static constexpr int rowCount = 20000;

  static std::vector<int64_t> makeContinuousIncreasingValues(
      int64_t begin,
      int64_t end) {
    std::vector<int64_t> values;
    values.resize(end - begin);
    std::iota(values.begin(), values.end(), begin);
    return values;
  }

  std::vector<int64_t> makeRandomIncreasingValues(int64_t begin, int64_t end) {
    VELOX_CHECK(begin < end);
    std::mt19937 gen{0};
    std::vector<int64_t> values;
    values.reserve(end - begin);
    for (int i = begin; i < end; i++) {
      if (folly::Random::rand32(0, 10, gen) > 8) {
        values.push_back(i);
      }
    }
    return values;
  }

  /// @rowGroupSizesForFiles The key is the file name, and the value is a vector
  /// of RowGroup sizes.
  /// @deleteFilesForBaseDatafiles The key is the delete file name, and the
  /// value contains the information about the content of this delete file.
  void assertPositionalDeletes(
      const std::map<std::string, std::vector<int64_t>>& rowGroupSizesForFiles,
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles,
      int32_t numPrefetchSplits = 0,
      int32_t splitCount = 1) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths =
        writeDataFiles(rowGroupSizesForFiles);
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths = writePositionDeleteFiles(
            deleteFilesForBaseDatafiles, dataFilePaths);

    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;

    for (const auto& dataFile : dataFilePaths) {
      std::string baseFileName = dataFile.first;
      std::string baseFilePath = dataFile.second->getPath();

      std::vector<IcebergDeleteFile> deleteFiles;

      for (auto const& deleteFile : deleteFilesForBaseDatafiles) {
        std::string deleteFileName = deleteFile.first;
        std::multimap<std::string, std::vector<int64_t>> deleteFileContent =
            deleteFile.second;

        if (deleteFileContent.count(baseFileName) != 0) {
          auto deleteFilePath =
              deleteFilePaths[deleteFileName].second->getPath();
          IcebergDeleteFile icebergDeleteFile(
              FileContent::kPositionalDeletes,
              deleteFilePath,
              dwio::common::FileFormat::DWRF,
              deleteFilePaths[deleteFileName].first,
              getFileSize(deleteFilePath));
          deleteFiles.push_back(icebergDeleteFile);
        }
      }

      auto icebergSplits =
          makeIcebergSplits(baseFilePath, deleteFiles, {}, splitCount);
      splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
    }

    std::string duckdbSql =
        getDuckDBQuery(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
    auto plan = PlanBuilder()
                    .startTableScan()
                    .connectorId(kCudfIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();
    auto task = assertQuery(plan, splits, duckdbSql, numPrefetchSplits);

    auto planStats = toPlanStats(task->taskStats());
    auto it = planStats.find(plan->id());
    ASSERT_TRUE(it != planStats.end());
    // TODO (mh): enable once we start to track gpu memory
    // ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  void assertSingleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositionsVec) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles = {
        {"data_file_1", {10000, 10000}}};
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles = {
            {"delete_file_1", {{"data_file_1", deletePositionsVec}}}};
    assertPositionalDeletes(
        rowGroupSizesForFiles, deleteFilesForBaseDatafiles, 0);
  }

  void assertMultipleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositionsVec) {
    int64_t previousFileRowCount = 500;
    int64_t afterFileRowCount = 500;
    assertPositionalDeletes(
        {
            {"data_file_0", {previousFileRowCount}},
            {"data_file_1", {10000, 10000}},
            {"data_file_2", {afterFileRowCount}},
        },
        {{"delete_file_1", {{"data_file_1", deletePositionsVec}}}},
        0);
  }

  void assertSingleBaseFileMultipleDeleteFiles(
      const std::vector<std::vector<int64_t>>& deletePositionsVecs) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles = {
        {"data_file_1", {10000, 10000}}};
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles;
    for (int i = 0; i < deletePositionsVecs.size(); i++) {
      std::string deleteFileName = fmt::format("delete_file_{}", i);
      deleteFilesForBaseDatafiles[deleteFileName] = {
          {"data_file_1", deletePositionsVecs[i]}};
    }
    assertPositionalDeletes(
        rowGroupSizesForFiles, deleteFilesForBaseDatafiles, 0);
  }

  void assertMultipleSplits(
      const std::vector<int64_t>& deletePositions,
      int32_t fileCount,
      int32_t numPrefetchSplits,
      int rowCountPerFile = rowCount,
      int32_t splitCountPerFile = 1) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles;
    for (int32_t i = 0; i < fileCount; i++) {
      std::string dataFileName = fmt::format("data_file_{}", i);
      rowGroupSizesForFiles[dataFileName] = {rowCountPerFile};
    }
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles;
    for (int i = 0; i < fileCount; i++) {
      std::string deleteFileName = fmt::format("delete_file_{}", i);
      deleteFilesForBaseDatafiles[deleteFileName] = {
          {fmt::format("data_file_{}", i), deletePositions}};
    }
    assertPositionalDeletes(
        rowGroupSizesForFiles,
        deleteFilesForBaseDatafiles,
        numPrefetchSplits,
        splitCountPerFile);
  }

 private:
  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths;
    std::vector<RowVectorPtr> dataVectorsJoined;
    dataVectorsJoined.reserve(rowGroupSizesForFiles.size());

    int64_t startingValue = 0;
    for (auto& dataFile : rowGroupSizesForFiles) {
      dataFilePaths[dataFile.first] = TempFilePath::create();

      std::vector<RowVectorPtr> dataVectors =
          makeVectors(dataFile.second, startingValue);
      writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors);

      for (int i = 0; i < dataVectors.size(); i++) {
        dataVectorsJoined.push_back(dataVectors[i]);
      }
    }

    createDuckDbTable(dataVectorsJoined);
    return dataFilePaths;
  }

  std::unordered_map<
      std::string,
      std::pair<int64_t, std::shared_ptr<TempFilePath>>>
  writePositionDeleteFiles(
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles,
      std::map<std::string, std::shared_ptr<TempFilePath>> baseFilePaths) {
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths;
    deleteFilePaths.reserve(deleteFilesForBaseDatafiles.size());

    for (auto& deleteFile : deleteFilesForBaseDatafiles) {
      auto deleteFileName = deleteFile.first;
      auto deleteFileContent = deleteFile.second;
      auto deleteFilePath = TempFilePath::create();

      std::vector<RowVectorPtr> deleteFileVectors;
      int64_t totalPositionsInDeleteFile = 0;

      for (auto& deleteFileRowGroup : deleteFileContent) {
        auto baseFileName = deleteFileRowGroup.first;
        auto baseFilePath = baseFilePaths[baseFileName]->getPath();
        auto positionsInRowGroup = deleteFileRowGroup.second;

        auto filePathVector = makeFlatVector<std::string>(
            static_cast<vector_size_t>(positionsInRowGroup.size()),
            [&](vector_size_t) { return baseFilePath; });
        auto deletePosVector = makeFlatVector<int64_t>(positionsInRowGroup);

        RowVectorPtr deleteFileVector = makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {filePathVector, deletePosVector});

        deleteFileVectors.push_back(deleteFileVector);
        totalPositionsInDeleteFile += positionsInRowGroup.size();
      }

      writeDeleteFile(
          DeleteFileFormat::DWRF, deleteFilePath->getPath(), deleteFileVectors);

      deleteFilePaths[deleteFileName] =
          std::make_pair(totalPositionsInDeleteFile, deleteFilePath);
    }

    return deleteFilePaths;
  }

  std::vector<RowVectorPtr> makeVectors(
      std::vector<int64_t> vectorSizes,
      int64_t& startingValue) {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(vectorSizes.size());
    for (int j = 0; j < vectorSizes.size(); j++) {
      auto data = makeContinuousIncreasingValues(
          startingValue, startingValue + vectorSizes[j]);
      vectors.push_back(makeRowVector({makeFlatVector<int64_t>(data)}));
      startingValue += vectorSizes[j];
    }
    return vectors;
  }

  std::string getDuckDBQuery(
      const std::map<std::string, std::vector<int64_t>>& rowGroupSizesForFiles,
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles) {
    int64_t totalNumRowsInAllBaseFiles = 0;
    std::map<std::string, int64_t> baseFileSizes;
    for (auto rowGroupSizesInFile : rowGroupSizesForFiles) {
      baseFileSizes[rowGroupSizesInFile.first] += std::accumulate(
          rowGroupSizesInFile.second.begin(),
          rowGroupSizesInFile.second.end(),
          0LL);
      totalNumRowsInAllBaseFiles += baseFileSizes[rowGroupSizesInFile.first];
    }

    std::map<std::string, std::vector<std::vector<int64_t>>>
        deletePosVectorsForAllBaseFiles;
    for (auto& deleteFile : deleteFilesForBaseDatafiles) {
      auto deleteFileContent = deleteFile.second;
      for (auto& rowGroup : deleteFileContent) {
        auto baseFileName = rowGroup.first;
        deletePosVectorsForAllBaseFiles[baseFileName].push_back(
            rowGroup.second);
      }
    }

    std::map<std::string, std::vector<int64_t>>
        flattenedDeletePosVectorsForAllBaseFiles;
    int64_t totalNumDeletePositions = 0;
    for (auto& deleteVectorsForBaseFile : deletePosVectorsForAllBaseFiles) {
      auto baseFileName = deleteVectorsForBaseFile.first;
      auto deletePositionVectors = deleteVectorsForBaseFile.second;
      std::vector<int64_t> deletePositionVector =
          flattenAndDedup(deletePositionVectors, baseFileSizes[baseFileName]);
      flattenedDeletePosVectorsForAllBaseFiles[baseFileName] =
          deletePositionVector;
      totalNumDeletePositions += deletePositionVector.size();
    }

    if (totalNumDeletePositions == 0) {
      return "SELECT * FROM tmp";
    }
    if (totalNumDeletePositions >= totalNumRowsInAllBaseFiles) {
      return "SELECT * FROM tmp WHERE 1 = 0";
    }

    std::vector<int64_t> allDeleteValues;
    int64_t numRowsInPreviousBaseFiles = 0;
    for (auto& baseFileSize : baseFileSizes) {
      auto deletePositions =
          flattenedDeletePosVectorsForAllBaseFiles[baseFileSize.first];
      if (numRowsInPreviousBaseFiles > 0) {
        for (int64_t& deleteValue : deletePositions) {
          deleteValue += numRowsInPreviousBaseFiles;
        }
      }
      allDeleteValues.insert(
          allDeleteValues.end(),
          deletePositions.begin(),
          deletePositions.end());
      numRowsInPreviousBaseFiles += baseFileSize.second;
    }

    return fmt::format(
        "SELECT * FROM tmp WHERE c0 NOT IN ({})",
        folly::join(", ", allDeleteValues));
  }

  std::vector<int64_t> flattenAndDedup(
      const std::vector<std::vector<int64_t>>& deletePositionVectors,
      int64_t baseFileSize) {
    std::vector<int64_t> deletePositionVector;
    for (auto& vec : deletePositionVectors) {
      for (auto pos : vec) {
        if (pos >= 0 && pos < baseFileSize) {
          deletePositionVector.push_back(pos);
        }
      }
    }
    std::sort(deletePositionVector.begin(), deletePositionVector.end());
    auto last =
        std::unique(deletePositionVector.begin(), deletePositionVector.end());
    deletePositionVector.erase(last, deletePositionVector.end());
    return deletePositionVector;
  }

  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

/// Basic read without any deletes.
TEST_F(CudfIcebergReadTest, basicRead) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfIcebergConnectorId,
          "iceberg_table",
          facebook::velox::common::SubfieldFilters{},
          nullptr,
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, reusesFooterMetadataDuringReaderSetup) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto originalBuilder =
      ::facebook::velox::connector::hive::BufferedInputBuilder::getInstance();
  auto countingBuilder =
      std::make_shared<CountingBufferedInputBuilder>(originalBuilder);
  ::facebook::velox::connector::hive::BufferedInputBuilder::registerBuilder(
      countingBuilder);
  SCOPE_EXIT {
    ::facebook::velox::connector::hive::BufferedInputBuilder::registerBuilder(
        originalBuilder);
  };

  AssertQueryBuilder(makeTableScanPlan(rowType))
      .connectorSessionProperty(
          kCudfIcebergConnectorId,
          cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
          "true")
      .splits(makeIcebergSplits(dataFile->getPath()))
      .assertResults({data});

  EXPECT_EQ(countingBuilder->createCount(), 1);
}

/// Read with multiple columns.
TEST_F(CudfIcebergReadTest, multiColumn) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto data = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read a larger file to verify chunked reading works.
TEST_F(CudfIcebergReadTest, largerFile) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto values = makeContinuousIncreasingValues(0, 10000);
  auto data = makeRowVector({makeFlatVector<int64_t>(values)});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp", 0);
}

/// Read with multiple data files (multiple splits).
TEST_F(CudfIcebergReadTest, multipleSplits) {
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto splits1 = makeIcebergSplits(filePath1->getPath());
  auto splits2 = makeIcebergSplits(filePath2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  createDuckDbTable({data1, data2});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, allSplits, "SELECT * FROM tmp", 0);
}

TEST_F(CudfIcebergReadTest, generatedReadyFilesRunThroughTableScan) {
  using cudf_velox::connector::hive::iceberg::CudfPreparedIcebergFile;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<RowVectorPtr, 4> data{
      makeRowVector({makeFlatVector<int64_t>({0, 1, 2})}),
      makeRowVector({makeFlatVector<int64_t>({10, 11, 12})}),
      makeRowVector({makeFlatVector<int64_t>({20, 21, 22})}),
      makeRowVector({makeFlatVector<int64_t>({30, 31, 32})})};
  std::array<std::shared_ptr<TempFilePath>, 4> paths{
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create()};
  for (size_t i = 0; i < data.size(); ++i) {
    writeToFile(paths[i]->getPath(), data[i]);
  }

  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = makeTableScanPlan(rowType);
  params.maxDrivers = 2;
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(), "0", task.queryCtx()->executor());
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 20));
  };
  auto cursor = TaskCursor::create(params);
  ASSERT_NE(source, nullptr);
  cursor->start();

  const auto makePrepared = [&](uint64_t ordinal) {
    auto original = makeIcebergSplits(paths[ordinal]->getPath());
    auto descriptor =
        std::dynamic_pointer_cast<const HiveIcebergSplit>(original.front());
    VELOX_CHECK_NOT_NULL(descriptor);
    auto dataSource =
        std::make_shared<OwnedFileDataSource>(paths[ordinal]->getPath());
    return CudfPreparedIcebergFile{
        ordinal, descriptor, dataSource, dataSource->size()};
  };

  ASSERT_TRUE(source->publishPrepared(makePrepared(0)));
  ASSERT_TRUE(source->publishPrepared(makePrepared(2)));
  std::vector<int64_t> firstValues;
  while (firstValues.size() < 6) {
    ASSERT_TRUE(cursor->moveNext());
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      firstValues.push_back(decoded->valueAt(row));
    }
  }
  std::sort(firstValues.begin(), firstValues.end());
  EXPECT_EQ(firstValues, (std::vector<int64_t>{0, 1, 2, 20, 21, 22}));

  ASSERT_TRUE(source->publishPrepared(makePrepared(1)));
  ASSERT_TRUE(source->publishPrepared(makePrepared(3)));
  source->finish();

  std::vector<int64_t> remaining;
  while (cursor->moveNext()) {
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      remaining.push_back(decoded->valueAt(row));
    }
  }
  std::sort(remaining.begin(), remaining.end());
  EXPECT_EQ(remaining, (std::vector<int64_t>{10, 11, 12, 30, 31, 32}));

  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  const auto stats = cursor->task()->taskStats();
  EXPECT_GE(stats.numTotalSplits, 2);
  EXPECT_LE(stats.numTotalSplits, 4);
  EXPECT_EQ(stats.numFinishedSplits, stats.numTotalSplits);
  EXPECT_EQ(stats.numRunningSplits, 0);
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
}

TEST_F(CudfIcebergReadTest, preparedRowGroupsRunThroughHybridTableScan) {
  using cudf_velox::connector::hive::planParquetFileRanges;
  using cudf_velox::connector::hive::iceberg::CudfPreparedIcebergFile;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<std::shared_ptr<TempFilePath>, 2> paths{
      TempFilePath::create(), TempFilePath::create()};

  const auto writeRowGroupedFile = [&](const std::string& path, int64_t base) {
    std::vector<int64_t> values(kRowsPerFile);
    std::iota(values.begin(), values.end(), base);
    auto input = makeRowVector({makeFlatVector<int64_t>(values)});
    const auto stream = cudf::get_default_stream();
    auto table = with_arrow::toCudfTable(
        input, input->pool(), stream, cudf::get_current_device_resource_ref());
    cudf::io::table_input_metadata metadata(table->view());
    metadata.column_metadata[0].set_name("c0");
    auto options = cudf::io::parquet_writer_options::builder(
                       cudf::io::sink_info{path}, table->view())
                       .metadata(std::move(metadata))
                       .row_group_size_rows(kRowsPerGroup)
                       .max_page_size_rows(kRowsPerGroup / 4)
                       .compression(cudf::io::compression_type::SNAPPY)
                       .build();
    cudf::io::write_parquet(options, stream);
    stream.synchronize();
  };
  writeRowGroupedFile(paths[0]->getPath(), 0);
  writeRowGroupedFile(paths[1]->getPath(), 100'000);

  const std::array<cudf::size_type, 2> selectedRowGroups{2, 1};
  std::vector<CudfPreparedIcebergFile> prepared;
  prepared.reserve(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    auto original = makeIcebergSplits(paths[i]->getPath());
    auto descriptor =
        std::dynamic_pointer_cast<const HiveIcebergSplit>(original.front());
    VELOX_CHECK_NOT_NULL(descriptor);
    auto dataSource =
        std::make_shared<OwnedFileDataSource>(paths[i]->getPath());
    auto readerOptions = cudf::io::parquet_reader_options::builder().build();
    readerOptions.set_column_names({"c0"});
    auto plan = planParquetFileRanges(
        dataSource, readerOptions, cudf::get_default_stream());
    ASSERT_EQ(plan.rowGroups.size(), 3);
    const auto rowGroup = selectedRowGroups[i];
    const auto preparedBytes = std::accumulate(
        plan.rowGroups[rowGroup].ranges.begin(),
        plan.rowGroups[rowGroup].ranges.end(),
        uint64_t{0},
        [](uint64_t total, const auto& range) {
          return total + static_cast<uint64_t>(range.size());
        });
    prepared.push_back(
        CudfPreparedIcebergFile{
            i,
            std::move(descriptor),
            std::move(dataSource),
            getFileSize(paths[i]->getPath()),
            std::make_shared<const cudf::io::parquet::FileMetaData>(
                std::move(plan.metadata)),
            rowGroup,
            preparedBytes});
  }

  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = makeTableScanPlan(rowType);
  params.maxDrivers = 1;
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(), "0", task.queryCtx()->executor());
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 30));
    for (auto& file : prepared) {
      ASSERT_TRUE(source->publishPrepared(std::move(file)));
    }
    source->finish();
  };

  auto cursor = TaskCursor::create(params);
  std::vector<int64_t> actual;
  while (cursor->moveNext()) {
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      actual.push_back(decoded->valueAt(row));
    }
  }
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  std::sort(actual.begin(), actual.end());
  std::vector<int64_t> expected(2 * kRowsPerGroup);
  std::iota(expected.begin(), expected.begin() + kRowsPerGroup, 40'000);
  std::iota(expected.begin() + kRowsPerGroup, expected.end(), 120'000);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(cursor->task()->taskStats().numTotalSplits, 1);
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
}

TEST_F(CudfIcebergReadTest, generatedRowGroupsPrepareThroughTableScan) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto path = TempFilePath::create();
  std::vector<int64_t> expected(kRowsPerFile);
  std::iota(expected.begin(), expected.end(), 0);
  auto input = makeRowVector({makeFlatVector<int64_t>(expected)});
  const auto stream = cudf::get_default_stream();
  auto table = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(table->view());
  metadata.column_metadata[0].set_name("c0");
  auto writerOptions = cudf::io::parquet_writer_options::builder(
                           cudf::io::sink_info{path->getPath()}, table->view())
                           .metadata(std::move(metadata))
                           .row_group_size_rows(kRowsPerGroup)
                           .max_page_size_rows(kRowsPerGroup / 4)
                           .compression(cudf::io::compression_type::SNAPPY)
                           .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto original = makeIcebergSplits(path->getPath());
  auto descriptor =
      std::dynamic_pointer_cast<HiveIcebergSplit>(original.front());
  VELOX_CHECK_NOT_NULL(descriptor);
  auto admission = std::make_shared<HostRangeAdmission>(64 << 20, 1);
  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = makeTableScanPlan(rowType);
  params.maxDrivers = 1;
  params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "3"}};
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId("generated-row-group-preparation")
                        .build();
  params.queryCtx->setConnectorSessionOverridesUnsafe(
      kCudfIcebergConnectorId,
      {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
        "true"}});
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(),
        "0",
        task.queryCtx()->executor(),
        admission,
        CudfPreparationMode::kRowGroupHybrid);
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 3, 1 << 20));
    ASSERT_TRUE(source->addDescriptor(0, descriptor));
    source->sealDescriptors();
  };

  auto cursor = TaskCursor::create(params);
  std::vector<int64_t> actual;
  while (cursor->moveNext()) {
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      actual.push_back(decoded->valueAt(row));
    }
  }
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(source->preparedCount(), 3);
  EXPECT_EQ(source->fallbackCount(), 0);
  const auto sourceMetrics = source->metrics();
  EXPECT_EQ(sourceMetrics.originalInputs, 1);
  EXPECT_EQ(sourceMetrics.plannedFiles, 1);
  EXPECT_EQ(sourceMetrics.totalRowGroups, 3);
  EXPECT_EQ(sourceMetrics.selectedRowGroups, 3);
  EXPECT_EQ(sourceMetrics.prunedRowGroups, 0);
  EXPECT_EQ(sourceMetrics.preparedUnits, 3);
  EXPECT_GT(sourceMetrics.preparedPayloadBytes, 0);
  EXPECT_EQ(sourceMetrics.issuedSplits, 3);
  EXPECT_EQ(sourceMetrics.preparedBatches, 3);
  EXPECT_EQ(sourceMetrics.issuedPreparedUnits, 3);
  EXPECT_EQ(sourceMetrics.issuedPreparedSources, 3);
  EXPECT_EQ(sourceMetrics.preparationFailures, 0);
  const auto admissionAfterDrivers = admission->snapshot();
  EXPECT_EQ(admissionAfterDrivers.currentBytes, 0);
  EXPECT_EQ(admissionAfterDrivers.currentUnits, 0);
  EXPECT_EQ(admissionAfterDrivers.peakUnits, 1);
  const auto planStats = toPlanStats(cursor->task()->taskStats());
  const auto& scanStats = planStats.at("0").customStats;
  EXPECT_EQ(scanStats.at("cudfReadyBatches").sum, 3);
  EXPECT_EQ(scanStats.at("cudfReadyUnits").sum, 3);
  EXPECT_EQ(scanStats.at("cudfReadySources").sum, 3);
  EXPECT_GT(scanStats.at("cudfReadySelectedBytes").sum, 0);
  EXPECT_GT(scanStats.at("cudfPreparedLeaseReadCalls").sum, 0);
  EXPECT_GT(scanStats.at("cudfPreparedLeaseReadBytes").sum, 0);
  EXPECT_GT(scanStats.at("cudfPreparedHybridH2dCopies").sum, 0);
  EXPECT_GT(scanStats.at("cudfPreparedHybridH2dBytes").sum, 0);
  EXPECT_EQ(scanStats.at("cudfPreparedUnexpectedBackingReads").sum, 0);
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
}

TEST_F(CudfIcebergReadTest, generatedRowGroupsPreservePushdownFilter) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto path = TempFilePath::create();
  std::vector<int64_t> values(kRowsPerFile);
  std::iota(values.begin(), values.end(), 0);
  auto input = makeRowVector({makeFlatVector<int64_t>(values)});
  const auto stream = cudf::get_default_stream();
  auto table = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(table->view());
  metadata.column_metadata[0].set_name("c0");
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{path->getPath()}, table->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerGroup)
          .max_page_size_rows(kRowsPerGroup / 4)
          .compression(cudf::io::compression_type::SNAPPY)
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto original = makeIcebergSplits(path->getPath());
  auto descriptor =
      std::dynamic_pointer_cast<HiveIcebergSplit>(original.front());
  VELOX_CHECK_NOT_NULL(descriptor);
  auto admission = std::make_shared<HostRangeAdmission>(64 << 20, 2);
  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = PlanBuilder()
                        .startTableScan()
                        .connectorId(kCudfIcebergConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .subfieldFilters({"c0 BETWEEN 20000 AND 39999"})
                        .endTableScan()
                        .planNode();
  params.maxDrivers = 1;
  params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "2"}};
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId("generated-row-group-filter")
                        .build();
  params.queryCtx->setConnectorSessionOverridesUnsafe(
      kCudfIcebergConnectorId,
      {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
        "true"}});
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(),
        "0",
        task.queryCtx()->executor(),
        admission,
        CudfPreparationMode::kRowGroupHybrid);
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 20));
    ASSERT_TRUE(source->addDescriptor(0, descriptor));
    source->sealDescriptors();
  };

  auto cursor = TaskCursor::create(params);
  std::vector<int64_t> actual;
  while (cursor->moveNext()) {
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      actual.push_back(decoded->valueAt(row));
    }
  }
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  std::vector<int64_t> expected(kRowsPerGroup);
  std::iota(expected.begin(), expected.end(), 20'000);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(source->preparedCount(), 1);
  EXPECT_EQ(source->fallbackCount(), 0);
  const auto sourceMetrics = source->metrics();
  EXPECT_EQ(sourceMetrics.originalInputs, 1);
  EXPECT_EQ(sourceMetrics.plannedFiles, 1);
  EXPECT_EQ(sourceMetrics.totalRowGroups, 3);
  EXPECT_EQ(sourceMetrics.selectedRowGroups, 1);
  EXPECT_EQ(sourceMetrics.prunedRowGroups, 2);
  EXPECT_EQ(sourceMetrics.preparedUnits, 1);
  EXPECT_EQ(sourceMetrics.issuedSplits, 1);
  EXPECT_EQ(sourceMetrics.preparedBatches, 1);
  EXPECT_EQ(sourceMetrics.issuedPreparedUnits, 1);
  EXPECT_EQ(sourceMetrics.issuedPreparedSources, 1);
  EXPECT_EQ(sourceMetrics.preparationFailures, 0);
  EXPECT_EQ(admission->snapshot().currentBytes, 0);
  EXPECT_EQ(admission->snapshot().currentUnits, 0);
}

TEST_F(CudfIcebergReadTest, generatedRowGroupsCloseWhenAllPruned) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto path = TempFilePath::create();
  std::vector<int64_t> values(kRowsPerFile);
  std::iota(values.begin(), values.end(), 0);
  auto input = makeRowVector({makeFlatVector<int64_t>(values)});
  const auto stream = cudf::get_default_stream();
  auto table = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(table->view());
  metadata.column_metadata[0].set_name("c0");
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{path->getPath()}, table->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerGroup)
          .max_page_size_rows(kRowsPerGroup / 4)
          .compression(cudf::io::compression_type::SNAPPY)
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto original = makeIcebergSplits(path->getPath());
  auto descriptor =
      std::dynamic_pointer_cast<HiveIcebergSplit>(original.front());
  VELOX_CHECK_NOT_NULL(descriptor);
  auto admission = std::make_shared<HostRangeAdmission>(64 << 20, 2);
  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = PlanBuilder()
                        .startTableScan()
                        .connectorId(kCudfIcebergConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .subfieldFilters({"c0 BETWEEN 100000 AND 110000"})
                        .endTableScan()
                        .planNode();
  params.maxDrivers = 1;
  params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "2"}};
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId("generated-row-group-all-pruned")
                        .build();
  params.queryCtx->setConnectorSessionOverridesUnsafe(
      kCudfIcebergConnectorId,
      {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
        "true"}});
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(),
        "0",
        task.queryCtx()->executor(),
        admission,
        CudfPreparationMode::kRowGroupHybrid);
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 20));
    ASSERT_TRUE(source->addDescriptor(0, descriptor));
    source->sealDescriptors();
  };

  auto cursor = TaskCursor::create(params);
  EXPECT_FALSE(cursor->moveNext());
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  const auto sourceMetrics = source->metrics();
  EXPECT_EQ(sourceMetrics.originalInputs, 1);
  EXPECT_EQ(sourceMetrics.plannedFiles, 1);
  EXPECT_EQ(sourceMetrics.totalRowGroups, 3);
  EXPECT_EQ(sourceMetrics.selectedRowGroups, 0);
  EXPECT_EQ(sourceMetrics.prunedRowGroups, 3);
  EXPECT_EQ(sourceMetrics.preparedUnits, 0);
  EXPECT_EQ(sourceMetrics.issuedSplits, 0);
  EXPECT_EQ(sourceMetrics.preparationFailures, 0);
  EXPECT_EQ(admission->snapshot().currentBytes, 0);
  EXPECT_EQ(admission->snapshot().currentUnits, 0);
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
}

TEST_F(CudfIcebergReadTest, generatedDescriptorsPrepareThroughTableScan) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<RowVectorPtr, 4> data{
      makeRowVector({makeFlatVector<int64_t>({0, 1, 2})}),
      makeRowVector({makeFlatVector<int64_t>({10, 11, 12})}),
      makeRowVector({makeFlatVector<int64_t>({20, 21, 22})}),
      makeRowVector({makeFlatVector<int64_t>({30, 31, 32})})};
  std::array<std::shared_ptr<TempFilePath>, 4> paths{
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create()};
  std::vector<std::shared_ptr<HiveIcebergSplit>> descriptors;
  descriptors.reserve(paths.size());
  for (size_t i = 0; i < data.size(); ++i) {
    writeToFile(paths[i]->getPath(), data[i]);
    auto splits = makeIcebergSplits(paths[i]->getPath());
    auto descriptor =
        std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
    VELOX_CHECK_NOT_NULL(descriptor);
    descriptors.push_back(std::move(descriptor));
  }

  auto admission = std::make_shared<HostRangeAdmission>(64 << 20, 4);
  std::shared_ptr<CudfPreparedSplitSource> source;
  CursorParameters params;
  params.planNode = makeTableScanPlan(rowType);
  params.maxDrivers = 2;
  params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "4"}};
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .queryId("generated-descriptor-preparation")
                        .build();
  params.queryCtx->setConnectorSessionOverridesUnsafe(
      kCudfIcebergConnectorId,
      {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
        "true"}});
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(), "0", task.queryCtx()->executor(), admission);
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 20));
    for (size_t i = 0; i < descriptors.size(); ++i) {
      ASSERT_TRUE(source->addDescriptor(i, descriptors[i]));
    }
    source->sealDescriptors();
  };

  auto cursor = TaskCursor::create(params);
  ASSERT_NE(source, nullptr);
  std::vector<int64_t> values;
  while (cursor->moveNext()) {
    auto output = cursor->current();
    auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
    for (vector_size_t row = 0; row < output->size(); ++row) {
      values.push_back(decoded->valueAt(row));
    }
  }
  std::sort(values.begin(), values.end());
  EXPECT_EQ(
      values,
      (std::vector<int64_t>{0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32}));
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  EXPECT_EQ(source->preparedCount(), 4);
  EXPECT_EQ(source->fallbackCount(), 0);
  const auto sourceMetrics = source->metrics();
  EXPECT_EQ(sourceMetrics.originalInputs, 4);
  EXPECT_EQ(sourceMetrics.preparedUnits, 4);
  EXPECT_EQ(sourceMetrics.issuedPreparedUnits, 4);
  EXPECT_EQ(sourceMetrics.issuedPreparedSources, 4);
  EXPECT_GT(sourceMetrics.preparedPayloadBytes, 0);
  EXPECT_EQ(sourceMetrics.preparationFailures, 0);
  const auto releaseDeadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (admission->snapshot().currentUnits != 0 &&
         std::chrono::steady_clock::now() < releaseDeadline) {
    std::this_thread::yield();
  }
  const auto admissionAfterDrivers = admission->snapshot();
  EXPECT_EQ(admissionAfterDrivers.currentBytes, 0);
  EXPECT_EQ(admissionAfterDrivers.currentUnits, 0);

  const auto stats = cursor->task()->taskStats();
  EXPECT_GE(stats.numTotalSplits, 2);
  EXPECT_LE(stats.numTotalSplits, 4);
  EXPECT_EQ(stats.numFinishedSplits, stats.numTotalSplits);
  EXPECT_EQ(stats.numRunningSplits, 0);
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
}

TEST_F(CudfIcebergReadTest, oversizedPreparedDescriptorFallsBack) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({7, 8, 9})});
  auto path = TempFilePath::create();
  writeToFile(path->getPath(), data);
  auto splits = makeIcebergSplits(path->getPath());
  auto descriptor = std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
  VELOX_CHECK_NOT_NULL(descriptor);

  for (const auto mode :
       {CudfPreparationMode::kFileRegular,
        CudfPreparationMode::kRowGroupHybrid}) {
    auto admission = std::make_shared<HostRangeAdmission>(1, 1);
    std::shared_ptr<CudfPreparedSplitSource> source;
    CursorParameters params;
    params.planNode = makeTableScanPlan(rowType);
    params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "1"}};
    params.queryCtx = core::QueryCtx::Builder()
                          .executor(driverExecutor_.get())
                          .queryId(
                              mode == CudfPreparationMode::kFileRegular
                                  ? "oversized-file-fallback"
                                  : "oversized-row-group-fallback")
                          .build();
    params.queryCtx->setConnectorSessionOverridesUnsafe(
        kCudfIcebergConnectorId,
        {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
          "true"}});
    params.beforeTaskStart = [&](Task& task) {
      source = std::make_shared<CudfPreparedSplitSource>(
          task.shared_from_this(),
          "0",
          task.queryCtx()->executor(),
          admission,
          mode);
      task.setSplitsStore(
          "0", std::make_unique<CudfPreparedSplitsStore>(source, 1, 1 << 20));
      ASSERT_TRUE(source->addDescriptor(0, descriptor));
      source->sealDescriptors();
    };

    auto cursor = TaskCursor::create(params);
    std::vector<int64_t> values;
    while (cursor->moveNext()) {
      auto output = cursor->current();
      auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
      for (vector_size_t row = 0; row < output->size(); ++row) {
        values.push_back(decoded->valueAt(row));
      }
    }
    ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
    ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
    EXPECT_EQ(values, (std::vector<int64_t>{7, 8, 9}));
    EXPECT_EQ(source->preparedCount(), 0);
    EXPECT_EQ(source->fallbackCount(), 1);
    EXPECT_EQ(admission->snapshot().currentBytes, 0);
    EXPECT_EQ(admission->snapshot().currentUnits, 0);
  }
}

TEST_F(CudfIcebergReadTest, successfulLimitStopsUnissuedPreparedFiles) {
  using cudf_velox::connector::hive::iceberg::CudfPreparedIcebergFile;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<std::shared_ptr<TempFilePath>, 4> paths{
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create()};
  for (size_t i = 0; i < paths.size(); ++i) {
    writeToFile(
        paths[i]->getPath(),
        makeRowVector({makeFlatVector<int64_t>(
            {static_cast<int64_t>(i * 10 + 1),
             static_cast<int64_t>(i * 10 + 2),
             static_cast<int64_t>(i * 10 + 3)})}));
  }

  std::shared_ptr<CudfPreparedSplitSource> source;
  std::vector<std::weak_ptr<cudf::io::datasource>> preparedSources;
  CursorParameters params;
  params.planNode = PlanBuilder()
                        .startTableScan()
                        .connectorId(kCudfIcebergConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .endTableScan()
                        .limit(0, 1, true)
                        .planNode();
  params.beforeTaskStart = [&](Task& task) {
    source = std::make_shared<CudfPreparedSplitSource>(
        task.shared_from_this(), "0", task.queryCtx()->executor());
    task.setSplitsStore(
        "0", std::make_unique<CudfPreparedSplitsStore>(source, 1, 1 << 20));
    for (size_t i = 0; i < paths.size(); ++i) {
      auto original = makeIcebergSplits(paths[i]->getPath());
      auto descriptor =
          std::dynamic_pointer_cast<const HiveIcebergSplit>(original.front());
      VELOX_CHECK_NOT_NULL(descriptor);
      auto dataSource =
          std::make_shared<OwnedFileDataSource>(paths[i]->getPath());
      preparedSources.push_back(dataSource);
      ASSERT_TRUE(source->publishPrepared(
          CudfPreparedIcebergFile{
              i,
              descriptor,
              std::move(dataSource),
              getFileSize(paths[i]->getPath())}));
    }
  };

  auto cursor = TaskCursor::create(params);
  int64_t rows = 0;
  while (cursor->moveNext()) {
    rows += cursor->current()->size();
  }
  EXPECT_EQ(rows, 1);
  ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
  ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
  EXPECT_EQ(cursor->task()->state(), TaskState::kFinished);
  EXPECT_EQ(source->preparedCount(), paths.size());
  EXPECT_EQ(cursor->task()->taskStats().numTotalSplits, 1);
  size_t releasedSources = 0;
  for (const auto& weak : preparedSources) {
    releasedSources += weak.expired();
  }
  // The completed Task retains the one issued reader until Task destruction;
  // all inputs that were never issued must be released by the terminal hook.
  EXPECT_EQ(releasedSources, paths.size() - 1);
  EXPECT_FALSE(source->publishFallback(
      std::make_shared<facebook::velox::connector::hive::HiveConnectorSplit>(
          kCudfIcebergConnectorId,
          "file:/tmp/after-limit",
          dwio::common::FileFormat::PARQUET)));
}

TEST_F(CudfIcebergReadTest, uniformLocalModeComparison) {
  using cudf_velox::connector::hive::HostRangeAdmission;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  enum class Mode { kLegacy, kFileRegular, kRowGroupHybrid };
  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;
  constexpr size_t kRepetitions = 6;
  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<std::shared_ptr<TempFilePath>, 4> paths{
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create()};
  std::vector<int64_t> expected;
  expected.reserve(paths.size() * kRowsPerFile);
  const auto stream = cudf::get_default_stream();
  for (size_t file = 0; file < paths.size(); ++file) {
    std::vector<int64_t> values(kRowsPerFile);
    std::iota(
        values.begin(),
        values.end(),
        static_cast<int64_t>(file * kRowsPerFile));
    expected.insert(expected.end(), values.begin(), values.end());
    auto input = makeRowVector({makeFlatVector<int64_t>(values)});
    auto table = with_arrow::toCudfTable(
        input, input->pool(), stream, cudf::get_current_device_resource_ref());
    cudf::io::table_input_metadata metadata(table->view());
    metadata.column_metadata[0].set_name("c0");
    auto writerOptions =
        cudf::io::parquet_writer_options::builder(
            cudf::io::sink_info{paths[file]->getPath()}, table->view())
            .metadata(std::move(metadata))
            .row_group_size_rows(kRowsPerGroup)
            .max_page_size_rows(kRowsPerGroup / 4)
            .compression(cudf::io::compression_type::SNAPPY)
            .build();
    cudf::io::write_parquet(writerOptions, stream);
  }
  stream.synchronize();

  std::array<std::shared_ptr<HiveIcebergSplit>, 4> descriptors;
  for (size_t file = 0; file < paths.size(); ++file) {
    auto splits = makeIcebergSplits(paths[file]->getPath());
    descriptors[file] =
        std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
    VELOX_CHECK_NOT_NULL(descriptors[file]);
  }

  const auto makeLegacySplit = [&](size_t primary, size_t secondary) {
    return std::make_shared<HiveIcebergSplit>(
        kCudfIcebergConnectorId,
        paths[primary]->getPath(),
        dwio::common::FileFormat::PARQUET,
        0,
        getFileSize(paths[primary]->getPath()),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        true,
        std::vector<IcebergDeleteFile>{},
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        0,
        std::vector<IcebergCoalescedFile>{
            {paths[secondary]->getPath(),
             getFileSize(paths[secondary]->getPath())}});
  };

  const auto modeName = [](Mode mode) {
    switch (mode) {
      case Mode::kLegacy:
        return "legacy";
      case Mode::kFileRegular:
        return "file_regular";
      case Mode::kRowGroupHybrid:
        return "row_group_hybrid";
    }
    VELOX_UNREACHABLE();
  };

  const auto run = [&](Mode mode, size_t repetition, bool measured) {
    std::shared_ptr<CudfPreparedSplitSource> source;
    std::shared_ptr<HostRangeAdmission> admission;
    CursorParameters params;
    params.planNode = makeTableScanPlan(rowType);
    params.maxDrivers = 1;
    params.queryConfigs = {{core::QueryConfig::kMaxSplitPreloadPerTask, "12"}};
    params.queryCtx =
        core::QueryCtx::Builder()
            .executor(driverExecutor_.get())
            .queryId(
                fmt::format("p4-uniform-{}-{}", modeName(mode), repetition))
            .build();
    params.queryCtx->setConnectorSessionOverridesUnsafe(
        kCudfIcebergConnectorId,
        {{cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInputSession,
          "true"}});
    params.beforeTaskStart = [&](Task& task) {
      if (mode == Mode::kLegacy) {
        task.addSplit("0", Split(makeLegacySplit(0, 1)));
        task.addSplit("0", Split(makeLegacySplit(2, 3)));
        task.noMoreSplits("0");
        return;
      }
      admission = std::make_shared<HostRangeAdmission>(256 << 20, 12);
      source = std::make_shared<CudfPreparedSplitSource>(
          task.shared_from_this(),
          "0",
          task.queryCtx()->executor(),
          admission,
          mode == Mode::kFileRegular ? CudfPreparationMode::kFileRegular
                                     : CudfPreparationMode::kRowGroupHybrid);
      task.setSplitsStore(
          "0", std::make_unique<CudfPreparedSplitsStore>(source, 4, 64 << 20));
      for (size_t file = 0; file < descriptors.size(); ++file) {
        ASSERT_TRUE(source->addDescriptor(file, descriptors[file]));
      }
      source->sealDescriptors();
    };

    const auto start = std::chrono::steady_clock::now();
    auto cursor = TaskCursor::create(params);
    std::optional<std::chrono::steady_clock::time_point> firstOutput;
    std::vector<int64_t> actual;
    actual.reserve(expected.size());
    while (cursor->moveNext()) {
      if (!firstOutput.has_value()) {
        firstOutput = std::chrono::steady_clock::now();
      }
      auto output = cursor->current();
      auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
      for (vector_size_t row = 0; row < output->size(); ++row) {
        actual.push_back(decoded->valueAt(row));
      }
    }
    const auto finish = std::chrono::steady_clock::now();
    ASSERT_TRUE(waitForTaskCompletion(cursor->task().get()));
    ASSERT_TRUE(waitForTaskDriversToFinish(cursor->task().get()));
    ASSERT_TRUE(firstOutput.has_value());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);

    if (measured) {
      const auto firstOutputUs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              *firstOutput - start)
              .count();
      const auto totalUs =
          std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
              .count();
      const auto sourceMetrics = source == nullptr
          ? CudfPreparedSplitSource::Metrics{}
          : source->metrics();
      const auto admissionMetrics = admission == nullptr
          ? HostRangeAdmission::Snapshot{}
          : admission->snapshot();
      const auto planStats = toPlanStats(cursor->task()->taskStats());
      const auto& customStats = planStats.at("0").customStats;
      const auto metric = [&](std::string_view name) {
        const auto it = customStats.find(std::string{name});
        return it == customStats.end() ? int64_t{0} : it->second.sum;
      };
      std::cout << "P4_LOCAL_MODE_METRIC fixture=uniform mode="
                << modeName(mode) << " repetition=" << repetition
                << " firstOutputUs=" << firstOutputUs << " totalUs=" << totalUs
                << " rows=" << actual.size()
                << " generatedBatches=" << sourceMetrics.preparedBatches
                << " preparedUnits=" << sourceMetrics.preparedUnits
                << " fallbackCount="
                << sourceMetrics.fallbackIneligible +
              sourceMetrics.fallbackPreparationUnavailable +
              sourceMetrics.fallbackPreloadDisabled
                << " payloadBytes=" << sourceMetrics.preparedPayloadBytes
                << " hostPeakBytes=" << admissionMetrics.peakBytes
                << " hostPeakUnits=" << admissionMetrics.peakUnits
                << " h2dCopies=" << metric("cudfPreparedHybridH2dCopies")
                << " h2dBytes=" << metric("cudfPreparedHybridH2dBytes")
                << " leaseReadCalls=" << metric("cudfPreparedLeaseReadCalls")
                << " leaseReadBytes=" << metric("cudfPreparedLeaseReadBytes")
                << std::endl;
    }
  };

  for (const auto mode :
       {Mode::kLegacy, Mode::kFileRegular, Mode::kRowGroupHybrid}) {
    run(mode, 0, false);
  }
  const std::array<Mode, 3> modes{
      Mode::kLegacy, Mode::kFileRegular, Mode::kRowGroupHybrid};
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    for (size_t offset = 0; offset < modes.size(); ++offset) {
      run(modes[(repetition + offset) % modes.size()], repetition, true);
    }
  }
}

TEST_F(CudfIcebergReadTest, delayedInputLocalModeComparison) {
  using cudf_velox::connector::hive::planParquetFileRanges;
  using cudf_velox::connector::hive::iceberg::CudfPreparationMode;
  using cudf_velox::connector::hive::iceberg::CudfPreparedIcebergFile;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitSource;
  using cudf_velox::connector::hive::iceberg::CudfPreparedSplitsStore;

  enum class Mode { kLegacy, kFileRegular, kRowGroupHybrid };
  constexpr size_t kRepetitions = 4;
  auto rowType = ROW({"c0"}, {BIGINT()});
  std::array<std::shared_ptr<TempFilePath>, 4> paths{
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create(),
      TempFilePath::create()};
  std::array<std::shared_ptr<HiveIcebergSplit>, 4> descriptors;
  std::array<std::shared_ptr<const cudf::io::parquet::FileMetaData>, 4>
      metadatas;
  std::array<uint64_t, 4> preparedBytes{};
  std::vector<int64_t> expected;
  auto options = cudf::io::parquet_reader_options::builder().build();
  options.set_column_names({"c0"});
  for (size_t file = 0; file < paths.size(); ++file) {
    auto data = makeRowVector({makeFlatVector<int64_t>(
        {static_cast<int64_t>(file * 10),
         static_cast<int64_t>(file * 10 + 1),
         static_cast<int64_t>(file * 10 + 2)})});
    writeToFile(paths[file]->getPath(), data);
    for (vector_size_t row = 0; row < data->size(); ++row) {
      expected.push_back(
          data->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(row));
    }
    auto splits = makeIcebergSplits(paths[file]->getPath());
    descriptors[file] =
        std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
    VELOX_CHECK_NOT_NULL(descriptors[file]);
    auto dataSource =
        std::make_shared<OwnedFileDataSource>(paths[file]->getPath());
    auto plan =
        planParquetFileRanges(dataSource, options, cudf::get_default_stream());
    ASSERT_EQ(plan.rowGroups.size(), 1);
    metadatas[file] = std::make_shared<const cudf::io::parquet::FileMetaData>(
        std::move(plan.metadata));
    preparedBytes[file] = plan.rowGroups.front().logicalBytes;
  }

  const auto modeName = [](Mode mode) {
    switch (mode) {
      case Mode::kLegacy:
        return "legacy";
      case Mode::kFileRegular:
        return "file_regular";
      case Mode::kRowGroupHybrid:
        return "row_group_hybrid";
    }
    VELOX_UNREACHABLE();
  };
  const auto makeLegacySplit = [&](size_t primary, size_t secondary) {
    return std::make_shared<HiveIcebergSplit>(
        kCudfIcebergConnectorId,
        paths[primary]->getPath(),
        dwio::common::FileFormat::PARQUET,
        0,
        getFileSize(paths[primary]->getPath()),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        true,
        std::vector<IcebergDeleteFile>{},
        std::unordered_map<std::string, std::string>{},
        std::nullopt,
        0,
        std::vector<IcebergCoalescedFile>{
            {paths[secondary]->getPath(),
             getFileSize(paths[secondary]->getPath())}});
  };
  const auto makePrepared = [&](size_t file, Mode mode) {
    auto dataSource =
        std::make_shared<OwnedFileDataSource>(paths[file]->getPath());
    if (mode == Mode::kRowGroupHybrid) {
      return CudfPreparedIcebergFile{
          file,
          descriptors[file],
          std::move(dataSource),
          getFileSize(paths[file]->getPath()),
          metadatas[file],
          cudf::size_type{0},
          preparedBytes[file]};
    }
    return CudfPreparedIcebergFile{
        file,
        descriptors[file],
        std::move(dataSource),
        getFileSize(paths[file]->getPath())};
  };

  const auto run = [&](Mode mode, size_t repetition) {
    std::shared_ptr<Task> task;
    std::shared_ptr<CudfPreparedSplitSource> source;
    CursorParameters params;
    params.planNode = makeTableScanPlan(rowType);
    params.maxDrivers = 1;
    params.queryCtx =
        core::QueryCtx::Builder()
            .executor(driverExecutor_.get())
            .queryId(
                fmt::format("p4-delayed-{}-{}", modeName(mode), repetition))
            .build();
    params.beforeTaskStart = [&](Task& value) {
      task = value.shared_from_this();
      if (mode == Mode::kLegacy) {
        return;
      }
      source = std::make_shared<CudfPreparedSplitSource>(
          task,
          "0",
          value.queryCtx()->executor(),
          nullptr,
          mode == Mode::kFileRegular ? CudfPreparationMode::kFileRegular
                                     : CudfPreparationMode::kRowGroupHybrid);
      value.setSplitsStore(
          "0", std::make_unique<CudfPreparedSplitsStore>(source, 2, 1 << 20));
      ASSERT_TRUE(source->publishPrepared(makePrepared(0, mode)));
      ASSERT_TRUE(source->publishPrepared(makePrepared(2, mode)));
    };

    auto cursor = TaskCursor::create(params);
    std::atomic_bool delayedPublished{false};
    const auto start = std::chrono::steady_clock::now();
    std::thread producer([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      delayedPublished.store(true, std::memory_order_release);
      if (mode == Mode::kLegacy) {
        task->addSplit("0", Split(makeLegacySplit(0, 1)));
        task->addSplit("0", Split(makeLegacySplit(2, 3)));
        task->noMoreSplits("0");
      } else {
        VELOX_CHECK(source->publishPrepared(makePrepared(1, mode)));
        VELOX_CHECK(source->publishPrepared(makePrepared(3, mode)));
        source->finish();
      }
    });
    std::optional<std::chrono::steady_clock::time_point> firstOutput;
    bool firstOutputBeforeDelayed{false};
    std::vector<int64_t> actual;
    while (cursor->moveNext()) {
      if (!firstOutput.has_value()) {
        firstOutput = std::chrono::steady_clock::now();
        firstOutputBeforeDelayed =
            !delayedPublished.load(std::memory_order_acquire);
      }
      auto output = cursor->current();
      auto decoded = output->childAt(0)->as<SimpleVector<int64_t>>();
      for (vector_size_t row = 0; row < output->size(); ++row) {
        actual.push_back(decoded->valueAt(row));
      }
    }
    const auto finish = std::chrono::steady_clock::now();
    producer.join();
    ASSERT_TRUE(firstOutput.has_value());
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
    const auto firstOutputUs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            *firstOutput - start)
            .count();
    const auto totalUs =
        std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
            .count();
    std::cout << "P4_LOCAL_MODE_METRIC fixture=delayed_scheduler mode="
              << modeName(mode) << " repetition=" << repetition
              << " firstOutputUs=" << firstOutputUs << " totalUs=" << totalUs
              << " rows=" << actual.size()
              << " firstOutputBeforeDelayed=" << firstOutputBeforeDelayed
              << std::endl;
  };

  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    run(Mode::kLegacy, repetition);
    run(Mode::kFileRegular, repetition);
    run(Mode::kRowGroupHybrid, repetition);
  }
}

TEST_F(CudfIcebergReadTest, coalescedMultipleFiles) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits{std::make_shared<HiveIcebergSplit>(
          kCudfIcebergConnectorId,
          filePath1->getPath(),
          dwio::common::FileFormat::PARQUET,
          0,
          getFileSize(filePath1->getPath()),
          std::unordered_map<std::string, std::optional<std::string>>{},
          std::nullopt,
          std::unordered_map<std::string, std::string>{},
          nullptr,
          true,
          std::vector<IcebergDeleteFile>{},
          std::unordered_map<std::string, std::string>{},
          std::nullopt,
          0,
          std::vector<IcebergCoalescedFile>{
              {filePath2->getPath(), getFileSize(filePath2->getPath())}})};

  createDuckDbTable({data1, data2});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp", 0);
}

TEST_F(CudfIcebergReadTest, coalescedProjectedCachePrefetch) {
  using namespace cudf_velox::connector::hive;
  // Exercise the S3 cache-prefetch path with local file handles, without an
  // S3 service or credentials. Distinct file identities and contents catch
  // accidentally reusing the primary file's plan or data for another member.
  auto cache = std::make_unique<FileHandleCache>(1024);
  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<CudfCoalescedFile> sources;
  uint64_t totalFileBytes = 0;
  for (int i = 0; i < 4; ++i) {
    auto file = TempFilePath::create();
    auto data = makeRowVector(
        {"c0", "unused"},
        {makeFlatVector<int64_t>(3 + i, [i](auto row) { return 10 * i + row; }),
         makeFlatVector<std::string>(
             3 + i, [](auto row) { return std::string(1024 + row, 'x'); })});
    writeToFile(file->getPath(), data);
    const auto path = "s3://cache-prefetch-test" + file->getPath();
    auto handle = std::make_unique<FileHandle>(FileHandle{
        filesystems::getFileSystem(file->getPath(), nullptr)
            ->openFileForRead(file->getPath()),
        StringIdLease(fileIds(), path),
        StringIdLease(fileIds(), "cache-prefetch-test")});
    sources.push_back({path, handle->file->size()});
    totalFileBytes += sources.back().length;
    ASSERT_TRUE(cache->add(FileHandleKey{.filename = path}, handle.get(), 1));
    handle.release();
    files.push_back(std::move(file));
  }
  FileHandleFactory factory(
      std::move(cache), std::make_unique<FileHandleGenerator>());
  auto outputType = ROW({"c0"}, {BIGINT()});
  auto table =
      std::make_shared<::facebook::velox::connector::hive::HiveTableHandle>(
          kCudfIcebergConnectorId,
          "prefetch",
          common::SubfieldFilters{},
          nullptr,
          outputType);

  std::unordered_map<std::string, std::optional<std::string>> savedEnv;
  auto setEnv = [&](const std::string& key, const std::string& value) {
    if (!savedEnv.count(key)) {
      auto* previous = std::getenv(key.c_str());
      savedEnv[key] =
          previous ? std::optional<std::string>(previous) : std::nullopt;
    }
    setenv(key.c_str(), value.c_str(), 1);
  };
  SCOPE_EXIT {
    for (const auto& [key, value] : savedEnv) {
      if (value) {
        setenv(key.c_str(), value->c_str(), 1);
      } else {
        unsetenv(key.c_str());
      }
    }
  };
  setEnv("GLUTEN_CPP_S3_CRT", "0");
  setEnv("GLUTEN_CPP_S3_AWS_SDK", "0");
  setEnv("GLUTEN_CUDF_S3_ADAPTIVE_PREFETCH", "0");
  setEnv("GLUTEN_CUDF_CACHE_HINT_PREPARE_LOADS", "1");
  setEnv("GLUTEN_CPP_S3_ASYNC_CACHE_READ_GROUPS", "1");
  setEnv(
      "GLUTEN_CUDF_CACHE_HINT_FIRST_LOAD_GROUP_READY_MIN_QUERY_REGISTERED_SPLITS",
      "2");

  // The one-byte window is smaller than every source; demand must still
  // progress for a group larger than the window. Also cover single-source
  // reads and both preparation-time and scan-time readiness waits.
  for (const auto fileCount : {1, 4}) {
    for (const auto budget : {1, 1024 * 1024}) {
      for (const bool preload : {false, true}) {
        // Full readiness, first-load readiness, and nonblocking range demand.
        for (const auto mode : {0, 1, 2}) {
          asyncDataCache_->clear();
          SCOPED_TRACE(
              fmt::format(
                  "files={} budget={} preload={} mode={}",
                  fileCount,
                  budget,
                  preload,
                  mode));
          setEnv(
              "GLUTEN_CUDF_CACHE_HINT_READY_SPLIT_PRELOAD",
              preload ? "1" : "0");
          setEnv(
              "GLUTEN_CUDF_CACHE_HINT_FIRST_LOAD_GROUP_READY_SCAN",
              mode != 0 ? "1" : "0");
          auto config = std::make_shared<config::ConfigBase>(
              std::unordered_map<std::string, std::string>{
                  {CudfHiveConfig::kUseBufferedInputSession, "true"},
                  {CudfHiveConfig::kPrefetchMaxInFlightBytesSession,
                   std::to_string(budget)},
                  {CudfHiveConfig::kExecutorSplitPrefetchConcurrencySession,
                   "4"}});
          const auto queryId = fmt::format(
              "{}-{}-{}-{}-{}",
              sources[0].filePath,
              fileCount,
              budget,
              preload,
              mode);
          ::facebook::velox::connector::ConnectorQueryCtx context(
              pool_.get(),
              rootPool_.get(),
              config.get(),
              nullptr,
              {},
              nullptr,
              asyncDataCache_.get(),
              queryId,
              "task",
              "scan",
              0,
              "");
          SCOPE_EXIT {
            ExecutorSplitPrefetch::eraseQuery(ioExecutor_.get(), queryId);
          };
          ExecutorSplitPrefetch::setExpectedSplitCount(
              ioExecutor_.get(), queryId, mode == 1 ? 2 : 1, 2);
          auto split = std::make_shared<CudfHiveConnectorSplit>(
              kCudfIcebergConnectorId,
              sources[0].filePath,
              // Iceberg whole-file tasks exclude the four-byte PAR1 header.
              4,
              sources[0].length - 4,
              0,
              std::unordered_map<std::string, std::string>{},
              std::vector<CudfCoalescedFile>(
                  sources.begin() + 1, sources.begin() + fileCount));
          const auto before =
              ExecutorSplitPrefetch::cacheHintRangeStatsForTest();
          const auto waitsBefore =
              ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
          CudfSplitReader reader(
              split,
              table,
              outputType,
              {"c0"},
              &factory,
              ioExecutor_.get(),
              &context,
              std::make_shared<CudfHiveConfig>(config),
              std::make_shared<io::IoStatistics>(),
              std::make_shared<IoStats>(),
              false,
              nullptr);
          dwio::common::RuntimeStatistics runtimeStats;
          reader.prepareSplit(runtimeStats);
          auto chunk = reader.next(0);
          ASSERT_TRUE(chunk.has_value());
          std::vector<int64_t> expected;
          for (int i = 0; i < fileCount; ++i) {
            for (int row = 0; row < 3 + i; ++row) {
              expected.push_back(10 * i + row);
            }
          }
          auto output = with_arrow::toVeloxColumn(
              chunk.value()->view(),
              pool_.get(),
              outputType,
              reader.stream(),
              cudf::get_current_device_resource_ref());
          reader.stream().synchronize();
          ::facebook::velox::test::assertEqualVectors(
              makeRowVector({makeFlatVector<int64_t>(expected)}), output);
          // All four files fit in one decode/output chunk, not four batches.
          EXPECT_FALSE(reader.next(0).has_value());
          const auto after =
              ExecutorSplitPrefetch::cacheHintRangeStatsForTest();
          EXPECT_EQ(after.logicalRanges - before.logicalRanges, fileCount);
          EXPECT_GT(after.logicalBytes - before.logicalBytes, 0);
          EXPECT_LT(after.logicalBytes - before.logicalBytes, totalFileBytes);
          EXPECT_EQ(
              after.consumedRanges - before.consumedRanges,
              after.uniqueRanges - before.uniqueRanges);
          const auto waits = ExecutorSplitPrefetch::cacheHintWaitStatsForTest();
          if (mode == 2) {
            EXPECT_EQ(
                waits.rangeReadyRequests - waitsBefore.rangeReadyRequests,
                fileCount);
            EXPECT_EQ(
                waits.rangeReadyReleases - waitsBefore.rangeReadyReleases,
                fileCount);
          } else {
            EXPECT_EQ(
                mode == 1 ? waits.firstLoadGroupWaits -
                        waitsBefore.firstLoadGroupWaits
                    : preload
                    ? waits.splitPreloadWaits - waitsBefore.splitPreloadWaits
                    : waits.scanWaits - waitsBefore.scanWaits,
                fileCount);
          }
        }
      }
    }
  }
}

TEST_F(CudfIcebergReadTest, coalescedSchemaEvolution) {
  auto oldData = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2})});
  auto newData = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({3, 4}), makeFlatVector<int64_t>({30, 40})});
  auto oldFile = TempFilePath::create();
  auto newFile = TempFilePath::create();
  writeToFile(oldFile->getPath(), oldData);
  writeToFile(newFile->getPath(), newData);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits{std::make_shared<HiveIcebergSplit>(
          kCudfIcebergConnectorId,
          oldFile->getPath(),
          dwio::common::FileFormat::PARQUET,
          0,
          getFileSize(oldFile->getPath()),
          std::unordered_map<std::string, std::optional<std::string>>{},
          std::nullopt,
          std::unordered_map<std::string, std::string>{},
          nullptr,
          true,
          std::vector<IcebergDeleteFile>{},
          std::unordered_map<std::string, std::string>{},
          std::nullopt,
          0,
          std::vector<IcebergCoalescedFile>{
              {newFile->getPath(), getFileSize(newFile->getPath())}})};

  auto outputType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto expected = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 30, 40})});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(outputType)
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, coalescedAllSchemaEvolutionColumns) {
  auto data1 =
      makeRowVector({"old_col"}, {makeFlatVector<int64_t>({100, 200, 300})});
  auto data2 =
      makeRowVector({"old_col"}, {makeFlatVector<int64_t>({400, 500, 600})});
  auto file1 = TempFilePath::create();
  auto file2 = TempFilePath::create();
  writeToFile(file1->getPath(), data1);
  writeToFile(file2->getPath(), data2);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      splits{std::make_shared<HiveIcebergSplit>(
          kCudfIcebergConnectorId,
          file1->getPath(),
          dwio::common::FileFormat::PARQUET,
          0,
          getFileSize(file1->getPath()),
          std::unordered_map<std::string, std::optional<std::string>>{},
          std::nullopt,
          std::unordered_map<std::string, std::string>{},
          nullptr,
          true,
          std::vector<IcebergDeleteFile>{},
          std::unordered_map<std::string, std::string>{},
          std::nullopt,
          0,
          std::vector<IcebergCoalescedFile>{
              {file2->getPath(), getFileSize(file2->getPath())}})};

  auto outputType = ROW({"new_col"}, {BIGINT()});
  auto expected =
      makeRowVector({"new_col"}, {makeNullConstant(TypeKind::BIGINT, 6)});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(outputType)
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read a single data file as multiple byte-range sub-splits with no deletes
TEST_F(CudfIcebergReadTest, subSplitsNoDeletes) {
  auto rowType = ROW({"c0"}, {BIGINT()});

  // Two separate write() calls produce two row groups.
  auto rowGroup0 = makeRowVector(
      {makeFlatVector<int64_t>(makeContinuousIncreasingValues(0, 10000))});
  auto rowGroup1 = makeRowVector(
      {makeFlatVector<int64_t>(makeContinuousIncreasingValues(10000, 20000))});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {rowGroup0, rowGroup1});
  createDuckDbTable({rowGroup0, rowGroup1});

  auto splits =
      makeIcebergSplits(filePath->getPath(), {}, {}, /*splitCount=*/2);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp", 0);
}

/// All  missing (schema evolution) columns
TEST_F(CudfIcebergReadTest, allSchemaEvolutionColumns) {
  auto dataVector = makeRowVector(
      {"old_col"},
      {
          makeFlatVector<int64_t>({100, 200, 300}),
      });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  auto newRowType = ROW({"new_col"}, {BIGINT()});
  auto expected = makeRowVector(
      {"new_col"},
      {
          makeNullConstant(TypeKind::BIGINT, 3),
      });

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .dataColumns(newRowType)
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({expected});

  // Filtering a schema-evolution column that is NULL for every row in an old
  // file must return an empty result (Iceberg NULL semantics).
  auto filteredPlan = PlanBuilder()
                          .startTableScan()
                          .connectorId(kCudfIcebergConnectorId)
                          .outputType(newRowType)
                          .dataColumns(newRowType)
                          .subfieldFilter("new_col = 1")
                          .endTableScan()
                          .planNode();

  auto filteredEmptyExpected = makeRowVector(
      {"new_col"},
      {
          makeFlatVector<int64_t>(std::vector<int64_t>{}),
      });

  AssertQueryBuilder(filteredPlan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({filteredEmptyExpected});

  // Filter-only evolution col (not in output) with a projected file column.
  auto mixedType = ROW({"old_col", "new_col"}, {BIGINT(), BIGINT()});
  auto fileOnlyOutput = ROW({"old_col"}, {BIGINT()});
  auto filterOnlyPlan = PlanBuilder()
                            .startTableScan()
                            .connectorId(kCudfIcebergConnectorId)
                            .outputType(fileOnlyOutput)
                            .dataColumns(mixedType)
                            .subfieldFilter("new_col = 1")
                            .endTableScan()
                            .planNode();

  AssertQueryBuilder(filterOnlyPlan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({makeRowVector(
          {"old_col"}, {makeFlatVector<int64_t>(std::vector<int64_t>{})})});
}

/// Column alias: output column `a` -> physical file column `c0`
TEST_F(CudfIcebergReadTest, columnAliasUsesPhysicalFileName) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto fileType = ROW({"c0"}, {BIGINT()});
  auto outputType = ROW({"a"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(fileType)
                  .columnAliases({{"a", "c0"}})
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath()))
      .assertResults({expected});
}

/// A nonempty data file in a NULL partition must return one NULL partition
/// value per row
TEST_F(CudfIcebergReadTest, nullPartitionColumn) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", std::nullopt}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"c0", "country"},
      {
          data->childAt(0),
          makeNullConstant(TypeKind::VARCHAR, 3),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});
}

/// A projection containing only injected columns. e.g., partition column, leads
/// to no columns being read from the data file and the output is synthesized
/// from the injected columns only
TEST_F(CudfIcebergReadTest, partitionOnlyProjection) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"country"},
      {
          makeFlatVector<std::string>({"US", "US", "US"}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});

  // Filter `country = 'CA'` excludes the partition value 'US', so the output is
  // an empty table. Reader is bypassed due to no physical columns to read.
  auto filteredNoMatchPlan = PlanBuilder()
                                 .startTableScan()
                                 .connectorId(kCudfIcebergConnectorId)
                                 .outputType(outputType)
                                 .dataColumns(tableType)
                                 .assignments(assignments)
                                 .subfieldFilter("country = 'CA'")
                                 .endTableScan()
                                 .planNode();

  auto emptyExpected =
      makeRowVector({"country"}, {makeFlatVector<std::string>({})});

  AssertQueryBuilder(filteredNoMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({emptyExpected});

  // A matching subfield filter keeps all rows.
  auto filteredMatchPlan = PlanBuilder()
                               .startTableScan()
                               .connectorId(kCudfIcebergConnectorId)
                               .outputType(outputType)
                               .dataColumns(tableType)
                               .assignments(assignments)
                               .subfieldFilter("country = 'US'")
                               .endTableScan()
                               .planNode();

  AssertQueryBuilder(filteredMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});

  // Mixed physical + injected projection with a non-matching filter on the
  // injected partition column must return no rows. The filter cannot be pushed
  // into the parquet reader because `country` is not a file column.
  auto mixedAssignments = assignments;
  mixedAssignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});

  auto mixedPlan = PlanBuilder()
                       .startTableScan()
                       .connectorId(kCudfIcebergConnectorId)
                       .outputType(tableType)
                       .dataColumns(tableType)
                       .assignments(mixedAssignments)
                       .subfieldFilter("country = 'CA'")
                       .endTableScan()
                       .planNode();

  auto mixedEmptyExpected = makeRowVector(
      {"c0", "country"},
      {
          makeFlatVector<int64_t>(std::vector<int64_t>{}),
          makeFlatVector<std::string>(std::vector<std::string>{}),
      });

  AssertQueryBuilder(mixedPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({mixedEmptyExpected});

  // Matching filter on the injected column with a physical column present.
  auto mixedMatchPlan = PlanBuilder()
                            .startTableScan()
                            .connectorId(kCudfIcebergConnectorId)
                            .outputType(tableType)
                            .dataColumns(tableType)
                            .assignments(mixedAssignments)
                            .subfieldFilter("country = 'US'")
                            .endTableScan()
                            .planNode();

  auto mixedMatchExpected = makeRowVector(
      {"c0", "country"},
      {
          data->childAt(0),
          makeFlatVector<std::string>({"US", "US", "US"}),
      });

  AssertQueryBuilder(mixedMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({mixedMatchExpected});

  // Filter-only partition (not in output): inject, filter, then strip.
  auto fileOnlyAssignments = facebook::velox::connector::ColumnHandleMap{};
  fileOnlyAssignments["c0"] = mixedAssignments["c0"];
  auto fileOnlyOutput = ROW({"c0"}, {BIGINT()});

  auto filterOnlyMatchPlan = PlanBuilder()
                                 .startTableScan()
                                 .connectorId(kCudfIcebergConnectorId)
                                 .outputType(fileOnlyOutput)
                                 .dataColumns(tableType)
                                 .assignments(fileOnlyAssignments)
                                 .subfieldFilter("country = 'US'")
                                 .endTableScan()
                                 .planNode();

  AssertQueryBuilder(filterOnlyMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({makeRowVector({"c0"}, {data->childAt(0)})});

  auto filterOnlyNoMatchPlan = PlanBuilder()
                                   .startTableScan()
                                   .connectorId(kCudfIcebergConnectorId)
                                   .outputType(fileOnlyOutput)
                                   .dataColumns(tableType)
                                   .assignments(fileOnlyAssignments)
                                   .subfieldFilter("country = 'CA'")
                                   .endTableScan()
                                   .planNode();

  AssertQueryBuilder(filterOnlyNoMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({makeRowVector(
          {"c0"}, {makeFlatVector<int64_t>(std::vector<int64_t>{})})});

  // Experimental hybrid reader must also support injected-only projections.
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kCudfIcebergConnectorId,
          cudf_velox::connector::hive::CudfHiveConfig::
              kUseExperimentalCudfReaderSession,
          "true")
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});
}

/// All-injected projection with positional deletes and a deferred (since no
/// physical columns) subfield filter.
TEST_F(CudfIcebergReadTest, allInjectedProjectionWithPositionalDeletes) {
  auto data =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30, 40, 50})});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Positional delete removing file rows 0 and 2.
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteVector = makeRowVector(
      {pathColumn->name, posColumn->name},
      {
          makeFlatVector<std::string>(
              2, [&](vector_size_t) { return dataFile->getPath(); }),
          makeFlatVector<int64_t>({0, 2}),
      });
  writeDeleteFile(
      DeleteFileFormat::DWRF, deleteFilePath->getPath(), {deleteVector});
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()));

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});
  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .subfieldFilter("country = 'US'")
                  .endTableScan()
                  .planNode();
  auto expected = makeRowVector(
      {"country"}, {makeFlatVector<std::string>({"US", "US", "US"})});
  AssertQueryBuilder(plan)
      .splits(
          makeIcebergSplits(dataFile->getPath(), {deleteFile}, partitionKeys))
      .assertResults({expected});

  // Read a physical column with prepended row indices. Filter is still deferred
  // but row indices are contiguous.
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  auto mixedPlan = PlanBuilder()
                       .startTableScan()
                       .connectorId(kCudfIcebergConnectorId)
                       .outputType(tableType)
                       .dataColumns(tableType)
                       .assignments(assignments)
                       .subfieldFilter("country = 'US'")
                       .endTableScan()
                       .planNode();
  auto mixedExpected = makeRowVector(
      {"c0", "country"},
      {
          makeFlatVector<int64_t>({20, 40, 50}),
          makeFlatVector<std::string>({"US", "US", "US"}),
      });
  AssertQueryBuilder(mixedPlan)
      .splits(
          makeIcebergSplits(dataFile->getPath(), {deleteFile}, partitionKeys))
      .assertResults({mixedExpected});
}

/// Verifies a deletion vector with an injected-only projection.
TEST_F(CudfIcebergReadTest, deletionVectorWithInjectedOnlyProjection) {
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30, 40, 50})}));

  auto bitmapData = serializeRoaringBitmapNoRun<int64_t>({0, 2});
  auto dvFile = writeDvFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(dvFile->getPath(), bitmapData.size(), 2);

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});
  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();
  auto expected = makeRowVector(
      {"country"}, {makeFlatVector<std::string>({"US", "US", "US"})});
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {dvDelete}, partitionKeys))
      .assertResults({expected});
}

/// Verifies sub-splits over an all-injected projection.
TEST_F(CudfIcebergReadTest, subSplitAllInjectedProjection) {
  auto firstRowGroup = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>(10'000, [](vector_size_t row) {
        return (static_cast<int64_t>(row) * 1'000'003) % 10'000'019;
      })});
  auto secondRowGroup = makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>(
          makeContinuousIncreasingValues(10'000, 10'001))});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), {firstRowGroup, secondRowGroup});

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});
  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  // Without deletes.
  auto expected = makeRowVector(
      {"country"}, {makeFlatVector<std::string>(10'001, [](vector_size_t) {
        return "US";
      })});
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(
          dataFile->getPath(), {}, partitionKeys, /*splitCount=*/2))
      .assertResults({expected});

  // With positional deletes.
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return dataFile->getPath(); }),
              makeFlatVector<int64_t>({0, 10'000}),
          })});
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()));

  auto expectedWithDeletes = makeRowVector(
      {"country"},
      {makeFlatVector<std::string>(9'999, [](vector_size_t) { return "US"; })});
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(
          dataFile->getPath(),
          {deleteFile},
          partitionKeys,
          /*splitCount=*/2))
      .assertResults({expectedWithDeletes});
}

/// A remaining filter on an unprojected column (`c1`) adds `c1` to the
/// projection for post-scan filtering and removal
TEST_F(CudfIcebergReadTest, unprojectedRemainingFilterColumn) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto dataType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(dataType)
                  .remainingFilter("c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({20, 40}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath()))
      .assertResults({expected});
}

/// Remaining filter referencing an injected filter-only partition column and an
/// unprojected file column.
TEST_F(CudfIcebergReadTest, remainingFilterOnInjectedColumn) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType =
      ROW({"c0", "c1", "country"}, {BIGINT(), BIGINT(), VARCHAR()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["c1"] = std::make_shared<HiveColumnHandle>(
      "c1",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .remainingFilter("country = 'CA' OR c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  // Partition 'US': `country = 'CA'` is false, so the filter reduces to
  // `c1 % 2 = 0`, keeping c1={2,4} -> c0={20,40}.
  std::unordered_map<std::string, std::optional<std::string>> usPartition = {
      {"country", "US"}};
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, usPartition))
      .assertResults(
          {makeRowVector({"c0"}, {makeFlatVector<int64_t>({20, 40})})});

  // Partition 'CA': `country = 'CA'` is true for every row, so all rows pass.
  std::unordered_map<std::string, std::optional<std::string>> caPartition = {
      {"country", "CA"}};
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, caPartition))
      .assertResults(
          {makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30, 40})})});
}

// A timezone-less TIMESTAMP partition value is
// interpreted in the default timezone and converted to UTC when the Hive config
// is true.
TEST_F(CudfIcebergReadTest, timestampPartitionHonorsLocalTimeSetting) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "pkey"}, {BIGINT(), TIMESTAMP()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["pkey"] = std::make_shared<HiveColumnHandle>(
      "pkey",
      HiveColumnHandle::ColumnType::kPartitionKey,
      TIMESTAMP(),
      TIMESTAMP(),
      std::vector<common::Subfield>{});

  const std::string partitionValue = "2023-10-14 07:00:00";
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"pkey", partitionValue}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto utcTimestamp =
      util::fromTimestampString(
          StringView(partitionValue), util::TimestampParseMode::kPrestoCast)
          .value();

  auto localTimestamp = utcTimestamp;
  localTimestamp.toGMT(Timestamp::defaultTimezone());

  auto assertValue = [&](bool readAsLocalTime, Timestamp expectedTimestamp) {
    auto expected = makeRowVector(
        {"c0", "pkey"},
        {
            data->childAt(0),
            makeFlatVector<Timestamp>(
                {expectedTimestamp, expectedTimestamp, expectedTimestamp}),
        });

    AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kCudfIcebergConnectorId,
            facebook::velox::connector::hive::HiveConfig::
                kReadTimestampPartitionValueAsLocalTimeSession,
            readAsLocalTime ? "true" : "false")
        .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
        .assertResults({expected});
  };

  // Interpret the partition string as already being in UTC.
  assertValue(false, utcTimestamp);

  // Interpret the partition string as local time and convert it to UTC.
  assertValue(true, localTimestamp);
}

/// A remaining filter is allowed with positional deletes
TEST_F(CudfIcebergReadTest, remainingFilterAndPositionalDeletes) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Positional delete file removing row 1 (c0=20, c1=2).
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteVector = makeRowVector(
      {pathColumn->name, posColumn->name},
      {
          makeFlatVector<std::string>(
              1, [&](vector_size_t) { return dataFile->getPath(); }),
          makeFlatVector<int64_t>({1}),
      });
  writeDeleteFile(
      DeleteFileFormat::DWRF, deleteFilePath->getPath(), {deleteVector});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(deleteFilePath->getPath()));

  auto dataType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(dataType)
                  .remainingFilter("c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  // Row 1 (c1=2) is deleted; of the remaining c1={1,3,4}, `c1 % 2 = 0` keeps
  // c1=4, i.e. c0=40.
  auto expected = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({40}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {deleteFile}))
      .assertResults({expected});
}

// Subfield filter (no injection) and positional deletes. Filter is not
// deferred.
TEST_F(CudfIcebergReadTest, subfieldFilterWithPositionalDeletes) {
  auto data = makeRowVector({makeFlatVector<int64_t>({10, 20, 30, 40})});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Positional delete file removing (c0=10) and (c0=30).
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteVector = makeRowVector(
      {pathColumn->name, posColumn->name},
      {
          makeFlatVector<std::string>(
              2, [&](vector_size_t) { return dataFile->getPath(); }),
          makeFlatVector<int64_t>({0, 2}),
      });
  writeDeleteFile(
      DeleteFileFormat::DWRF, deleteFilePath->getPath(), {deleteVector});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()));

  // The pushed-down filter keeps gapped file rows {0, 2, 3}. Positional
  // deletes remove rows 0 and 2, leaving c0=40.
  auto dataType = ROW({"c0"}, {BIGINT()});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(dataType)
                  .dataColumns(dataType)
                  .subfieldFilter("c0 != 20")
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({40})});

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {deleteFile}))
      .assertResults({expected});
}

TEST_F(CudfIcebergReadTest, singleBaseFileSinglePositionalDeleteFile) {
  assertSingleBaseFileSingleDeleteFile({{0, 1, 2, 3}});
  assertSingleBaseFileSingleDeleteFile({{0, 9999, 10000, 19999}});
  assertSingleBaseFileSingleDeleteFile({{10000, 10002, 19999}});
  assertSingleBaseFileSingleDeleteFile({makeRandomIncreasingValues(0, 20000)});
  assertSingleBaseFileSingleDeleteFile({});
  assertSingleBaseFileSingleDeleteFile(
      {makeContinuousIncreasingValues(0, 20000)});
  assertSingleBaseFileSingleDeleteFile({{20000, 29999}});
}

TEST_F(CudfIcebergReadTest, multipleBaseFilesSinglePositionalDeleteFile) {
  assertMultipleBaseFileSingleDeleteFile({0, 1, 2, 3});
  assertMultipleBaseFileSingleDeleteFile({0, 9999, 10000, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile(
      makeRandomIncreasingValues(0, rowCount));
  assertMultipleBaseFileSingleDeleteFile({});
  assertMultipleBaseFileSingleDeleteFile(
      makeContinuousIncreasingValues(0, rowCount));
}

TEST_F(CudfIcebergReadTest, singleBaseFileMultiplePositionalDeleteFiles) {
  assertSingleBaseFileMultipleDeleteFiles({{1}, {2}, {3}, {4}});
  assertSingleBaseFileMultipleDeleteFiles({{0}, {9999}, {10000}, {19999}});
  assertSingleBaseFileMultipleDeleteFiles({{500, 21000}});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 10000),
       makeRandomIncreasingValues(10000, 20000),
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 10000),
       makeContinuousIncreasingValues(10000, 20000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 10000),
       makeContinuousIncreasingValues(10000, 20000),
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 20000),
       makeContinuousIncreasingValues(0, 20000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 20000),
       {},
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles({{}, {}});
}

TEST_F(CudfIcebergReadTest, multipleBaseFileMultiplePositionalDeleteFiles) {
  std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles;
  std::unordered_map<
      std::string,
      std::multimap<std::string, std::vector<int64_t>>>
      deleteFilesForBaseDatafiles;

  rowGroupSizesForFiles["data_file_1"] = {100, 85};
  rowGroupSizesForFiles["data_file_2"] = {99, 1};

  deleteFilesForBaseDatafiles["delete_file_1"] = {{"data_file_1", {0, 1, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {100, 101, 184}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeContinuousIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(186, 300)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 100, 102, 184}}, {"data_file_2", {1, 98, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}},
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", {1, 3, 5, 7}},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}}, {"data_file_2", {1, 3, 5, 7}}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", {98, 99, 100, 101, 184}},
      {"data_file_2", {3, 5, 7, 9}},
      {"data_file_2", {98, 99, 100}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", makeRandomIncreasingValues(10, 120)},
      {"data_file_2", makeRandomIncreasingValues(50, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
}

TEST_F(CudfIcebergReadTest, positionalDeletesMultipleSplits) {
  assertMultipleSplits({1, 2, 3, 4}, 10, 5);
  assertMultipleSplits({1, 2, 3, 4}, 10, 0);
  assertMultipleSplits({1, 2, 3, 4}, 10, 10);
  assertMultipleSplits({0, 9999, 10000, 19999}, 10, 3);
  assertMultipleSplits(makeRandomIncreasingValues(0, 20000), 10, 3);
  assertMultipleSplits(makeContinuousIncreasingValues(0, 20000), 10, 3);
  assertMultipleSplits({}, 10, 3);

  assertMultipleSplits({1, 2, 3, 4}, 10, 5, 30000, 3);
  assertPositionalDeletes(
      {
          {"data_file_0", {500}},
          {"data_file_1", {10000, 10000}},
          {"data_file_2", {500}},
      },
      {{"delete_file_1",
        {{"data_file_1", makeRandomIncreasingValues(0, 20000)}}}},
      0,
      3);

  assertMultipleSplits({1000, 9000, 20000}, 1, 0, 20000, 3);
}

// Applies positional deletes to a nonzero sub-split by absolute file row.
TEST_F(CudfIcebergReadTest, subSplitPositionalDeletes) {
  // Two row groups: (rows 0..89) and (rows 90..99).
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      {makeRowVector(
           {makeFlatVector<int64_t>(makeContinuousIncreasingValues(0, 90))}),
       makeRowVector({makeFlatVector<int64_t>(
           makeContinuousIncreasingValues(90, 100))})});

  // Rows 0, 1, and 2 are outside the selected split; row 95 is inside it.
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  4, [&](vector_size_t) { return dataFile->getPath(); }),
              makeFlatVector<int64_t>({0, 1, 2, 95}),
          })});
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      4,
      getFileSize(deleteFilePath->getPath()));

  const auto encodePositionBound = [](uint64_t position) {
    const auto littleEndianPosition = folly::Endian::little(position);
    return encoding::Base64::encode(
        std::string_view(
            reinterpret_cast<const char*>(&littleEndianPosition),
            sizeof(littleEndianPosition)));
  };
  auto beforeSplitDeleteFilePath = TempFilePath::create();
  IcebergDeleteFile beforeSplitDeleteFile(
      FileContent::kPositionalDeletes,
      beforeSplitDeleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      0,
      {},
      {},
      {{posColumn->id, encodePositionBound(2)}});
  auto afterSplitDeleteFilePath = TempFilePath::create();
  IcebergDeleteFile afterSplitDeleteFile(
      FileContent::kPositionalDeletes,
      afterSplitDeleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      0,
      {},
      {{posColumn->id, encodePositionBound(100)}},
      {});

  // Use only the second sub-split (start != 0), which reads the trailing row
  // group (file rows 90..99).
  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {deleteFile, beforeSplitDeleteFile, afterSplitDeleteFile},
      {},
      /*splitCount=*/2);
  ASSERT_EQ(splits.size(), 2);
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      secondSplit = {splits.back()};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({90, 91, 92, 93, 94, 96, 97, 98, 99})});
  AssertQueryBuilder(plan).splits(secondSplit).assertResults({expected});
}

// Applies a deletion vector to filtered rows in a nonzero sub-split.
TEST_F(CudfIcebergReadTest, deletionVectorWithFilteredSubSplit) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      {
          makeRowVector({makeFlatVector<int64_t>(
              90, [](vector_size_t row) { return row; })}),
          makeRowVector({makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row + 90; })}),
      });

  auto bitmapData = serializeRoaringBitmapNoRun<int64_t>({95});
  auto dvFile = writeDvFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(dvFile->getPath(), bitmapData.size(), 1);
  auto splits =
      makeIcebergSplits(dataFile->getPath(), {dvDelete}, {}, /*splitCount=*/2);
  ASSERT_EQ(splits.size(), 2);
  std::vector<std::shared_ptr<::facebook::velox::connector::ConnectorSplit>>
      secondSplit = {splits.back()};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .subfieldFilter("c0 != 92")
                  .endTableScan()
                  .planNode();
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({90, 91, 93, 94, 96, 97, 98, 99})});
  AssertQueryBuilder(plan).splits(secondSplit).assertResults({expected});
}

// Verifies positional deletes with a row-group filter.
TEST_F(CudfIcebergReadTest, positionalDeleteFileWithRowGroupFilter) {
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      {
          makeRowVector(
              {"id"},
              {makeFlatVector<int64_t>(
                  makeContinuousIncreasingValues(100, 200))}),
          makeRowVector(
              {"id"},
              {makeFlatVector<int64_t>(
                  makeContinuousIncreasingValues(0, 100))}),
          makeRowVector(
              {"id"},
              {makeFlatVector<int64_t>(
                  makeContinuousIncreasingValues(200, 300))}),
      });
  const auto& path = dataFile->getPath();

  const int32_t deletedPositionSize = 100;
  std::vector<int64_t> deletePositions(deletedPositionSize);
  std::iota(deletePositions.begin(), deletePositions.end(), 100);

  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  deletedPositionSize, [&](vector_size_t) { return path; }),
              makeFlatVector<int64_t>(deletePositions),
          })});
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      deletedPositionSize,
      getFileSize(deleteFilePath->getPath()));

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(ROW({"id"}, {BIGINT()}))
                  .remainingFilter("id >= 100")
                  .endTableScan()
                  .planNode();
  assertQuery(
      plan,
      makeIcebergSplits(path, {deleteFile}),
      "SELECT i AS id FROM range(100, 300) AS t(i)",
      0);
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberApplied) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/10);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/5);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberSkipped) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

// Subfield filter + positional delete skipped by sequence-number resolution:
// no delete readers remain, so the filter can still be pushed into parquet.
TEST_F(CudfIcebergReadTest, subfieldFilterWithSkippedPositionalDelete) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  // Positional delete with dataSequenceNumber (5) < the data file's sequence
  // number (10), so it is skipped by sequence-number resolution.
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .subfieldFilter("c0 < 4")
                  .endTableScan()
                  .planNode();

  // The delete is skipped, so `c0 < 4` simply keeps {0, 1, 2, 3}.
  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

// Subfield filter + positional delete for a different data file.
TEST_F(
    CudfIcebergReadTest,
    subfieldFilterWithPositionalDeleteForDifferentFile) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto baseFilePath = dataFilePath->getPath();
  auto otherDataFilePath = TempFilePath::create();

  // The delete file's positions reference a different data file, so its
  // file_path column statistics exclude `baseFilePath` and `testFilters` prunes
  // the reader during construction.
  auto deleteFilePath = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2,
                  [&](vector_size_t) { return otherDataFilePath->getPath(); }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  // Equal sequence numbers make the positional delete applicable, ensuring it
  // reaches file-path pruning inside `PositionalDeleteFileReader`.
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/10);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .subfieldFilter("c0 < 4")
                  .endTableScan()
                  .planNode();

  // The positional delete file contains no entries for this data file, so
  // `c0 < 4` simply keeps {0, 1, 2, 3}.
  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberEqualApplied) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/5);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberZeroDisablesFilter) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/0);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/100);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Edge-case tests from Sirius project

/// Combined positional + equality deletes on the same data file.
/// Verifies that positional deletes (position-based) and equality deletes
/// (value-based) compose correctly when applied sequentially.
TEST_F(CudfIcebergReadTest, combinedPositionalAndEqualityDeletes) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Positional delete: remove rows at positions 1 and 3 (c0=1, c0=3)
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      2, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({1, 3});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posIcebergDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(posDeleteFile->getPath()));

  // Equality delete: remove rows where c0={5, 7}
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({5, 7}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {posIcebergDelete, eqIcebergDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Positional removes pos 1,3 (c0=1,3). Equality removes c0=5,7.
  // Surviving: 0, 2, 4, 6, 8, 9
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 2, 4, 6, 8, 9}),
      makeFlatVector<int64_t>({10, 12, 14, 16, 18, 19}),
  });

  assertEqualResults({expected}, {result});
}

/// Non-projected equality delete key column.
/// Query selects only c1, but equality delete key is c0.
/// The reader must force-project c0 for filtering then strip it from output.
TEST_F(CudfIcebergReadTest, nonProjectedDeleteKeyColumn) {
  auto fullType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c1"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Equality delete on c0 (NOT in the output projection)
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqIcebergDelete});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2,4 deleted -> c1=20,40 removed
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Insert-delete-insert interleaving: data written after a delete (higher
/// sequence number) should NOT be affected by that delete.
TEST_F(CudfIcebergReadTest, insertDeleteInsertInterleaving) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // File 1: original data (seq=1)
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: data written AFTER the delete (seq=3)
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({200, 400}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // Equality delete at seq=2: delete c0=2
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  // File 1: data seq=1, delete seq=2 -> delete APPLIES (2 > 1)
  auto splits1 = makeIcebergSplits(
      dataFile1->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/1);
  // File 2: data seq=3, delete seq=2 -> delete SKIPPED (2 < 3)
  auto splits2 = makeIcebergSplits(
      dataFile2->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/3);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(allSplits).copyResults(pool());

  // File1 loses c0=2 (delete applies), file2 keeps c0=2 (delete skipped)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 2, 4}),
      makeFlatVector<int64_t>({10, 30, 200, 400}),
  });

  assertEqualResults({expected}, {result});
}

TEST_F(CudfIcebergReadTest, schemaEvolutionRemoveColumn) {
  auto oldRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});
  auto newRowType = ROW({"c0", "c2"}, {BIGINT(), VARCHAR()});

  // Write data file with old schema (c0, c1, c2).
  auto dataVector = makeRowVector(
      oldRowType->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });

  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  auto icebergSplits = makeIcebergSplits(dataFilePath->getPath());

  // Expected result: c0 and c2 have values, c1 is not present.
  auto expected = makeRowVector(
      newRowType->names(),
      {
          dataVector->childAt(0),
          dataVector->childAt(2),
      });

  // Read with new schema (c0 and c2 only, c1 removed).
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, schemaEvolutionAddColumns) {
  auto oldRowType = ROW({"c0"}, {BIGINT()});
  auto newRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});

  // Write data file with old schema (only c0).
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({100, 200, 300}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);
  auto icebergSplits = makeIcebergSplits(dataFilePath->getPath());

  // Expected result: c0 has values, c1 and c2 are NULL.
  auto expected = makeRowVector({
      dataVector->childAt(0),
      makeNullConstant(TypeKind::INTEGER, 3),
      makeNullConstant(TypeKind::VARCHAR, 3),
  });

  // Read with new schema (c0, c1, and c2).
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .dataColumns(newRowType)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

// Hive-migrated partition columns: values come from the partitionKeys map,
// not the data file, and are injected by adaptColumns.
TEST_F(CudfIcebergReadTest, partitionColumnsFromHive) {
  auto fileRowType = ROW({"c0", "c1"}, {BIGINT(), INTEGER()});
  auto tableRowType =
      ROW({"c0", "c1", "region", "year"},
          {BIGINT(), INTEGER(), VARCHAR(), INTEGER()});

  // Write data file with only non-partition columns (c0, c1).
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  // Set partition keys for region and year.
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  partitionKeys["region"] = "US";
  partitionKeys["year"] = "2025";

  auto icebergSplits =
      makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);

  // Build column handles marking partition columns.
  facebook::velox::connector::ColumnHandleMap assignments;
  for (uint32_t i = 0; i < tableRowType->size(); ++i) {
    const auto& name = tableRowType->nameOf(i);
    auto columnType = (i >= fileRowType->size())
        ? HiveColumnHandle::ColumnType::kPartitionKey
        : HiveColumnHandle::ColumnType::kRegular;
    assignments[name] = std::make_shared<HiveColumnHandle>(
        name,
        columnType,
        tableRowType->childAt(i),
        tableRowType->childAt(i),
        std::vector<common::Subfield>{});
  }

  // Expected result: c0 and c1 from file, region and year from partition keys.
  auto expected = makeRowVector(
      tableRowType->names(),
      {
          dataVector->childAt(0),
          dataVector->childAt(1),
          makeFlatVector<std::string>({"US", "US", "US"}),
          makeFlatVector<int32_t>({2025, 2025, 2025}),
      });

  // Read with table schema including partition columns.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableRowType)
                  .dataColumns(tableRowType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

// Test reading a DATE identity partition column. DATE partition values arrive
// in two encodings: Hive-migrated tables store a date string ("2025-06-05")
// while Iceberg-native tables store days-since-epoch ("20244"). Both must
// decode to the same DATE result.
TEST_F(CudfIcebergReadTest, partitionColumnsDate) {
  auto fileRowType = ROW({"c0"}, {BIGINT()});
  auto tableRowType = ROW({"c0", "partitiondate"}, {BIGINT(), DATE()});

  // Write a data file with only the non-partition column c0.
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  // 2025-06-05 is 20244 days since the Unix epoch.
  const int32_t kDays = DATE()->toDays("2025-06-05");

  // Build column handles marking partitiondate as a partition key.
  facebook::velox::connector::ColumnHandleMap assignments;
  for (uint32_t i = 0; i < tableRowType->size(); ++i) {
    const auto& name = tableRowType->nameOf(i);
    auto columnType = (i >= fileRowType->size())
        ? HiveColumnHandle::ColumnType::kPartitionKey
        : HiveColumnHandle::ColumnType::kRegular;
    assignments[name] = std::make_shared<HiveColumnHandle>(
        name,
        columnType,
        tableRowType->childAt(i),
        tableRowType->childAt(i),
        std::vector<common::Subfield>{});
  }

  auto expected = makeRowVector(
      tableRowType->names(),
      {
          dataVector->childAt(0),
          makeFlatVector<int32_t>({kDays, kDays, kDays}, DATE()),
      });

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableRowType)
                  .dataColumns(tableRowType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  // Encoding 1: Hive-migrated date string.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    partitionKeys["partitiondate"] = "2025-06-05";
    auto splits = makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);
    AssertQueryBuilder(plan).splits(splits).assertResults({expected});
  }

  // Encoding 2: Iceberg-native days-since-epoch.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    partitionKeys["partitiondate"] = folly::to<std::string>(kDays);
    auto splits = makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);
    AssertQueryBuilder(plan).splits(splits).assertResults({expected});
  }
}

} // namespace facebook::velox::cudf_velox::exec::test
