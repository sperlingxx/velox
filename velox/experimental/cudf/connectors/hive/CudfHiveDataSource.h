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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/RandomUtil.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <future>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox {
class PinnedHostBuffer;
} // namespace facebook::velox::cudf_velox

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

using CudfParquetReader = cudf::io::chunked_parquet_reader;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;

using CudfHybridScanReader =
    cudf::io::parquet::experimental::hybrid_scan_reader;
using CudfHybridScanReaderPtr = std::unique_ptr<CudfHybridScanReader>;

class CudfHiveDataSource : public DataSource, public NvtxHelper {
 public:
  CudfHiveDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& CudfHiveConfig);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<facebook::velox::common::Filter>& /*filter*/)
      override {
    VELOX_NYI(
        "Dynamic filters not yet implemented by cudf::CudfHiveConnector.");
  }

  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& /* future */) override;

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  const common::SubfieldFilters* getFilters() const override {
    return &subfieldFilters_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  // Create a CudfParquetReader with the given split.
  CudfParquetReaderPtr createSplitReader();
  CudfHybridScanReaderPtr createExperimentalSplitReader();

  // Clear split_ and splitReader after split has been fully processed.  Keep
  // readers around to hold adaptation.
  void resetSplit();
  // Clear cudfTable_ and currentCudfTableView_ once we have successfully
  // converted it to `RowVectorPtr` and returned.
  void resetCudfTableAndView();
  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  // Setup the cuDF data source and options
  void setupCudfDataSourceAndOptions();

  RowVectorPtr emptyOutput_;

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;

  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  facebook::velox::FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;

  memory::MemoryPool* const pool_;

  // cuDF split reader stuff.
  cudf::io::parquet_reader_options readerOptions_;
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::unique_ptr<std::once_flag> tableMaterialized_;
  CudfParquetReaderPtr splitReader_;
  CudfHybridScanReaderPtr exptSplitReader_;
  bool useExperimentalSplitReader_;
  rmm::cuda_stream_view stream_;

  // Output type from file reader.  This is different from outputType_ that it
  // contains column names before assignment, and columns that only used in
  // remaining filter.
  RowTypePtr readerOutputType_;

  // Columns to read from the Parquet file (excludes partition key columns).
  std::vector<std::string> readColumnNames_;

  // Partition key columns that must be materialised as constants from the
  // split metadata.  Each entry records the output column index, column name,
  // and Velox type so we can inject the right constant vector after the cuDF
  // read.
  struct PartitionColumnInfo {
    size_t outputIndex;
    std::string name;
    TypePtr type;
  };
  std::vector<PartitionColumnInfo> partitionColumns_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;

  dwio::common::ReaderOptions baseReaderOpts_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  // The row type for the data source output, not including filter-only columns
  const RowTypePtr outputType_;

  // outputType_ minus partition-key columns; used for the cudf reader/converter
  // which only sees columns actually stored in the Parquet file.
  RowTypePtr dataOutputType_;

  // Expression evaluator for remaining filter.
  core::ExpressionEvaluator* const expressionEvaluator_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  std::shared_ptr<velox::cudf_velox::CudfExpression> cudfExpressionEvaluator_;

  // Expression evaluator for subfield filter.
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  cudf::ast::tree subfieldTree_;
  common::SubfieldFilters subfieldFilters_;
  // Cached combined subfield filter expression owned by 'subfieldTree_'.
  cudf::ast::expression const* subfieldFilterExpr_{nullptr};

  dwio::common::RuntimeStatistics runtimeStats_;
  std::atomic<uint64_t> totalRemainingFilterTime_{0};

  // Create callback data for total scan timing calculation
  struct TotalScanTimeCallbackData {
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };

  // Host callback function to calculate total scan time
  static void totalScanTimeCalculator(void* userData);

  // --- Cross-split accumulation for coalesced multi-file reads ---
  // Pending file ranges from a coalesced split, processed one at a time.
  std::vector<CoalescedFileRange> pendingFiles_;
  size_t nextFileIndex_{0};
  // Accumulated cudf tables from multiple files, flushed when target reached.
  std::vector<std::unique_ptr<cudf::table>> accumulatedTables_;
  int64_t accumulatedBytes_{0};
  int64_t numCoalescedBatches_{0};

  // Coalesced scan timing metrics.
  std::atomic<uint64_t> totalCoalesceBufferTimeNs_{0};
  std::atomic<uint64_t> totalFileAdvanceTimeNs_{0};
  std::atomic<uint64_t> totalPreReadTimeNs_{0};
  int64_t numFilesCoalesced_{0};

  // Max number of I/O threads active concurrently during pre-read.
  int32_t maxActiveIoThreads_{0};

  // Per-file async read futures launched in addSplit().
  struct AsyncFileRead {
    std::shared_future<std::shared_ptr<cudf_velox::PinnedHostBuffer>> future;
    size_t fileSize;
    uint64_t start;
    uint64_t length;
    // Prevents the ReadFile from being destroyed on the IO executor
    // thread.  HDFS files close via JNI; keeping the shared_ptr here
    // ensures the file handle outlives the async read and is destroyed
    // on the calling thread (when asyncFileReads_ is cleared).
    std::shared_ptr<ReadFile> readFile;
  };
  std::vector<AsyncFileRead> asyncFileReads_;

  // Multi-source reader: all pinned buffers kept alive while the single
  // chunked_parquet_reader references them via host_span in source_info.
  std::vector<std::shared_ptr<cudf_velox::PinnedHostBuffer>>
      coalescedPinnedBuffers_;
  // True when addSplit() prepared async reads for multi-source; the reader
  // will be created lazily on the first next() call.
  bool coalescedMultiSourcePending_{false};

  // Create a single multi-source chunked_parquet_reader from all async-read
  // pinned buffers (regular reader path only).
  void createCoalescedMultiSourceReader();

  // Keeps the current file's pinned buffer alive for the experimental
  // reader per-file path (advanceToNextCoalescedFile only).
  std::shared_ptr<cudf_velox::PinnedHostBuffer> currentFilePinnedBuffer_;

  // Advance to the next file in the coalesced split (experimental reader
  // path only). Returns false if no more files.
  bool advanceToNextCoalescedFile();
  // Flush accumulated tables into one CudfVector output.
  RowVectorPtr flushAccumulated();

  // Wrap a data-only RowVector into the full outputType_ by inserting constant
  // partition-key columns at the positions recorded in partitionColumns_.
  RowVectorPtr injectPartitionColumns(
      RowVectorPtr dataVector,
      vector_size_t nRows);

  // --- Chunked experimental reader state ---
  void initExperimentalReaderMetadata();
  std::unique_ptr<cudf::table> readNextExperimentalBatch(
      cudf::io::table_metadata& metadata);

  bool exptMetadataInitialized_{false};
  std::vector<cudf::size_type> exptFilteredRowGroups_;
  size_t exptNextRGIndex_{0};
  cudf::io::parquet_reader_options exptResolvedOptions_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
