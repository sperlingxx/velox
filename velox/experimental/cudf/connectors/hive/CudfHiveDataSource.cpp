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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.hpp"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/FieldReference.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda_runtime.h>

#include <future>
#include <limits>
#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {

using cudf_velox::GpuGuard;

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      baseReaderOpts_(pool_),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters.
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
    // Add fields in the filter to the columns to read if not there
    for (const auto& [field, _] : subfieldFilters_) {
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field.toString()) == readColumnNames_.end()) {
        readColumnNames_.push_back(field.toString());
      }
    }
  }

  // Create remaining filter
  auto remainingFilter = tableHandle_->remainingFilter();
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    for (const auto& field : remainingFilterExprSet_->distinctFields()) {
      // Add fields in the filter to the columns to read if not there
      if (std::find(
              readColumnNames_.begin(),
              readColumnNames_.end(),
              field->name()) == readColumnNames_.end()) {
        readColumnNames_.push_back(field->name());
      }
    }

    const RowTypePtr remainingFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    cudfExpressionEvaluator_ = velox::cudf_velox::createCudfExpression(
        remainingFilterExprSet_->exprs()[0], remainingFilterType_);
  }

  // Build a combined AST for all subfield filters once. This is query-constant
  // and doesn't depend on split-specific state.
  // STRING/Bytes filters are not supported by cudf AST/Jitify and will be
  // skipped. If all filters are unsupported, subfieldFilterExpr_ stays null
  // and the downstream FilterProject handles them on CPU.
  if (!subfieldFilters_.empty()) {
    const RowTypePtr readerFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> newNames;
        std::vector<TypePtr> newTypes;

        for (const auto& name : readColumnNames_) {
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          newNames.emplace_back(std::move(name));
          newTypes.push_back(parsedType);
        }

        return ROW(std::move(newNames), std::move(newTypes));
      } else {
        return outputType_;
      }
    }();

    bool hasStringColumns = false;
    for (const auto& [subfield, filterPtr] : subfieldFilters_) {
      if (!filterPtr || subfield.path().empty()) {
        continue;
      }
      auto* nestedField =
          dynamic_cast<const common::Subfield::NestedField*>(
              subfield.path()[0].get());
      if (nestedField && readerFilterType->containsChild(nestedField->name())) {
        auto colType =
            readerFilterType->findChild(nestedField->name());
        if (colType->isVarchar() || colType->isVarbinary()) {
          hasStringColumns = true;
          break;
        }
      }
    }

    if (hasStringColumns) {
      LOG(WARNING)
          << "Subfield filters involve STRING/VARBINARY columns; "
          << "skipping GPU subfield filtering to avoid Jitify errors.";
      subfieldFilterExpr_ = nullptr;
    } else {
      try {
        subfieldFilterExpr_ = &createAstFromSubfieldFilters(
            subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
      } catch (const VeloxException& e) {
        LOG(WARNING) << "Could not build AST for subfield filters: "
                     << e.message()
                     << ". Skipping GPU subfield filtering.";
        subfieldFilterExpr_ = nullptr;
      }
    }
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();

  // Whether to use the experimental split reader
  useExperimentalSplitReader_ =
      cudfHiveConfig_->useExperimentalCudfReaderSession(
          connectorQueryCtx_->sessionProperties());
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /* future */) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  // Basic sanity checks
  VELOX_CHECK_NOT_NULL(split_, "No split to process. Call addSplit first.");
  VELOX_CHECK(
      splitReader_ or exptSplitReader_ or coalescedMultiSourcePending_,
      "No split reader present");

  std::unique_ptr<cudf::table> cudfTable;
  cudf::io::table_metadata metadata;

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  const bool hasCoalescedFiles = !pendingFiles_.empty();
  const auto targetBytes = CudfConfig::getInstance().gpuTargetBatchBytes;

  // GPU guard for single-file and experimental paths; emplaced before first
  // GPU operation and held through common post-processing.
  std::optional<GpuGuard> gpuGuard;

  if (not useExperimentalSplitReader_) {
    // Lazily create the multi-source reader on first next() call.
    if (coalescedMultiSourcePending_) {
      createCoalescedMultiSourceReader();
    }

    VELOX_CHECK_NOT_NULL(splitReader_, "Regular cudf split reader not present");

    if (hasCoalescedFiles) {
      // Multi-source coalesced read: a single chunked_parquet_reader sees
      // all files as multiple sources, reading row groups across them.
      const auto effectiveTarget = (targetBytes > 0)
          ? targetBytes
          : std::numeric_limits<int64_t>::max();
      GpuGuard coalescedGpuGuard;
      auto coalesceLoopStartUs = getCurrentTimeMicro();
      while (splitReader_->has_next()) {
        auto tableWithMetadata = splitReader_->read_chunk();
        if (tableWithMetadata.tbl && tableWithMetadata.tbl->num_rows() > 0) {
          auto& tbl = tableWithMetadata.tbl;
          if (remainingFilterExprSet_) {
            auto cols = tbl->release();
            const auto originalNumColumns = cols.size();
            std::vector<cudf::column_view> colViews;
            colViews.reserve(cols.size());
            for (auto& c : cols) {
              colViews.push_back(c->view());
            }
            auto filterResult = cudfExpressionEvaluator_->eval(
                colViews,
                stream_,
                cudf::get_current_device_resource_ref());
            std::vector<std::unique_ptr<cudf::column>> origCols;
            origCols.reserve(originalNumColumns);
            std::move(
                cols.begin(),
                cols.begin() + originalNumColumns,
                std::back_inserter(origCols));
            auto origTable =
                std::make_unique<cudf::table>(std::move(origCols));
            tbl = cudf::apply_boolean_mask(
                *origTable,
                asView(filterResult),
                stream_,
                cudf::get_current_device_resource_ref());
          }
          if (tbl->num_rows() > 0) {
            auto tableBytes = estimateTableBytes(tbl);
            accumulatedTables_.push_back(std::move(tbl));
            accumulatedBytes_ += tableBytes;
            if (accumulatedBytes_ >= effectiveTarget) {
              break;
            }
          }
        }
      }
      totalCoalesceBufferTimeNs_.fetch_add(
          (getCurrentTimeMicro() - coalesceLoopStartUs) * 1000,
          std::memory_order_relaxed);
      auto result = flushAccumulated();
      if (result == nullptr) {
        return nullptr;
      }
      TotalScanTimeCallbackData* callbackData =
          new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};
      cudaLaunchHostFunc(
          stream_.value(),
          &CudfHiveDataSource::totalScanTimeCalculator,
          callbackData);
      return result;
    }

    // Single-file path (no coalesced files).
    if (not splitReader_->has_next()) {
      return nullptr;
    }
    gpuGuard.emplace();
    auto tableWithMetadata = splitReader_->read_chunk();
    cudfTable = std::move(tableWithMetadata.tbl);
    metadata = std::move(tableWithMetadata.metadata);
  } else {
    // Chunked experimental reader: process row groups in batches.
    // Loops across coalesced files when the current file is exhausted.
    VELOX_CHECK_NOT_NULL(
        exptSplitReader_, "Experimental cudf split reader not present");

    gpuGuard.emplace();
    while (true) {
      if (!exptMetadataInitialized_) {
        initExperimentalReaderMetadata();
      }

      if (exptNextRGIndex_ < exptFilteredRowGroups_.size()) {
        cudfTable = readNextExperimentalBatch(metadata);
        if (cudfTable && cudfTable->num_rows() > 0) {
          break;
        }
      }

      if (!hasCoalescedFiles || !advanceToNextCoalescedFile()) {
        return nullptr;
      }
    }
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(),
      &CudfHiveDataSource::totalScanTimeCalculator,
      callbackData);

  uint64_t filterTimeUs{0};
  // Apply remaining filter if present
  if (remainingFilterExprSet_) {
    MicrosecondTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    // Filter may need additional computed columns
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult = cudfExpressionEvaluator_->eval(
        inputViews, stream_, cudf::get_current_device_resource_ref());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    // Keep only rows where the filter is true
    cudfTable = cudf::apply_boolean_mask(
        *originalTable,
        asView(filterResult),
        stream_,
        cudf::get_current_device_resource_ref());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  // Output RowVectorPtr
  const auto nRows = cudfTable->num_rows();

  // keep only outputType_.size() columns in cudfTable_
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(originalColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(originalColumns));
  }

  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_->names(), stream_);

  // Check if conversion yielded a nullptr
  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  // Update completedRows_.
  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

void CudfHiveDataSource::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTime(elapsedNs);

  delete data;
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = [&]() {
    // Dynamic cast split to `CudfHiveConnectorSplit`
    if (std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
      return std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split);
      // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
    } else if (std::dynamic_pointer_cast<hive::HiveConnectorSplit>(split)) {
      const auto hiveSplit =
          std::dynamic_pointer_cast<hive::HiveConnectorSplit>(split);
      VELOX_CHECK_EQ(
          hiveSplit->fileFormat,
          dwio::common::FileFormat::PARQUET,
          "Unsupported file format for conversion from HiveConnectorSplit to CudfHiveConnectorSplit");
      // Remove "file:" prefix from the file path if present
      std::string cleanedPath = hiveSplit->filePath;
      constexpr std::string_view kFilePrefix = "file:";
      constexpr std::string_view kS3APrefix = "s3a:";
      if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
        cleanedPath = cleanedPath.substr(kFilePrefix.size());
      } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
        // KvikIO does not support "s3a:" prefix. We need to translate it to
        // "s3:".
        cleanedPath.erase(kS3APrefix.size() - 2, 1);
      }
      auto cudfHiveSplitBuilder = CudfHiveConnectorSplitBuilder(cleanedPath)
                                      .start(hiveSplit->start)
                                      .length(hiveSplit->length)
                                      .connectorId(hiveSplit->connectorId)
                                      .splitWeight(hiveSplit->splitWeight);
      for (auto const& infoColumn : hiveSplit->infoColumns) {
        cudfHiveSplitBuilder.infoColumn(infoColumn.first, infoColumn.second);
      }
      return cudfHiveSplitBuilder.build();
    } else {
      VELOX_FAIL("Unsupported split type: {}", split->toString());
    }
  }();

  VLOG(1) << "Adding split " << split_->toString();

  // Reset cross-split accumulation state.
  pendingFiles_.clear();
  nextFileIndex_ = 0;
  accumulatedTables_.clear();
  accumulatedBytes_ = 0;

  // Launch async reads into PinnedHostBuffers for coalesced files.
  // Each file is read in full so the cuDF Parquet reader can parse
  // the header/footer; skip_bytes/num_bytes handle row-group filtering.
  asyncFileReads_.clear();
  coalescedPinnedBuffers_.clear();
  coalescedMultiSourcePending_ = false;

  if (!split_->coalescedFiles.empty()) {
    pendingFiles_ = split_->coalescedFiles;
    for (const auto& fileRange : pendingFiles_) {
      completedBytes_ += fileRange.length;
    }

    auto launchAsyncRead = [this](const std::string& path, uint64_t start,
                              uint64_t length)
        -> AsyncFileRead {
      auto colNames = readColumnNames_;
      const auto fileHandleKey = FileHandleKey{
          .filename = path,
          .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
      auto fileProperties = FileProperties{};
      auto fileHandleCachePtr = fileHandleFactory_->generate(
          fileHandleKey, &fileProperties,
          ioStats_ ? ioStats_.get() : nullptr);
      auto readFile = fileHandleCachePtr->file;
      const auto fsize = readFile->size();

      if (executor_) {
        // Dispatch onto the IO executor whose threads are JVM-attached,
        // so HDFS (libhdfs JNI) pread operations work correctly —
        // unlike std::async threads which are not attached to the JVM.
        auto promise = std::make_shared<
            std::promise<std::shared_ptr<cudf_velox::PinnedHostBuffer>>>();
        auto future = promise->get_future().share();

        // Capture readFile by value (shared_ptr copy) to keep the file
        // handle alive for the duration of the async read.
        executor_->add(
            [promise, readFile, colNames = std::move(colNames), start]() {
              try {
                auto buf =
                    selectiveParquetRead(readFile.get(), colNames, start);
                promise->set_value(std::move(buf));
              } catch (...) {
                promise->set_exception(std::current_exception());
              }
            });

        return {std::move(future), fsize, start, length,
                std::move(readFile)};
      }

      // Fallback: no executor — run synchronously.
      auto buf = selectiveParquetRead(readFile.get(), colNames, start);
      std::promise<std::shared_ptr<cudf_velox::PinnedHostBuffer>> promise;
      promise.set_value(std::move(buf));
      return {promise.get_future().share(), fsize, start, length,
              std::move(readFile)};
    };

    auto preReadStartUs = getCurrentTimeMicro();

    if (!useExperimentalSplitReader_) {
      // Multi-source approach (like spark-rapids): async-read ALL files
      // (primary + coalesced) into individual pinned buffers, then create
      // ONE chunked_parquet_reader with all buffers as multiple sources.
      const size_t totalFiles = 1 + pendingFiles_.size();
      asyncFileReads_.reserve(totalFiles);
      asyncFileReads_.push_back(launchAsyncRead(
          split_->filePath, split_->start, split_->length));
      for (const auto& fileRange : pendingFiles_) {
        asyncFileReads_.push_back(launchAsyncRead(
            fileRange.filePath, fileRange.start, fileRange.length));
      }
      coalescedMultiSourcePending_ = true;
    } else {
      // Experimental reader: per-file approach. Only async-read coalesced
      // files; the primary file's reader is created synchronously below.
      asyncFileReads_.reserve(pendingFiles_.size());
      for (const auto& fileRange : pendingFiles_) {
        asyncFileReads_.push_back(launchAsyncRead(
            fileRange.filePath, fileRange.start, fileRange.length));
      }
    }

    totalPreReadTimeNs_.fetch_add(
        (getCurrentTimeMicro() - preReadStartUs) * 1000,
        std::memory_order_relaxed);
  }

  // Split reader already exists, reset
  if (splitReader_ or exptSplitReader_) {
    splitReader_.reset();
    exptSplitReader_.reset();
    tableMaterialized_.reset();
  }

  if (coalescedMultiSourcePending_) {
    // Multi-source path: defer reader creation to first next() call,
    // after all async reads have completed.
    stream_ = cudfGlobalStreamPool().get_stream();
    tableMaterialized_ = std::make_unique<std::once_flag>();
    completedBytes_ += split_->length;
    numFilesCoalesced_ = static_cast<int64_t>(asyncFileReads_.size());
    return;
  }

  // Create a cudf split reader (single-file or experimental coalesced)
  if (useExperimentalSplitReader_) {
    exptSplitReader_ = createExperimentalSplitReader();
  } else {
    splitReader_ = createSplitReader();
  }

  tableMaterialized_ = std::make_unique<std::once_flag>();

  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    if (fileHandleCachePtr.get() and fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

void CudfHiveDataSource::setupCudfDataSourceAndOptions() {
  // Build source info for the chunked parquet reader
  auto sourceInfo = [&]() {
    // Use file data source if we don't want to use the BufferedInput source
    if (not cudfHiveConfig_->useBufferedInputSession(
            connectorQueryCtx_->sessionProperties())) {
      VLOG(1) << "Using file data source for CudfHiveDataSource";
      return cudf::io::source_info{split_->filePath};
    }

    auto fileHandleCachePtr = FileHandleCachedPtr{};
    try {
      const auto fileHandleKey = FileHandleKey{
          .filename = split_->filePath,
          .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
      auto fileProperties = FileProperties{};
      fileHandleCachePtr = fileHandleFactory_->generate(
          fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
      VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
    } catch (const VeloxRuntimeError& e) {
      LOG(WARNING) << fmt::format(
          "Failed to generate file handle cache for file {}, falling back to file data source for CudfHiveDataSource",
          split_->filePath);
      return cudf::io::source_info{split_->filePath};
    }

    // Here we keep adding new entries to CacheTTLController when new
    // fileHandles are generated, if CacheTTLController was created. Creator of
    // CacheTTLController needs to make sure a size control strategy was
    // available such as removing aged out entries.
    if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
      cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
    }

    auto bufferedInput =
        velox::connector::hive::BufferedInputBuilder::getInstance()->create(
            *fileHandleCachePtr,
            baseReaderOpts_,
            connectorQueryCtx_,
            ioStatistics_,
            ioStats_,
            executor_);
    if (not bufferedInput) {
      LOG(WARNING) << fmt::format(
          "Failed to create buffered input source for file {}, falling back to file data source for CudfHiveDataSource",
          split_->filePath);
      return cudf::io::source_info{split_->filePath};
    }
    dataSource_ =
        std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
    return cudf::io::source_info{dataSource_.get()};
  }();

  if (dataSource_ == nullptr) {
    dataSource_ = std::move(makeDataSourcesFromSourceInfo(sourceInfo).front());
  }

  RowTypePtr readerFilterType = nullptr;
  bool hasDecimalFilter = false;
  if (subfieldFilters_.size()) {
    readerFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> newNames;
        std::vector<TypePtr> newTypes;

        for (const auto& name : readColumnNames_) {
          // Ensure all columns being read are available to the filter
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          newNames.emplace_back(std::move(name));
          newTypes.push_back(parsedType);
        }

        return ROW(std::move(newNames), std::move(newTypes));
      } else {
        return outputType_;
      }
    }();

    for (const auto& [field, _] : subfieldFilters_) {
      if (!field.valid()) {
        continue;
      }
      const auto& fieldName = field.baseName();
      const auto fieldType = readerFilterType->findChild(fieldName);
      if (fieldType && fieldType->isDecimal()) {
        hasDecimalFilter = true;
        break;
      }
    }
  }

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .skip_bytes(split_->start)
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .use_jit_filter(hasDecimalFilter)
          .build();

  // Set num_bytes only if available
  if (split_->size() != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(split_->size());
  }

  // Set filter expression created in constructor if any subfield filters
  if (subfieldFilterExpr_ != nullptr) {
    readerOptions_.set_filter(*subfieldFilterExpr_);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }
}

CudfParquetReaderPtr CudfHiveDataSource::createSplitReader() {
  setupCudfDataSourceAndOptions();
  stream_ = cudfGlobalStreamPool().get_stream();

  // Create a parquet reader
  return std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      readerOptions_,
      stream_,
      cudf::get_current_device_resource_ref());
}

CudfHybridScanReaderPtr CudfHiveDataSource::createExperimentalSplitReader() {
  setupCudfDataSourceAndOptions();
  stream_ = cudfGlobalStreamPool().get_stream();

  // Create a hybrid scan reader
  auto const footerBytes = fetchFooterBytes(dataSource_);
  auto exptSplitReader = std::make_unique<CudfHybridScanReader>(
      cudf::host_span<uint8_t const>{footerBytes->data(), footerBytes->size()},
      readerOptions_);

  // Setup page index if available
  auto const pageIndexByteRange = exptSplitReader->page_index_byte_range();
  if (not pageIndexByteRange.is_empty()) {
    auto const pageIndexBytes = dataSource_->host_read(
        pageIndexByteRange.offset(), pageIndexByteRange.size());
    exptSplitReader->setup_page_index(
        cudf::host_span<uint8_t const>{
            pageIndexBytes->data(), pageIndexBytes->size()});
  }

  return exptSplitReader;
}

void CudfHiveDataSource::resetSplit() {
  split_.reset();
  splitReader_.reset();
  exptSplitReader_.reset();
  tableMaterialized_.reset();
  dataSource_.reset();
  coalescedPinnedBuffers_.clear();
  currentFilePinnedBuffer_.reset();
  coalescedMultiSourcePending_ = false;
  exptMetadataInitialized_ = false;
  exptFilteredRowGroups_.clear();
  exptNextRGIndex_ = 0;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  res.insert({
      {"totalScanTime",
       RuntimeMetric(
           ioStatistics_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
      {"totalRemainingFilterTime",
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  if (numCoalescedBatches_ > 0) {
    res.emplace(
        "numCoalescedBatches", RuntimeMetric(numCoalescedBatches_));
    res.emplace(
        "totalCoalesceBufferTime",
        RuntimeMetric(
            totalCoalesceBufferTimeNs_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos));
    res.emplace(
        "totalFileAdvanceTime",
        RuntimeMetric(
            totalFileAdvanceTimeNs_.load(std::memory_order_relaxed),
            RuntimeCounter::Unit::kNanos));
    res.emplace("numFilesCoalesced", RuntimeMetric(numFilesCoalesced_));
    auto preReadNs = totalPreReadTimeNs_.load(std::memory_order_relaxed);
    if (preReadNs > 0) {
      res.emplace(
          "totalPreReadTime",
          RuntimeMetric(preReadNs, RuntimeCounter::Unit::kNanos));
    }
    if (!asyncFileReads_.empty()) {
      res.emplace(
          "asyncFileReadsLaunched",
          RuntimeMetric(static_cast<int64_t>(asyncFileReads_.size())));
    }
    if (!coalescedPinnedBuffers_.empty()) {
      size_t totalPinnedBytes = 0;
      for (const auto& buf : coalescedPinnedBuffers_) {
        if (buf) {
          totalPinnedBytes += buf->size();
        }
      }
      res.emplace(
          "multiSourcePinnedBuffers",
          RuntimeMetric(
              static_cast<int64_t>(coalescedPinnedBuffers_.size())));
      res.emplace(
          "multiSourcePinnedBytes",
          RuntimeMetric(
              static_cast<int64_t>(totalPinnedBytes),
              RuntimeCounter::Unit::kBytes));
    }
  }
  const auto& ioStats = ioStats_->stats();
  for (const auto& storageStats : ioStats) {
    res.emplace(storageStats.first, storageStats.second);
  }

  if (auto* bids = dynamic_cast<BufferedInputDataSource*>(dataSource_.get())) {
    res.emplace(
        "pinnedAllocBytes",
        RuntimeMetric(bids->pinnedAllocBytes(), RuntimeCounter::Unit::kBytes));
    res.emplace(
        "pageableAllocBytes",
        RuntimeMetric(
            bids->pageableAllocBytes(), RuntimeCounter::Unit::kBytes));
  }

  return res;
}

void CudfHiveDataSource::createCoalescedMultiSourceReader() {
  VELOX_CHECK(
      coalescedMultiSourcePending_,
      "createCoalescedMultiSourceReader called without pending multi-source");
  VELOX_CHECK(
      !asyncFileReads_.empty(),
      "No async file reads for multi-source reader");

  auto waitStartUs = getCurrentTimeMicro();

  // Wait for all async reads and collect pinned buffers.
  coalescedPinnedBuffers_.reserve(asyncFileReads_.size());
  std::vector<cudf::host_span<const std::byte>> bufferSpans;
  bufferSpans.reserve(asyncFileReads_.size());

  for (auto& asyncRead : asyncFileReads_) {
    auto pinnedBuf = asyncRead.future.get();
    VELOX_CHECK_NOT_NULL(pinnedBuf, "Async file read returned null");
    bufferSpans.push_back(cudf::host_span<const std::byte>(
        reinterpret_cast<const std::byte*>(pinnedBuf->data()),
        pinnedBuf->size()));
    coalescedPinnedBuffers_.push_back(std::move(pinnedBuf));
  }

  totalPreReadTimeNs_.fetch_add(
      (getCurrentTimeMicro() - waitStartUs) * 1000,
      std::memory_order_relaxed);

  // Build multi-source source_info: cuDF treats each buffer as a separate
  // Parquet file and merges their row groups into a single reader pipeline.
  auto sourceInfo = cudf::io::source_info(
      cudf::host_span<cudf::host_span<const std::byte>>(
          bufferSpans.data(), bufferSpans.size()));

  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  if (subfieldFilterExpr_ != nullptr) {
    readerOptions_.set_filter(*subfieldFilterExpr_);
  }
  if (!readColumnNames_.empty()) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      readerOptions_,
      stream_,
      cudf::get_current_device_resource_ref());

  coalescedMultiSourcePending_ = false;

  LOG(INFO) << "Created multi-source parquet reader with "
            << asyncFileReads_.size() << " files ("
            << coalescedPinnedBuffers_.size() << " pinned buffers)";
}

bool CudfHiveDataSource::advanceToNextCoalescedFile() {
  if (nextFileIndex_ >= pendingFiles_.size()) {
    return false;
  }

  const auto& fileRange = pendingFiles_[nextFileIndex_];
  const size_t fileIdx = nextFileIndex_;
  ++nextFileIndex_;

  splitReader_.reset();
  exptSplitReader_.reset();
  dataSource_.reset();
  tableMaterialized_ = std::make_unique<std::once_flag>();
  exptMetadataInitialized_ = false;
  exptFilteredRowGroups_.clear();
  exptNextRGIndex_ = 0;

  // Build a temporary CudfHiveConnectorSplit for this file.
  split_ = std::make_shared<CudfHiveConnectorSplit>(
      split_->connectorId,
      fileRange.filePath,
      fileRange.start,
      fileRange.length,
      0,
      fileRange.infoColumns);

  // Try to use the async pre-read pinned buffer (pipelined IO).
  // The future blocks only until THIS file's IO completes — other files'
  // reads continue in the background, overlapping with GPU processing.
  const bool hasAsyncRead = fileIdx < asyncFileReads_.size();
  if (hasAsyncRead) {
    auto& asyncRead = asyncFileReads_[fileIdx];
    // Wait for this file's IO to complete (likely already done while
    // GPU was processing the previous file).
    currentFilePinnedBuffer_ = asyncRead.future.get();
    VELOX_CHECK_NOT_NULL(
        currentFilePinnedBuffer_, "Async file read returned null");

    // Build source_info from the pinned buffer.
    auto sourceInfo = cudf::io::source_info(
        cudf::host_span<const std::byte>(
            reinterpret_cast<const std::byte*>(
                currentFilePinnedBuffer_->data()),
            currentFilePinnedBuffer_->size()));

    // If selectiveParquetRead produced a compact buffer (smaller than the
    // original file), the internal offsets are already self-contained.
    // skip_bytes/num_bytes must not be applied to reconstructed buffers.
    const bool isSelectiveBuffer =
        currentFilePinnedBuffer_->size() != asyncRead.fileSize;
    auto builder =
        cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
            .skip_bytes(isSelectiveBuffer ? 0 : asyncRead.start)
            .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
            .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
            .allow_mismatched_pq_schemas(
                cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
            .timestamp_type(cudfHiveConfig_->timestampType());
    readerOptions_ = builder.build();
    if (!isSelectiveBuffer && asyncRead.length < asyncRead.fileSize) {
      readerOptions_.set_num_bytes(asyncRead.length);
    }
    if (subfieldFilterExpr_ != nullptr) {
      readerOptions_.set_filter(*subfieldFilterExpr_);
    }
    if (readColumnNames_.size()) {
      readerOptions_.set_column_names(readColumnNames_);
    }

    if (useExperimentalSplitReader_) {
      dataSource_ =
          std::move(makeDataSourcesFromSourceInfo(
                        cudf::io::source_info(
                            cudf::host_span<const std::byte>(
                                reinterpret_cast<const std::byte*>(
                                    currentFilePinnedBuffer_->data()),
                                currentFilePinnedBuffer_->size())))
                        .front());
      auto const footerBytes = fetchFooterBytes(dataSource_);
      exptSplitReader_ = std::make_unique<CudfHybridScanReader>(
          cudf::host_span<uint8_t const>{
              footerBytes->data(), footerBytes->size()},
          readerOptions_);
      auto const pageIndexByteRange =
          exptSplitReader_->page_index_byte_range();
      if (not pageIndexByteRange.is_empty()) {
        auto const pageIndexBytes = dataSource_->host_read(
            pageIndexByteRange.offset(), pageIndexByteRange.size());
        exptSplitReader_->setup_page_index(
            cudf::host_span<uint8_t const>{
                pageIndexBytes->data(), pageIndexBytes->size()});
      }
    } else {
      splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          cudfHiveConfig_->maxChunkReadLimit(),
          cudfHiveConfig_->maxPassReadLimit(),
          readerOptions_,
          stream_,
          cudf::get_current_device_resource_ref());
    }
  } else {
    // No async pre-read available; read from disk synchronously.
    if (useExperimentalSplitReader_) {
      setupCudfDataSourceAndOptions();
      auto const footerBytes = fetchFooterBytes(dataSource_);
      exptSplitReader_ = std::make_unique<CudfHybridScanReader>(
          cudf::host_span<uint8_t const>{
              footerBytes->data(), footerBytes->size()},
          readerOptions_);
      auto const pageIndexByteRange =
          exptSplitReader_->page_index_byte_range();
      if (not pageIndexByteRange.is_empty()) {
        auto const pageIndexBytes = dataSource_->host_read(
            pageIndexByteRange.offset(), pageIndexByteRange.size());
        exptSplitReader_->setup_page_index(
            cudf::host_span<uint8_t const>{
                pageIndexBytes->data(), pageIndexBytes->size()});
      }
    } else {
      setupCudfDataSourceAndOptions();
      splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
          cudfHiveConfig_->maxChunkReadLimit(),
          cudfHiveConfig_->maxPassReadLimit(),
          readerOptions_,
          stream_,
          cudf::get_current_device_resource_ref());
    }
  }

  ++numFilesCoalesced_;
  VLOG(1) << "Advanced to coalesced file " << fileRange.filePath
           << " (" << nextFileIndex_ << "/" << pendingFiles_.size() << ")";
  return true;
}

RowVectorPtr CudfHiveDataSource::flushAccumulated() {
  if (accumulatedTables_.empty()) {
    return nullptr;
  }
  ++numCoalescedBatches_;

  std::unique_ptr<cudf::table> cudfTable;
  if (accumulatedTables_.size() == 1) {
    cudfTable = std::move(accumulatedTables_[0]);
  } else {
    cudfTable = concatenateTables(std::move(accumulatedTables_), stream_);
  }
  accumulatedTables_.clear();
  accumulatedBytes_ = 0;

  const auto nRows = cudfTable->num_rows();
  if (nRows == 0) {
    return nullptr;
  }

  // Keep only outputType_.size() columns
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> originalColumns;
    originalColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(originalColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(originalColumns));
  }

  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_->names(), stream_);

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");
  completedRows_ += output->size();
  return output;
}

void CudfHiveDataSource::initExperimentalReaderMetadata() {
  VELOX_CHECK(!exptMetadataInitialized_);
  exptFilteredRowGroups_ = exptSplitReader_->all_row_groups(readerOptions_);

  exptResolvedOptions_ = readerOptions_;

  if (readerOptions_.get_filter().has_value()) {
    auto exprConverter = referenceToNameConverter(
        readerOptions_.get_filter(),
        exptSplitReader_->parquet_metadata().schema,
        readColumnNames_);
    exptResolvedOptions_.set_filter(exprConverter.convertedExpression());

    auto footerBytes = fetchFooterBytes(dataSource_);
    auto tmpExptSplitReader = std::make_unique<CudfHybridScanReader>(
        cudf::host_span<uint8_t const>{
            footerBytes->data(), footerBytes->size()},
        exptResolvedOptions_);
    exptFilteredRowGroups_ =
        tmpExptSplitReader->filter_row_groups_with_stats(
            exptFilteredRowGroups_, exptResolvedOptions_, stream_);
  }

  if (not exptResolvedOptions_.get_filter().has_value()) {
    auto scalar = cudf::numeric_scalar<int32_t>(0, false, stream_);
    auto literal = cudf::ast::literal(scalar);
    auto filter =
        cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);
    exptResolvedOptions_.set_filter(filter);
  }

  exptNextRGIndex_ = 0;
  exptMetadataInitialized_ = true;
}

std::unique_ptr<cudf::table> CudfHiveDataSource::readNextExperimentalBatch(
    cudf::io::table_metadata& metadata) {
  VELOX_CHECK(exptMetadataInitialized_);
  if (exptNextRGIndex_ >= exptFilteredRowGroups_.size()) {
    return nullptr;
  }

  const auto targetBytes = CudfConfig::getInstance().gpuTargetBatchBytes;

  size_t batchEnd = exptNextRGIndex_;
  int64_t batchCompressedBytes = 0;
  while (batchEnd < exptFilteredRowGroups_.size()) {
    auto singleRG = cudf::host_span<cudf::size_type const>(
        &exptFilteredRowGroups_[batchEnd], 1);
    auto byteRanges =
        exptSplitReader_->payload_column_chunks_byte_ranges(
            singleRG, exptResolvedOptions_);
    int64_t rgBytes = 0;
    for (auto const& br : byteRanges) {
      rgBytes += br.size();
    }
    batchCompressedBytes += rgBytes;
    ++batchEnd;
    if (targetBytes > 0 && batchCompressedBytes >= targetBytes) {
      break;
    }
  }

  auto batchSpan = cudf::host_span<cudf::size_type const>(
      exptFilteredRowGroups_.data() + exptNextRGIndex_,
      batchEnd - exptNextRGIndex_);
  exptNextRGIndex_ = batchEnd;

  const auto columnChunkByteRanges =
      exptSplitReader_->payload_column_chunks_byte_ranges(
          batchSpan, exptResolvedOptions_);

  std::vector<rmm::device_buffer> columnChunkBuffers(
      columnChunkByteRanges.size());
  std::vector<std::future<size_t>> ioFutures{};
  ioFutures.reserve(columnChunkByteRanges.size());
  std::for_each(
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator(columnChunkByteRanges.size()),
      [&](auto idx) {
        const auto& byteRange = columnChunkByteRanges[idx];
        auto& buffer = columnChunkBuffers[idx];

        constexpr size_t bufferPaddingMultiple = 8;
        buffer = rmm::device_buffer(
            cudf::util::round_up_safe<size_t>(
                byteRange.size(), bufferPaddingMultiple),
            stream_,
            cudf::get_current_device_resource_ref());
        if (auto bufferedInput =
                dynamic_cast<BufferedInputDataSource*>(dataSource_.get())) {
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              static_cast<uint8_t*>(buffer.data()));
        } else if (
            dataSource_->supports_device_read() and
            dataSource_->is_device_read_preferred(byteRange.size())) {
          ioFutures.emplace_back(dataSource_->device_read_async(
              byteRange.offset(),
              byteRange.size(),
              static_cast<uint8_t*>(buffer.data()),
              stream_));
        } else {
          auto hostBuffer =
              dataSource_->host_read(byteRange.offset(), byteRange.size());
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              buffer.data(),
              hostBuffer->data(),
              byteRange.size(),
              cudaMemcpyHostToDevice,
              stream_.value()));
        }
      });

  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource_.get())) {
    bufferedInput->load(stream_);
  }
  std::for_each(ioFutures.begin(), ioFutures.end(), [](auto& future) {
    future.get();
  });

  std::vector<cudf::device_span<uint8_t const>> columnChunkData;
  columnChunkData.reserve(columnChunkBuffers.size());
  std::transform(
      columnChunkBuffers.begin(),
      columnChunkBuffers.end(),
      std::back_inserter(columnChunkData),
      [](auto& buffer) {
        return cudf::device_span<uint8_t const>{
            static_cast<uint8_t*>(buffer.data()), buffer.size()};
      });

  const auto totalRows =
      exptSplitReader_->total_rows_in_row_groups(batchSpan);
  auto const scalarTrue = cudf::numeric_scalar<bool>(true, true, stream_);
  auto allTrueRowMask =
      cudf::make_column_from_scalar(scalarTrue, totalRows, stream_);

  auto tableWithMetadata = exptSplitReader_->materialize_payload_columns(
      batchSpan,
      columnChunkData,
      allTrueRowMask->view(),
      cudf::io::parquet::experimental::use_data_page_mask::NO,
      readerOptions_,
      stream_,
      cudf::get_current_device_resource_ref());

  metadata = std::move(tableWithMetadata.metadata);

  if (readerOptions_.get_filter().has_value()) {
    std::unique_ptr<cudf::table> table = std::move(tableWithMetadata.tbl);
    try {
      auto filterMask = cudf::compute_column(
          *table, readerOptions_.get_filter().value(), stream_);
      return cudf::apply_boolean_mask(
          table->view(),
          filterMask->view(),
          stream_,
          cudf::get_current_device_resource_ref());
    } catch (const std::exception& e) {
      LOG(WARNING)
          << "Subfield filter compute_column failed (possibly Jitify): "
          << e.what() << ". Returning unfiltered data for this chunk.";
      return table;
    }
  }
  return std::move(tableWithMetadata.tbl);
}

} // namespace facebook::velox::cudf_velox::connector::hive
