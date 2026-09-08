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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/stream_compaction.hpp>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

bool isSupportedCudfReaderFilterType(const TypePtr& type) {
  if (type == nullptr) {
    return false;
  }

  switch (type->kind()) {
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
      return false;
    default:
      return true;
  }
}

TypePtr topLevelSubfieldType(
    const hive::HiveTableHandle& tableHandle,
    const RowTypePtr& outputType,
    const common::Subfield& field) {
  if (!field.valid()) {
    return nullptr;
  }

  const auto& baseName = field.baseName();
  if (tableHandle.dataColumns() &&
      tableHandle.dataColumns()->containsChild(baseName)) {
    return tableHandle.dataColumns()->findChild(baseName);
  }
  if (outputType && outputType->containsChild(baseName)) {
    return outputType->findChild(baseName);
  }
  return nullptr;
}

} // namespace

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
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  auto addReadColumn = [&](std::string_view name) {
    auto readName = toTopLevelReadColumnName(name);
    if (readColumnSet_.emplace(readName).second) {
      readColumnNames_.emplace_back(std::move(readName));
    }
  };

  // Set up column projection if needed
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    auto outputReadName = toTopLevelReadColumnName(handle->name());
    outputReadColumnNames_.emplace_back(outputReadName);
    addReadColumn(outputReadName);
  }

  // Copy subfield filters. Keep the originals separate from filters extracted
  // from the remaining expression so unsupported reader filters can only fall
  // back when the post-scan expression contains an equivalent predicate.
  common::SubfieldFilters originalSubfieldFilters;
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    originalSubfieldFilters.emplace(k.clone(), v->clone());
    subfieldFilters_.emplace(k.clone(), v->clone());
  }

  // Extract additional simple filters from remainingFilter (same as CPU path).
  // This extracts single-column filters like "col = 'X'" or "col <> 'Y'" from
  // complex expressions and adds them to subfieldFilters_ for pushdown.
  common::SubfieldFilters extractedSubfieldFilters;
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter =
      facebook::velox::connector::hive::extractFiltersFromRemainingFilter(
          tableHandle_->remainingFilter(),
          expressionEvaluator_,
          extractedSubfieldFilters,
          sampleRate);
  for (auto& [field, filter] : extractedSubfieldFilters) {
    if (auto it = subfieldFilters_.find(field); it != subfieldFilters_.end()) {
      filter = filter->mergeWith(it->second.get());
    }
    subfieldFilters_.insert_or_assign(field.clone(), filter);
  }

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    addReadColumn(field.toString());
  }

  common::SubfieldFilters readerSubfieldFilters;
  bool skippedReaderFilter = false;
  for (const auto& [field, filter] : subfieldFilters_) {
    const auto& path = field.path();
    if (path.size() != 1) {
      if (auto original = originalSubfieldFilters.find(field);
          original != originalSubfieldFilters.end()) {
        auto extracted = extractedSubfieldFilters.find(field);
        if (extracted == extractedSubfieldFilters.end() ||
            original->second->toString() != extracted->second->toString()) {
          VELOX_UNSUPPORTED(
              "Nested cuDF reader filter '{}' requires an equivalent "
              "post-scan predicate",
              field.toString());
        }
      }
      skippedReaderFilter = true;
      VLOG(1) << "Skipping nested cuDF reader filter pushdown for subfield: "
              << field.toString();
      continue;
    }

    const auto type = topLevelSubfieldType(*tableHandle_, outputType_, field);
    if (!isSupportedCudfReaderFilterType(type)) {
      if (auto original = originalSubfieldFilters.find(field);
          original != originalSubfieldFilters.end()) {
        auto extracted = extractedSubfieldFilters.find(field);
        if (extracted == extractedSubfieldFilters.end() ||
            original->second->toString() != extracted->second->toString()) {
          VELOX_UNSUPPORTED(
              "Complex cuDF reader filter '{}' requires an equivalent "
              "post-scan predicate",
              field.toString());
        }
      }
      skippedReaderFilter = true;
      VLOG(1) << "Skipping complex cuDF reader filter pushdown for subfield: "
              << field.toString();
      continue;
    }

    readerSubfieldFilters.emplace(field.clone(), filter);
  }

  const auto postScanFilter =
      skippedReaderFilter ? tableHandle_->remainingFilter() : remainingFilter;
  // Optimize (rewrites + constant folding) the remaining filter before
  // evaluator selection so CudfFunctions never see scalar-only operand sets.
  // TODO: ConnectorQueryCtx does not expose the session QueryCtx, only an
  // ExpressionEvaluator, so constant folding here runs against a transient
  // QueryCtx with default query config rather than the session's. Passing the
  // real session QueryCtx (e.g. by exposing it on ConnectorQueryCtx) should be
  // figured out later. A local QueryCtx is required because
  // expression::optimize constant-folds through exec::ExprSet, whose
  // constructor dereferences the QueryCtx unconditionally; a null QueryCtx
  // would crash.
  auto optimizeQueryCtx = core::QueryCtx::create();
  optimizedRemainingFilter_ = postScanFilter
      ? expression::optimize(postScanFilter, optimizeQueryCtx.get(), pool_)
      : nullptr;
  if (optimizedRemainingFilter_) {
    // Add fields referenced by the filter to the columns to read. Collect from
    // the optimized expression since folding may drop branches and the columns
    // they reference. Read-column order does not affect results: the data
    // source projects its output to the requested output type.
    for (const auto& name : referencedInputFields(optimizedRemainingFilter_)) {
      addReadColumn(name);
    }

    // TODO: Prune struct columns to the subfields referenced by the remaining
    // filter; currently the whole column is read even if only one field is
    // used.

    // The filter is already optimized and constant folded above, so compile it
    // directly.
    auto const remainingFilterType = getTableRowType();
    cudfRemainingFilterExpression_ = createCudfExpression(
        optimizedRemainingFilter_, remainingFilterType, pool_);
  }

  // Build a combined AST for all subfield filters once. This is query-constant
  // and doesn't depend on split-specific state.
  if (!readerSubfieldFilters.empty()) {
    auto const readerFilterType = getTableRowType();
    subfieldFilterExpr_ = &createAstFromSubfieldFilters(
        readerSubfieldFilters,
        subfieldTree_,
        subfieldScalars_,
        readerFilterType);
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();

  // Whether to use the experimental cuDF reader
  useExperimentalCudfReader_ =
      cudfHiveConfig_->useExperimentalCudfReaderSession(
          connectorQueryCtx_->sessionProperties());
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_);
}

void CudfHiveDataSource::convertSplit(std::shared_ptr<ConnectorSplit> split) {
  // The cuDF connector receives CudfHiveConnectorSplit instances from the
  // Gluten split builders (including the MPP coordinator). Keep the cast
  // free of RTTI for the same reason as the async DataSource handoff below.
  if (split->connectorId == "cudf-hive") {
    split_ = std::static_pointer_cast<CudfHiveConnectorSplit>(split);
    VELOX_CHECK_NOT_NULL(split_, "Null CudfHiveConnectorSplit");
    return;
  }

  // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
  auto hiveSplit = checkedPointerCast<hive::HiveConnectorSplit>(split);

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
    // KvikIO does not support "s3a:" prefix. We need to translate it to "s3:".
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
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Virtual method for class-specific conversion of the split
  convertSplit(split);

  cudfSplitReader_ = createCudfSplitReader();
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
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
    // Unable to get the file size, log a warning and continue
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

void CudfHiveDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  // The TableScan preloader creates this DataSource through the same
  // CudfHiveConnector instance, so the source type is fixed by the connector
  // factory. Avoid RTTI here: this handoff runs across the async preload
  // boundary, and the combined Gluten/Velox build can load incompatible RTTI
  // metadata for this connector.
  VELOX_CHECK_NOT_NULL(sourceUnique, "Null DataSource");
  auto* source = static_cast<CudfHiveDataSource*>(sourceUnique.get());

  split_ = std::move(source->split_);
  runtimeStats_ = std::move(source->runtimeStats_);
  completedRows_ += source->completedRows_;
  completedBytes_ += source->completedBytes_;
  cudfSplitReader_ = std::move(source->cudfSplitReader_);
  optimizedRemainingFilter_ = std::move(source->optimizedRemainingFilter_);
  cudfRemainingFilterExpression_ =
      std::move(source->cudfRemainingFilterExpression_);
  // The preloaded reader keeps a non-owning pointer into the source's AST
  // tree. Move the owning tree and scalars together with the reader so those
  // pointers remain valid after the preloaded DataSource is destroyed.
  subfieldTree_ = std::move(source->subfieldTree_);
  subfieldScalars_ = std::move(source->subfieldScalars_);
  subfieldFilterExpr_ = source->subfieldFilterExpr_;
  totalRemainingFilterTime_.fetch_add(
      source->totalRemainingFilterTime_.load(std::memory_order_relaxed),
      std::memory_order_relaxed);

  source->ioStatistics_->merge(*ioStatistics_);
  ioStatistics_ = std::move(source->ioStatistics_);
  source->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(source->ioStats_);

  if (cudfSplitReader_) {
    cudfSplitReader_->setDataSourceContext(
        connectorQueryCtx_, runtimeStats_, subfieldFilterExpr_);
  }
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  CudaAllocationTraceScope allocationTrace(
      fmt::format(
          "CudfHiveDataSource instance={} reader={} method=next",
          static_cast<const void*>(this),
          static_cast<const void*>(cudfSplitReader_.get())));
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  auto chunkOpt = cudfSplitReader_->next(size);
  if (!chunkOpt.has_value()) {
    return nullptr;
  }
  auto cudfTable = std::move(chunkOpt.value());
  auto stream = cudfSplitReader_->stream();

  uint64_t filterTimeUs{0};
  if (optimizedRemainingFilter_) {
    MicrosecondWallTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult =
        cudfRemainingFilterExpression_->eval(inputViews, stream, get_temp_mr());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    cudfTable = cudf::apply_boolean_mask(
        *originalTable, asView(filterResult), stream, get_output_mr());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  const auto nRows = cudfTable->num_rows();

  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(outputColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  }

  // TODO (dm): Should we only enable table scan if cudf is registered?
  // Earlier we could enable cudf table scans without using other cudf operators
  // We still can, but I'm wondering if this is the right thing to do
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  stream.synchronize();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  facebook::velox::connector::hive::addIoStatsToRuntimeStats(
      *ioStatistics_, "", result);
  if (numFilesCoalesced_ > 0) {
    result.emplace("numFilesCoalesced", RuntimeMetric(numFilesCoalesced_));
  }
  result.insert({
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  const auto& ioStats = ioStats_->stats();
  for (const auto& storageStats : ioStats) {
    result.emplace(storageStats.first, storageStats.second);
  }
  return result;
}

const RowTypePtr CudfHiveDataSource::getTableRowType() {
  if (cachedTableRowType_) {
    return cachedTableRowType_;
  }
  if (tableHandle_->dataColumns()) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (const auto& name : readColumnNames_) {
      auto parsedType = tableHandle_->dataColumns()->findChild(name);
      names.emplace_back(std::move(name));
      types.push_back(parsedType);
    }
    cachedTableRowType_ = ROW(std::move(names), std::move(types));
    return cachedTableRowType_;
  }
  cachedTableRowType_ = outputType_;
  return cachedTableRowType_;
}

std::string CudfHiveDataSource::toTopLevelReadColumnName(
    std::string_view name) const {
  if (tableHandle_ && tableHandle_->dataColumns()) {
    const auto& dataColumns = tableHandle_->dataColumns();
    if (dataColumns->containsChild(name)) {
      return std::string{name};
    }

    const auto dot = name.find('.');
    if (dot != std::string_view::npos) {
      const auto topLevelName = name.substr(0, dot);
      if (dataColumns->containsChild(topLevelName)) {
        return std::string{topLevelName};
      }
    }
  }

  return std::string{name};
}

} // namespace facebook::velox::cudf_velox::connector::hive
