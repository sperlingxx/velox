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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileConnectorUtil.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#ifdef VELOX_ENABLE_S3
#include "velox/connectors/hive/storage_adapters/s3fs/S3FileSystem.h"
#endif
#ifdef VELOX_ENABLE_ABFS
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#endif

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/version_config.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <ranges>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

std::size_t multiFileChunkReadLimit(
    std::size_t configuredLimit,
    const config::ConfigBase* session) {
  const auto batchTarget = session->get<uint64_t>(
      CudfConfig::kCudfBatchSizeMinThresholdBytes,
      CudfConfig::getInstance().batchSizeMinThresholdBytes);
  // The compute batch threshold is commonly 8 MiB. Reusing it directly for
  // scans turns a split containing thousands of small files into hundreds of
  // tiny decode calls. Keep the safety bound, but do not shrink the reader
  // below its 256 MiB multi-file default.
  return facebook::velox::cudf_velox::connector::hive::multiFileChunkReadLimit(
      configuredLimit, batchTarget);
}

bool experimentalPrepareIoEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_EXPERIMENTAL_PREPARE_IO");
  return value != nullptr && std::string_view(value) == "1";
}

bool experimentalPrepareHostOnlyEnabled() {
  const auto* value = std::getenv("GLUTEN_CPP_S3_PREPARE_HOST_ONLY");
  return value != nullptr && std::string_view(value) == "1";
}

bool envFlagEnabled(const char* value) {
  return value != nullptr &&
      (std::string_view(value) == "1" || std::string_view(value) == "true");
}

bool adaptiveS3PrefetchEnabled() {
  return envFlagEnabled(std::getenv("GLUTEN_CUDF_S3_ADAPTIVE_PREFETCH"));
}

bool envFlagOrAdaptive(const char* name) {
  const auto* value = std::getenv(name);
  return value == nullptr || *value == '\0' ? adaptiveS3PrefetchEnabled()
                                            : envFlagEnabled(value);
}

bool prepareCacheHintLoadsEnabled() {
  return envFlagOrAdaptive("GLUTEN_CUDF_CACHE_HINT_PREPARE_LOADS");
}

bool cacheHintReadySplitPreloadEnabled() {
  return envFlagOrAdaptive("GLUTEN_CUDF_CACHE_HINT_READY_SPLIT_PRELOAD");
}

bool cacheHintFirstLoadGroupReadyScanEnabled() {
  return envFlagOrAdaptive(
      "GLUTEN_CUDF_CACHE_HINT_FIRST_LOAD_GROUP_READY_SCAN");
}

uint64_t cacheHintFirstLoadGroupReadyMinQueryRegisteredSplits() {
  static const auto minSplits = [] {
    const auto* value = std::getenv(
        "GLUTEN_CUDF_CACHE_HINT_FIRST_LOAD_GROUP_READY_MIN_QUERY_REGISTERED_SPLITS");
    if (value == nullptr || *value == '\0') {
      // Keep smaller scans on the range-ready path while allowing roughly
      // 500-split scans to overlap their first cache load with GPU execution.
      return adaptiveS3PrefetchEnabled() ? uint64_t{500} : uint64_t{0};
    }
    try {
      size_t parsedCharacters{0};
      const auto parsed = std::stoull(value, &parsedCharacters);
      VELOX_CHECK_EQ(
          value[parsedCharacters],
          '\0',
          "Invalid first-load-ready minimum query registered splits: {}",
          value);
      return static_cast<uint64_t>(parsed);
    } catch (const std::exception&) {
      VELOX_FAIL(
          "Invalid first-load-ready minimum query registered splits: {}",
          value);
    }
  }();
  return minSplits;
}

#ifdef VELOX_ENABLE_S3
std::shared_ptr<ReadFile> openNativeS3ReadFile(
    const std::shared_ptr<filesystems::S3FileSystem>& fileSystem,
    const std::string& path,
    std::optional<std::size_t> fileSize) {
  filesystems::FileOptions options;
  if (fileSize.has_value()) {
    options.fileSize = static_cast<int64_t>(fileSize.value());
  }
  return std::shared_ptr<ReadFile>(fileSystem->openFileForRead(path, options));
}
#endif

} // namespace

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

struct CachedPrefetchPlan {
  std::vector<cudf::io::text::byte_range_info> ranges;
  CacheHintRangeStats rangeStats;
};

class CachePrefetchPlanCache {
 public:
  static CachePrefetchPlanCache& instance() {
    static CachePrefetchPlanCache cache;
    return cache;
  }

  bool enabled() const {
    return maxEntries_ > 0;
  }

  std::shared_ptr<const CachedPrefetchPlan> get(const std::string& key) {
    if (maxEntries_ == 0) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = plans_.find(key);
    if (found == plans_.end()) {
      ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(false);
      return nullptr;
    }
    ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(true);
    return found->second;
  }

  std::shared_ptr<const CachedPrefetchPlan> putIfAbsent(
      std::string key,
      std::shared_ptr<const CachedPrefetchPlan> plan) {
    if (maxEntries_ == 0) {
      return plan;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto found = plans_.find(key); found != plans_.end()) {
      return found->second;
    }
    while (plans_.size() >= maxEntries_ && !insertionOrder_.empty()) {
      plans_.erase(insertionOrder_.front());
      insertionOrder_.pop_front();
    }
    insertionOrder_.push_back(key);
    plans_.emplace(std::move(key), plan);
    ExecutorSplitPrefetch::recordCachePrefetchPlanEntries(plans_.size());
    return plan;
  }

 private:
  CachePrefetchPlanCache()
      : maxEntries_([] {
          const auto* value =
              std::getenv("GLUTEN_CUDF_CACHE_PREFETCH_PLAN_CACHE_ENTRIES");
          if (value == nullptr || *value == '\0') {
            return size_t{0};
          }
          try {
            size_t parsedCharacters{0};
            const auto parsed = std::stoull(value, &parsedCharacters);
            if (value[parsedCharacters] != '\0') {
              return size_t{0};
            }
            return static_cast<size_t>(parsed);
          } catch (...) {
            return size_t{0};
          }
        }()) {}

  const size_t maxEntries_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<const CachedPrefetchPlan>>
      plans_;
  std::deque<std::string> insertionOrder_;
};

std::string cachePrefetchPlanKey(
    const std::string& filePath,
    uint64_t start,
    uint64_t length,
    const HiveTableHandle& tableHandle,
    const std::vector<std::string>& readColumnNames) {
  std::ostringstream key;
  key << filePath << '\0' << start << '\0' << length << '\0'
      << tableHandle.toString();
  for (const auto& column : readColumnNames) {
    key << '\0' << column;
  }
  return key.str();
}

// Checks whether the `path` uses an ABFS scheme
bool isAbfsPath([[maybe_unused]] const std::string_view path) {
#ifdef VELOX_ENABLE_ABFS
  return ::facebook::velox::filesystems::isAbfsFile(path);
#else
  return false;
#endif
}

// Rebuilds a struct/list column in-place after possibly transforming (e.g.,
// decimal-casting) its children.
template <typename TransformChildrenFn>
std::unique_ptr<cudf::column> rebuildWithTransformedChildren(
    std::unique_ptr<cudf::column> col,
    TransformChildrenFn&& transformFn) {
  auto const type = col->type();
  auto const size = col->size();
  auto const nullCount = col->null_count();
  auto contents = col->release();
  transformFn(contents.children);
  return std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount,
      std::move(contents.children));
}

// Recursively casts columns to the expected Velox type iff the column is:
//  - Decimal type but not the expected Velox type.
//  - Struct type: with any of its children being decimal type but not the
//  expected Velox type. Rebuilt in place with the casted children.
//  - List type: with its `child` being decimal type but not the expected Velox
//  type. Rebuilt in place with the casted children.
std::unique_ptr<cudf::column> castDecimalColumns(
    std::unique_ptr<cudf::column> col,
    const TypePtr& veloxType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Decimal type (base case)
  if (veloxType->isDecimal()) {
    auto const targetType = veloxToCudfDataType(veloxType);
    if (col->type() != targetType) {
      return cudf::cast(col->view(), targetType, stream, mr);
    }
    return col;
  }

  // Struct type
  if (veloxType->kind() == TypeKind::ROW) {
    auto const& rowType = veloxType->asRow();
    auto const numChildren = static_cast<size_t>(col->num_children());
    VELOX_CHECK_EQ(
        numChildren,
        rowType.size(),
        "Scanned STRUCT column has {} fields but the expected schema has {}.",
        numChildren,
        rowType.size());
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      for (size_t i = 0; i < numChildren; ++i) {
        children[i] = castDecimalColumns(
            std::move(children[i]), rowType.childAt(i), stream, mr);
      }
    });
  }

  // List type
  if (veloxType->kind() == TypeKind::ARRAY) {
    // A LIST column stores [offsets, child]; only the child may hold decimal
    // data.
    VELOX_CHECK_EQ(
        col->num_children(),
        2,
        "LIST column must have exactly 2 children: [offsets, child]");
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      auto const childIdx = cudf::lists_column_view::child_column_index;
      children[childIdx] = castDecimalColumns(
          std::move(children[childIdx]), veloxType->childAt(0), stream, mr);
    });
  }

  return col;
}

std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    const RowTypePtr& rowType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numColumns =
      std::min<size_t>(table->view().num_columns(), rowType->size());
  auto columns = table->release();
  for (size_t i = 0; i < numColumns; ++i) {
    columns[i] = castDecimalColumns(
        std::move(columns[i]), rowType->childAt(i), stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace

CudfSplitReader::CudfSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    bool useExperimentalCudfReader,
    cudf::ast::expression const* subfieldFilterExpr)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241},
          std::nullopt,
          fmt::format("[split:{}]", split ? split->filePath : "unknown")),
      split_(std::move(split)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      cudfHiveConfig_(cudfHiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      useExperimentalCudfReader_(useExperimentalCudfReader),
      baseReaderOpts_(pool_),
      subfieldFilterExpr_(subfieldFilterExpr) {
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);
  facebook::velox::connector::hive::configureReaderOptions(
      std::make_shared<facebook::velox::connector::hive::HiveConfig>(
          cudfHiveConfig_->config()),
      connectorQueryCtx_,
      baseReaderOpts_);
}

CudfSplitReader::~CudfSplitReader() {
  releaseCachePrefetchHint();
}

void CudfSplitReader::setDataSourceContext(
    const ConnectorQueryCtx* connectorQueryCtx,
    dwio::common::RuntimeStatistics& /*runtimeStats*/,
    cudf::ast::expression const* subfieldFilterExpr) {
  connectorQueryCtx_ = connectorQueryCtx;
  subfieldFilterExpr_ = subfieldFilterExpr;
}

void CudfSplitReader::setupReader() {
  if (useExperimentalCudfReader_) {
    createExperimentalReader();
  } else {
    createCudfReader();
  }
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStatistics& /*runtimeStats*/) {
  setupReader();
  if (useExperimentalCudfReader_ && split_->filePath.starts_with("s3://")) {
    if (experimentalPrepareHostOnlyEnabled()) {
      prepareExperimentalHostRead();
    } else if (experimentalPrepareIoEnabled()) {
      setupExperimentalScan();
    }
  }
}

void CudfSplitReader::prepareSplit(
    dwio::common::RuntimeStatistics& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool
  stream_ = cudfGlobalStreamPool().get_stream();

  // Perform split-specific setup.
  prepareSplitInternal(runtimeStats);

  // Update runtime stats
  runtimeStats.processedSplits++;
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::next(
    uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  auto chunkOpt = readNextChunk();
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.value(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return std::move(chunkOpt.value());
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::readNextChunk() {
  auto output_mr = determineCudfMemoryResource();

  if (!useExperimentalCudfReader_) {
    waitForCachePrefetchHint();
    // Read table using the regular cudf parquet reader
    VELOX_CHECK_NOT_NULL(splitReader_, "cudf parquet reader not present");

    if (!splitReader_->has_next()) {
      releaseCachePrefetchHint();
      return std::nullopt;
    }

    auto tableWithMetadata = splitReader_->read_chunk();
    return castDecimalColumnsToVeloxTypes(
        std::move(tableWithMetadata.tbl), outputType_, stream_, output_mr);
  }

  // Read table using the experimental parquet reader
  VELOX_CHECK_NOT_NULL(exptSplitReader_, "cuDF hybrid scan reader not present");
  VELOX_CHECK_NOT_NULL(hybridScanState_, "hybrid scan state not present");

  setupExperimentalScan();

  if (!exptSplitReader_->has_next_table_chunk()) {
    return std::nullopt;
  }

  return std::move(exptSplitReader_->materialize_all_columns_chunk().tbl);
}

void CudfSplitReader::setupCachePrefetchHint() {
  if (useExperimentalCudfReader_ || connectorQueryCtx_->cache() == nullptr ||
      !cudfHiveConfig_->useBufferedInputSession(
          connectorQueryCtx_->sessionProperties()) ||
      readColumnNames_.empty()) {
    return;
  }
  VELOX_CHECK_EQ(fileMetaData_.size(), 1 + coalescedDataSources_.size());

  // Register every member before waiting. Each source keeps its own cache
  // identity, admission bytes and load lifetime; the existing scheduler can
  // overlap files without holding an entire group's budget behind one read.
  for (size_t sourceIndex = 0; sourceIndex < fileMetaData_.size();
       ++sourceIndex) {
    setupCachePrefetchHintForSource(sourceIndex);
  }
}

void CudfSplitReader::setupCachePrefetchHintForSource(size_t sourceIndex) {
  const auto& source =
      sourceIndex == 0 ? dataSource_ : coalescedDataSources_[sourceIndex - 1];
  const auto& filePath = sourceIndex == 0
      ? split_->filePath
      : split_->coalescedFiles[sourceIndex - 1].filePath;
  if (!filePath.starts_with("s3://")) {
    return;
  }
  auto buffered = std::dynamic_pointer_cast<BufferedInputDataSource>(source);
  if (!buffered) {
    return;
  }
  // Multi-source readers select whole files. Use those same bounds in the
  // per-file plan cache, rather than applying the primary split to members.
  const auto start = split_->coalescedFiles.empty() ? split_->start : 0;
  try {
    const auto length =
        split_->coalescedFiles.empty() ? split_->length : source->size();
    auto& planCache = CachePrefetchPlanCache::instance();
    std::string planKey;
    std::shared_ptr<const CachedPrefetchPlan> cachedPlan;
    if (planCache.enabled()) {
      planKey = cachePrefetchPlanKey(
          filePath, start, length, *tableHandle_, readColumnNames_);
      cachedPlan = planCache.get(planKey);
    }
    if (cachedPlan == nullptr) {
      // This reader is a metadata-only planner. It never materializes
      // columns, allocates device input buffers, or changes the regular
      // reader's state.
      CudfHybridScanReader planner(fileMetaData_[sourceIndex], readerOptions_);
      auto rowGroups = planner.all_row_groups(readerOptions_);
      if (readerOptions_.get_skip_bytes() > 0 ||
          readerOptions_.get_num_bytes().has_value()) {
        rowGroups = planner.filter_row_groups_with_byte_range(
            rowGroups, readerOptions_);
      }
      // Match the regular reader's row-group pruning before converting the
      // projected chunks into cache ranges. A planning failure is safe: the
      // surrounding best-effort fallback leaves demand reads authoritative.
      if (readerOptions_.get_filter().has_value()) {
        rowGroups = planner.filter_row_groups_with_stats(
            rowGroups, readerOptions_, stream_);
      }
      auto plan = std::make_shared<CachedPrefetchPlan>();
      plan->ranges =
          planner.all_column_chunks_byte_ranges(rowGroups, readerOptions_);
      plan->rangeStats = buffered->canonicalCacheStats(plan->ranges);
      cachedPlan = planCache.enabled()
          ? planCache.putIfAbsent(planKey, std::move(plan))
          : std::move(plan);
    }
    if (cachedPlan->rangeStats.uniqueBytes == 0) {
      return;
    }

    const auto hintKey = fmt::format(
        "{}:{}:{}:{}", connectorQueryCtx_->scanId(), filePath, start, length);
    cachePrefetchQueryId_ = connectorQueryCtx_->queryId();
    const auto* session = connectorQueryCtx_->sessionProperties();
    const auto prepareLoads = prepareCacheHintLoadsEnabled();
    ExecutorSplitPrefetch::CacheHintLoad load;
    std::shared_ptr<CacheHintFirstLoadSignal> firstLoadReady;
    if (prepareLoads) {
      std::function<void()> firstLoadReadyCallback;
      cachePrefetchFirstLoadPolicyEnabled_ =
          cacheHintFirstLoadGroupReadyScanEnabled();
      cachePrefetchFirstLoadReady_ =
          cachePrefetchFirstLoadPolicyEnabled_ &&
          ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
              executor_,
              *cachePrefetchQueryId_,
              cacheHintFirstLoadGroupReadyMinQueryRegisteredSplits());
      if (cachePrefetchFirstLoadReady_) {
        firstLoadReady = std::make_shared<CacheHintFirstLoadSignal>();
        firstLoadReadyCallback = [firstLoadReady]() {
          firstLoadReady->signal();
        };
      }
      auto prepared = buffered->prepareCachePrefetch(
          cachedPlan->ranges, std::move(firstLoadReadyCallback));
      VELOX_CHECK(
          static_cast<bool>(prepared),
          "Projected cache hint lost its cache-backed input while preparing loads");
      load = [buffered = std::move(buffered),
              prepared = std::move(prepared)]() mutable { prepared(); };
    } else {
      load = [buffered = std::move(buffered),
              ranges = cachedPlan->ranges]() mutable {
        VELOX_CHECK(
            buffered->prefetchToCache(ranges),
            "Projected cache hint lost its cache-backed input");
      };
    }
    cachePrefetchHints_.push_back({hintKey, filePath});
    ExecutorSplitPrefetch::registerCacheHint(
        executor_,
        *cachePrefetchQueryId_,
        hintKey,
        cachedPlan->rangeStats,
        std::move(load),
        cudfHiveConfig_->executorSplitPrefetchConcurrencySession(session),
        cudfHiveConfig_->prefetchMaxInFlightBytesSession(session),
        std::move(firstLoadReady));
  } catch (const std::exception& e) {
    ExecutorSplitPrefetch::recordCacheHintFallback();
    LOG(WARNING) << "Falling back to demand reads after cuDF cache hint "
                    "planning failed for "
                 << filePath << ": " << e.what();
  }
}

void CudfSplitReader::waitForCachePrefetchHint() {
  waitForCachePrefetchHint(/*splitPreload=*/false);
}

void CudfSplitReader::waitForCachePrefetchHint(bool splitPreload) {
  // After preload, prioritize each member's remaining demand reads.
  if (!splitPreload && cachePrefetchHintWaited_ &&
      !cachePrefetchDemandPrioritized_) {
    cachePrefetchDemandPrioritized_ = true;
    for (const auto& hint : cachePrefetchHints_) {
      cachePrefetchDemandPrioritized_ &= prioritizeNativeS3File(hint.filePath);
    }
  }
  if (cachePrefetchHintWaited_ || !cachePrefetchQueryId_ ||
      cachePrefetchHints_.empty()) {
    return;
  }
  cachePrefetchHintWaited_ = true;
  for (const auto& hint : cachePrefetchHints_) {
    if (cachePrefetchFirstLoadPolicyEnabled_) {
      if (!cachePrefetchFirstLoadReady_) {
        ExecutorSplitPrefetch::requestCacheHint(
            executor_, *cachePrefetchQueryId_, hint.key);
        cachePrefetchNonBlockingRequested_ = true;
      } else {
        ExecutorSplitPrefetch::takeCacheHint(
            executor_,
            *cachePrefetchQueryId_,
            hint.key,
            CacheHintWaitMode::kFirstLoadGroup);
      }
    } else {
      ExecutorSplitPrefetch::takeCacheHint(
          executor_,
          *cachePrefetchQueryId_,
          hint.key,
          splitPreload ? CacheHintWaitMode::kSplitPreload
                       : CacheHintWaitMode::kScan);
    }
  }
}

void CudfSplitReader::releaseCachePrefetchHint() {
  if (cachePrefetchNonBlockingRequested_ && cachePrefetchQueryId_) {
    for (const auto& hint : cachePrefetchHints_) {
      ExecutorSplitPrefetch::releaseCacheHint(
          executor_, *cachePrefetchQueryId_, hint.key);
    }
  }
  cachePrefetchQueryId_.reset();
  cachePrefetchHints_.clear();
  cachePrefetchDemandPrioritized_ = false;
  cachePrefetchNonBlockingRequested_ = false;
  cachePrefetchFirstLoadPolicyEnabled_ = false;
  cachePrefetchFirstLoadReady_ = false;
}

void CudfSplitReader::prepareExperimentalHostRead() {
  VELOX_CHECK(
      useExperimentalCudfReader_,
      "Experimental host preparation requested for the regular cuDF reader");
  VELOX_CHECK_NOT_NULL(exptSplitReader_, "cuDF hybrid scan reader not present");
  VELOX_CHECK_NOT_NULL(hybridScanState_, "hybrid scan state not present");
  VELOX_CHECK(
      !hybridScanState_->preparedHostByteRanges_,
      "Experimental host ranges were already prepared");

  auto rowGroupIndices = exptSplitReader_->all_row_groups(readerOptions_);
  if (readerOptions_.get_skip_bytes() > 0 or
      readerOptions_.get_num_bytes().has_value()) {
    rowGroupIndices = exptSplitReader_->filter_row_groups_with_byte_range(
        rowGroupIndices, readerOptions_);
  }
  if (readerOptions_.get_filter().has_value()) {
    rowGroupIndices = exptSplitReader_->filter_row_groups_with_stats(
        rowGroupIndices, readerOptions_, stream_);
  }
  auto columnChunkByteRanges = exptSplitReader_->all_column_chunks_byte_ranges(
      rowGroupIndices, readerOptions_);
  auto prepared = prepareByteRangesToHost(dataSource_, columnChunkByteRanges);
  VELOX_CHECK_NOT_NULL(
      prepared,
      "Host-only experimental preparation requires a packed remote source");
  hybridScanState_->rowGroupIndices_ = std::move(rowGroupIndices);
  hybridScanState_->columnChunkByteRanges_ = std::move(columnChunkByteRanges);
  hybridScanState_->preparedHostByteRanges_ = std::move(prepared);
}

void CudfSplitReader::setupExperimentalScan() {
  VELOX_CHECK(
      useExperimentalCudfReader_,
      "Experimental scan setup requested for the regular cuDF reader");
  VELOX_CHECK_NOT_NULL(exptSplitReader_, "cuDF hybrid scan reader not present");
  VELOX_CHECK_NOT_NULL(hybridScanState_, "hybrid scan state not present");

  std::call_once(*hybridScanState_->isHybridScanSetup_, [&]() {
    auto output_mr = determineCudfMemoryResource();
    std::vector<cudf::size_type> rowGroupIndices;
    std::vector<cudf::io::text::byte_range_info> columnChunkByteRanges;
    if (hybridScanState_->rowGroupIndices_.has_value()) {
      VELOX_CHECK(
          hybridScanState_->columnChunkByteRanges_.has_value(),
          "Prepared row groups are missing projected byte ranges");
      rowGroupIndices = std::move(hybridScanState_->rowGroupIndices_.value());
      columnChunkByteRanges =
          std::move(hybridScanState_->columnChunkByteRanges_.value());
      hybridScanState_->rowGroupIndices_.reset();
      hybridScanState_->columnChunkByteRanges_.reset();
    } else {
      rowGroupIndices = exptSplitReader_->all_row_groups(readerOptions_);

      // Filter row groups using row group byte ranges.
      if (readerOptions_.get_skip_bytes() > 0 or
          readerOptions_.get_num_bytes().has_value()) {
        rowGroupIndices = exptSplitReader_->filter_row_groups_with_byte_range(
            rowGroupIndices, readerOptions_);
      }

      // Filter row groups using column chunk statistics.
      if (readerOptions_.get_filter().has_value()) {
        rowGroupIndices = exptSplitReader_->filter_row_groups_with_stats(
            rowGroupIndices, readerOptions_, stream_);
      }

      columnChunkByteRanges = exptSplitReader_->all_column_chunks_byte_ranges(
          rowGroupIndices, readerOptions_);
    }

    // Fetch column chunk byte ranges
    nvtxRangePush("fetchByteRanges");

    // Tuple containing a vector of device buffers, a vector of device spans
    // for each input byte range, and a future to wait for all reads to
    // complete
    auto ioData = hybridScanState_->preparedHostByteRanges_
        ? copyPreparedByteRangesToDevice(
              std::move(hybridScanState_->preparedHostByteRanges_),
              columnChunkByteRanges,
              stream_,
              get_temp_mr())
        : fetchByteRangesAsync(
              dataSource_, columnChunkByteRanges, stream_, get_temp_mr());

    // Wait for all pending reads to complete and propagate any I/O failure.
    std::get<2>(ioData).get();
    nvtxRangePop();

    // Save state for hybrid scan reader for future calls to `next()`
    hybridScanState_->columnChunkBuffers_ = std::move(std::get<0>(ioData));
    hybridScanState_->columnChunkData_ = std::move(std::get<1>(ioData));

    const auto setupStart = std::chrono::steady_clock::now();
    exptSplitReader_->setup_chunking_for_all_columns(
        cudfHiveConfig_->maxChunkReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        cudfHiveConfig_->maxPassReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        rowGroupIndices,
        hybridScanState_->columnChunkData_,
        readerOptions_,
        stream_,
        output_mr);
    const auto setupNanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - setupStart)
            .count();
    static std::atomic<uint64_t> completed{0};
    static std::atomic<uint64_t> totalSetupNanos{0};
    const auto count = completed.fetch_add(1, std::memory_order_relaxed) + 1;
    totalSetupNanos.fetch_add(setupNanos, std::memory_order_relaxed);
    if (std::getenv("GLUTEN_CPP_S3_DIAGNOSTICS") != nullptr &&
        count % 64 == 0) {
      LOG(WARNING) << "CPP_S3_HYBRID_SETUP completed=" << count
                   << " avgSetupMs="
                   << totalSetupNanos.load(std::memory_order_relaxed) / count /
              1e6;
    }
    // TODO: check remainingFilterExprSet_ flag here to choose mr
  });
}

void CudfSplitReader::resetSplit() {
  releaseCachePrefetchHint();
  splitReader_.reset();
  exptSplitReader_.reset();
  hybridScanState_.reset();
  dataSource_.reset();
  coalescedDataSources_.clear();
  selectivePreloadBuffers_.clear();
  selectivePreloadResult_.reset();
  fileMetaData_.clear();
  cachePrefetchHintWaited_ = false;
  cachePrefetchDemandPrioritized_ = false;
}

cudf::ast::expression const* CudfSplitReader::pushdownFilter() const {
  return subfieldFilter();
}

cudf::ast::expression const* CudfSplitReader::subfieldFilter() const {
  return subfieldFilterExpr_;
}

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_) {
    return;
  }

  std::optional<std::size_t> primaryFileSize;
  if (!split_->coalescedFiles.empty() && split_->start == 0 &&
      split_->length != std::numeric_limits<uint64_t>::max()) {
    primaryFileSize = split_->length;
  }
  dataSource_ = createCudfDataSource(split_->filePath, primaryFileSize);
  coalescedDataSources_.reserve(split_->coalescedFiles.size());
  for (const auto& file : split_->coalescedFiles) {
    coalescedDataSources_.push_back(
        createCudfDataSource(file.filePath, file.length));
  }
}

void CudfSplitReader::setupSelectivePreloadDataSources() {
  // Iceberg needs file metadata while preparing the split. In that path the
  // preload is intentionally set up before schema inspection, and
  // createCudfReader() calls this method again later. Keep the second call a
  // no-op so that we neither fetch the same remote file twice nor replace the
  // pinned data sources and invalidate the cached footer metadata.
  if (selectivePreloadResult_ || !selectivePreloadBuffers_.empty()) {
    return;
  }

  const auto* session = connectorQueryCtx_->sessionProperties();
  const auto enabled = cudfHiveConfig_->selectivePreloadEnabledSession(session);
  if (useExperimentalCudfReader_ || !enabled || readColumnNames_.empty()) {
    return;
  }
  VELOX_CHECK_NOT_NULL(
      executor_,
      "Selective Parquet preload requires a non-zero Velox IOThreads setting");

  auto broker = ExecutorReadBroker::get(
      executor_,
      cudfHiveConfig_->prefetchMaxInFlightBytesSession(session),
      cudfHiveConfig_->prefetchThreadsSession(session));

#ifdef VELOX_ENABLE_S3
  if (split_->filePath.starts_with("s3://") &&
      ExecutorSplitPrefetch::contains(
          executor_, connectorQueryCtx_->queryId(), split_->filePath)) {
    const auto splitConcurrency =
        cudfHiveConfig_->executorSplitPrefetchConcurrencySession(session);
    const auto maxReadyBytes =
        cudfHiveConfig_->prefetchMaxInFlightBytesSession(session);
    ExecutorSplitPrefetch::initialize(
        executor_,
        connectorQueryCtx_->queryId(),
        [config = cudfHiveConfig_->config()](
            const std::string& filePath, std::optional<std::size_t> fileSize) {
          auto fileSystem = filesystems::getFileSystem(filePath, config);
          auto s3FileSystem =
              std::dynamic_pointer_cast<filesystems::S3FileSystem>(fileSystem);
          VELOX_CHECK_NOT_NULL(
              s3FileSystem,
              "S3 path resolved to a non-S3 filesystem: {}",
              filePath);
          const auto credentials = s3FileSystem->getCredentialSnapshot();
          auto source = std::make_shared<KvikioS3DataSource>(
              filePath,
              credentials.accessKeyId,
              credentials.secretAccessKey,
              credentials.sessionToken,
              credentials.region,
              credentials.endpoint,
              openNativeS3ReadFile(s3FileSystem, filePath, fileSize),
              fileSize);
          return PrefetchReadFunction{
              [source = std::move(source), filePath](
                  uint64_t offset, uint64_t size, uint8_t* destination) {
                const auto bytes = source->host_read(offset, size, destination);
                VELOX_CHECK_EQ(
                    bytes,
                    size,
                    "Short datasource read for {} at offset {}",
                    filePath,
                    offset);
              }};
        },
        broker,
        splitConcurrency,
        maxReadyBytes);
    selectivePreloadResult_ = ExecutorSplitPrefetch::take(
        executor_, connectorQueryCtx_->queryId(), split_->filePath);
    if (selectivePreloadResult_) {
      VELOX_CHECK_EQ(
          selectivePreloadResult_->buffers.size(),
          1 + split_->coalescedFiles.size());
      dataSource_.reset();
      coalescedDataSources_.clear();
      selectivePreloadBuffers_.clear();
      coalescedDataSources_.reserve(
          selectivePreloadResult_->buffers.size() - 1);
      selectivePreloadBuffers_.reserve(selectivePreloadResult_->buffers.size());
      for (size_t index = 0; index < selectivePreloadResult_->buffers.size();
           ++index) {
        const auto& buffer = selectivePreloadResult_->buffers[index];
        auto span = cudf::host_span<const std::byte>(
            reinterpret_cast<const std::byte*>(buffer->data()), buffer->size());
        auto source = std::shared_ptr<cudf::io::datasource>(
            cudf::io::datasource::create(span));
        if (index == 0) {
          dataSource_ = std::move(source);
        } else {
          coalescedDataSources_.push_back(std::move(source));
        }
        selectivePreloadBuffers_.push_back(buffer);
      }
      fileMetaData_.clear();
      return;
    }
  }
#endif

  struct PendingFile {
    std::shared_ptr<PinnedHostBuffer> buffer;
    std::future<void> future;
  };

  constexpr uint64_t kReadRangeBytes = 16ULL << 20;
  std::vector<PendingFile> pending;
  pending.reserve(1 + split_->coalescedFiles.size());

  using DataSourceFactory = std::function<std::shared_ptr<cudf::io::datasource>(
      const std::string&, std::optional<std::size_t>)>;
  DataSourceFactory sourceFactory =
      [this](const std::string& path, std::optional<std::size_t> fileSize) {
        return createCudfDataSource(path, fileSize);
      };
#ifdef VELOX_ENABLE_S3
  const auto useBufferedInput = cudfHiveConfig_->useBufferedInputSession(
      connectorQueryCtx_->sessionProperties());
  if (!useBufferedInput && split_->filePath.starts_with("s3://")) {
    sourceFactory = [config = cudfHiveConfig_->config()](
                        const std::string& path,
                        std::optional<std::size_t> fileSize) {
      VELOX_CHECK(
          path.starts_with("s3://"),
          "Grouped S3 split contains a non-S3 path: {}",
          path);
      auto fileSystem = filesystems::getFileSystem(path, config);
      auto s3FileSystem =
          std::dynamic_pointer_cast<filesystems::S3FileSystem>(fileSystem);
      VELOX_CHECK_NOT_NULL(
          s3FileSystem, "S3 path resolved to a non-S3 filesystem: {}", path);
      const auto credentials = s3FileSystem->getCredentialSnapshot();
      return std::make_shared<KvikioS3DataSource>(
          path,
          credentials.accessKeyId,
          credentials.secretAccessKey,
          credentials.sessionToken,
          credentials.region,
          credentials.endpoint,
          openNativeS3ReadFile(s3FileSystem, path, fileSize),
          fileSize);
    };
  }
#endif

  struct PreparedFile {
    std::string path;
    std::optional<std::size_t> knownFileSize;
    std::shared_ptr<cudf::io::datasource> originalSource;
    uint64_t sourceSize;
  };
  std::vector<PreparedFile> prepared;
  prepared.reserve(1 + split_->coalescedFiles.size());
  uint64_t batchBytes = 0;

  auto prepareFile = [&](const std::string& path,
                         uint64_t start,
                         uint64_t length,
                         bool wholeFile) {
    const std::optional<std::size_t> knownFileSize = wholeFile && start == 0 &&
            length != std::numeric_limits<uint64_t>::max()
        ? std::optional<std::size_t>{length}
        : std::nullopt;
    std::shared_ptr<cudf::io::datasource> originalSource;
    const auto sourceSize =
        knownFileSize.has_value() ? knownFileSize.value() : ([&]() {
          originalSource = sourceFactory(path, knownFileSize);
          VELOX_CHECK_NOT_NULL(originalSource);
          return originalSource->size();
        })();
    VELOX_CHECK_LE(
        sourceSize,
        std::numeric_limits<uint64_t>::max() - batchBytes,
        "Selective preload byte count overflow");
    batchBytes += sourceSize;
    prepared.push_back(
        {path, knownFileSize, std::move(originalSource), sourceSize});
  };

  // A split range that begins at zero is not necessarily a whole file. The
  // coalesced-file contract is the only caller-provided proof that these
  // lengths cover complete physical files.
  prepareFile(
      split_->filePath,
      split_->start,
      split_->length,
      !split_->coalescedFiles.empty());
  for (const auto& file : split_->coalescedFiles) {
    prepareFile(file.filePath, 0, file.length, true);
  }

  // Reserve the whole grouped split before allocating any pinned buffer. The
  // broker admits one oversize group only when no other reservation is live.
  auto reservation = broker->reserve(batchBytes);

  // Submit all physical files before waiting for any one of them. This removes
  // the per-file synchronization barrier inside a grouped split and lets the
  // executor-scoped broker maintain one request window across scan drivers.
  auto submitFile = [&](PreparedFile file) {
    auto buffer =
        std::make_shared<PinnedHostBuffer>(file.sourceSize, reservation);

    std::vector<PrefetchRange> ranges;
    ranges.reserve((file.sourceSize + kReadRangeBytes - 1) / kReadRangeBytes);
    for (uint64_t offset = 0; offset < file.sourceSize;
         offset += kReadRangeBytes) {
      ranges.push_back(
          {offset,
           std::min<uint64_t>(kReadRangeBytes, file.sourceSize - offset),
           offset});
    }
    auto makeReadFunction =
        [path = file.path](std::shared_ptr<cudf::io::datasource> source) {
          VELOX_CHECK_NOT_NULL(source);
          return PrefetchReadFunction{
              [source = std::move(source), path](
                  uint64_t offset, uint64_t size, uint8_t* destination) {
                const auto bytes = source->host_read(offset, size, destination);
                VELOX_CHECK_EQ(
                    bytes,
                    size,
                    "Short datasource read for {} at offset {}",
                    path,
                    offset);
              }};
        };
    std::future<void> future;
    if (file.originalSource) {
      // Unknown-size sources must still be opened synchronously to size the
      // destination. Preserve the existing range-parallel path for them.
      future = broker->read(
          makeReadFunction(std::move(file.originalSource)),
          file.sourceSize,
          std::move(ranges),
          buffer,
          reservation);
    } else {
      future = broker->readPrepared(
          [sourceFactory,
           makeReadFunction,
           path = file.path,
           knownFileSize = file.knownFileSize]() mutable {
            return makeReadFunction(sourceFactory(path, knownFileSize));
          },
          file.sourceSize,
          std::move(ranges),
          buffer,
          reservation);
    }
    pending.push_back({std::move(buffer), std::move(future)});
  };

  selectivePreloadBuffers_.clear();
  for (auto& file : prepared) {
    submitFile(std::move(file));
  }

  std::exception_ptr firstFailure;
  for (auto& file : pending) {
    try {
      file.future.get();
    } catch (...) {
      if (!firstFailure) {
        firstFailure = std::current_exception();
      }
    }
  }
  if (firstFailure) {
    std::rethrow_exception(firstFailure);
  }

  dataSource_.reset();
  coalescedDataSources_.clear();
  coalescedDataSources_.reserve(pending.empty() ? 0 : pending.size() - 1);
  selectivePreloadBuffers_.reserve(pending.size());
  for (size_t index = 0; index < pending.size(); ++index) {
    auto& file = pending[index];
    auto span = cudf::host_span<const std::byte>(
        reinterpret_cast<const std::byte*>(file.buffer->data()),
        file.buffer->size());
    auto source = std::shared_ptr<cudf::io::datasource>(
        cudf::io::datasource::create(span));
    if (index == 0) {
      dataSource_ = std::move(source);
    } else {
      coalescedDataSources_.push_back(std::move(source));
    }
    selectivePreloadBuffers_.push_back(std::move(file.buffer));
  }
  fileMetaData_.clear();
}

uint64_t CudfSplitReader::primaryDataSourceSize() const {
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");
  return dataSource_->size();
}

std::shared_ptr<cudf::io::datasource> CudfSplitReader::createCudfDataSource(
    const std::string& filePath,
    std::optional<std::size_t> fileSize) {
  const auto useBufferedInput = cudfHiveConfig_->useBufferedInputSession(
      connectorQueryCtx_->sessionProperties());

  VELOX_CHECK(
      not isAbfsPath(filePath) or useBufferedInput,
      "ABFS blobs require buffered input data source. "
      "Set the session property '{}' (or connector property '{}') to 'true'. "
      "Blob Path: {}.",
      CudfHiveConfig::kUseBufferedInputSession,
      CudfHiveConfig::kUseBufferedInput,
      filePath);

  // Use KvikIO data source if we don't want to use the BufferedInput source
  if (not useBufferedInput) {
#ifdef VELOX_ENABLE_S3
    if (filePath.starts_with("s3://")) {
      const auto useCrtS3Reader = cudfHiveConfig_->useCrtS3ReaderSession(
          connectorQueryCtx_->sessionProperties());
      if (useCrtS3Reader) {
        VELOX_CHECK(
            crtS3RangeReaderAvailable(),
            "AWS CRT S3 reader was requested but its JVM bridge is unavailable");
        VLOG(1) << fmt::format(
            "Using AWS CRT S3 data source for file: {}", filePath);
        return std::make_shared<CrtS3DataSource>(filePath, fileSize);
      }
      auto fileSystem =
          filesystems::getFileSystem(filePath, cudfHiveConfig_->config());
      auto s3FileSystem =
          std::dynamic_pointer_cast<filesystems::S3FileSystem>(fileSystem);
      VELOX_CHECK_NOT_NULL(
          s3FileSystem,
          "S3 path resolved to a non-S3 filesystem: {}",
          filePath);
      const auto credentials = s3FileSystem->getCredentialSnapshot();
      VLOG(1) << fmt::format(
          "Using IRSA-aware KvikIO S3 data source for file: {}", filePath);
      return std::make_shared<KvikioS3DataSource>(
          filePath,
          credentials.accessKeyId,
          credentials.secretAccessKey,
          credentials.sessionToken,
          credentials.region,
          credentials.endpoint,
          openNativeS3ReadFile(s3FileSystem, filePath, fileSize),
          fileSize);
    }
#endif
    VLOG(1) << fmt::format("Using KvikIO data source for file: {}", filePath);
    return std::move(
        cudf::io::make_datasources(cudf::io::source_info{filePath}).front());
  }

  auto fileHandleCachePtr = FileHandleCachedPtr{};
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    // ABFS blobs can not fall back to KvikIO. Throw the original error.
    if (isAbfsPath(filePath)) {
      VELOX_USER_FAIL(
          "Failed to generate file handle cache for ABFS blob. Ensure "
          "registerAbfsFileSystem() and registerAzureClientProvider() have "
          "been called and the connector config provides Azure credentials. "
          "Blob path: {}. Error: {}.",
          filePath,
          e.what());
    }

    LOG(WARNING) << fmt::format(
        "Failed to generate file handle cache for file. Falling back to KvikIO. Path: {}",
        filePath);
    return std::move(
        cudf::io::make_datasources(cudf::io::source_info{filePath}).front());
  }

  // Here we keep adding new entries to CacheTTLController when new
  // fileHandles are generated, if CacheTTLController was created. Creator of
  // CacheTTLController needs to make sure a size control strategy was
  // available such as removing aged out entries.
  if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
    cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
  }

  const FileHandle* bufferedFileHandle = fileHandleCachePtr.get();
#ifdef VELOX_ENABLE_S3
  std::optional<FileHandle> scheduledFileHandle;
  if (connectorQueryCtx_->cache() != nullptr && filePath.starts_with("s3://") &&
      nativeS3ScheduledReadEnabled()) {
    scheduledFileHandle.emplace(
        FileHandle{
            makeNativeScheduledS3ReadFile(bufferedFileHandle->file, filePath),
            bufferedFileHandle->uuid,
            bufferedFileHandle->groupId});
    bufferedFileHandle = &scheduledFileHandle.value();
  }
#endif
  auto bufferedInput =
      velox::connector::hive::BufferedInputBuilder::getInstance()->create(
          *bufferedFileHandle,
          baseReaderOpts_,
          connectorQueryCtx_,
          ioStatistics_,
          ioStats_,
          executor_);
  if (not bufferedInput) {
    // ABFS blobs can not fall back to KvikIO
    if (isAbfsPath(filePath)) {
      VELOX_USER_FAIL(
          "Failed to create buffered input data source for the ABFS blob. Ensure that the registered "
          "BufferedInputBuilder is ABFS-aware. Blob path: {}.",
          filePath);
    }

    LOG(WARNING) << fmt::format(
        "Failed to create buffered input data source for file. Falling back to the KvikIO. Path: {}",
        filePath);
    return std::move(
        cudf::io::make_datasources(cudf::io::source_info{filePath}).front());
  }
  return std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
}

std::vector<std::unique_ptr<cudf::io::datasource>>
CudfSplitReader::makeDataSourceViews() {
  VELOX_CHECK_NOT_NULL(dataSource_);
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.reserve(1 + coalescedDataSources_.size());
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));
  for (const auto& source : coalescedDataSources_) {
    sources.push_back(cudf::io::datasource::create(source.get()));
  }
  return sources;
}

void CudfSplitReader::setupReaderOptions() {
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");
  auto sourceInfo = cudf::io::source_info{dataSource_.get()};

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  // cuDF only supports byte bounds for a single source. Coalesced Iceberg
  // splits contain whole files, so read every source without byte bounds.
  if (split_->coalescedFiles.empty() && split_->start != 0) {
    readerOptions_.set_skip_bytes(split_->start);
  }
  if (split_->coalescedFiles.empty() &&
      split_->size() != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(split_->size());
  }

  if (auto* filter = pushdownFilter(); filter != nullptr) {
    readerOptions_.set_filter(*filter);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  if (prependRowIndex_) {
#if CUDF_VERSION_MAJOR > 26 || \
    (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 8)
    readerOptions_.enable_prepend_row_index_column(true);
#else
    VELOX_FAIL("Prepending a row-index column requires cuDF 26.08 or newer");
#endif
  }
}

rmm::device_async_resource_ref CudfSplitReader::determineCudfMemoryResource()
    const {
  return get_output_mr();
}

void CudfSplitReader::fileMetaDatas() {
  if (not fileMetaData_.empty()) {
    return;
  }

  // Setup the datasource
  setupCudfDataSource();

  // Check that the datasource is set up
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  // Wrap the existing datasources without transferring ownership.
  auto sources = makeDataSourceViews();
  fileMetaData_ = cudf::io::read_parquet_footers(sources);
  VELOX_CHECK_GE(
      fileMetaData_.size(),
      1,
      "CudfSplitReader failed to read any parquet metadatas");
}

void CudfSplitReader::createCudfReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  setupCachePrefetchHint();

  // Make the existing TableScan ready-first selection reflect projected cache
  // range readiness, not just footer parsing and reader construction.
  if (cacheHintReadySplitPreloadEnabled() && !cachePrefetchHints_.empty()) {
    waitForCachePrefetchHint(/*splitPreload=*/true);
  }

  auto sources = makeDataSourceViews();

  auto chunkReadLimit = cudfHiveConfig_->maxChunkReadLimitSession(
      connectorQueryCtx_->sessionProperties());
  if (!split_->coalescedFiles.empty()) {
    // An unbounded reader is safe for one physical file, but a coalesced split
    // can contain many files. Decoding all of them into one table bypasses the
    // downstream byte thresholds and makes a single scan batch consume most
    // of the GPU before aggregation can apply backpressure.
    chunkReadLimit = multiFileChunkReadLimit(
        chunkReadLimit, connectorQueryCtx_->sessionProperties());
  }

  // Create a parquet reader
  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      chunkReadLimit,
      cudfHiveConfig_->maxPassReadLimitSession(
          connectorQueryCtx_->sessionProperties()),
      std::move(sources),
      std::move(fileMetaData_),
      readerOptions_,
      stream_,
      determineCudfMemoryResource());

  // Metadata ingested
  fileMetaData_.clear();
}

void CudfSplitReader::createExperimentalReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "cuDF experimental reader requires exactly one parquet metadata");

  // Create a hybrid scan reader
  nvtxRangePush("hybridScanReader");
  auto reader = std::make_unique<CudfHybridScanReader>(
      std::move(fileMetaData_.front()), readerOptions_);
  nvtxRangePop();

  exptSplitReader_ = std::move(reader);
  hybridScanState_ = std::make_unique<HybridScanState>();

  // Metadata ingested
  fileMetaData_.clear();
}

void CudfSplitReader::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTimeNs(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
