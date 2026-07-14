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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/merge.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>

#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr uint64_t kMergePassBytes = 128ULL << 20;
constexpr uint64_t kSpillRowGroupBytes = 64ULL << 20;
constexpr uint64_t kOutputChunkBytes = 32ULL << 20;
constexpr size_t kFinalMergeRuns = 2;
constexpr cudf::size_type kMaxOutputRows = 262144;

std::atomic<uint64_t> orderBySpillDirectorySequence{0};
std::atomic<uint64_t> testingSortedRunBytes{0};
std::atomic<size_t> testingMergeFanIn{0};
std::atomic<uint64_t> mergeChunkBytes{kMergeChunkBytes};
std::atomic<uint64_t> outputChunkBytes{kOutputChunkBytes};
std::atomic<cudf::size_type> maxOutputRows{kMaxOutputRows};

// Read-only diagnostics used to assert the external-sort bounds in GPU tests.
// They never participate in operator scheduling or correctness.
std::atomic<uint64_t> observedMaxActiveRuns{0};
std::atomic<uint64_t> observedSourceChunks{0};
std::atomic<uint64_t> observedMergeOutputBatches{0};
std::atomic<uint64_t> observedEmittedChunks{0};
std::atomic<uint64_t> observedSpillCleanups{0};

bool isSpillSafeType(const TypePtr& type) {
  if (type == nullptr || type->providesCustomComparison()) {
    return false;
  }

  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
      return true;
    case TypeKind::INTEGER:
      // DATE is supported, but interval year-month has not been validated
      // through the internal Parquet representation.
      return !type->isIntervalYearMonth();
    case TypeKind::BIGINT:
      // Plain BIGINT and short DECIMAL are supported. Interval day-time has
      // not been validated through the internal Parquet representation.
      return !type->isIntervalDayTime();
    case TypeKind::HUGEINT:
      return type->isDecimal();
    case TypeKind::TIMESTAMP:
      switch (CudfConfig::getInstance().timestampUnit) {
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
          return true;
        default:
          // Parquet changes TIMESTAMP_SECONDS to milliseconds unless the
          // reader is given an explicit target, making spill data-dependent.
          return false;
      }
    case TypeKind::ARRAY:
      return type->size() == 1 && isSpillSafeType(type->childAt(0));
    case TypeKind::ROW:
      if (type->size() == 0) {
        return false;
      }
      for (uint32_t i = 0; i < type->size(); ++i) {
        if (!isSpillSafeType(type->childAt(i))) {
          return false;
        }
      }
      return true;
    case TypeKind::VARBINARY:
    case TypeKind::MAP:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      return false;
  }
  return false;
}

bool isSupportedSortKeyType(const TypePtr& type) {
  if (!isSpillSafeType(type) || !type->isOrderable()) {
    return false;
  }

  // Nested cuDF sort semantics have not been validated against Velox. Nested
  // values may still be carried as payload when every leaf is spill-safe.
  return type->kind() != TypeKind::ARRAY && type->kind() != TypeKind::ROW;
}

void updateAtomicMax(std::atomic<uint64_t>& target, uint64_t value) {
  auto current = target.load();
  while (current < value &&
         !target.compare_exchange_weak(current, value)) {
  }
}

void resetTestingStats() {
  observedMaxActiveRuns.store(0);
  observedSourceChunks.store(0);
  observedMergeOutputBatches.store(0);
  observedEmittedChunks.store(0);
  observedSpillCleanups.store(0);
}

std::unique_ptr<cudf::table> copyTableSlice(
    cudf::table_view input,
    cudf::size_type begin,
    cudf::size_type end,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LE(begin, end);
  auto slices = cudf::slice(input, {begin, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);
  return std::make_unique<cudf::table>(slices.front(), stream, mr);
}

cudf::size_type firstSearchPosition(
    cudf::column_view positions,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(positions.size(), 1);
  cudf::size_type result{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &result,
      positions.data<cudf::size_type>(),
      sizeof(result),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return result;
}

} // namespace

bool CudfOrderBy::isSupported(
    const RowTypePtr& outputType,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys) {
  if (outputType == nullptr || sortingKeys.empty() ||
      !isSpillSafeType(outputType)) {
    return false;
  }

  return std::all_of(
      sortingKeys.begin(), sortingKeys.end(), [](const auto& key) {
        return key != nullptr && isSupportedSortKeyType(key->type());
      });
}

bool CudfOrderBy::isSupported(
    const std::shared_ptr<const core::OrderByNode>& orderByNode) {
  return orderByNode != nullptr &&
      isSupported(orderByNode->outputType(), orderByNode->sortingKeys());
}

void CudfOrderBy::testingSetMemoryLimits(
    uint64_t runBytes,
    uint64_t chunkBytes,
    uint64_t outputBytes,
    cudf::size_type outputRows,
    size_t fanIn) {
  VELOX_CHECK_GT(runBytes, 0);
  VELOX_CHECK_GT(chunkBytes, 0);
  VELOX_CHECK_GT(outputBytes, 0);
  VELOX_CHECK_GT(outputRows, 0);
  VELOX_CHECK_GE(fanIn, 2);
  testingSortedRunBytes.store(runBytes);
  testingMergeFanIn.store(fanIn);
  mergeChunkBytes.store(chunkBytes);
  outputChunkBytes.store(outputBytes);
  maxOutputRows.store(outputRows);
  resetTestingStats();
}

void CudfOrderBy::testingResetMemoryLimits() {
  testingSortedRunBytes.store(0);
  testingMergeFanIn.store(0);
  mergeChunkBytes.store(kMergeChunkBytes);
  outputChunkBytes.store(kOutputChunkBytes);
  maxOutputRows.store(kMaxOutputRows);
  resetTestingStats();
}

uint64_t CudfOrderBy::testingMaxActiveRuns() {
  return observedMaxActiveRuns.load();
}

uint64_t CudfOrderBy::testingSourceChunks() {
  return observedSourceChunks.load();
}

uint64_t CudfOrderBy::testingMergeOutputBatches() {
  return observedMergeOutputBatches.load();
}

uint64_t CudfOrderBy::testingEmittedChunks() {
  return observedEmittedChunks.load();
}

uint64_t CudfOrderBy::testingSpillCleanups() {
  return observedSpillCleanups.load();
}

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderByNode>& orderByNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          orderByNode->outputType(),
          orderByNode->id(),
          "CudfOrderBy",
          nvtx3::rgb{64, 224, 208}, // Turquoise
          NvtxMethodFlag::kAll,
          std::nullopt,
          orderByNode),
      orderByNode_(orderByNode),
      stateStream_(cudfGlobalStreamPool().get_stream()),
      sortedRunBytes_(
          testingSortedRunBytes.load() > 0
              ? testingSortedRunBytes.load()
              : CudfConfig::getInstance().orderBySortedRunBytes),
      mergeFanIn_(
          testingMergeFanIn.load() > 0
              ? testingMergeFanIn.load()
              : static_cast<size_t>(
                    CudfConfig::getInstance().orderByMergeFanIn)) {
  VELOX_CHECK(
      isSupported(orderByNode),
      "CudfOrderBy received an unsupported external-spill schema or sorting "
      "key type");
  sortKeys_.reserve(orderByNode->sortingKeys().size());
  columnOrder_.reserve(orderByNode->sortingKeys().size());
  nullOrder_.reserve(orderByNode->sortingKeys().size());
  for (int i = 0; i < orderByNode->sortingKeys().size(); ++i) {
    const auto channel =
        exec::exprToChannel(orderByNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "OrderBy doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    auto const& sortingOrder = orderByNode->sortingOrders()[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfOrderBy::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  try {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

    const auto inputStream = cudfInput->stream();
    if (inputStream.value() != stateStream_.value()) {
      std::vector<rmm::cuda_stream_view> inputStreams{inputStream};
      cudf::detail::join_streams(inputStreams, stateStream_);
    }
    // A packed-table backing buffer can retain a different deallocation stream
    // even when the vector's logical stream already equals stateStream_.
    VELOX_CHECK(
        cudfInput->rebindStream(stateStream_),
        "CudfOrderBy cannot rebind its input to the state stream");

    bufferedBytes_ += cudfInput->estimateFlatSize();
    inputs_.push_back(std::move(cudfInput));
    if (bufferedBytes_ >= sortedRunBytes_) {
      spillSortedRun();
    }
  } catch (...) {
    cleanupSpillStateAfterFailure("addInput");
    throw;
  }
}

void CudfOrderBy::doNoMoreInput() {
  Operator::noMoreInput();

  try {
    if (spilled_ && !inputs_.empty()) {
      spillSortedRun();
    }
    if (spilled_) {
      prepareSpilledOutput();
      return;
    }

    if (inputs_.empty()) {
      finished_ = true;
      return;
    }

    auto input = getConcatenatedTable(
        std::exchange(inputs_, {}), outputType_, stateStream_, get_output_mr());
    bufferedBytes_ = 0;
    VELOX_CHECK_NOT_NULL(input);

    auto sorted = cudf::sort_by_key(
        input->view(),
        input->view().select(sortKeys_),
        columnOrder_,
        nullOrder_,
        stateStream_,
        get_output_mr());
    setPendingOutput(std::move(sorted));
  } catch (...) {
    cleanupSpillStateAfterFailure("noMoreInput");
    throw;
  }
}

RowVectorPtr CudfOrderBy::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  try {
    if (auto output = takePendingOutput()) {
      return output;
    }

    if (!spilled_) {
      finished_ = true;
      return nullptr;
    }

    prepareSpilledOutput();
    auto merged = mergeNextSortedBatch();
    if (merged && merged->num_rows() > 0) {
      setPendingOutput(std::move(merged));
      return takePendingOutput();
    }

    finished_ = true;
    cleanupSpillFiles();
    return nullptr;
  } catch (...) {
    cleanupSpillStateAfterFailure("getOutput");
    throw;
  }
}

void CudfOrderBy::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }

  namespace fs = std::filesystem;
  if (!spilled_) {
    const auto sequence = orderBySpillDirectorySequence.fetch_add(1);
    spillDirectory_ = (fs::temp_directory_path() /
                       fmt::format(
                           "velox-cudf-orderby-spill-{}-{}",
                           static_cast<int64_t>(::getpid()),
                           sequence))
                          .string();
    fs::create_directories(spillDirectory_);
    spilled_ = true;
  }

  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.concatenate.begin "
      "bufferedBytes={} bufferedInputs={} existingRuns={} "
      "sortedRunBytes={} mergeFanIn={}",
      orderByNode_->id(),
      bufferedBytes_,
      inputs_.size(),
      sortedRuns_.size(),
      sortedRunBytes_,
      mergeFanIn_));
  auto input = getConcatenatedTable(
      std::exchange(inputs_, {}), outputType_, stateStream_, get_output_mr());
  bufferedBytes_ = 0;
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.concatenate.end rows={}",
      orderByNode_->id(),
      input->num_rows()));
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.sort.begin rows={}",
      orderByNode_->id(),
      input->num_rows()));
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(sortKeys_),
      columnOrder_,
      nullOrder_,
      stateStream_,
      get_output_mr());
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.sort.end rows={}",
      orderByNode_->id(),
      sorted->num_rows()));

  auto path = fmt::format(
      "{}/run-{:06}.parquet", spillDirectory_, spillFileSequence_++);
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{path}, sorted->view())
                     .row_group_size_bytes(kSpillRowGroupBytes)
                     .build();
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.write.begin rows={} "
      "existingRuns={} path={}",
      orderByNode_->id(),
      sorted->num_rows(),
      sortedRuns_.size(),
      path));
  cudf::io::write_parquet(options, stateStream_);

  SortedRun run;
  run.path = std::move(path);
  sortedRuns_.push_back(std::move(run));
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=sortRun.write.end rows={} runs={} "
      "path={}",
      orderByNode_->id(),
      sorted->num_rows(),
      sortedRuns_.size(),
      sortedRuns_.back().path));
  ::malloc_trim(0);
}

uint64_t CudfOrderBy::measureTableBytes(
    std::unique_ptr<cudf::table>& table,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(table);
  auto vector = std::make_shared<CudfVector>(
      pool(), outputType_, table->num_rows(), std::move(table), stream);
  const auto bytes = vector->estimateFlatSize();
  table = vector->release();
  return bytes;
}

bool CudfOrderBy::loadPausedChunk(
    SortedRun& run,
    rmm::cuda_stream_view stream,
    MergeStats& stats) {
  VELOX_CHECK(stream.value() == stateStream_.value());
  VELOX_CHECK_NOT_NULL(run.reader);

  if (run.chunk && run.chunkOffset < run.chunk->num_rows()) {
    return true;
  }
  run.chunk.reset();
  run.chunkOffset = 0;
  run.chunkBytes = 0;

  while (run.reader->has_next()) {
    auto chunk = run.reader->read_chunk();
    ++stats.sourceChunks;
    observedSourceChunks.fetch_add(1);
    stats.sourceRows += chunk.tbl->num_rows();
    if (chunk.tbl->num_rows() == 0) {
      continue;
    }
    run.chunk = std::move(chunk.tbl);
    run.chunkBytes = measureTableBytes(run.chunk, stream);
    stats.sourceBytes += run.chunkBytes;
    return true;
  }
  return false;
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeNextPausedBatch(
    std::vector<SortedRun*>& runs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finished,
    MergeStats& stats) {
  VELOX_CHECK(stream.value() == stateStream_.value());
  if (finished) {
    return nullptr;
  }

  std::vector<SortedRun*> activeRuns;
  std::vector<cudf::table_view> remainingViews;
  activeRuns.reserve(runs.size());
  remainingViews.reserve(runs.size());
  uint64_t residentRows{0};
  uint64_t residentBytes{0};
  for (auto* run : runs) {
    VELOX_CHECK_NOT_NULL(run);
    if (!loadPausedChunk(*run, stream, stats)) {
      continue;
    }
    auto slices = cudf::slice(
        run->chunk->view(),
        {run->chunkOffset, run->chunk->num_rows()},
        stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    activeRuns.push_back(run);
    remainingViews.push_back(slices.front());
    residentRows += slices.front().num_rows();
    // Count the complete owning chunk, which safely overestimates a suffix.
    residentBytes += run->chunkBytes;
  }

  stats.maxActiveRuns =
      std::max<uint64_t>(stats.maxActiveRuns, activeRuns.size());
  stats.maxResidentRows =
      std::max(stats.maxResidentRows, residentRows);
  stats.maxResidentBytes =
      std::max(stats.maxResidentBytes, residentBytes);
  updateAtomicMax(observedMaxActiveRuns, activeRuns.size());
  VELOX_CHECK_LE(
      activeRuns.size(),
      mergeFanIn_,
      "CudfOrderBy opened more readers than its configured merge fan-in");

  if (activeRuns.empty()) {
    finished = true;
    return nullptr;
  }

  std::vector<cudf::table_view> safeViews;
  std::vector<cudf::size_type> consumed(activeRuns.size(), 0);
  if (activeRuns.size() == 1) {
    safeViews.push_back(remainingViews.front());
    consumed.front() = remainingViews.front().num_rows();
  } else {
    // The smallest current tail is a safe global boundary: unread rows in
    // every sorted run compare after (or equal to) it. Keep each unconsumed
    // suffix in its owning reader chunk instead of copying a growing carry.
    std::vector<cudf::table_view> boundaryRows;
    boundaryRows.reserve(remainingViews.size());
    for (const auto& view : remainingViews) {
      auto last = cudf::slice(
          view, {view.num_rows() - 1, view.num_rows()}, stream);
      boundaryRows.push_back(last.front());
    }
    auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
    auto sortedBoundaries = cudf::sort_by_key(
        boundaryCandidates->view(),
        boundaryCandidates->view().select(sortKeys_),
        columnOrder_,
        nullOrder_,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);

    for (size_t index = 0; index < remainingViews.size(); ++index) {
      auto positions = cudf::upper_bound(
          remainingViews[index].select(sortKeys_),
          boundary.front().select(sortKeys_),
          columnOrder_,
          nullOrder_,
          stream,
          mr);
      consumed[index] = firstSearchPosition(positions->view(), stream);
      if (consumed[index] == 0) {
        continue;
      }
      auto safe =
          cudf::slice(remainingViews[index], {0, consumed[index]}, stream);
      safeViews.push_back(safe.front());
    }
  }

  VELOX_CHECK(!safeViews.empty(), "Paused OrderBy merge made no progress");
  auto output = safeViews.size() == 1
      ? std::make_unique<cudf::table>(safeViews.front(), stream, mr)
      : cudf::merge(
            safeViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);

  for (size_t index = 0; index < activeRuns.size(); ++index) {
    activeRuns[index]->chunkOffset += consumed[index];
  }

  ++stats.outputBatches;
  observedMergeOutputBatches.fetch_add(1);
  stats.outputRows += output->num_rows();
  const auto batchBytes = measureTableBytes(output, stream);
  stats.outputBytes += batchBytes;
  stats.maxOutputBytes = std::max(stats.maxOutputBytes, batchBytes);

  finished = true;
  for (auto* run : runs) {
    if ((run->chunk && run->chunkOffset < run->chunk->num_rows()) ||
        (run->reader && run->reader->has_next())) {
      finished = false;
      break;
    }
  }
  return output;
}

void CudfOrderBy::compactSortedRunsForMerge() {
  MergeStats compactionStats;
  while (sortedRuns_.size() > kFinalMergeRuns) {
    const auto inputRunCount = sortedRuns_.size();
    std::vector<SortedRun> nextLevel;
    nextLevel.reserve((sortedRuns_.size() + mergeFanIn_ - 1) / mergeFanIn_);
    std::vector<std::string> obsoletePaths;
    MergeStats levelStats;

    for (size_t begin = 0; begin < sortedRuns_.size(); begin += mergeFanIn_) {
      const auto end = std::min(sortedRuns_.size(), begin + mergeFanIn_);
      if (end - begin == 1) {
        nextLevel.push_back(std::move(sortedRuns_[begin]));
        continue;
      }

      std::vector<SortedRun*> runs;
      runs.reserve(end - begin);
      for (size_t index = begin; index < end; ++index) {
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{sortedRuns_[index].path})
                           .build();
        auto& run = sortedRuns_[index];
        run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
            mergeChunkBytes.load(),
            kMergePassBytes,
            options,
            stateStream_,
            get_output_mr());
        run.chunk.reset();
        run.chunkOffset = 0;
        run.chunkBytes = 0;
        runs.push_back(&run);
      }

      const auto outputPath = fmt::format(
          "{}/merge-{:06}.parquet", spillDirectory_, spillFileSequence_++);
      auto writerOptions =
          cudf::io::chunked_parquet_writer_options::builder(
              cudf::io::sink_info{outputPath})
              .row_group_size_bytes(kSpillRowGroupBytes)
              .build();
      cudf::io::chunked_parquet_writer writer(writerOptions, stateStream_);
      bool groupFinished{false};
      while (!groupFinished) {
        auto merged = mergeNextPausedBatch(
            runs,
            stateStream_,
            get_output_mr(),
            groupFinished,
            levelStats);
        if (merged && merged->num_rows() > 0) {
          writer.write(merged->view());
        }
      }
      writer.close();

      for (size_t index = begin; index < end; ++index) {
        auto& run = sortedRuns_[index];
        run.reader.reset();
        run.chunk.reset();
        run.chunkOffset = 0;
        run.chunkBytes = 0;
        obsoletePaths.push_back(run.path);
      }
      SortedRun outputRun;
      outputRun.path = outputPath;
      nextLevel.push_back(std::move(outputRun));
    }

    // Complete I/O and stream-ordered frees before deleting the previous level.
    stateStream_.synchronize();
    for (const auto& path : obsoletePaths) {
      std::error_code error;
      std::filesystem::remove(path, error);
      if (error) {
        LOG(WARNING) << "CudfOrderBy failed to remove compacted run path="
                     << path << " error=" << error.message();
      }
    }
    sortedRuns_ = std::move(nextLevel);

    compactionStats.sourceChunks += levelStats.sourceChunks;
    compactionStats.sourceRows += levelStats.sourceRows;
    compactionStats.sourceBytes += levelStats.sourceBytes;
    compactionStats.outputBatches += levelStats.outputBatches;
    compactionStats.outputRows += levelStats.outputRows;
    compactionStats.outputBytes += levelStats.outputBytes;
    compactionStats.maxResidentRows = std::max(
        compactionStats.maxResidentRows, levelStats.maxResidentRows);
    compactionStats.maxResidentBytes = std::max(
        compactionStats.maxResidentBytes, levelStats.maxResidentBytes);
    compactionStats.maxOutputBytes = std::max(
        compactionStats.maxOutputBytes, levelStats.maxOutputBytes);
    compactionStats.maxActiveRuns = std::max(
        compactionStats.maxActiveRuns, levelStats.maxActiveRuns);

    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfOrderBy node={} state=compaction.level.end "
        "inputRuns={} outputRuns={} sourceChunks={} outputBatches={} "
        "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
        orderByNode_->id(),
        inputRunCount,
        sortedRuns_.size(),
        levelStats.sourceChunks,
        levelStats.outputBatches,
        levelStats.maxResidentBytes,
        levelStats.maxOutputBytes,
        levelStats.maxActiveRuns));
  }

  stateStream_.synchronize();
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=compaction.end runs={} "
      "sourceChunks={} outputBatches={} maxResidentBytes={} "
      "maxOutputBytes={} maxActiveRuns={}",
      orderByNode_->id(),
      sortedRuns_.size(),
      compactionStats.sourceChunks,
      compactionStats.outputBatches,
      compactionStats.maxResidentBytes,
      compactionStats.maxOutputBytes,
      compactionStats.maxActiveRuns));
}

void CudfOrderBy::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  VELOX_CHECK_LE(sortedRuns_.size(), kFinalMergeRuns);
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        mergeChunkBytes.load(),
        kMergePassBytes,
        options,
        stateStream_,
        get_output_mr());
    run.chunk.reset();
    run.chunkOffset = 0;
    run.chunkBytes = 0;
  }
  readersInitialized_ = true;
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfOrderBy node={} state=output.merge.begin runs={} "
      "chunkReadLimit={} passReadLimit={}",
      orderByNode_->id(),
      sortedRuns_.size(),
      mergeChunkBytes.load(),
      kMergePassBytes));
}

void CudfOrderBy::prepareSpilledOutput() {
  if (readersInitialized_) {
    return;
  }
  compactSortedRunsForMerge();
  initializeSortedRunReaders();
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeNextSortedBatch() {
  if (mergeFinished_) {
    return nullptr;
  }
  std::vector<SortedRun*> runs;
  runs.reserve(sortedRuns_.size());
  for (auto& run : sortedRuns_) {
    runs.push_back(&run);
  }
  auto result = mergeNextPausedBatch(
      runs,
      stateStream_,
      get_output_mr(),
      mergeFinished_,
      outputMergeStats_);
  if (mergeFinished_) {
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfOrderBy node={} state=output.merge.end runs={} "
        "sourceChunks={} sourceRows={} sourceBytes={} outputBatches={} "
        "outputRows={} outputBytes={} maxResidentRows={} "
        "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
        orderByNode_->id(),
        sortedRuns_.size(),
        outputMergeStats_.sourceChunks,
        outputMergeStats_.sourceRows,
        outputMergeStats_.sourceBytes,
        outputMergeStats_.outputBatches,
        outputMergeStats_.outputRows,
        outputMergeStats_.outputBytes,
        outputMergeStats_.maxResidentRows,
        outputMergeStats_.maxResidentBytes,
        outputMergeStats_.maxOutputBytes,
        outputMergeStats_.maxActiveRuns));
  }
  return result;
}

void CudfOrderBy::setPendingOutput(std::unique_ptr<cudf::table> output) {
  VELOX_CHECK(!pendingOutput_);
  if (!output || output->num_rows() == 0) {
    return;
  }
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = measureTableBytes(output, stateStream_);
  pendingOutput_ = std::move(output);
}

CudfVectorPtr CudfOrderBy::takePendingOutput() {
  if (!pendingOutput_) {
    return nullptr;
  }

  const auto totalRows = pendingOutput_->num_rows();
  VELOX_CHECK_LT(pendingOutputOffset_, totalRows);
  const auto remainingRows = totalRows - pendingOutputOffset_;
  const auto byteLimit = outputChunkBytes.load();
  cudf::size_type targetRows =
      std::min(remainingRows, maxOutputRows.load());
  if (pendingOutputBytes_ > byteLimit) {
    const auto proportionalRows = static_cast<cudf::size_type>(
        std::max<uint64_t>(
            1,
            static_cast<uint64_t>(totalRows) * byteLimit /
                pendingOutputBytes_));
    targetRows = std::min(targetRows, proportionalRows);
  }

  while (true) {
    auto chunk = copyTableSlice(
        pendingOutput_->view(),
        pendingOutputOffset_,
        pendingOutputOffset_ + targetRows,
        stateStream_,
        get_output_mr());
    auto output = std::make_shared<CudfVector>(
        pool(), outputType_, targetRows, std::move(chunk), stateStream_);
    const auto actualBytes = output->estimateFlatSize();
    if (actualBytes <= byteLimit || targetRows == 1) {
      if (actualBytes > byteLimit) {
        LOG(WARNING) << "CudfOrderBy node=" << orderByNode_->id()
                     << " emitted one oversized row bytes=" << actualBytes
                     << " byteLimit=" << byteLimit;
      }
      pendingOutputOffset_ += targetRows;
      observedEmittedChunks.fetch_add(1);
      if (pendingOutputOffset_ == totalRows) {
        pendingOutput_.reset();
        pendingOutputOffset_ = 0;
        pendingOutputBytes_ = 0;
      }
      return output;
    }

    const auto proportionalRows = static_cast<cudf::size_type>(
        std::max<uint64_t>(
            1,
            static_cast<uint64_t>(targetRows) * byteLimit / actualBytes));
    targetRows =
        std::min<cudf::size_type>(targetRows - 1, proportionalRows);
  }
}

void CudfOrderBy::cleanupSpillFiles() {
  // Finish reader/writer work before destroying owners, then wait for their
  // stream-ordered frees before removing spill files.
  stateStream_.synchronize();
  sortedRuns_.clear();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;
  stateStream_.synchronize();

  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (error) {
      LOG(WARNING) << "CudfOrderBy failed to remove spill directory path="
                   << spillDirectory_ << " error=" << error.message();
    } else {
      spillDirectory_.clear();
      observedSpillCleanups.fetch_add(1);
    }
  }
  ::malloc_trim(0);
}

void CudfOrderBy::cleanupSpillStateAfterFailure(
    std::string_view context) noexcept {
  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfOrderBy " << context
                 << " pre-destruction cleanup failed: " << error.what();
  }

  inputs_.clear();
  sortedRuns_.clear();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;

  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfOrderBy " << context
                 << " post-destruction cleanup failed: " << error.what();
  }

  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (error) {
      LOG(WARNING) << "CudfOrderBy " << context
                   << " failed to remove spill directory path="
                   << spillDirectory_ << " error=" << error.message();
    } else {
      spillDirectory_.clear();
      observedSpillCleanups.fetch_add(1);
    }
  }
}

void CudfOrderBy::doClose() {
  // close() also runs during task failure; cleanup preserves the original
  // exception while draining and destroying all state-stream owners.
  cleanupSpillStateAfterFailure("close");
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
