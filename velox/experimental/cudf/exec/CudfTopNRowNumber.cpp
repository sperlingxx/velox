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
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/OperatorUtils.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <malloc.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kCandidateRunBytes = 128ULL << 20;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr size_t kMergeFanIn = 4;
constexpr cudf::size_type kMaxCompleteOutputRows = 262144;
constexpr std::string_view kConditionalTopNMarker = "__gluten_mpp_topn_active";
std::atomic<uint64_t> spillDirectorySequence{0};

bool isSupportedKeyType(const TypePtr& type) {
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

bool CudfTopNRowNumber::shouldReplace(
    const std::shared_ptr<const core::TopNRowNumberNode>& node) {
  if (node == nullptr || node->limit() != 1) {
    return false;
  }
  const auto rankFunction = node->rankFunction();
  if (rankFunction != core::TopNRowNumberNode::RankFunction::kRowNumber &&
      rankFunction != core::TopNRowNumberNode::RankFunction::kRank &&
      rankFunction != core::TopNRowNumberNode::RankFunction::kDenseRank) {
    return false;
  }
  if (rankFunction != core::TopNRowNumberNode::RankFunction::kRowNumber &&
      node->sortingKeys().empty()) {
    return false;
  }

  for (const auto& key : node->partitionKeys()) {
    if (!isSupportedKeyType(key->type())) {
      return false;
    }
  }

  for (const auto& key : node->sortingKeys()) {
    if (!isSupportedKeyType(key->type())) {
      return false;
    }
  }

  return true;
}

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          node->outputType(),
          node->id(),
          "CudfTopNRowNumber",
          nvtx3::rgb{255, 140, 0},
          NvtxMethodFlag::kAll,
          std::nullopt,
          node),
      limit_(node->limit()),
      rankFunction_(node->rankFunction()),
      generateRowNumber_(node->generateRowNumber()),
      inputType_(node->inputType()),
      diagnosticNodeId_(node->id()) {
  VELOX_CHECK_EQ(limit_, 1, "CudfTopNRowNumber only supports limit=1");
  VELOX_CHECK(
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kDenseRank,
      "CudfTopNRowNumber only supports row_number, rank, or dense_rank");

  for (const auto& key : node->partitionKeys()) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopNRowNumber doesn't allow constant partition keys");
    partitionKeys_.push_back(channel);
    const auto& keyName = inputType_->nameOf(channel);
    if (keyName.compare(
            0, kConditionalTopNMarker.size(), kConditionalTopNMarker) == 0) {
      VELOX_CHECK(
          !passthroughKey_.has_value(),
          "TopNRowNumber allows only one conditional pass-through key");
      VELOX_CHECK(
          inputType_->childAt(channel)->kind() == TypeKind::BOOLEAN,
          "Conditional TopNRowNumber marker must be boolean");
      passthroughKey_ = channel;
    }
  }

  const auto& sortingKeys = node->sortingKeys();
  const auto& sortingOrders = node->sortingOrders();

  for (const auto& key : sortingKeys) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopNRowNumber doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
  }

  allKeyIndices_ = partitionKeys_;
  allKeyIndices_.insert(
      allKeyIndices_.end(), sortKeys_.begin(), sortKeys_.end());

  for (size_t i = 0; i < partitionKeys_.size(); ++i) {
    columnOrders_.push_back(cudf::order::ASCENDING);
    nullOrders_.push_back(cudf::null_order::BEFORE);
  }

  for (const auto& order : sortingOrders) {
    columnOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfTopNRowNumber::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

  if (passthroughKey_.has_value()) {
    auto stream = cudfInput->stream();
    auto inputView = cudfInput->getTableView();
    auto activeMask = inputView.column(*passthroughKey_);
    VELOX_CHECK(
        activeMask.type().id() == cudf::type_id::BOOL8,
        "Conditional TopNRowNumber marker must be BOOL8");

    auto inactiveMask = cudf::unary_operation(
        activeMask, cudf::unary_operator::NOT, stream, get_temp_mr());
    auto inactive = cudf::apply_boolean_mask(
        inputView, inactiveMask->view(), stream, get_output_mr());
    if (inactive->num_rows() > 0) {
      passthroughOutputs_.push_back(
          std::make_shared<CudfVector>(
              pool(),
              inputType_,
              inactive->num_rows(),
              std::move(inactive),
              stream));
    }

    auto active = cudf::apply_boolean_mask(
        inputView, activeMask, stream, get_output_mr());
    if (active->num_rows() == 0) {
      return;
    }
    cudfInput = std::make_shared<CudfVector>(
        pool(), inputType_, active->num_rows(), std::move(active), stream);
  }

  auto stream = cudfInput->stream();
  auto mr = get_output_mr();
  auto batchCandidates =
      reduceToCandidates(cudfInput->getTableView(), stream, mr);
  if (candidates_ && candidates_->num_rows() > 0) {
    std::vector<cudf::table_view> pieces{
        candidates_->view(), batchCandidates->view()};
    auto merged = cudf::concatenate(pieces, stream, mr);
    candidates_ = reduceToCandidates(merged->view(), stream, mr);
  } else {
    candidates_ = std::move(batchCandidates);
  }

  auto candidateVector = std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidates_->num_rows(),
      std::move(candidates_),
      stream);
  const auto candidateBytes = candidateVector->estimateFlatSize();
  if (candidateBytes >= kCandidateRunBytes) {
    inputs_.push_back(std::move(candidateVector));
    bufferedBytes_ = candidateBytes;
    spillSortedRun();
  } else {
    candidates_ = candidateVector->release();
  }
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  if (spilled_ && candidates_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto candidateVector = std::make_shared<CudfVector>(
        pool(),
        inputType_,
        candidates_->num_rows(),
        std::move(candidates_),
        stream);
    bufferedBytes_ = candidateVector->estimateFlatSize();
    inputs_.push_back(std::move(candidateVector));
    spillSortedRun();
  }
  if (spilled_) {
    compactSortedRunsForMerge();
    initializeSortedRunReaders();
  }
  if (!candidates_ && passthroughOutputs_.empty()) {
    finished_ = !spilled_;
  }
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (!passthroughOutputs_.empty()) {
    auto output = std::move(passthroughOutputs_.front());
    passthroughOutputs_.pop_front();
    return output;
  }

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (spilled_) {
    auto result = computeNextSortedOutput();
    if (result != nullptr) {
      return result;
    }
    finished_ = true;
    cleanupSpillFiles();
    return nullptr;
  }

  if (!candidates_) {
    finished_ = true;
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  auto input = std::exchange(candidates_, nullptr);
  auto result =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input->view(), stream, mr)
      : computeLimitOneRankLike(input->view(), stream, mr);
  finished_ = true;
  return result;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::reduceToCandidates(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto reduced =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input, stream, mr)
      : computeLimitOneRankLike(input, stream, mr);
  auto table = reduced->release();
  if (generateRowNumber_) {
    auto columns = table->release();
    VELOX_CHECK_EQ(
        columns.size(),
        inputType_->size() + 1,
        "Incremental TopN candidate has unexpected generated rank column");
    columns.pop_back();
    table = std::make_unique<cudf::table>(std::move(columns));
  }
  return table;
}

void CudfTopNRowNumber::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }

  namespace fs = std::filesystem;
  if (!spilled_) {
    const auto sequence = spillDirectorySequence.fetch_add(1);
    spillDirectory_ = (fs::temp_directory_path() /
                       fmt::format(
                           "velox-cudf-topn-spill-{}-{}",
                           static_cast<int64_t>(::getpid()),
                           sequence))
                          .string();
    fs::create_directories(spillDirectory_);
    spilled_ = true;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.concatenate.begin "
          "bufferedBytes={} bufferedInputs={}",
          diagnosticNodeId_,
          bufferedBytes_,
          inputs_.size()));
  auto input =
      getConcatenatedTable(std::exchange(inputs_, {}), inputType_, stream, mr);
  bufferedBytes_ = 0;

  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.sort.begin rows={}",
          diagnosticNodeId_,
          input->num_rows()));
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(allKeyIndices_),
      columnOrders_,
      nullOrders_,
      stream,
      mr);
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.sort.end rows={}",
          diagnosticNodeId_,
          input->num_rows()));

  auto path = fmt::format(
      "{}/run-{:06}.parquet", spillDirectory_, spillFileSequence_++);
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{path}, sorted->view())
                     .build();
  cudf::io::write_parquet(options, stream);
  sortedRuns_.push_back({std::move(path), nullptr});
  ::malloc_trim(0);
}

void CudfTopNRowNumber::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        kMergeChunkBytes, 0, options, stream, mr);
  }
  readersInitialized_ = true;
}

void CudfTopNRowNumber::compactSortedRunsForMerge() {
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  while (sortedRuns_.size() > kMergeFanIn) {
    std::vector<SortedRun> nextLevel;
    nextLevel.reserve((sortedRuns_.size() + kMergeFanIn - 1) / kMergeFanIn);

    for (size_t begin = 0; begin < sortedRuns_.size(); begin += kMergeFanIn) {
      const auto end = std::min(sortedRuns_.size(), begin + kMergeFanIn);
      if (end - begin == 1) {
        nextLevel.push_back(std::move(sortedRuns_[begin]));
        continue;
      }

      std::vector<std::unique_ptr<cudf::io::chunked_parquet_reader>> readers;
      readers.reserve(end - begin);
      for (size_t index = begin; index < end; ++index) {
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{sortedRuns_[index].path})
                           .build();
        readers.push_back(
            std::make_unique<cudf::io::chunked_parquet_reader>(
                kMergeChunkBytes, 0, options, stream, mr));
      }

      const auto outputPath = fmt::format(
          "{}/merge-{:06}.parquet", spillDirectory_, spillFileSequence_++);
      auto writerOptions = cudf::io::chunked_parquet_writer_options::builder(
                               cudf::io::sink_info{outputPath})
                               .build();
      cudf::io::chunked_parquet_writer writer(writerOptions, stream);
      std::unique_ptr<cudf::table> carry;

      while (true) {
        std::vector<std::unique_ptr<cudf::table>> chunks;
        std::vector<cudf::table_view> mergeViews;
        std::vector<cudf::table_view> boundaryRows;
        if (carry && carry->num_rows() > 0) {
          mergeViews.push_back(carry->view());
        }
        for (auto& reader : readers) {
          if (!reader->has_next()) {
            continue;
          }
          auto chunk = reader->read_chunk();
          if (chunk.tbl->num_rows() == 0) {
            continue;
          }
          chunks.push_back(std::move(chunk.tbl));
          mergeViews.push_back(chunks.back()->view());
          if (reader->has_next()) {
            auto last = cudf::slice(
                chunks.back()->view(),
                {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
                stream);
            boundaryRows.push_back(last.front());
          }
        }

        if (mergeViews.empty()) {
          break;
        }
        std::unique_ptr<cudf::table> merged = mergeViews.size() == 1
            ? std::make_unique<cudf::table>(mergeViews.front(), stream, mr)
            : cudf::merge(
                  mergeViews,
                  allKeyIndices_,
                  columnOrders_,
                  nullOrders_,
                  stream,
                  mr);
        carry.reset();
        if (boundaryRows.empty()) {
          writer.write(merged->view());
          break;
        }

        auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
        auto sortedBoundaries = cudf::sort_by_key(
            boundaryCandidates->view(),
            boundaryCandidates->view().select(allKeyIndices_),
            columnOrders_,
            nullOrders_,
            stream,
            mr);
        auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
        auto positions = cudf::upper_bound(
            merged->view().select(allKeyIndices_),
            boundary.front().select(allKeyIndices_),
            columnOrders_,
            nullOrders_,
            stream,
            mr);
        const auto safeEnd = firstSearchPosition(positions->view(), stream);
        carry = copyTableSlice(
            merged->view(), safeEnd, merged->num_rows(), stream, mr);
        if (safeEnd > 0) {
          auto safe = cudf::slice(merged->view(), {0, safeEnd}, stream);
          writer.write(safe.front());
        }
      }
      writer.close();

      for (size_t index = begin; index < end; ++index) {
        std::error_code error;
        std::filesystem::remove(sortedRuns_[index].path, error);
      }
      nextLevel.push_back({outputPath, nullptr});
    }
    sortedRuns_ = std::move(nextLevel);
  }
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextSortedBatch(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finalBatch) {
  // Once all readers are exhausted, a subsequent call may exist solely to
  // drain partitionCarry_. Mark it final so the last partition is emitted
  // instead of being retained forever as a possibly-incomplete peer group.
  finalBatch = mergeFinished_;
  while (!mergeFinished_) {
    std::vector<std::unique_ptr<cudf::table>> chunks;
    std::vector<cudf::table_view> mergeViews;
    std::vector<cudf::table_view> boundaryRows;
    if (mergeCarry_ && mergeCarry_->num_rows() > 0) {
      mergeViews.push_back(mergeCarry_->view());
    }
    for (auto& run : sortedRuns_) {
      if (!run.reader || !run.reader->has_next()) {
        continue;
      }
      auto chunk = run.reader->read_chunk();
      if (chunk.tbl->num_rows() == 0) {
        continue;
      }
      chunks.push_back(std::move(chunk.tbl));
      mergeViews.push_back(chunks.back()->view());
      if (run.reader->has_next()) {
        auto last = cudf::slice(
            chunks.back()->view(),
            {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
            stream);
        boundaryRows.push_back(last.front());
      }
    }
    if (mergeViews.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return std::exchange(mergeCarry_, nullptr);
    }

    std::unique_ptr<cudf::table> merged;
    if (mergeViews.size() == 1) {
      merged = std::make_unique<cudf::table>(mergeViews.front(), stream, mr);
    } else {
      merged = cudf::merge(
          mergeViews, allKeyIndices_, columnOrders_, nullOrders_, stream, mr);
    }
    mergeCarry_.reset();
    if (boundaryRows.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return merged;
    }

    auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
    auto sortedBoundaries = cudf::sort_by_key(
        boundaryCandidates->view(),
        boundaryCandidates->view().select(allKeyIndices_),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
    // Future rows are >= each run's current tail. Rows equal to the minimum
    // tail are therefore safe to emit as well (the sort is not required to be
    // stable across runs). Keeping them in carry makes low-cardinality keys
    // grow without bound.
    auto positions = cudf::upper_bound(
        merged->view().select(allKeyIndices_),
        boundary.front().select(allKeyIndices_),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    const auto safeEnd = firstSearchPosition(positions->view(), stream);
    mergeCarry_ =
        copyTableSlice(merged->view(), safeEnd, merged->num_rows(), stream, mr);
    if (safeEnd > 0) {
      return copyTableSlice(merged->view(), 0, safeEnd, stream, mr);
    }
  }
  return nullptr;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::takeCompletePartitions(
    std::unique_ptr<cudf::table> sorted,
    bool finalBatch,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (partitionCarry_ && partitionCarry_->num_rows() > 0) {
    if (sorted && sorted->num_rows() > 0) {
      std::vector<cudf::table_view> pieces{
          partitionCarry_->view(), sorted->view()};
      sorted = cudf::concatenate(pieces, stream, mr);
      partitionCarry_.reset();
    } else {
      sorted = std::exchange(partitionCarry_, nullptr);
    }
  }
  if (!sorted || sorted->num_rows() == 0) {
    return nullptr;
  }

  auto partitionColumns = sorted->view().select(partitionKeys_);
  std::vector<cudf::order> orders(
      partitionKeys_.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> nullOrders(
      partitionKeys_.size(), cudf::null_order::BEFORE);
  cudf::size_type completeEnd = sorted->num_rows();
  if (!finalBatch) {
    auto lastPartition = cudf::slice(
        partitionColumns, {sorted->num_rows() - 1, sorted->num_rows()}, stream);
    auto positions = cudf::lower_bound(
        partitionColumns,
        lastPartition.front(),
        orders,
        nullOrders,
        stream,
        mr);
    completeEnd = firstSearchPosition(positions->view(), stream);
  }

  // Keep each downstream Top-N reduction/gather bounded. Select a partition
  // boundary at or before the row target so a rank peer group is never split.
  cudf::size_type emitEnd = completeEnd;
  if (completeEnd > kMaxCompleteOutputRows) {
    auto boundaryPartition = cudf::slice(
        partitionColumns,
        {kMaxCompleteOutputRows, kMaxCompleteOutputRows + 1},
        stream);
    auto positions = cudf::lower_bound(
        partitionColumns,
        boundaryPartition.front(),
        orders,
        nullOrders,
        stream,
        mr);
    const auto boundary = firstSearchPosition(positions->view(), stream);
    // A single giant peer group must remain intact for rank semantics.
    emitEnd = boundary > 0 ? boundary : completeEnd;
  }

  partitionCarry_ =
      copyTableSlice(sorted->view(), emitEnd, sorted->num_rows(), stream, mr);
  if (emitEnd == 0) {
    return nullptr;
  }
  return copyTableSlice(sorted->view(), 0, emitEnd, stream, mr);
}

CudfVectorPtr CudfTopNRowNumber::computeNextSortedOutput() {
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  while (!mergeFinished_ || mergeCarry_ || partitionCarry_) {
    bool finalBatch = false;
    auto sorted = mergeNextSortedBatch(stream, mr, finalBatch);
    sorted = takeCompletePartitions(std::move(sorted), finalBatch, stream, mr);
    if (!sorted || sorted->num_rows() == 0) {
      if (finalBatch) {
        return nullptr;
      }
      continue;
    }
    // The merge output is already globally ordered. The existing limit-one
    // helpers are reused initially for exact rank/tie semantics; they operate
    // on a bounded complete-partition batch.
    return rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
        ? computeLimitOneRowNumber(sorted->view(), stream, mr)
        : computeLimitOneRankLike(sorted->view(), stream, mr);
  }
  return nullptr;
}

void CudfTopNRowNumber::cleanupSpillFiles() {
  sortedRuns_.clear();
  mergeCarry_.reset();
  partitionCarry_.reset();
  if (spillDirectory_.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove_all(spillDirectory_, error);
  spillDirectory_.clear();
  ::malloc_trim(0);
}

void CudfTopNRowNumber::doClose() {
  inputs_.clear();
  candidates_.reset();
  passthroughOutputs_.clear();
  cleanupSpillFiles();
  Operator::close();
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRowNumber(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::table> result;

  if (input.num_rows() == 0) {
    result = std::make_unique<cudf::table>(input, stream, mr);
  } else if (partitionKeys_.empty()) {
    auto keyView = input.select(sortKeys_);
    std::vector<cudf::order> sortOrders(
        columnOrders_.begin() + partitionKeys_.size(), columnOrders_.end());
    std::vector<cudf::null_order> sortNullOrders(
        nullOrders_.begin() + partitionKeys_.size(), nullOrders_.end());

    auto sortedIndices = cudf::stable_sorted_order(
        keyView, sortOrders, sortNullOrders, stream, mr);
    auto firstIndex = cudf::split(sortedIndices->view(), {1}, stream).front();
    result = cudf::gather(
        input,
        firstIndex,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  } else if (
      sortKeys_.size() == 1 &&
      nullOrders_[partitionKeys_.size()] == cudf::null_order::AFTER) {
    auto partitionView = input.select(partitionKeys_);
    cudf::groupby::groupby grouper(partitionView, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = input.column(sortKeys_.front());
    if (columnOrders_[partitionKeys_.size()] == cudf::order::ASCENDING) {
      requests[0].aggregations.push_back(
          cudf::make_min_aggregation<cudf::groupby_aggregation>());
    } else {
      requests[0].aggregations.push_back(
          cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    auto [groupKeys, aggregateResults] =
        grouper.aggregate(requests, stream, mr);
    VELOX_CHECK_EQ(aggregateResults.size(), 1);
    VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
    auto topKeyColumns = groupKeys->release();
    topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
    auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));
    auto probeKeys = input.select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeys->view(),
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices = lookup.inner_join(probeKeys, std::nullopt, stream, mr);
    auto probeIndices = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    auto bestPeers = cudf::gather(
        input,
        probeIndices,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    result = cudf::unique(
        bestPeers->view(),
        partitionKeys_,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream,
        mr);
  } else {
    auto allKeysView = input.select(allKeyIndices_);
    auto sortedIndices = cudf::stable_sorted_order(
        allKeysView, columnOrders_, nullOrders_, stream, mr);
    auto sortedTable = cudf::gather(
        input,
        sortedIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);

    result = cudf::unique(
        sortedTable->view(),
        partitionKeys_,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream,
        mr);
  }

  if (generateRowNumber_) {
    auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
    auto rowNumber =
        cudf::make_column_from_scalar(one, result->num_rows(), stream, mr);
    auto columns = result->release();
    columns.push_back(std::move(rowNumber));
    result = std::make_unique<cudf::table>(std::move(columns));
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRankLike(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(!sortKeys_.empty(), "Rank-like TopNRowNumber requires sort keys");

  std::unique_ptr<cudf::table> result;
  if (input.num_rows() == 0) {
    result = std::make_unique<cudf::table>(input, stream, mr);
  } else if (
      !partitionKeys_.empty() && sortKeys_.size() == 1 &&
      nullOrders_[partitionKeys_.size()] == cudf::null_order::AFTER) {
    // Fast grouped Top-1 for the common single scalar order key. A full sort
    // is unnecessary: compute each partition's best key, then join that key
    // back to the input. The inner join intentionally preserves every peer of
    // the best key, which is exactly rank/dense_rank limit=1 semantics.
    auto partitionView = input.select(partitionKeys_);
    cudf::groupby::groupby grouper(partitionView, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = input.column(sortKeys_.front());
    if (columnOrders_[partitionKeys_.size()] == cudf::order::ASCENDING) {
      requests[0].aggregations.push_back(
          cudf::make_min_aggregation<cudf::groupby_aggregation>());
    } else {
      requests[0].aggregations.push_back(
          cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    auto [groupKeys, aggregateResults] =
        grouper.aggregate(requests, stream, mr);
    VELOX_CHECK_EQ(aggregateResults.size(), 1);
    VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
    auto topKeyColumns = groupKeys->release();
    topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
    auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));
    auto probeKeys = input.select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeys->view(),
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices = lookup.inner_join(probeKeys, std::nullopt, stream, mr);
    auto probeIndices = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    result = cudf::gather(
        input,
        probeIndices,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  } else {
    auto allKeysView = input.select(allKeyIndices_);
    auto sortedIndices = cudf::stable_sorted_order(
        allKeysView, columnOrders_, nullOrders_, stream, mr);
    auto sortedTable = cudf::gather(
        input,
        sortedIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);

    std::unique_ptr<cudf::table> topRows;
    if (partitionKeys_.empty()) {
      auto firstIndex = cudf::split(sortedIndices->view(), {1}, stream).front();
      topRows = cudf::gather(
          input,
          firstIndex,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
    } else {
      topRows = cudf::unique(
          sortedTable->view(),
          partitionKeys_,
          cudf::duplicate_keep_option::KEEP_FIRST,
          cudf::null_equality::EQUAL,
          stream,
          mr);
    }

    auto topKeyView = topRows->view().select(allKeyIndices_);
    auto probeKeyView = sortedTable->view().select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeyView,
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices =
        lookup.inner_join(probeKeyView, std::nullopt, stream, mr);
    auto leftIndicesCol = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    result = cudf::gather(
        sortedTable->view(),
        leftIndicesCol,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  }

  if (generateRowNumber_) {
    auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
    auto rowNumber =
        cudf::make_column_from_scalar(one, result->num_rows(), stream, mr);
    auto columns = result->release();
    columns.push_back(std::move(rowNumber));
    result = std::make_unique<cudf::table>(std::move(columns));
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

} // namespace facebook::velox::cudf_velox
