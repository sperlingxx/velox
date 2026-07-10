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
#include <mutex>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kCandidateRunBytes = 128ULL << 20;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr uint64_t kMergePassBytes = 128ULL << 20;
constexpr uint64_t kSpillRowGroupBytes = 64ULL << 20;
constexpr uint64_t kOutputChunkBytes = 32ULL << 20;
constexpr size_t kMergeFanIn = 2;
constexpr size_t kFinalMergeRuns = 2;
constexpr cudf::size_type kMaxCompleteOutputRows = 262144;
constexpr std::string_view kConditionalTopNMarker = "__gluten_mpp_topn_active";
std::atomic<uint64_t> spillDirectorySequence{0};
std::atomic<uint64_t> candidateRunBytes{kCandidateRunBytes};
std::atomic<uint64_t> mergeChunkBytes{kMergeChunkBytes};
std::atomic<uint64_t> outputChunkBytes{kOutputChunkBytes};
std::atomic<cudf::size_type> maxOutputRows{kMaxCompleteOutputRows};

size_t topNCompactionConcurrency() {
  return static_cast<size_t>(
      CudfConfig::getInstance().topNCompactionConcurrency);
}

class TopNCompactionAdmission {
 public:
  enum class RequestState { kQueued, kAcquired, kCancelled };

  struct Request {
    RequestState state{RequestState::kQueued};
    std::optional<ContinuePromise> promise;
  };

  struct Reservation {
    std::shared_ptr<Request> request;
    ContinueFuture future{ContinueFuture::makeEmpty()};
  };

  Reservation reserve(size_t concurrency) {
    if (concurrency == 0) {
      return {};
    }

    Reservation reservation;
    reservation.request = std::make_shared<Request>();
    std::lock_guard<std::mutex> lock(mutex_);
    if (capacity_ == 0) {
      capacity_ = concurrency;
    }
    VELOX_CHECK_EQ(
        capacity_,
        concurrency,
        "Concurrent TopN compaction requests must use one process-wide "
        "concurrency limit");

    if (waiters_.empty() && active_ < capacity_) {
      reservation.request->state = RequestState::kAcquired;
      ++active_;
      return reservation;
    }

    auto [promise, future] = makeVeloxContinuePromiseContract(
        "CudfTopNRowNumber compaction admission");
    reservation.request->promise = std::move(promise);
    reservation.future = std::move(future);
    waiters_.push_back(reservation.request);
    return reservation;
  }

  bool acquired(const std::shared_ptr<Request>& request) const {
    if (!request) {
      // A zero concurrency setting disables admission limiting.
      return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return request->state == RequestState::kAcquired;
  }

  void cancelOrRelease(const std::shared_ptr<Request>& request) {
    if (!request) {
      return;
    }

    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (request->state == RequestState::kAcquired) {
        VELOX_CHECK_GT(active_, 0);
        --active_;
        request->state = RequestState::kCancelled;
      } else if (request->state == RequestState::kQueued) {
        request->state = RequestState::kCancelled;
        movePromise(*request, promises);
      }
      grantWaitersLocked(promises);
      if (active_ == 0 && waiters_.empty()) {
        capacity_ = 0;
      }
    }
    for (auto& promise : promises) {
      promise.setValue();
    }
  }

  size_t active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
  }

 private:
  static void movePromise(
      Request& request,
      std::vector<ContinuePromise>& promises) {
    if (request.promise.has_value()) {
      promises.push_back(std::move(*request.promise));
      request.promise.reset();
    }
  }

  void grantWaitersLocked(std::vector<ContinuePromise>& promises) {
    while (active_ < capacity_ && !waiters_.empty()) {
      auto request = std::move(waiters_.front());
      waiters_.pop_front();
      if (request->state == RequestState::kCancelled) {
        continue;
      }
      VELOX_CHECK(request->state == RequestState::kQueued);
      request->state = RequestState::kAcquired;
      ++active_;
      movePromise(*request, promises);
    }
  }

  mutable std::mutex mutex_;
  std::deque<std::shared_ptr<Request>> waiters_;
  size_t capacity_{0};
  size_t active_{0};
};

TopNCompactionAdmission topNCompactionAdmission;

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

// nullOrders_ uses cuDF's direction-adjusted convention from the other cuDF
// sort operators. Convert it back to the SQL meaning before deciding whether
// min/max may ignore nulls.
bool semanticNullsLast(
    cudf::order order,
    cudf::null_order nullOrder) {
  return (order == cudf::order::ASCENDING &&
          nullOrder == cudf::null_order::AFTER) ||
      (order == cudf::order::DESCENDING &&
       nullOrder == cudf::null_order::BEFORE);
}

} // namespace

struct CudfTopNRowNumber::CompactionPermit::State {
  std::shared_ptr<TopNCompactionAdmission::Request> request;
  ContinueFuture future{ContinueFuture::makeEmpty()};
};

CudfTopNRowNumber::CompactionPermit::CompactionPermit(size_t concurrency)
    : state_(std::make_unique<State>()) {
  auto reservation = topNCompactionAdmission.reserve(concurrency);
  state_->request = std::move(reservation.request);
  state_->future = std::move(reservation.future);
}

CudfTopNRowNumber::CompactionPermit::~CompactionPermit() {
  topNCompactionAdmission.cancelOrRelease(state_->request);
}

bool CudfTopNRowNumber::CompactionPermit::ready() const {
  return topNCompactionAdmission.acquired(state_->request);
}

bool CudfTopNRowNumber::CompactionPermit::hasWaitFuture() const {
  return state_->future.valid();
}

ContinueFuture CudfTopNRowNumber::CompactionPermit::takeWaitFuture() {
  VELOX_CHECK(state_->future.valid());
  return std::move(state_->future);
}

size_t CudfTopNRowNumber::CompactionPermit::testingActivePermits() {
  return topNCompactionAdmission.active();
}

void CudfTopNRowNumber::testingSetMemoryLimits(
    uint64_t candidateBytes,
    uint64_t chunkBytes,
    uint64_t outputBytes,
    cudf::size_type outputRows) {
  VELOX_CHECK_GT(candidateBytes, 0);
  VELOX_CHECK_GT(chunkBytes, 0);
  VELOX_CHECK_GT(outputBytes, 0);
  VELOX_CHECK_GT(outputRows, 0);
  candidateRunBytes.store(candidateBytes);
  mergeChunkBytes.store(chunkBytes);
  outputChunkBytes.store(outputBytes);
  maxOutputRows.store(outputRows);
}

void CudfTopNRowNumber::testingResetMemoryLimits() {
  candidateRunBytes.store(kCandidateRunBytes);
  mergeChunkBytes.store(kMergeChunkBytes);
  outputChunkBytes.store(kOutputChunkBytes);
  maxOutputRows.store(kMaxCompleteOutputRows);
}

exec::BlockingReason CudfTopNRowNumber::isBlocked(ContinueFuture* future) {
  if (compactionPermit_ && compactionPermit_->hasWaitFuture()) {
    *future = compactionPermit_->takeWaitFuture();
    return exec::BlockingReason::kWaitForMemory;
  }
  return exec::BlockingReason::kNotBlocked;
}

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
      diagnosticNodeId_(node->id()),
      stateStream_(cudfGlobalStreamPool().get_stream()) {
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

  const auto inputStream = cudfInput->stream();
  if (inputStream.value() != stateStream_.value()) {
    std::vector<rmm::cuda_stream_view> inputStreams{inputStream};
    cudf::detail::join_streams(inputStreams, stateStream_);
  }
  // Rebind unconditionally before taking any views. A packed-table backing
  // buffer may still carry a different deallocation stream even when the
  // CudfVector's logical stream already equals stateStream_.
  VELOX_CHECK(
      cudfInput->rebindStream(stateStream_),
      "CudfTopNRowNumber cannot rebind its input to the state stream");
  auto stream = stateStream_;

  if (passthroughKey_.has_value()) {
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
  if (candidateBytes >= candidateRunBytes.load()) {
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
    auto stream = stateStream_;
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
    const auto compactionConcurrency = topNCompactionConcurrency();
    if (!compactionPermit_) {
      if (compactionConcurrency > 0) {
        logDeviceMemorySnapshot(fmt::format(
            "operator=CudfTopNRowNumber node={} state=compaction.admission.wait "
            "runs={} concurrency={}",
            diagnosticNodeId_,
            sortedRuns_.size(),
            compactionConcurrency));
      }
      // reserve() never waits on this driver thread. A queued request exposes a
      // ContinueFuture from isBlocked() and resumes in getOutput after its FIFO
      // turn is granted.
      compactionPermit_ =
          std::make_unique<CompactionPermit>(compactionConcurrency);
    }
    try {
      if (compactionPermit_->ready()) {
        prepareSpilledOutput();
      }
    } catch (...) {
      cleanupSpillStateAfterFailure("noMoreInput");
      throw;
    }
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
    try {
      prepareSpilledOutput();
      auto result = computeNextSortedOutput();
      if (result != nullptr) {
        return result;
      }
      finished_ = true;
      cleanupSpillFiles();
      return nullptr;
    } catch (...) {
      // Preserve the original merge failure while ensuring all GPU owners are
      // destroyed before handing the admission slot to another operator.
      cleanupSpillStateAfterFailure("getOutput");
      throw;
    }
  }

  if (auto output = takePendingOutput()) {
    return output;
  }

  if (!candidates_) {
    finished_ = true;
    return nullptr;
  }

  auto stream = stateStream_;
  auto mr = get_output_mr();
  auto input = std::exchange(candidates_, nullptr);
  auto result =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input->view(), stream, mr)
      : computeLimitOneRankLike(input->view(), stream, mr);
  setPendingOutput(result->release());
  return takePendingOutput();
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

  auto stream = stateStream_;
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
                     .row_group_size_bytes(kSpillRowGroupBytes)
                     .build();
  cudf::io::write_parquet(options, stream);
  sortedRuns_.push_back({std::move(path), nullptr});
  ::malloc_trim(0);
}

void CudfTopNRowNumber::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = stateStream_;
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        mergeChunkBytes.load(), kMergePassBytes, options, stream, mr);
    run.chunk.reset();
    run.chunkOffset = 0;
    run.chunkBytes = 0;
  }
  readersInitialized_ = true;
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfTopNRowNumber node={} state=output.merge.begin runs={} "
      "chunkReadLimit={} passReadLimit={}",
      diagnosticNodeId_,
      sortedRuns_.size(),
      mergeChunkBytes.load(),
      kMergePassBytes));
}

uint64_t CudfTopNRowNumber::measureTableBytes(
    std::unique_ptr<cudf::table>& table,
    const TypePtr& type,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(table);
  const auto rows = table->num_rows();
  auto vector = std::make_shared<CudfVector>(
      pool(), type, rows, std::move(table), stream);
  const auto bytes = vector->estimateFlatSize();
  table = vector->release();
  return bytes;
}

bool CudfTopNRowNumber::loadPausedChunk(
    SortedRun& run,
    rmm::cuda_stream_view stream,
    PausedMergeStats& stats) {
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
    stats.sourceRows += chunk.tbl->num_rows();
    if (chunk.tbl->num_rows() == 0) {
      continue;
    }
    run.chunk = std::move(chunk.tbl);
    run.chunkBytes =
        measureTableBytes(run.chunk, inputType_, stateStream_);
    stats.sourceBytes += run.chunkBytes;
    return true;
  }
  return false;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextPausedBatch(
    std::vector<SortedRun*>& runs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finished,
    PausedMergeStats& stats) {
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
    // Count the whole owning chunk. This deliberately overestimates a paused
    // suffix and therefore remains a safe resident-byte high-water mark.
    residentBytes += run->chunkBytes;
  }

  stats.maxActiveRuns = std::max<uint64_t>(
      stats.maxActiveRuns, activeRuns.size());
  stats.maxResidentRows =
      std::max(stats.maxResidentRows, residentRows);
  stats.maxResidentBytes =
      std::max(stats.maxResidentBytes, residentBytes);
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
    // Every run is sorted. The minimum current tail is a global safe boundary:
    // future rows in every run are >= its current tail. Consume only each
    // run's prefix through that boundary and leave its suffix in the owning
    // chunk. A leading reader is therefore paused instead of copied into an
    // ever-growing carry table.
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
        boundaryCandidates->view().select(allKeyIndices_),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);

    for (size_t index = 0; index < remainingViews.size(); ++index) {
      auto positions = cudf::upper_bound(
          remainingViews[index].select(allKeyIndices_),
          boundary.front().select(allKeyIndices_),
          columnOrders_,
          nullOrders_,
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

  VELOX_CHECK(!safeViews.empty(), "Paused Top-N merge made no progress");
  std::unique_ptr<cudf::table> output = safeViews.size() == 1
      ? std::make_unique<cudf::table>(safeViews.front(), stream, mr)
      : cudf::merge(
            safeViews,
            allKeyIndices_,
            columnOrders_,
            nullOrders_,
            stream,
            mr);

  for (size_t index = 0; index < activeRuns.size(); ++index) {
    activeRuns[index]->chunkOffset += consumed[index];
  }
  ++stats.outputBatches;
  stats.outputRows += output->num_rows();
  const auto batchBytes =
      measureTableBytes(output, inputType_, stateStream_);
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

void CudfTopNRowNumber::prepareSpilledOutput() {
  if (readersInitialized_) {
    return;
  }
  VELOX_CHECK_NOT_NULL(compactionPermit_);
  VELOX_CHECK(
      compactionPermit_->ready(),
      "TopN spill compaction started before admission was granted");
  try {
    compactSortedRunsForMerge();
    initializeSortedRunReaders();
  } catch (...) {
    cleanupSpillStateAfterFailure("prepareSpilledOutput");
    throw;
  }
}

void CudfTopNRowNumber::compactSortedRunsForMerge() {
  auto stream = stateStream_;
  auto mr = get_output_mr();
  const auto compactionConcurrency = topNCompactionConcurrency();
  VELOX_CHECK_NOT_NULL(compactionPermit_);
  VELOX_CHECK(compactionPermit_->ready());
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfTopNRowNumber node={} state=compaction.admission.held "
      "runs={} concurrency={}",
      diagnosticNodeId_,
      sortedRuns_.size(),
      compactionConcurrency));

  PausedMergeStats compactionStats;

  try {
    // Stop at two runs. getOutput performs the final merge with the same paused
    // cursors. compactionPermit_ deliberately remains held until those readers
    // and all pending output have been drained and cleaned up.
    while (sortedRuns_.size() > kFinalMergeRuns) {
      const auto inputRunCount = sortedRuns_.size();
      std::vector<SortedRun> nextLevel;
      nextLevel.reserve((sortedRuns_.size() + kMergeFanIn - 1) / kMergeFanIn);
      std::vector<std::string> obsoletePaths;
      PausedMergeStats levelStats;

      for (size_t begin = 0; begin < sortedRuns_.size(); begin += kMergeFanIn) {
        const auto end = std::min(sortedRuns_.size(), begin + kMergeFanIn);
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
              mergeChunkBytes.load(), kMergePassBytes, options, stream, mr);
          run.chunk.reset();
          run.chunkOffset = 0;
          run.chunkBytes = 0;
          runs.push_back(&run);
        }

        const auto outputPath = fmt::format(
            "{}/merge-{:06}.parquet", spillDirectory_, spillFileSequence_++);
        auto writerOptions = cudf::io::chunked_parquet_writer_options::builder(
                                 cudf::io::sink_info{outputPath})
                                 .row_group_size_bytes(kSpillRowGroupBytes)
                                 .build();
        cudf::io::chunked_parquet_writer writer(writerOptions, stream);
        bool groupFinished{false};
        while (!groupFinished) {
          auto merged = mergeNextPausedBatch(
              runs, stream, mr, groupFinished, levelStats);
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
        nextLevel.push_back({outputPath, nullptr});
      }

      // Complete all I/O and stream-ordered frees before deleting source runs
      // and starting the next level. The async allocator can then immediately
      // reuse the completed level's storage.
      stream.synchronize();
      for (const auto& path : obsoletePaths) {
        std::error_code error;
        std::filesystem::remove(path, error);
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
          "operator=CudfTopNRowNumber node={} state=compaction.level.end "
          "inputRuns={} outputRuns={} sourceChunks={} sourceRows={} sourceBytes={} "
          "outputBatches={} outputRows={} outputBytes={} maxResidentRows={} "
          "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
          diagnosticNodeId_,
          inputRunCount,
          sortedRuns_.size(),
          levelStats.sourceChunks,
          levelStats.sourceRows,
          levelStats.sourceBytes,
          levelStats.outputBatches,
          levelStats.outputRows,
          levelStats.outputBytes,
          levelStats.maxResidentRows,
          levelStats.maxResidentBytes,
          levelStats.maxOutputBytes,
          levelStats.maxActiveRuns));
    }

    stream.synchronize();
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfTopNRowNumber node={} state=compaction.end.admission-held "
        "runs={} concurrency={} sourceChunks={} sourceRows={} sourceBytes={} "
        "outputBatches={} outputRows={} outputBytes={} maxResidentRows={} "
        "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
        diagnosticNodeId_,
        sortedRuns_.size(),
        compactionConcurrency,
        compactionStats.sourceChunks,
        compactionStats.sourceRows,
        compactionStats.sourceBytes,
        compactionStats.outputBatches,
        compactionStats.outputRows,
        compactionStats.outputBytes,
        compactionStats.maxResidentRows,
        compactionStats.maxResidentBytes,
        compactionStats.maxOutputBytes,
        compactionStats.maxActiveRuns));
  } catch (...) {
    // Local cuDF objects have unwound. Clear member-owned readers/chunks and
    // drain their async frees before another operator receives this slot.
    cleanupSpillStateAfterFailure("compaction");
    throw;
  }
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextSortedBatch(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finalBatch) {
  // finalBatch is retained for diagnostics/callers; sorted streaming state no
  // longer buffers a whole trailing partition.
  finalBatch = mergeFinished_;
  if (mergeFinished_) {
    return nullptr;
  }

  std::vector<SortedRun*> runs;
  runs.reserve(sortedRuns_.size());
  for (auto& run : sortedRuns_) {
    runs.push_back(&run);
  }
  auto result = mergeNextPausedBatch(
      runs, stream, mr, mergeFinished_, outputMergeStats_);
  finalBatch = mergeFinished_;
  if (mergeFinished_ && !outputMergeEndLogged_) {
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfTopNRowNumber node={} state=output.merge.end "
        "runs={} sourceChunks={} sourceRows={} sourceBytes={} outputBatches={} "
        "outputRows={} outputBytes={} maxResidentRows={} maxResidentBytes={} "
        "maxOutputBytes={} maxActiveRuns={}",
        diagnosticNodeId_,
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
    outputMergeEndLogged_ = true;
  }
  return result;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::appendGeneratedRank(
    std::unique_ptr<cudf::table> table,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  if (!generateRowNumber_ || !table) {
    return table;
  }
  auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
  auto rank =
      cudf::make_column_from_scalar(one, table->num_rows(), stream, mr);
  auto columns = table->release();
  columns.push_back(std::move(rank));
  return std::make_unique<cudf::table>(std::move(columns));
}

void CudfTopNRowNumber::updateSortedPartitionState(
    cudf::table_view sorted,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GT(sorted.num_rows(), 0);

  cudf::size_type firstRowOfLastPartition{0};
  if (!partitionKeys_.empty()) {
    auto partitionColumns = sorted.select(partitionKeys_);
    auto lastPartition = cudf::slice(
        partitionColumns, {sorted.num_rows() - 1, sorted.num_rows()}, stream);
    std::vector<cudf::order> orders(
        partitionKeys_.size(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> nullOrders(
        partitionKeys_.size(), cudf::null_order::BEFORE);
    auto positions = cudf::lower_bound(
        partitionColumns,
        lastPartition.front(),
        orders,
        nullOrders,
        stream,
        mr);
    firstRowOfLastPartition =
        firstSearchPosition(positions->view(), stream);
  }

  auto stateRow = cudf::slice(
      sorted,
      {firstRowOfLastPartition, firstRowOfLastPartition + 1},
      stream);
  VELOX_CHECK_EQ(stateRow.size(), 1);
  if (partitionKeys_.empty()) {
    currentPartitionKey_.reset();
  } else {
    currentPartitionKey_ = std::make_unique<cudf::table>(
        stateRow.front().select(partitionKeys_), stream, mr);
  }
  if (rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber) {
    currentPeerKey_.reset();
  } else {
    currentPeerKey_ = std::make_unique<cudf::table>(
        stateRow.front().select(allKeyIndices_), stream, mr);
  }
  hasCurrentPartition_ = true;
}

std::unique_ptr<cudf::table>
CudfTopNRowNumber::reduceSortedBatchToTopOne(
    std::unique_ptr<cudf::table> sorted,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!sorted || sorted->num_rows() == 0) {
    return nullptr;
  }

  const auto sortedView = sorted->view();
  const auto numRows = sorted->num_rows();
  cudf::size_type continuationEnd{0};
  if (hasCurrentPartition_) {
    if (partitionKeys_.empty()) {
      // Empty partition keys mean one global partition. Never construct or
      // search a zero-column cuDF table.
      continuationEnd = numRows;
    } else {
      VELOX_CHECK_NOT_NULL(currentPartitionKey_);
      auto partitionColumns = sortedView.select(partitionKeys_);
      std::vector<cudf::order> orders(
          partitionKeys_.size(), cudf::order::ASCENDING);
      std::vector<cudf::null_order> nullOrders(
          partitionKeys_.size(), cudf::null_order::BEFORE);
      auto positions = cudf::upper_bound(
          partitionColumns,
          currentPartitionKey_->view(),
          orders,
          nullOrders,
          stream,
          mr);
      continuationEnd = firstSearchPosition(positions->view(), stream);
    }
  }

  std::vector<std::unique_ptr<cudf::table>> outputPieces;
  if (continuationEnd > 0 &&
      rankFunction_ != core::TopNRowNumberNode::RankFunction::kRowNumber) {
    VELOX_CHECK_NOT_NULL(currentPeerKey_);
    auto continuation =
        cudf::slice(sortedView, {0, continuationEnd}, stream);
    auto positions = cudf::upper_bound(
        continuation.front().select(allKeyIndices_),
        currentPeerKey_->view(),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    const auto peerEnd = firstSearchPosition(positions->view(), stream);
    if (peerEnd > 0) {
      outputPieces.push_back(appendGeneratedRank(
          copyTableSlice(sortedView, 0, peerEnd, stream, mr), stream, mr));
    }
  }

  const auto newPartitionBegin = hasCurrentPartition_ ? continuationEnd : 0;
  if (newPartitionBegin < numRows) {
    auto newPartitions =
        cudf::slice(sortedView, {newPartitionBegin, numRows}, stream);
    auto reduced =
        rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
        ? computeLimitOneRowNumber(newPartitions.front(), stream, mr)
        : computeLimitOneRankLike(newPartitions.front(), stream, mr);
    outputPieces.push_back(reduced->release());
  }

  // Only a batch containing a newly-started partition changes the one-row
  // state. A batch containing only the continuation of the current partition
  // retains its original first peer key.
  if (!hasCurrentPartition_ || newPartitionBegin < numRows) {
    updateSortedPartitionState(sortedView, stream, mr);
  }

  if (outputPieces.empty()) {
    return nullptr;
  }
  if (outputPieces.size() == 1) {
    return std::move(outputPieces.front());
  }
  std::vector<cudf::table_view> outputViews;
  outputViews.reserve(outputPieces.size());
  for (const auto& piece : outputPieces) {
    outputViews.push_back(piece->view());
  }
  return cudf::concatenate(outputViews, stream, mr);
}

void CudfTopNRowNumber::setPendingOutput(
    std::unique_ptr<cudf::table> output) {
  VELOX_CHECK(!pendingOutput_);
  if (!output || output->num_rows() == 0) {
    return;
  }
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ =
      measureTableBytes(output, outputType_, stateStream_);
  pendingOutput_ = std::move(output);
}

CudfVectorPtr CudfTopNRowNumber::takePendingOutput() {
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
    const auto proportionalRows = static_cast<cudf::size_type>(std::max<uint64_t>(
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
        LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                     << " emitted one oversized row bytes=" << actualBytes
                     << " byteLimit=" << byteLimit;
      }
      pendingOutputOffset_ += targetRows;
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

CudfVectorPtr CudfTopNRowNumber::computeNextSortedOutput() {
  auto stream = stateStream_;
  auto mr = get_output_mr();
  if (auto output = takePendingOutput()) {
    return output;
  }
  while (!mergeFinished_) {
    bool finalBatch = false;
    auto sorted = mergeNextSortedBatch(stream, mr, finalBatch);
    (void)finalBatch;
    auto reduced =
        reduceSortedBatchToTopOne(std::move(sorted), stream, mr);
    if (!reduced || reduced->num_rows() == 0) {
      continue;
    }
    setPendingOutput(std::move(reduced));
    return takePendingOutput();
  }
  return nullptr;
}

void CudfTopNRowNumber::cleanupSpillStateAfterFailure(
    std::string_view context) noexcept {
  // A failed synchronization must not jump directly to permit release while
  // readers or tables are still live. First make a best-effort drain, always
  // destroy every GPU owner, then drain the deallocations before releasing (or
  // cancelling) admission. This preserves the original task exception.
  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber " << context
                 << " pre-destruction cleanup failed: " << error.what();
  }

  inputs_.clear();
  candidates_.reset();
  passthroughOutputs_.clear();
  sortedRuns_.clear();
  currentPartitionKey_.reset();
  currentPeerKey_.reset();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;
  hasCurrentPartition_ = false;

  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber " << context
                 << " post-destruction cleanup failed: " << error.what();
  }

  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    spillDirectory_.clear();
  }
  compactionPermit_.reset();
}

void CudfTopNRowNumber::cleanupSpillFiles() {
  // Readers, one-row streaming state, pending output, and Parquet writes all
  // use stateStream_. Finish their work before destroying owners or releasing
  // the process-wide admission permit.
  stateStream_.synchronize();
  sortedRuns_.clear();
  currentPartitionKey_.reset();
  currentPeerKey_.reset();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;
  hasCurrentPartition_ = false;
  // Destroying cuDF owners enqueues their deallocations on stateStream_. Wait
  // for those async frees before admitting the next memory-heavy TopN.
  stateStream_.synchronize();
  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    spillDirectory_.clear();
  }
  compactionPermit_.reset();
  ::malloc_trim(0);
}

void CudfTopNRowNumber::doClose() {
  // close() also runs while unwinding a failed task. Preserve the original
  // failure if the CUDA context is already poisoned, but never free live state
  // before first attempting to drain its stream.
  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber state stream cleanup failed: "
                 << error.what();
  }
  inputs_.clear();
  candidates_.reset();
  passthroughOutputs_.clear();
  sortedRuns_.clear();
  currentPartitionKey_.reset();
  currentPeerKey_.reset();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;
  hasCurrentPartition_ = false;
  try {
    // The resets above enqueue async deallocations. Keep the admission permit
    // until they have completed so the next TopN cannot overlap this state.
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber post-destruction cleanup failed: "
                 << error.what();
  }
  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    spillDirectory_.clear();
  }
  compactionPermit_.reset();
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
      semanticNullsLast(
          columnOrders_[partitionKeys_.size()],
          nullOrders_[partitionKeys_.size()])) {
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
      semanticNullsLast(
          columnOrders_[partitionKeys_.size()],
          nullOrders_[partitionKeys_.size()])) {
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
