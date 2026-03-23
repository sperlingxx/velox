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
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>
#include <thread>

namespace {

static constexpr int kOomMaxRetries = 3;

void recoverGpuMemory() {
  cudaDeviceSynchronize();
  cudaGetLastError();
}

void trimGpuMemoryPool() {
  cudaMemPool_t pool = nullptr;
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess &&
      pool != nullptr) {
    cudaMemPoolTrimTo(pool, 0);
  }
  cudaGetLastError();
}

// cuDF has no VARBINARY type — all string-like columns come back as VARCHAR.
// kindEquals(VARCHAR, VARBINARY) returns false, so setType() crashes even
// though they share FlatVector<StringView> layout.  This function rebuilds
// vectors bottom-up with the target type while sharing data buffers, so
// setType() can succeed (or be skipped entirely).
void fixStringBinaryMismatch(
    facebook::velox::VectorPtr& vec,
    const facebook::velox::TypePtr& target) {
  using namespace facebook::velox;
  if (!vec || !target || vec->type()->kindEquals(target)) {
    return;
  }
  auto actualKind = vec->type()->kind();
  auto targetKind = target->kind();

  if (is_string_kind(actualKind) && is_string_kind(targetKind)) {
    auto* flat = vec->asFlatVector<StringView>();
    vec = std::make_shared<FlatVector<StringView>>(
        flat->pool(), target, flat->nulls(), flat->size(),
        flat->values(),
        std::vector<BufferPtr>(flat->stringBuffers()));
    return;
  }

  if (actualKind == TypeKind::ROW && targetKind == TypeKind::ROW) {
    auto* row = vec->as<RowVector>();
    for (column_index_t i = 0;
         i < row->childrenSize() && i < target->size();
         ++i) {
      fixStringBinaryMismatch(row->childAt(i), target->childAt(i));
    }
    if (!vec->type()->kindEquals(target)) {
      std::vector<VectorPtr> children;
      children.reserve(row->childrenSize());
      for (column_index_t i = 0; i < row->childrenSize(); ++i) {
        children.push_back(row->childAt(i));
      }
      vec = std::make_shared<RowVector>(
          row->pool(), target, row->nulls(), row->size(),
          std::move(children));
    }
  } else if (actualKind == TypeKind::ARRAY && targetKind == TypeKind::ARRAY) {
    auto* arr = vec->as<ArrayVector>();
    VectorPtr elems = arr->elements();
    fixStringBinaryMismatch(elems, target->childAt(0));
    if (elems != arr->elements()) {
      vec = std::make_shared<ArrayVector>(
          arr->pool(), target, arr->nulls(), arr->size(),
          arr->offsets(), arr->sizes(), std::move(elems));
    }
  } else if (actualKind == TypeKind::MAP && targetKind == TypeKind::MAP) {
    auto* map = vec->as<MapVector>();
    VectorPtr keys = map->mapKeys();
    VectorPtr vals = map->mapValues();
    fixStringBinaryMismatch(keys, target->childAt(0));
    fixStringBinaryMismatch(vals, target->childAt(1));
    if (keys != map->mapKeys() || vals != map->mapValues()) {
      vec = std::make_shared<MapVector>(
          map->pool(), target, map->nulls(), map->size(),
          map->offsets(), map->sizes(),
          std::move(keys), std::move(vals));
    }
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

namespace {

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfFromVelox"),
      NvtxHelper(
          nvtx3::rgb{255, 140, 0}, // Orange
          operatorId,
          fmt::format("[{}]", planNodeId)) {}

void CudfFromVelox::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Accumulate inputs
    currentOutputBytes_ += input->estimateFlatSize();
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
  }
}

RowVectorPtr CudfFromVelox::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  const auto targetBytes = CudfConfig::getInstance().gpuTargetBatchBytes;
  const auto targetRows =
      preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());

  finished_ = noMoreInput_ && inputs_.empty();

  bool belowRowThreshold =
      currentOutputSize_ < static_cast<std::size_t>(targetRows);
  bool belowByteThreshold =
      (targetBytes > 0) ? (currentOutputBytes_ < targetBytes) : true;
  bool belowThreshold = belowRowThreshold && belowByteThreshold;

  if (finished_ or (belowThreshold and not noMoreInput_) or inputs_.empty()) {
    return nullptr;
  }

  // Select inputs that don't exceed the max vector size limit
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  int64_t totalBytes = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    if (totalSize + input->size() <= maxVectorSize) {
      selectedInputs.push_back(input);
      totalSize += input->size();
      totalBytes += input->estimateFlatSize();
    } else {
      break;
    }
  }

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;
  currentOutputBytes_ -= totalBytes;

  // Early return if no input
  if (totalSize == 0) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  if (selectedInputs[0]->type()->size() == 0) {
    // Zero-column input (e.g. COUNT(*) without GROUP BY).
    // Arrow/cudf tables cannot preserve row count with 0 columns,
    // so create a dummy INT8 column to carry the row count through.
    // Note: outputType_ may differ from actual input type because
    // ToCudf uses the downstream operator's output type, not the
    // upstream operator's output type.
    auto mr = cudf::get_current_device_resource_ref();
    auto dummy = cudf::make_numeric_column(
        cudf::data_type(cudf::type_id::INT8),
        totalSize,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(dummy));
    auto tbl = std::make_unique<cudf::table>(std::move(cols));
    return std::make_shared<CudfVector>(
        selectedInputs[0]->pool(),
        outputType_,
        totalSize,
        std::move(tbl),
        stream);
  }

  // Batched HtoD with OOM resilience: if GPU memory is exhausted, reduce the
  // batch size and put excess inputs back for the next getOutput() call.
  // This prevents std::bad_alloc from crashing the process (SIGABRT) when
  // multiple concurrent tasks saturate GPU memory (e.g. Q18 Stage 35).
  std::unique_ptr<cudf::table> tbl;
  for (int attempt = 0;; ++attempt) {
    try {
      tbl = with_arrow::toCudfTableBatched(
          selectedInputs, selectedInputs[0]->pool(), stream);
      break;
    } catch (const std::bad_alloc& e) {
      if (selectedInputs.size() > 1) {
        // Reduce batch: put the second half back into inputs_ for next call.
        auto half = selectedInputs.size() / 2;
        LOG(WARNING) << "CudfFromVelox[" << planNodeId() << "]: GPU OOM "
                     << "with " << selectedInputs.size() << " batches, "
                     << "reducing to " << half << " (attempt "
                     << (attempt + 1) << ")";
        for (size_t i = selectedInputs.size(); i > half; --i) {
          auto& v = selectedInputs[i - 1];
          currentOutputSize_ += v->size();
          currentOutputBytes_ += v->estimateFlatSize();
          inputs_.insert(inputs_.begin(), std::move(v));
        }
        selectedInputs.resize(half);
        recoverGpuMemory();
        continue;
      }
      if (attempt >= kOomMaxRetries) {
        trimGpuMemoryPool();
        VELOX_FAIL(
            "CudfFromVelox[{}]: GPU OOM after {} retries with a single "
            "batch ({} rows, {} bytes): {}",
            planNodeId(),
            kOomMaxRetries,
            selectedInputs[0]->size(),
            selectedInputs[0]->estimateFlatSize(),
            e.what());
      }
      LOG(WARNING) << "CudfFromVelox[" << planNodeId() << "]: GPU OOM "
                   << "with single batch (" << selectedInputs[0]->size()
                   << " rows), retrying (attempt " << (attempt + 1)
                   << "/" << kOomMaxRetries << ")";
      recoverGpuMemory();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100 * (1 << attempt)));
    }
  }

  VELOX_CHECK_NOT_NULL(tbl);

  if (selectedInputs.size() > 1) {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "numCoalescedBatches", RuntimeCounter(1));
  }

  const auto size = tbl->num_rows();
  return std::make_shared<CudfVector>(
      selectedInputs[0]->pool(), outputType_, size, std::move(tbl), stream);
}

void CudfFromVelox::close() {
  cudf::get_default_stream().synchronize();
  exec::Operator::close();
  inputs_.clear();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfToVelox"),
      NvtxHelper(
          nvtx3::rgb{148, 0, 211}, // Purple
          operatorId,
          fmt::format("[{}]", planNodeId)) {}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

std::optional<uint64_t> CudfToVelox::averageRowSize() {
  if (!averageRowSize_) {
    if (inputs_.empty() || inputs_.front()->size() == 0) {
      return std::nullopt;
    }
    averageRowSize_ =
        inputs_.front()->estimateFlatSize() / inputs_.front()->size();
  }
  return averageRowSize_;
}

RowVectorPtr CudfToVelox::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  if (finished_ || inputs_.empty()) {
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  // Get the target batch size
  const auto targetBatchSize = outputBatchRows(averageRowSize());
  auto stream = inputs_.front()->stream();

  // Process single input directly in these cases:
  // 1. In passthrough mode
  // 2. If we only have one input and it's smaller than or equal to the target
  // batch size
  if (isPassthroughMode() ||
      (inputs_.size() == 1 && inputs_.front()->size() <= targetBatchSize)) {
    // Move the CudfVector out to keep it alive while we use the view.
    // This avoids expensive materialization when constructed from packed_table.
    auto cudfVector = std::move(inputs_.front());
    inputs_.pop_front();

    auto tableView = cudfVector->getTableView();
    if (tableView.num_rows() == 0) {
      finished_ = noMoreInput_ && inputs_.empty();
      return nullptr;
    }
    RowVectorPtr output =
        with_arrow::toVeloxColumn(tableView, pool(), outputType_, "", stream, cudf::get_current_device_resource_ref());
    stream.synchronize();
    finished_ = noMoreInput_ && inputs_.empty();
    if (output->type()->kindEquals(outputType_)) {
      output->setType(outputType_);
    } else {
      for (column_index_t i = 0; i < output->childrenSize(); ++i) {
        if (i < outputType_->size()) {
          fixStringBinaryMismatch(
              output->childAt(i), outputType_->childAt(i));
        }
      }
      std::vector<VectorPtr> children;
      children.reserve(output->childrenSize());
      for (column_index_t i = 0; i < output->childrenSize(); ++i) {
        children.push_back(output->childAt(i));
      }
      output = std::make_shared<RowVector>(
          pool(), outputType_, output->nulls(), output->size(),
          std::move(children));
    }
    return output;
  }

  // Calculate how many tables we need to concatenate to reach the target batch
  // size and collect them in a vector
  std::vector<CudfVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;

  while (!inputs_.empty() && totalSize < targetBatchSize) {
    auto& input = inputs_.front();
    if (totalSize + input->size() <= targetBatchSize) {
      totalSize += input->size();
      selectedInputs.push_back(std::move(input));
      inputs_.pop_front();
    } else {
      // If the next input would exceed targetBatchSize,
      // we need to split it and only take what we need.
      // The input may have been produced on a different stream than the
      // first input's stream; ensure ordering before using its data.
      if (input->stream() != stream) {
        cudf::detail::join_streams(
            std::vector<rmm::cuda_stream_view>{input->stream()}, stream);
      }
      auto cudfTableView = input->getTableView();
      auto partitions = std::vector<cudf::size_type>{
          static_cast<cudf::size_type>(targetBatchSize - totalSize)};
      auto tableSplits = cudf::split(cudfTableView, partitions, stream);

      // Create new CudfVector from the first part
      auto firstPart = std::make_unique<cudf::table>(tableSplits[0], stream);
      auto firstPartSize = firstPart->num_rows();
      auto firstPartVector = std::make_shared<CudfVector>(
          pool(), input->type(), firstPartSize, std::move(firstPart), stream);

      // Create new CudfVector from the second part
      auto secondPart = std::make_unique<cudf::table>(tableSplits[1], stream);
      auto secondPartSize = secondPart->num_rows();
      auto secondPartVector = std::make_shared<CudfVector>(
          pool(), input->type(), secondPartSize, std::move(secondPart), stream);

      // Sync before releasing the original input: the table copies above are
      // async on `stream` and read from the original input's device buffers.
      // Without this, the original's GPU memory may be freed (when `input` is
      // reassigned below) while the copy kernels are still reading from it.
      stream.synchronize();

      // Replace the original input with the second part
      input = std::move(secondPartVector);

      // Add the first part to selectedInputs
      selectedInputs.push_back(std::move(firstPartVector));
      totalSize += firstPartSize;
      break;
    }
  }

  finished_ = noMoreInput_ && inputs_.empty();

  // If we have no inputs to process, return nullptr
  if (selectedInputs.empty()) {
    return nullptr;
  }

  // Concatenate the selected tables on the GPU
  auto resultTable = getConcatenatedTable(selectedInputs, outputType_, stream);

  // Convert the concatenated table to a RowVector
  const auto size = resultTable->num_rows();
  VELOX_CHECK_NOT_NULL(resultTable);
  if (size == 0) {
    return nullptr;
  }

  RowVectorPtr output = with_arrow::toVeloxColumn(
      resultTable->view(), pool(), outputType_, "", stream, cudf::get_current_device_resource_ref());
  stream.synchronize();
  finished_ = noMoreInput_ && inputs_.empty();
  if (output->type()->kindEquals(outputType_)) {
    output->setType(outputType_);
  } else {
    for (column_index_t i = 0; i < output->childrenSize(); ++i) {
      if (i < outputType_->size()) {
        fixStringBinaryMismatch(
            output->childAt(i), outputType_->childAt(i));
      }
    }
    std::vector<VectorPtr> children;
    children.reserve(output->childrenSize());
    for (column_index_t i = 0; i < output->childrenSize(); ++i) {
      children.push_back(output->childAt(i));
    }
    output = std::make_shared<RowVector>(
        pool(), outputType_, output->nulls(), output->size(),
        std::move(children));
  }
  return output;
}

void CudfToVelox::close() {
  exec::Operator::close();
  inputs_.clear();
}

} // namespace facebook::velox::cudf_velox
