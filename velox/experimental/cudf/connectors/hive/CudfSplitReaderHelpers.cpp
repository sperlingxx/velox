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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/Casts.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <folly/futures/Future.h>

#include <exception>
#include <future>
#include <mutex>
#include <vector>

namespace {

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back({dst, size, std::move(sharedStream)});
}

void BufferedInputDataSource::prepareDeviceLoadBatch(size_t count) {
  VELOX_CHECK(
      pendingDeviceLoads_.empty(),
      "Cannot prepare a device-load batch while another batch is pending");
  pendingDeviceLoads_.reserve(count);
}

void BufferedInputDataSource::discardPendingDeviceLoads() noexcept {
  pendingDeviceLoads_.clear();
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  std::vector<PendingDeviceLoad> deviceLoads;
  std::vector<std::vector<uint8_t>> hostBuffers;

  // Serialize complete IO batches and detach this batch up front. If any
  // operation fails, pendingDeviceLoads_ must not retain destinations owned by
  // the caller that is unwinding.
  std::lock_guard<std::mutex> lock(ioBatchMutex());
  deviceLoads.swap(pendingDeviceLoads_);

  bool copyQueued = false;
  try {
    input_->load(velox::dwio::common::LogType::FILE);
    hostBuffers.reserve(deviceLoads.size());
    for (const auto& deviceLoad : deviceLoads) {
      auto& buffer = hostBuffers.emplace_back(deviceLoad.size);
      deviceLoad.inputStream->readFully(
          reinterpret_cast<char*>(buffer.data()), deviceLoad.size);
      if (deviceLoad.size > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            deviceLoad.destination,
            buffer.data(),
            deviceLoad.size,
            cudaMemcpyHostToDevice,
            stream.value()));
        copyQueued = true;
      }
    }

    // The host buffers must outlive their asynchronous H2D copies. Complete
    // the stream before either the future or this storage is released.
    if (copyQueued) {
      stream.synchronize();
    }
  } catch (...) {
    if (copyQueued) {
      // Do not let an exception destroy a host buffer while a queued copy may
      // still reference it. Preserve the original exception after draining.
      (void)cudaStreamSynchronize(stream.value());
    }
    throw;
  }
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      const auto readSize = hostBuffer->size();
                      if (readSize > 0) {
                        CUDF_CUDA_TRY(cudaMemcpyAsync(
                            dst,
                            hostBuffer->data(),
                            readSize,
                            cudaMemcpyHostToDevice,
                            stream.value()));
                        // device_read_async's future may only become ready once
                        // dst is safe for the caller to consume. This also keeps
                        // hostBuffer alive for the full asynchronous copy.
                        stream.synchronize();
                      }
                      return readSize;
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource) {
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_footer_to_host(*dataSource);
}

std::unique_ptr<cudf::io::datasource::buffer> fetchPageIndexBytes(
    std::shared_ptr<cudf::io::datasource> dataSource,
    const cudf::io::text::byte_range_info pageIndexBytes) {
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_page_index_to_host(
      *dataSource, pageIndexBytes);
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) { return acc + byteRange.size(); });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData + acc, static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    auto syncFunction = [](std::shared_ptr<cudf::io::datasource> dataSource,
                           rmm::cuda_stream_view stream) {
      auto buffer =
          checkedPointerCast<BufferedInputDataSource>(dataSource.get());
      buffer->load(stream);
    };

    bool batchPrepared = false;
    try {
      bufferedInput->prepareDeviceLoadBatch(byteRanges.size());
      batchPrepared = true;
      auto iter =
          cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
      std::for_each(
          iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
            const auto& byteRange = cuda::std::get<0>(tuple);
            const auto& destination = cuda::std::get<1>(tuple);
            bufferedInput->enqueueForDevice(
                static_cast<uint64_t>(byteRange.offset()),
                static_cast<uint64_t>(byteRange.size()),
                const_cast<uint8_t*>(destination.data()));
          });

      auto loadFuture =
          std::async(std::launch::deferred, syncFunction, dataSource, stream);
      return {
          std::move(columnChunkBuffers),
          std::move(columnChunkData),
          std::move(loadFuture)};
    } catch (...) {
      if (batchPrepared) {
        bufferedInput->discardPendingDeviceLoads();
      }
      throw;
    }
  }

  // KvikIO dataSource: Impl borrowed from `fetch_byte_ranges_to_device_async()`
  // in `parquet_io_utils.cpp` in cuDF.
  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<uint8_t*> destinations;

  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      if (nextOffset != ioOffset + ioSize) {
        break;
      }
      ioSize += byteRanges[nextChunk].size();
      nextChunk++;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinations.push_back(
          const_cast<uint8_t*>(columnChunkData[chunk].data()));
    }
    chunk = nextChunk;
  }
  VELOX_CHECK_EQ(
      ioOffsets.size(),
      ioSizes.size(),
      "Number of IO offsets and sizes must be equal");
  VELOX_CHECK_EQ(
      ioSizes.size(),
      destinations.size(),
      "Number of IO sizes and destinations must be equal");

  auto iter = cuda::make_zip_iterator(
      ioOffsets.begin(), ioSizes.begin(), destinations.begin());

  struct ReadTaskState {
    std::vector<std::future<size_t>> deviceReadTasks;
    std::vector<
        std::future<std::unique_ptr<cudf::io::datasource::buffer>>>
        hostReadTasks;
    std::vector<uint8_t*> hostReadDestinations;
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>> hostBuffers;
  };
  auto readState = std::make_shared<ReadTaskState>();
  readState->deviceReadTasks.reserve(ioOffsets.size());
  readState->hostReadTasks.reserve(ioOffsets.size());
  readState->hostReadDestinations.reserve(ioOffsets.size());
  // Allocate result slots before any task can write into a device destination.
  readState->hostBuffers.resize(ioOffsets.size());

  auto collectReadTasks = [](ReadTaskState& state) noexcept {
    std::exception_ptr firstError;
    for (size_t i = 0; i < state.hostReadTasks.size(); ++i) {
      try {
        state.hostBuffers[i] = state.hostReadTasks[i].get();
      } catch (...) {
        if (!firstError) {
          firstError = std::current_exception();
        }
      }
    }
    for (auto& task : state.deviceReadTasks) {
      try {
        task.get();
      } catch (...) {
        if (!firstError) {
          firstError = std::current_exception();
        }
      }
    }
    return firstError;
  };

  // device_read_async is not guaranteed to follow stream-ordering (see
  // datasource API docs)
  stream.synchronize();

  try {
    std::lock_guard<std::mutex> lock(ioBatchMutex());
    std::for_each(
        iter, iter + ioOffsets.size(), [&](const auto& tuple) {
          const auto ioOffset = cuda::std::get<0>(tuple);
          const auto ioSize = cuda::std::get<1>(tuple);
          const auto dest = cuda::std::get<2>(tuple);

          if (dataSource->supports_device_read() and
              dataSource->is_device_read_preferred(ioSize)) {
            readState->deviceReadTasks.emplace_back(
                dataSource->device_read_async(ioOffset, ioSize, dest, stream));
          } else {
            readState->hostReadTasks.emplace_back(
                std::async(
                    std::launch::async,
                    [dataSource, ioOffset, ioSize]() {
                      return dataSource->host_read(ioOffset, ioSize);
                    }));
            readState->hostReadDestinations.push_back(dest);
          }
        });
  } catch (...) {
    const auto schedulingError = std::current_exception();
    // Futures returned by arbitrary datasources do not have to wait in their
    // destructors. Drain every task before destination buffers unwind.
    (void)collectReadTasks(*readState);
    std::rethrow_exception(schedulingError);
  }

  auto syncFunction = [readState, collectReadTasks, stream]() {
    const auto firstError = collectReadTasks(*readState);
    if (firstError) {
      std::rethrow_exception(firstError);
    }

    bool copyQueued = false;
    try {
      for (size_t i = 0; i < readState->hostReadTasks.size(); ++i) {
        const auto& hostBuffer = readState->hostBuffers[i];
        if (hostBuffer->size() > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              readState->hostReadDestinations[i],
              hostBuffer->data(),
              hostBuffer->size(),
              cudaMemcpyHostToDevice,
              stream.value()));
          copyQueued = true;
        }
      }
      if (copyQueued) {
        stream.synchronize();
      }
    } catch (...) {
      if (copyQueued) {
        (void)cudaStreamSynchronize(stream.value());
      }
      throw;
    }
  };

  try {
    // The shared state is allocated before launching tasks, so even a failure
    // to allocate the deferred future leaves this scope able to drain them.
    auto readFuture =
        std::async(std::launch::deferred, std::move(syncFunction));
    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::move(readFuture)};
  } catch (...) {
    const auto futureError = std::current_exception();
    (void)collectReadTasks(*readState);
    std::rethrow_exception(futureError);
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive
