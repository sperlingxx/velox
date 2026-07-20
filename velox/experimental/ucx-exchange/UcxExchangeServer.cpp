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
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include <glog/logging.h>
#include <malloc.h>
#include <rmm/cuda_stream_view.hpp>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include "cuda_runtime.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
void accountFreedHostBytesAndTrim(uint64_t bytes) {
  constexpr uint64_t kTrimInterval = 64ULL * 1024 * 1024;
  static std::atomic<uint64_t> freedSinceTrim{0};
  if (bytes == 0) {
    return;
  }
  const auto accumulated =
      freedSinceTrim.fetch_add(bytes, std::memory_order_acq_rel) + bytes;
  if (accumulated >= kTrimInterval) {
    const auto claimed = freedSinceTrim.exchange(0, std::memory_order_acq_rel);
    if (claimed >= kTrimInterval) {
      malloc_trim(0);
    }
  }
}

void retireRequest(
    std::shared_ptr<ucxx::Request>& current,
    std::vector<std::shared_ptr<ucxx::Request>>& inFlight) {
  inFlight.erase(
      std::remove_if(
          inFlight.begin(),
          inFlight.end(),
          [](const auto& request) {
            return request == nullptr || request->isCompleted();
          }),
      inFlight.end());
  if (current != nullptr && !current->isCompleted()) {
    inFlight.push_back(std::move(current));
  } else {
    current.reset();
  }
}

const folly::F14FastMap<UcxExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxExchangeServer::ServerState::Created, "Created"},
              {UcxExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxExchangeServer::ServerState::DataRequestReady,
               "DataRequestReady"},
              {UcxExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxExchangeServer::ServerState::WaitingForIntraNodeRetrieve,
               "WaitingForIntraNodeRetrieve"},
              {UcxExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}

bool intraNodeProducerPollRequeueEnabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("GLUTEN_UCX_INTRANODE_PRODUCER_POLL_REQUEUE");
    return value != nullptr && value[0] != '\0' &&
        !(value[0] == '0' && value[1] == '\0');
  }();
  return enabled;
}

int64_t intraNodeProducerPollRequeueLimit() {
  static const int64_t limit = [] {
    const char* value =
        std::getenv("GLUTEN_UCX_INTRANODE_PRODUCER_POLL_REQUEUE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
      return int64_t{-1};
    }
    try {
      return static_cast<int64_t>(std::stoll(value));
    } catch (...) {
      return int64_t{-1};
    }
  }();
  return limit;
}

int64_t maxInFlightSendHostBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_SEND_HOST_BYTES")) {
      try {
        const auto parsed = static_cast<int64_t>(std::stoll(value));
        if (parsed > 0) {
          return parsed;
        }
      } catch (...) {
      }
    }
    return static_cast<int64_t>(2) * 1024 * 1024 * 1024;
  }();
  return limit;
}

std::atomic<int64_t> inFlightSendHostBytes{0};

bool tryReserveSendHostBytes(int64_t bytes) {
  VELOX_CHECK_GE(bytes, 0);
  auto current = inFlightSendHostBytes.load(std::memory_order_relaxed);
  while (true) {
    // Permit one oversized transfer when no other host staging is active. A
    // single packed table cannot be split by the current wire protocol, and
    // rejecting it forever would deadlock the exchange.
    if (current > 0 && current + bytes > maxInFlightSendHostBytes()) {
      return false;
    }
    if (inFlightSendHostBytes.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return true;
    }
  }
}

void releaseSendHostBytes(int64_t bytes) {
  if (bytes <= 0) {
    return;
  }
  const auto previous =
      inFlightSendHostBytes.fetch_sub(bytes, std::memory_order_acq_rel);
  VELOX_CHECK_GE(previous, bytes);
}

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeServer,
    ServerState,
    serverStateNames)

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime (which should be freed promptly after DMA completes).
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory. The context remains alive as an empty shell for the
// lifetime of the Request, which is safe and costs negligible memory.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  // The UCX build used by Gluten MPP may not include CUDA memory-type
  // transports.  In that case handing an rmm device pointer to tagSend makes
  // the shared-memory transport memcpy from an inaccessible address.  Keep a
  // host staging buffer alive with the request and let UCX move host memory.
  std::shared_ptr<std::vector<uint8_t>> hostData;
  int64_t reservedHostBytes{0};

  ~DataSendContext() {
    releaseHostReservation();
  }

  bool reserveHostBytes(int64_t bytes) {
    VELOX_CHECK_EQ(reservedHostBytes, 0);
    if (!tryReserveSendHostBytes(bytes)) {
      return false;
    }
    reservedHostBytes = bytes;
    return true;
  }

  void releaseHostReservation() {
    if (reservedHostBytes > 0) {
      releaseSendHostBytes(reservedHostBytes);
      reservedHostBytes = 0;
    }
  }
};

void UcxExchangeServer::setState(ServerState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private
UcxExchangeServer::UcxExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer)
    : CommElement(communicator, endpointRef),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      isIntraNodeTransfer_(isIntraNodeTransfer),
      queueMgr_(UcxOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);

  if (isIntraNodeTransfer_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Detected same-node source (intra-node transfer) for "
            << partitionKey_.toString();
  }
}

// static
std::shared_ptr<UcxExchangeServer> UcxExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> endpointRef,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  auto ptr = std::shared_ptr<UcxExchangeServer>(new UcxExchangeServer(
      communicator, endpointRef, key, isIntraNodeTransfer));
  return ptr;
}

void UcxExchangeServer::process() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  switch (state_) {
    case ServerState::Created:
      setState(ServerState::ReadyToTransfer);
      wakeCommunicator();
      break;
    case ServerState::ReadyToTransfer:
      // Count-only / rendezvous push (Presto-style): no consumer credit
      // request. Go straight to dequeue + send; the data tagSend blocks at
      // rendezvous until the source posts its matching tagRecv
      // (getMetadata/getData), which is the sole flow-control mechanism.
      setState(ServerState::DataRequestReady);
      wakeCommunicator();
      break;
    case ServerState::DataRequestReady:
      setState(ServerState::WaitingForDataFromQueue);
      // Register the callback with the destination queue to get data.
      // If the queue doesn't exist yet, getData will create an empty
      // queue and the callback will be triggered once the corresponding
      // source task has initialized the queue and added data to it.
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      {
        std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
        queueMgr_->getData(
            partitionKey_.taskId,
            partitionKey_.destination,
            // Unbounded per-fetch cap; rendezvous + queue-occupancy
            // backpressure are the flow control (no byte-credit).
            std::numeric_limits<uint64_t>::max(),
            static_cast<int64_t>(sequenceNumber_),
            [weakQueue](
                std::shared_ptr<cudf::packed_columns> data,
                int64_t sequence,
                std::vector<int64_t> remainingBytes) {
              auto self = weakQueue.lock();
              if (!self) {
                return; // Object was destroyed, safe to ignore
              }
              // Check if close() was called - avoid processing if we're
              // shutting down
              if (self->closed_.load(std::memory_order_acquire)) {
                VLOG(3) << "@" << self->partitionKey_.taskId
                        << " getData callback called after close, ignoring";
                return;
              }
              if (sequence != static_cast<int64_t>(self->sequenceNumber_)) {
                // The destination queue has already advanced beyond this
                // duplicate/retried server.  Close only this stale server.
                // In particular, do not deleteResults(): another active
                // server owns the already-advanced queue and may still need
                // its remaining pages.
                LOG(WARNING)
                    << "Closing stale UCX exchange server for task="
                    << self->partitionKey_.taskId
                    << " destination=" << self->partitionKey_.destination
                    << " requestedSequence=" << self->sequenceNumber_
                    << " acknowledgedSequence=" << sequence;
                self->skipQueueDeleteOnClose_.store(
                    true, std::memory_order_release);
                self->setState(ServerState::Done);
                self->wakeCommunicator();
                return;
              }
              // This upcall may be called from another thread than the
              // communicator thread. It is called
              // when data on the queue becomes available.
              VLOG(3) << "@" << self->partitionKey_.taskId
                      << " Found data for client: "
                      << self->partitionKey_.toString()
                      << " sequence=" << sequence;
              std::lock_guard<std::recursive_mutex> lock(self->dataMutex_);
              VELOX_CHECK(
                  self->dataPtr_ == nullptr,
                  "Data pointer exists: Illegal state!");
              self->dataPtr_ = std::move(data);
              self->setState(ServerState::DataReady);
              self->wakeCommunicator();
            });
      }
      wakeCommunicator();
      break;
    case ServerState::WaitingForDataFromQueue:
      // Waiting for data is handled by an upcall from the data queue. Nothing
      // to do
      break;
    case ServerState::DataReady:
      sendData();
      break;
    case ServerState::WaitingForSendComplete:
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
      break;
    case ServerState::WaitingForIntraNodeRetrieve:
      // Intra-node transfer: the registry re-enqueues us when the source has
      // retrieved the data. Do a non-blocking check only for that wakeup (or a
      // defensive spurious work item); do not self-requeue and spin.
      if (intraNodeRetrieveFuture_.valid()) {
        auto status =
            intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
          intraNodeRetrieveFuture_.get(); // Clear the future
          intraNodePollCount_ = 0;
          onIntraNodeRetrieveComplete();
        } else if (
            intraNodeProducerPollRequeueEnabled() &&
            (intraNodeProducerPollRequeueLimit() < 0 ||
             intraNodePollCount_ <
                 static_cast<uint32_t>(intraNodeProducerPollRequeueLimit()))) {
          ++intraNodePollCount_;
          if (intraNodePollCount_ % 100 == 0) {
            VLOG(2) << "[INTRA] [ExSrv " << partitionKey_.toString()
                    << " seq=" << sequenceNumber_
                    << "] still waiting for source retrieval, polls="
                    << intraNodePollCount_;
          }
          wakeCommunicator();
        }
      }
      break;
    case ServerState::Done:
      close();
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      break;
  };
}

void UcxExchangeServer::close() {
  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }
  VLOG(2) << "[UCX-SERVER-CLOSE] task=" << partitionKey_.taskId
          << " key=" << partitionKey_.toString() << " peer="
          << (endpointRef_ ? endpointRef_->getPeerAddress() : "(unknown)")
          << " state=" << toName(getState()) << " seq=" << sequenceNumber_
          << " hasMetaRequest=" << (metaRequest_ != nullptr)
          << " hasDataRequest=" << (dataRequest_ != nullptr)
          << " hasDataPtr=" << (dataPtr_ != nullptr);

  if (queueMgr_ && !skipQueueDeleteOnClose_.load(std::memory_order_acquire)) {
    queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
  }

  // Cancel any outstanding requests. With weak_ptr callbacks, the callbacks
  // will safely no-op if we're destroyed before they complete.
  if (metaRequest_ && !metaRequest_->isCompleted()) {
    metaRequest_->cancel();
  }
  if (dataRequest_ && !dataRequest_->isCompleted()) {
    dataRequest_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  auto communicator = communicator_.lock();
  if (communicator) {
    if (metaRequest_) {
      communicator->deferRequestCleanup(std::move(metaRequest_));
    }
    if (dataRequest_) {
      communicator->deferRequestCleanup(std::move(dataRequest_));
    }
    for (auto& req : completedRequests_) {
      communicator->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (communicator) {
    communicator->unregister(getSelfPtr());
  }
}

std::string UcxExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxExchangeServer> UcxExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void UcxExchangeServer::wakeCommunicator() {
  if (auto communicator = tryCommunicator()) {
    communicator->addToWorkQueue(getSelfPtr());
  }
}

void UcxExchangeServer::sendData() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);

  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_
          << "] sendData hasData=" << (dataPtr_ != nullptr)
          << (dataPtr_ && dataPtr_->gpu_data
                  ? " size=" + std::to_string(dataPtr_->gpu_data->size())
                  : "");

  if (isIntraNodeTransfer_) {
    // INTRA-NODE TRANSFER PATH: Use registry for all communication, no UCXX
    // needed
    sendStart_ = std::chrono::high_resolution_clock::now();

    if (dataPtr_) {
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      const auto stream = dataPtr_->gpu_data->stream();
      // The consumer tags uniquely owned pages with this stream so downstream
      // reads and stream-ordered async frees remain ordered with the buffer.
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key,
              dataPtr_,
              stream,
              /*atEnd=*/false,
              makeIntraNodeRetrieveWakeup());
      dataPtr_.reset();
      intraNodeAtEndPublished_ = false;

      // Go dormant until the source retrieves the entry and the registry wakeup
      // re-enqueues this server.
      setState(ServerState::WaitingForIntraNodeRetrieve);
      if (intraNodeProducerPollRequeueEnabled()) {
        wakeCommunicator();
      }
    } else {
      // Data pointer is null, so no more data will be coming.
      // Publish atEnd marker to registry
      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing atEnd for sequence "
              << sequenceNumber_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key,
              nullptr,
              rmm::cuda_stream_default,
              /*atEnd=*/true,
              makeIntraNodeRetrieveWakeup());
      intraNodeAtEndPublished_ = true;

      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);

      // Wait for source to acknowledge atEnd before finishing. The registry
      // wakeup re-enqueues this server when that happens.
      setState(ServerState::WaitingForIntraNodeRetrieve);
      if (intraNodeProducerPollRequeueEnabled()) {
        wakeCommunicator();
      }
    }
  } else {
    // REMOTE EXCHANGE PATH: Use UCXX for metadata and data transfer
    const bool useHostStaging = !communicator->hasCudaTransport();
    std::shared_ptr<DataSendContext> dataCtx;
    if (dataPtr_) {
      const auto hostBytes = static_cast<int64_t>(dataPtr_->gpu_data->size());
      dataCtx = std::make_shared<DataSendContext>();
      if (useHostStaging && !dataCtx->reserveHostBytes(hostBytes)) {
        // Keep dataPtr_ and state=DataReady.  Completed UCX callbacks release
        // process-wide credit; requeueing lets this server retry without
        // dequeuing or staging another packed table.
        wakeCommunicator();
        return;
      }
    }
    std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

    if (dataPtr_) {
      // Copy metadata (not move) because in broadcast mode, the same
      // packed_columns may be shared across multiple destination queues.
      // Metadata is small (CPU-side), so copying is negligible.
      metadataMsg->cudfMetadata =
          std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
      metadataMsg->dataSizeBytes = dataPtr_->gpu_data->size();
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = true;
    }

    auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

    // send metadata.
    uint64_t metadataTag =
        getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
    retireRequest(metaRequest_, completedRequests_);

    // Wrap the serialized metadata in a context so the callback can release
    // it after the send completes, while the Request (and context shell)
    // stays alive for UCP wireup replay.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;

    metaRequest_ = endpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta](
            ucs_status_t status, std::shared_ptr<void> arg) {
          // Release the metadata buffer from the context. The context
          // shell stays alive with the Request; only the payload is freed.
          auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata); // release CPU buffer

          auto self = weakMeta.lock();
          if (!self) {
            return; // Object was destroyed, safe to ignore
          }
          // Check if close() was called
          if (self->closed_.load(std::memory_order_acquire)) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata send callback called after close, ignoring";
            return;
          }
          if (status == UCS_OK) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata successfully sent to " << tid
                    << " with tag: " << std::hex << metadataTag;
          } else {
            VLOG(0) << "[UCX-SERVER-METADATA-SEND-ERROR] task="
                    << self->partitionKey_.taskId << " key=" << tid
                    << " seq=" << self->sequenceNumber_ << " tag=" << std::hex
                    << metadataTag << std::dec
                    << " status=" << ucs_status_string(status);
            self->setState(ServerState::Done);
            self->wakeCommunicator();
          }
        },
        metaCtx);

    // send the data chunk (if any)
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Sending rmm::buffer: " << std::hex
              << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      setState(ServerState::WaitingForSendComplete);
      uint64_t dataTag =
          getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
      retireRequest(dataRequest_, completedRequests_);

      // Wrap the GPU data buffer in a context so the callback can release
      // it after the DMA completes, while the Request (and context shell)
      // stays alive for UCP wireup replay.
      dataCtx->data = dataPtr_;
      void* sendBuffer = dataCtx->data->gpu_data->data();
      if (useHostStaging) {
        dataCtx->hostData = std::make_shared<std::vector<uint8_t>>(bytes_);
        const auto producerStream = dataCtx->data->gpu_data->stream();
        CUDF_CUDA_TRY(cudaStreamSynchronize(producerStream.value()));
        CUDF_CUDA_TRY(cudaMemcpy(
            dataCtx->hostData->data(),
            dataCtx->data->gpu_data->data(),
            bytes_,
            cudaMemcpyDeviceToHost));
        sendBuffer = dataCtx->hostData->data();
      }
      VLOG(2) << "@" << partitionKey_.taskId << " posting "
              << (useHostStaging ? "host-staged" : "direct-device")
              << " send for " << bytes_ << " bytes";

      dataRequest_ = endpointRef_->endpoint_->tagSend(
          sendBuffer,
          static_cast<size_t>(bytes_),
          ucxx::Tag{dataTag},
          false,
          [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
            // Release both payload buffers from the context. completedRequests_
            // deliberately retains the UCXX Request (and therefore
            // callbackData) for wireup replay safety, so leaving hostData in
            // this context leaks one complete host-staging copy per batch until
            // the exchange server is destroyed.  Large MPP exchanges otherwise
            // consume hundreds of GiB even though every send has completed. The
            // callback means UCX has finished with both payloads; only the
            // empty context shell must remain alive with the Request.
            auto ctx = std::static_pointer_cast<DataSendContext>(arg);
            auto dataHolder = std::move(ctx->data);
            auto hostDataHolder = std::move(ctx->hostData);
            const auto releasedHostBytes = ctx->reservedHostBytes;
            ctx->releaseHostReservation();
            hostDataHolder.reset();
            // The default allocator retains these very large vector arenas in
            // the executor even after free().  A long exchange therefore has
            // bounded live staging but unbounded RSS. Return completed large
            // transfers to the OS instead of waiting for process teardown.
            accountFreedHostBytesAndTrim(releasedHostBytes);

            if (auto self = weakData.lock()) {
              self->sendComplete(status, arg);
            }
            // The holders are destroyed here, releasing the GPU buffer if
            // sendComplete() already reset the server's dataPtr_, and always
            // releasing the completed transfer's host staging allocation.
          },
          dataCtx);
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "@" << partitionKey_.taskId
              << " Finished transferring partition for task "
              << partitionKey_.toString();
      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
      setState(ServerState::Done);
      wakeCommunicator();
    }
  }
}

void UcxExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SERVER-SEND-COMPLETE-AFTER-CLOSE] task="
            << partitionKey_.taskId << " key=" << partitionKey_.toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    return;
  }
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    VELOX_CHECK_NOT_NULL(dataPtr_, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = bytes_ / micros;

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";

    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    VLOG(3) << "@" << partitionKey_.taskId
            << " Releasing dataPtr_ in sendComplete.";
    setState(ServerState::ReadyToTransfer);
  } else {
    VLOG(0) << "[UCX-SERVER-DATA-SEND-ERROR] task=" << partitionKey_.taskId
            << " key=" << partitionKey_.toString() << " seq=" << sequenceNumber_
            << " bytes=" << bytes_ << " status=" << ucs_status_string(status);
    setState(ServerState::Done);
  }
  wakeCommunicator();
}

std::function<void()> UcxExchangeServer::makeIntraNodeRetrieveWakeup() {
  std::weak_ptr<CommElement> weakSelf = getSelfPtr();
  auto weakCommunicator = communicator_;
  return [weakSelf, weakCommunicator]() {
    if (auto server = weakSelf.lock(); server) {
      auto communicator = weakCommunicator.lock();
      if (!communicator) {
        return;
      }
      communicator->addToWorkQueue(server);
    }
  };
}

void UcxExchangeServer::onIntraNodeRetrieveComplete() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " onIntraNodeRetrieveComplete called after close, ignoring";
    return;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - sendStart_;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

  VLOG(3)
      << "@" << partitionKey_.taskId << " Intra-node transfer duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer throughput: " << throughput << " MByte/s";

  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer complete for sequence " << sequenceNumber_;

  if (intraNodeAtEndPublished_) {
    // This was the final atEnd marker, we're done
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: atEnd acknowledged, finishing";
    setState(ServerState::Done);
  } else {
    // More data may be coming, continue transfer loop
    this->sequenceNumber_++;
    setState(ServerState::ReadyToTransfer);
  }
  wakeCommunicator();
}

} // namespace facebook::velox::ucx_exchange
