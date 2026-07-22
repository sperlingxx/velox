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

#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <folly/String.h>
#include <folly/Uri.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
void retireRequest(
    std::shared_ptr<ucxx::Request>& current,
    std::vector<std::shared_ptr<ucxx::Request>>& inFlight) {
  // UCXX removes a request from the endpoint/worker inflight collections and
  // frees its UCP handle before invoking the user completion callback.  A
  // completed Request therefore does not need to be retained.  Keeping every
  // completed receive until exchange teardown grows without bound on packet-
  // heavy shuffles.  Preserve only requests that are genuinely still in
  // progress, and prune those as soon as a later packet observes completion.
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

const folly::F14FastMap<UcxExchangeSource::ReceiverState, std::string_view>&
receiverStateNames() {
  static const folly::F14FastMap<
      UcxExchangeSource::ReceiverState,
      std::string_view>
      kNames = {
          {UcxExchangeSource::ReceiverState::Created, "Created"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeComplete,
           "WaitingForHandshakeComplete"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeResponse,
           "WaitingForHandshakeResponse"},
          {UcxExchangeSource::ReceiverState::ReadyToReceive, "ReadyToReceive"},
          {UcxExchangeSource::ReceiverState::WaitingForMetadata,
           "WaitingForMetadata"},
          {UcxExchangeSource::ReceiverState::WaitingForReceiveCredit,
           "WaitingForReceiveCredit"},
          {UcxExchangeSource::ReceiverState::WaitingForData, "WaitingForData"},
          {UcxExchangeSource::ReceiverState::WaitingForIntraNodeData,
           "WaitingForIntraNodeData"},
          {UcxExchangeSource::ReceiverState::Done, "Done"},
      };
  return kNames;
}

int64_t maxInFlightRecvHostBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_HOST_BYTES")) {
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

std::atomic<int64_t> inFlightRecvHostBytes{0};

rmm::mr::statistics_resource_adaptor& receiveDeviceMemoryResource() {
  // Allocate UCX receive pages from the same async resource as cuDF operators.
  // A separate cuda_memory_resource cannot reuse blocks cached by the async
  // pool and can therefore fail cudaMalloc while scan/aggregation has ample
  // reusable memory in that pool. Keep a dedicated statistics adaptor so
  // receive credit continues to account for packed pages after ownership moves
  // out of UcxExchangeSource.
  static rmm::mr::statistics_resource_adaptor resource{
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          rmm::mr::get_current_device_resource_ref()}};
  return resource;
}

int64_t maxInFlightRecvDeviceBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_DEVICE_BYTES")) {
      try {
        const auto parsed = static_cast<int64_t>(std::stoll(value));
        if (parsed > 0) {
          return parsed;
        }
      } catch (...) {
      }
    }
    // A query-wide credit pool needs dependency-aware wakeups. A blind global
    // cap can deadlock a DAG when an edge holding credit is downstream of an
    // edge waiting for credit. Keep this diagnostic mechanism opt-in until
    // the coordinator owns that dependency-aware arbitration.
    return static_cast<int64_t>(0);
  }();
  return limit;
}

bool hasRecvDeviceCredit(int64_t bytes) {
  const auto limit = maxInFlightRecvDeviceBytes();
  if (limit <= 0) {
    return true;
  }
  const auto current = receiveDeviceMemoryResource().get_bytes_counter().value;
  // Packed tables cannot be split. Permit one oversized receive only when no
  // other receive page is live, matching the host-credit behavior.
  return current == 0 || current + bytes <= limit;
}

std::atomic<int64_t> lastReportedReceiveDevicePeakBucket{0};

bool tryReserveRecvHostBytes(int64_t bytes) {
  VELOX_CHECK_GE(bytes, 0);
  auto current = inFlightRecvHostBytes.load(std::memory_order_relaxed);
  while (true) {
    // The current protocol cannot split one packed table. Permit one
    // oversized receive when the process has no other host receive buffer.
    if (current > 0 && current + bytes > maxInFlightRecvHostBytes()) {
      return false;
    }
    if (inFlightRecvHostBytes.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return true;
    }
  }
}

void releaseRecvHostBytes(int64_t bytes) {
  if (bytes <= 0) {
    return;
  }
  const auto previous =
      inFlightRecvHostBytes.fetch_sub(bytes, std::memory_order_acq_rel);
  VELOX_CHECK_GE(previous, bytes);
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeSource,
    ReceiverState,
    receiverStateNames)

int64_t UcxExchangeSource::maxInFlightRecvBytes() {
  // Read once. See header for rationale: recv buffers are off the operator
  // pool, so an unbounded byte footprint scales O(#peers) and exhausts the GPU
  // at 4 peers. This cap makes the producer's tagSend block at rendezvous,
  // leaving the async operator pool headroom. Deadlock-safe: the count-based
  // resume path (UcxExchangeClient::next) drains both count and bytes.
  static const int64_t kBytes = [] {
    if (const char* env = std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_BYTES")) {
      try {
        const int64_t v = std::stoll(env);
        if (v > 0) {
          return v;
        }
      } catch (...) {
      }
    }
    return static_cast<int64_t>(8) * 1024 * 1024 * 1024; // 8 GiB default
  }();
  return kBytes;
}

void UcxExchangeSource::setState(ReceiverState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private.
UcxExchangeSource::UcxExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    std::string_view taskId,
    std::string_view host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<UcxExchangeQueue> queue)
    : CommElement(communicator),
      host_(host),
      port_(port),
      taskId_(taskId),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/*static*/
std::shared_ptr<UcxExchangeSource> UcxExchangeSource::create(
    std::string_view taskId,
    std::string_view url,
    const std::shared_ptr<UcxExchangeQueue>& queue) {
  folly::Uri uri(url);
  // Note that there is no distinct schema for the UCXX exchange.
  // The approach is to ignore the schema and not check for HTTP or HTTPS.
  // FIXME: Can't use the HTTP port as this conflicts with Prestissimo!
  // For the time being, there's an ugly hack that just increases the port by 3.
  const std::string host = uri.host();
  int port = uri.port() + 3;
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source = std::shared_ptr<UcxExchangeSource>(
      new UcxExchangeSource(communicator, taskId, host, port, key, queue));
  // register the exchange source with the communicator. This makes sure that
  // "progress" is called.
  communicator->registerCommElement(source);
  VLOG(3) << source->toString()
          << " creating UcxExchangeSource for url: " << url;
  return source;
}

void UcxExchangeSource::process() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    deliverEndMarker();
    return;
  }
  if (closed_) {
    // Driver thread called closed
    cleanUp();
    return;
  }

  switch (state_) {
    case ReceiverState::Created: {
      // Get the endpoint.
      HostPort hp{host_, port_};
      std::shared_ptr<UcxExchangeSource> selfPtr = getSelfPtr();
      auto epRef = communicator->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        setStateIf(
            ReceiverState::Created, ReceiverState::WaitingForHandshakeComplete);
        sendHandshake();
      } else {
        // connection failed.
        VLOG(0) << toString() << " Failed to connect to " << host_ << ":"
                << std::to_string(port_);
        deliverEndMarker();
        setState(ReceiverState::Done);
      }
      wakeCommunicator();
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Waiting for handshake send completion is handled by callback.
      break;
    case ReceiverState::WaitingForHandshakeResponse:
      // Waiting for HandshakeResponse is handled by callback.
      break;
    case ReceiverState::ReadyToReceive: {
      // Backpressure: don't post the next receive if the consumer queue is
      // overloaded. The source goes dormant (not in work queue) and will be
      // woken by UcxExchangeClient::next() calling resumeFromBackpressure()
      // when the queue drains below the low water mark.
      //
      // This creates natural backpressure: the server's tagSend for data
      // will block at rendezvous until we post a matching tagRecv. For
      // intra-node: the server's publish future won't resolve until we poll.
      UcxExchangeQueue::BackpressureStats stats;
      // Backpressure on BOTH item count and aggregate bytes. The byte cap is
      // what bounds the off-pool receive-buffer footprint that otherwise scales
      // O(#peers) and OOMs/deadlocks the GPU at 4 peers (the count cap alone
      // let a few large chunks fill the device before pausing).
      if (queue_->shouldPauseReceive(
              kBackpressureHighWaterMark, maxInFlightRecvBytes(), &stats)) {
        if (!backpressureActive_.exchange(true, std::memory_order_acq_rel)) {
          VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
                  << "] pausing, queueSize=" << stats.queueSize
                  << " (high=" << kBackpressureHighWaterMark
                  << "), queueBytes=" << stats.queuedBytes
                  << ", pendingReceiveBytes=" << stats.pendingReceiveBytes
                  << " (cap=" << maxInFlightRecvBytes() << ")";
        }
        // Go dormant — do NOT re-enqueue into work queue.
        // UcxExchangeClient::next() will call resumeFromBackpressure().
        break;
      }

      // Count-only backpressure (Presto-style): post the next receive directly.
      // The server's tagSend blocks at rendezvous until we post the matching
      // tagRecv, so no explicit byte-credit request is needed. Intra-node waits
      // on the registry; remote waits for UCX metadata/data tags.
      if (isIntraNodeTransfer_) {
        setStateIf(
            ReceiverState::ReadyToReceive,
            ReceiverState::WaitingForIntraNodeData);
        waitForIntraNodeData();
      } else {
        setStateIf(
            ReceiverState::ReadyToReceive, ReceiverState::WaitingForMetadata);
        getMetadata();
      }
    } break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::WaitingForReceiveCredit:
      tryStartDataReceive(
          pendingReceive_, ReceiverState::WaitingForReceiveCredit);
      break;
    case ReceiverState::WaitingForData:
      // Waiting for data is handled by an upcall from UCXX. Nothing to do.
      break;
    case ReceiverState::WaitingForIntraNodeData:
      // Poll for intra-node transfer data
      waitForIntraNodeData();
      break;
    case ReceiverState::Done:
      // We need to call clean-up in this thread to remove any state
      cleanUp();
      break;
  }
}

void UcxExchangeSource::cleanUp() {
  releaseReceiveReservation();
  pendingReceive_.reset();

  uint32_t value = static_cast<uint32_t>(getState());
  if (value != static_cast<uint32_t>(ReceiverState::Done)) {
    // Unexpected cleanup
    VLOG(3) << toString()
            << " In UcxExchangeSource::cleanUp state == " << value;
  }

  // Cancel any outstanding request. With weak_ptr callbacks, the callback
  // will safely no-op if we're destroyed before it completes.
  if (request_ && !request_->isCompleted()) {
    // The Task has failed and we may need to cancel outstanding requests
    request_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  auto communicator = communicator_.lock();
  if (communicator) {
    if (request_) {
      communicator->deferRequestCleanup(std::move(request_));
    }
    for (auto& req : completedRequests_) {
      communicator->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator) {
    communicator->unregister(getSelfPtr());
  }
}

void UcxExchangeSource::close() {
  // This is called by the driver thread so we need to be careful to
  // indicate to the process thread that we are closing and
  // let it do the actual cleaning up.

  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }

  VLOG(2) << "[UCX-SOURCE-CLOSE] " << toString()
          << " state=" << toName(getState()) << " seq=" << sequenceNumber_
          << " hasRequest=" << (request_ != nullptr) << " atEnd=" << atEnd_;

  // Guarantee the end marker is delivered before transitioning to Done.
  deliverEndMarker();

  // Let the Communicator progress thread do the actual clean-up.
  setState(ReceiverState::Done);
  wakeCommunicator();
}

void UcxExchangeSource::closeWithError(std::string error) {
  if (!closed_.load(std::memory_order_acquire) &&
      getState() != ReceiverState::Done && !atEnd_) {
    queue_->setError(std::move(error));
  }
  close();
}

void UcxExchangeSource::resumeFromBackpressure() {
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel)) {
    VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
            << "] resumed by consumer, queueSize=" << queue_->size();
    wakeCommunicator();
  }
}

folly::F14FastMap<std::string, int64_t> UcxExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;

  // these metrics will be aggregated over all exchange sources of the same
  // exchange client.
  map["ucxExchangeSource.numPackedColumns"] = metrics_.numPackedColumns_;
  map["ucxExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["ucxExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// private methods ---
PartitionKey UcxExchangeSource::extractTaskAndDestinationId(
    std::string_view path) {
  // The URL path has the form: /v1/task/<taskId>/results/<destinationId>"
  std::vector<folly::StringPiece> components;
  folly::split('/', path, components, true);

  VELOX_CHECK_EQ(components[0], "v1");
  VELOX_CHECK_EQ(components[1], "task");
  VELOX_CHECK_EQ(components[3], "results");

  uint32_t destinationId;
  try {
    destinationId = static_cast<uint32_t>(std::stoul(components[4].str()));
  } catch (const std::exception& e) {
    VELOX_UNSUPPORTED("Illegal destination in task URL: {}", path);
  }

  return PartitionKey{components[2].str(), destinationId};
}

std::shared_ptr<UcxExchangeSource> UcxExchangeSource::getSelfPtr() {
  std::shared_ptr<UcxExchangeSource> ptr;
  try {
    ptr = shared_from_this();
  } catch (std::bad_weak_ptr& exp) {
    ptr = nullptr;
  }
  return ptr;
}

void UcxExchangeSource::wakeCommunicator() {
  if (auto communicator = tryCommunicator()) {
    communicator->addToWorkQueue(getSelfPtr());
  }
}

void UcxExchangeSource::enqueue(
    PackedTableWithStreamPtr data,
    int64_t reservedReceiveBytes) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    queue_->enqueueLocked(std::move(data), queuePromises, reservedReceiveBytes);
  }
  // wake up consumers of the UcxExchangeQueue
  for (auto& promise : queuePromises) {
    promise.setValue();
  }
}

void UcxExchangeSource::deliverEndMarker() {
  if (!registered_.load(std::memory_order_acquire)) {
    // Never registered with queue -- don't deliver end marker to avoid
    // spurious numCompleted_ increments.
    return;
  }
  bool expected = false;
  if (!endMarkerDelivered_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    // Already delivered by another thread/path.
    return;
  }
  VLOG(3) << toString() << " delivering end-of-stream marker to queue";
  enqueue(nullptr);
}

void UcxExchangeSource::releaseReceiveReservation() {
  if (reservedReceiveBytes_ > 0) {
    queue_->releaseReservedReceive(reservedReceiveBytes_);
    reservedReceiveBytes_ = 0;
  }
  if (reservedGlobalHostReceiveBytes_ > 0) {
    releaseRecvHostBytes(reservedGlobalHostReceiveBytes_);
    reservedGlobalHostReceiveBytes_ = 0;
  }
}

void UcxExchangeSource::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void UcxExchangeSource::sendHandshake() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    deliverEndMarker();
    return;
  }
  handshakeRequestBuffer_ = std::make_shared<HandshakeMsg>();
  auto& handshakeReq = handshakeRequestBuffer_;
  handshakeReq->destination = partitionKey_.destination;
  // Use sizeof(...) - 1 and explicitly null-terminate to prevent buffer
  // overread if taskId is longer than the destination buffer.
  strncpy(
      handshakeReq->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshakeReq->taskId) - 1);
  handshakeReq->taskId[sizeof(handshakeReq->taskId) - 1] = '\0';
  handshakeReq->workerId = communicator->getWorkerId();

  VLOG(2) << "[UCX-SOURCE-HANDSHAKE-SEND] localTask=" << taskId_
          << " remoteTask=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination << " peer=" << host_
          << ":" << port_ << " workerId=" << handshakeReq->workerId;

  // Create the handshake which will register client's existence with the server
  ucxx::AmReceiverCallbackInfo info(
      communicator->kAmCallbackOwner, communicator->kAmCallbackId);
  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  // Pass handshakeReq as the callback arg to keep the send buffer alive until
  // the async amSend completes. UCXX stores it as shared_ptr<void> but the
  // type-erased deleter still calls ~HandshakeMsg correctly.
  request_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(*handshakeReq),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshake(status, arg);
        }
      },
      handshakeReq);
}

void UcxExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  // arg holds the HandshakeMsg that was sent — it is unused here because this
  // is a send completion callback (the outgoing data has already been
  // transmitted). The parameter exists only because UCXX uses it as a lifetime
  // handle; letting it go out of scope releases the send buffer.

  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshake called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
  } else {
    VLOG(3) << toString() << "+ onHandshake " << ucs_status_string(status)
            << " peer=" << host_ << ":" << port_;
    // Now wait for the HandshakeResponse from the server
    setStateIf(
        ReceiverState::WaitingForHandshakeComplete,
        ReceiverState::WaitingForHandshakeResponse);
    receiveHandshakeResponse();
  }
}

void UcxExchangeSource::getMetadata() {
  // Use kMaxMetaBufSize to support tables with many columns.
  // The sender allocates exact size needed; receiver pre-allocates max. A
  // source has only one receive in flight, so reuse this buffer for all
  // packets instead of creating one mmap-backed allocation per request.
  if (metadataReceiveBuffer_ == nullptr) {
    metadataReceiveBuffer_ =
        std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
  }
  auto metadataReq = metadataReceiveBuffer_;
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, sequenceNumber_);

  VLOG(3) << toString()
          << " waiting for metadata for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << metadataTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      kMaxMetaBufSize,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        auto metadata = std::static_pointer_cast<std::vector<uint8_t>>(arg);
        if (auto self = weak.lock()) {
          self->onMetadata(status, metadata);
        }
      },
      metadataReq);
}

void UcxExchangeSource::onMetadata(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SOURCE-METADATA-AFTER-CLOSE] " << toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForMetadata) {
    VLOG(2) << toString() << " onMetadata called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onMetadata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
  } else {
    VELOX_CHECK_NOT_NULL(arg, "Didn't get metadata");

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    auto ptr = std::make_shared<DataAndMetadata>();

    ptr->metadata =
        std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));

    VLOG(3) << toString()
            << " Datasize bytes == " << ptr->metadata.dataSizeBytes;

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      atEnd_ = true;
      // enqueue a nullpointer to mark the end for this source.
      VLOG(3) << "There is no more data to transfer for " << toString();
      deliverEndMarker();
      setStateIf(ReceiverState::WaitingForMetadata, ReceiverState::Done);
      wakeCommunicator();
      // jump out of this function.
      return;
    }

    pendingReceive_ = ptr;
    tryStartDataReceive(pendingReceive_, ReceiverState::WaitingForMetadata);
  }
}

bool UcxExchangeSource::tryStartDataReceive(
    const std::shared_ptr<DataAndMetadata>& ptr,
    ReceiverState expectedState) {
  if (ptr == nullptr) {
    return false;
  }
  auto communicator = tryCommunicator();
  if (!communicator) {
    return false;
  }

  UcxExchangeQueue::BackpressureStats stats;
  if (!queue_->tryReserveReceive(
          ptr->metadata.dataSizeBytes, maxInFlightRecvBytes(), &stats)) {
    if (!backpressureActive_.exchange(true, std::memory_order_acq_rel)) {
      VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
              << "] waiting for receive credit, requestedBytes="
              << ptr->metadata.dataSizeBytes
              << ", queueBytes=" << stats.queuedBytes
              << ", pendingReceiveBytes=" << stats.pendingReceiveBytes
              << " (cap=" << maxInFlightRecvBytes() << ")";
    }
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    return false;
  }
  reservedReceiveBytes_ = ptr->metadata.dataSizeBytes;

  const bool useHostStaging = !communicator->hasCudaTransport();
  if (useHostStaging && !tryReserveRecvHostBytes(ptr->metadata.dataSizeBytes)) {
    queue_->releaseReservedReceive(reservedReceiveBytes_);
    reservedReceiveBytes_ = 0;
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    // Global credit is shared across otherwise unrelated ExchangeQueues, so
    // a dequeue from this queue cannot wake us. Retry from the communicator
    // work loop; completed receives release the process-wide credit below.
    wakeCommunicator();
    return false;
  }
  if (useHostStaging) {
    reservedGlobalHostReceiveBytes_ = ptr->metadata.dataSizeBytes;
  }

  if (!hasRecvDeviceCredit(ptr->metadata.dataSizeBytes)) {
    releaseReceiveReservation();
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    // A different ExchangeQueue may release the process-wide credit. Retry
    // from the communicator loop instead of waiting on this queue's promise.
    wakeCommunicator();
    return false;
  }

  // REMOTE EXCHANGE PATH: Allocate buffer and receive via UCXX.
  auto stream =
      facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
  ptr->stream = stream;

  // UCX writes receive buffers from its progress thread, outside CUDA stream
  // ordering. Allocate from the shared async pool, then synchronize this
  // allocation stream before exposing the pointer to UCX. Receive completion
  // transfers ownership to a packed table that retains the same stream, so
  // downstream work and eventual deallocation remain stream ordered.
  auto& recvMemoryResource = receiveDeviceMemoryResource();
  const auto allocateReceiveBuffer = [&]() {
    ptr->dataBuf = std::make_unique<rmm::device_buffer>(
        ptr->metadata.dataSizeBytes,
        stream,
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            recvMemoryResource});
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  };
  try {
    allocateReceiveBuffer();
  } catch (const rmm::bad_alloc& firstError) {
    // On the allocation-failure slow path, wait for stream-ordered frees and
    // let the shared async pool reclaim completed blocks, then retry once.
    cudaGetLastError();
    const auto syncStatus = cudaDeviceSynchronize();
    if (syncStatus == cudaSuccess) {
      try {
        allocateReceiveBuffer();
        LOG(WARNING)
            << toString()
            << " recovered UCX receive allocation after device synchronize: "
            << ptr->metadata.dataSizeBytes
            << " bytes; first error: " << firstError.what();
      } catch (const rmm::bad_alloc& retryError) {
        VLOG(0) << toString() << " *** RMM failed to allocate "
                << ptr->metadata.dataSizeBytes
                << " bytes after device synchronize retry: "
                << retryError.what();
      }
    } else {
      VLOG(0) << toString()
              << " *** cudaDeviceSynchronize failed while recovering "
                 "receive allocation: "
              << cudaGetErrorString(syncStatus);
    }
  }
  if (ptr->dataBuf != nullptr) {
    if (facebook::velox::cudf_velox::deviceMemoryDiagnosticsEnabled()) {
      constexpr int64_t kReportStep = 512LL << 20;
      const auto bytes = recvMemoryResource.get_bytes_counter();
      const auto peakBucket = bytes.peak / kReportStep;
      auto reported =
          lastReportedReceiveDevicePeakBucket.load(std::memory_order_relaxed);
      bool crossedPeakBucket = false;
      while (peakBucket > reported) {
        if (lastReportedReceiveDevicePeakBucket.compare_exchange_weak(
                reported,
                peakBucket,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
          crossedPeakBucket = true;
          break;
        }
      }
      if (crossedPeakBucket ||
          ptr->metadata.dataSizeBytes >= static_cast<uint64_t>(64) << 20) {
        facebook::velox::cudf_velox::logDeviceMemorySnapshot(
            fmt::format(
                "operator=UcxExchangeSource state=receive.allocate "
                "task={} remoteTask={} destination={} allocationBytes={} "
                "ucxRecvCurrentBytes={} ucxRecvPeakBytes={} "
                "ucxRecvTotalBytes={} queueBytes={} pendingReceiveBytes={} cap={}",
                taskId_,
                partitionKey_.taskId,
                partitionKey_.destination,
                ptr->metadata.dataSizeBytes,
                bytes.value,
                bytes.peak,
                bytes.total,
                stats.queuedBytes,
                stats.pendingReceiveBytes,
                maxInFlightRecvBytes()));
      }
    }
  } else {
    releaseReceiveReservation();
    queue_->setError("Failed to alloc GPU memory");
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
    return false;
  }

  VLOG(3) << toString() << " Allocated " << ptr->metadata.dataSizeBytes
          << " bytes of device memory";

  // CUDA-aware transports can receive directly into the final device buffer.
  // Only stage through host memory when the active UCX context has confirmed
  // that no CUDA transport is available; sm/tcp cannot write through a device
  // pointer safely in that configuration.
  void* receiveBuffer = ptr->dataBuf->data();
  if (useHostStaging) {
    const auto receiveSize = static_cast<size_t>(ptr->metadata.dataSizeBytes);
    if (dataReceiveBuffer_ == nullptr) {
      dataReceiveBuffer_ = std::make_shared<std::vector<uint8_t>>(receiveSize);
    } else if (dataReceiveBuffer_->size() < receiveSize) {
      dataReceiveBuffer_->resize(receiveSize);
    }
    ptr->hostData = dataReceiveBuffer_;
    receiveBuffer = ptr->hostData->data();
  }
  VLOG(2) << toString() << " posting "
          << (useHostStaging ? "host-staged" : "direct-device")
          << " receive for " << ptr->metadata.dataSizeBytes << " bytes";

  uint64_t dataTag = getDataTag(partitionKeyHash_, sequenceNumber_);
  VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << dataTag << std::dec;

  if (!setStateIf(expectedState, ReceiverState::WaitingForData)) {
    releaseReceiveReservation();
    VLOG(1) << toString() << " tryStartDataReceive Invalid previous state ";
    return false;
  }

  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      receiveBuffer,
      ptr->metadata.dataSizeBytes,
      ucxx::Tag{dataTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onData(status, arg);
        }
      },
      ptr);
  pendingReceive_.reset();
  return true;
}

void UcxExchangeSource::onData(ucs_status_t status, std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SOURCE-DATA-AFTER-CLOSE] " << toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    releaseReceiveReservation();
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForData) {
    VLOG(2) << toString() << " onData called in state " << toName(getState())
            << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onData " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << "[UCX-SOURCE-DATA-ERROR] " << toString()
            << " seq=" << sequenceNumber_ << " state=" << toName(getState())
            << " error=" << errorMsg;
    releaseReceiveReservation();
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    if (ptr->hostData != nullptr) {
      CUDF_CUDA_TRY(cudaMemcpy(
          ptr->dataBuf->data(),
          ptr->hostData->data(),
          ptr->metadata.dataSizeBytes,
          cudaMemcpyHostToDevice));
      ptr->hostData.reset();
      if (reservedGlobalHostReceiveBytes_ > 0) {
        releaseRecvHostBytes(reservedGlobalHostReceiveBytes_);
        reservedGlobalHostReceiveBytes_ = 0;
      }
    }
    // The source-level pageable fallback buffer is intentionally retained and
    // reused by the next serial host-staged receive.

    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    // Create packed_columns from the received metadata and data buffer
    cudf::packed_columns packedCols(
        std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));

    // Unpack to get the table_view and create a packed_table
    cudf::table_view tableView = cudf::unpack(packedCols);
    auto packedTable = std::make_unique<cudf::packed_table>(
        cudf::packed_table{tableView, std::move(packedCols)});

    // Bundle the packed_table with the stream that was used for allocation
    auto data = std::make_unique<PackedTableWithStream>(
        std::move(packedTable), ptr->stream);

    const int64_t reservedReceiveBytes = reservedReceiveBytes_;
    enqueue(std::move(data), reservedReceiveBytes);
    reservedReceiveBytes_ = 0;
    setStateIf(ReceiverState::WaitingForData, ReceiverState::ReadyToReceive);
  }
  wakeCommunicator();
}

void UcxExchangeSource::receiveHandshakeResponse() {
  auto responseBuffer = std::make_shared<HandshakeResponse>();
  uint64_t responseTag = getHandshakeResponseTag(partitionKeyHash_);

  VLOG(3) << toString()
          << " waiting for HandshakeResponse with tag: " << std::hex
          << responseTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      responseBuffer.get(),
      sizeof(*responseBuffer),
      ucxx::Tag{responseTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshakeResponse(status, arg);
        }
      },
      responseBuffer);
}

void UcxExchangeSource::onHandshakeResponse(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // The peer has now consumed the AM payload and returned a response, so the
  // retained handshake buffer can finally be released.
  handshakeRequestBuffer_.reset();
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " onHandshakeResponse called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive HandshakeResponse from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
    return;
  }

  std::shared_ptr<HandshakeResponse> response =
      std::static_pointer_cast<HandshakeResponse>(arg);

  isIntraNodeTransfer_ = response->isIntraNodeTransfer;

  VLOG(2) << "[UCX-SOURCE-HANDSHAKE-RESPONSE] localTask=" << taskId_
          << " remoteTask=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination << " peer=" << host_
          << ":" << port_ << " isIntraNodeTransfer=" << isIntraNodeTransfer_;

  setStateIf(
      ReceiverState::WaitingForHandshakeResponse,
      ReceiverState::ReadyToReceive);
  wakeCommunicator();
}

void UcxExchangeSource::waitForIntraNodeData() {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " waitForIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  IntraNodeTransferKey key{
      partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};

  auto result = IntraNodeTransferRegistry::getInstance()->poll(key);

  if (!result.has_value()) {
    // Event-driven wait: re-queuing here would busy-spin the single-threaded
    // Communicator and can starve the same-process producer's publish() (which
    // needs that same thread), livelocking the intra-node transfer. Instead
    // register a one-shot wakeup and go dormant; publish()/cancelTask()
    // re-enqueues this source exactly once when data is available. The weak_ptr
    // keeps the wakeup safe if the source is destroyed first. Capture the
    // Communicator weakly as well so a dormant waiter cannot extend the
    // singleton lifetime during executor shutdown.
    auto self = getSelfPtr();
    std::weak_ptr<CommElement> weakSelf = self;
    auto weakCommunicator = communicator_;
    const bool readyNow =
        IntraNodeTransferRegistry::getInstance()->registerWaiter(
            key, [weakSelf, weakCommunicator]() {
              if (auto source = weakSelf.lock(); source) {
                auto communicator = weakCommunicator.lock();
                if (!communicator) {
                  return;
                }
                communicator->addToWorkQueue(source);
              }
            });
    if (readyNow) {
      // Data landed (or the task was cancelled) between poll and register —
      // re-poll once instead of going dormant.
      wakeCommunicator();
    }
    return;
  }

  intraNodePollCount_ = 0;
  onIntraNodeData(std::move(result->data), result->stream, result->atEnd);
}

void UcxExchangeSource::onIntraNodeData(
    std::shared_ptr<cudf::packed_columns> data,
    rmm::cuda_stream_view producerStream,
    bool atEnd) {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  if (atEnd) {
    // End of stream
    atEnd_ = true;
    VLOG(3) << toString() << " Intra-node transfer: end of stream";
    deliverEndMarker();
    setState(ReceiverState::Done);

    wakeCommunicator();
    return;
  }

  if (!data) {
    // Error - should not happen if atEnd is false
    std::string errorMsg = fmt::format(
        "Intra-node transfer data is null for task {}, dest {}, seq {}",
        partitionKey_.taskId,
        partitionKey_.destination,
        sequenceNumber_);
    VLOG(0) << toString() << " " << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
    return;
  }

  VLOG(3) << toString()
          << " Intra-node transfer: received data for seq=" << sequenceNumber_
          << " size=" << data->gpu_data->size();

  metrics_.numPackedColumns_.addValue(1);
  metrics_.totalBytes_.addValue(data->gpu_data->size());
  // Broadcast output can share the same packed_columns across multiple
  // destinations. Keep the zero-copy path for uniquely owned partitioned
  // pages, but clone shared pages before moving out of them.
  const bool sharedPage = data.use_count() > 1;
  // The received device buffer was allocated on `producerStream`, and with the
  // stream-ordered async MR its cudaFreeAsync stays bound to that stream. For
  // the uniquely-owned (partitioned) page we MOVE the buffer out, so tag the
  // rebuilt vector with `producerStream` itself: the downstream read and the
  // eventual async free then share one stream, so the free can never recycle
  // the block under a still-pending consumer read on a different pool stream
  // (previously seen as flaky garbage on Q17, both high and low). Shared pages
  // are cloned onto a fresh pool stream (the source buffer stays live in the
  // shared owner and the producer host-synchronizes before publishing).
  auto stream = sharedPage
      ? facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream()
      : producerStream;
  cudf::packed_columns packedCols(
      sharedPage ? std::make_unique<std::vector<uint8_t>>(*data->metadata)
                 : std::move(data->metadata),
      sharedPage ? std::make_unique<rmm::device_buffer>(
                       data->gpu_data->data(), data->gpu_data->size(), stream)
                 : std::move(data->gpu_data));

  // Unpack to get the table_view and create a packed_table
  cudf::table_view tableView = cudf::unpack(packedCols);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedCols)});

  auto tableWithStream =
      std::make_unique<PackedTableWithStream>(std::move(packedTable), stream);

  enqueue(std::move(tableWithStream));

  this->sequenceNumber_++;
  setStateIf(
      ReceiverState::WaitingForIntraNodeData, ReceiverState::ReadyToReceive);
  wakeCommunicator();
}

bool UcxExchangeSource::setStateIf(
    UcxExchangeSource::ReceiverState expected,
    UcxExchangeSource::ReceiverState desired) {
  ReceiverState exp = expected;
  // since spurious failures can happen even if state_ == expected, we need
  // to do this in a loop.
  while (!state_.compare_exchange_strong(
      exp, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    if (exp != expected) {
      // no spurious failure, state isn't what we've expected.
      return false;
    }
    // spurious failure.
    exp = expected; // reset for the next try
  }
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(expected) << " -> " << toName(desired);
  return true;
}

} // namespace facebook::velox::ucx_exchange
