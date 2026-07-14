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

#include "folly/Conv.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::cudf_velox {

CudfConfig& CudfConfig::getInstance() {
  static CudfConfig instance;
  return instance;
}

void CudfConfig::initialize(
    std::unordered_map<std::string, std::string>&& config) {
  if (config.find(kCudfEnabled) != config.end()) {
    enabled = folly::to<bool>(config[kCudfEnabled]);
  }
  if (config.find(kCudfDebugEnabled) != config.end()) {
    debugEnabled = folly::to<bool>(config[kCudfDebugEnabled]);
  }
  if (config.find(kCudfMemoryResource) != config.end()) {
    memoryResource = config[kCudfMemoryResource];
  }
  if (config.find(kCudfMemoryPercent) != config.end()) {
    memoryPercent = folly::to<int32_t>(config[kCudfMemoryPercent]);
  }
  if (config.find(kCudfOutputMr) != config.end()) {
    outputMemoryResource = config[kCudfOutputMr];
  }
  if (config.find(kCudfBatchSizeMinThreshold) != config.end()) {
    batchSizeMinThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMinThreshold]);
  }
  if (config.find(kCudfBatchSizeMaxThreshold) != config.end()) {
    batchSizeMaxThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMaxThreshold]);
  }
  if (config.find(kCudfConcatOptimizationEnabled) != config.end()) {
    concatOptimizationEnabled =
        folly::to<bool>(config[kCudfConcatOptimizationEnabled]);
  }
  if (config.find(kCudfFunctionNamePrefix) != config.end()) {
    functionNamePrefix = config[kCudfFunctionNamePrefix];
  }
  if (config.find(kCudfFunctionEngine) != config.end()) {
    functionEngine = config[kCudfFunctionEngine];
  }
  if (config.find(kCudfAstExpressionEnabled) != config.end()) {
    astExpressionEnabled = folly::to<bool>(config[kCudfAstExpressionEnabled]);
  }
  if (config.find(kCudfJitExpressionEnabled) != config.end()) {
    jitExpressionEnabled = folly::to<bool>(config[kCudfJitExpressionEnabled]);
  }
  if (config.find(kCudfAstExpressionPriority) != config.end()) {
    astExpressionPriority =
        folly::to<int32_t>(config[kCudfAstExpressionPriority]);
  }
  if (config.find(kCudfAllowCpuFallback) != config.end()) {
    allowCpuFallback = folly::to<bool>(config[kCudfAllowCpuFallback]);
  }
  if (config.find(kCudfLogFallback) != config.end()) {
    logFallback = folly::to<bool>(config[kCudfLogFallback]);
  }
  if (config.find(kCudfTopNBatchSize) != config.end()) {
    topNBatchSize = folly::to<int32_t>(config[kCudfTopNBatchSize]);
  }
  if (config.find(kUcxExchange) != config.end()) {
    exchange = folly::to<bool>(config[kUcxExchange]);
  }
  if (config.find(kUcxIntraNodeExchange) != config.end()) {
    intraNodeExchange = folly::to<bool>(config[kUcxIntraNodeExchange]);
  }
  if (config.find(kUcxxErrorHandling) != config.end()) {
    ucxxErrorHandling = folly::to<bool>(config[kUcxxErrorHandling]);
  }
  if (config.find(kUcxxBlockingPolling) != config.end()) {
    ucxxBlockingPolling = folly::to<bool>(config[kUcxxBlockingPolling]);
  }
  if (config.find(kUcxExchangeLogLevel) != config.end()) {
    exchangeLogLevel = folly::to<int32_t>(config[kUcxExchangeLogLevel]);
  }
  if (config.find(kCudfTimestampUnit) != config.end()) {
    const auto& unit = config[kCudfTimestampUnit];
    if (unit == "s") {
      timestampUnit = cudf::type_id::TIMESTAMP_SECONDS;
    } else if (unit == "ms") {
      timestampUnit = cudf::type_id::TIMESTAMP_MILLISECONDS;
    } else if (unit == "us") {
      timestampUnit = cudf::type_id::TIMESTAMP_MICROSECONDS;
    } else if (unit == "ns") {
      timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
    } else {
      VELOX_FAIL(
          "Invalid timestamp unit: {}. Valid values are: s, ms, us, ns", unit);
    }
  }
}

} // namespace facebook::velox::cudf_velox
