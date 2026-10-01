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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDataSource.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfPreparedIcebergSplit.h"

#include "velox/common/Casts.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"

#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {
std::string normalizeFilePath(std::string path) {
  constexpr std::string_view kFilePrefix{"file:"};
  constexpr std::string_view kS3APrefix{"s3a:"};
  if (path.starts_with(kFilePrefix)) {
    return path.substr(kFilePrefix.size());
  }
  if (path.starts_with(kS3APrefix)) {
    path.erase(kS3APrefix.size() - 2, 1);
  }
  return path;
}
} // namespace

CudfIcebergDataSource::CudfIcebergDataSource(
    const RowTypePtr& outputType,
    const velox_connector::ConnectorTableHandlePtr& tableHandle,
    const velox_connector::ColumnHandleMap& columnHandles,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig)
    : CudfHiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig),
      hiveConfig_(hiveConfig) {}

void CudfIcebergDataSource::convertSplit(
    std::shared_ptr<velox_connector::ConnectorSplit> split) {
  // Convert `ConnectorSplit` to `HiveIcebergSplit`
  icebergSplit_ =
      checkedPointerCast<const velox_iceberg::HiveIcebergSplit>(split);

  if (auto prepared =
          std::dynamic_pointer_cast<CudfPreparedIcebergSplit>(split)) {
    recordPreparedBatch(
        prepared->unitCount(), prepared->sourceCount(), prepared->size());
    if (prepared->isRowGroupBatch()) {
      setPreparedHybridScan(prepared->takePreparedHybridScan());
    } else {
      setPreparedDataSources(prepared->takeDataSources());
    }
  }

  // Convert `ConnectorSplit` to `CudfHiveConnectorSplit`
  CudfHiveDataSource::convertSplit(split);

  split_->coalescedFiles.reserve(icebergSplit_->coalescedFiles.size());
  for (const auto& file : icebergSplit_->coalescedFiles) {
    split_->coalescedFiles.push_back(
        {normalizeFilePath(file.filePath), file.length});
  }
}

bool CudfIcebergDataSource::prepareSplitOnly(
    const std::shared_ptr<velox_connector::ConnectorSplit>& split,
    CudfSplitReader& reader,
    dwio::common::RuntimeStatistics& runtimeStats) {
  auto request =
      std::dynamic_pointer_cast<const CudfFilePreparationSplit>(split);
  if (request == nullptr) {
    return false;
  }

  auto& icebergReader = static_cast<CudfIcebergSplitReader&>(reader);
  if (request->isRowGroupPlanRequest()) {
    auto plan = icebergReader.planHostReadyFile(
        request->admission(), request->cancellationToken(), runtimeStats);
    if (plan.has_value()) {
      request->result()->publishPlan(std::move(*plan));
    } else {
      request->result()->fallback();
    }
    return true;
  }
  if (request->isRowGroupPayloadRequest()) {
    auto dataSource = icebergReader.prepareHostReadyRowGroup(
        request->rowGroupPlan().ranges,
        request->admission(),
        request->cancellationToken());
    const auto fileSize = static_cast<uint64_t>(dataSource->size());
    request->result()->publish(
        CudfPreparedIcebergFile{
            request->descriptorOrdinal(),
            request->descriptor(),
            std::move(dataSource),
            fileSize,
            request->metadata(),
            request->rowGroupPlan().rowGroup,
            request->rowGroupPlan().logicalBytes});
    return true;
  }
  auto prepared = icebergReader.prepareHostReadyFile(
      request->admission(), request->cancellationToken(), runtimeStats);
  if (!prepared.has_value()) {
    request->result()->fallback();
    return true;
  }
  const auto preparedFileSize =
      static_cast<uint64_t>(prepared->dataSource->size());
  request->result()->publish(
      CudfPreparedIcebergFile{
          request->descriptorOrdinal(),
          request->descriptor(),
          std::move(prepared->dataSource),
          preparedFileSize,
          std::move(prepared->metadata)});
  return true;
}

std::unique_ptr<CudfSplitReader>
CudfIcebergDataSource::createCudfSplitReader() {
  return std::make_unique<CudfIcebergSplitReader>(
      split_,
      icebergSplit_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      hiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
