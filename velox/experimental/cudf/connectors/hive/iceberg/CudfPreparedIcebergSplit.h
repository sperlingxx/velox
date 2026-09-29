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

#pragma once

#include "velox/experimental/cudf/connectors/hive/CudfScanPreparation.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"

#include "velox/connectors/hive/iceberg/IcebergSplit.h"

#include <cudf/io/datasource.hpp>

#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

enum class CudfPreparationMode { kFileRegular, kRowGroupHybrid };

struct CudfPreparedIcebergFile {
  uint64_t descriptorOrdinal{0};
  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> descriptor;
  std::shared_ptr<cudf::io::datasource> dataSource;
  uint64_t fileSize{0};
  std::shared_ptr<const cudf::io::parquet::FileMetaData> metadata;
  std::optional<cudf::size_type> rowGroup;
  uint64_t preparedBytes{0};
};

/// Result channel shared between a singleton preparation split and the
/// generated split source. AsyncSource readiness establishes completion; this
/// mutex only protects cancellation and direct test use.
class CudfFilePreparationResult final {
 public:
  void publish(CudfPreparedIcebergFile file) {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!complete_);
    file_ = std::move(file);
    complete_ = true;
  }

  void publishPlan(PreparedParquetFilePlan plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!complete_);
    plan_ = std::move(plan);
    complete_ = true;
  }

  void fallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!complete_);
    complete_ = true;
  }

  std::optional<CudfPreparedIcebergFile> takeFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(complete_, "Preparation result is not complete");
    VELOX_CHECK(!plan_.has_value(), "Preparation result contains a file plan");
    return std::move(file_);
  }

  std::optional<PreparedParquetFilePlan> takePlan() {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(complete_, "Preparation result is not complete");
    VELOX_CHECK(!file_.has_value(), "Preparation result contains a ready file");
    return std::move(plan_);
  }

 private:
  std::mutex mutex_;
  std::optional<CudfPreparedIcebergFile> file_;
  std::optional<PreparedParquetFilePlan> plan_;
  bool complete_{false};
};

/// In-process singleton split consumed only by the TableScan preload path.
/// The DataSource prepares retained host ranges and publishes them through
/// result(); this split is never handed to a scan driver.
class CudfFilePreparationSplit final : public velox_iceberg::HiveIcebergSplit {
 public:
  CudfFilePreparationSplit(
      uint64_t descriptorOrdinal,
      std::shared_ptr<velox_iceberg::HiveIcebergSplit> descriptor,
      std::shared_ptr<HostRangeAdmission> admission,
      folly::CancellationToken cancellationToken,
      CudfPreparationMode mode = CudfPreparationMode::kFileRegular,
      std::shared_ptr<const cudf::io::parquet::FileMetaData> metadata = nullptr,
      std::optional<PreparedParquetRowGroupPlan> rowGroupPlan = std::nullopt)
      : velox_iceberg::HiveIcebergSplit(
            checked(descriptor).connectorId,
            checked(descriptor).filePath,
            checked(descriptor).fileFormat,
            checked(descriptor).start,
            checked(descriptor).length,
            checked(descriptor).partitionKeys,
            checked(descriptor).tableBucketNumber,
            preparationCustomInfo(
                checked(descriptor).customSplitInfo,
                mode,
                rowGroupPlan),
            checked(descriptor).extraFileInfo,
            checked(descriptor).cacheable,
            checked(descriptor).deleteFiles,
            checked(descriptor).infoColumns,
            checked(descriptor).properties,
            checked(descriptor).dataSequenceNumber,
            checked(descriptor).coalescedFiles),
        descriptorOrdinal_(descriptorOrdinal),
        descriptor_(std::move(descriptor)),
        admission_(std::move(admission)),
        cancellationToken_(std::move(cancellationToken)),
        mode_(mode),
        metadata_(std::move(metadata)),
        rowGroupPlan_(std::move(rowGroupPlan)),
        result_(std::make_shared<CudfFilePreparationResult>()) {
    VELOX_CHECK_NOT_NULL(admission_);
    VELOX_CHECK(descriptor_->coalescedFiles.empty());
    VELOX_CHECK_EQ(
        rowGroupPlan_.has_value(),
        metadata_ != nullptr,
        "Row-group payload requires its shared file metadata");
    VELOX_CHECK(
        !rowGroupPlan_.has_value() ||
        mode_ == CudfPreparationMode::kRowGroupHybrid);
  }

  uint64_t descriptorOrdinal() const {
    return descriptorOrdinal_;
  }

  const std::shared_ptr<velox_iceberg::HiveIcebergSplit>& descriptor() const {
    return descriptor_;
  }

  HostRangeAdmission& admission() const {
    return *admission_;
  }

  const folly::CancellationToken& cancellationToken() const {
    return cancellationToken_;
  }

  CudfPreparationMode mode() const {
    return mode_;
  }

  bool isRowGroupPlanRequest() const {
    return mode_ == CudfPreparationMode::kRowGroupHybrid &&
        !rowGroupPlan_.has_value();
  }

  bool isRowGroupPayloadRequest() const {
    return rowGroupPlan_.has_value();
  }

  const std::shared_ptr<const cudf::io::parquet::FileMetaData>& metadata()
      const {
    return metadata_;
  }

  const PreparedParquetRowGroupPlan& rowGroupPlan() const {
    VELOX_CHECK(rowGroupPlan_.has_value());
    return *rowGroupPlan_;
  }

  const std::shared_ptr<CudfFilePreparationResult>& result() const {
    return result_;
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED(
        "CudfFilePreparationSplit is an in-process preparation request");
  }

 private:
  static const velox_iceberg::HiveIcebergSplit& checked(
      const std::shared_ptr<velox_iceberg::HiveIcebergSplit>& split) {
    VELOX_CHECK_NOT_NULL(split);
    return *split;
  }

  static std::unordered_map<std::string, std::string> preparationCustomInfo(
      const std::unordered_map<std::string, std::string>& original,
      CudfPreparationMode mode,
      const std::optional<PreparedParquetRowGroupPlan>& rowGroupPlan) {
    auto result = original;
    result["cudf_prepare_file"] = "true";
    if (mode == CudfPreparationMode::kRowGroupHybrid) {
      result
          [rowGroupPlan.has_value() ? "cudf_prepare_row_group"
                                    : "cudf_plan_row_groups"] = "true";
    }
    return result;
  }

  const uint64_t descriptorOrdinal_;
  const std::shared_ptr<velox_iceberg::HiveIcebergSplit> descriptor_;
  const std::shared_ptr<HostRangeAdmission> admission_;
  const folly::CancellationToken cancellationToken_;
  const CudfPreparationMode mode_;
  const std::shared_ptr<const cudf::io::parquet::FileMetaData> metadata_;
  const std::optional<PreparedParquetRowGroupPlan> rowGroupPlan_;
  const std::shared_ptr<CudfFilePreparationResult> result_;
};

/// In-process split generated after complete host readiness. Serialization is
/// intentionally unsupported because the datasource owners are process-local.
class CudfPreparedIcebergSplit final : public velox_iceberg::HiveIcebergSplit {
 public:
  explicit CudfPreparedIcebergSplit(std::vector<CudfPreparedIcebergFile> files)
      : velox_iceberg::HiveIcebergSplit(
            primary(files).connectorId,
            primary(files).filePath,
            primary(files).fileFormat,
            primary(files).start,
            primary(files).length,
            primary(files).partitionKeys,
            primary(files).tableBucketNumber,
            preparedCustomInfo(primary(files).customSplitInfo, files),
            primary(files).extraFileInfo,
            primary(files).cacheable,
            primary(files).deleteFiles,
            primary(files).infoColumns,
            primary(files).properties,
            primary(files).dataSequenceNumber,
            makeCoalescedFiles(files)),
        files_(std::move(files)),
        selectedBytes_(sumFileSizes(files_)),
        sourceCount_(countDistinctSources(files_)) {
    VELOX_CHECK(!files_.empty());
    const auto& first = *files_.front().descriptor;
    const auto rowGroupBatch = files_.front().rowGroup.has_value();
    for (const auto& file : files_) {
      VELOX_CHECK_NOT_NULL(file.descriptor);
      VELOX_CHECK_NOT_NULL(file.dataSource);
      VELOX_CHECK_EQ(file.dataSource->size(), file.fileSize);
      VELOX_CHECK(file.descriptor->deleteFiles.empty());
      VELOX_CHECK(file.descriptor->coalescedFiles.empty());
      VELOX_CHECK(
          file.descriptor->partitionKeys == first.partitionKeys,
          "Prepared files have different partition constants");
      VELOX_CHECK(
          file.descriptor->infoColumns == first.infoColumns,
          "Prepared files have different metadata constants");
      VELOX_CHECK_EQ(file.descriptor->fileFormat, first.fileFormat);
      VELOX_CHECK_EQ(file.rowGroup.has_value(), rowGroupBatch);
      if (rowGroupBatch) {
        VELOX_CHECK_NOT_NULL(file.metadata);
      }
    }
    batchSizeHint = first.batchSizeHint;
  }

  const std::vector<CudfPreparedIcebergFile>& files() const {
    return files_;
  }

  std::vector<std::shared_ptr<cudf::io::datasource>> dataSources() const {
    std::vector<std::shared_ptr<cudf::io::datasource>> sources;
    sources.reserve(files_.size());
    for (const auto& file : files_) {
      sources.push_back(file.dataSource);
    }
    return sources;
  }

  std::vector<std::shared_ptr<cudf::io::datasource>> takeDataSources() {
    std::vector<std::shared_ptr<cudf::io::datasource>> sources;
    sources.reserve(files_.size());
    for (auto& file : files_) {
      VELOX_CHECK_NOT_NULL(file.dataSource);
      sources.push_back(std::move(file.dataSource));
    }
    return sources;
  }

  bool isRowGroupBatch() const {
    return files_.front().rowGroup.has_value();
  }

  PreparedHybridScanBatch takePreparedHybridScan() {
    VELOX_CHECK(isRowGroupBatch());
    PreparedHybridScanBatch batch;
    batch.dataSources.reserve(files_.size());
    batch.metadatas.reserve(files_.size());
    batch.rowGroups.reserve(files_.size());
    for (auto& file : files_) {
      VELOX_CHECK_NOT_NULL(file.dataSource);
      batch.dataSources.push_back(std::move(file.dataSource));
      batch.metadatas.push_back(file.metadata);
      batch.rowGroups.push_back({*file.rowGroup});
    }
    return batch;
  }

  uint64_t size() const override {
    return selectedBytes_;
  }

  size_t unitCount() const {
    return files_.size();
  }

  size_t sourceCount() const {
    return sourceCount_;
  }

  std::string toString() const override {
    return fmt::format(
        "CudfPreparedIcebergSplit[files={}, bytes={}]",
        files_.size(),
        selectedBytes_);
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED(
        "CudfPreparedIcebergSplit is an in-process generated split");
  }

 private:
  static const velox_iceberg::HiveIcebergSplit& primary(
      const std::vector<CudfPreparedIcebergFile>& files) {
    VELOX_CHECK(!files.empty(), "Prepared batch must contain a file");
    VELOX_CHECK_NOT_NULL(files.front().descriptor);
    return *files.front().descriptor;
  }

  static std::unordered_map<std::string, std::string> preparedCustomInfo(
      const std::unordered_map<std::string, std::string>& original,
      const std::vector<CudfPreparedIcebergFile>& files) {
    auto result = original;
    result["cudf_prepared_batch"] = "true";
    if (!files.empty() && files.front().rowGroup.has_value()) {
      result["cudf_prepared_row_group_batch"] = "true";
    }
    return result;
  }

  static std::vector<velox_iceberg::IcebergCoalescedFile> makeCoalescedFiles(
      const std::vector<CudfPreparedIcebergFile>& files) {
    std::vector<velox_iceberg::IcebergCoalescedFile> coalesced;
    if (files.size() <= 1) {
      return coalesced;
    }
    coalesced.reserve(files.size() - 1);
    for (size_t i = 1; i < files.size(); ++i) {
      VELOX_CHECK_NOT_NULL(files[i].descriptor);
      coalesced.push_back({files[i].descriptor->filePath, files[i].fileSize});
    }
    return coalesced;
  }

  static uint64_t sumFileSizes(
      const std::vector<CudfPreparedIcebergFile>& files) {
    uint64_t total = 0;
    for (const auto& file : files) {
      const auto selectedBytes =
          file.preparedBytes == 0 ? file.fileSize : file.preparedBytes;
      VELOX_CHECK_LE(
          selectedBytes,
          std::numeric_limits<uint64_t>::max() - total,
          "Prepared batch byte size overflow");
      total += selectedBytes;
    }
    return total;
  }

  static size_t countDistinctSources(
      const std::vector<CudfPreparedIcebergFile>& files) {
    std::unordered_set<uint64_t> ordinals;
    ordinals.reserve(files.size());
    for (const auto& file : files) {
      ordinals.insert(file.descriptorOrdinal);
    }
    return ordinals.size();
  }

  std::vector<CudfPreparedIcebergFile> files_;
  const uint64_t selectedBytes_;
  const size_t sourceCount_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
