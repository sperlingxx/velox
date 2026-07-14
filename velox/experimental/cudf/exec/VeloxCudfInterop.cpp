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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

#include <cstdlib>
#include <cstring>

namespace facebook::velox::cudf_velox {

namespace with_arrow {

namespace {

void replaceArrowFormatForCudfImport(ArrowSchema* schema, const char* format) {
  // Velox's Arrow exporter owns the original format pointer and may not have
  // allocated it with malloc, so do not free it on this import-side rewrite.
  const size_t bufferLen = std::strlen(format) + 1;
  auto* buffer = static_cast<char*>(std::malloc(bufferLen));
  VELOX_CHECK_NOT_NULL(buffer);
  std::memcpy(buffer, format, bufferLen);
  schema->format = buffer;
}

void setArrowSchemaTypesForCudfImport(
    ArrowSchema* schema,
    const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ROW: {
      if (schema->n_children != static_cast<int64_t>(type->size())) {
        break;
      }
      for (size_t i = 0; i < type->size(); ++i) {
        setArrowSchemaTypesForCudfImport(schema->children[i], type->childAt(i));
      }
      break;
    }
    case TypeKind::ARRAY: {
      if (schema->n_children == 1 && schema->children[0] != nullptr) {
        setArrowSchemaTypesForCudfImport(schema->children[0], type->childAt(0));
      }
      break;
    }
    case TypeKind::MAP: {
      // Velox exports MAP using Arrow's logical map format. libcudf imports
      // the same physical layout as LIST<STRUCT<key,value>>, but rejects the
      // Arrow MAP format itself, so downgrade only this import-side schema.
      replaceArrowFormatForCudfImport(schema, "+l");
      VELOX_CHECK_EQ(schema->n_children, 1);
      VELOX_CHECK_NOT_NULL(schema->children[0]);
      auto* entries = schema->children[0];
      replaceArrowFormatForCudfImport(entries, "+s");
      VELOX_CHECK_EQ(entries->n_children, 2);
      VELOX_CHECK_NOT_NULL(entries->children[0]);
      VELOX_CHECK_NOT_NULL(entries->children[1]);
      setArrowSchemaTypesForCudfImport(entries->children[0], type->childAt(0));
      setArrowSchemaTypesForCudfImport(entries->children[1], type->childAt(1));
      break;
    }
    default:
      break;
  }
}

} // namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<std::string> timestampTimeZone) {
  TimestampUnit unit;
  switch (CudfConfig::getInstance().timestampUnit) {
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      unit = TimestampUnit::kNano;
      break;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      unit = TimestampUnit::kMicro;
      break;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      unit = TimestampUnit::kMilli;
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
      unit = TimestampUnit::kSecond;
      break;
    default:
      VELOX_UNSUPPORTED();
  }
  // Need to flattenDictionary and flattenConstant, otherwise we observe issues
  // in the null mask. Also, libcudf does not support Arrow binary, so we export
  // VARBINARY as UTF-8.
  ArrowOptions arrowOptions{
      .flattenDictionary = true,
      .flattenConstant = true,
      .timestampUnit = unit,
      .timestampTimeZone = timestampTimeZone,
      .exportVarbinaryAsString = true,
      .useDecimalTypeWidth = true};
  auto exportVector =
      std::static_pointer_cast<BaseVector>(std::make_shared<RowVector>(
          veloxTable->pool(),
          veloxTable->type(),
          veloxTable->nulls(),
          veloxTable->size(),
          veloxTable->children()));
  BaseVector::flattenVector(exportVector);
  ArrowArray arrowArray;
  exportToArrow(exportVector, arrowArray, pool, arrowOptions);
  ArrowSchema arrowSchema;
  exportToArrow(exportVector, arrowSchema, arrowOptions);
  setArrowSchemaTypesForCudfImport(&arrowSchema, exportVector->type());
  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream, mr);

  // Synchronize before releasing Arrow resources.  cudf::from_arrow uses
  // cudaMemcpyBatchAsync (CUDA 13.0+) with cudaMemcpySrcAccessOrderStream,
  // which defers reading the host source buffers until the stream reaches
  // each copy.  The Arrow arrays must therefore stay alive until the stream
  // has executed those copies.
  stream.synchronize();

  // Release Arrow resources
  if (arrowArray.release) {
    arrowArray.release(&arrowArray);
  }
  if (arrowSchema.release) {
    arrowSchema.release(&arrowSchema);
  }
  return tbl;
}

namespace {

void replaceArrowFormat(ArrowSchema* schema, const char* format) {
  if (schema->format != nullptr) {
    std::free(const_cast<char*>(schema->format));
    schema->format = nullptr;
  }
  const size_t bufferLen = std::strlen(format) + 1;
  auto* buffer = static_cast<char*>(std::malloc(bufferLen));
  VELOX_CHECK_NOT_NULL(buffer);
  std::memcpy(buffer, format, bufferLen);
  schema->format = buffer;
}

void setArrowSchemaTypesFromVelox(ArrowSchema* schema, const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ROW: {
      if (schema->n_children != static_cast<int64_t>(type->size())) {
        break;
      }
      for (size_t i = 0; i < type->size(); ++i) {
        setArrowSchemaTypesFromVelox(schema->children[i], type->childAt(i));
      }
      break;
    }
    case TypeKind::ARRAY: {
      if (schema->n_children == 1 && schema->children[0] != nullptr) {
        setArrowSchemaTypesFromVelox(schema->children[0], type->childAt(0));
      }
      break;
    }
    case TypeKind::MAP: {
      replaceArrowFormat(schema, "+m");
      VELOX_CHECK_EQ(schema->n_children, 1);
      VELOX_CHECK_NOT_NULL(schema->children[0]);
      auto* entries = schema->children[0];
      replaceArrowFormat(entries, "+s");
      VELOX_CHECK_EQ(entries->n_children, 2);
      VELOX_CHECK_NOT_NULL(entries->children[0]);
      VELOX_CHECK_NOT_NULL(entries->children[1]);
      setArrowSchemaTypesFromVelox(entries->children[0], type->childAt(0));
      setArrowSchemaTypesFromVelox(entries->children[1], type->childAt(1));
      break;
    }
    case TypeKind::VARBINARY: {
      // Replace any format string with "z" to indicate VARBINARY.
      replaceArrowFormat(schema, "z");
      break;
    }
    default:
      break;
  }
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    const RowTypePtr* outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // To avoid ownership issues, we make copies of the Arrow objects
  // returned from cuDF as unique_ptrs, then mark the originals as
  // released so their destructors don't try to free the resources.
  //
  // A better solution would be alternative versions of the cuDF
  // to_arrow_host functions that return Arrow objects by value
  // or populate objects passed by reference, but that would require
  // changes to cuDF.
  //
  // seves 1/17/26

  auto arrowDeviceArray = cudf::to_arrow_host(table, stream, mr);
  ArrowArray arrayCopy = arrowDeviceArray->array;
  arrowDeviceArray->array.release = nullptr;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  ArrowSchema schemaCopy = *arrowSchema;
  arrowSchema->release = nullptr;

  // Override schema type recursively with outputType if provided. This is
  // needed for some types like VARBINARY which are exported as STRING (the
  // format is overridden to "z" when the exportVarbinaryAsString option is set
  // to true in the exportToArrow() call) because cuDF does not have a VARBINARY
  // type. MAP also needs logical format restoration because cuDF exposes it as
  // a LIST<STRUCT<key,value>> physical layout.
  if (outputType) {
    setArrowSchemaTypesFromVelox(&schemaCopy, *outputType);
  }

  auto veloxTable = importFromArrowAsOwner(schemaCopy, arrayCopy, pool);

  // BaseVector to RowVector
  auto castedPtr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(castedPtr);
  return castedPtr;
}

template <typename Iterator>
std::vector<cudf::column_metadata>
getMetadata(Iterator begin, Iterator end, const std::string& namePrefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(namePrefix + std::to_string(i)));
    metadata.back().children_meta = getMetadata(
        c->child_begin(), c->child_end(), namePrefix + std::to_string(i));
    i++;
  }
  return metadata;
}

// Recursively generate metadata using exact names from Velox RowType.
cudf::column_metadata getMetadataWithName(
    const facebook::velox::TypePtr& type,
    const std::string& name) {
  cudf::column_metadata meta(name);
  if (type->kind() == facebook::velox::TypeKind::ROW) {
    auto rowType =
        std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
    for (size_t i = 0; i < rowType->size(); ++i) {
      meta.children_meta.push_back(
          getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
    }
  } else if (type->kind() == facebook::velox::TypeKind::ARRAY) {
    // cudf::lists_column_view::child_column_index is 1, the first metadata is
    // offsets
    meta.children_meta.emplace_back(cudf::column_metadata(name + "_offsets"));
    meta.children_meta.push_back(
        getMetadataWithName(type->childAt(0), "element"));
  } else if (type->kind() == facebook::velox::TypeKind::MAP) {
    meta.children_meta.emplace_back(cudf::column_metadata(name + "_offsets"));
    auto entries = cudf::column_metadata("entries");
    entries.children_meta.push_back(
        getMetadataWithName(type->childAt(0), "key"));
    entries.children_meta.push_back(
        getMetadataWithName(type->childAt(1), "value"));
    meta.children_meta.push_back(std::move(entries));
  }
  return meta;
}

std::vector<cudf::column_metadata> getMetadataWithName(
    const RowTypePtr& rowType) {
  std::vector<cudf::column_metadata> metadata;
  for (size_t i = 0; i < rowType->size(); ++i) {
    metadata.push_back(
        getMetadataWithName(rowType->childAt(i), rowType->nameOf(i)));
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, nullptr, stream, mr);
}

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    const facebook::velox::RowTypePtr& outputType,
    std::string namePrefix,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, &outputType, stream, mr);
}

// New overload: Accepts a Velox TypePtr for recursive metadata construction.
RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Recursively generate metadata using Velox type names for all columns.
  // This assumes 'type' is a RowType and its children match the cudf table
  // columns.
  auto rowType =
      std::dynamic_pointer_cast<const facebook::velox::RowType>(type);
  VELOX_CHECK_NOT_NULL(rowType);
  auto metadata = getMetadataWithName(rowType);
  return toVeloxColumn(table, pool, metadata, &rowType, stream, mr);
}

} // namespace with_arrow

} // namespace facebook::velox::cudf_velox
