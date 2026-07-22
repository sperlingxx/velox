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

cudf::data_type veloxToCudfDataType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::data_type{cudf::type_id::BOOL8};
    case TypeKind::TINYINT:
      return cudf::data_type{cudf::type_id::INT8};
    case TypeKind::SMALLINT:
      return cudf::data_type{cudf::type_id::INT16};
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
      }
      return cudf::data_type{cudf::type_id::INT32};
    case TypeKind::BIGINT:
      // BIGINT is used for both INT64 and DECIMAL64
      if (type->isDecimal()) {
        auto const decimalType =
            std::dynamic_pointer_cast<const ShortDecimalType>(type);
        VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
        auto const cudfScale = numeric::scale_type{-decimalType->scale()};
        return cudf::data_type{cudf::type_id::DECIMAL64, cudfScale};
      }
      return cudf::data_type{cudf::type_id::INT64};
    case TypeKind::HUGEINT: {
      // HUGEINT is used only for DECIMAL128
      // per facebookincubator/velox PR 4434 (May 2, 2023)
      // although see commented-out HUGEINT -> DURATION_DAYS below
      VELOX_CHECK(
          type->isDecimal(), "HUGEINT should only be used for DECIMAL128");
      auto const decimalType =
          std::dynamic_pointer_cast<const LongDecimalType>(type);
      VELOX_CHECK(decimalType, "Invalid Decimal Type (failed dynamic_cast)");
      auto const cudfScale = numeric::scale_type{-decimalType->scale()};
      return cudf::data_type{cudf::type_id::DECIMAL128, cudfScale};
    }
    case TypeKind::REAL:
      return cudf::data_type{cudf::type_id::FLOAT32};
    case TypeKind::DOUBLE:
      return cudf::data_type{cudf::type_id::FLOAT64};
    case TypeKind::VARCHAR:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::VARBINARY:
      return cudf::data_type{cudf::type_id::STRING};
    case TypeKind::TIMESTAMP:
      return cudf::data_type{CudfConfig::getInstance().timestampUnit};
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    case TypeKind::ARRAY:
      return cudf::data_type{cudf::type_id::LIST};
    case TypeKind::MAP:
      return cudf::data_type{cudf::type_id::LIST};
    case TypeKind::ROW:
      return cudf::data_type{cudf::type_id::STRUCT};
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      break;
  }
  CUDF_FAIL(
      "Unsupported Velox type: " +
      std::string(TypeKindName::toName(type->kind())));
  return cudf::data_type{cudf::type_id::EMPTY};
}

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

bool containsUnknownType(const TypePtr& type) {
  if (type->kind() == TypeKind::UNKNOWN) {
    return true;
  }
  for (size_t i = 0; i < type->size(); ++i) {
    if (containsUnknownType(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

TypePtr physicalTypeForCudfImport(const TypePtr& type) {
  if (!containsUnknownType(type)) {
    return type;
  }
  switch (type->kind()) {
    case TypeKind::UNKNOWN:
      return TINYINT();
    case TypeKind::ROW: {
      const auto rowType = asRowType(type);
      std::vector<TypePtr> children;
      children.reserve(rowType->size());
      for (size_t i = 0; i < rowType->size(); ++i) {
        children.push_back(physicalTypeForCudfImport(rowType->childAt(i)));
      }
      return ROW(rowType->names(), std::move(children));
    }
    case TypeKind::ARRAY:
      return ARRAY(physicalTypeForCudfImport(type->childAt(0)));
    case TypeKind::MAP:
      return MAP(
          physicalTypeForCudfImport(type->childAt(0)),
          physicalTypeForCudfImport(type->childAt(1)));
    default:
      VELOX_UNREACHABLE(
          "Type {} reported an UNKNOWN child but is not complex", type);
  }
}

// Arrow's null type imports into cuDF as an EMPTY column. EMPTY is useful as
// an Arrow placeholder, but cuDF operators such as gather and hash join cannot
// dispatch it. Convert logical Velox UNKNOWN vectors to all-null INT8 transport
// vectors before Arrow export. The reverse conversion in restoreLogicalTypes()
// restores the requested UNKNOWN schema when data returns to Velox.
VectorPtr prepareLogicalTypesForCudfImport(
    const VectorPtr& vector,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(vector);
  const auto& type = vector->type();
  if (!containsUnknownType(type)) {
    return vector;
  }

  if (type->kind() == TypeKind::UNKNOWN) {
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      VELOX_CHECK(
          vector->isNullAt(i),
          "Cannot encode UNKNOWN non-null value for cuDF at row {}",
          i);
    }
    return BaseVector::createNullConstant(TINYINT(), vector->size(), pool);
  }

  switch (type->kind()) {
    case TypeKind::ROW: {
      const auto row = std::dynamic_pointer_cast<RowVector>(vector);
      VELOX_CHECK_NOT_NULL(row);
      std::vector<VectorPtr> children;
      children.reserve(row->childrenSize());
      for (size_t i = 0; i < row->childrenSize(); ++i) {
        children.push_back(
            prepareLogicalTypesForCudfImport(row->childAt(i), pool));
      }
      return std::make_shared<RowVector>(
          pool,
          physicalTypeForCudfImport(type),
          row->nulls(),
          row->size(),
          std::move(children),
          row->getNullCount());
    }
    case TypeKind::ARRAY: {
      const auto array = std::dynamic_pointer_cast<ArrayVector>(vector);
      VELOX_CHECK_NOT_NULL(array);
      return std::make_shared<ArrayVector>(
          pool,
          physicalTypeForCudfImport(type),
          array->nulls(),
          array->size(),
          array->offsets(),
          array->sizes(),
          prepareLogicalTypesForCudfImport(array->elements(), pool),
          array->getNullCount());
    }
    case TypeKind::MAP: {
      const auto map = std::dynamic_pointer_cast<MapVector>(vector);
      VELOX_CHECK_NOT_NULL(map);
      return std::make_shared<MapVector>(
          pool,
          physicalTypeForCudfImport(type),
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          prepareLogicalTypesForCudfImport(map->mapKeys(), pool),
          prepareLogicalTypesForCudfImport(map->mapValues(), pool),
          map->getNullCount(),
          map->hasSortedKeys());
    }
    default:
      VELOX_UNREACHABLE(
          "Type {} reported an UNKNOWN child but is not complex", type);
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
  auto exportVector = std::static_pointer_cast<BaseVector>(
      prepareLogicalTypesForCudfImport(veloxTable, pool));
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

// cuDF has no physical representation for Velox's UNKNOWN type. Producers use
// an all-null INT8 column as the transport representation. Arrow therefore
// imports that column as TINYINT, which makes RowVector::setType fail before it
// can restore the requested logical schema. Rebuild the complex-vector path and
// replace these placeholders with real UNKNOWN null constants first.
VectorPtr restoreLogicalTypes(
    const VectorPtr& vector,
    const TypePtr& type,
    memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(vector);
  VELOX_CHECK_NOT_NULL(type);

  if (type->kind() == TypeKind::UNKNOWN) {
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      VELOX_CHECK(
          vector->isNullAt(i),
          "Cannot restore UNKNOWN from a non-null physical value at row {}",
          i);
    }
    return BaseVector::createNullConstant(type, vector->size(), pool);
  }

  switch (type->kind()) {
    case TypeKind::ROW: {
      auto row = std::dynamic_pointer_cast<RowVector>(vector);
      VELOX_CHECK_NOT_NULL(
          row,
          "Expected ROW physical vector for logical type {}, got {}",
          type,
          vector->type());
      VELOX_CHECK_EQ(row->childrenSize(), type->size());
      std::vector<VectorPtr> children;
      children.reserve(type->size());
      for (size_t i = 0; i < type->size(); ++i) {
        children.push_back(
            restoreLogicalTypes(row->childAt(i), type->childAt(i), pool));
      }
      return std::make_shared<RowVector>(
          pool,
          type,
          row->nulls(),
          row->size(),
          std::move(children),
          row->getNullCount());
    }
    case TypeKind::ARRAY: {
      auto array = std::dynamic_pointer_cast<ArrayVector>(vector);
      VELOX_CHECK_NOT_NULL(
          array,
          "Expected ARRAY physical vector for logical type {}, got {}",
          type,
          vector->type());
      return std::make_shared<ArrayVector>(
          pool,
          type,
          array->nulls(),
          array->size(),
          array->offsets(),
          array->sizes(),
          restoreLogicalTypes(array->elements(), type->childAt(0), pool),
          array->getNullCount());
    }
    case TypeKind::MAP: {
      auto map = std::dynamic_pointer_cast<MapVector>(vector);
      VELOX_CHECK_NOT_NULL(
          map,
          "Expected MAP physical vector for logical type {}, got {}",
          type,
          vector->type());
      return std::make_shared<MapVector>(
          pool,
          type,
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          restoreLogicalTypes(map->mapKeys(), type->childAt(0), pool),
          restoreLogicalTypes(map->mapValues(), type->childAt(1), pool),
          map->getNullCount(),
          map->hasSortedKeys());
    }
    default:
      // An untyped NULL can acquire a concrete logical type after Spark/Velox
      // type coercion without adding a physical cuDF cast.  Its transport
      // column is still all-null INT8.  Rebuild such columns with the requested
      // logical type; never reinterpret a non-null physical value.
      if (!vector->type()->kindEquals(type)) {
        for (vector_size_t i = 0; i < vector->size(); ++i) {
          VELOX_CHECK(
              vector->isNullAt(i),
              "Cannot restore logical type {} from physical type {} with a "
              "non-null value at row {}",
              type,
              vector->type(),
              i);
        }
        return BaseVector::createNullConstant(type, vector->size(), pool);
      }
      VELOX_CHECK(
          vector->type()->kindEquals(type),
          "Cannot restore logical type {} from physical type {}",
          type,
          vector->type());
      vector->setType(type);
      return vector;
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
  if (outputType) {
    veloxTable = restoreLogicalTypes(veloxTable, *outputType, pool);
  }

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
