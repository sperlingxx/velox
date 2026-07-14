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
#include "velox/experimental/cudf/CudfTypeUtils.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/error.hpp>

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

} // namespace facebook::velox::cudf_velox
