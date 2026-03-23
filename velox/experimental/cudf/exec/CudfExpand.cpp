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

#include "velox/experimental/cudf/exec/CudfExpand.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

namespace {

// Dispatch helper: create a cudf scalar from a Velox ConstantTypedExpr.
// For null constants, produces an invalid (null) scalar of the correct
// cudf type. For non-null constants, dispatches on TypeKind to build
// a typed scalar.
template <TypeKind kind>
std::unique_ptr<cudf::scalar> createScalarDispatch(
    const std::shared_ptr<const core::ConstantTypedExpr>& constant,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  auto vec = constant->toConstantVector(pool);
  bool isNull = vec->isNullAt(0);
  auto type = constant->type();
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  if (isNull) {
    auto cudfType =
        cudf::data_type{veloxToCudfTypeId(type)};
    return cudf::make_default_constructed_scalar(
        cudfType, stream, mr);
  }

  auto constVec = vec->template as<SimpleVector<T>>();
  T value = constVec->valueAt(0);

  if constexpr (
      std::is_same_v<T, StringView> ||
      std::is_same_v<T, std::string_view>) {
    return std::make_unique<cudf::string_scalar>(
        std::string(value.data(), value.size()),
        true,
        stream,
        mr);
  } else if constexpr (cudf::is_fixed_width<T>()) {
    if (type->isDate()) {
      using CudfDateType = cudf::timestamp_D;
      return std::make_unique<
          cudf::timestamp_scalar<CudfDateType>>(
          value, true, stream, mr);
    }
    return std::make_unique<cudf::numeric_scalar<T>>(
        value, true, stream, mr);
  }
  VELOX_NYI(
      "CudfExpand: unsupported constant type {}",
      type->toString());
}

std::unique_ptr<cudf::scalar> createScalarFromConstant(
    const std::shared_ptr<const core::ConstantTypedExpr>&
        constant,
    memory::MemoryPool* pool) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createScalarDispatch,
      constant->type()->kind(),
      constant,
      pool);
}

} // namespace

CudfExpand::CudfExpand(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::ExpandNode>& expandNode)
    : Operator(
          driverCtx,
          expandNode->outputType(),
          operatorId,
          expandNode->id(),
          "CudfExpand"),
      NvtxHelper(
          nvtx3::rgb{186, 85, 211}, // Medium orchid
          operatorId,
          fmt::format("[{}]", expandNode->id())) {
  const auto& inputType = expandNode->inputType();
  const auto numRows = expandNode->projections().size();
  const auto numColumns = expandNode->names().size();

  fieldProjections_.reserve(numRows);
  constantScalars_.reserve(numRows);

  for (const auto& rowProjections : expandNode->projections()) {
    std::vector<column_index_t> rowFieldProj;
    std::vector<std::unique_ptr<cudf::scalar>> rowScalars;
    rowFieldProj.reserve(numColumns);
    rowScalars.reserve(numColumns);

    for (const auto& colProjection : rowProjections) {
      if (auto field =
              core::TypedExprs::asFieldAccess(colProjection)) {
        rowFieldProj.push_back(
            inputType->getChildIdx(field->name()));
        rowScalars.push_back(nullptr);
      } else if (
          auto constant =
              core::TypedExprs::asConstant(colProjection)) {
        rowFieldProj.push_back(kConstantChannel);
        rowScalars.push_back(
            createScalarFromConstant(constant, pool()));
      } else {
        VELOX_USER_FAIL(
            "CudfExpand: only column references and constants "
            "are supported. Got: {}",
            colProjection->toString());
      }
    }

    fieldProjections_.emplace_back(std::move(rowFieldProj));
    constantScalars_.emplace_back(std::move(rowScalars));
  }
}

bool CudfExpand::needsInput() const {
  return !noMoreInput_ && input_ == nullptr;
}

void CudfExpand::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

RowVectorPtr CudfExpand::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;

  if (!input_) {
    return nullptr;
  }

  auto cudfInput =
      std::dynamic_pointer_cast<cudf_velox::CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(
      cudfInput, "CudfExpand expects CudfVector input");

  const auto numInputRows = input_->size();
  auto inputView = cudfInput->getTableView();
  auto stream = cudfInput->stream();

  const auto& rowProjection = fieldProjections_[rowIndex_];
  const auto& rowScalars = constantScalars_[rowIndex_];
  const auto numColumns = rowProjection.size();

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(numColumns);

  for (size_t i = 0; i < numColumns; ++i) {
    if (rowProjection[i] == kConstantChannel) {
      outputColumns.push_back(cudf::make_column_from_scalar(
          *rowScalars[i], numInputRows, stream));
    } else {
      outputColumns.push_back(std::make_unique<cudf::column>(
          inputView.column(rowProjection[i]), stream));
    }
  }

  auto outputTable = std::make_unique<cudf::table>(
      std::move(outputColumns));

  ++rowIndex_;
  if (rowIndex_ ==
      static_cast<int32_t>(fieldProjections_.size())) {
    rowIndex_ = 0;
    input_ = nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      numInputRows,
      std::move(outputTable),
      stream);
}

} // namespace facebook::velox::cudf_velox
