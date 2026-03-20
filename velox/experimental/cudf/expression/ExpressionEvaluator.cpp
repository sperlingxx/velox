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
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/reduction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/search.hpp>
#include <cudf/concatenate.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

bool decimalScalarIsZero(
    const cudf::scalar& scalar,
    rmm::cuda_stream_view stream) {
  if (!scalar.is_valid(stream)) {
    return false;
  }
  if (scalar.type().id() == cudf::type_id::DECIMAL64) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(
            scalar);
    return dec.value(stream) == 0;
  }
  if (scalar.type().id() == cudf::type_id::DECIMAL128) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(
            scalar);
    return dec.value(stream) == 0;
  }
  return false;
}

bool hasDecimalZero(
    const cudf::column_view& col,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (col.is_empty()) {
    return false;
  }
  std::unique_ptr<cudf::scalar> zero;
  auto scale = numeric::scale_type{col.type().scale()};
  if (col.type().id() == cudf::type_id::DECIMAL64) {
    zero =
        cudf::make_fixed_point_scalar<numeric::decimal64>(0, scale, stream, mr);
  } else if (col.type().id() == cudf::type_id::DECIMAL128) {
    zero = cudf::make_fixed_point_scalar<numeric::decimal128>(
        0, scale, stream, mr);
  } else {
    return false;
  }

  auto equals = cudf::binary_operation(
      col,
      *zero,
      cudf::binary_operator::EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto anyAgg = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto anyScalar = cudf::reduce(
      equals->view(),
      *anyAgg,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto const& boolScalar =
      static_cast<cudf::numeric_scalar<bool> const&>(*anyScalar);
  return boolScalar.is_valid(stream) && boolScalar.value(stream);
}

struct CudfExpressionEvaluatorEntry {
  int priority;
  CudfExpressionEvaluatorCanEvaluate canEvaluate;
  CudfExpressionEvaluatorCreate create;
};

static std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getCudfExpressionEvaluatorRegistry() {
  static std::unordered_map<std::string, CudfExpressionEvaluatorEntry> registry;
  return registry;
}

static void ensureBuiltinExpressionEvaluatorsRegistered() {
  static bool registered = false;
  if (registered) {
    return;
  }

  // Default priority for function evaluator
  const int kFunctionPriority = 50;

  // Function evaluator
  registerCudfExpressionEvaluator(
      "function",
      kFunctionPriority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return FunctionExpression::canEvaluate(std::move(expr));
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return FunctionExpression::create(std::move(expr), row);
      },
      /*overwrite=*/false);

  registered = true;
}

} // namespace

bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite) {
  auto& registry = getCudfExpressionEvaluatorRegistry();
  if (!overwrite && registry.find(name) != registry.end()) {
    return false;
  }
  registry[name] = CudfExpressionEvaluatorEntry{
      priority, std::move(canEvaluate), std::move(create)};
  return true;
}

std::unordered_map<std::string, CudfFunctionSpec>& getCudfFunctionRegistry() {
  static std::unordered_map<std::string, CudfFunctionSpec> registry;
  return registry;
}

namespace {

static bool matchCallAgainstSignatures(
    const velox::exec::Expr& call,
    const std::vector<exec::FunctionSignaturePtr>& sigs) {
  const auto n = call.inputs().size();
  std::vector<TypePtr> argTypes;
  argTypes.reserve(n);
  for (const auto& in : call.inputs()) {
    argTypes.push_back(in->type());
  }
  for (const auto& sig : sigs) {
    exec::SignatureBinder binder(*sig, argTypes);
    if (!binder.tryBind()) {
      continue;
    }
    // binder does not confirm whether positional arguments are
    // constants(scalars) as expected. we have to check manually
    const auto& constArgs = sig->constantArguments();
    const size_t fixed = std::min(constArgs.size(), n);
    bool ok = true;
    for (size_t i = 0; i < fixed; ++i) {
      if (constArgs[i] && call.inputs()[i]->name() != "literal") {
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }
    return true;
  }
  return false;
}

} // namespace

class SplitFunction : public CudfFunction {
 public:
  SplitFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto delimiterExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(delimiterExpr, "split delimiter must be a constant");
    delimiterScalar_ = std::make_unique<cudf::string_scalar>(
        delimiterExpr->value()->toString(0), true, stream, mr);

    auto limitExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(limitExpr, "split limit must be a constant");
    maxSplitCount_ = std::stoll(limitExpr->value()->toString(0));

    // Presto specifies maxSplitCount as the maximum size of the returned array
    // while cuDF understands the parameter as how many splits can it perform.
    maxSplitCount_ -= 1;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::split_record(
        inputCol, *delimiterScalar_, maxSplitCount_, stream, mr);
  };

 private:
  std::unique_ptr<cudf::string_scalar> delimiterScalar_;
  cudf::size_type maxSplitCount_;
};

class CastFunction : public CudfFunction {
 public:
  CastFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "cast expects exactly 1 input");

    targetCudfType_ = cudf_velox::veloxToCudfDataType(expr->type());
    auto sourceType =
        cudf_velox::veloxToCudfDataType(expr->inputs()[0]->type());

    if (sourceType.id() == cudf::type_id::TIMESTAMP_DAYS &&
        targetCudfType_.id() == cudf::type_id::STRING) {
      dateToString_ = true;
    } else {
      VELOX_CHECK(
          cudf::is_supported_cast(sourceType, targetCudfType_),
          "Cast from {} to {} is not supported",
          expr->inputs()[0]->type()->toString(),
          expr->type()->toString());
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    if (dateToString_) {
      return cudf::strings::from_timestamps(inputCol, "%Y-%m-%d",
          cudf::strings_column_view(cudf::column_view{
              cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0}),
          stream, mr);
    }
    return cudf::cast(inputCol, targetCudfType_, stream, mr);
  }

 private:
  cudf::data_type targetCudfType_;
  bool dateToString_{false};
};

// Spark date_add function implementation.
// For the presto date_add, the first value is unit string,
// may need to get the function with prefix, if the prefix is "", it is Spark
// function.
class DateAddFunction : public CudfFunction {
 public:
  DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->isDate(),
        "First argument to date_add must be a date");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0]));
    // The date_add second argument could be int8_t, int16_t, int32_t.
    value_ = makeScalarFromConstantExpr(
        expr->inputs()[1], cudf::type_id::DURATION_DAYS);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::binary_operation(
        inputCol,
        *value_,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

 private:
  std::unique_ptr<cudf::scalar> value_;
};

class CardinalityFunction : public CudfFunction {
 public:
  CardinalityFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    // Cardinality doesn't need any pre-computed scalars, just validates input
    // count
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "cardinality expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::lists::count_elements(inputCol, stream, mr);
  }
};

class RoundFunction : public CudfFunction {
 public:
  explicit RoundFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    const auto argSize = expr->inputs().size();
    VELOX_CHECK(argSize >= 1 && argSize <= 2, "round expects 1 or 2 inputs");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<exec::ConstantExpr>(expr->inputs()[0]),
        "round expects first column is not literal");
    if (argSize == 2) {
      auto scaleExpr =
          std::dynamic_pointer_cast<exec::ConstantExpr>(expr->inputs()[1]);
      VELOX_CHECK_NOT_NULL(scaleExpr, "round scale must be a constant");
      scale_ = scaleExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputView = asView(inputColumns[0]);
    if (cudf::is_floating_point(inputView.type())) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      return cudf::round(
          inputView, scale_,
          cudf::rounding_method::HALF_UP, stream, mr);
#pragma GCC diagnostic pop
    }
    return cudf::round_decimal(
        inputView, scale_,
        cudf::rounding_method::HALF_UP, stream, mr);
  }

 private:
  int32_t scale_ = 0;
};

class BinaryFunction : public CudfFunction {
 public:
  BinaryFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::binary_operator op)
      : op_(op), type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "Binary function expects exactly 2 inputs");
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0])) {
      auto constValue = constExpr->value();
      left_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    } else if (
        auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[1])) {
      auto constValue = constExpr->value();
      right_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }

    VELOX_CHECK(
        !(left_ != nullptr && right_ != nullptr),
        "Binary function on two literals is not supported");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto isComparisonOp = [](cudf::binary_operator op) {
      switch (op) {
        case cudf::binary_operator::EQUAL:
        case cudf::binary_operator::NOT_EQUAL:
        case cudf::binary_operator::GREATER:
        case cudf::binary_operator::GREATER_EQUAL:
        case cudf::binary_operator::LESS:
        case cudf::binary_operator::LESS_EQUAL:
          return true;
        default:
          return false;
      }
    };
    if (left_ == nullptr && right_ == nullptr) {
      if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
        auto lhsView = asView(inputColumns[0]);
        auto rhsView = asView(inputColumns[1]);
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (type_.id() == cudf::type_id::DECIMAL128) {
          if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, lhsView.type().scale()};
            lhsCast = cudf::cast(lhsView, castType, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, rhsView.type().scale()};
            rhsCast = cudf::cast(rhsView, castType, stream, mr);
            rhsView = rhsCast->view();
          }
        }
        if (hasDecimalZero(rhsView, stream, mr)) {
          VELOX_USER_FAIL("Division by zero");
        }
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto outScale = -type_.scale();
        auto aRescale = outScale - lhsScale + rhsScale;
        return decimalDivide(lhsView, rhsView, type_, aRescale, stream);
      }
      auto lhsView = asView(inputColumns[0]);
      auto rhsView = asView(inputColumns[1]);
      if (isComparisonOp(op_) && cudf::is_fixed_point(lhsView.type()) &&
          cudf::is_fixed_point(rhsView.type())) {
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
        auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                             rhsView.type().id() == cudf::type_id::DECIMAL128)
            ? cudf::type_id::DECIMAL128
            : cudf::type_id::DECIMAL64;
        auto targetType =
            cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (lhsView.type() != targetType) {
          lhsCast = cudf::cast(lhsView, targetType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type() != targetType) {
          rhsCast = cudf::cast(rhsView, targetType, stream, mr);
          rhsView = rhsCast->view();
        }
        return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
      }
      if (cudf::is_fixed_point(type_)) {
        if (op_ == cudf::binary_operator::ADD ||
            op_ == cudf::binary_operator::SUB ||
            op_ == cudf::binary_operator::MOD) {
          std::unique_ptr<cudf::column> lhsCast;
          std::unique_ptr<cudf::column> rhsCast;
          if (lhsView.type() != type_) {
            lhsCast = cudf::cast(lhsView, type_, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type() != type_) {
            rhsCast = cudf::cast(rhsView, type_, stream, mr);
            rhsView = rhsCast->view();
          }
          return cudf::binary_operation(
              lhsView, rhsView, op_, type_, stream, mr);
        }
        if (op_ == cudf::binary_operator::MUL) {
          std::unique_ptr<cudf::column> lhsCast;
          std::unique_ptr<cudf::column> rhsCast;
          if (type_.id() == cudf::type_id::DECIMAL128) {
            if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, lhsView.type().scale()};
              lhsCast = cudf::cast(lhsView, castType, stream, mr);
              lhsView = lhsCast->view();
            }
            if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, rhsView.type().scale()};
              rhsCast = cudf::cast(rhsView, castType, stream, mr);
              rhsView = rhsCast->view();
            }
          }
          return cudf::binary_operation(
              lhsView, rhsView, op_, type_, stream, mr);
        }
      }
      return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
    } else if (left_ == nullptr) {
      if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
        if (decimalScalarIsZero(*right_, stream)) {
          VELOX_USER_FAIL("Division by zero");
        }
        auto lhsView = asView(inputColumns[0]);
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -right_->type().scale();
        auto outScale = -type_.scale();
        auto aRescale = outScale - lhsScale + rhsScale;
        auto rhsCol =
            cudf::make_column_from_scalar(*right_, lhsView.size(), stream, mr);
        auto rhsView = rhsCol->view();
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (type_.id() == cudf::type_id::DECIMAL128) {
          if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, lhsView.type().scale()};
            lhsCast = cudf::cast(lhsView, castType, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, rhsView.type().scale()};
            rhsCast = cudf::cast(rhsView, castType, stream, mr);
            rhsView = rhsCast->view();
          }
        }
        return decimalDivide(lhsView, rhsView, type_, aRescale, stream);
      }
      auto lhsView = asView(inputColumns[0]);
      if (isComparisonOp(op_) && cudf::is_fixed_point(lhsView.type()) &&
          cudf::is_fixed_point(right_->type())) {
        auto rhsCol =
            cudf::make_column_from_scalar(*right_, lhsView.size(), stream, mr);
        auto rhsView = rhsCol->view();
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
        auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                             rhsView.type().id() == cudf::type_id::DECIMAL128)
            ? cudf::type_id::DECIMAL128
            : cudf::type_id::DECIMAL64;
        auto targetType =
            cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (lhsView.type() != targetType) {
          lhsCast = cudf::cast(lhsView, targetType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type() != targetType) {
          rhsCast = cudf::cast(rhsView, targetType, stream, mr);
          rhsView = rhsCast->view();
        }
        return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
      }
      if (cudf::is_fixed_point(type_)) {
        auto rhsCol =
            cudf::make_column_from_scalar(*right_, lhsView.size(), stream, mr);
        auto rhsView = rhsCol->view();
        if (op_ == cudf::binary_operator::ADD ||
            op_ == cudf::binary_operator::SUB ||
            op_ == cudf::binary_operator::MOD) {
          std::unique_ptr<cudf::column> lhsCast;
          std::unique_ptr<cudf::column> rhsCast;
          if (lhsView.type() != type_) {
            lhsCast = cudf::cast(lhsView, type_, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type() != type_) {
            rhsCast = cudf::cast(rhsView, type_, stream, mr);
            rhsView = rhsCast->view();
          }
          return cudf::binary_operation(
              lhsView, rhsView, op_, type_, stream, mr);
        }
        if (op_ == cudf::binary_operator::MUL) {
          std::unique_ptr<cudf::column> lhsCast;
          std::unique_ptr<cudf::column> rhsCast;
          if (type_.id() == cudf::type_id::DECIMAL128) {
            if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, lhsView.type().scale()};
              lhsCast = cudf::cast(lhsView, castType, stream, mr);
              lhsView = lhsCast->view();
            }
            if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, rhsView.type().scale()};
              rhsCast = cudf::cast(rhsView, castType, stream, mr);
              rhsView = rhsCast->view();
            }
          }
          return cudf::binary_operation(
              lhsView, rhsView, op_, type_, stream, mr);
        }
      }
      return cudf::binary_operation(
          asView(inputColumns[0]), *right_, op_, type_, stream, mr);
    }
    if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
      auto rhsView = asView(inputColumns[0]);
      if (hasDecimalZero(rhsView, stream, mr)) {
        VELOX_USER_FAIL("Division by zero");
      }
      auto lhsScale = -left_->type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto outScale = -type_.scale();
      auto aRescale = outScale - lhsScale + rhsScale;
      auto lhsCol =
          cudf::make_column_from_scalar(*left_, rhsView.size(), stream, mr);
      auto lhsView = lhsCol->view();
      std::unique_ptr<cudf::column> lhsCast;
      std::unique_ptr<cudf::column> rhsCast;
      if (type_.id() == cudf::type_id::DECIMAL128) {
        if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
          auto castType = cudf::data_type{
              cudf::type_id::DECIMAL128, lhsView.type().scale()};
          lhsCast = cudf::cast(lhsView, castType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
          auto castType = cudf::data_type{
              cudf::type_id::DECIMAL128, rhsView.type().scale()};
          rhsCast = cudf::cast(rhsView, castType, stream, mr);
          rhsView = rhsCast->view();
        }
      }
      return decimalDivide(lhsView, rhsView, type_, aRescale, stream);
    }
    auto rhsView = asView(inputColumns[0]);
    if (isComparisonOp(op_) && cudf::is_fixed_point(left_->type()) &&
        cudf::is_fixed_point(rhsView.type())) {
      auto lhsCol =
          cudf::make_column_from_scalar(*left_, rhsView.size(), stream, mr);
      auto lhsView = lhsCol->view();
      auto lhsScale = -lhsView.type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
      auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                           rhsView.type().id() == cudf::type_id::DECIMAL128)
          ? cudf::type_id::DECIMAL128
          : cudf::type_id::DECIMAL64;
      auto targetType =
          cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
      std::unique_ptr<cudf::column> lhsCast;
      std::unique_ptr<cudf::column> rhsCast;
      if (lhsView.type() != targetType) {
        lhsCast = cudf::cast(lhsView, targetType, stream, mr);
        lhsView = lhsCast->view();
      }
      if (rhsView.type() != targetType) {
        rhsCast = cudf::cast(rhsView, targetType, stream, mr);
        rhsView = rhsCast->view();
      }
      return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
    }
    if (cudf::is_fixed_point(type_)) {
      auto lhsCol =
          cudf::make_column_from_scalar(*left_, rhsView.size(), stream, mr);
      auto lhsView = lhsCol->view();
      if (op_ == cudf::binary_operator::ADD ||
          op_ == cudf::binary_operator::SUB ||
          op_ == cudf::binary_operator::MOD) {
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (lhsView.type() != type_) {
          lhsCast = cudf::cast(lhsView, type_, stream, mr);
          lhsView = lhsCast->view();
        }
        if (rhsView.type() != type_) {
          rhsCast = cudf::cast(rhsView, type_, stream, mr);
          rhsView = rhsCast->view();
        }
        return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
      }
      if (op_ == cudf::binary_operator::MUL) {
        std::unique_ptr<cudf::column> lhsCast;
        std::unique_ptr<cudf::column> rhsCast;
        if (type_.id() == cudf::type_id::DECIMAL128) {
          if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, lhsView.type().scale()};
            lhsCast = cudf::cast(lhsView, castType, stream, mr);
            lhsView = lhsCast->view();
          }
          if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, rhsView.type().scale()};
            rhsCast = cudf::cast(rhsView, castType, stream, mr);
            rhsView = rhsCast->view();
          }
        }
        return cudf::binary_operation(lhsView, rhsView, op_, type_, stream, mr);
      }
    }
    return cudf::binary_operation(*left_, rhsView, op_, type_, stream, mr);
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

class UnaryFunction : public CudfFunction {
 public:
  UnaryFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::unary_operator op)
      : op_(op) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "Unary function expects exactly 1 input");
    auto constExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NULL(
        constExpr, "Unary function on literal input is not supported");
    // @TODO (seves 1/28/26)
    // binary functions require at least ONE input to be non-literal
    // do we need to support unary functions with ONLY a literal input?
    // assuming not for now
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::unary_operation(asView(inputColumns[0]), op_, stream, mr);
  }

 private:
  const cudf::unary_operator op_;
};

class LogicalFunction : public CudfFunction {
 public:
  LogicalFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::binary_operator op)
      : op_(op) {
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "Logical function expects at least 2 inputs");
    literals_.reserve(expr->inputs().size());
    for (const auto& input : expr->inputs()) {
      auto constExpr =
          std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input);
      if (constExpr) {
        literals_.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createCudfScalar,
            constExpr->value()->typeKind(),
            constExpr->value()));
      } else {
        literals_.push_back(nullptr);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    size_t rowCount = 0;
    if (!inputColumns.empty()) {
      rowCount = asView(inputColumns[0]).size();
    }
    if (rowCount == 0 && inputColumns.empty()) {
      rowCount = 1;
    }

    std::vector<std::unique_ptr<cudf::column>> literalColumns;
    literalColumns.reserve(literals_.size());
    std::vector<cudf::column_view> operands;
    operands.reserve(literals_.size());

    size_t columnIndex = 0;
    for (const auto& literal : literals_) {
      if (literal) {
        auto column = cudf::make_column_from_scalar(*literal, rowCount, stream);
        operands.push_back(column->view());
        literalColumns.push_back(std::move(column));
      } else {
        VELOX_CHECK_LT(columnIndex, inputColumns.size());
        operands.push_back(asView(inputColumns[columnIndex++]));
      }
    }

    VELOX_CHECK(!operands.empty());
    if (operands.size() == 1) {
      if (!literalColumns.empty()) {
        return std::move(literalColumns[0]);
      }
      return operands[0];
    }

    auto result = cudf::binary_operation(
        operands[0], operands[1], op_, kBoolType, stream, mr);
    for (size_t i = 2; i < operands.size(); ++i) {
      result = cudf::binary_operation(
          result->view(), operands[i], op_, kBoolType, stream, mr);
    }
    return result;
  }

 private:
  static constexpr cudf::data_type kBoolType{cudf::type_id::BOOL8};
  const cudf::binary_operator op_;
  std::vector<std::unique_ptr<cudf::scalar>> literals_;
};

class BetweenFunction : public CudfFunction {
 public:
  BetweenFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    // must have exactly three inputs: value, min, max
    VELOX_CHECK_EQ(
        expr->inputs().size(), 3, "Between function expects exactly 3 inputs");
    // value must not be a literal
    auto constExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NULL(
        constExpr, "Between function with literal input is not supported");
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[1])) {
      // min is a literal
      auto constValue = constExpr->value();
      minLiteral_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[2])) {
      // max is a literal
      auto constValue = constExpr->value();
      maxLiteral_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // return (value >= min) && (value <= max)
    std::unique_ptr<cudf::column> geResultColumn, leResultColumn;
    if (minLiteral_) {
      geResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          *minLiteral_,
          cudf::binary_operator::GREATER_EQUAL,
          kBoolType,
          stream,
          mr);
    } else {
      geResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          asView(inputColumns[1]),
          cudf::binary_operator::GREATER_EQUAL,
          kBoolType,
          stream,
          mr);
    }
    if (maxLiteral_) {
      leResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          *maxLiteral_,
          cudf::binary_operator::LESS_EQUAL,
          kBoolType,
          stream,
          mr);
    } else {
      leResultColumn = cudf::binary_operation(
          asView(inputColumns[0]),
          asView(inputColumns[2]),
          cudf::binary_operator::LESS_EQUAL,
          kBoolType,
          stream,
          mr);
    }
    return cudf::binary_operation(
        geResultColumn->view(),
        leResultColumn->view(),
        cudf::binary_operator::LOGICAL_AND,
        kBoolType,
        stream,
        mr);
  }

 private:
  static constexpr cudf::data_type kBoolType{cudf::type_id::BOOL8};
  std::unique_ptr<cudf::scalar> minLiteral_;
  std::unique_ptr<cudf::scalar> maxLiteral_;
};

class InFunction : public CudfFunction {
 public:
  InFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      bool skipCast)
      : skipCast_(skipCast) {
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "in expects exactly 2 inputs");
    auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
        expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        constExpr, "in second argument must be a constant array");
    auto value = constExpr->value();
    VELOX_CHECK_NOT_NULL(value, "ConstantExpr value is null");

    cudf::data_type targetType{cudf::type_id::EMPTY};
    if (skipCast_) {
      auto srcType = expr->inputs()[0]->inputs()[0]->type();
      targetType = cudf_velox::veloxToCudfDataType(srcType);
    }
    buildHaystackColumn(value, targetType);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty(), "in requires at least one input column");
    auto needles = asView(inputColumns[0]);
    auto haystackView = haystack_->view();
    if (haystackView.type() != needles.type()) {
      if (cudf::is_supported_cast(haystackView.type(), needles.type())) {
        auto castedHaystack =
            cudf::cast(haystackView, needles.type(), stream, mr);
        return cudf::contains(castedHaystack->view(), needles, stream, mr);
      }
      if (cudf::is_supported_cast(needles.type(), haystackView.type())) {
        auto castedNeedles =
            cudf::cast(needles, haystackView.type(), stream, mr);
        return cudf::contains(haystackView, castedNeedles->view(), stream, mr);
      }
      VELOX_FAIL(
          "IN type mismatch: haystack={} needles={}",
          static_cast<int>(haystackView.type().id()),
          static_cast<int>(needles.type().id()));
    }
    return cudf::contains(haystackView, needles, stream, mr);
  }

  static bool shouldSkipCast(const std::shared_ptr<velox::exec::Expr>& expr) {
    auto& valueExpr = expr->inputs()[0];
    if (valueExpr->name() != "cast" && valueExpr->name() != "try_cast") {
      return false;
    }
    if (valueExpr->inputs().empty()) {
      return false;
    }
    auto srcCudf =
        cudf_velox::veloxToCudfDataType(valueExpr->inputs()[0]->type());
    auto dstCudf = cudf_velox::veloxToCudfDataType(valueExpr->type());
    // CastFunction handles DATE→STRING via cudf::strings::from_timestamps,
    // so don't skip the cast even though cudf::cast doesn't support it.
    if (srcCudf.id() == cudf::type_id::TIMESTAMP_DAYS &&
        dstCudf.id() == cudf::type_id::STRING) {
      return false;
    }
    return !cudf::is_supported_cast(srcCudf, dstCudf);
  }

 private:
  template <TypeKind Kind>
  static std::unique_ptr<cudf::scalar> scalarFromElement(
      const VectorPtr& vec,
      const TypePtr& type,
      vector_size_t idx) {
    using T = typename TypeTraits<Kind>::NativeType;
    auto val = vec->template as<SimpleVector<T>>()->valueAt(idx);
    return makeScalarFromValue<T>(type, val, false);
  }

  void buildHaystackColumn(
      const VectorPtr& vector,
      cudf::data_type castTarget) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    VELOX_CHECK(
        vector->isConstantEncoding(), "Expected constant vector for IN list");

    auto constantVector = vector->asUnchecked<ConstantVector<ComplexType>>();
    VELOX_CHECK(!constantVector->isNullAt(0), "NULL array in IN not supported");

    auto valueVector = constantVector->valueVector();
    VELOX_CHECK(
        valueVector->encoding() == VectorEncoding::Simple::ARRAY,
        "Expected ARRAY encoding for IN list");

    auto arrayVector = valueVector->as<ArrayVector>();
    auto index = constantVector->index();
    auto size = arrayVector->sizeAt(index);
    auto offset = arrayVector->offsetAt(index);
    auto elements = arrayVector->elements();
    VELOX_CHECK_GT(size, 0, "Empty IN list");

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(size);

    for (auto i = offset; i < offset + size; ++i) {
      if (elements->isNullAt(i)) {
        continue;
      }
      auto scalar = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          scalarFromElement,
          elements->typeKind(),
          elements,
          elements->type(),
          i);
      columns.push_back(cudf::make_column_from_scalar(*scalar, 1, stream, mr));
    }

    VELOX_CHECK(!columns.empty(), "IN list has no non-null values");

    if (columns.size() == 1) {
      haystack_ = std::move(columns[0]);
    } else {
      std::vector<cudf::column_view> views;
      views.reserve(columns.size());
      for (const auto& col : columns) {
        views.push_back(col->view());
      }
      haystack_ = cudf::concatenate(views, stream, mr);
    }

    if (castTarget.id() != cudf::type_id::EMPTY &&
        haystack_->view().type() != castTarget) {
      if (haystack_->view().type().id() == cudf::type_id::STRING &&
          cudf::is_timestamp(castTarget)) {
        haystack_ = cudf::strings::to_timestamps(
            cudf::strings_column_view(haystack_->view()),
            castTarget,
            "%Y-%m-%d",
            stream,
            mr);
      } else {
        haystack_ = cudf::cast(haystack_->view(), castTarget, stream, mr);
      }
    }
  }

  bool skipCast_;
  std::unique_ptr<cudf::column> haystack_;
};

// Bloom filter probe: might_contain(bloom_filter, value).
// Bloom filter is a probabilistic data structure used by Spark for runtime
// filtering. Returning all-true is always correct (false positives allowed).
// TODO: implement actual GPU bloom filter probe for better performance.
class MightContainFunction : public CudfFunction {
 public:
  MightContainFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "might_contain expects at least 2 inputs");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_GE(
        inputColumns.size(), 1, "might_contain needs at least 1 input column");
    auto probeCol = asView(inputColumns.back());
    auto trueScalar = cudf::numeric_scalar<bool>(true, true, stream, mr);
    return cudf::make_column_from_scalar(trueScalar, probeCol.size(), stream, mr);
  }
};

class IsNullFunction : public CudfFunction {
 public:
  IsNullFunction(bool negate) : negate_(negate) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "isnull/isnotnull expects 1 input");
    auto col = asView(inputColumns[0]);
    auto n = col.size();

    if (!col.nullable() || col.null_count() == 0) {
      auto scalar = cudf::numeric_scalar<bool>(negate_, true, stream, mr);
      return cudf::make_column_from_scalar(scalar, n, stream, mr);
    }

    if (col.null_count() == n) {
      auto scalar = cudf::numeric_scalar<bool>(!negate_, true, stream, mr);
      return cudf::make_column_from_scalar(scalar, n, stream, mr);
    }

    // Build isnotnull: all-true column with input's null mask, then replace
    // nulls with false. This yields true for valid rows, false for null rows.
    auto trueScalar = cudf::numeric_scalar<bool>(true, true, stream, mr);
    auto allTrue = cudf::make_column_from_scalar(trueScalar, n, stream, mr);
    auto maskBuf = cudf::copy_bitmask(col, stream, mr);
    allTrue->set_null_mask(std::move(maskBuf), col.null_count());

    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    auto isnotnull = cudf::replace_nulls(allTrue->view(), falseScalar, stream, mr);

    if (negate_) {
      return isnotnull;
    }
    return cudf::unary_operation(
        isnotnull->view(), cudf::unary_operator::NOT, stream, mr);
  }

 private:
  bool negate_; // false = isnull, true = isnotnull
};

class GreatestLeastFunction : public CudfFunction {
 public:
  GreatestLeastFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::binary_operator op)
      : op_(op), type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    // must have at least three inputs
    VELOX_CHECK_GE(
        expr->inputs().size(),
        3,
        "Greatest/Least function expects at least 3 inputs");
    // scan inputs for literals
    for (size_t i = 0; i < expr->inputs().size(); ++i) {
      auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
          expr->inputs()[i]);
      if (constExpr) {
        literals_.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createCudfScalar,
            constExpr->value()->typeKind(),
            constExpr->value()));
      } else {
        literals_.push_back(nullptr);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // construct a chain of NULL_MIN or NULL_MAX operations
    std::unique_ptr<cudf::column> result;
    // the first pair of values
    if (literals_[0] && literals_[1]) {
      // no variant of cudf::binary_operation that takes two scalars so we must
      // create columns
      auto col0 = cudf::make_column_from_scalar(*literals_[0], 1, stream);
      auto col1 = cudf::make_column_from_scalar(*literals_[1], 1, stream);
      result = cudf::binary_operation(
          col0->view(), col1->view(), op_, type_, stream, mr);
    } else if (literals_[0]) {
      result = cudf::binary_operation(
          *literals_[0], asView(inputColumns[1]), op_, type_, stream, mr);
    } else if (literals_[1]) {
      result = cudf::binary_operation(
          asView(inputColumns[0]), *literals_[1], op_, type_, stream, mr);
    } else {
      result = cudf::binary_operation(
          asView(inputColumns[0]),
          asView(inputColumns[1]),
          op_,
          type_,
          stream,
          mr);
    }
    // remaining values
    for (size_t i = 2; i < inputColumns.size(); ++i) {
      if (literals_[i]) {
        result = cudf::binary_operation(
            result->view(), *literals_[i], op_, type_, stream, mr);
      } else {
        result = cudf::binary_operation(
            result->view(), asView(inputColumns[i]), op_, type_, stream, mr);
      }
    }
    return result;
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  std::vector<std::unique_ptr<cudf::scalar>> literals_;
};

class SwitchFunction : public CudfFunction {
 public:
  SwitchFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 3, "case when expects exactly 3 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::BOOLEAN,
        "The switch condition result type should be boolean");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr),
        "The condition should not be constant");
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[1])) {
      auto constValue = constExpr->value();
      left_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
    if (auto constExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[2])) {
      auto constValue = constExpr->value();
      right_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, constValue->typeKind(), constValue);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    if (left_ == nullptr && right_ == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]),
          asView(inputColumns[2]),
          asView(inputColumns[0]),
          stream,
          mr);
    } else if (left_ == nullptr) {
      return cudf::copy_if_else(
          asView(inputColumns[1]),
          *right_,
          asView(inputColumns[0]),
          stream,
          mr);
    } else if (right_ == nullptr) {
      return cudf::copy_if_else(
          *left_, asView(inputColumns[1]), asView(inputColumns[0]), stream, mr);
    }
    // right != null and left != null
    return cudf::copy_if_else(
        *left_, *right_, asView(inputColumns[0]), stream, mr);
  }

 private:
  std::unique_ptr<cudf::scalar> left_;
  std::unique_ptr<cudf::scalar> right_;
};

int64_t getIntConstant(
    const std::shared_ptr<velox::exec::ConstantExpr>& constExpr) {
  auto value = constExpr->value();
  switch (value->typeKind()) {
    case TypeKind::INTEGER:
      return value->as<SimpleVector<int32_t>>()->valueAt(0);
    case TypeKind::BIGINT:
      return value->as<SimpleVector<int64_t>>()->valueAt(0);
    default:
      VELOX_FAIL(
          "Expected integer or bigint constant, got {}",
          value->type()->toString());
  }
}

class SubstrFunction : public CudfFunction {
 public:
  SubstrFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substr expects at least 2 inputs");
    VELOX_CHECK_LE(expr->inputs().size(), 3, "substr expects at most 3 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto startExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(startExpr, "substr start must be a constant");

    auto startValue = getIntConstant(startExpr);
    cudf::size_type adjustedStart = static_cast<cudf::size_type>(startValue);
    if (startValue >= 1) {
      // cuDF indexing starts at 0.
      // Presto indexing starts at 1.
      // Positive indices need to substract 1.
      adjustedStart = static_cast<cudf::size_type>(startValue - 1);
    }

    startScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
        adjustedStart, true, stream, mr);

    if (expr->inputs().size() > 2) {
      auto lengthExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(lengthExpr, "substr length must be a constant");

      auto lengthValue = getIntConstant(lengthExpr);
      // cuDF uses indices [begin, end).
      // Presto uses length as the length of the substring.
      // We compute the end as start + length.
      cudf::size_type endPosition =
          adjustedStart + static_cast<cudf::size_type>(lengthValue);

      endScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
          endPosition, true, stream, mr);
    } else {
      endScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
          0, false, stream, mr);
    }

    stepScalar_ = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(
        1, true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::slice_strings(
        inputCol, *startScalar_, *endScalar_, *stepScalar_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> startScalar_;
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> endScalar_;
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> stepScalar_;
};

// Constructs a struct column from input columns with null propagation.
// NullMode 0: struct is never null (row_constructor)
// NullMode 1: struct is null if ANY input is null
//   (row_constructor_with_null)
// NullMode 2: struct is null only if ALL inputs are null
//   (row_constructor_with_all_null)
template <int NullMode>
class RowConstructorFunction : public CudfFunction {
 public:
  RowConstructorFunction(
      const std::shared_ptr<velox::exec::Expr>& /*expr*/) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "row_constructor requires at least one input");

    auto const numRows = asView(inputColumns[0]).size();

    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(inputColumns.size());
    for (auto& col : inputColumns) {
      auto view = asView(col);
      children.push_back(
          std::make_unique<cudf::column>(view, stream, mr));
    }

    rmm::device_buffer null_mask{};
    cudf::size_type null_count = 0;

    if constexpr (NullMode == 1) {
      // ANY child null -> struct null
      auto tbl_view = cudf::table_view(
          [&]() {
            std::vector<cudf::column_view> views;
            for (auto& c : children)
              views.push_back(c->view());
            return views;
          }());
      auto [mask, nc] =
          cudf::bitmask_and(tbl_view, stream, mr);
      null_mask = std::move(mask);
      null_count = nc;
    } else if constexpr (NullMode == 2) {
      // ALL children null -> struct null
      auto tbl_view = cudf::table_view(
          [&]() {
            std::vector<cudf::column_view> views;
            for (auto& c : children)
              views.push_back(c->view());
            return views;
          }());
      auto [mask, nc] =
          cudf::bitmask_or(tbl_view, stream, mr);
      null_mask = std::move(mask);
      null_count = nc;
    }
    // NullMode 0: no null mask, struct is never null

    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::STRUCT},
        numRows,
        rmm::device_buffer{},
        std::move(null_mask),
        null_count,
        std::move(children));
  }
};


class CoalesceFunction : public CudfFunction {
 public:
  CoalesceFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    // Storing the first literal that appears in inputs because we don't need to
    // process after that. This is the last fallback.
    numColumnsBeforeLiteral_ = expr->inputs().size();
    for (size_t i = 0; i < expr->inputs().size(); ++i) {
      const auto& input = expr->inputs()[i];
      if (auto c =
              std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input)) {
        if (!c->value()->isNullAt(0)) {
          literalScalar_ = makeScalarFromConstantExpr(c);
          numColumnsBeforeLiteral_ = i;
          break;
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // Coalesce is practically a cudf::replace_nulls over multiple columns.
    // Starting from first column, we keep calling replace nulls with
    // subsequent cols until we get an all valid col or run out of columns

    // If a literal comes before any column input, fill the result with it.
    if (literalScalar_ && numColumnsBeforeLiteral_ == 0) {
      if (inputColumns.empty()) {
        // We need at least one column to tell us the required output size
        VELOX_NYI("coalesce with only literal inputs is not supported");
      }
      auto size = asView(inputColumns[0]).size();
      return cudf::make_column_from_scalar(*literalScalar_, size, stream, mr);
    }

    VELOX_CHECK(
        !inputColumns.empty(),
        "coalesce requires at least one non-literal input");
    ColumnOrView result = asView(inputColumns[0]);
    size_t stop = std::min(numColumnsBeforeLiteral_, inputColumns.size());
    for (size_t i = 1; i < stop && asView(result).has_nulls(); ++i) {
      result = cudf::replace_nulls(
          asView(result), asView(inputColumns[i]), stream, mr);
    }

    if (literalScalar_ && asView(result).has_nulls()) {
      result = cudf::replace_nulls(asView(result), *literalScalar_, stream, mr);
    }

    return result;
  }

 private:
  size_t numColumnsBeforeLiteral_;
  std::unique_ptr<cudf::scalar> literalScalar_;
};

class HashFunction : public CudfFunction {
 public:
  HashFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(expr->inputs().size(), 2, "hash expects at least 2 inputs");
    auto seedExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(seedExpr, "hash seed must be a constant");
    int32_t seedValue =
        seedExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
    VELOX_CHECK_GE(seedValue, 0);
    seedValue_ = seedValue;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty());
    auto inputTableView = convertToTableView(inputColumns);
    return cudf::hashing::murmurhash3_x86_32(
        inputTableView, seedValue_, stream, mr);
  }

 private:
  static cudf::table_view convertToTableView(
      std::vector<ColumnOrView>& inputColumns) {
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());

    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }

    return cudf::table_view(columns);
  }

  uint32_t seedValue_;
};

class XxHash64Function : public CudfFunction {
 public:
  XxHash64Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "xxhash64 expects at least 2 inputs");
    auto seedExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(seedExpr, "xxhash64 seed must be a constant");
    auto seedVec = seedExpr->value();
    if (seedVec->typeKind() == TypeKind::BIGINT) {
      seedValue_ = static_cast<uint64_t>(
          seedVec->as<SimpleVector<int64_t>>()->valueAt(0));
    } else {
      seedValue_ = static_cast<uint64_t>(
          seedVec->as<SimpleVector<int32_t>>()->valueAt(0));
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty());
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());
    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }
    return cudf::hashing::xxhash_64(
        cudf::table_view(columns), seedValue_, stream, mr);
  }

 private:
  uint64_t seedValue_;
};

class YearFunction : public CudfFunction {
 public:
  explicit YearFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "year expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_datetime_component(
        inputCol, cudf::datetime::datetime_component::YEAR, stream, mr);
  }
};

class LengthFunction : public CudfFunction {
 public:
  explicit LengthFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "length expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::count_characters(inputCol, stream, mr);
  }
};

class LowerFunction : public CudfFunction {
 public:
  explicit LowerFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "lower expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_lower(inputCol, stream, mr);
  }
};

class UpperFunction : public CudfFunction {
 public:
  explicit UpperFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "upper expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::to_upper(inputCol, stream, mr);
  }
};

class LikeFunction : public CudfFunction {
 public:
  explicit LikeFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "like expects 2 or 3 inputs, got {}",
        expr->inputs().size());

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "like pattern must be a constant");
    pattern_ = patternExpr->value()->toString(0);

    if (expr->inputs().size() == 3) {
      auto escapeExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(escapeExpr, "like escape char must be a constant");
      escape_ = escapeExpr->value()->toString(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::like(inputCol, pattern_, escape_, stream, mr);
  }

 private:
  std::string pattern_;
  std::string escape_;
};

class StartswithFunction : public CudfFunction {
 public:
  explicit StartswithFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "startswith expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "startswith pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::starts_with(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

class EndswithFunction : public CudfFunction {
 public:
  explicit EndswithFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "endswith expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "endswith pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::ends_with(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

class ContainsFunction : public CudfFunction {
 public:
  explicit ContainsFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "contains expects 2 inputs");

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(patternExpr, "contains pattern must be a constant");
    pattern_ = std::make_unique<cudf::string_scalar>(
        patternExpr->value()->toString(0), true, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::strings::contains(inputCol, *pattern_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::string_scalar> pattern_;
};

class AbsFunction : public CudfFunction {
 public:
  explicit AbsFunction(
      const std::shared_ptr<velox::exec::Expr>& /*expr*/) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::unary_operation(
        asView(inputColumns[0]),
        cudf::unary_operator::ABS,
        stream,
        mr);
  }
};

class ConcatFunction : public CudfFunction {
 public:
  explicit ConcatFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    numInputs_ = 0;
    for (const auto& input : expr->inputs()) {
      if (auto c =
              std::dynamic_pointer_cast<exec::ConstantExpr>(input)) {
        auto val = c->value();
        VELOX_CHECK(val->isConstantEncoding());
        auto sv =
            val->as<SimpleVector<StringView>>()->valueAt(0);
        literals_.push_back(std::string(sv.data(), sv.size()));
        literalPositions_.push_back(numInputs_ + literals_.size() - 1);
      } else {
        columnPositions_.push_back(
            numInputs_ + literals_.size());
        numInputs_++;
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    if (inputColumns.empty() && literals_.empty()) {
      VELOX_FAIL("concat requires at least one input");
    }

    cudf::size_type numRows = 0;
    for (auto& col : inputColumns) {
      numRows = std::max(numRows, asView(col).size());
    }

    std::vector<std::unique_ptr<cudf::column>> literalCols;
    std::vector<cudf::column_view> allCols;
    allCols.resize(
        inputColumns.size() + literals_.size());

    size_t colIdx = 0;
    size_t litIdx = 0;
    for (size_t i = 0;
         i < inputColumns.size() + literals_.size();
         ++i) {
      bool isLit = litIdx < literalPositions_.size() &&
          literalPositions_[litIdx] == i;
      if (isLit) {
        auto sc = cudf::string_scalar(
            literals_[litIdx], true, stream, mr);
        literalCols.push_back(
            cudf::make_column_from_scalar(sc, numRows, stream, mr));
        allCols[i] = literalCols.back()->view();
        litIdx++;
      } else {
        allCols[i] = asView(inputColumns[colIdx]);
        colIdx++;
      }
    }

    auto tv = cudf::table_view(allCols);
    auto emptySep = cudf::string_scalar("", true, stream, mr);
    auto nullRep = cudf::string_scalar("", false, stream, mr);
    return cudf::strings::concatenate(
        tv, emptySep, nullRep,
        cudf::strings::separator_on_nulls::YES,
        stream, mr);
  }

 private:
  size_t numInputs_{0};
  std::vector<std::string> literals_;
  std::vector<size_t> literalPositions_;
  std::vector<size_t> columnPositions_;
};

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  auto& registry = getCudfFunctionRegistry();
  if (!overwrite && registry.find(name) != registry.end()) {
    return false;
  }
  registry[name] = CudfFunctionSpec{std::move(factory), signatures};
  return true;
}

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite) {
  for (const auto& name : aliases) {
    registerCudfFunction(name, factory, signatures, overwrite);
  }
}

std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const std::shared_ptr<velox::exec::Expr>& expr) {
  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second.factory(name, expr);
  }
  return nullptr;
}

bool registerBuiltinFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "split",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SplitFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("array(varchar)")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           // cuDF expects cudf::size_type (int32) but we may get bigint from
           // presto. SplitFunction hacks around this by converting to string
           // and back
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "cardinality",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CardinalityFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("array(any)")
           .build()});

  registerCudfFunctions(
      {prefix + "substr", prefix + "substring"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SubstrFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("bigint")
           .constantArgumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("integer")
           .constantArgumentType("integer")
           .build()});

  // Coalesce is special form and doesn't have a prefix in its name.
  registerCudfFunction(
      "coalesce",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CoalesceFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("T")
           .variableArity("T")
           .build()});

  registerCudfFunction(
      "and",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LogicalFunction>(
            expr, cudf::binary_operator::LOGICAL_AND);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .variableArity("boolean")
           .build()});

  registerCudfFunction(
      "or",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LogicalFunction>(
            expr, cudf::binary_operator::LOGICAL_OR);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .variableArity("boolean")
           .build()});

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .constantArgumentType("integer")
           .argumentType("any")
           .variableArity()
           .build()});

  registerCudfFunction(
      prefix + "xxhash64_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<XxHash64Function>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("bigint")
           .argumentType("any")
           .variableArity()
           .build()});

  registerCudfFunction(
      prefix + "round",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RoundFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("real")
           .argumentType("real")
           .build(),
       FunctionSignatureBuilder()
           .returnType("real")
           .argumentType("real")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .build(),
       FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "year",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<YearFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("timestamp")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("date")
           .build()});

  registerCudfFunction(
      prefix + "length",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LengthFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "lower",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LowerFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "upper",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<UpperFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "like",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LikeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "startswith",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<StartswithFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "endswith",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<EndswithFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "contains",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ContainsFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  // Our cudf binary ops can take all numeric types but instead of listing them
  // all, we're testing if input types can be casted to double. Coersion will
  // pass because all numerics can be casted to double.
  // TODO (dm): This could break for decimal
  auto cmpSigs = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("double")
          .argumentType("double")
          .build()};

  // NOTE: "greaterthan"/"gt" and "divide" are NOT registered here.
  // They are registered below via registerComparisonOp / registerBinaryOp
  // which include both double AND decimal signatures.  Registering them
  // here with only the double signature would shadow the decimal-capable
  // registration (overwrite=false), breaking decimal comparisons and
  // decimal division (e.g. TPC-DS Q18 AVG on decimals).

  auto intShiftSigs = std::vector<exec::FunctionSignaturePtr>{
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .argumentType("integer")
          .build()};

  registerCudfFunction(
      prefix + "shiftright",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<BinaryFunction>(
            expr, cudf::binary_operator::SHIFT_RIGHT);
      },
      intShiftSigs);

  registerCudfFunction(
      prefix + "shiftleft",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<BinaryFunction>(
            expr, cudf::binary_operator::SHIFT_LEFT);
      },
      intShiftSigs);

  // No prefix because switch and if are special form
  registerCudfFunctions(
      {"switch", "if"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SwitchFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("T")
           .argumentType("boolean")
           .argumentType("T")
           .argumentType("T")
           .build()});

  registerCudfFunctions(
      // No signatures required for cast and try_cast. They are special forms.
      {"try_cast", "cast"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<CastFunction>(expr);
      },
      {
          // Cast needs special handling dynamically using cudf.
      });

  //
  // regular binary operators
  //

  auto registerBinaryOp = [&](const std::vector<std::string>& aliases,
                              cudf::binary_operator op) {
    auto decimalBinarySignature = [&](cudf::binary_operator decimalOp) {
      std::string rPrecisionConstraint;
      std::string rScaleConstraint;
      switch (decimalOp) {
        case cudf::binary_operator::ADD:
        case cudf::binary_operator::SUB:
          rPrecisionConstraint =
              "min(38, max(a_precision - a_scale, b_precision - b_scale) + "
              "max(a_scale, b_scale) + 1)";
          rScaleConstraint = "max(a_scale, b_scale)";
          break;
        case cudf::binary_operator::MUL:
          rPrecisionConstraint = "min(38, a_precision + b_precision)";
          rScaleConstraint = "a_scale + b_scale";
          break;
        case cudf::binary_operator::DIV:
          rPrecisionConstraint =
              "min(38, a_precision + b_scale + max(0, b_scale - a_scale))";
          rScaleConstraint = "max(a_scale, b_scale)";
          break;
        case cudf::binary_operator::MOD:
          rPrecisionConstraint =
              "min(b_precision - b_scale, a_precision - a_scale) + "
              "max(a_scale, b_scale)";
          rScaleConstraint = "max(a_scale, b_scale)";
          break;
        default:
          VELOX_FAIL("Unsupported decimal binary operator");
      }

      return FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .integerVariable("r_precision", rPrecisionConstraint)
          .integerVariable("r_scale", rScaleConstraint)
          .returnType("decimal(r_precision, r_scale)")
          .argumentType("decimal(a_precision, a_scale)")
          .argumentType("decimal(b_precision, b_scale)")
          .build();
    };

    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<BinaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("double")
             .argumentType("double")
             .argumentType("double")
             .build(),
         decimalBinarySignature(op)});
  };

  registerBinaryOp(
      {prefix + "plus", prefix + "add"}, cudf::binary_operator::ADD);
  registerBinaryOp(
      {prefix + "minus", prefix + "subtract"}, cudf::binary_operator::SUB);
  registerBinaryOp({prefix + "multiply"}, cudf::binary_operator::MUL);
  registerBinaryOp({prefix + "divide"}, cudf::binary_operator::DIV);
  registerBinaryOp({prefix + "mod"}, cudf::binary_operator::MOD);

  //
  // regular comparison operators
  //

  auto registerComparisonOp = [&](const std::vector<std::string>& aliases,
                                  cudf::binary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<BinaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("boolean")
             .argumentType("double")
             .argumentType("double")
             .build(),
         FunctionSignatureBuilder()
             .integerVariable("a_precision")
             .integerVariable("a_scale")
             .integerVariable("b_precision")
             .integerVariable("b_scale")
             .returnType("boolean")
             .argumentType("decimal(a_precision, a_scale)")
             .argumentType("decimal(b_precision, b_scale)")
             .build()});
  };

  registerComparisonOp(
      {prefix + "equal", prefix + "eq", prefix + "equalto"},
      cudf::binary_operator::EQUAL);
  registerComparisonOp(
      {prefix + "notequal", prefix + "neq", prefix + "notequalto"},
      cudf::binary_operator::NOT_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthanorequal", prefix + "gte"},
      cudf::binary_operator::GREATER_EQUAL);
  registerComparisonOp(
      {prefix + "lessthanorequal", prefix + "lte"},
      cudf::binary_operator::LESS_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthan", prefix + "gt"}, cudf::binary_operator::GREATER);
  registerComparisonOp(
      {prefix + "lessthan", prefix + "lt"}, cudf::binary_operator::LESS);

  //
  // regular unary operators
  //

  auto registerUnaryOp = [&](const std::vector<std::string>& aliases,
                             cudf::unary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<UnaryFunction>(expr, op);
        },
        {FunctionSignatureBuilder()
             .returnType("double")
             .argumentType("double")
             .build(),
         FunctionSignatureBuilder()
             .integerVariable("p")
             .integerVariable("s")
             .returnType("decimal(p,s)")
             .argumentType("decimal(p,s)")
             .build()});
  };

  registerUnaryOp({prefix + "abs"}, cudf::unary_operator::ABS);
  registerUnaryOp({prefix + "negate"}, cudf::unary_operator::NEGATE);
  registerUnaryOp({prefix + "floor"}, cudf::unary_operator::FLOOR);
  registerUnaryOp({prefix + "ceil"}, cudf::unary_operator::CEIL);

  // @TODO (seves 1/28/26)
  // truncate
  // no direct cudf mapping
  // perhaps a compound operation using round/round_decimal

  //
  // between
  //

  registerCudfFunction(
      prefix + "between",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<BetweenFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("double")
           .argumentType("double")
           .argumentType("double")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("boolean")
           .argumentType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .build()});

  //
  // greatest & least
  //

  registerCudfFunction(
      prefix + "greatest",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MAX);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .variableArity("double")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .variableArity("decimal(p,s)")
           .build()});

  registerCudfFunction(
      prefix + "least",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MIN);
      },
      {FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .variableArity("double")
           .build(),
       FunctionSignatureBuilder()
           .integerVariable("p")
           .integerVariable("s")
           .returnType("decimal(p,s)")
           .argumentType("decimal(p,s)")
           .variableArity("decimal(p,s)")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      "row_constructor_with_null",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<
            RowConstructorFunction<1>>(expr);
      },
      {});

  registerCudfFunction(
      "row_constructor",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<
            RowConstructorFunction<0>>(expr);
      },
      {});

  registerCudfFunction(
      "row_constructor_with_all_null",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<
            RowConstructorFunction<2>>(expr);
      },
      {});

  registerCudfFunction(
      prefix + "abs",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<AbsFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("tinyint")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("smallint")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("real")
           .argumentType("real")
           .build(),
       FunctionSignatureBuilder()
           .returnType("double")
           .argumentType("double")
           .build()});

  registerCudfFunction(
      prefix + "concat",
      [](const std::string&,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ConcatFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .variableArity("varchar")
           .build()});

  return true;
}

std::shared_ptr<FunctionExpression> FunctionExpression::create(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const RowTypePtr& inputRowSchema) {
  auto node = std::make_shared<FunctionExpression>();
  node->expr_ = expr;
  node->inputRowSchema_ = inputRowSchema;

  auto name = expr->name();
  if (name == "in") {
    bool skipCast = InFunction::shouldSkipCast(expr);
    node->function_ = std::make_shared<InFunction>(expr, skipCast);
    if (skipCast) {
      auto& innerExpr = expr->inputs()[0]->inputs()[0];
      if (innerExpr->name() != "literal") {
        node->subexpressions_.push_back(
            createCudfExpression(innerExpr, inputRowSchema));
      }
    } else {
      auto& valueExpr = expr->inputs()[0];
      if (valueExpr->name() != "literal") {
        node->subexpressions_.push_back(
            createCudfExpression(valueExpr, inputRowSchema));
      }
    }
  } else if (name == "might_contain") {
    node->function_ = std::make_shared<MightContainFunction>(expr);
    for (const auto& input : expr->inputs()) {
      if (input->name() != "literal") {
        node->subexpressions_.push_back(
            createCudfExpression(input, inputRowSchema));
      }
    }
  } else if (name == "isnull" || name == "isnotnull") {
    node->function_ =
        std::make_shared<IsNullFunction>(name == "isnotnull");
    for (const auto& input : expr->inputs()) {
      if (input->name() != "literal") {
        node->subexpressions_.push_back(
            createCudfExpression(input, inputRowSchema));
      }
    }
  } else {
    node->function_ = createCudfFunction(name, expr);
    if (node->function_) {
      for (const auto& input : expr->inputs()) {
        if (input->name() != "literal") {
          node->subexpressions_.push_back(
              createCudfExpression(input, inputRowSchema));
        }
      }
    }
  }

  return node;
}

ColumnOrView FunctionExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  using velox::exec::FieldReference;

  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr_)) {
    auto name = fieldExpr->name();
    auto columnIndex = inputRowSchema_->getChildIdx(name);
    return inputColumnViews[columnIndex];
  }

  if (function_) {
    std::vector<ColumnOrView> subexprResults;
    subexprResults.reserve(subexpressions_.size());

    for (const auto& subexpr : subexpressions_) {
      subexprResults.push_back(subexpr->eval(inputColumnViews, stream, mr));
    }

    auto result = function_->eval(subexprResults, stream, mr);
    if (finalize) {
      const auto requestedType = cudf_velox::veloxToCudfDataType(expr_->type());
      auto resultView = asView(result);
      if (resultView.type() != requestedType) {
        return cudf::cast(resultView, requestedType, stream, mr);
      }
    }
    return result;
  }

  VELOX_FAIL(
      "Unsupported expression for recursive evaluation: " + expr_->name());
}

void FunctionExpression::close() {
  function_.reset();
  subexpressions_.clear();
}

bool FunctionExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  using velox::exec::FieldReference;

  if (std::dynamic_pointer_cast<FieldReference>(expr)) {
    return true;
  }

  const auto& opName = expr->name();
  if (opName == "cast" || opName == "try_cast") {
    const auto& srcType =
        expr->inputs().empty() ? nullptr : expr->inputs()[0]->type();
    const auto& dstType = expr->type();
    if (srcType == nullptr || dstType == nullptr) {
      return false;
    }
    auto src = cudf_velox::veloxToCudfDataType(srcType);
    auto dst = cudf_velox::veloxToCudfDataType(dstType);
    if (src.id() == cudf::type_id::TIMESTAMP_DAYS &&
        dst.id() == cudf::type_id::STRING) {
      return true;
    }
    return cudf::is_supported_cast(src, dst);
  }
  // row_constructor variants always accepted: the return type is a
  // struct dynamically derived from input types, so static signature
  // matching cannot express this. The actual GPU implementation
  // (RowConstructorFunction) handles any input combination.
  if (opName == "row_constructor_with_null" ||
      opName == "row_constructor" ||
      opName == "row_constructor_with_all_null") {
    return true;
  }

  if (opName == "in") {
    if (expr->inputs().size() != 2) {
      return false;
    }
    return std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
               expr->inputs()[1]) != nullptr;
  }

  if (opName == "might_contain") {
    return expr->inputs().size() >= 2;
  }

  if (opName == "isnull" || opName == "isnotnull") {
    return expr->inputs().size() == 1;
  }

  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(expr->name());
  if (it == registry.end()) {
    return false;
  }
  const auto& spec = it->second;
  return matchCallAgainstSignatures(*expr, spec.signatures);
}

bool canBeEvaluatedByCudf(std::shared_ptr<velox::exec::Expr> expr, bool deep) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  bool supported = false;
  for (const auto& [name, entry] : registry) {
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      supported = true;
      break;
    }
  }
  if (!supported) {
    LOG_FALLBACK(expr->toString());
    return false;
  }

  if (deep) {
    for (const auto& input : expr->inputs()) {
      if (input->name() != "literal" && !canBeEvaluatedByCudf(input, true)) {
        return false;
      }
    }
  }

  return true;
}

std::shared_ptr<CudfExpression> createCudfExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema,
    std::optional<std::string> except) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  std::vector<const CudfExpressionEvaluatorEntry*> candidates;
  for (const auto& [name, entry] : registry) {
    if (except && name == *except) {
      continue;
    }
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      candidates.push_back(&entry);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
      [](const auto* a, const auto* b) {
        return a->priority > b->priority;
      });

  for (const auto* entry : candidates) {
    try {
      return entry->create(expr, inputRowSchema);
    } catch (const VeloxException& e) {
      LOG(WARNING) << "Expression evaluator failed for expr '"
                   << expr->toString()
                   << "': " << e.message() << ". Trying next evaluator.";
    }
  }

  if (!candidates.empty()) {
    LOG(WARNING) << "All " << candidates.size()
                 << " expression evaluator candidates failed for expr '"
                 << expr->toString()
                 << "'. Falling back to FunctionExpression.";
  }

  return FunctionExpression::create(expr, inputRowSchema);
}

} // namespace facebook::velox::cudf_velox
