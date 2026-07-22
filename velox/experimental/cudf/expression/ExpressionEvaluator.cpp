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
#include "velox/experimental/cudf/expression/NullMask.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/Time.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

thread_local const core::QueryConfig* currentFunctionQueryConfig = nullptr;

class ScopedCudfFunctionQueryConfig {
 public:
  explicit ScopedCudfFunctionQueryConfig(const core::QueryConfig* queryConfig)
      : previous_(currentFunctionQueryConfig) {
    currentFunctionQueryConfig = queryConfig;
  }

  ~ScopedCudfFunctionQueryConfig() {
    currentFunctionQueryConfig = previous_;
  }

 private:
  const core::QueryConfig* previous_;
};

cudf::size_type resolveFieldReferenceIndex(
    velox::exec::FieldReference& fieldExpr,
    const RowTypePtr& parentRowType) {
  // FieldReference::index(EvalCtx) may retain the index resolved against the
  // original, outer input row. Re-evaluating it with a synthetic child row is
  // therefore unsafe for nested dereferences and selected the wrong struct
  // child in complex projection plans. Resolve the nested field directly
  // against its declared parent ROW instead.
  return static_cast<cudf::size_type>(
      parentRowType->getChildIdx(fieldExpr.field()));
}

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

std::unique_ptr<cudf::scalar> castDecimalScalar(
    const cudf::scalar& src,
    cudf::data_type targetType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!src.is_valid(stream)) {
    VELOX_CHECK(
        targetType.id() == cudf::type_id::DECIMAL64 ||
            targetType.id() == cudf::type_id::DECIMAL128,
        "castDecimalScalar: target must be DECIMAL64 or DECIMAL128");
    if (targetType.id() == cudf::type_id::DECIMAL128) {
      return std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
          0, numeric::scale_type{targetType.scale()}, false, stream, mr);
    }
    return std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        0, numeric::scale_type{targetType.scale()}, false, stream, mr);
  }

  __int128_t rep;
  if (src.type().id() == cudf::type_id::DECIMAL64) {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(src);
    rep = static_cast<int64_t>(dec.value(stream));
  } else {
    auto const& dec =
        static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(src);
    rep = static_cast<__int128_t>(dec.value(stream));
  }

  int32_t scaleDiff = src.type().scale() - targetType.scale();
  if (scaleDiff > 0) {
    for (int32_t i = 0; i < scaleDiff; ++i) {
      rep *= 10;
    }
  } else if (scaleDiff < 0) {
    for (int32_t i = 0; i < -scaleDiff; ++i) {
      rep /= 10;
    }
  }

  if (targetType.id() == cudf::type_id::DECIMAL128) {
    return cudf::make_fixed_point_scalar<numeric::decimal128>(
        rep, numeric::scale_type{targetType.scale()}, stream, mr);
  }
  return cudf::make_fixed_point_scalar<numeric::decimal64>(
      static_cast<int64_t>(rep),
      numeric::scale_type{targetType.scale()},
      stream,
      mr);
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

const core::QueryConfig* currentCudfFunctionQueryConfig() {
  return currentFunctionQueryConfig;
}

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

std::unordered_map<std::string, std::vector<CudfFunctionSpec>>&
getCudfFunctionRegistry() {
  static std::unordered_map<std::string, std::vector<CudfFunctionSpec>>
      registry;
  return registry;
}

bool isNullComplexConstantLiteral(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto constant =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  if (constant == nullptr || constant->value() == nullptr ||
      !constant->value()->isNullAt(0)) {
    return false;
  }

  switch (constant->type()->kind()) {
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
      return true;
    default:
      return false;
  }
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
    exec::SignatureBinder binder(*sig, argTypes, TypeCoercer::defaults());
    if (!binder.tryBind()) {
      continue;
    }
    // binder does not confirm whether positional arguments are
    // constants(scalars) as expected. we have to check manually
    const auto& constArgs = sig->constantArguments();
    const size_t fixed = std::min(constArgs.size(), n);
    bool ok = true;
    for (size_t i = 0; i < fixed; ++i) {
      if (constArgs[i] &&
          !std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
              call.inputs()[i])) {
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

bool hasFunctionNameSuffix(std::string_view name, std::string_view suffix) {
  return name.size() >= suffix.size() &&
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isSupportedFromJsonRowOfStringsExpr(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 1 ||
      expr->inputs()[0]->type()->kind() != TypeKind::VARCHAR ||
      expr->type()->kind() != TypeKind::ROW) {
    return false;
  }

  const auto& rowType = expr->type()->asRow();
  return std::all_of(
      rowType.children().begin(),
      rowType.children().end(),
      [](const TypePtr& child) { return child->kind() == TypeKind::VARCHAR; });
}

bool isSupportedDirectJsonLeafType(const TypePtr& type) {
  if (type->isDate() || type->isDecimal()) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
      return true;
    default:
      return false;
  }
}

bool isSupportedDirectJsonType(const TypePtr& type, bool isRoot) {
  switch (type->kind()) {
    case TypeKind::ROW:
      return std::all_of(
          type->asRow().children().begin(),
          type->asRow().children().end(),
          [](const TypePtr& child) {
            return isSupportedDirectJsonType(child, false);
          });
    case TypeKind::ARRAY:
      return isSupportedDirectJsonType(type->childAt(0), false);
    default:
      return !isRoot && isSupportedDirectJsonLeafType(type);
  }
}

bool isSupportedFromJsonExpr(const std::shared_ptr<velox::exec::Expr>& expr) {
  return isSupportedFromJsonRowOfStringsExpr(expr) ||
      (expr->inputs().size() == 1 &&
       expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR &&
       isSupportedDirectJsonType(expr->type(), true));
}

bool isCudfRegexFunction(std::string_view name) {
  return hasFunctionNameSuffix(name, "regexp_extract") ||
      hasFunctionNameSuffix(name, "regexp_replace") ||
      hasFunctionNameSuffix(name, "rlike");
}

constexpr std::string_view kLeadingZeroPreserveOneJavaPattern{"^0+(?!$)"};
constexpr std::string_view kLeadingZeroPreserveOneCudfPattern{"^0+(.+)$"};
constexpr std::string_view kLeadingZeroPreserveOneCudfReplacement{"\\1"};

bool isLeadingZeroPreserveOneRegexpReplace(
    std::string_view pattern,
    std::string_view replacement) {
  return pattern == kLeadingZeroPreserveOneJavaPattern && replacement.empty();
}

bool hasUnsupportedCudfRegexPattern(std::string_view pattern) {
  bool escaped = false;
  for (size_t i = 0; i < pattern.size(); ++i) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (pattern[i] == '\\') {
      escaped = true;
      continue;
    }
    if (pattern[i] != '(' || i + 2 >= pattern.size() || pattern[i + 1] != '?') {
      continue;
    }
    if (pattern[i + 2] == '=' || pattern[i + 2] == '!') {
      return true;
    }
    if (pattern[i + 2] == '<' && i + 3 < pattern.size() &&
        (pattern[i + 3] == '=' || pattern[i + 3] == '!')) {
      return true;
    }
  }
  return false;
}

bool hasUnsupportedCudfRegexPatternArg(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() < 2) {
    return false;
  }
  auto patternExpr =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[1]);
  if (patternExpr == nullptr || patternExpr->value() == nullptr ||
      patternExpr->value()->isNullAt(0)) {
    return false;
  }
  const auto pattern = patternExpr->value()->toString(0);
  if (!hasUnsupportedCudfRegexPattern(pattern)) {
    return false;
  }
  if (hasFunctionNameSuffix(expr->name(), "regexp_replace") &&
      expr->inputs().size() >= 3) {
    auto replacementExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[2]);
    if (replacementExpr != nullptr && replacementExpr->value() != nullptr &&
        !replacementExpr->value()->isNullAt(0) &&
        isLeadingZeroPreserveOneRegexpReplace(
            pattern, replacementExpr->value()->toString(0))) {
      return false;
    }
  }
  return true;
}

bool isSupportedSparkDatetimePattern(std::string_view pattern) {
  return pattern == "yyyyMMdd" || pattern == "yyyy-MM-dd" ||
      pattern == "yyyy/MM/dd" || pattern == "yyyy-MM-dd HH:mm:ss" ||
      pattern == "yyyy-MM-dd HH:mm:ss.SSS" || pattern == "MMM-yyyy";
}

bool hasSupportedConstantSparkDatetimePattern(
    const std::shared_ptr<velox::exec::Expr>& expr,
    size_t argumentIndex) {
  if (expr->inputs().size() <= argumentIndex) {
    return false;
  }
  const auto patternExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
      expr->inputs()[argumentIndex]);
  if (patternExpr == nullptr || patternExpr->value() == nullptr) {
    return false;
  }
  if (patternExpr->value()->isNullAt(0)) {
    return true;
  }
  return isSupportedSparkDatetimePattern(patternExpr->value()->toString(0));
}

bool isUtf8DecodeCharset(std::string_view charset) {
  if (charset.size() != 5 && charset.size() != 4) {
    return false;
  }

  char normalized[5];
  for (size_t i = 0; i < charset.size(); ++i) {
    normalized[i] =
        static_cast<char>(std::tolower(static_cast<unsigned char>(charset[i])));
  }
  return charset.size() == 5 ? std::string_view(normalized, 5) == "utf-8"
                             : std::string_view(normalized, 4) == "utf8";
}

bool hasSupportedConstantDecodeCharset(
    const std::shared_ptr<velox::exec::Expr>& expr,
    size_t argumentIndex) {
  if (expr->inputs().size() <= argumentIndex) {
    return false;
  }
  const auto charsetExpr = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
      expr->inputs()[argumentIndex]);
  if (charsetExpr == nullptr || charsetExpr->value() == nullptr ||
      charsetExpr->value()->isNullAt(0)) {
    return false;
  }
  return isUtf8DecodeCharset(charsetExpr->value()->toString(0));
}

bool isIntegralNonDecimalVeloxType(const TypePtr& type) {
  if (type == nullptr || type->isDate()) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return true;
    default:
      return false;
  }
}

int64_t readConstantIntegralValue(const velox::exec::ConstantExpr& expr) {
  switch (expr.type()->kind()) {
    case TypeKind::TINYINT:
      return expr.value()->as<SimpleVector<int8_t>>()->valueAt(0);
    case TypeKind::SMALLINT:
      return expr.value()->as<SimpleVector<int16_t>>()->valueAt(0);
    case TypeKind::INTEGER:
      return expr.value()->as<SimpleVector<int32_t>>()->valueAt(0);
    case TypeKind::BIGINT:
      return expr.value()->as<SimpleVector<int64_t>>()->valueAt(0);
    default:
      VELOX_UNSUPPORTED(
          "Unsupported integral constant type {}", expr.type()->toString());
  }
}

bool isFloatingPointVeloxType(const TypePtr& type) {
  if (type == nullptr) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      return true;
    default:
      return false;
  }
}

bool isNumericToBooleanVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  if (srcVelox == nullptr || dstVelox == nullptr ||
      dstVelox->kind() != TypeKind::BOOLEAN || srcVelox->isDate()) {
    return false;
  }
  return isIntegralNonDecimalVeloxType(srcVelox) ||
      isFloatingPointVeloxType(srcVelox) || srcVelox->isDecimal();
}

bool isStringToBooleanVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      srcVelox->kind() == TypeKind::VARCHAR &&
      dstVelox->kind() == TypeKind::BOOLEAN;
}

bool isStringToDateVeloxCast(const TypePtr& srcVelox, const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      srcVelox->kind() == TypeKind::VARCHAR && dstVelox->isDate();
}

bool isStringToTimestampVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      srcVelox->kind() == TypeKind::VARCHAR &&
      dstVelox->kind() == TypeKind::TIMESTAMP;
}

bool isTimestampToStringVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      srcVelox->kind() == TypeKind::TIMESTAMP &&
      dstVelox->kind() == TypeKind::VARCHAR;
}

bool isNumericToTimestampVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      dstVelox->kind() == TypeKind::TIMESTAMP &&
      (isIntegralNonDecimalVeloxType(srcVelox) ||
       isFloatingPointVeloxType(srcVelox));
}

bool isListIntegralToStringVeloxCast(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  return srcVelox != nullptr && dstVelox != nullptr &&
      srcVelox->kind() == TypeKind::ARRAY &&
      dstVelox->kind() == TypeKind::ARRAY &&
      isIntegralNonDecimalVeloxType(srcVelox->childAt(0)) &&
      dstVelox->childAt(0)->kind() == TypeKind::VARCHAR;
}

bool isSupportedCudfScalarLiteralType(const TypePtr& type) {
  if (type == nullptr) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::HUGEINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      return true;
    default:
      return false;
  }
}

bool isUnsupportedConstantLiteral(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto constant =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  if (constant == nullptr || constant->value() == nullptr) {
    return false;
  }
  return !isSupportedCudfScalarLiteralType(expr->type());
}

bool isUntypedNullConstantLiteral(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto constant =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  return constant != nullptr && constant->value() != nullptr &&
      constant->type()->kind() == TypeKind::UNKNOWN &&
      constant->value()->isNullAt(0);
}

bool isNonNullEmptyArrayConstantLiteral(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto constant =
      std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  if (constant == nullptr || constant->value() == nullptr ||
      constant->type()->kind() != TypeKind::ARRAY ||
      constant->value()->isNullAt(0) ||
      !constant->value()->isConstantEncoding()) {
    return false;
  }

  const auto* constantVector =
      constant->value()->asUnchecked<ConstantVector<ComplexType>>();
  const auto valueVector = constantVector->valueVector();
  if (valueVector == nullptr ||
      valueVector->encoding() != VectorEncoding::Simple::ARRAY) {
    return false;
  }
  const auto* arrayVector = valueVector->as<ArrayVector>();
  return arrayVector->sizeAt(constantVector->index()) == 0;
}

bool hasUnsupportedComplexConstantLiteral(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const bool isRowConstructor = expr->name() == "row_constructor" ||
      expr->name() == "row_constructor_with_null" ||
      expr->name() == "row_constructor_with_all_null";
  for (const auto& input : expr->inputs()) {
    if (isUnsupportedConstantLiteral(input) &&
        !(isRowConstructor && isUntypedNullConstantLiteral(input)) &&
        !(expr->name() == "coalesce" &&
          isNonNullEmptyArrayConstantLiteral(input))) {
      return true;
    }
    if (input->distinctFields().empty() && input->inputs().size() == 1 &&
        isUnsupportedConstantLiteral(input->inputs()[0]) &&
        !(expr->name() == "coalesce" &&
          isNonNullEmptyArrayConstantLiteral(input->inputs()[0]))) {
      return true;
    }
  }
  return false;
}

bool isPlainCudfCastSourceType(cudf::data_type type) {
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
      return true;
    default:
      return false;
  }
}

bool isPlainCudfCastSupported(
    const TypePtr& srcVelox,
    const TypePtr& dstVelox) {
  if (srcVelox == nullptr || dstVelox == nullptr) {
    return false;
  }

  auto src = cudf_velox::veloxToCudfDataType(srcVelox);
  auto dst = cudf_velox::veloxToCudfDataType(dstVelox);
  if (src == dst) {
    return true;
  }

  if (src.id() == cudf::type_id::STRING || dst.id() == cudf::type_id::STRING ||
      src.id() == cudf::type_id::LIST || dst.id() == cudf::type_id::LIST ||
      src.id() == cudf::type_id::STRUCT || dst.id() == cudf::type_id::STRUCT) {
    return false;
  }

  if (!isPlainCudfCastSourceType(src)) {
    return false;
  }

  return cudf::is_supported_cast(src, dst);
}

std::unique_ptr<cudf::column> makeAllNullStringColumn(
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::size_type> offsets(size + 1, 0);
  rmm::device_buffer offsetsBuffer(
      offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsetsBuffer),
      rmm::device_buffer{},
      0);
  return cudf::make_strings_column(
      size,
      std::move(offsetsColumn),
      rmm::device_buffer{},
      size,
      cudf::create_null_mask(size, cudf::mask_state::ALL_NULL, stream, mr));
}

std::unique_ptr<cudf::column> parseTimestampsWithInvalidsAsNulls(
    cudf::column_view inputCol,
    cudf::data_type targetType,
    std::string_view format,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto strings = cudf::strings_column_view(inputCol);
  auto validMask = cudf::strings::is_timestamp(strings, format, stream, mr);
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, mr);
  auto validNoNulls =
      cudf::replace_nulls(validMask->view(), falseScalar, stream, mr);

  auto parsed =
      cudf::strings::to_timestamps(strings, targetType, format, stream, mr);
  auto nullScalar =
      cudf::make_default_constructed_scalar(targetType, stream, mr);
  nullScalar->set_valid_async(false, stream);
  return cudf::copy_if_else(
      parsed->view(), *nullScalar, validNoNulls->view(), stream, mr);
}

std::unique_ptr<cudf::column> parseSparkStringDateCast(
    cudf::column_view inputCol,
    cudf::data_type targetType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto strings = cudf::strings_column_view(inputCol);
  auto dateValidMask =
      cudf::strings::is_timestamp(strings, "%Y-%m-%d", stream, mr);
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, mr);
  auto dateValidNoNulls =
      cudf::replace_nulls(dateValidMask->view(), falseScalar, stream, mr);

  auto dateParsed = parseTimestampsWithInvalidsAsNulls(
      inputCol, targetType, "%Y-%m-%d", stream, mr);
  auto timestampParsed = parseTimestampsWithInvalidsAsNulls(
      inputCol, targetType, "%Y-%m-%d %H:%M:%S", stream, mr);
  return cudf::copy_if_else(
      dateParsed->view(),
      timestampParsed->view(),
      dateValidNoNulls->view(),
      stream,
      mr);
}

double timestampUnitFactor(cudf::data_type targetType) {
  switch (targetType.id()) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return 1.0;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return 1000.0;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return 1000000.0;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return 1000000000.0;
    default:
      VELOX_UNSUPPORTED(
          "Unsupported numeric to timestamp target type {}",
          static_cast<int32_t>(targetType.id()));
  }
}

std::unique_ptr<cudf::column> castNumericSecondsToTimestamp(
    cudf::column_view inputCol,
    cudf::data_type targetType,
    bool sourceIsFloating,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::column> secondsOwner;
  auto secondsView = inputCol;
  if (inputCol.type().id() != cudf::type_id::FLOAT64) {
    secondsOwner = cudf::cast(
        inputCol, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
    secondsView = secondsOwner->view();
  }

  cudf::numeric_scalar<double> factor(
      timestampUnitFactor(targetType), true, stream, mr);
  auto scaled = cudf::binary_operation(
      secondsView,
      factor,
      cudf::binary_operator::MUL,
      cudf::data_type{cudf::type_id::FLOAT64},
      stream,
      mr);
  secondsOwner.reset();

  auto ticks = cudf::cast(
      scaled->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  scaled.reset();

  if (sourceIsFloating) {
    auto valid = cudf::is_not_nan(inputCol, stream, mr);
    auto nullScalar = cudf::numeric_scalar<int64_t>(0, false, stream, mr);
    ticks = cudf::copy_if_else(
        ticks->view(), nullScalar, valid->view(), stream, mr);
  }

  return std::make_unique<cudf::column>(
      cudf::bit_cast(ticks->view(), targetType), stream, mr);
}

std::unique_ptr<cudf::column> makeEmptyColumnForListScalar(
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (type->kind() == TypeKind::ARRAY) {
    std::vector<cudf::size_type> offsets{0};
    rmm::device_buffer offsetsBuffer(
        offsets.data(), sizeof(cudf::size_type), stream, mr);
    auto offsetsColumn = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        1,
        std::move(offsetsBuffer),
        rmm::device_buffer{},
        0);
    return cudf::make_lists_column(
        0,
        std::move(offsetsColumn),
        makeEmptyColumnForListScalar(type->childAt(0), stream, mr),
        0,
        rmm::device_buffer{});
  }
  return cudf::make_empty_column(cudf_velox::veloxToCudfDataType(type));
}

std::unique_ptr<cudf::scalar> makeEmptyArrayScalar(
    const TypePtr& arrayType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(arrayType->kind(), TypeKind::ARRAY);
  auto elements =
      makeEmptyColumnForListScalar(arrayType->childAt(0), stream, mr);
  auto scalar = cudf::make_list_scalar(elements->view(), stream, mr);
  stream.synchronize();
  return scalar;
}

std::unique_ptr<cudf::column> makeAllNullArrayColumn(
    const TypePtr& arrayType,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(arrayType->kind(), TypeKind::ARRAY);
  std::vector<cudf::size_type> offsets(size + 1, 0);
  rmm::device_buffer offsetsBuffer(
      offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsetsBuffer),
      rmm::device_buffer{},
      0);
  return cudf::make_lists_column(
      size,
      std::move(offsetsColumn),
      makeEmptyColumnForListScalar(arrayType->childAt(0), stream, mr),
      size,
      cudf::create_null_mask(size, cudf::mask_state::ALL_NULL, stream, mr));
}

std::unique_ptr<cudf::column> castListIntegralToString(
    cudf::column_view inputCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto listView = cudf::lists_column_view(inputCol);
  auto stringElements = cudf::strings::from_integers(
      listView.get_sliced_child(stream), stream, mr);
  auto offsets = std::make_unique<cudf::column>(listView.offsets(), stream, mr);
  rmm::device_buffer nullMask;
  if (inputCol.nullable()) {
    nullMask = cudf::copy_bitmask(inputCol, stream, mr);
  }
  return cudf::make_lists_column(
      inputCol.size(),
      std::move(offsets),
      std::move(stringElements),
      inputCol.null_count(),
      std::move(nullMask));
}

} // namespace

class SplitFunction : public CudfFunction {
 public:
  SplitFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    auto delimiterExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(delimiterExpr, "split delimiter must be a constant");
    delimiter_ = delimiterExpr->value()->toString(0);

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
    cudf::string_scalar delimiterScalar(delimiter_, true, stream, mr);
    return cudf::strings::split_record(
        inputCol, delimiterScalar, maxSplitCount_, stream, mr);
  };

 private:
  std::string delimiter_;
  cudf::size_type maxSplitCount_;
};

class RegexpExtractFunction : public CudfFunction {
 public:
  explicit RegexpExtractFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_EQ(
        expr->inputs().size(), 3, "regexp_extract expects exactly 3 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "regexp_extract input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "regexp_extract output must be VARCHAR");

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        patternExpr, "regexp_extract pattern must be a constant");
    patternIsNull_ = patternExpr->value()->isNullAt(0);
    if (!patternIsNull_) {
      pattern_ = patternExpr->value()->toString(0);
    }

    auto groupExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(groupExpr, "regexp_extract group must be a constant");
    groupIsNull_ = groupExpr->value()->isNullAt(0);
    if (!groupIsNull_) {
      group_ = static_cast<cudf::size_type>(
          std::stoll(groupExpr->value()->toString(0)));
      VELOX_USER_CHECK_GE(
          group_, 0, "regexp_extract group index must be non-negative");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "regexp_extract requires exactly one non-literal input column");
    auto inputCol = asView(inputColumns[0]);

    if (patternIsNull_ || groupIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    // Spark group 0 means the full match. libcudf extract_single group 0 means
    // the first capture group, so wrap the pattern only for Spark's group 0.
    const auto cudfPattern = group_ == 0 ? "(" + pattern_ + ")" : pattern_;
    const auto cudfGroup = group_ == 0 ? 0 : group_ - 1;
    auto regexProgram = cudf::strings::regex_program::create(cudfPattern);
    auto result = cudf::strings::extract_single(
        cudf::strings_column_view(inputCol),
        *regexProgram,
        cudfGroup,
        stream,
        mr);

    // Spark returns an empty string for no match or unmatched group, while
    // libcudf returns null. Replace extraction nulls with empty strings, then
    // merge the original input nulls back to preserve Spark null propagation.
    if (result->has_nulls()) {
      cudf::string_scalar emptyString("", true, stream, mr);
      result = cudf::replace_nulls(result->view(), emptyString, stream, mr);
    }
    mergeNullSourceNullsIntoResult(*result, inputCol, stream, mr);
    return result;
  }

 private:
  bool patternIsNull_{false};
  bool groupIsNull_{false};
  std::string pattern_;
  cudf::size_type group_{0};
};

class RegexpReplaceFunction : public CudfFunction {
 public:
  explicit RegexpReplaceFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK(
        expr->inputs().size() == 3 || expr->inputs().size() == 4,
        "regexp_replace expects 3 or 4 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "regexp_replace input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "regexp_replace output must be VARCHAR");

    auto patternExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        patternExpr, "regexp_replace pattern must be a constant");
    patternIsNull_ = patternExpr->value()->isNullAt(0);
    if (!patternIsNull_) {
      pattern_ = patternExpr->value()->toString(0);
    }

    auto replacementExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
    VELOX_CHECK_NOT_NULL(
        replacementExpr, "regexp_replace replacement must be a constant");
    replacementIsNull_ = replacementExpr->value()->isNullAt(0);
    if (!replacementIsNull_) {
      replacement_ = replacementExpr->value()->toString(0);
    }

    if (!patternIsNull_ && !replacementIsNull_ &&
        isLeadingZeroPreserveOneRegexpReplace(pattern_, replacement_)) {
      pattern_ = kLeadingZeroPreserveOneCudfPattern;
      replacement_ = kLeadingZeroPreserveOneCudfReplacement;
      useBackrefReplacement_ = true;
    }

    if (expr->inputs().size() == 4) {
      auto positionExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[3]);
      VELOX_CHECK_NOT_NULL(
          positionExpr, "regexp_replace position must be a constant");
      positionIsNull_ = positionExpr->value()->isNullAt(0);
      if (!positionIsNull_) {
        const auto position = std::stoll(positionExpr->value()->toString(0));
        VELOX_USER_CHECK_EQ(
            position,
            1,
            "cuDF regexp_replace currently supports only position = 1");
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "regexp_replace requires exactly one non-literal input column");
    auto inputCol = asView(inputColumns[0]);

    if (patternIsNull_ || replacementIsNull_ || positionIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    auto regexProgram = cudf::strings::regex_program::create(pattern_);
    if (useBackrefReplacement_) {
      return cudf::strings::replace_with_backrefs(
          cudf::strings_column_view(inputCol),
          *regexProgram,
          replacement_,
          stream,
          mr);
    }
    cudf::string_scalar replacementScalar(replacement_, true, stream, mr);
    return cudf::strings::replace_re(
        cudf::strings_column_view(inputCol),
        *regexProgram,
        replacementScalar,
        std::nullopt,
        stream,
        mr);
  }

 private:
  bool patternIsNull_{false};
  bool replacementIsNull_{false};
  bool positionIsNull_{false};
  bool useBackrefReplacement_{false};
  std::string pattern_;
  std::string replacement_;
};

class ReplaceFunction : public CudfFunction {
 public:
  explicit ReplaceFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "replace expects 2 or 3 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "replace input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "replace output must be VARCHAR");

    auto targetExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(targetExpr, "replace target must be a constant");
    targetIsNull_ = targetExpr->value()->isNullAt(0);
    if (!targetIsNull_) {
      target_ = targetExpr->value()->toString(0);
      VELOX_USER_CHECK(!target_.empty(), "cuDF replace target cannot be empty");
    }

    if (expr->inputs().size() == 3) {
      auto replacementExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(
          replacementExpr, "replace replacement must be a constant");
      replacementIsNull_ = replacementExpr->value()->isNullAt(0);
      if (!replacementIsNull_) {
        replacement_ = replacementExpr->value()->toString(0);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "replace requires exactly one non-literal input column");
    auto inputCol = asView(inputColumns[0]);

    if (targetIsNull_ || replacementIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    cudf::string_scalar targetScalar(target_, true, stream, mr);
    cudf::string_scalar replacementScalar(replacement_, true, stream, mr);
    return cudf::strings::replace(
        cudf::strings_column_view(inputCol),
        targetScalar,
        replacementScalar,
        -1,
        stream,
        mr);
  }

 private:
  bool targetIsNull_{false};
  bool replacementIsNull_{false};
  std::string target_;
  std::string replacement_;
};

class TrimFunction : public CudfFunction {
 public:
  TrimFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::strings::side_type side)
      : side_(side) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK(
        expr->inputs().size() == 1 || expr->inputs().size() == 2,
        "trim expects 1 or 2 inputs");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "trim output must be VARCHAR");

    if (expr->inputs().size() == 1) {
      VELOX_CHECK(
          expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
          "trim input must be VARCHAR");
      trimChars_ = " ";
      return;
    }

    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR &&
            expr->inputs()[1]->type()->kind() == TypeKind::VARCHAR,
        "trim inputs must be VARCHAR");
    auto trimExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(trimExpr, "trim characters must be a constant");
    trimCharsIsNull_ = trimExpr->value()->isNullAt(0);
    if (!trimCharsIsNull_) {
      trimChars_ = trimExpr->value()->toString(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "trim expects exactly one non-literal input column");
    auto inputCol = asView(inputColumns[0]);
    if (trimCharsIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    cudf::string_scalar trimChars(trimChars_, true, stream, mr);
    return cudf::strings::strip(
        cudf::strings_column_view(inputCol), side_, trimChars, stream, mr);
  }

 private:
  cudf::strings::side_type side_;
  std::string trimChars_;
  bool trimCharsIsNull_{false};
};

class GetJsonObjectFunction : public CudfFunction {
 public:
  GetJsonObjectFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "get_json_object expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "get_json_object json input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "get_json_object output must be VARCHAR");
    auto pathExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(pathExpr, "get_json_object path must be a constant");
    pathIsNull_ = pathExpr->value()->isNullAt(0);
    if (!pathIsNull_) {
      jsonPath_ = pathExpr->value()->toString(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "get_json_object expects one non-literal input column");
    auto inputCol = asView(inputColumns[0]);
    if (pathIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    cudf::string_scalar pathScalar(jsonPath_, true, stream, mr);
    cudf::get_json_object_options options;
    options.set_strip_quotes_from_single_strings(true);
    options.set_missing_fields_as_nulls(true);
    return cudf::get_json_object(
        cudf::strings_column_view(inputCol), pathScalar, options, stream, mr);
  }

 private:
  std::string jsonPath_;
  bool pathIsNull_{false};
};

class CastFunction : public CudfFunction {
 public:
  enum class CastMode {
    kIdentity,
    kCudfCast,
    kDateToString,
    kTimestampToString,
    kIntToString,
    kFloatToString,
    kStringToInt,
    kStringToFloat,
    kStringToBool,
    kStringToDate,
    kStringToTimestamp,
    kNumericToTimestamp,
    kNumericToBool,
    kListIntegralToString
  };

  CastFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "cast expects exactly 1 input");

    const auto& sourceVeloxType = expr->inputs()[0]->type();
    const auto& targetVeloxType = expr->type();
    targetCudfType_ = cudf_velox::veloxToCudfDataType(expr->type());
    auto sourceType =
        cudf_velox::veloxToCudfDataType(expr->inputs()[0]->type());

    if (isListIntegralToStringVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kListIntegralToString;
    } else if (sourceType == targetCudfType_) {
      castMode_ = CastMode::kIdentity;
    } else if (
        sourceVeloxType->isDate() &&
        targetVeloxType->kind() == TypeKind::VARCHAR) {
      castMode_ = CastMode::kDateToString;
    } else if (isTimestampToStringVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kTimestampToString;
    } else if (
        isIntegralNonDecimalVeloxType(sourceVeloxType) &&
        targetVeloxType->kind() == TypeKind::VARCHAR) {
      castMode_ = CastMode::kIntToString;
    } else if (
        isFloatingPointVeloxType(sourceVeloxType) &&
        targetVeloxType->kind() == TypeKind::VARCHAR) {
      castMode_ = CastMode::kFloatToString;
    } else if (
        sourceVeloxType->kind() == TypeKind::VARCHAR &&
        isIntegralNonDecimalVeloxType(targetVeloxType)) {
      castMode_ = CastMode::kStringToInt;
    } else if (
        sourceVeloxType->kind() == TypeKind::VARCHAR &&
        isFloatingPointVeloxType(targetVeloxType)) {
      castMode_ = CastMode::kStringToFloat;
    } else if (isStringToBooleanVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kStringToBool;
    } else if (isStringToDateVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kStringToDate;
    } else if (isStringToTimestampVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kStringToTimestamp;
    } else if (isNumericToTimestampVeloxCast(
                   sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kNumericToTimestamp;
      numericToTimestampSourceIsFloating_ =
          isFloatingPointVeloxType(sourceVeloxType);
    } else if (isNumericToBooleanVeloxCast(sourceVeloxType, targetVeloxType)) {
      castMode_ = CastMode::kNumericToBool;
      numericToBoolSourceIsFloating_ =
          isFloatingPointVeloxType(sourceVeloxType);
    } else {
      VELOX_CHECK(
          isPlainCudfCastSupported(sourceVeloxType, targetVeloxType),
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
    switch (castMode_) {
      case CastMode::kIdentity:
        return ColumnOrView(inputCol);
      case CastMode::kDateToString:
        return cudf::strings::from_timestamps(
            inputCol,
            "%Y-%m-%d",
            cudf::strings_column_view(
                cudf::column_view{
                    cudf::data_type{cudf::type_id::STRING},
                    0,
                    nullptr,
                    nullptr,
                    0}),
            stream,
            mr);
      case CastMode::kTimestampToString:
        return cudf::strings::from_timestamps(
            inputCol,
            "%Y-%m-%d %H:%M:%S",
            cudf::strings_column_view(
                cudf::column_view{
                    cudf::data_type{cudf::type_id::STRING},
                    0,
                    nullptr,
                    nullptr,
                    0}),
            stream,
            mr);
      case CastMode::kIntToString:
        return cudf::strings::from_integers(inputCol, stream, mr);
      case CastMode::kFloatToString:
        return cudf::strings::from_floats(inputCol, stream, mr);
      case CastMode::kStringToInt:
        return cudf::strings::to_integers(
            cudf::strings_column_view(inputCol), targetCudfType_, stream, mr);
      case CastMode::kStringToFloat:
        return cudf::strings::to_floats(
            cudf::strings_column_view(inputCol), targetCudfType_, stream, mr);
      case CastMode::kStringToBool: {
        cudf::string_scalar stripChars("", true, stream, mr);
        auto trimmed = cudf::strings::strip(
            cudf::strings_column_view(inputCol),
            cudf::strings::side_type::BOTH,
            stripChars,
            stream,
            mr);
        auto lowered = cudf::strings::to_lower(
            cudf::strings_column_view(trimmed->view()), stream, mr);
        trimmed.reset();

        auto trueProgram =
            cudf::strings::regex_program::create("^(t|true|y|yes|1)$");
        auto trueMask = cudf::strings::matches_re(
            cudf::strings_column_view(lowered->view()),
            *trueProgram,
            stream,
            mr);
        trueProgram.reset();

        auto falseProgram =
            cudf::strings::regex_program::create("^(f|false|n|no|0)$");
        auto falseMask = cudf::strings::matches_re(
            cudf::strings_column_view(lowered->view()),
            *falseProgram,
            stream,
            mr);
        falseProgram.reset();
        lowered.reset();

        auto validMask = cudf::binary_operation(
            trueMask->view(),
            falseMask->view(),
            cudf::binary_operator::LOGICAL_OR,
            targetCudfType_,
            stream,
            mr);

        auto trueScalar = cudf::numeric_scalar<bool>(true, true, stream, mr);
        auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
        auto boolResult = cudf::copy_if_else(
            trueScalar, falseScalar, trueMask->view(), stream, mr);
        trueMask.reset();
        falseMask.reset();

        auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);
        return cudf::copy_if_else(
            boolResult->view(), nullScalar, validMask->view(), stream, mr);
      }
      case CastMode::kStringToDate:
        return parseSparkStringDateCast(inputCol, targetCudfType_, stream, mr);
      case CastMode::kStringToTimestamp:
        return parseSparkStringDateCast(inputCol, targetCudfType_, stream, mr);
      case CastMode::kNumericToTimestamp:
        return castNumericSecondsToTimestamp(
            inputCol,
            targetCudfType_,
            numericToTimestampSourceIsFloating_,
            stream,
            mr);
      case CastMode::kNumericToBool: {
        auto zero =
            cudf::make_default_constructed_scalar(inputCol.type(), stream, mr);
        auto nonZero = cudf::binary_operation(
            inputCol,
            *zero,
            cudf::binary_operator::NOT_EQUAL,
            targetCudfType_,
            stream,
            mr);
        if (!numericToBoolSourceIsFloating_) {
          return nonZero;
        }

        auto notNan = cudf::is_not_nan(inputCol, stream, mr);
        auto result = cudf::binary_operation(
            nonZero->view(),
            notNan->view(),
            cudf::binary_operator::NULL_LOGICAL_AND,
            targetCudfType_,
            stream,
            mr);
        mergeNullSourceNullsIntoResult(*result, inputCol, stream, mr);
        return result;
      }
      case CastMode::kListIntegralToString:
        return castListIntegralToString(inputCol, stream, mr);
      case CastMode::kCudfCast:
        return cudf::cast(inputCol, targetCudfType_, stream, mr);
    }
    VELOX_UNREACHABLE();
  }

 private:
  cudf::data_type targetCudfType_;
  CastMode castMode_{CastMode::kCudfCast};
  bool numericToBoolSourceIsFloating_{false};
  bool numericToTimestampSourceIsFloating_{false};
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

class ArrayContainsFunction : public CudfFunction {
 public:
  explicit ArrayContainsFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "array_contains expects exactly 2 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::ARRAY,
        "array_contains expects an ARRAY input");
    VELOX_CHECK_EQ(
        expr->type()->kind(),
        TypeKind::BOOLEAN,
        "array_contains output must be BOOLEAN");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        2,
        "array_contains expects an array column and a search-key column");

    auto positions = cudf::lists::index_of(
        cudf::lists_column_view(asView(inputColumns[0])),
        asView(inputColumns[1]),
        cudf::lists::duplicate_find_option::FIND_FIRST,
        stream,
        mr);
    cudf::numeric_scalar<cudf::size_type> firstValidPosition(
        0, true, stream, mr);
    return cudf::binary_operation(
        positions->view(),
        firstValidPosition,
        cudf::binary_operator::GREATER_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
};

class ArrayConstructorFunction : public CudfFunction {
 public:
  explicit ArrayConstructorFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(expr->inputs().size(), 1, "array expects at least 1 input");
    VELOX_CHECK_EQ(
        expr->type()->kind(), TypeKind::ARRAY, "array output must be ARRAY");
    numInputs_ = expr->inputs().size();
    literals_.reserve(numInputs_);
    inputColumnIndices_.reserve(numInputs_);

    int32_t nextInputColumnIndex = 0;
    bool hasNonLiteralInput = false;
    for (const auto& input : expr->inputs()) {
      if (auto constant = std::dynamic_pointer_cast<ConstantExpr>(input)) {
        literals_.push_back(makeScalarFromConstantExpr(constant));
        inputColumnIndices_.push_back(-1);
      } else {
        hasNonLiteralInput = true;
        literals_.push_back(nullptr);
        inputColumnIndices_.push_back(nextInputColumnIndex++);
      }
    }
    if (!hasNonLiteralInput) {
      VELOX_NYI("array with only literal inputs is not supported");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty(), "array requires non-literal inputs");
    const auto outputSize = asView(inputColumns[0]).size();

    std::vector<std::unique_ptr<cudf::column>> materializedLiterals;
    std::vector<cudf::column_view> views;
    materializedLiterals.reserve(numInputs_);
    views.reserve(numInputs_);

    for (size_t arg = 0; arg < numInputs_; ++arg) {
      if (literals_[arg]) {
        materializedLiterals.push_back(
            cudf::make_column_from_scalar(
                *literals_[arg], outputSize, stream, mr));
        views.push_back(materializedLiterals.back()->view());
      } else {
        const auto inputColumnIndex = inputColumnIndices_[arg];
        VELOX_CHECK_GE(inputColumnIndex, 0);
        VELOX_CHECK_LT(inputColumnIndex, inputColumns.size());
        auto view = asView(inputColumns[inputColumnIndex]);
        VELOX_CHECK_EQ(view.size(), outputSize);
        views.push_back(view);
      }
    }

    auto values = cudf::interleave_columns(cudf::table_view(views), stream, mr);
    std::vector<cudf::size_type> offsets(outputSize + 1);
    for (cudf::size_type row = 0; row <= outputSize; ++row) {
      offsets[row] = row * static_cast<cudf::size_type>(numInputs_);
    }
    rmm::device_buffer offsetsBuffer(
        offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
    auto offsetsColumn = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets.size()),
        std::move(offsetsBuffer),
        rmm::device_buffer{},
        0);

    return cudf::make_lists_column(
        outputSize,
        std::move(offsetsColumn),
        std::move(values),
        0,
        rmm::device_buffer{});
  }

 private:
  std::vector<std::unique_ptr<cudf::scalar>> literals_;
  std::vector<int32_t> inputColumnIndices_;
  size_t numInputs_{0};
};

class SizeFunction : public CudfFunction {
 public:
  SizeFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "size expects exactly 2 inputs");
    auto legacyExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        legacyExpr, "size legacySizeOfNull argument must be a constant");
    legacySizeOfNull_ =
        legacyExpr->value()->as<SimpleVector<bool>>()->valueAt(0);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto inputCol = asView(inputColumns[0]);
    auto result = cudf::lists::count_elements(inputCol, stream, mr);
    if (legacySizeOfNull_ && result->has_nulls()) {
      auto minusOne = cudf::numeric_scalar<int32_t>(-1, true, stream, mr);
      result = cudf::replace_nulls(result->view(), minusOne, stream, mr);
    }
    return result;
  }

 private:
  bool legacySizeOfNull_{false};
};

class SortArrayFunction : public CudfFunction {
 public:
  explicit SortArrayFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK(
        expr->inputs().size() == 1 || expr->inputs().size() == 2,
        "sort_array expects 1 or 2 inputs");
    if (expr->inputs().size() == 2) {
      auto ascExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
      VELOX_CHECK_NOT_NULL(ascExpr, "sort_array asc flag must be constant");
      asc_ = ascExpr->value()->as<SimpleVector<bool>>()->valueAt(0);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    return cudf::lists::sort_lists(
        cudf::lists_column_view(asView(inputColumns[0])),
        asc_ ? cudf::order::ASCENDING : cudf::order::DESCENDING,
        asc_ ? cudf::null_order::BEFORE : cudf::null_order::AFTER,
        stream,
        mr);
  }

 private:
  bool asc_{true};
};

class ArrayDistinctFunction : public CudfFunction {
 public:
  explicit ArrayDistinctFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "array_distinct expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    return cudf::lists::distinct(
        cudf::lists_column_view(asView(inputColumns[0])),
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        cudf::duplicate_keep_option::KEEP_FIRST,
        stream,
        mr);
  }
};

class ArrayExceptFunction : public CudfFunction {
 public:
  explicit ArrayExceptFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "array_except expects exactly 2 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::ARRAY,
        "array_except expects an ARRAY left input");
    VELOX_CHECK_EQ(
        expr->inputs()[1]->type()->kind(),
        TypeKind::ARRAY,
        "array_except expects an ARRAY right input");
    VELOX_CHECK(
        expr->inputs()[0]->type()->childAt(0)->equivalent(
            *expr->inputs()[1]->type()->childAt(0)),
        "array_except inputs must have the same element type");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 2);
    return cudf::lists::difference_distinct(
        cudf::lists_column_view(asView(inputColumns[0])),
        cudf::lists_column_view(asView(inputColumns[1])),
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        stream,
        mr);
  }
};

class FlattenFunction : public CudfFunction {
 public:
  explicit FlattenFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "flatten expects exactly 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    return cudf::lists::concatenate_list_elements(
        asView(inputColumns[0]),
        cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW,
        stream,
        mr);
  }
};

class IsNullFunction : public CudfFunction {
 public:
  explicit IsNullFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "is_null expects 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "is_null expects 1 input");
    return cudf::is_null(asView(inputColumns[0]), stream, mr);
  }
};

class IsNotNullFunction : public CudfFunction {
 public:
  explicit IsNotNullFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "isnotnull expects 1 input");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "isnotnull expects 1 input");
    return cudf::is_valid(asView(inputColumns[0]), stream, mr);
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
    decimalsScalar_ = std::unique_ptr<cudf::numeric_scalar<int32_t>>(
        static_cast<cudf::numeric_scalar<int32_t>*>(
            makeScalarFromValue<int32_t>(INTEGER(), scale_, false).release()));
    factorScalar_ = std::unique_ptr<cudf::numeric_scalar<double>>(
        static_cast<cudf::numeric_scalar<double>*>(
            makeScalarFromValue<double>(
                DOUBLE(), std::pow(10.0, static_cast<double>(scale_)), false)
                .release()));
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto inputTypeId = inputCol.type().id();

    if (inputTypeId == cudf::type_id::FLOAT32) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
      auto result = cudf::round(
          inputCol, scale_, cudf::rounding_method::HALF_UP, stream, mr);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
      return result;
    }

    if (inputTypeId == cudf::type_id::FLOAT64) {
      // JIT-compile a CUDA UDF that mirrors the Velox CPU round implementation
      // (see velox/functions/prestosql/ArithmeticImpl.h). This makes
      // Velox-cuDF produce bit-identical results to the Velox CPU path for
      // round(double, scale) (e.g. round(2.675, 2) == 2.68) by relying on the
      // same `round(x * factor) / factor` trick.
      static constexpr char const* kUdf = R"***(
__device__ void velox_round_double(
    double* out, double number, int decimals, double factor) {
  if (!isfinite(number)) { *out = number; return; }
  if (decimals == 0) { *out = round(number); return; }
  if (decimals < 0) {
    *out = round(number * factor) / factor;
    return;
  }
  const double truncated = trunc(number);
  const double fraction = number - truncated;
  if (fraction == 0.0) { *out = number; return; }
  // Threshold matches Velox CPU: for small magnitudes the factor-multiply
  // path has less precision loss than the truncate + fraction path.
  if (fabs(number) < 17592186044415.0) {
    *out = round(number * factor) / factor;
    return;
  }
  const double roundedFractions = round(fraction * factor) / factor;
  *out = truncated + roundedFractions;
}
)***";

      // The scale-derived `decimals` and `factor` scalars are built once in
      // the constructor (see makeScalarFromValue). Each eval only constructs
      // two non-owning column_views over their device storage.
      const cudf::column_view decimalsView{
          cudf::data_type{cudf::type_id::INT32},
          /*size=*/1,
          decimalsScalar_->data(),
          /*null_mask=*/nullptr,
          /*null_count=*/0};
      const cudf::column_view factorView{
          cudf::data_type{cudf::type_id::FLOAT64},
          1,
          factorScalar_->data(),
          nullptr,
          0};

      const cudf::transform_input transformInputs[] = {
          inputCol,
          cudf::scalar_column_view(decimalsView),
          cudf::scalar_column_view(factorView),
      };
      return cudf::transform_extended(
          transformInputs,
          kUdf,
          cudf::data_type{cudf::type_id::FLOAT64},
          cudf::udf_source_type::CUDA,
          std::nullopt,
          cudf::null_aware::NO,
          std::nullopt,
          cudf::output_nullability::PRESERVE,
          stream,
          mr);
    }
    return cudf::round_decimal(
        inputCol, scale_, cudf::rounding_method::HALF_UP, stream, mr);
  }

 private:
  int32_t scale_ = 0;
  std::unique_ptr<cudf::numeric_scalar<int32_t>> decimalsScalar_;
  std::unique_ptr<cudf::numeric_scalar<double>> factorScalar_;
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
        case cudf::binary_operator::NULL_EQUALS:
        case cudf::binary_operator::NOT_EQUAL:
        case cudf::binary_operator::NULL_NOT_EQUALS:
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
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -rhsView.type().scale();
        auto outScale = -type_.scale();
        auto aRescale = outScale - lhsScale + rhsScale;
        return decimalDivide(lhsView, rhsView, type_, aRescale, stream, mr);
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
        // @TODO Check for divide-by-zero as in the DECIMAL case above?
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
          // @TODO Check for divide-by-zero as in the DECIMAL case above?
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
          // @TODO Check for divide-by-zero as in the DECIMAL case above?
          return cudf::binary_operation(
              lhsView, rhsView, op_, type_, stream, mr);
        }
      }
      // @TODO Check for divide-by-zero as in the DECIMAL case above?
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
        return decimalDivide(lhsView, *right_, type_, aRescale, stream, mr);
      }
      auto lhsView = asView(inputColumns[0]);
      if (isComparisonOp(op_) && cudf::is_fixed_point(lhsView.type()) &&
          cudf::is_fixed_point(right_->type())) {
        auto lhsScale = -lhsView.type().scale();
        auto rhsScale = -right_->type().scale();
        auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
        auto targetTypeId = (lhsView.type().id() == cudf::type_id::DECIMAL128 ||
                             right_->type().id() == cudf::type_id::DECIMAL128)
            ? cudf::type_id::DECIMAL128
            : cudf::type_id::DECIMAL64;
        auto targetType =
            cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
        std::unique_ptr<cudf::column> lhsCast;
        if (lhsView.type() != targetType) {
          lhsCast = cudf::cast(lhsView, targetType, stream, mr);
          lhsView = lhsCast->view();
        }
        if (right_->type() != targetType) {
          auto rhsScalar = castDecimalScalar(*right_, targetType, stream, mr);
          return cudf::binary_operation(
              lhsView, *rhsScalar, op_, type_, stream, mr);
        }
        return cudf::binary_operation(lhsView, *right_, op_, type_, stream, mr);
      }
      if (cudf::is_fixed_point(type_)) {
        if (op_ == cudf::binary_operator::ADD ||
            op_ == cudf::binary_operator::SUB ||
            op_ == cudf::binary_operator::MOD) {
          std::unique_ptr<cudf::column> lhsCast;
          if (lhsView.type() != type_) {
            lhsCast = cudf::cast(lhsView, type_, stream, mr);
            lhsView = lhsCast->view();
          }
          if (right_->type() != type_) {
            auto rhsScalar = castDecimalScalar(*right_, type_, stream, mr);
            return cudf::binary_operation(
                lhsView, *rhsScalar, op_, type_, stream, mr);
          }
          return cudf::binary_operation(
              lhsView, *right_, op_, type_, stream, mr);
        }
        if (op_ == cudf::binary_operator::MUL) {
          std::unique_ptr<cudf::column> lhsCast;
          std::unique_ptr<cudf::scalar> rhsScalar;
          if (type_.id() == cudf::type_id::DECIMAL128) {
            if (lhsView.type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, lhsView.type().scale()};
              lhsCast = cudf::cast(lhsView, castType, stream, mr);
              lhsView = lhsCast->view();
            }
            if (right_->type().id() == cudf::type_id::DECIMAL64) {
              auto castType = cudf::data_type{
                  cudf::type_id::DECIMAL128, right_->type().scale()};
              rhsScalar = castDecimalScalar(*right_, castType, stream, mr);
            }
          }
          return cudf::binary_operation(
              lhsView,
              rhsScalar ? *rhsScalar : *right_,
              op_,
              type_,
              stream,
              mr);
        }
      }
      return cudf::binary_operation(
          asView(inputColumns[0]), *right_, op_, type_, stream, mr);
    }
    if (op_ == cudf::binary_operator::DIV && cudf::is_fixed_point(type_)) {
      auto rhsView = asView(inputColumns[0]);
      auto lhsScale = -left_->type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto outScale = -type_.scale();
      auto aRescale = outScale - lhsScale + rhsScale;
      return decimalDivide(*left_, rhsView, type_, aRescale, stream, mr);
    }
    auto rhsView = asView(inputColumns[0]);
    if (isComparisonOp(op_) && cudf::is_fixed_point(left_->type()) &&
        cudf::is_fixed_point(rhsView.type())) {
      auto lhsScale = -left_->type().scale();
      auto rhsScale = -rhsView.type().scale();
      auto targetScale = lhsScale > rhsScale ? lhsScale : rhsScale;
      auto targetTypeId = (left_->type().id() == cudf::type_id::DECIMAL128 ||
                           rhsView.type().id() == cudf::type_id::DECIMAL128)
          ? cudf::type_id::DECIMAL128
          : cudf::type_id::DECIMAL64;
      auto targetType =
          cudf::data_type{targetTypeId, numeric::scale_type{-targetScale}};
      std::unique_ptr<cudf::column> rhsCast;
      if (rhsView.type() != targetType) {
        rhsCast = cudf::cast(rhsView, targetType, stream, mr);
        rhsView = rhsCast->view();
      }
      if (left_->type() != targetType) {
        auto lhsScalar = castDecimalScalar(*left_, targetType, stream, mr);
        return cudf::binary_operation(
            *lhsScalar, rhsView, op_, type_, stream, mr);
      }
      return cudf::binary_operation(*left_, rhsView, op_, type_, stream, mr);
    }
    if (cudf::is_fixed_point(type_)) {
      if (op_ == cudf::binary_operator::ADD ||
          op_ == cudf::binary_operator::SUB ||
          op_ == cudf::binary_operator::MOD) {
        std::unique_ptr<cudf::column> rhsCast;
        if (rhsView.type() != type_) {
          rhsCast = cudf::cast(rhsView, type_, stream, mr);
          rhsView = rhsCast->view();
        }
        if (left_->type() != type_) {
          auto lhsScalar = castDecimalScalar(*left_, type_, stream, mr);
          return cudf::binary_operation(
              *lhsScalar, rhsView, op_, type_, stream, mr);
        }
        return cudf::binary_operation(*left_, rhsView, op_, type_, stream, mr);
      }
      if (op_ == cudf::binary_operator::MUL) {
        std::unique_ptr<cudf::column> rhsCast;
        std::unique_ptr<cudf::scalar> lhsScalar;
        if (type_.id() == cudf::type_id::DECIMAL128) {
          if (rhsView.type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, rhsView.type().scale()};
            rhsCast = cudf::cast(rhsView, castType, stream, mr);
            rhsView = rhsCast->view();
          }
          if (left_->type().id() == cudf::type_id::DECIMAL64) {
            auto castType = cudf::data_type{
                cudf::type_id::DECIMAL128, left_->type().scale()};
            lhsScalar = castDecimalScalar(*left_, castType, stream, mr);
          }
        }
        return cudf::binary_operation(
            lhsScalar ? *lhsScalar : *left_, rhsView, op_, type_, stream, mr);
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

// @TODO 4/22/26
// Simplify or remove the logic in this class that handles constant-folding or
// short-circuiting of logical operations, once the cuDF expression optimizer
// enhancements land (Velox PR #17108, see also Velox Issue #17307).
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
        VELOX_CHECK_EQ(
            constExpr->value()->typeKind(),
            TypeKind::BOOLEAN,
            "Logical function only supports boolean literals");
        auto boolConst = constExpr->value()->as<ConstantVector<bool>>();
        VELOX_CHECK_NOT_NULL(boolConst);
        if (!shortCircuitScalar_ && !boolConst->isNullAt(0)) {
          const bool v = boolConst->valueAt(0);
          if ((op_ == cudf::binary_operator::NULL_LOGICAL_AND && !v) ||
              (op_ == cudf::binary_operator::NULL_LOGICAL_OR && v)) {
            // If we encounter non-null false (for AND) or true (for OR), we
            // know what the final result must be, although it will still need
            // to be expanded to a column the same size as the input columns. No
            // need to continue capturing literals in that case.
            shortCircuitScalar_ =
                createCudfScalar<TypeKind::BOOLEAN>(constExpr->value());
            break;
          }
        }
        literals_.push_back(
            createCudfScalar<TypeKind::BOOLEAN>(constExpr->value()));
      } else {
        literals_.push_back(nullptr);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // If there are no input columns, the result is a scalar.
    const size_t rowCount =
        inputColumns.empty() ? 1 : asView(inputColumns[0]).size();

    // If we determined a short-circuit result in the constructor, we
    // return it directly here, expanded to the size of the input
    // columns if not all the inputs are literals.
    if (shortCircuitScalar_) {
      return cudf::make_column_from_scalar(
          *shortCircuitScalar_, rowCount, stream, mr);
    }

    // Now build the vector of actual operands, each of which is either a
    // pre-computed literal or an input column.
    struct Operand {
      const cudf::scalar* scalar;
      cudf::column_view column;
    };
    std::vector<Operand> operands;
    operands.reserve(literals_.size());
    size_t columnIndex = 0;
    for (const auto& literal : literals_) {
      if (literal) {
        operands.push_back(Operand{literal.get(), {}});
      } else {
        VELOX_CHECK_LT(columnIndex, inputColumns.size());
        operands.push_back(
            Operand{nullptr, asView(inputColumns[columnIndex++])});
      }
    }

    // There must be at least one operand.
    VELOX_CHECK(!operands.empty());

    // If there is only one operand, we can return it directly,
    // again expanded to the size of the input columns if needed.
    if (operands.size() == 1) {
      const auto& only = operands[0];
      if (only.scalar) {
        return cudf::make_column_from_scalar(
            *only.scalar, rowCount, stream, mr);
      }
      return ColumnOrView(only.column);
    }

    // If we get this far, we have at least two operands. We can
    // now compute the result by iterating over the operands and
    // applying the binary operator to each pair of operands.
    const auto& left = operands[0];
    const auto& right = operands[1];
    std::unique_ptr<cudf::column> result;
    if (left.scalar && right.scalar) {
      // This case may still happen even in the case where a short-circuit
      // result was not determined in the constructor, for example, if the
      // inputs are 'true OR true' or 'false AND false'.
      auto tmp =
          cudf::make_column_from_scalar(*left.scalar, rowCount, stream, mr);
      result = cudf::binary_operation(
          tmp->view(), *right.scalar, op_, kBoolType, stream, mr);
    } else if (left.scalar) {
      result = cudf::binary_operation(
          *left.scalar, right.column, op_, kBoolType, stream, mr);
    } else if (right.scalar) {
      result = cudf::binary_operation(
          left.column, *right.scalar, op_, kBoolType, stream, mr);
    } else {
      result = cudf::binary_operation(
          left.column, right.column, op_, kBoolType, stream, mr);
    }
    for (size_t i = 2; i < operands.size(); ++i) {
      const auto& next = operands[i];
      if (next.scalar) {
        result = cudf::binary_operation(
            result->view(), *next.scalar, op_, kBoolType, stream, mr);
      } else {
        result = cudf::binary_operation(
            result->view(), next.column, op_, kBoolType, stream, mr);
      }
    }
    return result;
  }

 private:
  static constexpr cudf::data_type kBoolType{cudf::type_id::BOOL8};
  const cudf::binary_operator op_;
  std::unique_ptr<cudf::scalar> shortCircuitScalar_;
  std::vector<std::unique_ptr<cudf::scalar>> literals_;
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

class NullPredicateFunction : public CudfFunction {
 public:
  NullPredicateFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      bool negate)
      : negate_(negate) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "Null predicate expects exactly 1 input");
    auto constExpr =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NULL(
        constExpr, "Null predicate on literal input is not supported");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto result = cudf::is_null(asView(inputColumns[0]), stream, mr);
    if (!negate_) {
      return result;
    }
    return cudf::unary_operation(
        result->view(), cudf::unary_operator::NOT, stream, mr);
  }

 private:
  const bool negate_;
};

class GetStructFieldFunction : public CudfFunction {
 public:
  explicit GetStructFieldFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "get_struct_field expects exactly 2 inputs");
    auto fieldExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        fieldExpr, "get_struct_field field index must be a constant");
    VELOX_CHECK(
        !fieldExpr->value()->isNullAt(0),
        "get_struct_field field index must be non-null");

    fieldIndex_ = static_cast<cudf::size_type>(
        std::stoll(fieldExpr->value()->toString(0)));
    VELOX_CHECK_GE(fieldIndex_, 0);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "get_struct_field expects exactly one non-literal input column");
    return FunctionExpression::makeStructChildColumn(
        inputColumns[0], fieldIndex_, stream, mr);
  }

 private:
  cudf::size_type fieldIndex_{0};
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

template <TypeKind Kind>
static VectorPtr foldConstantPair(
    const VectorPtr& a,
    const VectorPtr& b,
    cudf::binary_operator op) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (a->isNullAt(0))
    return b;
  if (b->isNullAt(0))
    return a;
  auto aVal = a->as<ConstantVector<T>>()->value();
  auto bVal = b->as<ConstantVector<T>>()->value();
  bool bWins =
      (op == cudf::binary_operator::NULL_MAX) ? (bVal > aVal) : (bVal < aVal);
  return bWins ? b : a;
}

class GreatestLeastFunction : public CudfFunction {
 public:
  GreatestLeastFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::binary_operator op)
      : op_(op), type_(cudf_velox::veloxToCudfDataType(expr->type())) {
    VELOX_CHECK_GE(
        expr->inputs().size(),
        2,
        "Greatest/Least function expects at least 2 inputs");
    // Separate column inputs (into order_) from constant inputs (folded into
    // a single scalar). Column indices refer to positions in the packed
    // inputColumns vector that eval() receives (which excludes literals).
    std::vector<VectorPtr> constValues;
    size_t columnIndex = 0;
    for (const auto& input : expr->inputs()) {
      if (auto constExpr =
              std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input)) {
        constValues.push_back(constExpr->value());
      } else {
        order_.push_back(columnIndex++);
      }
    }
    // Fold all constant values into a single scalar on the host.
    if (!constValues.empty()) {
      auto winner = constValues[0];
      for (size_t i = 1; i < constValues.size(); ++i) {
        winner = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            foldConstantPair, winner->typeKind(), winner, constValues[i], op);
      }
      foldedScalar_ = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createCudfScalar, winner->typeKind(), winner);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // All inputs were constant -- return the pre-folded scalar as a column.
    if (order_.empty()) {
      return cudf::make_column_from_scalar(*foldedScalar_, 1, stream, mr);
    }

    // Accumulate across column inputs.
    std::unique_ptr<cudf::column> result;
    for (size_t i = 1; i < order_.size(); ++i) {
      cudf::column_view lhs =
          result ? result->view() : asView(inputColumns[order_[0]]);
      result = cudf::binary_operation(
          lhs, asView(inputColumns[order_[i]]), op_, type_, stream, mr);
    }

    // Apply the folded constant as a final (column, scalar) operation.
    if (foldedScalar_) {
      cudf::column_view lhs =
          result ? result->view() : asView(inputColumns[order_[0]]);
      result =
          cudf::binary_operation(lhs, *foldedScalar_, op_, type_, stream, mr);
    }
    return result;
  }

 private:
  const cudf::binary_operator op_;
  const cudf::data_type type_;
  std::unique_ptr<cudf::scalar> foldedScalar_;
  std::vector<size_t> order_;
};

bool isSwitchExpressionSupported(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto& inputs = expr->inputs();
  if (inputs.size() < 3 || inputs.size() % 2 == 0) {
    return false;
  }

  const auto& resultType = expr->type();
  if (resultType == nullptr) {
    return false;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->type() == nullptr) {
      return false;
    }
    const bool isCondition = (i % 2 == 0) && (i + 1 < inputs.size());
    if (isCondition) {
      if (inputs[i]->type()->kind() != TypeKind::BOOLEAN) {
        return false;
      }
    } else if (!inputs[i]->type()->equivalent(*resultType)) {
      return false;
    }

    const auto constant =
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(inputs[i]);
    const auto isNullArrayLiteral = constant != nullptr &&
        constant->value() != nullptr && constant->value()->isNullAt(0) &&
        inputs[i]->type()->kind() == TypeKind::ARRAY;
    const auto isEmptyArrayLiteral =
        isNonNullEmptyArrayConstantLiteral(inputs[i]);
    if (isUnsupportedConstantLiteral(inputs[i]) && !isNullArrayLiteral &&
        !isEmptyArrayLiteral) {
      return false;
    }
  }

  return true;
}

class SwitchFunction : public CudfFunction {
 public:
  SwitchFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK(isSwitchExpressionSupported(expr));
    const auto& inputs = expr->inputs();
    VELOX_CHECK_GE(inputs.size(), 3, "case when expects at least 3 inputs");
    VELOX_CHECK_EQ(
        inputs.size() % 2,
        1,
        "case when expects condition/value pairs followed by else");

    size_t columnIndex = 0;
    operands_.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const bool isCondition = (i % 2 == 0) && (i + 1 < inputs.size());
      if (isCondition) {
        VELOX_CHECK_EQ(
            inputs[i]->type()->kind(),
            TypeKind::BOOLEAN,
            "The switch condition result type should be boolean");
      } else {
        VELOX_CHECK(
            inputs[i]->type()->equivalent(*expr->type()),
            "The switch branch type should match the result type");
      }

      if (auto constExpr =
              std::dynamic_pointer_cast<velox::exec::ConstantExpr>(inputs[i])) {
        auto constValue = constExpr->value();
        VELOX_CHECK_NOT_NULL(constValue);
        if (constValue->isNullAt(0) &&
            inputs[i]->type()->kind() == TypeKind::ARRAY) {
          operands_.push_back(Operand{nullptr, inputs[i]->type(), nullptr, 0});
        } else if (isNonNullEmptyArrayConstantLiteral(inputs[i])) {
          operands_.push_back(Operand{nullptr, nullptr, inputs[i]->type(), 0});
        } else {
          operands_.push_back(
              Operand{
                  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
                      createCudfScalar, constValue->typeKind(), constValue),
                  nullptr,
                  nullptr,
                  0});
        }
      } else {
        operands_.push_back(Operand{nullptr, nullptr, nullptr, columnIndex++});
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_GE(operands_.size(), 3);
    const size_t rowCount =
        inputColumns.empty() ? 1 : asView(inputColumns[0]).size();
    std::vector<std::unique_ptr<cudf::column>> materializedScalars;
    materializedScalars.reserve(operands_.size());

    auto operandView = [&](const Operand& operand) -> cudf::column_view {
      if (operand.scalar) {
        materializedScalars.push_back(
            cudf::make_column_from_scalar(
                *operand.scalar, rowCount, stream, mr));
        return materializedScalars.back()->view();
      }
      if (operand.nullArrayType) {
        materializedScalars.push_back(makeAllNullArrayColumn(
            operand.nullArrayType, rowCount, stream, mr));
        return materializedScalars.back()->view();
      }
      if (operand.emptyArrayType) {
        auto scalar = makeEmptyArrayScalar(operand.emptyArrayType, stream, mr);
        materializedScalars.push_back(
            cudf::make_column_from_scalar(*scalar, rowCount, stream, mr));
        return materializedScalars.back()->view();
      }
      VELOX_CHECK_LT(operand.columnIndex, inputColumns.size());
      return asView(inputColumns[operand.columnIndex]);
    };

    std::unique_ptr<cudf::column> result;
    for (int i = static_cast<int>(operands_.size()) - 3; i >= 0; i -= 2) {
      auto condition = operandView(operands_[i]);
      auto thenValue = operandView(operands_[i + 1]);
      auto elseValue = result ? result->view() : operandView(operands_.back());
      result = cudf::copy_if_else(thenValue, elseValue, condition, stream, mr);
    }
    return result;
  }

 private:
  struct Operand {
    std::unique_ptr<cudf::scalar> scalar;
    TypePtr nullArrayType;
    TypePtr emptyArrayType;
    size_t columnIndex;
  };

  std::vector<Operand> operands_;
};

class SubstrFunction : public CudfFunction {
 public:
  SubstrFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "substr expects at least 2 inputs");
    VELOX_CHECK_LE(expr->inputs().size(), 3, "substr expects at most 3 inputs");

    auto startExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(startExpr, "substr start must be a constant");

    auto startValue = readConstantIntegralValue(*startExpr);
    start_ = static_cast<cudf::size_type>(startValue);
    if (startValue >= 1) {
      // cuDF indexing starts at 0.
      // Presto indexing starts at 1.
      // Positive indices need to substract 1.
      start_ = static_cast<cudf::size_type>(startValue - 1);
    }

    if (expr->inputs().size() > 2) {
      auto lengthExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(lengthExpr, "substr length must be a constant");

      auto lengthValue = readConstantIntegralValue(*lengthExpr);
      // cuDF uses indices [begin, end).
      // Presto uses length as the length of the substring.
      // We compute the end as start + length.
      end_ = start_ + static_cast<cudf::size_type>(lengthValue);
      hasEnd_ = true;
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    cudf::numeric_scalar<cudf::size_type> startScalar(start_, true, stream, mr);
    cudf::numeric_scalar<cudf::size_type> endScalar(
        hasEnd_ ? end_ : 0, hasEnd_, stream, mr);
    cudf::numeric_scalar<cudf::size_type> stepScalar(1, true, stream, mr);
    return cudf::strings::slice_strings(
        inputCol, startScalar, endScalar, stepScalar, stream, mr);
  }

 private:
  cudf::size_type start_{0};
  cudf::size_type end_{0};
  bool hasEnd_{false};
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
          if (isNonNullEmptyArrayConstantLiteral(c)) {
            emptyArrayLiteralType_ = c->type();
          } else {
            literalScalar_ = makeScalarFromConstantExpr(c);
          }
          numColumnsBeforeLiteral_ = i;
          break;
        }
      } else if (input->distinctFields().empty() && !input->inputs().empty()) {
        // Handle constant expressions that weren't folded (e.g., cast of
        // literal).
        if (auto innerConst =
                std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
                    input->inputs()[0])) {
          if (!innerConst->value()->isNullAt(0)) {
            if (isNonNullEmptyArrayConstantLiteral(innerConst)) {
              emptyArrayLiteralType_ = innerConst->type();
            } else {
              literalScalar_ = makeScalarFromConstantExpr(innerConst);
            }
            numColumnsBeforeLiteral_ = i;
            break;
          }
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
    if ((literalScalar_ || emptyArrayLiteralType_) &&
        numColumnsBeforeLiteral_ == 0) {
      if (inputColumns.empty()) {
        // We need at least one column to tell us the required output size
        VELOX_NYI("coalesce with only literal inputs is not supported");
      }
      auto size = asView(inputColumns[0]).size();
      auto scalar = emptyArrayLiteralType_
          ? makeEmptyArrayScalar(emptyArrayLiteralType_, stream, mr)
          : nullptr;
      return cudf::make_column_from_scalar(
          scalar ? *scalar : *literalScalar_, size, stream, mr);
    }

    VELOX_CHECK(
        !inputColumns.empty(),
        "coalesce requires at least one non-literal input");
    ColumnOrView result = asView(inputColumns[0]);
    size_t stop = std::min(numColumnsBeforeLiteral_, inputColumns.size());
    for (size_t i = 1; i < stop && asView(result).has_nulls(); ++i) {
      const auto current = asView(result);
      if (current.type().id() == cudf::type_id::LIST ||
          current.type().id() == cudf::type_id::STRUCT) {
        // libcudf replace_nulls does not specialize nested columns. Select
        // whole rows instead, preserving child nulls inside a valid map/list/
        // struct value while replacing only null parent rows.
        auto nullRows = cudf::is_null(current, stream, mr);
        result = cudf::copy_if_else(
            asView(inputColumns[i]), current, nullRows->view(), stream, mr);
      } else {
        result =
            cudf::replace_nulls(current, asView(inputColumns[i]), stream, mr);
      }
    }

    if ((literalScalar_ || emptyArrayLiteralType_) &&
        asView(result).has_nulls()) {
      auto scalar = emptyArrayLiteralType_
          ? makeEmptyArrayScalar(emptyArrayLiteralType_, stream, mr)
          : nullptr;
      const auto current = asView(result);
      if (current.type().id() == cudf::type_id::LIST ||
          current.type().id() == cudf::type_id::STRUCT) {
        auto nullRows = cudf::is_null(current, stream, mr);
        result = cudf::copy_if_else(
            scalar ? *scalar : *literalScalar_,
            current,
            nullRows->view(),
            stream,
            mr);
      } else {
        result = cudf::replace_nulls(
            current, scalar ? *scalar : *literalScalar_, stream, mr);
      }
    }

    return result;
  }

 private:
  size_t numColumnsBeforeLiteral_;
  std::unique_ptr<cudf::scalar> literalScalar_;
  TypePtr emptyArrayLiteralType_;
};

class ExtractComponentFunction : public CudfFunction {
 public:
  ExtractComponentFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      cudf::datetime::datetime_component component)
      : component_(component) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "extract expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_datetime_component(
        inputCol, component_, stream, mr);
  }

 private:
  cudf::datetime::datetime_component component_;
};

// Builds an ExtractComponentFunction for a fixed datetime component, avoiding a
// near-identical registration lambda for every component (year, month, ...).
struct ExtractComponentFactory {
  cudf::datetime::datetime_component component;

  std::shared_ptr<CudfFunction> operator()(
      const std::string&,
      const std::shared_ptr<velox::exec::Expr>& expr) const {
    return std::make_shared<ExtractComponentFunction>(expr, component);
  }
};

class QuarterFunction : public CudfFunction {
 public:
  explicit QuarterFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "quarter expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::extract_quarter(inputCol, stream, mr);
  }
};

class DayOfYearFunction : public CudfFunction {
 public:
  explicit DayOfYearFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "day_of_year expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::day_of_year(inputCol, stream, mr);
  }
};

class WeekFunction : public CudfFunction {
 public:
  explicit WeekFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "week expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto weekStrings = cudf::strings::from_timestamps(
        inputCol, "%V", cudf::strings_column_view{}, stream, mr);
    return cudf::strings::to_integers(
        cudf::strings_column_view(weekStrings->view()),
        cudf::data_type(cudf::type_id::INT32),
        stream,
        mr);
  }
};

class YearOfWeekFunction : public CudfFunction {
 public:
  explicit YearOfWeekFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        1,
        "year_of_week expects exactly 1 input column");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    auto yearStrings = cudf::strings::from_timestamps(
        inputCol, "%G", cudf::strings_column_view{}, stream, mr);
    return cudf::strings::to_integers(
        cudf::strings_column_view(yearStrings->view()),
        cudf::data_type(cudf::type_id::INT32),
        stream,
        mr);
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
        "like expects 2 or 3 inputs");

    if (auto inputExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0])) {
      inputIsConstant_ = true;
      inputIsNull_ = inputExpr->value()->isNullAt(0);
      if (!inputIsNull_) {
        input_ = inputExpr->value()->toString(0);
      }
    }

    if (auto patternExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1])) {
      patternIsConstant_ = true;
      patternIsNull_ = patternExpr->value()->isNullAt(0);
      if (!patternIsNull_) {
        pattern_ = patternExpr->value()->toString(0);
      }
    }

    VELOX_CHECK(
        !(inputIsConstant_ && patternIsConstant_),
        "like with constant input and pattern is not supported by the cuDF "
        "evaluator because there is no input column to derive the output row "
        "count from");

    hasEscape_ = expr->inputs().size() == 3;
    if (expr->inputs().size() == 3) {
      auto escapeExpr =
          std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[2]);
      VELOX_CHECK_NOT_NULL(escapeExpr, "like escape must be a constant");
      escapeIsNull_ = escapeExpr->value()->isNullAt(0);
      if (!escapeIsNull_) {
        escape_ = escapeExpr->value()->toString(0);
        if (!patternIsConstant_ && escape_.size() == 1) {
          // Column-pattern LIKE ESCAPE reuses the same three legal escape
          // sequences for every batch, so cache those tiny helper columns once
          // here. Constant patterns are validated on the host and don't need
          // these columns.
          auto stream = cudf::get_default_stream(cudf::allow_default_stream);
          auto mr = get_temp_mr();
          targetsColumn_ = makeEscapeTargetsColumn(escape_[0], stream, mr);
          replacementsColumn_ = makeEscapeReplacementsColumn(stream, mr);
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    size_t nextInput = 0;
    VELOX_CHECK(
        !inputColumns.empty(),
        "like requires at least one non-literal input column");
    // inputColumns only contains non-literal children, so the first entry
    // determines the output row count even when the input itself is constant.
    auto outputRowCount = asView(inputColumns[0]).size();

    std::unique_ptr<cudf::column> inputColumnHolder;
    cudf::column_view inputCol;
    if (inputIsConstant_) {
      cudf::string_scalar inputScalar(input_, !inputIsNull_, stream, mr);
      inputColumnHolder = cudf::make_column_from_scalar(
          inputScalar, outputRowCount, stream, mr);
      inputCol = inputColumnHolder->view();
    } else {
      inputCol = asView(inputColumns[nextInput++]);
    }

    auto makeAllNullResult = [&]() {
      // Match Velox CPU null propagation for constant null pattern/escape
      // inputs by producing a fully null boolean column.
      auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);
      return cudf::make_column_from_scalar(
          nullScalar, inputCol.size(), stream, mr);
    };

    if (patternIsConstant_ && patternIsNull_) {
      return makeAllNullResult();
    }
    if (hasEscape_ && escapeIsNull_) {
      return makeAllNullResult();
    }

    char escapeChar{0};
    if (hasEscape_) {
      VELOX_USER_CHECK_EQ(
          escape_.size(), 1, "Escape string must be a single character");
      escapeChar = escape_[0];
    }

    if (patternIsConstant_) {
      if (hasEscape_) {
        validateConstantPattern(pattern_, escapeChar);
      }
      return cudf::strings::like(inputCol, pattern_, escape_, stream, mr);
    }

    auto patternCol = asView(inputColumns[nextInput]);
    if (hasEscape_) {
      validatePatternColumn(patternCol, escapeChar, stream, mr);
    }

    std::unique_ptr<cudf::column> sanitizedPatternHolder;
    auto sanitizedPattern = patternCol;
    if (patternCol.has_nulls()) {
      // libcudf rejects null pattern rows for column/column LIKE. Replace them
      // only for the LIKE call after validation so the temporary column
      // overlaps less with the validation intermediates above.
      cudf::string_scalar emptyPattern("", true, stream, mr);
      sanitizedPatternHolder =
          cudf::replace_nulls(patternCol, emptyPattern, stream, mr);
      sanitizedPattern = sanitizedPatternHolder->view();
    }

    cudf::string_scalar escapeScalar(escape_, true, stream, mr);
    auto result = cudf::strings::like(
        inputCol, sanitizedPattern, escapeScalar, stream, mr);
    sanitizedPatternHolder.reset();
    // Velox returns null if either the input or pattern row is null. cuDF
    // already propagated input nulls into the result, so only merge the pattern
    // nulls back when needed.
    mergeNullSourceNullsIntoResult(*result, patternCol, stream, mr);
    return result;
  }

 private:
  static void validateConstantPattern(std::string_view pattern, char escape);

  static std::unique_ptr<cudf::column> makeEscapeTargetsColumn(
      char escape,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  static std::unique_ptr<cudf::column> makeEscapeReplacementsColumn(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  void validatePatternColumn(
      cudf::column_view patternColumn,
      char escape,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  bool inputIsConstant_{false};
  bool inputIsNull_{false};
  bool patternIsConstant_{false};
  bool patternIsNull_{false};
  bool hasEscape_{false};
  bool escapeIsNull_{false};
  std::string input_;
  std::string pattern_;
  std::string escape_;
  std::unique_ptr<cudf::column> targetsColumn_;
  std::unique_ptr<cudf::column> replacementsColumn_;
};

void LikeFunction::validateConstantPattern(
    std::string_view pattern,
    char escape) {
  // Match Velox CPU invalid escape validation before calling libcudf LIKE.
  for (size_t index = 0; index < pattern.size(); ++index) {
    if (pattern[index] != escape) {
      continue;
    }

    VELOX_USER_CHECK_LT(
        index + 1,
        pattern.size(),
        "Escape character must be followed by '%', '_' or the escape character itself");
    auto next = pattern[index + 1];
    if (next != escape && next != '%' && next != '_') {
      VELOX_USER_FAIL(
          "Escape character must be followed by '%', '_' or the escape character itself");
    }
    ++index;
  }
}

std::unique_ptr<cudf::column> LikeFunction::makeEscapeTargetsColumn(
    char escape,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Build the three legal escape sequences so column-pattern LIKE ESCAPE can
  // strip them before checking whether any invalid escape usage remains. This
  // keeps cuDF aligned with Velox CPU pattern validation semantics.
  cudf::string_scalar escapedEscapeScalar(
      std::string(2, escape), true, stream, mr);
  cudf::string_scalar escapedPercentScalar(
      std::string(1, escape) + '%', true, stream, mr);
  cudf::string_scalar escapedUnderscoreScalar(
      std::string(1, escape) + '_', true, stream, mr);
  auto escapedEscapeView = escapedEscapeScalar.value(stream);
  auto escapedPercentView = escapedPercentScalar.value(stream);
  auto escapedUnderscoreView = escapedUnderscoreScalar.value(stream);
  rmm::device_uvector<cudf::string_view> deviceTargetViews(3, stream, mr);
  deviceTargetViews.set_element_async(0, escapedEscapeView, stream);
  deviceTargetViews.set_element_async(1, escapedPercentView, stream);
  deviceTargetViews.set_element_async(2, escapedUnderscoreView, stream);
  auto targetsColumn = cudf::make_strings_column(
      cudf::device_span<cudf::string_view const>{deviceTargetViews},
      cudf::string_view{nullptr, 0},
      stream,
      mr);
  // The temporary scalars and string_view array above back async work used to
  // build the output column, so wait for the stream before returning.
  stream.synchronize();
  return targetsColumn;
}

std::unique_ptr<cudf::column> LikeFunction::makeEscapeReplacementsColumn(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  cudf::string_scalar emptyString("", true, stream, mr);
  auto replacementsColumn =
      cudf::make_column_from_scalar(emptyString, 1, stream, mr);
  // make_column_from_scalar(string_scalar) reads the scalar's device string
  // data asynchronously, so keep the scalar alive until the stream completes.
  stream.synchronize();
  return replacementsColumn;
}

void LikeFunction::validatePatternColumn(
    cudf::column_view patternColumn,
    char escape,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_NOT_NULL(targetsColumn_);
  VELOX_CHECK_NOT_NULL(replacementsColumn_);

  // Remove the three legal escape forms first. Any remaining escape character
  // must be dangling or followed by an unsupported character, which matches the
  // Velox CPU invalid-escape check.
  auto normalized = cudf::strings::replace_multiple(
      cudf::strings_column_view(patternColumn),
      cudf::strings_column_view(targetsColumn_->view()),
      cudf::strings_column_view(replacementsColumn_->view()),
      stream,
      mr);

  auto escapeScalar =
      cudf::string_scalar(std::string(1, escape), true, stream, mr);
  auto hasDanglingEscape = cudf::strings::contains(
      cudf::strings_column_view(normalized->view()), escapeScalar, stream, mr);
  normalized.reset();

  auto anyAggregation = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto invalidScalar = cudf::reduce(
      hasDanglingEscape->view(),
      *anyAggregation,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  hasDanglingEscape.reset();

  auto const& invalidBool =
      static_cast<cudf::numeric_scalar<bool> const&>(*invalidScalar);
  auto hasInvalidEscapeUsageValue =
      invalidBool.is_valid(stream) && invalidBool.value(stream);
  invalidScalar.reset();

  VELOX_USER_CHECK(
      !hasInvalidEscapeUsageValue,
      "Escape character must be followed by '%', '_' or the escape character itself");
}

class StringPatternPredicateFunction : public CudfFunction {
 public:
  explicit StringPatternPredicateFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      std::string_view functionName) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "{} expects 2 inputs", functionName);

    if (auto inputExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0])) {
      inputIsConstant_ = true;
      inputIsNull_ = inputExpr->value()->isNullAt(0);
      if (!inputIsNull_) {
        input_ = inputExpr->value()->toString(0);
      }
    }

    if (auto patternExpr =
            std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1])) {
      patternIsConstant_ = true;
      patternIsNull_ = patternExpr->value()->isNullAt(0);
      if (!patternIsNull_) {
        pattern_ = patternExpr->value()->toString(0);
      }
    }

    // Fully constant string-match calls stay off the cuDF path because the
    // evaluator has no input column to derive the output row count from.
    VELOX_CHECK(
        !(inputIsConstant_ && patternIsConstant_),
        "{} with two constant inputs is not supported by the cuDF evaluator",
        functionName);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    size_t nextInput = 0;
    auto rowCount = inputColumns.empty() ? vector_size_t{1}
                                         : asView(inputColumns[0]).size();

    std::unique_ptr<cudf::column> inputColumnHolder;
    cudf::column_view inputCol;
    if (inputIsConstant_) {
      cudf::string_scalar inputScalar(input_, !inputIsNull_, stream, mr);
      inputColumnHolder =
          cudf::make_column_from_scalar(inputScalar, rowCount, stream, mr);
      inputCol = inputColumnHolder->view();
    } else {
      inputCol = asView(inputColumns[nextInput++]);
    }

    if (patternIsConstant_) {
      if (patternIsNull_) {
        auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);
        return cudf::make_column_from_scalar(
            nullScalar, inputCol.size(), stream, mr);
      }
      cudf::string_scalar patternScalar(pattern_, true, stream, mr);
      return evaluateMatch(inputCol, patternScalar, stream, mr);
    }

    auto patternCol = asView(inputColumns[nextInput]);
    auto result = evaluateMatch(inputCol, patternCol, stream, mr);
    // Match Velox CPU null propagation for column/column evaluation: libcudf
    // can return a valid false when the pattern row is null, but Velox returns
    // null if either side is null. cuDF already propagated input nulls into the
    // result, so only merge the pattern nulls back when needed.
    mergeNullSourceNullsIntoResult(*result, patternCol, stream, mr);
    return result;
  }

 protected:
  virtual std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const = 0;

  virtual std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const = 0;

  bool inputIsConstant_{false};
  bool inputIsNull_{false};
  bool patternIsNull_{false};
  bool patternIsConstant_{false};
  std::string input_;
  std::string pattern_;
};

class StartswithFunction : public StringPatternPredicateFunction {
 public:
  explicit StartswithFunction(const std::shared_ptr<velox::exec::Expr>& expr)
      : StringPatternPredicateFunction(expr, "startswith") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::starts_with(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::starts_with(inputCol, patternCol, stream, mr);
  }
};

class EndswithFunction : public StringPatternPredicateFunction {
 public:
  explicit EndswithFunction(const std::shared_ptr<velox::exec::Expr>& expr)
      : StringPatternPredicateFunction(expr, "endswith") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::ends_with(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::ends_with(inputCol, patternCol, stream, mr);
  }
};

class ContainsFunction : public StringPatternPredicateFunction {
 public:
  explicit ContainsFunction(const std::shared_ptr<velox::exec::Expr>& expr)
      : StringPatternPredicateFunction(expr, "contains") {}

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const& patternScalar,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::contains(inputCol, patternScalar, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::column_view patternCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return cudf::strings::contains(inputCol, patternCol, stream, mr);
  }
};

class RLikeFunction : public StringPatternPredicateFunction {
 public:
  explicit RLikeFunction(const std::shared_ptr<velox::exec::Expr>& expr)
      : StringPatternPredicateFunction(expr, "rlike") {
    VELOX_CHECK(patternIsConstant_, "cuDF rlike requires a constant pattern");
  }

 protected:
  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view inputCol,
      cudf::string_scalar const&,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto regexProgram = cudf::strings::regex_program::create(pattern_);
    return cudf::strings::contains_re(
        cudf::strings_column_view(inputCol), *regexProgram, stream, mr);
  }

  std::unique_ptr<cudf::column> evaluateMatch(
      cudf::column_view,
      cudf::column_view,
      rmm::cuda_stream_view,
      rmm::device_async_resource_ref) const override {
    VELOX_UNSUPPORTED("cuDF rlike requires a constant pattern");
  }
};

class ConcatFunction : public CudfFunction {
 public:
  explicit ConcatFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    numInputs_ = expr->inputs().size();
    VELOX_CHECK_GE(numInputs_, 2, "concat expects at least 2 inputs");

    // Scan inputs for literals and store strings in map by input index.
    for (size_t i = 0; i < numInputs_; ++i) {
      if (auto constant =
              std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[i])) {
        inputIndexToLiteral_[i] = constant->value()->toString(0);
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    // Validate sizes.
    VELOX_CHECK_EQ(
        inputColumns.size() + inputIndexToLiteral_.size(),
        numInputs_,
        "Unexpected number of input columns");

    // If there is at least one input column, fetch its size as the output size.
    // If there are no input columns, this means that all the inputs are
    // literals, and the output size will be 1.
    const size_t outputSize =
        inputColumns.empty() ? 1u : asView(inputColumns[0]).size();

    // Iterate the inputs, building a vector of column views, either a literal
    // from the map, or the next input column. We also keep a vector of the
    // columns created for literals, so that they persist while their views
    // are used in the concatenation.
    std::vector<cudf::column_view> columnViews;
    std::vector<std::unique_ptr<cudf::column>> literalColumns;
    size_t nextInputColumnIndex = 0u;
    for (size_t i = 0; i < numInputs_; ++i) {
      auto it = inputIndexToLiteral_.find(i);
      if (it == inputIndexToLiteral_.end()) {
        // No literal for this input. Use the next input column.
        auto& column = inputColumns[nextInputColumnIndex++];
        columnViews.push_back(asView(column));
      } else {
        // Create a column of the literal repeated for the entire output size.
        auto const& literal = it->second;
        cudf::string_scalar scalar(literal, true, stream, mr);
        auto col =
            cudf::make_column_from_scalar(scalar, outputSize, stream, mr);
        columnViews.push_back(col->view());
        literalColumns.emplace_back(std::move(col));
      }
    }

    // Concatenate the columns, nulls as empty strings, no separators.
    cudf::string_scalar emptyString("", true, stream, mr);
    return cudf::strings::concatenate(
        cudf::table_view(columnViews),
        emptyString,
        emptyString,
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
  }

 private:
  std::map<int, std::string> inputIndexToLiteral_;
  size_t numInputs_{0};
};

class ConcatWsFunction : public CudfFunction {
 public:
  explicit ConcatWsFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    numInputs_ = expr->inputs().size();
    VELOX_CHECK_GE(numInputs_, 1, "concat_ws expects at least 1 input");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::VARCHAR,
        "concat_ws separator must be VARCHAR");
    VELOX_CHECK_EQ(
        expr->type()->kind(),
        TypeKind::VARCHAR,
        "concat_ws output must be VARCHAR");

    auto separatorExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(
        separatorExpr, "cuDF concat_ws requires a constant separator");
    auto separatorValue = separatorExpr->value();
    VELOX_CHECK_NOT_NULL(
        separatorValue, "concat_ws separator value is missing");
    if (separatorValue->isNullAt(0)) {
      nullSeparator_ = true;
    } else {
      separator_ = separatorValue->toString(0);
    }

    literals_.reserve(numInputs_ - 1);
    for (size_t i = 1; i < numInputs_; ++i) {
      const auto& inputType = expr->inputs()[i]->type();
      const auto inputKind = inputType->kind();
      VELOX_CHECK(
          inputKind == TypeKind::VARCHAR ||
              (inputKind == TypeKind::ARRAY &&
               inputType->childAt(0)->kind() == TypeKind::VARCHAR),
          "cuDF concat_ws only supports VARCHAR or ARRAY<VARCHAR> value inputs");
      auto& literal = literals_.emplace_back();
      literal.isArray = inputKind == TypeKind::ARRAY;
      if (auto constant =
              std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[i])) {
        VELOX_CHECK(
            !literal.isArray,
            "cuDF concat_ws does not support literal ARRAY<VARCHAR> inputs yet");
        literal.isLiteral = true;
        auto value = constant->value();
        VELOX_CHECK_NOT_NULL(value, "concat_ws literal value is missing");
        if (value->isNullAt(0)) {
          literal.isNull = true;
        } else {
          literal.value = value->toString(0);
        }
      }
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    const auto literalCount =
        std::count_if(literals_.begin(), literals_.end(), [](const auto& arg) {
          return arg.isLiteral;
        });
    VELOX_CHECK_EQ(
        inputColumns.size() + literalCount,
        numInputs_ - 1,
        "Unexpected number of concat_ws value columns");

    const auto outputSize =
        inputColumns.empty() ? 1u : asView(inputColumns[0]).size();
    if (nullSeparator_) {
      cudf::string_scalar nullString("", false, stream, mr);
      auto result =
          cudf::make_column_from_scalar(nullString, outputSize, stream, mr);
      stream.synchronize();
      return result;
    }

    cudf::string_scalar emptyString("", true, stream, mr);
    cudf::string_scalar separator(separator_, true, stream, mr);
    if (literals_.empty()) {
      return cudf::make_column_from_scalar(emptyString, outputSize, stream, mr);
    }

    std::vector<cudf::column_view> columnViews;
    std::vector<std::unique_ptr<cudf::column>> literalColumns;
    columnViews.reserve(literals_.size());
    literalColumns.reserve(literals_.size());
    size_t nextInputColumnIndex = 0;

    for (const auto& literal : literals_) {
      if (!literal.isLiteral) {
        VELOX_CHECK_LT(nextInputColumnIndex, inputColumns.size());
        auto inputView = asView(inputColumns[nextInputColumnIndex++]);
        if (literal.isArray) {
          auto joined = cudf::strings::join_list_elements(
              cudf::lists_column_view(inputView),
              separator,
              emptyString,
              cudf::strings::separator_on_nulls::NO,
              cudf::strings::output_if_empty_list::EMPTY_STRING,
              stream,
              mr);
          columnViews.push_back(joined->view());
          literalColumns.push_back(std::move(joined));
        } else {
          columnViews.push_back(inputView);
        }
        continue;
      }

      cudf::string_scalar scalar(literal.value, !literal.isNull, stream, mr);
      auto column =
          cudf::make_column_from_scalar(scalar, outputSize, stream, mr);
      columnViews.push_back(column->view());
      literalColumns.push_back(std::move(column));
    }

    if (columnViews.size() == 1) {
      if (!columnViews[0].has_nulls()) {
        return std::make_unique<cudf::column>(columnViews[0], stream, mr);
      }
      return cudf::replace_nulls(columnViews[0], emptyString, stream, mr);
    }

    return cudf::strings::concatenate(
        cudf::table_view(columnViews),
        separator,
        emptyString,
        cudf::strings::separator_on_nulls::NO,
        stream,
        mr);
  }

 private:
  struct LiteralArg {
    bool isLiteral{false};
    bool isNull{false};
    bool isArray{false};
    std::string value;
  };

  std::string separator_;
  bool nullSeparator_{false};
  size_t numInputs_{0};
  std::vector<LiteralArg> literals_;
};

enum class RowParentNullPolicy {
  kNever,
  kAnyInputNull,
  kAllInputsNull,
};

class RowConstructorFunction : public CudfFunction {
 public:
  explicit RowConstructorFunction(
      const std::shared_ptr<velox::exec::Expr>& expr,
      RowParentNullPolicy parentNullPolicy = RowParentNullPolicy::kNever,
      std::string functionName = "row_constructor")
      : parentNullPolicy_(parentNullPolicy),
        functionName_(std::move(functionName)) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(
        expr->inputs().size(), 1, "{} expects at least 1 input", functionName_);
    numInputs_ = expr->inputs().size();
    bool hasNonLiteralInput = false;
    literals_.reserve(numInputs_);
    for (const auto& input : expr->inputs()) {
      if (auto constant = std::dynamic_pointer_cast<ConstantExpr>(input)) {
        const bool isUntypedNull = isUntypedNullConstantLiteral(input);
        literals_.push_back(
            isUntypedNull ? nullptr : makeScalarFromConstantExpr(constant));
        untypedNullLiterals_.push_back(isUntypedNull);
      } else {
        hasNonLiteralInput = true;
        literals_.push_back(nullptr);
        untypedNullLiterals_.push_back(false);
      }
    }
    if (!hasNonLiteralInput) {
      VELOX_NYI("{} with only literal inputs is not supported", functionName_);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(
        !inputColumns.empty(),
        "row_constructor requires at least one non-literal input column");
    const cudf::size_type outputSize = asView(inputColumns[0]).size();

    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(numInputs_);

    size_t nextInputColumnIndex = 0;
    for (size_t i = 0; i < literals_.size(); ++i) {
      const auto& literal = literals_[i];
      if (literal) {
        children.push_back(
            cudf::make_column_from_scalar(*literal, outputSize, stream, mr));
      } else if (untypedNullLiterals_[i]) {
        // Velox UNKNOWN is the logical type of an untyped NULL literal and
        // has no cuDF physical type. Keep the logical UNKNOWN in the row
        // schema and use an all-null byte column as its inert physical child.
        children.push_back(
            cudf::make_fixed_width_column(
                cudf::data_type{cudf::type_id::INT8},
                outputSize,
                cudf::mask_state::ALL_NULL,
                stream,
                mr));
      } else {
        VELOX_CHECK_LT(nextInputColumnIndex, inputColumns.size());
        children.push_back(
            makeOwnedColumn(inputColumns[nextInputColumnIndex++], stream, mr));
      }
    }

    VELOX_CHECK_EQ(nextInputColumnIndex, inputColumns.size());

    if (parentNullPolicy_ == RowParentNullPolicy::kNever) {
      return cudf::make_structs_column(
          outputSize, std::move(children), 0, rmm::device_buffer{}, stream, mr);
    }

    std::vector<cudf::column_view> childViews;
    childViews.reserve(children.size());
    for (const auto& child : children) {
      childViews.push_back(child->view());
    }
    auto [nullMask, nullCount] =
        parentNullPolicy_ == RowParentNullPolicy::kAnyInputNull
        ? cudf::bitmask_and(cudf::table_view(childViews), stream, mr)
        : cudf::bitmask_or(cudf::table_view(childViews), stream, mr);
    return cudf::make_structs_column(
        outputSize,
        std::move(children),
        nullCount,
        std::move(nullMask),
        stream,
        mr);
  }

 private:
  static std::unique_ptr<cudf::column> makeOwnedColumn(
      ColumnOrView& holder,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::vector<std::unique_ptr<cudf::scalar>> literals_;
  std::vector<bool> untypedNullLiterals_;
  RowParentNullPolicy parentNullPolicy_;
  std::string functionName_;
  size_t numInputs_{0};
};

std::unique_ptr<cudf::column> RowConstructorFunction::makeOwnedColumn(
    ColumnOrView& holder,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return std::visit(
      [&](auto& value) -> std::unique_ptr<cudf::column> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, cudf::column_view>) {
          return std::make_unique<cudf::column>(value, stream, mr);
        } else {
          return std::move(value);
        }
      },
      holder);
}

class MapFunction : public CudfFunction {
 public:
  explicit MapFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK(
        expr->inputs().size() >= 2 && expr->inputs().size() % 2 == 0,
        "map expects an even number of inputs");
    numInputs_ = expr->inputs().size();
    literals_.reserve(numInputs_);
    inputColumnIndices_.reserve(numInputs_);

    int32_t nextInputColumnIndex = 0;
    bool hasNonLiteralInput = false;
    for (const auto& input : expr->inputs()) {
      if (auto constant = std::dynamic_pointer_cast<ConstantExpr>(input)) {
        literals_.push_back(makeScalarFromConstantExpr(constant));
        inputColumnIndices_.push_back(-1);
      } else {
        hasNonLiteralInput = true;
        literals_.push_back(nullptr);
        inputColumnIndices_.push_back(nextInputColumnIndex++);
      }
    }
    if (!hasNonLiteralInput) {
      VELOX_NYI("map with only literal inputs is not supported");
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty(), "map requires non-literal inputs");
    const auto outputSize = asView(inputColumns[0]).size();
    const auto mapSize = static_cast<cudf::size_type>(numInputs_ / 2);
    const auto entryCount = outputSize * mapSize;

    auto keys = interleaveMapChild(0, inputColumns, outputSize, stream, mr);
    auto values = interleaveMapChild(1, inputColumns, outputSize, stream, mr);

    std::vector<std::unique_ptr<cudf::column>> entryChildren;
    entryChildren.reserve(2);
    entryChildren.push_back(std::move(keys));
    entryChildren.push_back(std::move(values));
    auto entries = cudf::make_structs_column(
        entryCount,
        std::move(entryChildren),
        0,
        rmm::device_buffer{},
        stream,
        mr);

    std::vector<cudf::size_type> offsets(outputSize + 1);
    for (cudf::size_type row = 0; row <= outputSize; ++row) {
      offsets[row] = row * mapSize;
    }
    rmm::device_buffer offsetsBuffer(
        offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
    auto offsetsColumn = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets.size()),
        std::move(offsetsBuffer),
        rmm::device_buffer{},
        0);

    return cudf::make_lists_column(
        outputSize,
        std::move(offsetsColumn),
        std::move(entries),
        0,
        rmm::device_buffer{});
  }

 private:
  std::unique_ptr<cudf::column> interleaveMapChild(
      size_t firstArg,
      std::vector<ColumnOrView>& inputColumns,
      cudf::size_type outputSize,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const {
    std::vector<std::unique_ptr<cudf::column>> materializedLiterals;
    std::vector<cudf::column_view> views;
    materializedLiterals.reserve(numInputs_ / 2);
    views.reserve(numInputs_ / 2);

    for (size_t arg = firstArg; arg < numInputs_; arg += 2) {
      if (literals_[arg]) {
        materializedLiterals.push_back(
            cudf::make_column_from_scalar(
                *literals_[arg], outputSize, stream, mr));
        views.push_back(materializedLiterals.back()->view());
      } else {
        const auto inputColumnIndex = inputColumnIndices_[arg];
        VELOX_CHECK_GE(inputColumnIndex, 0);
        VELOX_CHECK_LT(inputColumnIndex, inputColumns.size());
        auto view = asView(inputColumns[inputColumnIndex]);
        VELOX_CHECK_EQ(view.size(), outputSize);
        views.push_back(view);
      }
    }

    return cudf::interleave_columns(cudf::table_view(views), stream, mr);
  }

  std::vector<std::unique_ptr<cudf::scalar>> literals_;
  std::vector<int32_t> inputColumnIndices_;
  size_t numInputs_{0};
};

std::vector<exec::FunctionSignaturePtr> mapSignatures() {
  constexpr int kMaxKeyValuePairs = 64;
  std::vector<exec::FunctionSignaturePtr> signatures;
  signatures.reserve(kMaxKeyValuePairs);
  for (int i = 1; i <= kMaxKeyValuePairs; ++i) {
    auto builder = exec::FunctionSignatureBuilder()
                       .knownTypeVariable("K")
                       .typeVariable("V")
                       .returnType("map(K,V)");
    for (int arg = 0; arg < i; ++arg) {
      builder.argumentType("K").argumentType("V");
    }
    signatures.push_back(builder.build());
  }
  return signatures;
}

std::vector<exec::FunctionSignaturePtr> greatestLeastSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .returnType("tinyint")
          .argumentType("tinyint")
          .variableArity("tinyint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("smallint")
          .argumentType("smallint")
          .variableArity("smallint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("integer")
          .variableArity("integer")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .variableArity("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("real")
          .argumentType("real")
          .variableArity("real")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("double")
          .variableArity("double")
          .build(),
      exec::FunctionSignatureBuilder()
          .integerVariable("p")
          .integerVariable("s")
          .returnType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .variableArity("decimal(p,s)")
          .build()};
}

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
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());
    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }
    return cudf::hashing::murmurhash3_x86_32(
        cudf::table_view(columns), seedValue_, stream, mr);
  }

 private:
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
    seedValue_ = static_cast<uint64_t>(
        seedExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0));
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

class Sha2Function : public CudfFunction {
 public:
  explicit Sha2Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "sha2 expects exactly 2 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::VARBINARY,
        "sha2 input must be VARBINARY");
    VELOX_CHECK_EQ(
        expr->type()->kind(), TypeKind::VARCHAR, "sha2 output must be VARCHAR");

    auto bitLengthExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(bitLengthExpr, "sha2 bit length must be a constant");
    auto bitLengthValue = bitLengthExpr->value();
    VELOX_CHECK_NOT_NULL(bitLengthValue, "sha2 bit length value is missing");

    if (bitLengthValue->isNullAt(0)) {
      returnsNull_ = true;
      return;
    }

    auto bitLength = bitLengthValue->as<SimpleVector<int32_t>>()->valueAt(0);
    if (bitLength == 0) {
      bitLength = 256;
    }

    switch (bitLength) {
      case 224:
        kind_ = Kind::kSha224;
        break;
      case 256:
        kind_ = Kind::kSha256;
        break;
      case 384:
        kind_ = Kind::kSha384;
        break;
      case 512:
        kind_ = Kind::kSha512;
        break;
      default:
        returnsNull_ = true;
        break;
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto input = asView(inputColumns[0]);
    if (returnsNull_) {
      cudf::string_scalar nullString("", false, stream, mr);
      auto result =
          cudf::make_column_from_scalar(nullString, input.size(), stream, mr);
      stream.synchronize();
      return result;
    }

    std::vector<cudf::column_view> columns{input};
    auto table = cudf::table_view(columns);
    std::unique_ptr<cudf::column> result;
    switch (kind_) {
      case Kind::kSha224:
        result = cudf::hashing::sha224(table, stream, mr);
        break;
      case Kind::kSha256:
        result = cudf::hashing::sha256(table, stream, mr);
        break;
      case Kind::kSha384:
        result = cudf::hashing::sha384(table, stream, mr);
        break;
      case Kind::kSha512:
        result = cudf::hashing::sha512(table, stream, mr);
        break;
    }
    mergeNullSourceNullsIntoResult(*result, input, stream, mr);
    return result;
  }

 private:
  enum class Kind { kSha224, kSha256, kSha384, kSha512 };

  Kind kind_{Kind::kSha256};
  bool returnsNull_{false};
};

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite,
    CudfCanEvaluate canEvaluate) {
  auto& registry = getCudfFunctionRegistry();
  if (!overwrite && !registry[name].empty()) {
    return false;
  }
  registry[name].push_back(
      CudfFunctionSpec{std::move(factory), signatures, std::move(canEvaluate)});
  return true;
}

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite,
    CudfCanEvaluate canEvaluate) {
  for (const auto& name : aliases) {
    registerCudfFunction(name, factory, signatures, overwrite, canEvaluate);
  }
}

std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const std::shared_ptr<velox::exec::Expr>& expr) {
  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(name);
  if (it == registry.end()) {
    return nullptr;
  }
  for (const auto& spec : it->second) {
    // Empty signatures matching must be allowed to handle
    // the special case of cast.
    if (!spec.signatures.empty() &&
        !matchCallAgainstSignatures(*expr, spec.signatures)) {
      continue;
    }
    if (spec.canEvaluate && !spec.canEvaluate(expr)) {
      continue;
    }
    return spec.factory(name, expr);
  }
  return nullptr;
}

bool registerBuiltinFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .constantArgumentType("integer")
           .argumentType("any")
           .variableArity("any")
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
           .variableArity("any")
           .build()});

  registerCudfFunction(
      prefix + "sha2",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<Sha2Function>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varbinary")
           .constantArgumentType("integer")
           .build()});

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
      prefix + "regexp_extract",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RegexpExtractFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("bigint")
           .build()});

  registerCudfFunction(
      prefix + "regexp_replace",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RegexpReplaceFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "replace",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ReplaceFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  const std::vector<exec::FunctionSignaturePtr> trimSignatures{
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .build(),
      FunctionSignatureBuilder()
          .returnType("varchar")
          .constantArgumentType("varchar")
          .argumentType("varchar")
          .build()};

  registerCudfFunction(
      prefix + "trim",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<TrimFunction>(
            expr, cudf::strings::side_type::BOTH);
      },
      trimSignatures);

  registerCudfFunction(
      prefix + "ltrim",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<TrimFunction>(
            expr, cudf::strings::side_type::LEFT);
      },
      trimSignatures);

  registerCudfFunction(
      prefix + "rtrim",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<TrimFunction>(
            expr, cudf::strings::side_type::RIGHT);
      },
      trimSignatures);

  registerCudfFunction(
      prefix + "get_json_object",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GetJsonObjectFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
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

  registerCudfFunction(
      prefix + "array_contains",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ArrayContainsFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("array(T)")
           .argumentType("T")
           .build()});

  registerCudfFunction(
      prefix + "array",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ArrayConstructorFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("T")
           .variableArity("T")
           .build()});

  registerCudfFunction(
      prefix + "size",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SizeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .argumentType("array(any)")
           .constantArgumentType("boolean")
           .build()});

  auto arrayIdentitySignature = FunctionSignatureBuilder()
                                    .typeVariable("T")
                                    .returnType("array(T)")
                                    .argumentType("array(T)")
                                    .build();
  registerCudfFunction(
      prefix + "sort_array",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SortArrayFunction>(expr);
      },
      {arrayIdentitySignature,
       FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("array(T)")
           .constantArgumentType("boolean")
           .build()});

  registerCudfFunction(
      prefix + "array_distinct",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ArrayDistinctFunction>(expr);
      },
      {arrayIdentitySignature});

  registerCudfFunction(
      prefix + "array_except",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ArrayExceptFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("array(T)")
           .argumentType("array(T)")
           .build()});

  registerCudfFunction(
      prefix + "flatten",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FlattenFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("array(array(T))")
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
           .constantArgumentType("integer")
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
           .constantArgumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("integer")
           .constantArgumentType("bigint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .constantArgumentType("bigint")
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

  // row_constructor is a special form and doesn't have a prefix in its name.
  registerCudfFunction(
      "row_constructor",
      [](const std::string& name,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RowConstructorFunction>(
            expr, RowParentNullPolicy::kNever, name);
      },
      {});

  registerCudfFunction(
      "row_constructor_with_null",
      [](const std::string& name,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RowConstructorFunction>(
            expr, RowParentNullPolicy::kAnyInputNull, name);
      },
      {});

  registerCudfFunction(
      "row_constructor_with_all_null",
      [](const std::string& name,
         const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RowConstructorFunction>(
            expr, RowParentNullPolicy::kAllInputsNull, name);
      },
      {});

  registerCudfFunction(
      prefix + "map",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<MapFunction>(expr);
      },
      mapSignatures());

  registerCudfFunction(
      "and",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<LogicalFunction>(
            expr, cudf::binary_operator::NULL_LOGICAL_AND);
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
            expr, cudf::binary_operator::NULL_LOGICAL_OR);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .variableArity("boolean")
           .build()});

  registerCudfFunctions(
      {"is_null", "isnull", prefix + "is_null", prefix + "isnull"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<NullPredicateFunction>(expr, false);
      },
      {});

  registerCudfFunctions(
      {"isnotnull", prefix + "isnotnull"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<NullPredicateFunction>(expr, true);
      },
      {});

  registerCudfFunction(
      "not",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<UnaryFunction>(expr, cudf::unary_operator::NOT);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("boolean")
           .build()});

  registerCudfFunction(
      "is_null",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<IsNullFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("T")
           .build()});

  registerCudfFunction(
      "isnotnull",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<IsNotNullFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("boolean")
           .argumentType("T")
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

  const std::vector<exec::FunctionSignaturePtr> timestampDateIntegerSignatures{
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("integer")
          .argumentType("date")
          .build()};

  registerCudfFunction(
      prefix + "year",
      ExtractComponentFactory{cudf::datetime::datetime_component::YEAR},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "month",
      ExtractComponentFactory{cudf::datetime::datetime_component::MONTH},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "day",
      ExtractComponentFactory{cudf::datetime::datetime_component::DAY},
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "dow", prefix + "day_of_week"},
      ExtractComponentFactory{cudf::datetime::datetime_component::WEEKDAY},
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "doy", prefix + "day_of_year"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DayOfYearFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "week", prefix + "week_of_year"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<WeekFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "quarter",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<QuarterFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunctions(
      {prefix + "yow", prefix + "year_of_week"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<YearOfWeekFunction>(expr);
      },
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "hour",
      ExtractComponentFactory{cudf::datetime::datetime_component::HOUR},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "minute",
      ExtractComponentFactory{cudf::datetime::datetime_component::MINUTE},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "second",
      ExtractComponentFactory{cudf::datetime::datetime_component::SECOND},
      timestampDateIntegerSignatures);

  registerCudfFunction(
      prefix + "millisecond",
      ExtractComponentFactory{cudf::datetime::datetime_component::MILLISECOND},
      timestampDateIntegerSignatures);

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
           .argumentType("varchar")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "rlike",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<RLikeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
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
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "endswith",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<EndswithFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "contains",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ContainsFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .argumentType("varchar")
           .argumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "concat",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ConcatFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .variableArity("varchar")
           .build()});

  registerCudfFunction(
      prefix + "concat_ws",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<ConcatWsFunction>(expr);
      },
      {});

  // No prefix because switch and if are special form
  registerCudfFunctions(
      {"switch", "if"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SwitchFunction>(expr);
      },
      {
          // Switch is a special form: condition/value pairs followed by else.
          // The branch positions are homogeneous, but condition positions are
          // boolean, so a single Velox variable-arity signature cannot model
          // the alternating argument pattern.
      });

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
    auto decimalBinarySignature = [&]() {
      return FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .integerVariable("r_precision")
          .integerVariable("r_scale")
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
         decimalBinarySignature()});
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

  auto comparisonSignatureFor = [](const std::string& typeName) {
    return FunctionSignatureBuilder()
        .returnType("boolean")
        .argumentType(typeName)
        .argumentType(typeName)
        .build();
  };

  const std::vector<exec::FunctionSignaturePtr> comparisonSignatures{
      comparisonSignatureFor("boolean"),
      comparisonSignatureFor("tinyint"),
      comparisonSignatureFor("smallint"),
      comparisonSignatureFor("integer"),
      comparisonSignatureFor("bigint"),
      comparisonSignatureFor("real"),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("double")
          .argumentType("double")
          .build(),
      comparisonSignatureFor("varchar"),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("date")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("b_precision")
          .integerVariable("b_scale")
          .returnType("boolean")
          .argumentType("decimal(a_precision, a_scale)")
          .argumentType("decimal(b_precision, b_scale)")
          .build()};

  auto registerComparisonOp = [&](const std::vector<std::string>& aliases,
                                  cudf::binary_operator op) {
    registerCudfFunctions(
        aliases,
        [op](
            const std::string&,
            const std::shared_ptr<velox::exec::Expr>& expr) {
          return std::make_shared<BinaryFunction>(expr, op);
        },
        comparisonSignatures);
  };

  registerComparisonOp(
      {prefix + "equalto", prefix + "eq", prefix + "decimal_equalto"},
      cudf::binary_operator::EQUAL);
  registerComparisonOp(
      {prefix + "equalnullsafe", "equalnullsafe"},
      cudf::binary_operator::NULL_EQUALS);
  registerComparisonOp(
      {prefix + "notequalto", prefix + "neq", prefix + "decimal_notequalto"},
      cudf::binary_operator::NOT_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthanorequal",
       prefix + "gte",
       prefix + "decimal_greaterthanorequal"},
      cudf::binary_operator::GREATER_EQUAL);
  registerComparisonOp(
      {prefix + "lessthanorequal",
       prefix + "lte",
       prefix + "decimal_lessthanorequal"},
      cudf::binary_operator::LESS_EQUAL);
  registerComparisonOp(
      {prefix + "greaterthan", prefix + "gt", prefix + "decimal_greaterthan"},
      cudf::binary_operator::GREATER);
  registerComparisonOp(
      {prefix + "lessthan", prefix + "lt", prefix + "decimal_lessthan"},
      cudf::binary_operator::LESS);

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

  const std::vector<exec::FunctionSignaturePtr> betweenSignatures{
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("double")
          .argumentType("double")
          .argumentType("double")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .build(),
      FunctionSignatureBuilder()
          .returnType("boolean")
          .argumentType("date")
          .argumentType("date")
          .argumentType("date")
          .build(),
      FunctionSignatureBuilder()
          .integerVariable("p")
          .integerVariable("s")
          .returnType("boolean")
          .argumentType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .argumentType("decimal(p,s)")
          .build()};

  registerCudfFunction(
      prefix + "between",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<BetweenFunction>(expr);
      },
      betweenSignatures);

  //
  // greatest & least
  //

  registerCudfFunction(
      prefix + "greatest",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MAX);
      },
      greatestLeastSignatures());

  registerCudfFunction(
      prefix + "least",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GreatestLeastFunction>(
            expr, cudf::binary_operator::NULL_MIN);
      },
      greatestLeastSignatures());

  // Note: Spark and Presto functions are now registered separately via
  // registerSparkFunctions() and registerPrestoFunctions()
  return true;
}

bool shouldEvaluateAsFunctionSubexpression(
    const std::shared_ptr<velox::exec::Expr>& parent,
    const std::shared_ptr<velox::exec::Expr>& input) {
  if (std::dynamic_pointer_cast<velox::exec::ConstantExpr>(input)) {
    return false;
  }
  if (parent->name() == "map_filter" && input->isLambda()) {
    return false;
  }
  return true;
}

std::shared_ptr<FunctionExpression> FunctionExpression::create(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const RowTypePtr& inputRowSchema,
    const core::QueryConfig* queryConfig) {
  using velox::exec::FieldReference;

  auto node = std::make_shared<FunctionExpression>();
  node->expr_ = expr;
  node->inputRowSchema_ = inputRowSchema;
  node->queryConfig_ = queryConfig;

  auto name = expr->name();
  if (name == "get_struct_field" || name == "spark_get_struct_field") {
    node->function_ = std::make_shared<GetStructFieldFunction>(expr);
  } else {
    ScopedCudfFunctionQueryConfig scope(queryConfig);
    node->function_ = createCudfFunction(name, expr);
  }

  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    if (!fieldExpr->inputs().empty()) {
      VELOX_CHECK_EQ(
          fieldExpr->inputs().size(),
          1,
          "Nested field reference expects exactly one input");
      auto parentRowType = asRowType(fieldExpr->inputs()[0]->type());
      VELOX_CHECK_NOT_NULL(
          parentRowType,
          "Nested FieldReference parent must be a ROW: {}",
          expr->toString());
      node->fieldIndex_ = resolveFieldReferenceIndex(*fieldExpr, parentRowType);
    }
  }

  if (node->function_ || std::dynamic_pointer_cast<FieldReference>(expr)) {
    for (const auto& input : expr->inputs()) {
      if (shouldEvaluateAsFunctionSubexpression(expr, input)) {
        node->subexpressions_.push_back(
            createCudfExpression(input, inputRowSchema, queryConfig));
      }
    }
  }

  return node;
}

std::unique_ptr<cudf::column> FunctionExpression::makeStructChildColumn(
    ColumnOrView& structColumn,
    cudf::size_type childIndex,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return std::visit(
      [&](auto& value) -> std::unique_ptr<cudf::column> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, cudf::column_view>) {
          VELOX_CHECK(
              value.type().id() == cudf::type_id::STRUCT,
              "Nested field reference expects a ROW input");
          VELOX_CHECK_LT(static_cast<size_t>(childIndex), value.num_children());
          auto const childView =
              cudf::structs_column_view(value).get_sliced_child(
                  childIndex, stream);
          auto child = std::make_unique<cudf::column>(childView, stream, mr);
          mergeNullSourceNullsIntoResult(*child, value, stream, mr);
          return child;
        } else {
          auto const structView = value->view();
          VELOX_CHECK(
              structView.type().id() == cudf::type_id::STRUCT,
              "Nested field reference expects a ROW input");
          VELOX_CHECK_LT(
              static_cast<size_t>(childIndex), structView.num_children());

          // `structView` points into `value`. Keep `contents` alive until after
          // merging parent nulls so structView's null mask remains valid.
          auto contents = value->release();
          auto child = std::move(contents.children[childIndex]);
          mergeNullSourceNullsIntoResult(*child, structView, stream, mr);
          return child;
        }
      },
      structColumn);
}

ColumnOrView FunctionExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  const auto inputRowCount = inputColumnViews.empty()
      ? cudf::size_type{0}
      : inputColumnViews.front().size();
  return eval(std::move(inputColumnViews), inputRowCount, stream, mr, finalize);
}

ColumnOrView FunctionExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    cudf::size_type inputRowCount,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  using velox::exec::FieldReference;

  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr_)) {
    if (fieldExpr->inputs().empty()) {
      auto columnIndex = inputRowSchema_->getChildIdx(fieldExpr->name());
      return inputColumnViews[columnIndex];
    }

    VELOX_CHECK_EQ(
        fieldExpr->inputs().size(),
        1,
        "Nested field reference expects exactly one input");
    VELOX_CHECK_EQ(
        subexpressions_.size(),
        1,
        "Nested field reference expects exactly one subexpression");

    auto parent = subexpressions_[0]->eval(inputColumnViews, stream, mr);
    VELOX_DCHECK_GE(fieldIndex_, 0);
    auto child = FunctionExpression::makeStructChildColumn(
        parent, static_cast<cudf::size_type>(fieldIndex_), stream, mr);
    VELOX_DCHECK(
        child->view().type() == cudf_velox::veloxToCudfDataType(expr_->type()));
    return child;
  }

  if (function_) {
    std::vector<ColumnOrView> subexprResults;
    subexprResults.reserve(subexpressions_.size());

    for (const auto& subexpr : subexpressions_) {
      subexprResults.push_back(subexpr->eval(inputColumnViews, stream, mr));
    }

    auto result = function_->eval(subexprResults, inputRowCount, stream, mr);
    if (finalize) {
      const auto requestedType = cudf_velox::veloxToCudfDataType(expr_->type());
      auto resultView = asView(result);
      if (resultView.type() != requestedType) {
        return cudf::cast(resultView, requestedType, stream, mr);
      }
    }
    const bool resultIsView = std::holds_alternative<cudf::column_view>(result);
    const bool hasOwnedSubexpr = std::any_of(
        subexprResults.begin(), subexprResults.end(), [](auto& subexprResult) {
          return std::holds_alternative<std::unique_ptr<cudf::column>>(
              subexprResult);
        });
    if (resultIsView && hasOwnedSubexpr) {
      return std::make_unique<cudf::column>(asView(result), stream, mr);
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

  if (auto constExpr =
          std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr)) {
    return (constExpr->value() != nullptr &&
            isSupportedCudfScalarLiteralType(constExpr->type())) ||
        isNullComplexConstantLiteral(expr);
  }

  const auto& opName = expr->name();
  if (opName == "switch" || opName == "if") {
    return isSwitchExpressionSupported(expr);
  }

  if (opName == "get_struct_field" || opName == "spark_get_struct_field") {
    return expr->inputs().size() == 2 &&
        !std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
               expr->inputs()[0]) &&
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[1]);
  }

  if (opName == "is_null" || opName == "isnull" || opName == "spark_is_null" ||
      opName == "spark_isnull" || opName == "isnotnull" ||
      opName == "spark_isnotnull") {
    return expr->inputs().size() == 1 &&
        !std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
               expr->inputs()[0]);
  }

  if (opName == "cast" || opName == "try_cast") {
    const auto& srcType =
        expr->inputs().empty() ? nullptr : expr->inputs()[0]->type();
    const auto& dstType = expr->type();
    if (srcType == nullptr || dstType == nullptr) {
      return false;
    }
    if (isStringToBooleanVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isStringToDateVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isStringToTimestampVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isNumericToTimestampVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isNumericToBooleanVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isListIntegralToStringVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isPlainCudfCastSupported(srcType, dstType)) {
      return true;
    }
    if (srcType->isDate() && dstType->kind() == TypeKind::VARCHAR) {
      return true;
    }
    if (isTimestampToStringVeloxCast(srcType, dstType)) {
      return true;
    }
    if (isIntegralNonDecimalVeloxType(srcType) &&
        dstType->kind() == TypeKind::VARCHAR) {
      return true;
    }
    if (isFloatingPointVeloxType(srcType) &&
        dstType->kind() == TypeKind::VARCHAR) {
      return true;
    }
    if (srcType->kind() == TypeKind::VARCHAR &&
        isIntegralNonDecimalVeloxType(dstType)) {
      return true;
    }
    if (srcType->kind() == TypeKind::VARCHAR &&
        isFloatingPointVeloxType(dstType)) {
      return true;
    }
    return false;
  }

  if (hasFunctionNameSuffix(opName, "date_format") ||
      hasFunctionNameSuffix(opName, "get_timestamp")) {
    if (!hasSupportedConstantSparkDatetimePattern(expr, 1)) {
      return false;
    }
  }

  if (hasFunctionNameSuffix(opName, "from_json")) {
    return isSupportedFromJsonExpr(expr);
  }

  if (hasFunctionNameSuffix(opName, "decode")) {
    if (expr->inputs().size() != 2 ||
        expr->inputs()[0]->type()->kind() != TypeKind::VARBINARY ||
        expr->type()->kind() != TypeKind::VARCHAR) {
      return false;
    }
    if (std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0])) {
      return false;
    }
    if (!hasSupportedConstantDecodeCharset(expr, 1)) {
      return false;
    }
  }

  if ((opName == "coalesce" || opName == "row_constructor" ||
       opName == "row_constructor_with_null" ||
       opName == "row_constructor_with_all_null") &&
      hasUnsupportedComplexConstantLiteral(expr)) {
    return false;
  }

  if (isCudfRegexFunction(opName) && hasUnsupportedCudfRegexPatternArg(expr)) {
    return false;
  }

  auto& registry = getCudfFunctionRegistry();
  auto it = registry.find(expr->name());
  if (it == registry.end()) {
    return false;
  }
  for (const auto& spec : it->second) {
    if (!spec.signatures.empty() &&
        !matchCallAgainstSignatures(*expr, spec.signatures)) {
      continue;
    }
    if (spec.canEvaluate && !spec.canEvaluate(expr)) {
      continue;
    }
    return true;
  }
  return false;
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
      if (shouldEvaluateAsFunctionSubexpression(expr, input) &&
          !canBeEvaluatedByCudf(input, true)) {
        return false;
      }
    }
  }

  return true;
}

std::shared_ptr<CudfExpression> createCudfExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema,
    const core::QueryConfig* queryConfig) {
  ensureBuiltinExpressionEvaluatorsRegistered();
  const auto& registry = getCudfExpressionEvaluatorRegistry();

  const CudfExpressionEvaluatorEntry* best = nullptr;
  const std::string* bestName = nullptr;
  for (const auto& [name, entry] : registry) {
    if (entry.canEvaluate && entry.canEvaluate(expr)) {
      if (best == nullptr || entry.priority > best->priority) {
        best = &entry;
        bestName = &name;
      }
    }
  }

  if (best != nullptr) {
    if (bestName != nullptr && *bestName == "function") {
      return FunctionExpression::create(expr, inputRowSchema, queryConfig);
    }
    return best->create(expr, inputRowSchema);
  }

  return FunctionExpression::create(expr, inputRowSchema, queryConfig);
}

void unregisterFunctions() {
  auto& registry = getCudfFunctionRegistry();
  registry.clear();
}

} // namespace facebook::velox::cudf_velox
