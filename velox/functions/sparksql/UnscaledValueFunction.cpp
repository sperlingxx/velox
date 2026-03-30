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
#include "velox/functions/sparksql/UnscaledValueFunction.h"

#include "velox/expression/DecodedArgs.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Return the unscaled bigint value of a decimal.
// Supports both short decimals (precision <= 18, backed by int64_t)
// and long decimals (precision > 18, backed by int128_t).
class UnscaledValueFunction final : public exec::VectorFunction {
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const final {
    auto type = args[0]->type();
    VELOX_USER_CHECK(
        type->isShortDecimal() || type->isLongDecimal(),
        "Expect decimal type, but got: {}",
        type);
    VectorPtr localResult;
    const auto& arg = args[0];
    if (type->isShortDecimal()) {
      if (arg->isConstantEncoding()) {
        auto value = arg->as<ConstantVector<int64_t>>()->valueAt(0);
        localResult = std::make_shared<ConstantVector<int64_t>>(
            context.pool(), rows.end(), false, BIGINT(), std::move(value));
      } else {
        auto flatInput = arg->asFlatVector<int64_t>();
        localResult = std::make_shared<FlatVector<int64_t>>(
            context.pool(),
            BIGINT(),
            nullptr,
            rows.end(),
            flatInput->values(),
            std::vector<BufferPtr>());
      }
    } else {
      if (arg->isConstantEncoding()) {
        auto value = static_cast<int64_t>(
            arg->as<ConstantVector<int128_t>>()->valueAt(0));
        localResult = std::make_shared<ConstantVector<int64_t>>(
            context.pool(), rows.end(), false, BIGINT(), std::move(value));
      } else {
        auto flatInput = arg->asFlatVector<int128_t>();
        auto flatResult =
            BaseVector::create(BIGINT(), rows.end(), context.pool());
        auto rawResults =
            flatResult->as<FlatVector<int64_t>>()->mutableRawValues();
        rows.applyToSelected([&](auto row) {
          rawResults[row] = static_cast<int64_t>(flatInput->valueAt(row));
        });
        if (flatInput->nulls()) {
          flatResult->setNulls(flatInput->nulls());
        }
        localResult = flatResult;
      }
    }
    context.moveOrCopyResult(localResult, rows, result);
  }
};
} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>>
unscaledValueSignatures() {
  return {exec::FunctionSignatureBuilder()
              .integerVariable("precision")
              .integerVariable("scale")
              .returnType("bigint")
              .argumentType("DECIMAL(precision, scale)")
              .build()};
}

std::unique_ptr<exec::VectorFunction> makeUnscaledValue() {
  return std::make_unique<UnscaledValueFunction>();
}

} // namespace facebook::velox::functions::sparksql
