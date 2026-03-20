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
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/JitExpression.h"

#include <cudf/column/column_factories.hpp>

namespace facebook::velox::cudf_velox {

JitExpression::JitExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema)
    : expr_{expr, inputRowSchema} {}

void JitExpression::close() {
  expr_.close();
}

ColumnOrView JitExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputColumnViews,
      expr_.precomputeInstructions_,
      expr_.scalars_,
      expr_.inputRowSchema_,
      stream);

  auto result = [&]() -> ColumnOrView {
    if (auto colRefPtr = dynamic_cast<cudf::ast::column_reference const*>(
            &expr_.cudfTree_.back())) {
      auto columnIndex = colRefPtr->get_column_index();
      if (columnIndex < inputColumnViews.size()) {
        return inputColumnViews[columnIndex];
      } else {
        return std::move(
            precomputedColumns[columnIndex - inputColumnViews.size()]);
      }
    } else if (
        auto litPtr = dynamic_cast<cudf::ast::literal const*>(
            &expr_.cudfTree_.back())) {
      auto numRows = inputColumnViews.empty()
          ? 0
          : inputColumnViews[0].size();
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(WARNING) << "JitExpr: literal shortcut, rows="
                     << numRows;
      }
      return cudf::make_column_from_scalar(
          litPtr->get_scalar(), numRows, stream, mr);
    } else {
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(WARNING) << "JitExpr: compute_column_jit path";
      }
      std::vector<cudf::column_view> allColumnViews(inputColumnViews);
      allColumnViews.reserve(
          inputColumnViews.size() + precomputedColumns.size());
      for (auto& precomputedCol : precomputedColumns) {
        allColumnViews.push_back(asView(precomputedCol));
      }
      cudf::table_view astInputTableView(allColumnViews);
      return cudf::compute_column_jit(
          astInputTableView, expr_.cudfTree_.back(), stream, mr);
    }
  }();
  if (finalize) {
    const auto requestedType =
        cudf_velox::veloxToCudfDataType(expr_.expr_->type());
    auto resultView = asView(result);
    if (resultView.type() != requestedType) {
      result = cudf::cast(resultView, requestedType, stream, mr);
    }
  }
  return result;
}

bool JitExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  return ASTExpression::canEvaluate(expr);
}

void registerJitEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kJitEvaluatorName,
      priority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return JitExpression::canEvaluate(expr);
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return std::make_shared<JitExpression>(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
