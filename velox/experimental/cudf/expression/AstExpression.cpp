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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/AstPrinter.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/vector/TableViewPrinter.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/column/column_factories.hpp>

namespace facebook::velox::cudf_velox {

// Create tree from Expr
// and collect precompute instructions for non-ast operations
cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precomputeInstructions) {
  AstContext context{
      tree, scalars, {inputRowSchema}, {precomputeInstructions}, expr};
  return context.pushExprToTree(expr);
}

cudf::ast::expression const& createAstTree(
    const std::shared_ptr<velox::exec::Expr>& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& leftPrecomputeInstructions,
    std::vector<PrecomputeInstruction>& rightPrecomputeInstructions) {
  AstContext context{
      tree,
      scalars,
      {leftRowSchema, rightRowSchema},
      {leftPrecomputeInstructions, rightPrecomputeInstructions},
      expr};
  return context.pushExprToTree(expr);
}

ASTExpression::ASTExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema)
    : expr_(expr), inputRowSchema_(inputRowSchema) {
  createAstTree(
      expr, cudfTree_, scalars_, inputRowSchema, precomputeInstructions_);
}

void ASTExpression::close() {
  cudfTree_ = {};
  scalars_.clear();
  precomputeInstructions_.clear();
}

ColumnOrView ASTExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputColumnViews,
      precomputeInstructions_,
      scalars_,
      inputRowSchema_,
      stream);

  auto result = [&]() -> ColumnOrView {
    if (auto colRefPtr = dynamic_cast<cudf::ast::column_reference const*>(
            &cudfTree_.back())) {
      auto columnIndex = colRefPtr->get_column_index();
      if (columnIndex < inputColumnViews.size()) {
        return inputColumnViews[columnIndex];
      } else {
        return std::move(
            precomputedColumns[columnIndex - inputColumnViews.size()]);
      }
    } else if (
        auto litPtr = dynamic_cast<cudf::ast::literal const*>(
            &cudfTree_.back())) {
      auto numRows = inputColumnViews.empty()
          ? 0
          : inputColumnViews[0].size();
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(WARNING) << "AstExpr: literal shortcut, rows="
                     << numRows;
      }
      return cudf::make_column_from_scalar(
          litPtr->get_scalar(), numRows, stream, mr);
    } else {
      // Build table_view lazily: only needed for compute_column.
      // Precomputed columns referenced via column_reference may have a
      // different row count (from a nested evaluator), so constructing
      // the table_view eagerly would trigger a spurious size mismatch.
      std::vector<cudf::column_view> allColumnViews(inputColumnViews);
      allColumnViews.reserve(
          inputColumnViews.size() + precomputedColumns.size());
      for (auto& precomputedCol : precomputedColumns) {
        allColumnViews.push_back(asView(precomputedCol));
      }
      cudf::table_view astInputTableView(allColumnViews);
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(WARNING) << "AstExpr: compute_column path, expr="
            << cudf::ast::expression_to_string(cudfTree_.back());
        LOG(WARNING) << "AstExpr: table_schema="
            << cudf::table_schema_to_string(astInputTableView);
      }
      try {
        return cudf::compute_column(
            astInputTableView, cudfTree_.back(), stream, mr);
      } catch (const std::exception& e) {
        VELOX_FAIL(
            "cudf::compute_column failed (possible operand type mismatch): {}",
            e.what());
      }
    }
  }();
  if (finalize) {
    const auto requestedType = cudf_velox::veloxToCudfDataType(expr_->type());
    auto resultView = asView(result);
    if (resultView.type() != requestedType) {
      result = cudf::cast(resultView, requestedType, stream, mr);
    }
  }
  return result;
}

bool ASTExpression::canEvaluate(std::shared_ptr<velox::exec::Expr> expr) {
  return std::dynamic_pointer_cast<velox::exec::FieldReference>(expr) !=
      nullptr ||
      detail::isAstExprSupported(expr);
}

void registerAstEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kAstEvaluatorName,
      priority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return ASTExpression::canEvaluate(expr);
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return std::make_shared<ASTExpression>(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
