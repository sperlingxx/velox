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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/CudfExpand.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfNestedLoopJoin.h"
#include "velox/experimental/cudf/exec/CudfLimit.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/type/TypeUtil.h"
#include "velox/exec/AssignUniqueId.h"
#include "velox/exec/CallbackSink.h"
#include "velox/exec/Expand.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/NestedLoopJoinBuild.h"
#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/exec/Limit.h"
#include "velox/exec/LocalPartition.h"
#include "velox/exec/OrderBy.h"
#include "velox/exec/StreamingAggregation.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/Task.h"
#include "velox/exec/TopN.h"
#include "velox/exec/TopNRowNumber.h"
#include "velox/exec/Values.h"
#include "velox/exec/Window.h"

namespace facebook::velox::cudf_velox {

namespace {

// Recursively check that all FieldAccessTypedExpr references in the
// expression tree can be found in the given inputType.  Returns false
// if any top-level field reference is missing from inputType.
bool allFieldsResolvable(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputType) {
  if (auto fieldAccess =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    if (fieldAccess->inputs().empty()) {
      if (!inputType->containsChild(fieldAccess->name())) {
        return false;
      }
    }
  }
  for (const auto& input : expr->inputs()) {
    if (!allFieldsResolvable(input, inputType)) {
      return false;
    }
  }
  return true;
}

} // namespace

/// OperatorAdapterRegistry Implementation
OperatorAdapterRegistry& OperatorAdapterRegistry::getInstance() {
  static OperatorAdapterRegistry instance;
  return instance;
}

void OperatorAdapterRegistry::registerAdapter(
    std::unique_ptr<OperatorAdapter> adapter) {
  adapters_.push_back(std::move(adapter));
}

const OperatorAdapter* OperatorAdapterRegistry::findAdapter(
    const exec::Operator* op) const {
  for (const auto& adapter : adapters_) {
    if (adapter->canHandle(op)) {
      return adapter.get();
    }
  }
  // Note: It is possible to have priority based adapter search.
  // But this is not implemented because it is not needed for now.
  return nullptr;
}

const std::vector<std::unique_ptr<OperatorAdapter>>&
OperatorAdapterRegistry::getAdapters() const {
  return adapters_;
}

void OperatorAdapterRegistry::clear() {
  adapters_.clear();
}

/// TableScanAdapter - Keeps original operator (produces GPU output)
class TableScanAdapter : public OperatorAdapter {
 public:
  TableScanAdapter() : OperatorAdapter("TableScan") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TableScan*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto tableScanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    if (!tableScanNode) {
      return false;
    }
    auto const& connector = velox::connector::getConnector(
        tableScanNode->tableHandle()->connectorId());
    auto cudfHiveConnector = std::dynamic_pointer_cast<
        facebook::velox::cudf_velox::connector::hive::CudfHiveConnector>(
        connector);
    return cudfHiveConnector != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// FilterProjectAdapter - Replaces with CudfFilterProject
class FilterProjectAdapter : public OperatorAdapter {
 public:
  FilterProjectAdapter() : OperatorAdapter("FilterProject") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    if (!filterProjectOp) {
      return false;
    }

    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterNode = filterProjectOp->filterNode();

    if (projectPlanNode) {
      if (projectPlanNode->sources()[0]->outputType()->size() == 0 ||
          projectPlanNode->outputType()->size() == 0) {
        return false;
      }
    }

    // Check filter separately
    if (filterNode) {
      if (!canBeEvaluatedByCudf(
              {filterNode->filter()}, ctx->task->queryCtx().get())) {
        return false;
      }
    }

    // Check projects separately
    if (projectPlanNode) {
      if (!canBeEvaluatedByCudf(
              projectPlanNode->projections(), ctx->task->queryCtx().get())) {
        return false;
      }
      const auto& inputType = projectPlanNode->sources()[0]->outputType();
      for (const auto& proj : projectPlanNode->projections()) {
        if (!allFieldsResolvable(proj, inputType)) {
          LOG(WARNING)
              << "CudfFilterProject: unresolvable field reference in "
              << proj->toString() << ", falling back to CPU";
          return false;
        }
      }
    }
    if (filterNode) {
      const auto& inputType = filterNode->sources()[0]->outputType();
      if (!allFieldsResolvable(filterNode->filter(), inputType)) {
        LOG(WARNING)
            << "CudfFilterProject: unresolvable field reference in filter, "
            << "falling back to CPU";
        return false;
      }
    }
    return true;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterPlanNode = filterProjectOp->filterNode();

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfFilterProject>(
            operatorId, ctx, filterPlanNode, projectPlanNode));
    return result;
  }
};

/// AggregationAdapter - Replaces with CudfHashAggregation
class AggregationAdapter : public OperatorAdapter {
 public:
  AggregationAdapter() : OperatorAdapter("Aggregation") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashAggregation*>(op) != nullptr ||
        dynamic_cast<const exec::StreamingAggregation*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      return false;
    }

    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
    if (!aggregationPlanNode) {
      return false;
    }

    return canBeEvaluatedByCudf(
        *aggregationPlanNode, ctx->task->queryCtx().get());
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashAggregation>(
            operatorId, ctx, aggregationPlanNode));
    return result;
  }
};

class CudfHashJoinBaseAdapter : public OperatorAdapter {
 public:
  using OperatorAdapter::OperatorAdapter;

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      return false;
    }

    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) {
      return false;
    }

    if (!CudfHashJoinProbe::isSupportedJoinType(joinPlanNode->joinType())) {
      return false;
    }

    // Null-aware left semi project with filter requires per-row
    // null analysis that depends on the filter result. Fall back
    // to CPU for correctness.
    if (joinPlanNode->isNullAware() &&
        joinPlanNode->joinType() ==
            core::JoinType::kLeftSemiProject &&
        joinPlanNode->filter()) {
      return false;
    }

    if (joinPlanNode->filter()) {
      if (!canBeEvaluatedByCudf(
              {joinPlanNode->filter()}, ctx->task->queryCtx().get())) {
        return false;
      }
      // Verify that all field references in the filter exist in the
      // combined left+right schema. Substrait plan conversion can produce
      // expressions that reference fields from a different node ID than
      // the join's actual probe/build output types.
      auto combinedType = type::concatRowTypes(
          {joinPlanNode->sources()[0]->outputType(),
           joinPlanNode->sources()[1]->outputType()});
      if (!allFieldsResolvable(joinPlanNode->filter(), combinedType)) {
        LOG(WARNING)
            << "CudfHashJoin: unresolvable field reference in filter '"
            << joinPlanNode->filter()->toString()
            << "', falling back to CPU";
        return false;
      }
    }
    return true;
  }
};

/// HashJoinBuildAdapter - Replaces with CudfHashJoinBuild
class HashJoinBuildAdapter : public CudfHashJoinBaseAdapter {
 public:
  HashJoinBuildAdapter() : CudfHashJoinBaseAdapter("HashJoinBuild") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

/// HashJoinProbeAdapter - Replaces with CudfHashJoinProbe
class HashJoinProbeAdapter : public CudfHashJoinBaseAdapter {
 public:
  HashJoinProbeAdapter() : CudfHashJoinBaseAdapter("HashJoinProbe") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashJoinProbe>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

class CudfNestedLoopJoinBaseAdapter : public OperatorAdapter {
 public:
  using OperatorAdapter::OperatorAdapter;

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      return false;
    }
    auto nlNode =
        std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
            planNode);
    if (!nlNode) {
      return false;
    }
    if (!CudfNestedLoopJoinProbe::isSupportedJoinType(
            nlNode->joinType())) {
      return false;
    }
    if (nlNode->joinCondition()) {
      if (!canBeEvaluatedByCudf(
              {nlNode->joinCondition()},
              ctx->task->queryCtx().get())) {
        return false;
      }
    }
    return true;
  }
};

class NestedLoopJoinBuildAdapter
    : public CudfNestedLoopJoinBaseAdapter {
 public:
  NestedLoopJoinBuildAdapter()
      : CudfNestedLoopJoinBaseAdapter("NestedLoopJoinBuild") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::NestedLoopJoinBuild*>(op) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }
  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator*,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto nlNode =
        std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
            planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfNestedLoopJoinBuild>(
            operatorId, ctx, nlNode));
    return result;
  }
};

class NestedLoopJoinProbeAdapter
    : public CudfNestedLoopJoinBaseAdapter {
 public:
  NestedLoopJoinProbeAdapter()
      : CudfNestedLoopJoinBaseAdapter("NestedLoopJoinProbe") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::NestedLoopJoinProbe*>(op) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }
  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator*,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto nlNode =
        std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(
            planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfNestedLoopJoinProbe>(
            operatorId, ctx, nlNode));
    return result;
  }
};

/// OrderByAdapter - Replaces with CudfOrderBy
class OrderByAdapter : public OperatorAdapter {
 public:
  OrderByAdapter() : OperatorAdapter("OrderBy") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::OrderBy*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::OrderByNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto orderByPlanNode =
        std::dynamic_pointer_cast<const core::OrderByNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfOrderBy>(operatorId, ctx, orderByPlanNode));
    return result;
  }
};

/// TopNAdapter - Replaces with CudfTopN
class TopNAdapter : public OperatorAdapter {
 public:
  TopNAdapter() : OperatorAdapter("TopN") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TopN*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::TopNNode>(planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto topNPlanNode =
        std::dynamic_pointer_cast<const core::TopNNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<CudfTopN>(operatorId, ctx, topNPlanNode));
    return result;
  }
};

/// LimitAdapter - Replaces with CudfLimit
class LimitAdapter : public OperatorAdapter {
 public:
  LimitAdapter() : OperatorAdapter("Limit") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Limit*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::LimitNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto limitPlanNode =
        std::dynamic_pointer_cast<const core::LimitNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLimit>(operatorId, ctx, limitPlanNode));
    return result;
  }
};

/// LocalPartitionAdapter - Conditionally replaces with CudfLocalPartition
class LocalPartitionAdapter : public OperatorAdapter {
 public:
  LocalPartitionAdapter() : OperatorAdapter("LocalPartition") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalPartition*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);
    return canHandle(op) && localPartitionPlanNode &&
        CudfLocalPartition::shouldReplace(localPartitionPlanNode);
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLocalPartition>(
            operatorId, ctx, localPartitionPlanNode));
    return result;
  }

  bool keepOperator() const override {
    return false;
  }
};

/// LocalExchangeAdapter - Keeps original operator
class LocalExchangeAdapter : public OperatorAdapter {
 public:
  LocalExchangeAdapter() : OperatorAdapter("LocalExchange") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalExchange*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// AssignUniqueIdAdapter - Replaces with CudfAssignUniqueId
class AssignUniqueIdAdapter : public OperatorAdapter {
 public:
  AssignUniqueIdAdapter() : OperatorAdapter("AssignUniqueId") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::AssignUniqueId*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(
               planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto assignUniqueIdPlanNode =
        std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfAssignUniqueId>(
            operatorId,
            ctx,
            assignUniqueIdPlanNode,
            assignUniqueIdPlanNode->taskUniqueId(),
            assignUniqueIdPlanNode->uniqueIdCounter()));
    return result;
  }
};

/// ValuesAdapter - Keeps original operator
class ValuesAdapter : public OperatorAdapter {
 public:
  ValuesAdapter() : OperatorAdapter("Values") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Values*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// ExpandAdapter - Replaces with CudfExpand
class ExpandAdapter : public OperatorAdapter {
 public:
  ExpandAdapter() : OperatorAdapter("Expand") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Expand*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<
               const core::ExpandNode>(planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto expandNode =
        std::dynamic_pointer_cast<const core::ExpandNode>(
            planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<CudfExpand>(
        operatorId, ctx, expandNode));
    return result;
  }
};

/// CallbackSinkAdapter - Keeps original operator
class CallbackSinkAdapter : public OperatorAdapter {
 public:
  CallbackSinkAdapter() : OperatorAdapter("CallbackSink") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::CallbackSink*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// TopNRowNumberAdapter - Replaces with CudfTopNRowNumber
class TopNRowNumberAdapter : public OperatorAdapter {
 public:
  TopNRowNumberAdapter() : OperatorAdapter("TopNRowNumber") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TopNRowNumber*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    if (!canHandle(op)) {
      return false;
    }
    auto node =
        std::dynamic_pointer_cast<const core::TopNRowNumberNode>(planNode);
    return node != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto node =
        std::dynamic_pointer_cast<const core::TopNRowNumberNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfTopNRowNumber>(operatorId, ctx, node));
    return result;
  }
};

/// WindowAdapter - Replaces with CudfWindow
class WindowAdapter : public OperatorAdapter {
 public:
  WindowAdapter() : OperatorAdapter("Window") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Window*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    if (!canHandle(op)) {
      return false;
    }
    auto windowNode =
        std::dynamic_pointer_cast<const core::WindowNode>(
            planNode);
    return windowNode != nullptr &&
        isSupportedCudfWindowNode(windowNode);
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>>
  createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto windowNode =
        std::dynamic_pointer_cast<const core::WindowNode>(
            planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<CudfWindow>(
        operatorId, ctx, windowNode));
    return result;
  }
};

/// Registration Function
void registerAllOperatorAdapters() {
  auto& registry = OperatorAdapterRegistry::getInstance();

  // Clear any existing adapters
  registry.clear();

  // Register all adapters
  registry.registerAdapter(std::make_unique<TableScanAdapter>());
  registry.registerAdapter(std::make_unique<FilterProjectAdapter>());
  registry.registerAdapter(std::make_unique<AggregationAdapter>());
  registry.registerAdapter(std::make_unique<HashJoinBuildAdapter>());
  registry.registerAdapter(std::make_unique<HashJoinProbeAdapter>());
  registry.registerAdapter(
      std::make_unique<NestedLoopJoinBuildAdapter>());
  registry.registerAdapter(
      std::make_unique<NestedLoopJoinProbeAdapter>());
  registry.registerAdapter(std::make_unique<OrderByAdapter>());
  registry.registerAdapter(std::make_unique<TopNAdapter>());
  registry.registerAdapter(std::make_unique<LimitAdapter>());
  registry.registerAdapter(std::make_unique<LocalPartitionAdapter>());
  registry.registerAdapter(std::make_unique<LocalExchangeAdapter>());
  registry.registerAdapter(std::make_unique<AssignUniqueIdAdapter>());
  registry.registerAdapter(std::make_unique<TopNRowNumberAdapter>());
  registry.registerAdapter(std::make_unique<ExpandAdapter>());
  registry.registerAdapter(std::make_unique<WindowAdapter>());
  registry.registerAdapter(std::make_unique<ValuesAdapter>());
  registry.registerAdapter(std::make_unique<CallbackSinkAdapter>());
}

} // namespace facebook::velox::cudf_velox
