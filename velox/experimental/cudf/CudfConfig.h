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

#pragma once

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfExchangeBatchSizeMinThreshold{
      "cudf.exchange_batch_size_min_threshold"};
  static constexpr const char* kCudfBatchSizeMinThresholdBytes{
      "cudf.batch_size_min_threshold_bytes"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfGroupbyStreamingMaxDistinctKeys{
      "cudf.groupby_streaming_max_distinct_keys"};
  static constexpr const char* kCudfOrderBySortedRunBytes{
      "cudf.order_by_sorted_run_bytes"};
  static constexpr const char* kCudfOrderByMergeFanIn{
      "cudf.order_by_merge_fan_in"};
  static constexpr const char* kCudfWindowSortedRunBytes{
      "cudf.window.sorted_run_bytes"};
  static constexpr const char* kCudfOrderByOutputChunkBytes{
      "cudf.order_by_output_chunk_bytes"};
  static constexpr const char* kCudfOrderByMaxOutputRows{
      "cudf.order_by_max_output_rows"};
  static constexpr const char* kCudfExchangeConcatOptimizationEnabled{
      "cudf.exchange_concat_optimization_enabled"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  // The value could be either spark or presto.
  static constexpr const char* kCudfFunctionEngine{"cudf.function_engine"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};
  // Maximum retained candidate bytes before CudfTopNRowNumber writes a sorted
  // run. The value must be a plain unsigned decimal byte count with no unit
  // suffix; the default is 134217728 bytes (128 MiB). If this threshold causes
  // a spill, the Task must be configured with an explicit spill root.
  static constexpr const char* kCudfTopNRowNumberCandidateRunBytes{
      "cudf.topn_row_number.candidate_run_bytes"};
  // Hard limit for the complete build table retained by CudfNestedLoopJoin.
  // The default is intentionally unbounded for non-MPP callers; Gluten MPP
  // supplies a conservative value for replicated Cartesian joins.
  static constexpr const char* kCudfNestedLoopJoinMaxBuildBytes{
      "cudf.nested_loop_join.max_build_bytes"};
  static constexpr const char* kCudfSkipOutputToVelox{
      "velox.cudf.skip_output_to_velox"};

  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingPolling{"ucxx.blocking_polling"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources, and the retained-memory release threshold for async.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Register Spark or Presto specific cuDF functions.
  std::string functionEngine{"presto"};

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat before supported Cudf operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators by collecting a minimum number
  /// of rows before concatenating and passing on to the next operator.
  /// This batch size is determined by batchSizeMinThreshold and
  /// batchSizeMaxThreshold
  bool concatOptimizationEnabled{false};

  /// Maximum distinct keys retained by FINAL streaming groupby. Zero keeps
  /// the all-GPU levelled aggregation path and does not construct streaming
  /// state.
  int32_t groupbyStreamingMaxDistinctKeys{0};

  /// Approximate bytes buffered before OrderBy spills an independently sorted
  /// run.
  uint64_t orderBySortedRunBytes{256ULL << 20};

  /// Number of sorted runs compacted by one OrderBy merge group.
  int32_t orderByMergeFanIn{8};

  /// Approximate bytes buffered before a partitioned Window writes an
  /// independently sorted run.
  uint64_t windowSortedRunBytes{3ULL << 30};

  /// Maximum bytes and rows returned by one OrderBy output batch.
  uint64_t orderByOutputChunkBytes{32ULL << 20};
  int32_t orderByMaxOutputRows{262144};

  /// Minimum rows to accumulate before GPU-side concatenation in
  /// `CudfBatchConcat` (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Target rows for CudfBatchConcat immediately after a UCX Exchange. Keep
  /// this below the aggregation target because a fragment can buffer several
  /// inbound exchanges concurrently while its join builds are still live.
  int32_t exchangeBatchSizeMinThreshold{32000000};

  /// Target bytes to accumulate before GPU-side concatenation. Zero disables
  /// the byte threshold. When both row and byte thresholds are configured,
  /// reaching either one flushes the batch.
  uint64_t batchSizeMinThresholdBytes{0};

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;

  /// Whether UCX exchange is enabled.
  bool exchange{false};

  /// Whether to enable error handling in UCXX endpoints.
  bool ucxxErrorHandling{true};

  /// Whether intra-node exchange optimization is enabled.
  bool intraNodeExchange{false};

  /// Whether to use blocking polling in UCXX.
  bool ucxxBlockingPolling{true};

  /// VLOG level for ucx-exchange source files.
  int32_t exchangeLogLevel{0};

  /// Whether to insert CudfBatchConcat immediately after ordinary UCX
  /// Exchange sources. Keep this independent from aggregation-side concat.
  /// This field is intentionally appended so incremental builds that reuse
  /// stable UCX objects preserve the offsets of every pre-existing field.
  bool exchangeConcatOptimizationEnabled{true};
};

} // namespace facebook::velox::cudf_velox
