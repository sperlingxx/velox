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
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfGpuTargetBatchRows{
      "cudf.gpu_target_batch_rows"};
  static constexpr const char* kCudfPinnedPoolSize{
      "cudf.pinned_pool_size"};
  static constexpr const char* kCudfHostAsPinnedThreshold{
      "cudf.host_as_pinned_threshold"};
  static constexpr const char* kCudfPackedDtoH{"cudf.packed_dtoh"};
  static constexpr const char* kCudfSkipOutputToVelox{
      "velox.cudf.skip_output_to_velox"};
  static constexpr const char* kCudfGpuTargetBatchBytes{
      "cudf.gpu_target_batch_bytes"};

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
  /// resources.
  int32_t memoryPercent{50};

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation
  bool jitExpressionEnabled{false};

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

  /// Size of the pinned host memory pool in bytes.
  /// Used for HtoD/DtoH transfers to achieve full PCIe bandwidth.
  /// 0 means use cudf default (0.5% of device memory, capped at 64 MB).
  size_t pinnedPoolSize{0};

  /// Threshold in bytes for allocating host memory as pinned.
  /// Host allocations <= this size use the pinned pool; larger ones use
  /// pageable. 0 disables (cudf default). Use SIZE_MAX to pin everything.
  size_t hostAsPinnedThreshold{0};

  /// Use cudf::pack for D2H transfers (single contiguous copy instead of
  /// per-buffer).  Disable to revert to legacy per-buffer path for A/B testing.
  bool packedDtoH{true};

  /// Target minimum number of rows for a GPU batch.  Operators that may
  /// produce or receive tiny batches (e.g. hash-probe after many small
  /// shuffled partitions, filter with high selectivity, or partial
  /// aggregation fed by scan with small row-groups) accumulate rows
  /// until this threshold is reached before launching GPU kernels.
  /// Set to 0 to disable accumulation (process each batch individually).
  int32_t gpuTargetBatchRows{1'000'000};

  /// Target minimum byte size for a GPU batch.  When non-zero, operators
  /// use this as the primary coalescing threshold instead of row counts.
  /// This produces better GPU utilization for wide tables where a small
  /// number of rows already consumes significant memory.
  /// Default 2 GiB.  Set to 0 to fall back to row-based accumulation.
  int64_t gpuTargetBatchBytes{2LL * 1024 * 1024 * 1024};
};

} // namespace facebook::velox::cudf_velox
