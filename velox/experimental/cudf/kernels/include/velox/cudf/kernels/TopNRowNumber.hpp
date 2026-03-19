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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <optional>
#include <vector>

namespace velox::cudf_kernels {

enum class RankFunction { kRowNumber, kRank, kDenseRank };

/// Compute top-N rows per partition using a ranking function on GPU.
///
/// The input table must already be sorted by (partition_keys, sorting_keys).
/// This function computes rank values within each partition and returns only
/// rows whose rank <= limit.
///
/// @param sorted_input  Already sorted table view.
/// @param partition_key_indices  Column indices of partition keys (may be
///                               empty for single-partition).
/// @param function  Ranking function to use.
/// @param limit  Per-partition limit (rows with rank > limit are dropped).
/// @param emit_rank  If true, an INT64 rank column is appended.
/// @param stream  CUDA stream.
/// @param mr  Device memory resource.
/// @return A pair: (filtered table, optional rank column). The filtered table
///         contains only input columns. The rank column, when requested, is
///         separate so the caller can place it at the desired output position.
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>>
topNRowNumber(
    cudf::table_view const& sorted_input,
    std::vector<cudf::size_type> const& partition_key_indices,
    RankFunction function,
    int32_t limit,
    bool emit_rank,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace velox::cudf_kernels
