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

#include "velox/cudf/kernels/TopNRowNumber.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

namespace velox::cudf_kernels {

namespace {

std::unique_ptr<cudf::column> computeRowNumbers(
    cudf::table_view const& sorted_input,
    std::vector<cudf::size_type> const& partition_key_indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const num_rows = sorted_input.num_rows();

  if (partition_key_indices.empty()) {
    auto col = cudf::sequence(
        num_rows,
        cudf::numeric_scalar<int64_t>(1, true, stream),
        cudf::numeric_scalar<int64_t>(1, true, stream),
        stream,
        mr);
    return col;
  }

  // Build a ones column, then groupby-scan(SUM) = cumulative row number.
  auto ones = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      num_rows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto ones_scalar = cudf::numeric_scalar<int64_t>(1, true, stream);
  auto ones_mv = ones->mutable_view();
  cudf::fill_in_place(ones_mv, 0, num_rows, ones_scalar);

  cudf::table_view keys = sorted_input.select(partition_key_indices);
  cudf::groupby::groupby gb(keys, cudf::null_policy::INCLUDE,
                            cudf::sorted::YES);

  std::vector<cudf::groupby::scan_request> requests;
  cudf::groupby::scan_request req;
  req.values = ones->view();
  req.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>());
  requests.push_back(std::move(req));

  auto [group_keys, results] = gb.scan(requests);
  return std::move(results[0].results[0]);
}

std::unique_ptr<cudf::column> computeRank(
    cudf::table_view const& sorted_input,
    std::vector<cudf::size_type> const& partition_key_indices,
    RankFunction function,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (function == RankFunction::kRowNumber) {
    return computeRowNumbers(sorted_input, partition_key_indices, stream, mr);
  }

  auto const num_rows = sorted_input.num_rows();

  // Determine cudf rank_method.
  cudf::rank_method method = (function == RankFunction::kRank)
      ? cudf::rank_method::MIN
      : cudf::rank_method::DENSE;

  // Build a ones column as the "values" for the scan.
  auto ones = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      num_rows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto ones_scalar = cudf::numeric_scalar<int64_t>(1, true, stream);
  auto ones_mv = ones->mutable_view();
  cudf::fill_in_place(ones_mv, 0, num_rows, ones_scalar);

  cudf::table_view keys;
  std::unique_ptr<cudf::column> const_key_col;
  if (partition_key_indices.empty()) {
    // No partitions: create a constant key so groupby treats all rows as one.
    const_key_col = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        num_rows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    auto zero = cudf::numeric_scalar<int32_t>(0, true, stream);
    auto ck_mv = const_key_col->mutable_view();
    cudf::fill_in_place(ck_mv, 0, num_rows, zero);
    keys = cudf::table_view{{const_key_col->view()}};
  } else {
    keys = sorted_input.select(partition_key_indices);
  }

  cudf::groupby::groupby gb(keys, cudf::null_policy::INCLUDE,
                            cudf::sorted::YES);

  std::vector<cudf::groupby::scan_request> requests;
  cudf::groupby::scan_request req;
  req.values = ones->view();
  req.aggregations.push_back(
      cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(method));
  requests.push_back(std::move(req));

  auto [group_keys_out, results] = gb.scan(requests);
  return std::move(results[0].results[0]);
}

} // namespace

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>>
topNRowNumber(
    cudf::table_view const& sorted_input,
    std::vector<cudf::size_type> const& partition_key_indices,
    RankFunction function,
    int32_t limit,
    bool emit_rank,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const num_rows = sorted_input.num_rows();

  if (num_rows == 0 || limit <= 0) {
    auto empty = cudf::empty_like(sorted_input);
    std::unique_ptr<cudf::column> rank_col;
    if (emit_rank) {
      rank_col = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT64}, 0,
          cudf::mask_state::UNALLOCATED, stream, mr);
    }
    return {std::move(empty), std::move(rank_col)};
  }

  auto ranks = computeRank(
      sorted_input, partition_key_indices, function, stream, mr);

  // Build boolean mask: rank <= limit
  auto limit_scalar = cudf::numeric_scalar<int64_t>(limit, true, stream);
  auto mask = cudf::binary_operation(
      ranks->view(),
      limit_scalar,
      cudf::binary_operator::LESS_EQUAL,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);

  auto filtered = cudf::apply_boolean_mask(sorted_input, mask->view(),
                                           stream, mr);

  std::unique_ptr<cudf::column> rank_out;
  if (emit_rank) {
    auto rank_table = cudf::apply_boolean_mask(
        cudf::table_view{{ranks->view()}}, mask->view(), stream, mr);
    auto cols = rank_table->release();
    rank_out = std::move(cols[0]);
  }

  return {std::move(filtered), std::move(rank_out)};
}

} // namespace velox::cudf_kernels
