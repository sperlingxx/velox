/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using cudf::io::parquet::experimental::hybrid_scan_multifile;
using ByteRange = cudf::io::text::byte_range_info;

struct PreparedSources {
  std::vector<std::unique_ptr<cudf::io::datasource>> datasources;
  std::vector<std::reference_wrapper<cudf::io::datasource>> datasourceRefs;
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footerBuffers;
  std::vector<cudf::host_span<const uint8_t>> footerSpans;
};

struct DeviceRanges {
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<const uint8_t>> spans;
};

std::unique_ptr<cudf::column> makeIntColumn(
    const std::vector<int32_t>& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  rmm::device_buffer data(values.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      data.size(),
      cudaMemcpyHostToDevice,
      stream.value()));
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(values.size()),
      std::move(data),
      rmm::device_buffer{},
      0);
}

std::vector<char> writeFile(
    int32_t firstValue,
    int32_t sourceValue,
    cudf::size_type numRows,
    cudf::size_type rowsPerGroup) {
  std::vector<int32_t> values(numRows);
  std::iota(values.begin(), values.end(), firstValue);
  std::vector<int32_t> sources(numRows, sourceValue);

  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.emplace_back(makeIntColumn(values, stream, mr));
  columns.emplace_back(makeIntColumn(sources, stream, mr));
  cudf::table table{std::move(columns)};

  cudf::io::table_input_metadata metadata(table.view());
  metadata.column_metadata[0].set_name("value");
  metadata.column_metadata[1].set_name("source");

  std::vector<char> output;
  cudf::io::parquet_writer_options options =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{&output}, table.view())
          .metadata(std::move(metadata))
          .row_group_size_rows(rowsPerGroup)
          .max_page_size_rows(rowsPerGroup / 4)
          .compression(cudf::io::compression_type::SNAPPY)
          .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(options);
  return output;
}

cudf::io::source_info makeSourceInfo(
    const std::vector<std::vector<char>>& files) {
  std::vector<cudf::host_span<const char>> spans;
  spans.reserve(files.size());
  for (const auto& file : files) {
    spans.emplace_back(file.data(), file.size());
  }
  return cudf::io::source_info{
      cudf::host_span<cudf::host_span<const char>>{spans}};
}

PreparedSources prepareSources(const cudf::io::source_info& sourceInfo) {
  PreparedSources inputs;
  inputs.datasources = cudf::io::make_datasources(sourceInfo);
  inputs.datasourceRefs.reserve(inputs.datasources.size());
  inputs.footerBuffers.reserve(inputs.datasources.size());
  inputs.footerSpans.reserve(inputs.datasources.size());

  for (const auto& source : inputs.datasources) {
    inputs.datasourceRefs.emplace_back(*source);
    inputs.footerBuffers.emplace_back(
        cudf::io::parquet::fetch_footer_to_host(inputs.datasourceRefs.back()));
    inputs.footerSpans.emplace_back(*inputs.footerBuffers.back());
  }
  return inputs;
}

DeviceRanges fetchRanges(
    const PreparedSources& inputs,
    const std::pair<std::vector<ByteRange>, std::vector<cudf::size_type>>&
        planned,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto& [ranges, sourceIndices] = planned;
  CUDF_EXPECTS(ranges.size() == sourceIndices.size(), "Invalid source map");

  std::vector<std::vector<ByteRange>> perSource(inputs.datasources.size());
  for (size_t i = 0; i < ranges.size(); ++i) {
    CUDF_EXPECTS(
        sourceIndices[i] >= 0 &&
            static_cast<size_t>(sourceIndices[i]) < inputs.datasources.size(),
        "Invalid source index");
    perSource[sourceIndices[i]].push_back(ranges[i]);
  }

  auto [buffers, perSourceSpans, reads] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          inputs.datasourceRefs, perSource, stream, mr);
  reads.get();

  std::vector<cudf::device_span<const uint8_t>> flatSpans;
  flatSpans.reserve(ranges.size());
  for (const auto& sourceSpans : perSourceSpans) {
    flatSpans.insert(flatSpans.end(), sourceSpans.begin(), sourceSpans.end());
  }
  CUDF_EXPECTS(ranges.size() == flatSpans.size(), "Invalid device span map");
  return {std::move(buffers), std::move(flatSpans)};
}

TEST(HybridScanMultifileApiTest, NonPrefixRowGroupsProduceMultipleChunks) {
  constexpr cudf::size_type kRowsPerFile = 60'000;
  constexpr cudf::size_type kRowsPerGroup = 20'000;

  std::vector<std::vector<char>> files;
  files.emplace_back(writeFile(0, 10, kRowsPerFile, kRowsPerGroup));
  files.emplace_back(writeFile(100'000, 20, kRowsPerFile, kRowsPerGroup));

  auto sourceInfo = makeSourceInfo(files);
  auto inputs = prepareSources(sourceInfo);
  auto options = cudf::io::parquet_reader_options::builder().build();
  hybrid_scan_multifile reader{inputs.footerSpans, options};

  const auto allRowGroups = reader.all_row_groups(options);
  ASSERT_EQ(allRowGroups.size(), 2);
  EXPECT_EQ(allRowGroups[0], (std::vector<cudf::size_type>{0, 1, 2}));
  EXPECT_EQ(allRowGroups[1], (std::vector<cudf::size_type>{0, 1, 2}));

  // Select neither source's prefix: file 0 RG 2 and file 1 RG 1.
  const std::vector<std::vector<cudf::size_type>> selected{{2}, {1}};
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const auto planned = reader.all_column_chunks_byte_ranges(selected, options);
  ASSERT_FALSE(planned.first.empty());
  ASSERT_EQ(planned.first.size(), planned.second.size());

  auto deviceRanges = fetchRanges(inputs, planned, stream, mr);
  constexpr size_t kOneRowGroupOutputBytes =
      kRowsPerGroup * 2 * sizeof(int32_t);
  reader.setup_chunking_for_all_columns(
      kOneRowGroupOutputBytes,
      0,
      selected,
      deviceRanges.spans,
      options,
      stream,
      mr);

  std::vector<std::unique_ptr<cudf::table>> chunks;
  while (reader.has_next_table_chunk()) {
    chunks.emplace_back(reader.materialize_all_columns_chunk().tbl);
  }
  ASSERT_GE(chunks.size(), 2);

  std::vector<cudf::table_view> chunkViews;
  chunkViews.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    chunkViews.emplace_back(chunk->view());
  }
  auto actual = cudf::concatenate(chunkViews, stream, mr);

  std::vector<int32_t> expectedValues(2 * kRowsPerGroup);
  std::iota(
      expectedValues.begin(),
      expectedValues.begin() + kRowsPerGroup,
      2 * kRowsPerGroup);
  std::iota(
      expectedValues.begin() + kRowsPerGroup,
      expectedValues.end(),
      100'000 + kRowsPerGroup);
  std::vector<int32_t> expectedSources(2 * kRowsPerGroup, 10);
  std::fill(expectedSources.begin() + kRowsPerGroup, expectedSources.end(), 20);
  ASSERT_EQ(actual->num_columns(), 2);
  ASSERT_EQ(actual->num_rows(), expectedValues.size());
  std::vector<int32_t> actualValues(actual->num_rows());
  std::vector<int32_t> actualSources(actual->num_rows());
  const auto actualView = actual->view();
  ASSERT_EQ(actualView.column(0).type().id(), cudf::type_id::INT32);
  ASSERT_EQ(actualView.column(1).type().id(), cudf::type_id::INT32);
  ASSERT_FALSE(actualView.column(0).has_nulls());
  ASSERT_FALSE(actualView.column(1).has_nulls());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actualValues.data(),
      actualView.column(0).data<int32_t>(),
      actualValues.size() * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actualSources.data(),
      actualView.column(1).data<int32_t>(),
      actualSources.size() * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  EXPECT_EQ(actualValues, expectedValues);
  EXPECT_EQ(actualSources, expectedSources);
}

} // namespace
