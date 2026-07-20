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

#include <gtest/gtest.h>

#include <limits>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, CudfConfig) {
  CudfConfig defaults;
  EXPECT_FALSE(defaults.concatOptimizationEnabled);
  EXPECT_TRUE(defaults.exchangeConcatOptimizationEnabled);
  EXPECT_EQ(defaults.batchSizeMinThreshold, 100000);
  EXPECT_EQ(defaults.exchangeBatchSizeMinThreshold, 32000000);

  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys, "16777216"},
      {CudfConfig::kCudfOrderBySortedRunBytes, "67108864"},
      {CudfConfig::kCudfOrderByOutputChunkBytes, "134217728"},
      {CudfConfig::kCudfOrderByMaxOutputRows, "1048576"},
      {CudfConfig::kCudfExchangeConcatOptimizationEnabled, "false"},
      {CudfConfig::kCudfOrderByMergeFanIn, "7"},
      {CudfConfig::kCudfWindowSortedRunBytes, "134217728"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.exchangeConcatOptimizationEnabled, false);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.groupbyStreamingMaxDistinctKeys, 16777216);
  ASSERT_EQ(config.orderBySortedRunBytes, 67108864);
  ASSERT_EQ(config.orderByMergeFanIn, 7);
  ASSERT_EQ(config.windowSortedRunBytes, 134217728);
  ASSERT_EQ(config.orderByOutputChunkBytes, 134217728);
  ASSERT_EQ(config.orderByMaxOutputRows, 1048576);
}

TEST(ConfigTest, WindowBounds) {
  CudfConfig defaultConfig;
  EXPECT_EQ(defaultConfig.windowSortedRunBytes, 3ULL << 30);

  CudfConfig zeroRunBytes;
  EXPECT_ANY_THROW(
      zeroRunBytes.initialize({{CudfConfig::kCudfWindowSortedRunBytes, "0"}}));
}

TEST(ConfigTest, OrderByBounds) {
  CudfConfig defaultConfig;
  EXPECT_EQ(defaultConfig.orderBySortedRunBytes, 256ULL << 20);
  EXPECT_EQ(defaultConfig.orderByMergeFanIn, 8);
  EXPECT_EQ(defaultConfig.orderByOutputChunkBytes, 32ULL << 20);
  EXPECT_EQ(defaultConfig.orderByMaxOutputRows, 262144);

  CudfConfig zeroRunBytes;
  EXPECT_ANY_THROW(
      zeroRunBytes.initialize({{CudfConfig::kCudfOrderBySortedRunBytes, "0"}}));

  CudfConfig lowFanIn;
  EXPECT_ANY_THROW(
      lowFanIn.initialize({{CudfConfig::kCudfOrderByMergeFanIn, "1"}}));

  CudfConfig highFanIn;
  EXPECT_ANY_THROW(
      highFanIn.initialize({{CudfConfig::kCudfOrderByMergeFanIn, "65"}}));

  CudfConfig zeroOutputBytes;
  EXPECT_ANY_THROW(zeroOutputBytes.initialize(
      {{CudfConfig::kCudfOrderByOutputChunkBytes, "0"}}));

  CudfConfig zeroOutputRows;
  EXPECT_ANY_THROW(zeroOutputRows.initialize(
      {{CudfConfig::kCudfOrderByMaxOutputRows, "0"}}));

  CudfConfig overflowOutputRows;
  EXPECT_ANY_THROW(overflowOutputRows.initialize(
      {{CudfConfig::kCudfOrderByMaxOutputRows, "2147483648"}}));
}

TEST(ConfigTest, GroupbyStreamingMaxDistinctKeysRange) {
  CudfConfig defaultConfig;
  ASSERT_EQ(defaultConfig.groupbyStreamingMaxDistinctKeys, 0);

  CudfConfig maxConfig;
  maxConfig.initialize(
      {{CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys,
        std::to_string(std::numeric_limits<int32_t>::max())}});
  ASSERT_EQ(
      maxConfig.groupbyStreamingMaxDistinctKeys,
      std::numeric_limits<int32_t>::max());

  CudfConfig negativeConfig;
  EXPECT_ANY_THROW(negativeConfig.initialize(
      {{CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys, "-1"}}));

  CudfConfig overflowConfig;
  EXPECT_ANY_THROW(overflowConfig.initialize(
      {{CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys, "2147483648"}}));
}

} // namespace facebook::velox::cudf_velox::test
