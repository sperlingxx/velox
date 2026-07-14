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
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys, "16777216"},
      {CudfConfig::kCudfOrderBySortedRunBytes, "67108864"},
      {CudfConfig::kCudfOrderByMergeFanIn, "7"}};

  CudfConfig config;
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_EQ(config.groupbyStreamingMaxDistinctKeys, 16777216);
  ASSERT_EQ(config.orderBySortedRunBytes, 67108864);
  ASSERT_EQ(config.orderByMergeFanIn, 7);
}

TEST(ConfigTest, OrderByBounds) {
  CudfConfig defaultConfig;
  EXPECT_EQ(defaultConfig.orderBySortedRunBytes, 256ULL << 20);
  EXPECT_EQ(defaultConfig.orderByMergeFanIn, 8);

  CudfConfig zeroRunBytes;
  EXPECT_ANY_THROW(zeroRunBytes.initialize(
      {{CudfConfig::kCudfOrderBySortedRunBytes, "0"}}));

  CudfConfig lowFanIn;
  EXPECT_ANY_THROW(lowFanIn.initialize(
      {{CudfConfig::kCudfOrderByMergeFanIn, "1"}}));

  CudfConfig highFanIn;
  EXPECT_ANY_THROW(highFanIn.initialize(
      {{CudfConfig::kCudfOrderByMergeFanIn, "65"}}));
}

TEST(ConfigTest, GroupbyStreamingMaxDistinctKeysRange) {
  CudfConfig defaultConfig;
  ASSERT_EQ(defaultConfig.groupbyStreamingMaxDistinctKeys, 0);

  CudfConfig maxConfig;
  maxConfig.initialize({
      {CudfConfig::kCudfGroupbyStreamingMaxDistinctKeys,
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
