// Copyright (c) 2023 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tools/objdump/extract_source.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/util/cli_consumer.h"

namespace {

constexpr auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_6;

using ExtractedSource =
    std::pair<bool, std::unordered_map<std::string, std::string>>;

uint32_t SwapBytes(const uint32_t word) {
  return ((word & uint32_t{0x000000ff}) << 24) |
         ((word & uint32_t{0x0000ff00}) << 8) |
         ((word & uint32_t{0x00ff0000}) >> 8) |
         ((word & uint32_t{0xff000000}) >> 24);
}

ExtractedSource ExtractSourceFromBinary(const std::vector<uint32_t>& binary) {
  std::unordered_map<std::string, std::string> output;
  const bool result = ExtractSourceFromModule(binary, &output);
  return std::make_pair(result, std::move(output));
}

ExtractedSource ExtractSource(const std::string& spv_source) {
  std::unique_ptr<spvtools::opt::IRContext> ctx = spvtools::BuildModule(
      kDefaultEnvironment, spvtools::utils::CLIMessageConsumer, spv_source,
      spvtools::SpirvTools::kDefaultAssembleOption |
          SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  if (!ctx) {
    ADD_FAILURE() << "Failed to build module:\n" << spv_source;
    return {false, {}};
  }
  std::vector<uint32_t> binary;
  ctx->module()->ToBinary(&binary, /* skip_nop = */ false);

  const auto result = ExtractSourceFromBinary(binary);
  std::transform(binary.begin(), binary.end(), binary.begin(), SwapBytes);
  const auto swapped_result = ExtractSourceFromBinary(binary);
  EXPECT_EQ(result, swapped_result);
  return result;
}

}  // namespace

TEST(ExtractSourceTest, no_debug) {
  std::string source = R"(
           OpCapability Shader
           OpCapability Linkage
           OpMemoryModel Logical GLSL450
   %void = OpTypeVoid
      %2 = OpTypeFunction %void
   %bool = OpTypeBool
      %4 = OpUndef %bool
      %5 = OpFunction %void None %2
      %6 = OpLabel
           OpReturn
           OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 0);
}

TEST(ExtractSourceTest, SimpleSource) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2 "[numthreads(1, 1, 1)] void compute_1(){ }"
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(),
               "[numthreads(1, 1, 1)] void compute_1(){ }");
}

TEST(ExtractSourceTest, StringsWithEveryTerminatorByteOffset) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
 %1 = OpString "abcd"
 %2 = OpString "a"
 %3 = OpString "ab"
 %4 = OpString "abc"
      OpSource HLSL 660 %1 "WXYZ"
      OpSourceContinued "1234"
      OpSource HLSL 660 %2 "W"
      OpSourceContinued "1"
      OpSource HLSL 660 %3 "WX"
      OpSourceContinued "12"
      OpSource HLSL 660 %4 "WXY"
      OpSourceContinued "123"
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(4u, result.size());
  EXPECT_EQ("WXYZ1234", result.at("abcd"));
  EXPECT_EQ("W1", result.at("a"));
  EXPECT_EQ("WX12", result.at("ab"));
  EXPECT_EQ("WXY123", result.at("abc"));
}

TEST(ExtractSourceTest, EmptySourceAndContinued) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
 %1 = OpString "empty.hlsl"
 %2 = OpString "empty-continued.hlsl"
 %3 = OpString "prefix.hlsl"
      OpSource HLSL 660 %1 ""
      OpSource HLSL 660 %2 ""
      OpSourceContinued ""
      OpSource HLSL 660 %3 "prefix"
      OpSourceContinued ""
  )";

  auto [success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(3u, result.size());
  EXPECT_EQ(std::string{}, result.at("empty.hlsl"));
  EXPECT_EQ(std::string{}, result.at("empty-continued.hlsl"));
  EXPECT_EQ(std::string("prefix"), result.at("prefix.hlsl"));
}

TEST(ExtractSourceTest, Utf8AcrossWordBoundaries) {
  // Each UTF-8 sequence starts at byte 3 and crosses a SPIR-V word boundary.
  const std::string filename = "abc\xc3\xa9.hlsl";
  const std::string prefix = "ABC\xe2\x82\xac";
  const std::string continuation = "123\xf0\x9f\x98\x80";
  const std::string source =
      "OpCapability Shader\n"
      "OpMemoryModel Logical GLSL450\n"
      "%1 = OpString \"" +
      filename + "\"\nOpSource HLSL 660 %1 \"" + prefix +
      "\"\nOpSourceContinued \"" + continuation + "\"\n";

  auto [success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(1u, result.size());
  EXPECT_EQ(prefix + continuation, result.at(filename));
}

TEST(ExtractSourceTest, EscapesAcrossWordBoundaries) {
  // The assembler consumes one escape layer. Each binary escape pair starts
  // at byte 3, so its two bytes straddle a word boundary.
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
 %1 = OpString "abc\\\".hlsl"
 %2 = OpString "abc\\\\.hlsl"
      OpSource HLSL 660 %1 "ABC\\\"tail"
      OpSourceContinued "123\\\"tail"
      OpSource HLSL 660 %2 "ABC\\\\tail"
      OpSourceContinued "123\\\\tail"
  )";

  auto [success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(2u, result.size());
  EXPECT_EQ(std::string("ABC\"tail123\"tail"), result.at("abc\".hlsl"));
  EXPECT_EQ(std::string("ABC\\tail123\\tail"), result.at("abc\\.hlsl"));
}

TEST(ExtractSourceTest, TrailingBackslashWithEveryTerminatorByteOffset) {
  // Each assembly backslash pair becomes one trailing binary backslash.
  // The four strings terminate at byte offsets 0, 1, 2, and 3, respectively.
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
 %1 = OpString "abc\\"
 %2 = OpString "\\"
 %3 = OpString "a\\"
 %4 = OpString "ab\\"
      OpSource HLSL 660 %1 "ABC\\"
      OpSourceContinued "123\\"
      OpSource HLSL 660 %2 "\\"
      OpSourceContinued "\\"
      OpSource HLSL 660 %3 "A\\"
      OpSourceContinued "1\\"
      OpSource HLSL 660 %4 "AB\\"
      OpSourceContinued "12\\"
  )";

  auto [success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(4u, result.size());
  EXPECT_EQ(std::string("ABC\\123\\"), result.at("abc\\"));
  EXPECT_EQ(std::string("\\\\"), result.at("\\"));
  EXPECT_EQ(std::string("A\\1\\"), result.at("a\\"));
  EXPECT_EQ(std::string("AB\\12\\"), result.at("ab\\"));
}

TEST(ExtractSourceTest, SourceWithOpenCLImport) {
  std::string source = R"(
      OpCapability Addresses
      OpCapability Kernel
 %1 = OpExtInstImport "OpenCL.std"
      OpMemoryModel Physical64 OpenCL
 %2 = OpString "kernel.cl"
      OpSource OpenCL_C 200000 %2 "kernel void compute() {}"
  )";

  auto [success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_EQ(1u, result.size());
  EXPECT_EQ(std::string("kernel void compute() {}"), result.at("kernel.cl"));
}

TEST(ExtractSourceTest, SourceContinued) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2 "[numthreads(1, 1, 1)] "
      OpSourceContinued "void compute_1(){ }"
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(),
               "[numthreads(1, 1, 1)] void compute_1(){ }");
}

TEST(ExtractSourceTest, OnlyFilename) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute.hlsl"
      OpSource HLSL 660 %2
      OpName %1 "compute_1"
 %3 = OpTypeVoid
 %4 = OpTypeFunction %3
 %1 = OpFunction %3 None %4
 %5 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(), "");
}

TEST(ExtractSourceTest, MultipleFiles) {
  std::string source = R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %1 "compute_1"
      OpExecutionMode %1 LocalSize 1 1 1
 %2 = OpString "compute1.hlsl"
 %3 = OpString "compute2.hlsl"
      OpSource HLSL 660 %2 "some instruction"
      OpSource HLSL 660 %3 "some other instruction"
      OpName %1 "compute_1"
 %4 = OpTypeVoid
 %5 = OpTypeFunction %4
 %1 = OpFunction %4 None %5
 %6 = OpLabel
      OpLine %2 1 41
      OpReturn
      OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 2);
  std::string compute1HlslResult = result["compute1.hlsl"];
  ASSERT_STREQ(compute1HlslResult.c_str(), "some instruction");
  std::string compute2HlslResult = result["compute2.hlsl"];
  ASSERT_STREQ(compute2HlslResult.c_str(), "some other instruction");
}

TEST(ExtractSourceTest, MultilineCode) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute_1"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2 "[numthreads(1, 1, 1)]
void compute_1() {
}
"
               OpName %1 "compute_1"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 3 1
               OpReturn
               OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(),
               "[numthreads(1, 1, 1)]\nvoid compute_1() {\n}\n");
}

TEST(ExtractSourceTest, EmptyFilename) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute_1"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString ""
               OpSource HLSL 660 %2 "void compute(){}"
               OpName %1 "compute_1"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 3 1
               OpReturn
               OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string unnamed0HlslResult = result["unnamed-0.hlsl"];
  ASSERT_STREQ(unnamed0HlslResult.c_str(), "void compute(){}");
}

TEST(ExtractSourceTest, EscapeEscaped) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2 "// check \" and \\\\ escapes removed"
               OpName %1 "compute"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 6 1
               OpReturn
               OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(), "// check \" and \\ escapes removed");
}

TEST(ExtractSourceTest, OpSourceWithNoSource) {
  std::string source = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %1 "compute"
               OpExecutionMode %1 LocalSize 1 1 1
          %2 = OpString "compute.hlsl"
               OpSource HLSL 660 %2
               OpName %1 "compute"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %1 = OpFunction %3 None %4
          %5 = OpLabel
               OpLine %2 6 1
               OpReturn
               OpFunctionEnd
  )";

  auto[success, result] = ExtractSource(source);
  ASSERT_TRUE(success);
  ASSERT_TRUE(result.size() == 1);
  std::string computeHlslResult = result["compute.hlsl"];
  ASSERT_STREQ(computeHlslResult.c_str(), "");
}
