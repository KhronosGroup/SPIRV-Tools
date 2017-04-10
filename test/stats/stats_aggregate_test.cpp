// Copyright (c) 2017 Google Inc.
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

// Tests for unique type declaration rules validator.

#include <string>

#include "source/spirv_stats.h"
#include "test_fixture.h"
#include "unit_spirv.h"

namespace {

using libspirv::SpirvStats;
using spvtest::ScopedContext;

// Calls libspirv::AggregateStats for binary compiled from |code|.
void CompileAndAggregateStats(const std::string& code, SpirvStats* stats,
                    spv_target_env env = SPV_ENV_UNIVERSAL_1_1) {
  ScopedContext ctx(env);
  spv_binary binary;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(
      ctx.context, code.c_str(), code.size(), &binary, nullptr));

  ASSERT_EQ(SPV_SUCCESS, AggregateStats(*ctx.context, binary->code,
                                        binary->wordCount, nullptr, stats));
  spvBinaryDestroy(binary);
}

TEST(AggregateStats, CapabilityHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(4, stats.capability_hist.size());
  EXPECT_EQ(0, stats.capability_hist.count(SpvCapabilityShader));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5, stats.capability_hist.size());
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(5, stats.capability_hist.size());
  EXPECT_EQ(1, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(3, stats.capability_hist.at(SpvCapabilityLinkage));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5, stats.capability_hist.size());
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityShader));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityAddresses));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityKernel));
  EXPECT_EQ(2, stats.capability_hist.at(SpvCapabilityGenericPointer));
  EXPECT_EQ(4, stats.capability_hist.at(SpvCapabilityLinkage));
}

TEST(AggregateStats, ExtensionHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Physical32 OpenCL
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_viewport_array2"
OpExtension "greatest_extension_ever"
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1, stats.extension_hist.size());
  EXPECT_EQ(0, stats.extension_hist.count("SPV_NV_viewport_array2"));
  EXPECT_EQ(1, stats.extension_hist.at("SPV_KHR_16bit_storage"));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(3, stats.extension_hist.size());
  EXPECT_EQ(1, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(1, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(1, stats.extension_hist.at("greatest_extension_ever"));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(3, stats.extension_hist.size());
  EXPECT_EQ(1, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(2, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(1, stats.extension_hist.at("greatest_extension_ever"));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(3, stats.extension_hist.size());
  EXPECT_EQ(2, stats.extension_hist.at("SPV_NV_viewport_array2"));
  EXPECT_EQ(2, stats.extension_hist.at("SPV_KHR_16bit_storage"));
  EXPECT_EQ(2, stats.extension_hist.at("greatest_extension_ever"));
}

TEST(AggregateStats, VersionHistogram) {
  const std::string code1 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1, stats.version_hist.size());
  EXPECT_EQ(1, stats.version_hist.at(0x00010100));

  CompileAndAggregateStats(code1, &stats, SPV_ENV_UNIVERSAL_1_0);
  EXPECT_EQ(2, stats.version_hist.size());
  EXPECT_EQ(1, stats.version_hist.at(0x00010100));
  EXPECT_EQ(1, stats.version_hist.at(0x00010000));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(2, stats.version_hist.size());
  EXPECT_EQ(2, stats.version_hist.at(0x00010100));
  EXPECT_EQ(1, stats.version_hist.at(0x00010000));

  CompileAndAggregateStats(code1, &stats, SPV_ENV_UNIVERSAL_1_0);
  EXPECT_EQ(2, stats.version_hist.size());
  EXPECT_EQ(2, stats.version_hist.at(0x00010100));
  EXPECT_EQ(2, stats.version_hist.at(0x00010000));
}

TEST(AggregateStats, GeneratorHistogram) {
  const std::string code1 = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  const uint32_t kGeneratorKhronosAssembler =
      SPV_GENERATOR_KHRONOS_ASSEMBLER << 16;

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1, stats.generator_hist.size());
  EXPECT_EQ(1, stats.generator_hist.at(kGeneratorKhronosAssembler));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(1, stats.generator_hist.size());
  EXPECT_EQ(2, stats.generator_hist.at(kGeneratorKhronosAssembler));
}

TEST(AggregateStats, OpcodeHistogram) {
  const std::string code1 = R"(
OpCapability Addresses
OpCapability Kernel
OpCapability GenericPointer
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
%i32 = OpTypeInt 32 1
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
)";

  const std::string code2 = R"(
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_NV_viewport_array2"
OpMemoryModel Logical GLSL450
)";

  SpirvStats stats;

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(4, stats.opcode_hist.size());
  EXPECT_EQ(4, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(1, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(1, stats.opcode_hist.at(SpvOpTypeFloat));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5, stats.opcode_hist.size());
  EXPECT_EQ(6, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(1, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(1, stats.opcode_hist.at(SpvOpExtension));

  CompileAndAggregateStats(code1, &stats);
  EXPECT_EQ(5, stats.opcode_hist.size());
  EXPECT_EQ(10, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(3, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(4, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(1, stats.opcode_hist.at(SpvOpExtension));

  CompileAndAggregateStats(code2, &stats);
  EXPECT_EQ(5, stats.opcode_hist.size());
  EXPECT_EQ(12, stats.opcode_hist.at(SpvOpCapability));
  EXPECT_EQ(4, stats.opcode_hist.at(SpvOpMemoryModel));
  EXPECT_EQ(4, stats.opcode_hist.at(SpvOpTypeInt));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpTypeFloat));
  EXPECT_EQ(2, stats.opcode_hist.at(SpvOpExtension));
}

}  // namespace
