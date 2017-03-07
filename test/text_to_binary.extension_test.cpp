// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Assembler tests for instructions in the "Extension Instruction" section
// of the SPIR-V spec.

#include "unit_spirv.h"

#include "test_fixture.h"
#include "gmock/gmock.h"
#include "spirv/1.0/GLSL.std.450.h"
#include "spirv/1.0/OpenCL.std.h"

namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using spvtest::TextToBinaryTest;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Values;
using ::testing::ValuesIn;

TEST_F(TextToBinaryTest, InvalidExtInstImportName) {
  EXPECT_THAT(CompileFailure("%1 = OpExtInstImport \"Haskell.std\""),
              Eq("Invalid extended instruction import 'Haskell.std'"));
}

TEST_F(TextToBinaryTest, InvalidImportId) {
  EXPECT_THAT(CompileFailure("%1 = OpTypeVoid\n"
                             "%2 = OpExtInst %1 %1"),
              Eq("Invalid extended instruction import Id 2"));
}

TEST_F(TextToBinaryTest, InvalidImportInstruction) {
  const std::string input = R"(%1 = OpTypeVoid
                               %2 = OpExtInstImport "OpenCL.std"
                               %3 = OpExtInst %1 %2 not_in_the_opencl)";
  EXPECT_THAT(CompileFailure(input),
              Eq("Invalid extended instruction name 'not_in_the_opencl'."));
}

TEST_F(TextToBinaryTest, MultiImport) {
  const std::string input = R"(%2 = OpExtInstImport "OpenCL.std"
                               %2 = OpExtInstImport "OpenCL.std")";
  EXPECT_THAT(CompileFailure(input),
              Eq("Import Id is being defined a second time"));
}

TEST_F(TextToBinaryTest, TooManyArguments) {
  const std::string input = R"(%opencl = OpExtInstImport "OpenCL.std"
                               %2 = OpExtInst %float %opencl cos %x %oops")";
  EXPECT_THAT(CompileFailure(input), Eq("Expected '=', found end of stream."));
}

TEST_F(TextToBinaryTest, ExtInstFromTwoDifferentImports) {
  const std::string input = R"(%1 = OpExtInstImport "OpenCL.std"
%2 = OpExtInstImport "GLSL.std.450"
%4 = OpExtInst %3 %1 native_sqrt %5
%7 = OpExtInst %6 %2 MatrixInverse %8
)";

  // Make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate({
          MakeInstruction(SpvOpExtInstImport, {1}, MakeVector("OpenCL.std")),
          MakeInstruction(SpvOpExtInstImport, {2}, MakeVector("GLSL.std.450")),
          MakeInstruction(
              SpvOpExtInst,
              {3, 4, 1, uint32_t(OpenCLLIB::Entrypoints::Native_sqrt), 5}),
          MakeInstruction(SpvOpExtInst,
                          {6, 7, 2, uint32_t(GLSLstd450MatrixInverse), 8}),
      })));

  // Make sure it disassembles correctly.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), Eq(input));
}


// A test case for assembling into words in an instruction.
struct AssemblyCase {
  std::string input;
  std::vector<uint32_t> expected;
};

using ExtensionRoundTripTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::tuple<spv_target_env, AssemblyCase>>>;

TEST_P(ExtensionRoundTripTest, Samples) {
  const spv_target_env& env = std::get<0>(GetParam());
  const AssemblyCase& ac = std::get<1>(GetParam());

  // Check that it assembles correctly.
  EXPECT_THAT(CompiledInstructions(ac.input, env), Eq(ac.expected));

  // Check round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(ac.input,
                                          SPV_BINARY_TO_TEXT_OPTION_NONE, env),
              Eq(ac.input));
}

// SPV_KHR_shader_ballot

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_shader_ballot, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability SubgroupBallotKHR\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilitySubgroupBallotKHR})},
                {"%2 = OpSubgroupBallotKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupBallotKHR, {1, 2, 3})},
                {"%2 = OpSubgroupFirstInvocationKHR %1 %3\n",
                 MakeInstruction(SpvOpSubgroupFirstInvocationKHR, {1, 2, 3})},
                {"OpDecorate %1 BuiltIn SubgroupEqMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupEqMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupGeMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGeMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupGtMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupGtMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupLeMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLeMaskKHR})},
                {"OpDecorate %1 BuiltIn SubgroupLtMaskKHR\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInSubgroupLtMaskKHR})},
            })), );

// SPV_KHR_shader_draw_parameters

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_shader_draw_parameters, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpCapability DrawParameters\n",
             MakeInstruction(SpvOpCapability, {SpvCapabilityDrawParameters})},
            {"OpDecorate %1 BuiltIn BaseVertex\n",
             MakeInstruction(SpvOpDecorate,
                             {1, SpvDecorationBuiltIn, SpvBuiltInBaseVertex})},
            {"OpDecorate %1 BuiltIn BaseInstance\n",
             MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                             SpvBuiltInBaseInstance})},
            {"OpDecorate %1 BuiltIn DrawIndex\n",
             MakeInstruction(SpvOpDecorate,
                             {1, SpvDecorationBuiltIn, SpvBuiltInDrawIndex})},
        })), );

// SPV_KHR_subgroup_vote

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_subgroup_vote, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(
        Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
               SPV_ENV_VULKAN_1_0),
        ValuesIn(std::vector<AssemblyCase>{
            {"OpCapability SubgroupVoteKHR\n",
             MakeInstruction(SpvOpCapability, {SpvCapabilitySubgroupVoteKHR})},
            {"%2 = OpSubgroupAnyKHR %1 %3\n",
             MakeInstruction(SpvOpSubgroupAnyKHR, {1, 2, 3})},
            {"%2 = OpSubgroupAllKHR %1 %3\n",
             MakeInstruction(SpvOpSubgroupAllKHR, {1, 2, 3})},
            {"%2 = OpSubgroupAllEqualKHR %1 %3\n",
             MakeInstruction(SpvOpSubgroupAllEqualKHR, {1, 2, 3})},
        })), );

// SPV_KHR_16bit_storage

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_16bit_storage, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability StorageUniformBufferBlock16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageUniformBufferBlock16})},
                {"OpCapability StorageUniform16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageUniform16})},
                {"OpCapability StoragePushConstant16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStoragePushConstant16})},
                {"OpCapability StorageInputOutput16\n",
                 MakeInstruction(SpvOpCapability,
                                 {SpvCapabilityStorageInputOutput16})},
            })), );

// SPV_KHR_device_group

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_device_group, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability DeviceGroup\n",
                 MakeInstruction(SpvOpCapability, {SpvCapabilityDeviceGroup})},
                {"OpDecorate %1 BuiltIn DeviceIndex\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInDeviceIndex})},
            })), );

// SPV_KHR_multiview

INSTANTIATE_TEST_CASE_P(
    SPV_KHR_multiview, ExtensionRoundTripTest,
    // We'll get coverage over operand tables by trying the universal
    // environments, and at least one specific environment.
    Combine(Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                   SPV_ENV_VULKAN_1_0),
            ValuesIn(std::vector<AssemblyCase>{
                {"OpCapability MultiView\n",
                 MakeInstruction(SpvOpCapability, {SpvCapabilityMultiView})},
                {"OpDecorate %1 BuiltIn ViewIndex\n",
                 MakeInstruction(SpvOpDecorate, {1, SpvDecorationBuiltIn,
                                                 SpvBuiltInViewIndex})},
            })), );

}  // anonymous namespace
