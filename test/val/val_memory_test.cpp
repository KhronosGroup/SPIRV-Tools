// Copyright (c) 2018 Google Inc.
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

// Validation tests for memory/storage

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

using ValidateMemory = spvtest::ValidateBase<bool>;

TEST_F(ValidateMemory, VulkanUniformConstantOnNonOpaqueResourceBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%float_ptr = OpTypePointer UniformConstant %float
%2 = OpVariable %float_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the UniformConstant storage class "
                "are used only as handles to refer to opaque resources. Such "
                "variables must be typed as OpTypeImage, OpTypeSampler, "
                "OpTypeSampledImage, OpTypeAccelerationStructureNV, or an "
                "array of one of these types."));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler = OpTypeSampler
%sampler_ptr = OpTypePointer UniformConstant %sampler
%2 = OpVariable %sampler_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnNonOpaqueResourceArrayBad) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%array_size = OpConstant %uint 5
%array = OpTypeArray %float %array_size
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_VULKAN_1_1));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("From Vulkan spec, section 14.5.2:\n"
                "Variables identified with the UniformConstant storage class "
                "are used only as handles to refer to opaque resources. Such "
                "variables must be typed as OpTypeImage, OpTypeSampler, "
                "OpTypeSampledImage, OpTypeAccelerationStructureNV, or an "
                "array of one of these types."));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceArrayGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler = OpTypeSampler
%uint = OpTypeInt 32 0
%array_size = OpConstant %uint 5
%array = OpTypeArray %sampler %array_size
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

TEST_F(ValidateMemory, VulkanUniformConstantOnOpaqueResourceRuntimeArrayGood) {
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%sampler = OpTypeSampler
%uint = OpTypeInt 32 0
%array = OpTypeRuntimeArray %sampler
%array_ptr = OpTypePointer UniformConstant %array
%2 = OpVariable %array_ptr UniformConstant
%void = OpTypeVoid
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str(), SPV_ENV_VULKAN_1_1);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_VULKAN_1_1));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
