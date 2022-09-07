// Copyright (c) 2022 The Khronos Group Inc.
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

// Tests instructions from SPV_EXT_mesh_shader

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Values;

using ValidateMeshShading = spvtest::ValidateBase<bool>;

TEST_F(ValidateMeshShading, EmitMeshTasksEXTNotLastInstructionUniversal) {
  const std::string body = R"(
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main" %p
               OpExecutionModeId %main LocalSizeId %uint_1 %uint_1 %uint_1
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
  %arr_float = OpTypeArray %float %uint_1
    %Payload = OpTypeStruct %arr_float
%ptr_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload
          %p = OpVariable %ptr_Payload TaskPayloadWorkgroupEXT
       %main = OpFunction %void None %func
     %label1 = OpLabel
               OpEmitMeshTasksEXT %uint_1 %uint_1 %uint_1 %p
               OpBranch %label2
     %label2 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(body, SPV_ENV_UNIVERSAL_1_4);
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_4));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Branch must appear in a block"));
}

TEST_F(ValidateMeshShading, EmitMeshTasksEXTNotLastInstructionVulkan) {
  const std::string body = R"(
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main" %p
               OpExecutionModeId %main LocalSizeId %uint_1 %uint_1 %uint_1
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
  %arr_float = OpTypeArray %float %uint_1
    %Payload = OpTypeStruct %arr_float
%ptr_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload
          %p = OpVariable %ptr_Payload TaskPayloadWorkgroupEXT
       %main = OpFunction %void None %func
     %label1 = OpLabel
               OpEmitMeshTasksEXT %uint_1 %uint_1 %uint_1 %p
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(body, SPV_ENV_VULKAN_1_2);
  EXPECT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions(SPV_ENV_VULKAN_1_2));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Return must appear in a block"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
