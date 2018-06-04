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

// Common validation fixtures for unit tests

#include "gmock/gmock.h"
#include "source/val/decoration.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using libspirv::Decoration;
using std::string;
using std::vector;
using ::testing::Eq;
using ::testing::HasSubstr;

using ValidateDecorations = spvtest::ValidateBase<bool>;

TEST_F(ValidateDecorations, ValidateOpDecorateRegistration) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpDecorate %1 ArrayStride 4
    OpDecorate %1 Uniform
    %2 = OpTypeFloat 32
    %1 = OpTypeRuntimeArray %2
    ; Since %1 is used first in Decoration, it gets id 1.
)";
  const uint32_t id = 1;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  // Must have 2 decorations.
  EXPECT_THAT(vstate_->id_decorations(id),
              Eq(vector<Decoration>{Decoration(SpvDecorationArrayStride, {4}),
                                    Decoration(SpvDecorationUniform)}));
}

TEST_F(ValidateDecorations, ValidateOpMemberDecorateRegistration) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpDecorate %_arr_double_uint_6 ArrayStride 4
    OpMemberDecorate %_struct_115 2 NonReadable
    OpMemberDecorate %_struct_115 2 Offset 2
    OpDecorate %_struct_115 BufferBlock
    %float = OpTypeFloat 32
    %uint = OpTypeInt 32 0
    %uint_6 = OpConstant %uint 6
    %_arr_double_uint_6 = OpTypeArray %float %uint_6
    %_struct_115 = OpTypeStruct %float %float %_arr_double_uint_6
)";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());

  // The array must have 1 decoration.
  const uint32_t arr_id = 1;
  EXPECT_THAT(
      vstate_->id_decorations(arr_id),
      Eq(vector<Decoration>{Decoration(SpvDecorationArrayStride, {4})}));

  // The struct must have 3 decorations.
  const uint32_t struct_id = 2;
  EXPECT_THAT(vstate_->id_decorations(struct_id),
              Eq(vector<Decoration>{Decoration(SpvDecorationNonReadable, {}, 2),
                                    Decoration(SpvDecorationOffset, {2}, 2),
                                    Decoration(SpvDecorationBufferBlock)}));
}

TEST_F(ValidateDecorations, ValidateGroupDecorateRegistration) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %1 DescriptorSet 0
               OpDecorate %1 NonWritable
               OpDecorate %1 Restrict
          %1 = OpDecorationGroup
               OpGroupDecorate %1 %2 %3
               OpGroupDecorate %1 %4
  %float = OpTypeFloat 32
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_9 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_9 = OpTypePointer Uniform %_struct_9
         %2 = OpVariable %_ptr_Uniform__struct_9 Uniform
 %_struct_10 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_10 = OpTypePointer Uniform %_struct_10
         %3 = OpVariable %_ptr_Uniform__struct_10 Uniform
 %_struct_11 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_11 = OpTypePointer Uniform %_struct_11
         %4 = OpVariable %_ptr_Uniform__struct_11 Uniform
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());

  // Decoration group has 3 decorations.
  auto expected_decorations = vector<Decoration>{
      Decoration(SpvDecorationDescriptorSet, {0}),
      Decoration(SpvDecorationNonWritable), Decoration(SpvDecorationRestrict)};

  // Decoration group is applied to id 1, 2, 3, and 4. Note that id 1 (which is
  // the decoration group id) also has all the decorations.
  EXPECT_THAT(vstate_->id_decorations(1), Eq(expected_decorations));
  EXPECT_THAT(vstate_->id_decorations(2), Eq(expected_decorations));
  EXPECT_THAT(vstate_->id_decorations(3), Eq(expected_decorations));
  EXPECT_THAT(vstate_->id_decorations(4), Eq(expected_decorations));
}

TEST_F(ValidateDecorations, ValidateGroupMemberDecorateRegistration) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %1 Offset 3
          %1 = OpDecorationGroup
               OpGroupMemberDecorate %1 %_struct_1 3 %_struct_2 3 %_struct_3 3
      %float = OpTypeFloat 32
%_runtimearr = OpTypeRuntimeArray %float
  %_struct_1 = OpTypeStruct %float %float %float %_runtimearr
  %_struct_2 = OpTypeStruct %float %float %float %_runtimearr
  %_struct_3 = OpTypeStruct %float %float %float %_runtimearr
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  // Decoration group has 1 decoration.
  auto expected_decorations =
      vector<Decoration>{Decoration(SpvDecorationOffset, {3}, 3)};

  // Decoration group is applied to id 2, 3, and 4.
  EXPECT_THAT(vstate_->id_decorations(2), Eq(expected_decorations));
  EXPECT_THAT(vstate_->id_decorations(3), Eq(expected_decorations));
  EXPECT_THAT(vstate_->id_decorations(4), Eq(expected_decorations));
}

TEST_F(ValidateDecorations, LinkageImportUsedForInitializedVariableBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %target LinkageAttributes "link_ptr" Import
      %float = OpTypeFloat 32
 %_ptr_float = OpTypePointer Uniform %float
       %zero = OpConstantNull %float
     %target = OpVariable %_ptr_float Uniform %zero
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("A module-scope OpVariable with initialization value "
                        "cannot be marked with the Import Linkage Type."));
}
TEST_F(ValidateDecorations, LinkageExportUsedForInitializedVariableGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %target LinkageAttributes "link_ptr" Export
      %float = OpTypeFloat 32
 %_ptr_float = OpTypePointer Uniform %float
       %zero = OpConstantNull %float
     %target = OpVariable %_ptr_float Uniform %zero
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, StructAllMembersHaveBuiltInDecorationsGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpMemberDecorate %_struct_1 0 BuiltIn Position
               OpMemberDecorate %_struct_1 1 BuiltIn Position
               OpMemberDecorate %_struct_1 2 BuiltIn Position
               OpMemberDecorate %_struct_1 3 BuiltIn Position
      %float = OpTypeFloat 32
%_runtimearr = OpTypeRuntimeArray %float
  %_struct_1 = OpTypeStruct %float %float %float %_runtimearr
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, MixedBuiltInDecorationsBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpMemberDecorate %_struct_1 0 BuiltIn Position
               OpMemberDecorate %_struct_1 1 BuiltIn Position
      %float = OpTypeFloat 32
%_runtimearr = OpTypeRuntimeArray %float
  %_struct_1 = OpTypeStruct %float %float %float %_runtimearr
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("When BuiltIn decoration is applied to a structure-type "
                "member, all members of that structure type must also be "
                "decorated with BuiltIn (No allowed mixing of built-in "
                "variables and non-built-in variables within a single "
                "structure). Structure id 1 does not meet this requirement."));
}

TEST_F(ValidateDecorations, StructContainsBuiltInStructBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpMemberDecorate %_struct_1 0 BuiltIn Position
               OpMemberDecorate %_struct_1 1 BuiltIn Position
               OpMemberDecorate %_struct_1 2 BuiltIn Position
               OpMemberDecorate %_struct_1 3 BuiltIn Position
      %float = OpTypeFloat 32
%_runtimearr = OpTypeRuntimeArray %float
  %_struct_1 = OpTypeStruct %float %float %float %_runtimearr
  %_struct_2 = OpTypeStruct %_struct_1
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Structure <id> 1 contains members with BuiltIn "
                        "decoration. Therefore this structure may not be "
                        "contained as a member of another structure type. "
                        "Structure <id> 4 contains structure <id> 1."));
}

TEST_F(ValidateDecorations, StructContainsNonBuiltInStructGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
      %float = OpTypeFloat 32
  %_struct_1 = OpTypeStruct %float
  %_struct_2 = OpTypeStruct %_struct_1
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, MultipleBuiltInObjectsConsumedByOpEntryPointBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %in_1 %in_2
               OpMemberDecorate %struct_1 0 BuiltIn InvocationId
               OpMemberDecorate %struct_2 0 BuiltIn Position
      %int = OpTypeInt 32 1
     %void = OpTypeVoid
     %func = OpTypeFunction %void
    %float = OpTypeFloat 32
 %struct_1 = OpTypeStruct %int
 %struct_2 = OpTypeStruct %float
%ptr_builtin_1 = OpTypePointer Input %struct_1
%ptr_builtin_2 = OpTypePointer Input %struct_2
%in_1 = OpVariable %ptr_builtin_1 Input
%in_2 = OpVariable %ptr_builtin_2 Input
       %main = OpFunction %void None %func
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("There must be at most one object per Storage Class "
                        "that can contain a structure type containing members "
                        "decorated with BuiltIn, consumed per entry-point."));
}

TEST_F(ValidateDecorations,
       OneBuiltInObjectPerStorageClassConsumedByOpEntryPointGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %in_1 %out_1
               OpMemberDecorate %struct_1 0 BuiltIn InvocationId
               OpMemberDecorate %struct_2 0 BuiltIn Position
      %int = OpTypeInt 32 1
     %void = OpTypeVoid
     %func = OpTypeFunction %void
    %float = OpTypeFloat 32
 %struct_1 = OpTypeStruct %int
 %struct_2 = OpTypeStruct %float
%ptr_builtin_1 = OpTypePointer Input %struct_1
%ptr_builtin_2 = OpTypePointer Output %struct_2
%in_1 = OpVariable %ptr_builtin_1 Input
%out_1 = OpVariable %ptr_builtin_2 Output
       %main = OpFunction %void None %func
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, NoBuiltInObjectsConsumedByOpEntryPointGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %in_1 %out_1
      %int = OpTypeInt 32 1
     %void = OpTypeVoid
     %func = OpTypeFunction %void
    %float = OpTypeFloat 32
 %struct_1 = OpTypeStruct %int
 %struct_2 = OpTypeStruct %float
%ptr_builtin_1 = OpTypePointer Input %struct_1
%ptr_builtin_2 = OpTypePointer Output %struct_2
%in_1 = OpVariable %ptr_builtin_1 Input
%out_1 = OpVariable %ptr_builtin_2 Output
       %main = OpFunction %void None %func
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, EntryPointFunctionHasLinkageAttributeBad) {
  string spirv = R"(
      OpCapability Shader
      OpCapability Linkage
      OpMemoryModel Logical GLSL450
      OpEntryPoint GLCompute %main "main"
      OpDecorate %main LinkageAttributes "import_main" Import
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%main = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CompileSuccessfully(spirv.c_str());
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("The LinkageAttributes Decoration (Linkage name: import_main) "
                "cannot be applied to function id 1 because it is targeted by "
                "an OpEntryPoint instruction."));
}

TEST_F(ValidateDecorations, FunctionDeclarationWithoutImportLinkageBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
     %void = OpTypeVoid
     %func = OpTypeFunction %void
       %main = OpFunction %void None %func
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateAndRetrieveValidationState());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Function declaration (id 3) must have a LinkageAttributes "
                "decoration with the Import Linkage type."));
}

TEST_F(ValidateDecorations, FunctionDeclarationWithImportLinkageGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %main LinkageAttributes "link_fn" Import
     %void = OpTypeVoid
     %func = OpTypeFunction %void
       %main = OpFunction %void None %func
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, FunctionDeclarationWithExportLinkageBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %main LinkageAttributes "link_fn" Export
     %void = OpTypeVoid
     %func = OpTypeFunction %void
       %main = OpFunction %void None %func
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateAndRetrieveValidationState());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Function declaration (id 1) must have a LinkageAttributes "
                "decoration with the Import Linkage type."));
}

TEST_F(ValidateDecorations, FunctionDefinitionWithImportLinkageBad) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %main LinkageAttributes "link_fn" Import
     %void = OpTypeVoid
     %func = OpTypeFunction %void
       %main = OpFunction %void None %func
      %label = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateAndRetrieveValidationState());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Function definition (id 1) may not be decorated with "
                        "Import Linkage type."));
}

TEST_F(ValidateDecorations, FunctionDefinitionWithoutImportLinkageGood) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
     %void = OpTypeVoid
     %func = OpTypeFunction %void
       %main = OpFunction %void None %func
      %label = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

TEST_F(ValidateDecorations, BuiltinVariablesGoodVulkan) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragCoord %_entryPointOutput
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 500
OpDecorate %gl_FragCoord BuiltIn FragCoord
OpDecorate %_entryPointOutput Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %3
%5 = OpLabel
OpStore %_entryPointOutput %14
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, env);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState(env));
}

TEST_F(ValidateDecorations, BuiltinVariablesWithLocationDecorationVulkan) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragCoord %_entryPointOutput
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 500
OpDecorate %gl_FragCoord BuiltIn FragCoord
OpDecorate %gl_FragCoord Location 0
OpDecorate %_entryPointOutput Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %3
%5 = OpLabel
OpStore %_entryPointOutput %14
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, env);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("A BuiltIn variable (id 2) cannot have any Location or "
                        "Component decorations"));
}
TEST_F(ValidateDecorations, BuiltinVariablesWithComponentDecorationVulkan) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragCoord %_entryPointOutput
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 500
OpDecorate %gl_FragCoord BuiltIn FragCoord
OpDecorate %gl_FragCoord Component 0
OpDecorate %_entryPointOutput Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%float_0 = OpConstant %float 0
%14 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %3
%5 = OpLabel
OpStore %_entryPointOutput %14
OpReturn
OpFunctionEnd
)";

  CompileSuccessfully(spirv, env);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("A BuiltIn variable (id 2) cannot have any Location or "
                        "Component decorations"));
}

// #version 440
// #extension GL_EXT_nonuniform_qualifier : enable
// layout(binding = 1) uniform sampler2D s2d[];
// layout(location = 0) in nonuniformEXT int i;
// void main()
// {
//     vec4 v = texture(s2d[i], vec2(0.3));
// }
TEST_F(ValidateDecorations, RuntimeArrayOfDescriptorSetsIsAllowed) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
               OpCapability Shader
               OpCapability ShaderNonUniformEXT
               OpCapability RuntimeDescriptorArrayEXT
               OpCapability SampledImageArrayNonUniformIndexingEXT
               OpExtension "SPV_EXT_descriptor_indexing"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %i
               OpSource GLSL 440
               OpSourceExtension "GL_EXT_nonuniform_qualifier"
               OpName %main "main"
               OpName %v "v"
               OpName %s2d "s2d"
               OpName %i "i"
               OpDecorate %s2d DescriptorSet 0
               OpDecorate %s2d Binding 1
               OpDecorate %i Location 0
               OpDecorate %i NonUniformEXT
               OpDecorate %18 NonUniformEXT
               OpDecorate %21 NonUniformEXT
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_runtimearr_11 = OpTypeRuntimeArray %11
%_ptr_UniformConstant__runtimearr_11 = OpTypePointer UniformConstant %_runtimearr_11
        %s2d = OpVariable %_ptr_UniformConstant__runtimearr_11 UniformConstant
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %i = OpVariable %_ptr_Input_int Input
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
    %v2float = OpTypeVector %float 2
%float_0_300000012 = OpConstant %float 0.300000012
         %24 = OpConstantComposite %v2float %float_0_300000012 %float_0_300000012
    %float_0 = OpConstant %float 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4float Function
         %18 = OpLoad %int %i
         %20 = OpAccessChain %_ptr_UniformConstant_11 %s2d %18
         %21 = OpLoad %11 %20
         %26 = OpImageSampleExplicitLod %v4float %21 %24 Lod %float_0
               OpStore %v %26
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(spirv, env);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
}

// #version 440
// #extension GL_EXT_nonuniform_qualifier : enable
// layout(binding = 1) uniform sampler2D s2d[][2];
// layout(location = 0) in nonuniformEXT int i;
// void main()
// {
//     vec4 v = texture(s2d[i][i], vec2(0.3));
// }
TEST_F(ValidateDecorations, RuntimeArrayOfArraysOfDescriptorSetsIsDisallowed) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
               OpCapability Shader
               OpCapability ShaderNonUniformEXT
               OpCapability RuntimeDescriptorArrayEXT
               OpCapability SampledImageArrayNonUniformIndexingEXT
               OpExtension "SPV_EXT_descriptor_indexing"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %i
               OpSource GLSL 440
               OpSourceExtension "GL_EXT_nonuniform_qualifier"
               OpName %main "main"
               OpName %v "v"
               OpName %s2d "s2d"
               OpName %i "i"
               OpDecorate %s2d DescriptorSet 0
               OpDecorate %s2d Binding 1
               OpDecorate %i Location 0
               OpDecorate %i NonUniformEXT
               OpDecorate %21 NonUniformEXT
               OpDecorate %22 NonUniformEXT
               OpDecorate %25 NonUniformEXT
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_arr_11_uint_2 = OpTypeArray %11 %uint_2
%_runtimearr__arr_11_uint_2 = OpTypeRuntimeArray %_arr_11_uint_2
%_ptr_UniformConstant__runtimearr__arr_11_uint_2 = OpTypePointer UniformConstant %_runtimearr__arr_11_uint_2
        %s2d = OpVariable %_ptr_UniformConstant__runtimearr__arr_11_uint_2 UniformConstant
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %i = OpVariable %_ptr_Input_int Input
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
    %v2float = OpTypeVector %float 2
%float_0_300000012 = OpConstant %float 0.300000012
         %28 = OpConstantComposite %v2float %float_0_300000012 %float_0_300000012
    %float_0 = OpConstant %float 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4float Function
         %21 = OpLoad %int %i
         %22 = OpLoad %int %i
         %24 = OpAccessChain %_ptr_UniformConstant_11 %s2d %21 %22
         %25 = OpLoad %11 %24
         %30 = OpImageSampleExplicitLod %v4float %25 %28 Lod %float_0
               OpStore %v %30
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(spirv, env);

  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Array of arrays is not allowed"));
}

// #version 440
// layout (set=1, binding=1) uniform sampler2D variableName[2][2];
// void main() {
// }
TEST_F(ValidateDecorations, ArrayOfArraysOfDescriptorSetsIsDisallowed) {
  const spv_target_env env = SPV_ENV_VULKAN_1_0;
  std::string spirv = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpSource GLSL 440
               OpName %main "main"
               OpName %variableName "variableName"
               OpDecorate %variableName DescriptorSet 1
               OpDecorate %variableName Binding 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %7 = OpTypeImage %float 2D 0 0 0 1 Unknown
          %8 = OpTypeSampledImage %7
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_arr_8_uint_2 = OpTypeArray %8 %uint_2
%_arr__arr_8_uint_2_uint_2 = OpTypeArray %_arr_8_uint_2 %uint_2
%_ptr_UniformConstant__arr__arr_8_uint_2_uint_2 = OpTypePointer UniformConstant %_arr__arr_8_uint_2_uint_2
%variableName = OpVariable %_ptr_UniformConstant__arr__arr_8_uint_2_uint_2 UniformConstant
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";
  CompileSuccessfully(spirv, env);

  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateAndRetrieveValidationState(env));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Array of arrays is not allowed"));
}

}  // anonymous namespace
