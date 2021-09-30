// Copyright (c) 2021 Google LLC
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

#include <string>

#include "gmock/gmock.h"
#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using ReplaceDescriptorVariableIndexAccessTest = PassTest<::testing::Test>;

TEST_F(ReplaceDescriptorVariableIndexAccessTest,
       ReplaceAccessChainToTextureArray) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %psmain "psmain" %gl_FragCoord %in_var_INSTANCEID %out_var_SV_TARGET
               OpExecutionMode %psmain OriginUpperLeft
               OpSource HLSL 600
               OpName %type_sampler "type.sampler"
               OpName %Sampler0 "Sampler0"
               OpName %type_2d_image "type.2d.image"
               OpName %Tex0 "Tex0"
               OpName %in_var_INSTANCEID "in.var.INSTANCEID"
               OpName %out_var_SV_TARGET "out.var.SV_TARGET"
               OpName %psmain "psmain"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %in_var_INSTANCEID Flat
               OpDecorate %in_var_INSTANCEID Location 0
               OpDecorate %out_var_SV_TARGET Location 0
               OpDecorate %Sampler0 DescriptorSet 0
               OpDecorate %Sampler0 Binding 1
               OpDecorate %Tex0 DescriptorSet 0
               OpDecorate %Tex0 Binding 2
       %bool = OpTypeBool
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
      %float = OpTypeFloat 32
%type_2d_image = OpTypeImage %float 2D 2 0 0 0 Unknown
%_arr_type_2d_image_uint_3 = OpTypeArray %type_2d_image %uint_3
%_ptr_UniformConstant__arr_type_2d_image_uint_3 = OpTypePointer UniformConstant %_arr_type_2d_image_uint_3
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Input_uint = OpTypePointer Input %uint
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
    %v2float = OpTypeVector %float 2
     %v2uint = OpTypeVector %uint 2
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
         %27 = OpConstantComposite %v2uint %uint_0 %uint_1
%type_sampled_image = OpTypeSampledImage %type_2d_image
   %Sampler0 = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
       %Tex0 = OpVariable %_ptr_UniformConstant__arr_type_2d_image_uint_3 UniformConstant
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%in_var_INSTANCEID = OpVariable %_ptr_Input_uint Input
%out_var_SV_TARGET = OpVariable %_ptr_Output_v4float Output
     %uint_2 = OpConstant %uint 2
         %66 = OpConstantNull %v4float

; CHECK: [[null_value:%\w+]] = OpConstantNull %v4float

     %psmain = OpFunction %void None %21
         %39 = OpLabel
         %29 = OpLoad %v4float %gl_FragCoord
         %30 = OpLoad %uint %in_var_INSTANCEID
         %37 = OpIEqual %bool %30 %uint_2
               OpSelectionMerge %38 None
               OpBranchConditional %37 %28 %40

; CHECK: [[var_index:%\w+]] = OpLoad %uint %in_var_INSTANCEID
; CHECK: OpSelectionMerge [[cond_branch_merge:%\w+]] None
; CHECK: OpBranchConditional {{%\w+}} {{%\w+}} [[bb_cond_br:%\w+]]

         %28 = OpLabel
         %31 = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %30
         %32 = OpLoad %type_2d_image %31
               OpImageWrite %32 %27 %29

; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default:%\w+]] 0 [[case0:%\w+]] 1 [[case1:%\w+]] 2 [[case2:%\w+]]
; CHECK: [[case0]] = OpLabel
; CHECK: OpAccessChain
; CHECK: OpLoad
; CHECK: OpImageWrite
; CHECK: OpBranch [[merge]]
; CHECK: [[case1]] = OpLabel
; CHECK: OpAccessChain
; CHECK: OpLoad
; CHECK: OpImageWrite
; CHECK: OpBranch [[merge]]
; CHECK: [[case2]] = OpLabel
; CHECK: OpAccessChain
; CHECK: OpLoad
; CHECK: OpImageWrite
; CHECK: OpBranch [[merge]]
; CHECK: [[default]] = OpLabel
; CHECK: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel

         %33 = OpLoad %type_sampler %Sampler0
         %34 = OpVectorShuffle %v2float %29 %29 0 1
         %35 = OpSampledImage %type_sampled_image %32 %33
         %36 = OpImageSampleImplicitLod %v4float %35 %34 None

; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default:%\w+]] 0 [[case0:%\w+]] 1 [[case1:%\w+]] 2 [[case2:%\w+]]
; CHECK: [[case0]] = OpLabel
; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_0
; CHECK: [[sam:%\w+]] = OpLoad %type_sampler %Sampler0
; CHECK: [[img:%\w+]] = OpLoad %type_2d_image [[ac]]
; CHECK: [[sampledImg:%\w+]] = OpSampledImage %type_sampled_image [[img]] [[sam]]
; CHECK: [[value0:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg]]
; CHECK: OpBranch [[merge]]
; CHECK: [[case1]] = OpLabel
; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_1
; CHECK: [[sam:%\w+]] = OpLoad %type_sampler %Sampler0
; CHECK: [[img:%\w+]] = OpLoad %type_2d_image [[ac]]
; CHECK: [[sampledImg:%\w+]] = OpSampledImage %type_sampled_image [[img]] [[sam]]
; CHECK: [[value1:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg]]
; CHECK: OpBranch [[merge]]
; CHECK: [[case2]] = OpLabel
; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_2
; CHECK: [[sam:%\w+]] = OpLoad %type_sampler %Sampler0
; CHECK: [[img:%\w+]] = OpLoad %type_2d_image [[ac]]
; CHECK: [[sampledImg:%\w+]] = OpSampledImage %type_sampled_image [[img]] [[sam]]
; CHECK: [[value2:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg]]
; CHECK: OpBranch [[merge]]
; CHECK: [[default]] = OpLabel
; CHECK: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK: [[phi0:%\w+]] = OpPhi %v4float [[value0]] [[case0]] [[value1]] [[case1]] [[value2]] [[case2]] [[null_value]] [[default]]

               OpBranch %38
         %40 = OpLabel
               OpBranch %38
         %38 = OpLabel
         %41 = OpPhi %v4float %36 %28 %29 %40

; CHECK: OpBranch [[cond_branch_merge]]
; CHECK: [[bb_cond_br]] = OpLabel
; CHECK: OpBranch [[cond_branch_merge]]
; CHECK: [[cond_branch_merge]] = OpLabel
; CHECK: [[phi1:%\w+]] = OpPhi %v4float [[phi0]] [[merge]] {{%\w+}} [[bb_cond_br]]
; CHECK: OpStore {{%\w+}} [[phi1]]

               OpStore %out_var_SV_TARGET %41
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<ReplaceDescriptorVariableIndexAccess>(text, true);
}

TEST_F(ReplaceDescriptorVariableIndexAccessTest,
       ReplaceAccessChainToTextureArrayAndSamplerArray) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %psmain "psmain" %gl_FragCoord %in_var_INSTANCEID %out_var_SV_TARGET
               OpExecutionMode %psmain OriginUpperLeft
               OpSource HLSL 600
               OpName %type_sampler "type.sampler"
               OpName %Sampler0 "Sampler0"
               OpName %type_2d_image "type.2d.image"
               OpName %Tex0 "Tex0"
               OpName %in_var_INSTANCEID "in.var.INSTANCEID"
               OpName %out_var_SV_TARGET "out.var.SV_TARGET"
               OpName %psmain "psmain"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %in_var_INSTANCEID Flat
               OpDecorate %in_var_INSTANCEID Location 0
               OpDecorate %out_var_SV_TARGET Location 0
               OpDecorate %Sampler0 DescriptorSet 0
               OpDecorate %Sampler0 Binding 1
               OpDecorate %Tex0 DescriptorSet 0
               OpDecorate %Tex0 Binding 2
%type_sampler = OpTypeSampler
       %uint = OpTypeInt 32 0
     %uint_2 = OpConstant %uint 2
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
%_arr_type_sampler_uint_2 = OpTypeArray %type_sampler %uint_2
%_ptr_UniformConstant__arr_type_sampler_uint_2 = OpTypePointer UniformConstant %_arr_type_sampler_uint_2
      %float = OpTypeFloat 32
%type_2d_image = OpTypeImage %float 2D 2 0 0 0 Unknown
%_arr_type_2d_image_uint_2 = OpTypeArray %type_2d_image %uint_2
%_ptr_UniformConstant__arr_type_2d_image_uint_2 = OpTypePointer UniformConstant %_arr_type_2d_image_uint_2
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Input_uint = OpTypePointer Input %uint
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
    %v2float = OpTypeVector %float 2
     %v2uint = OpTypeVector %uint 2
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
         %27 = OpConstantComposite %v2uint %uint_0 %uint_1
%type_sampled_image = OpTypeSampledImage %type_2d_image
   %Sampler0 = OpVariable %_ptr_UniformConstant__arr_type_sampler_uint_2 UniformConstant
       %Tex0 = OpVariable %_ptr_UniformConstant__arr_type_2d_image_uint_2 UniformConstant
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%in_var_INSTANCEID = OpVariable %_ptr_Input_uint Input
%out_var_SV_TARGET = OpVariable %_ptr_Output_v4float Output
         %66 = OpConstantNull %v4float
     %psmain = OpFunction %void None %21
         %28 = OpLabel
         %29 = OpLoad %v4float %gl_FragCoord
         %30 = OpLoad %uint %in_var_INSTANCEID
         %31 = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %30
         %32 = OpLoad %type_2d_image %31
               OpImageWrite %32 %27 %29

; CHECK: [[null_value:%\w+]] = OpConstantNull %v4float

; CHECK: [[var_index:%\w+]] = OpLoad %uint %in_var_INSTANCEID
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default:%\w+]] 0 [[case0:%\w+]] 1 [[case1:%\w+]]
; CHECK: [[case0]] = OpLabel
; CHECK: OpAccessChain
; CHECK: OpLoad
; CHECK: OpImageWrite
; CHECK: OpBranch [[merge]]
; CHECK: [[case1]] = OpLabel
; CHECK: OpAccessChain
; CHECK: OpLoad
; CHECK: OpImageWrite
; CHECK: OpBranch [[merge]]
; CHECK: [[default]] = OpLabel
; CHECK: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel

         %33 = OpAccessChain %_ptr_UniformConstant_type_sampler %Sampler0 %30
         %37 = OpLoad %type_sampler %33
         %34 = OpVectorShuffle %v2float %29 %29 0 1
         %35 = OpSampledImage %type_sampled_image %32 %37
         %36 = OpImageSampleImplicitLod %v4float %35 %34 None

; SPIR-V instructions to be replaced (will be killed by ADCE)
; CHECK: OpSelectionMerge
; CHECK: OpSwitch

; CHECK: OpSelectionMerge [[merge_sampler:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default_sampler:%\w+]] 0 [[case_sampler0:%\w+]] 1 [[case_sampler1:%\w+]]

; CHECK: [[case_sampler0]] = OpLabel
; CHECK: OpSelectionMerge [[merge_texture0:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default_texture:%\w+]] 0 [[case_texture0:%\w+]] 1 [[case_texture1:%\w+]]
; CHECK: [[case_texture0]] = OpLabel
; CHECK: [[pt0:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_0
; CHECK: [[ps0:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_sampler %Sampler0 %uint_0
; CHECK: [[s0:%\w+]] = OpLoad %type_sampler [[ps0]]
; CHECK: [[t0:%\w+]] = OpLoad %type_2d_image [[pt0]]
; CHECK: [[sampledImg0:%\w+]] = OpSampledImage %type_sampled_image [[t0]] [[s0]]
; CHECK: [[value0:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg0]]
; CHECK: OpBranch [[merge_texture0]]
; CHECK: [[case_texture1]] = OpLabel
; CHECK: [[pt1:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_1
; CHECK: [[ps0:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_sampler %Sampler0 %uint_0
; CHECK: [[s0:%\w+]] = OpLoad %type_sampler [[ps0]]
; CHECK: [[t1:%\w+]] = OpLoad %type_2d_image [[pt1]]
; CHECK: [[sampledImg1:%\w+]] = OpSampledImage %type_sampled_image [[t1]] [[s0]]
; CHECK: [[value1:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg1]]
; CHECK: OpBranch [[merge_texture0]]
; CHECK: [[default_texture]] = OpLabel
; CHECK: OpBranch [[merge_texture0]]
; CHECK: [[merge_texture0]] = OpLabel
; CHECK: [[phi0:%\w+]] = OpPhi %v4float [[value0]] [[case_texture0]] [[value1]] [[case_texture1]] [[null_value]] [[default_texture]]
; CHECK: OpBranch [[merge_sampler]]

; CHECK: [[case_sampler1]] = OpLabel
; CHECK: OpSelectionMerge [[merge_texture1:%\w+]] None
; CHECK: OpSwitch [[var_index]] [[default_texture:%\w+]] 0 [[case_texture0:%\w+]] 1 [[case_texture1:%\w+]]
; CHECK: [[case_texture0]] = OpLabel
; CHECK: [[pt0:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_0
; CHECK: [[ps1:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_sampler %Sampler0 %uint_1
; CHECK: [[s1:%\w+]] = OpLoad %type_sampler [[ps1]]
; CHECK: [[t0:%\w+]] = OpLoad %type_2d_image [[pt0]]
; CHECK: [[sampledImg0:%\w+]] = OpSampledImage %type_sampled_image [[t0]] [[s1]]
; CHECK: [[value0:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg0]]
; CHECK: OpBranch [[merge_texture1]]
; CHECK: [[case_texture1]] = OpLabel
; CHECK: [[pt1:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_1
; CHECK: [[ps1:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_sampler %Sampler0 %uint_1
; CHECK: [[s1:%\w+]] = OpLoad %type_sampler [[ps1]]
; CHECK: [[t1:%\w+]] = OpLoad %type_2d_image [[pt1]]
; CHECK: [[sampledImg1:%\w+]] = OpSampledImage %type_sampled_image [[t1]] [[s1]]
; CHECK: [[value1:%\w+]] = OpImageSampleImplicitLod %v4float [[sampledImg1]]
; CHECK: OpBranch [[merge_texture1]]
; CHECK: [[default_texture]] = OpLabel
; CHECK: OpBranch [[merge_texture1]]
; CHECK: [[merge_texture1]] = OpLabel
; CHECK: [[phi1:%\w+]] = OpPhi %v4float [[value0]] [[case_texture0]] [[value1]] [[case_texture1]] [[null_value]] [[default_texture]]

; CHECK: [[default_sampler]] = OpLabel
; CHECK: OpBranch [[merge_sampler]]
; CHECK: [[merge_sampler]] = OpLabel
; CHECK: OpPhi %v4float [[phi0]] [[merge_texture0]] [[phi1]] [[merge_texture1]] [[null_value]] [[default_sampler]]
; CHECK: OpStore

               OpStore %out_var_SV_TARGET %36
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<ReplaceDescriptorVariableIndexAccess>(text, true);
}

TEST_F(ReplaceDescriptorVariableIndexAccessTest,
       ReplaceAccessChainToTextureArrayWithSingleElement) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %psmain "psmain" %gl_FragCoord %in_var_INSTANCEID %out_var_SV_TARGET
               OpExecutionMode %psmain OriginUpperLeft
               OpSource HLSL 600
               OpName %type_sampler "type.sampler"
               OpName %Sampler0 "Sampler0"
               OpName %type_2d_image "type.2d.image"
               OpName %Tex0 "Tex0"
               OpName %in_var_INSTANCEID "in.var.INSTANCEID"
               OpName %out_var_SV_TARGET "out.var.SV_TARGET"
               OpName %psmain "psmain"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %in_var_INSTANCEID Flat
               OpDecorate %in_var_INSTANCEID Location 0
               OpDecorate %out_var_SV_TARGET Location 0
               OpDecorate %Sampler0 DescriptorSet 0
               OpDecorate %Sampler0 Binding 1
               OpDecorate %Tex0 DescriptorSet 0
               OpDecorate %Tex0 Binding 2
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
%type_2d_image = OpTypeImage %float 2D 2 0 0 0 Unknown
%_arr_type_2d_image_uint_1 = OpTypeArray %type_2d_image %uint_1
%_ptr_UniformConstant__arr_type_2d_image_uint_1 = OpTypePointer UniformConstant %_arr_type_2d_image_uint_1
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Input_uint = OpTypePointer Input %uint
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %void = OpTypeVoid
         %21 = OpTypeFunction %void
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
    %v2float = OpTypeVector %float 2
     %v2uint = OpTypeVector %uint 2
     %uint_0 = OpConstant %uint 0
         %27 = OpConstantComposite %v2uint %uint_0 %uint_1
%type_sampled_image = OpTypeSampledImage %type_2d_image
   %Sampler0 = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
       %Tex0 = OpVariable %_ptr_UniformConstant__arr_type_2d_image_uint_1 UniformConstant
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%in_var_INSTANCEID = OpVariable %_ptr_Input_uint Input
%out_var_SV_TARGET = OpVariable %_ptr_Output_v4float Output
     %uint_2 = OpConstant %uint 2
         %66 = OpConstantNull %v4float
     %psmain = OpFunction %void None %21
         %28 = OpLabel
         %29 = OpLoad %v4float %gl_FragCoord
         %30 = OpLoad %uint %in_var_INSTANCEID
         %31 = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %30
         %32 = OpLoad %type_2d_image %31
               OpImageWrite %32 %27 %29

; CHECK: [[ac:%\w+]] = OpAccessChain %_ptr_UniformConstant_type_2d_image %Tex0 %uint_0
; CHECK-NOT: OpAccessChain
; CHECK-NOT: OpSwitch
; CHECK-NOT: OpPhi

         %33 = OpLoad %type_sampler %Sampler0
         %34 = OpVectorShuffle %v2float %29 %29 0 1
         %35 = OpSampledImage %type_sampled_image %32 %33
         %36 = OpImageSampleImplicitLod %v4float %35 %34 None

               OpStore %out_var_SV_TARGET %36
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<ReplaceDescriptorVariableIndexAccess>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
