// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

using FixUniformStructOpaqueTest = PassTest<::testing::Test>;

TEST_F(FixUniformStructOpaqueTest, ExtractSampler) {
    const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %color %uv
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %color "color"
               OpName %UniformBlock "UniformBlock"
               OpMemberName %UniformBlock 0 "srctex"
               OpMemberName %UniformBlock 1 "salt"
               OpName %gl_DefaultUniformBlock "gl_DefaultUniformBlock"
               OpMemberName %gl_DefaultUniformBlock 0 "data"
               OpName %_ ""
               OpName %uv "uv"
               OpDecorate %color Location 0
               OpMemberDecorate %UniformBlock 0 Offset 0
               OpMemberDecorate %UniformBlock 1 Offset 16
               OpMemberDecorate %gl_DefaultUniformBlock 0 Offset 0
               OpDecorate %gl_DefaultUniformBlock Block
               OpDecorate %_ DescriptorSet 0
               OpDecorate %_ Binding 0
               OpDecorate %uv Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%UniformBlock = OpTypeStruct %11 %v4float
%gl_DefaultUniformBlock = OpTypeStruct %UniformBlock
%_ptr_UniformConstant_gl_DefaultUniformBlock = OpTypePointer UniformConstant %gl_DefaultUniformBlock
          %_ = OpVariable %_ptr_UniformConstant_gl_DefaultUniformBlock UniformConstant
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
         %uv = OpVariable %_ptr_Input_v2float Input
      %int_1 = OpConstant %int 1
%_ptr_UniformConstant_v4float = OpTypePointer UniformConstant %v4float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %19 = OpAccessChain %_ptr_UniformConstant_11 %_ %int_0 %int_0
         %20 = OpLoad %11 %19
         %24 = OpLoad %v2float %uv
         %25 = OpImageSampleImplicitLod %v4float %20 %24
         %28 = OpAccessChain %_ptr_UniformConstant_v4float %_ %int_0 %int_1
         %29 = OpLoad %v4float %28
         %30 = OpFAdd %v4float %25 %29
               OpStore %color %30
               OpReturn
               OpFunctionEnd
)";

    const std::string after =
        R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %color %uv
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %color "color"
OpName %UniformBlock "UniformBlock"
OpMemberName %UniformBlock 1 "salt"
OpName %gl_DefaultUniformBlock "gl_DefaultUniformBlock"
OpMemberName %gl_DefaultUniformBlock 0 "data"
OpName %_ ""
OpName %uv "uv"
OpName %data_srctex "data.srctex"
OpDecorate %color Location 0
OpMemberDecorate %UniformBlock 0 Offset 0
OpMemberDecorate %UniformBlock 1 Offset 16
OpMemberDecorate %gl_DefaultUniformBlock 0 Offset 0
OpDecorate %gl_DefaultUniformBlock Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %uv Location 0
OpDecorate %data_srctex DescriptorSet 0
OpDecorate %data_srctex Binding 1
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%color = OpVariable %_ptr_Output_v4float Output
%10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%11 = OpTypeSampledImage %10
%uint = OpTypeInt 32 0
%UniformBlock = OpTypeStruct %uint %v4float
%gl_DefaultUniformBlock = OpTypeStruct %UniformBlock
%_ptr_Uniform_gl_DefaultUniformBlock = OpTypePointer Uniform %gl_DefaultUniformBlock
%_ = OpVariable %_ptr_Uniform_gl_DefaultUniformBlock Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%uv = OpVariable %_ptr_Input_v2float Input
%int_1 = OpConstant %int 1
%_ptr_Uniform_v4float = OpTypePointer Uniform %v4float
%data_srctex = OpVariable %_ptr_UniformConstant_11 UniformConstant
%main = OpFunction %void None %3
%5 = OpLabel
%20 = OpLoad %11 %data_srctex
%24 = OpLoad %v2float %uv
%25 = OpImageSampleImplicitLod %v4float %20 %24
%28 = OpAccessChain %_ptr_Uniform_v4float %_ %int_0 %int_1
%29 = OpLoad %v4float %28
%30 = OpFAdd %v4float %25 %29
OpStore %color %30
OpReturn
OpFunctionEnd
)";

    SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    SinglePassRunAndCheck<FixUniformStructOpaquePass>(before, after, true, true);
}

} // namespace
} // namespace opt
} // namespace spvtools
