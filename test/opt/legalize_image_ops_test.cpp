// Copyright (c) 2021 Tencent Inc.
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

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using LegalizeImageOpsTest = PassTest<::testing::Test>;

// %OpTypeImage is using half types, convert to float when necessary
// and mark it's percision as relaxed.
TEST_F(LegalizeImageOpsTest, ConvertImageOpPercision) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Float16
OpCapability StorageInputOutput16
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_user_type"
OpExtension "SPV_KHR_16bit_storage"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPS "MainPS" %UV %OutColor
OpExecutionMode %MainPS OriginUpperLeft
OpSource HLSL 500
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %MainPS "MainPS"
OpName %InputTexture "InputTexture"
OpName %InputSampler "InputSampler"
OpName %UV "UV"
OpName %OutColor "OutColor"
OpDecorate %InputTexture DescriptorSet 0
OpDecorate %InputTexture Binding 0
OpDecorate %InputSampler DescriptorSet 0
OpDecorate %InputSampler Binding 0
OpDecorate %UV Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%Half = OpTypeFloat 16
%v4Half = OpTypeVector %Half 4
%14 = OpTypeImage %Half 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_14 = OpTypePointer UniformConstant %14
%InputTexture = OpVariable %_ptr_UniformConstant_14 UniformConstant
%18 = OpTypeSampler
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%InputSampler = OpVariable %_ptr_UniformConstant_18 UniformConstant
%22 = OpTypeSampledImage %14
%v2Half = OpTypeVector %Half 2
%_ptr_Input_v4Half = OpTypePointer Input %v4Half
%UV = OpVariable %_ptr_Input_v4Half Input
%_ptr_Output_v4Half = OpTypePointer Output %v4Half
%OutColor = OpVariable %_ptr_Output_v4Half Output
%MainPS = OpFunction %void None %3
%5 = OpLabel
%31 = OpLoad %v4Half %UV
%41 = OpLoad %14 %InputTexture
%42 = OpLoad %18 %InputSampler
%43 = OpSampledImage %22 %41 %42
%45 = OpVectorShuffle %v2Half %31 %31 0 1
%46 = OpImageSampleImplicitLod %v4Half %43 %45
OpStore %OutColor %46
OpReturn
OpFunctionEnd
)";

  const std::string after = 
      R"(OpCapability Shader
OpCapability Float16
OpCapability StorageInputOutput16
OpExtension "SPV_GOOGLE_hlsl_functionality1"
OpExtension "SPV_GOOGLE_user_type"
OpExtension "SPV_KHR_16bit_storage"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %MainPS "MainPS" %UV %OutColor
OpExecutionMode %MainPS OriginUpperLeft
OpSource HLSL 500
OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
OpSourceExtension "GL_GOOGLE_include_directive"
OpName %MainPS "MainPS"
OpName %InputTexture "InputTexture"
OpName %InputSampler "InputSampler"
OpName %UV "UV"
OpName %OutColor "OutColor"
OpDecorate %InputTexture DescriptorSet 0
OpDecorate %InputTexture Binding 0
OpDecorate %InputSampler DescriptorSet 0
OpDecorate %InputSampler Binding 0
OpDecorate %UV Location 0
OpDecorate %OutColor Location 0
OpDecorate %14 RelaxedPrecision
OpDecorate %46 RelaxedPrecision
OpDecorate %49 RelaxedPrecision
%float = OpTypeFloat 32
%void = OpTypeVoid
%3 = OpTypeFunction %void
%half = OpTypeFloat 16
%v4half = OpTypeVector %half 4
%14 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_14 = OpTypePointer UniformConstant %14
%InputTexture = OpVariable %_ptr_UniformConstant_14 UniformConstant
%18 = OpTypeSampler
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%InputSampler = OpVariable %_ptr_UniformConstant_18 UniformConstant
%22 = OpTypeSampledImage %14
%v2half = OpTypeVector %half 2
%_ptr_Input_v4half = OpTypePointer Input %v4half
%UV = OpVariable %_ptr_Input_v4half Input
%_ptr_Output_v4half = OpTypePointer Output %v4half
%OutColor = OpVariable %_ptr_Output_v4half Output
%v4float = OpTypeVector %float 4
%MainPS = OpFunction %void None %3
%5 = OpLabel
%31 = OpLoad %v4half %UV
%41 = OpLoad %14 %InputTexture
%42 = OpLoad %18 %InputSampler
%43 = OpSampledImage %22 %41 %42
%45 = OpVectorShuffle %v2half %31 %31 0 1
%46 = OpImageSampleImplicitLod %v4float %43 %45
%49 = OpFConvert %v4half %46
OpStore %OutColor %49
OpReturn
OpFunctionEnd
)";

  // Expected diff be like:
  // 
  // OpDecorate %14 RelaxedPrecision
  // OpDecorate %46 RelaxedPrecision
  // OpDecorate %49 RelaxedPrecision
  // [add relaxed percision for modified]
  // 
  // %float = OpTypeFloat 32
  // [Insert OpTypeFloat to the top of types]
  //
  // %14 = OpTypeImage %float 2D 0 0 0 1 Unknown
  // [Change half to float to bypass vulkan spec.]
  //
  // %v4float = OpTypeVector %float 4
  // [New v4float type]
  //
  // %46 = OpImageSampleImplicitLod %v4float %43 %45
  // %49 = OpFConvert %v4half %46
  // [Add convert op to convert sample back to half.]
  // OpStore %OutColor %49

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<LegalizeImageOpsPass>(before, after, true, true);
}


}  // namespace
}  // namespace opt
}  // namespace spvtools