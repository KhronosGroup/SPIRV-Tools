// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

// Debug Printf Instrumentation Tests.

#include <string>
#include <vector>

#include "test/opt/assembly_builder.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InstDebugPrintfTest = PassTest<::testing::Test>;

TEST_F(InstDebugPrintfTest, V4Float32) {
  // SamplerState g_sDefault;
  // Texture2D g_tColor;
  //
  // struct PS_INPUT
  // {
  //   float2 vBaseTexCoord : TEXCOORD0;
  // };
  //
  // struct PS_OUTPUT
  // {
  //   float4 vDiffuse : SV_Target0;
  // };
  //
  // PS_OUTPUT MainPs(PS_INPUT i)
  // {
  //   PS_OUTPUT o;
  //
  //   o.vDiffuse.rgba = g_tColor.Sample(g_sDefault, (i.vBaseTexCoord.xy).xy);
  //   debugPrintfEXT("diffuse: %v4f", o.vDiffuse.rgba);
  //   return o;
  // }

  const std::string orig_defs =
      R"(OpCapability Shader
OpExtension "SPV_KHR_non_semantic_info"
%1 = OpExtInstImport "NonSemantic.DebugPrintf"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "MainPs" %3 %4
OpExecutionMode %2 OriginUpperLeft
%5 = OpString "Color is %vn"
)";

  const std::string new_defs =
      R"(OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %2 "MainPs" %3 %4 %gl_FragCoord
OpExecutionMode %2 OriginUpperLeft
%5 = OpString "Color is %vn"
)";

  const std::string orig_decorates =
      R"(OpDecorate %6 DescriptorSet 0
OpDecorate %6 Binding 1
OpDecorate %7 DescriptorSet 0
OpDecorate %7 Binding 0
OpDecorate %3 Location 0
OpDecorate %4 Location 0
)";

  const std::string added_decorates =
      R"(OpDecorate %_runtimearr_uint ArrayStride 4
OpDecorate %_struct_47 Block
OpMemberDecorate %_struct_47 0 Offset 0
OpMemberDecorate %_struct_47 1 Offset 4
OpDecorate %49 DescriptorSet 7
OpDecorate %49 Binding 3
OpDecorate %gl_FragCoord BuiltIn FragCoord
)";

  const std::string orig_globals =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%13 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%6 = OpVariable %_ptr_UniformConstant_13 UniformConstant
%15 = OpTypeSampler
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%7 = OpVariable %_ptr_UniformConstant_15 UniformConstant
%17 = OpTypeSampledImage %13
%_ptr_Input_v2float = OpTypePointer Input %v2float
%3 = OpVariable %_ptr_Input_v2float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%4 = OpVariable %_ptr_Output_v4float Output
)";

  const std::string added_globals =
      R"(%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%38 = OpTypeFunction %void %uint %uint %uint %uint %uint %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
%_struct_47 = OpTypeStruct %uint %_runtimearr_uint
%_ptr_StorageBuffer__struct_47 = OpTypePointer StorageBuffer %_struct_47
%49 = OpVariable %_ptr_StorageBuffer__struct_47 StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_0 = OpConstant %uint 0
%uint_12 = OpConstant %uint 12
%uint_4 = OpConstant %uint 4
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%uint_23 = OpConstant %uint 23
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%v4uint = OpTypeVector %uint 4
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%uint_9 = OpConstant %uint 9
%uint_10 = OpConstant %uint 10
%uint_11 = OpConstant %uint 11
%uint_36 = OpConstant %uint 36
)";

  const std::string main_before_printf =
      R"(%2 = OpFunction %void None %9
%20 = OpLabel
%21 = OpLoad %v2float %3
%22 = OpLoad %13 %6
%23 = OpLoad %15 %7
%24 = OpSampledImage %17 %22 %23
%25 = OpImageSampleImplicitLod %v4float %24 %21
)";

  const std::string printf_instruction =
      R"(%26 = OpExtInst %void %1 1 %5 %25
)";

  const std::string printf_instrumentation =
      R"(%29 = OpCompositeExtract %float %25 0
%30 = OpBitcast %uint %29
%31 = OpCompositeExtract %float %25 1
%32 = OpBitcast %uint %31
%33 = OpCompositeExtract %float %25 2
%34 = OpBitcast %uint %33
%35 = OpCompositeExtract %float %25 3
%36 = OpBitcast %uint %35
%101 = OpFunctionCall %void %37 %uint_36 %uint_5 %30 %32 %34 %36
OpBranch %102
%102 = OpLabel
)";

  const std::string main_after_printf =
      R"(OpStore %4 %25
OpReturn
OpFunctionEnd
)";

  const std::string output_func =
      R"(%37 = OpFunction %void None %38
%39 = OpFunctionParameter %uint
%40 = OpFunctionParameter %uint
%41 = OpFunctionParameter %uint
%42 = OpFunctionParameter %uint
%43 = OpFunctionParameter %uint
%44 = OpFunctionParameter %uint
%45 = OpLabel
%52 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_0
%55 = OpAtomicIAdd %uint %52 %uint_4 %uint_0 %uint_12
%56 = OpIAdd %uint %55 %uint_12
%57 = OpArrayLength %uint %49 1
%59 = OpULessThanEqual %bool %56 %57
OpSelectionMerge %60 None
OpBranchConditional %59 %61 %60
%61 = OpLabel
%62 = OpIAdd %uint %55 %uint_0
%64 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %62
OpStore %64 %uint_12
%66 = OpIAdd %uint %55 %uint_1
%67 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %66
OpStore %67 %uint_23
%69 = OpIAdd %uint %55 %uint_2
%70 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %69
OpStore %70 %39
%72 = OpIAdd %uint %55 %uint_3
%73 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %72
OpStore %73 %uint_4
%76 = OpLoad %v4float %gl_FragCoord
%78 = OpBitcast %v4uint %76
%79 = OpCompositeExtract %uint %78 0
%80 = OpIAdd %uint %55 %uint_4
%81 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %80
OpStore %81 %79
%82 = OpCompositeExtract %uint %78 1
%83 = OpIAdd %uint %55 %uint_5
%84 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %83
OpStore %84 %82
%86 = OpIAdd %uint %55 %uint_7
%87 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %86
OpStore %87 %40
%89 = OpIAdd %uint %55 %uint_8
%90 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %89
OpStore %90 %41
%92 = OpIAdd %uint %55 %uint_9
%93 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %92
OpStore %93 %42
%95 = OpIAdd %uint %55 %uint_10
%96 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %95
OpStore %96 %43
%98 = OpIAdd %uint %55 %uint_11
%99 = OpAccessChain %_ptr_StorageBuffer_uint %49 %uint_1 %98
OpStore %99 %44
OpBranch %60
%60 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InstDebugPrintfPass>(
      orig_defs + orig_decorates + orig_globals + main_before_printf +
          printf_instruction + main_after_printf,
      new_defs + orig_decorates + added_decorates + orig_globals +
          added_globals + main_before_printf + printf_instrumentation +
          main_after_printf + output_func,
      true, true, 7u, 23u);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//   Compute shader
//   Geometry shader
//   Tesselation control shader
//   Tesselation eval shader
//   Vertex shader

}  // namespace
}  // namespace opt
}  // namespace spvtools
