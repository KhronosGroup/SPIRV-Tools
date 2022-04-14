// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#include <unordered_set>

#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using AnalyzeDeadOutputStoresTest = PassTest<::testing::Test>;

TEST_F(AnalyzeDeadOutputStoresTest, FragMultipleLocations) {
  // Should report locations {2, 5}
  //
  // #version 450
  //
  // layout(location = 2) in Vertex
  // {
  //         vec4 color0;
  //         vec4 color1;
  //         vec4 color2[3];
  // } iVert;
  //
  // layout(location = 0) out vec4 oFragColor;
  //
  // void main()
  // {
  //     oFragColor = iVert.color0 + iVert.color2[1];
  // }
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %oFragColor %iVert
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %oFragColor "oFragColor"
               OpName %Vertex "Vertex"
               OpMemberName %Vertex 0 "color0"
               OpMemberName %Vertex 1 "color1"
               OpMemberName %Vertex 2 "color2"
               OpName %iVert "iVert"
               OpDecorate %oFragColor Location 0
               OpDecorate %Vertex Block
               OpDecorate %iVert Location 2
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
 %oFragColor = OpVariable %_ptr_Output_v4float Output
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
%_arr_v4float_uint_3 = OpTypeArray %v4float %uint_3
     %Vertex = OpTypeStruct %v4float %v4float %_arr_v4float_uint_3
%_ptr_Input_Vertex = OpTypePointer Input %Vertex
      %iVert = OpVariable %_ptr_Input_Vertex Input
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
      %int_2 = OpConstant %int 2
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %19 = OpAccessChain %_ptr_Input_v4float %iVert %int_0
         %20 = OpLoad %v4float %19
         %23 = OpAccessChain %_ptr_Input_v4float %iVert %int_2 %int_1
         %24 = OpLoad %v4float %23
         %25 = OpFAdd %v4float %20 %24
               OpStore %oFragColor %25
               OpReturn
               OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  std::unordered_set<uint32_t> live_inputs;
  auto result = SinglePassRunToBinary<EliminateDeadOutputStoresPass>(
      text, true, &live_inputs, /* analyze */ true);

  auto itr0 = live_inputs.find(0);
  auto itr1 = live_inputs.find(1);
  auto itr2 = live_inputs.find(2);
  auto itr3 = live_inputs.find(3);
  auto itr4 = live_inputs.find(4);
  auto itr5 = live_inputs.find(5);
  auto itr6 = live_inputs.find(6);

  // Expect live_inputs == {2, 5}
  EXPECT_TRUE(itr0 == live_inputs.end());
  EXPECT_TRUE(itr1 == live_inputs.end());
  EXPECT_TRUE(itr2 != live_inputs.end());
  EXPECT_TRUE(itr3 == live_inputs.end());
  EXPECT_TRUE(itr4 == live_inputs.end());
  EXPECT_TRUE(itr5 != live_inputs.end());
  EXPECT_TRUE(itr6 == live_inputs.end());
}

TEST_F(AnalyzeDeadOutputStoresTest, FragMatrix) {
  // Should report locations {2, 8, 9, 10, 11}
  //
  // #version 450
  //
  // uniform ui_name {
  //     int i;
  // } ui_inst;
  //
  // layout(location = 2) in Vertex
  // {
  //         vec4 color0;
  //         vec4 color1;
  //         mat4 color2;
  //         mat4 color3;
  //         mat4 color4;
  // } iVert;
  //
  // // Output variable for the color
  // layout(location = 0) out vec4 oFragColor;
  //
  // void main()
  // {
  //     oFragColor = iVert.color0 + iVert.color3[ui_inst.i];
  // }
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %oFragColor %iVert %ui_inst
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %oFragColor "oFragColor"
               OpName %Vertex "Vertex"
               OpMemberName %Vertex 0 "color0"
               OpMemberName %Vertex 1 "color1"
               OpMemberName %Vertex 2 "color2"
               OpMemberName %Vertex 3 "color3"
               OpMemberName %Vertex 4 "color4"
               OpName %iVert "iVert"
               OpName %ui_name "ui_name"
               OpMemberName %ui_name 0 "i"
               OpName %ui_inst "ui_inst"
               OpDecorate %oFragColor Location 0
               OpDecorate %Vertex Block
               OpDecorate %iVert Location 2
               OpMemberDecorate %ui_name 0 Offset 0
               OpDecorate %ui_name Block
               OpDecorate %ui_inst DescriptorSet 0
               OpDecorate %ui_inst Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
 %oFragColor = OpVariable %_ptr_Output_v4float Output
%mat4v4float = OpTypeMatrix %v4float 4
     %Vertex = OpTypeStruct %v4float %v4float %mat4v4float %mat4v4float %mat4v4float
%_ptr_Input_Vertex = OpTypePointer Input %Vertex
      %iVert = OpVariable %_ptr_Input_Vertex Input
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
      %int_3 = OpConstant %int 3
    %ui_name = OpTypeStruct %int
%_ptr_Uniform_ui_name = OpTypePointer Uniform %ui_name
    %ui_inst = OpVariable %_ptr_Uniform_ui_name Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
       %main = OpFunction %void None %3
          %5 = OpLabel
         %17 = OpAccessChain %_ptr_Input_v4float %iVert %int_0
         %18 = OpLoad %v4float %17
         %24 = OpAccessChain %_ptr_Uniform_int %ui_inst %int_0
         %25 = OpLoad %int %24
         %26 = OpAccessChain %_ptr_Input_v4float %iVert %int_3 %25
         %27 = OpLoad %v4float %26
         %28 = OpFAdd %v4float %18 %27
               OpStore %oFragColor %28
               OpReturn
               OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  std::unordered_set<uint32_t> live_inputs;
  auto result = SinglePassRunToBinary<EliminateDeadOutputStoresPass>(
      text, true, &live_inputs, /* analyze */ true);

  auto itr0 = live_inputs.find(0);
  auto itr1 = live_inputs.find(1);
  auto itr2 = live_inputs.find(2);
  auto itr3 = live_inputs.find(3);
  auto itr4 = live_inputs.find(4);
  auto itr5 = live_inputs.find(5);
  auto itr6 = live_inputs.find(6);
  auto itr7 = live_inputs.find(7);
  auto itr8 = live_inputs.find(8);
  auto itr9 = live_inputs.find(9);
  auto itr10 = live_inputs.find(10);
  auto itr11 = live_inputs.find(11);
  auto itr12 = live_inputs.find(12);
  auto itr13 = live_inputs.find(13);
  auto itr14 = live_inputs.find(14);
  auto itr15 = live_inputs.find(15);

  // Expect live_inputs == {2, 8, 9, 10, 11}
  EXPECT_TRUE(itr0 == live_inputs.end());
  EXPECT_TRUE(itr1 == live_inputs.end());
  EXPECT_TRUE(itr2 != live_inputs.end());
  EXPECT_TRUE(itr3 == live_inputs.end());
  EXPECT_TRUE(itr4 == live_inputs.end());
  EXPECT_TRUE(itr5 == live_inputs.end());
  EXPECT_TRUE(itr6 == live_inputs.end());
  EXPECT_TRUE(itr7 == live_inputs.end());
  EXPECT_TRUE(itr8 != live_inputs.end());
  EXPECT_TRUE(itr9 != live_inputs.end());
  EXPECT_TRUE(itr10 != live_inputs.end());
  EXPECT_TRUE(itr11 != live_inputs.end());
  EXPECT_TRUE(itr12 == live_inputs.end());
  EXPECT_TRUE(itr13 == live_inputs.end());
  EXPECT_TRUE(itr14 == live_inputs.end());
  EXPECT_TRUE(itr15 == live_inputs.end());
}

TEST_F(AnalyzeDeadOutputStoresTest, FragMemberLocs) {
  // Should report location {1}
  //
  // #version 450
  //
  // in Vertex
  // {
  //     layout (location = 1) vec4 Cd;
  //     layout (location = 0) vec2 uv;
  // } iVert;
  //
  // layout (location = 0) out vec4 fragColor;
  //
  // void main()
  // {
  //     vec4 color = vec4(iVert.Cd);
  //     fragColor = color;
  // }
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %iVert %fragColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %color "color"
               OpName %Vertex "Vertex"
               OpMemberName %Vertex 0 "Cd"
               OpMemberName %Vertex 1 "uv"
               OpName %iVert "iVert"
               OpName %fragColor "fragColor"
               OpMemberDecorate %Vertex 0 Location 1
               OpMemberDecorate %Vertex 1 Location 0
               OpDecorate %Vertex Block
               OpDecorate %fragColor Location 0
               OpDecorate %_struct_27 Block
               OpMemberDecorate %_struct_27 0 Location 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
    %v2float = OpTypeVector %float 2
     %Vertex = OpTypeStruct %v4float %v2float
%_ptr_Input_Vertex = OpTypePointer Input %Vertex
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
  %fragColor = OpVariable %_ptr_Output_v4float Output
 %_struct_27 = OpTypeStruct %v4float
%_ptr_Input__struct_27 = OpTypePointer Input %_struct_27
      %iVert = OpVariable %_ptr_Input__struct_27 Input
       %main = OpFunction %void None %3
          %5 = OpLabel
      %color = OpVariable %_ptr_Function_v4float Function
         %17 = OpAccessChain %_ptr_Input_v4float %iVert %int_0
         %18 = OpLoad %v4float %17
         %19 = OpCompositeExtract %float %18 0
         %20 = OpCompositeExtract %float %18 1
         %21 = OpCompositeExtract %float %18 2
         %22 = OpCompositeExtract %float %18 3
         %23 = OpCompositeConstruct %v4float %19 %20 %21 %22
               OpStore %color %23
         %26 = OpLoad %v4float %color
               OpStore %fragColor %26
               OpReturn
               OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  std::unordered_set<uint32_t> live_inputs;
  auto result = SinglePassRunToBinary<EliminateDeadOutputStoresPass>(
      text, true, &live_inputs, /* analyze */ true);

  auto itr0 = live_inputs.find(0);
  auto itr1 = live_inputs.find(1);

  // Expect live_inputs == {2, 5}
  EXPECT_TRUE(itr0 == live_inputs.end());
  EXPECT_TRUE(itr1 != live_inputs.end());
}

TEST_F(AnalyzeDeadOutputStoresTest, ArrayedInput) {
  // Tests handling of arrayed input seen in Tesc, Tese and Geom shaders.
  //
  // Should report location {1, 10}.
  //
  // #version 450
  //
  // layout (vertices = 4) out;
  //
  // layout (location = 1) in Vertex
  // {
  //                 vec4 p;
  //                 vec3 n;
  //                 vec4 f[100];
  // } iVert[];
  //
  // layout (location = 0) out vec4 position[4];
  //
  // void main()
  // {
  //                 vec4 pos = iVert[gl_InvocationID].p *
  //                            iVert[gl_InvocationID].f[7];
  //                 position[gl_InvocationID] = pos;
  // }
  const std::string text = R"(
               OpCapability Tessellation
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %iVert %gl_InvocationID %position
               OpExecutionMode %main OutputVertices 4
               OpSource GLSL 450
               OpName %main "main"
               OpName %Vertex "Vertex"
               OpMemberName %Vertex 0 "p"
               OpMemberName %Vertex 1 "n"
               OpMemberName %Vertex 2 "f"
               OpName %iVert "iVert"
               OpName %gl_InvocationID "gl_InvocationID"
               OpName %position "position"
               OpDecorate %Vertex Block
               OpDecorate %iVert Location 1
               OpDecorate %gl_InvocationID BuiltIn InvocationId
               OpDecorate %position Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %v3float = OpTypeVector %float 3
       %uint = OpTypeInt 32 0
   %uint_100 = OpConstant %uint 100
%_arr_v4float_uint_100 = OpTypeArray %v4float %uint_100
     %Vertex = OpTypeStruct %v4float %v3float %_arr_v4float_uint_100
    %uint_32 = OpConstant %uint 32
%_arr_Vertex_uint_32 = OpTypeArray %Vertex %uint_32
%_ptr_Input__arr_Vertex_uint_32 = OpTypePointer Input %_arr_Vertex_uint_32
      %iVert = OpVariable %_ptr_Input__arr_Vertex_uint_32 Input
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_InvocationID = OpVariable %_ptr_Input_int Input
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
      %int_2 = OpConstant %int 2
      %int_7 = OpConstant %int 7
     %uint_4 = OpConstant %uint 4
%_arr_v4float_uint_4 = OpTypeArray %v4float %uint_4
%_ptr_Output__arr_v4float_uint_4 = OpTypePointer Output %_arr_v4float_uint_4
   %position = OpVariable %_ptr_Output__arr_v4float_uint_4 Output
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %22 = OpLoad %int %gl_InvocationID
         %25 = OpAccessChain %_ptr_Input_v4float %iVert %22 %int_0
         %26 = OpLoad %v4float %25
         %30 = OpAccessChain %_ptr_Input_v4float %iVert %22 %int_2 %int_7
         %31 = OpLoad %v4float %30
         %32 = OpFMul %v4float %26 %31
         %40 = OpAccessChain %_ptr_Output_v4float %position %22
               OpStore %40 %32
               OpReturn
               OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  std::unordered_set<uint32_t> live_inputs;
  auto result = SinglePassRunToBinary<EliminateDeadOutputStoresPass>(
      text, true, &live_inputs, /* analyze */ true);

  auto itr0 = live_inputs.find(0);
  auto itr1 = live_inputs.find(1);
  auto itr2 = live_inputs.find(2);
  auto itr3 = live_inputs.find(3);
  auto itr4 = live_inputs.find(4);
  auto itr5 = live_inputs.find(5);
  auto itr6 = live_inputs.find(6);
  auto itr7 = live_inputs.find(7);
  auto itr8 = live_inputs.find(8);
  auto itr9 = live_inputs.find(9);
  auto itr10 = live_inputs.find(10);
  auto itr11 = live_inputs.find(11);

  // Expect live_inputs == {1, 10}
  EXPECT_TRUE(itr0 == live_inputs.end());
  EXPECT_TRUE(itr1 != live_inputs.end());
  EXPECT_TRUE(itr2 == live_inputs.end());
  EXPECT_TRUE(itr3 == live_inputs.end());
  EXPECT_TRUE(itr4 == live_inputs.end());
  EXPECT_TRUE(itr5 == live_inputs.end());
  EXPECT_TRUE(itr6 == live_inputs.end());
  EXPECT_TRUE(itr7 == live_inputs.end());
  EXPECT_TRUE(itr8 == live_inputs.end());
  EXPECT_TRUE(itr9 == live_inputs.end());
  EXPECT_TRUE(itr10 != live_inputs.end());
  EXPECT_TRUE(itr11 == live_inputs.end());
}

TEST_F(AnalyzeDeadOutputStoresTest, ArrayedInputMemberLocs) {
  // Tests handling of member locs with arrayed input seen in Tesc, Tese
  // and Geom shaders.
  //
  // Should report location {1, 12}.
  //
  // #version 450
  //
  // layout (vertices = 4) out;
  //
  // in Vertex
  // {
  //     layout (location = 1) vec4 p;
  //     layout (location = 3) vec3 n;
  //     layout (location = 5) vec4 f[100];
  // } iVert[];
  //
  // layout (location = 0) out vec4 position[4];
  //
  // void main()
  // {
  //                 vec4 pos = iVert[gl_InvocationID].p *
  //                            iVert[gl_InvocationID].f[7];
  //                 position[gl_InvocationID] = pos;
  // }
  const std::string text = R"(
               OpCapability Tessellation
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %iVert %gl_InvocationID %position
               OpExecutionMode %main OutputVertices 4
               OpSource GLSL 450
               OpName %main "main"
               OpName %Vertex "Vertex"
               OpMemberName %Vertex 0 "p"
               OpMemberName %Vertex 1 "n"
               OpMemberName %Vertex 2 "f"
               OpName %iVert "iVert"
               OpName %gl_InvocationID "gl_InvocationID"
               OpName %position "position"
               OpMemberDecorate %Vertex 0 Location 1
               OpMemberDecorate %Vertex 1 Location 3
               OpMemberDecorate %Vertex 2 Location 5
               OpDecorate %Vertex Block
               OpDecorate %gl_InvocationID BuiltIn InvocationId
               OpDecorate %position Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %v3float = OpTypeVector %float 3
       %uint = OpTypeInt 32 0
   %uint_100 = OpConstant %uint 100
%_arr_v4float_uint_100 = OpTypeArray %v4float %uint_100
     %Vertex = OpTypeStruct %v4float %v3float %_arr_v4float_uint_100
    %uint_32 = OpConstant %uint 32
%_arr_Vertex_uint_32 = OpTypeArray %Vertex %uint_32
%_ptr_Input__arr_Vertex_uint_32 = OpTypePointer Input %_arr_Vertex_uint_32
      %iVert = OpVariable %_ptr_Input__arr_Vertex_uint_32 Input
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%gl_InvocationID = OpVariable %_ptr_Input_int Input
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
      %int_2 = OpConstant %int 2
      %int_7 = OpConstant %int 7
     %uint_4 = OpConstant %uint 4
%_arr_v4float_uint_4 = OpTypeArray %v4float %uint_4
%_ptr_Output__arr_v4float_uint_4 = OpTypePointer Output %_arr_v4float_uint_4
   %position = OpVariable %_ptr_Output__arr_v4float_uint_4 Output
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %22 = OpLoad %int %gl_InvocationID
         %25 = OpAccessChain %_ptr_Input_v4float %iVert %22 %int_0
         %26 = OpLoad %v4float %25
         %30 = OpAccessChain %_ptr_Input_v4float %iVert %22 %int_2 %int_7
         %31 = OpLoad %v4float %30
         %32 = OpFMul %v4float %26 %31
         %40 = OpAccessChain %_ptr_Output_v4float %position %22
               OpStore %40 %32
               OpReturn
               OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);

  std::unordered_set<uint32_t> live_inputs;
  auto result = SinglePassRunToBinary<EliminateDeadOutputStoresPass>(
      text, true, &live_inputs, /* analyze */ true);

  auto itr0 = live_inputs.find(0);
  auto itr1 = live_inputs.find(1);
  auto itr2 = live_inputs.find(2);
  auto itr3 = live_inputs.find(3);
  auto itr4 = live_inputs.find(4);
  auto itr5 = live_inputs.find(5);
  auto itr6 = live_inputs.find(6);
  auto itr7 = live_inputs.find(7);
  auto itr8 = live_inputs.find(8);
  auto itr9 = live_inputs.find(9);
  auto itr10 = live_inputs.find(10);
  auto itr11 = live_inputs.find(11);
  auto itr12 = live_inputs.find(12);
  auto itr13 = live_inputs.find(13);

  // Expect live_inputs == {1, 12}
  EXPECT_TRUE(itr0 == live_inputs.end());
  EXPECT_TRUE(itr1 != live_inputs.end());
  EXPECT_TRUE(itr2 == live_inputs.end());
  EXPECT_TRUE(itr3 == live_inputs.end());
  EXPECT_TRUE(itr4 == live_inputs.end());
  EXPECT_TRUE(itr5 == live_inputs.end());
  EXPECT_TRUE(itr6 == live_inputs.end());
  EXPECT_TRUE(itr7 == live_inputs.end());
  EXPECT_TRUE(itr8 == live_inputs.end());
  EXPECT_TRUE(itr9 == live_inputs.end());
  EXPECT_TRUE(itr10 == live_inputs.end());
  EXPECT_TRUE(itr11 == live_inputs.end());
  EXPECT_TRUE(itr12 != live_inputs.end());
  EXPECT_TRUE(itr13 == live_inputs.end());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
