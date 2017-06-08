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

#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;

using AggressiveDCETest = PassTest<::testing::Test>;

TEST_F(AggressiveDCETest, Simple) {
  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //  
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = Dead;
  //      gl_FragColor = v;
  //  }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function 
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%17 = OpLoad %v4float %Dead
OpStore %dv %17
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%18 = OpLoad %v4float %v
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before, 
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
}

TEST_F(AggressiveDCETest, DeadCycle) {
  //  

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %df "df"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_I"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %i "i"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_I"
OpName %_ ""
OpName %gl_FragColor "gl_FragColor"
)";

  const std::string predefs2 =
      R"(OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%U_t = OpTypeStruct %int
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
%bool = OpTypeBool
%float_0_5 = OpConstant %float 0.5
%int_1 = OpConstant %int 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %11
%27 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%df = OpVariable %_ptr_Function_float Function 
%i = OpVariable %_ptr_Function_int Function
%28 = OpLoad %v4float %BaseColor
OpStore %v %28
OpStore %df %float_0
OpStore %i %int_0
OpBranch %29
%29 = OpLabel
OpLoopMerge %30 %31 None
OpBranch %32
%32 = OpLabel
%33 = OpLoad %int %i
%34 = OpAccessChain %_ptr_Uniform_int %_ %int_0
%35 = OpLoad %int %34
%36 = OpSLessThan %bool %33 %35
OpBranchConditional %36 %37 %30
%37 = OpLabel
%38 = OpLoad %float %df
%39 = OpFMul %float %38 %float_0_5
OpStore %df %39
%40 = OpLoad %int %i
%41 = OpIAdd %int %40 %int_1
OpStore %i %41
OpBranch %31
%31 = OpLabel
OpBranch %29
%30 = OpLabel
%42 = OpLoad %v4float %v
OpStore %gl_FragColor %42
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %11
%27 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%i = OpVariable %_ptr_Function_int Function
%28 = OpLoad %v4float %BaseColor
OpStore %v %28
OpStore %i %int_0
OpBranch %29
%29 = OpLabel
OpLoopMerge %30 %31 None
OpBranch %32
%32 = OpLabel
%33 = OpLoad %int %i
%34 = OpAccessChain %_ptr_Uniform_int %_ %int_0
%35 = OpLoad %int %34
%36 = OpSLessThan %bool %33 %35
OpBranchConditional %36 %37 %30
%37 = OpLabel
%40 = OpLoad %int %i
%41 = OpIAdd %int %40 %int_1
OpStore %i %41
OpBranch %31
%31 = OpLabel
OpBranch %29
%30 = OpLabel
%42 = OpLoad %v4float %v
OpStore %gl_FragColor %42
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before, 
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Check that logical addressing required
//    Check that function calls inhibit optimization
//    Others?

}  // anonymous namespace
