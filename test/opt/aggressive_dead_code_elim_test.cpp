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

TEST_F(AggressiveDCETest, EliminateExtendedInst) {
  //  #version 140
  //
  //  in vec4 BaseColor;
  //  in vec4 Dead;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      vec4 dv = sqrt(Dead);
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
%18 = OpExtInst %v4float %1 Sqrt %17 
OpStore %dv %18
%19 = OpLoad %v4float %v
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%15 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%16 = OpLoad %v4float %BaseColor
OpStore %v %16
%19 = OpLoad %v4float %v
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateFrexp) {
  // Note: SPIR-V hand-edited to utilize Frexp
  //
  // #version 450
  //
  // in vec4 BaseColor;
  // in vec4 Dead;
  // out vec4 Color;
  // out ivec4 iv2;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     vec4 dv = frexp(Dead, iv2);
  //     Color = v;
  // }

  const std::string predefs1 =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %iv2 %Color
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
)";

  const std::string names_before =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %iv2 "iv2"
OpName %ResType "ResType"
OpName %Color "Color"
)";

  const std::string names_after =
      R"(OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %Dead "Dead"
OpName %iv2 "iv2"
OpName %ResType "ResType"
OpName %Color "Color"
)";

  const std::string predefs2 =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%int = OpTypeInt 32 1
%v4int = OpTypeVector %int 4
%_ptr_Output_v4int = OpTypePointer Output %v4int
%iv2 = OpVariable %_ptr_Output_v4int Output
%ResType = OpTypeStruct %v4float %v4int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%Color = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %11
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function 
%21 = OpLoad %v4float %BaseColor
OpStore %v %21
%22 = OpLoad %v4float %Dead
%23 = OpExtInst %v4float %1 Frexp %22 %iv2
OpStore %dv %23
%24 = OpLoad %v4float %v
OpStore %Color %24
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %11
%20 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %BaseColor
OpStore %v %21
%22 = OpLoad %v4float %Dead
%23 = OpExtInst %v4float %1 Frexp %22 %iv2
%24 = OpLoad %v4float %v
OpStore %Color %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDecorate) {
  // Note: The SPIR-V was hand-edited to add the OpDecorate
  //
  // #version 140
  //
  // in vec4 BaseColor;
  // in vec4 Dead;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     vec4 dv = Dead * 0.5;
  //     gl_FragColor = v;
  // }

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
OpDecorate %8 RelaxedPrecision
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
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%float_0_5 = OpConstant %float 0.5
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %10
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%dv = OpVariable %_ptr_Function_v4float Function 
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
%19 = OpLoad %v4float %Dead
%8 = OpVectorTimesScalar %v4float %19 %float_0_5
OpStore %dv %8
%20 = OpLoad %v4float %v
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %10
%17 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%18 = OpLoad %v4float %BaseColor
OpStore %v %18
%20 = OpLoad %v4float %v
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs1 + names_before + predefs2 + func_before,
      predefs1 + names_after + predefs2 + func_after, true, true);
}

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
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, DeadCycle) {
  // #version 140
  // in vec4 BaseColor;
  //
  // layout(std140) uniform U_t
  // {
  //     int g_I ;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     float df = 0.0;
  //     int i = 0;
  //     while (i < g_I) {
  //       df = df * 0.5;
  //       i = i + 1;
  //     }
  //     gl_FragColor = v;
  // }

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
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, OptWhitelistExtension) {
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
OpExtension "SPV_AMD_gpu_shader_int16"
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
      predefs1 + names_after + predefs2 + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoOptBlacklistExtension) {
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

  const std::string assembly =
      R"(OpCapability Shader
OpExtension "SPV_KHR_variable_pointers"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %9
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

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, ElimWithCall) {
  // This demonstrates that "dead" function calls are not eliminated.
  // Also demonstrates that DCE will happen in presence of function call.
  // #version 140
  // in vec4 i1;
  // in vec4 i2;
  //
  // void nothing(vec4 v)
  // {
  // }
  //
  // void main()
  // {
  //     vec4 v1 = i1;
  //     vec4 v2 = i2;
  //     nothing(v1);
  //     gl_FragColor = vec4(0.0);
  // }

  const std::string defs_before =
      R"( OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %i1 %i2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %nothing_vf4_ "nothing(vf4;" 
OpName %v "v"
OpName %v1 "v1"
OpName %i1 "i1"
OpName %v2 "v2"
OpName %i2 "i2"
OpName %param "param" 
OpName %gl_FragColor "gl_FragColor" 
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%16 = OpTypeFunction %void %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%i1 = OpVariable %_ptr_Input_v4float Input
%i2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0 
)";

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %i1 %i2 %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %nothing_vf4_ "nothing(vf4;"
OpName %v "v"
OpName %v1 "v1"
OpName %i1 "i1"
OpName %i2 "i2"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%16 = OpTypeFunction %void %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%i1 = OpVariable %_ptr_Input_v4float Input
%i2 = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %12
%21 = OpLabel
%v1 = OpVariable %_ptr_Function_v4float Function 
%v2 = OpVariable %_ptr_Function_v4float Function 
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %i1
OpStore %v1 %22
%23 = OpLoad %v4float %i2
OpStore %v2 %23
%24 = OpLoad %v4float %v1
OpStore %param %24
%25 = OpFunctionCall %void %nothing_vf4_ %param
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
%nothing_vf4_ = OpFunction %void None %16
%v = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %12
%21 = OpLabel
%v1 = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %i1
OpStore %v1 %22
%24 = OpLoad %v4float %v1
OpStore %param %24
%25 = OpFunctionCall %void %nothing_vf4_ %param
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
%nothing_vf4_ = OpFunction %void None %16
%v = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      defs_before + func_before, defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoParamElim) {
  // This demonstrates that unused parameters are not eliminated, but
  // dead uses of them are.
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // vec4 foo(vec4 v1, vec4 v2)
  // {
  //     vec4 t = -v1;
  //     return v2;
  // }
  //
  // void main()
  // {
  //     vec4 dead;
  //     gl_FragColor = foo(dead, BaseColor);
  // }

  const std::string defs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;" 
OpName %v1 "v1"
OpName %v2 "v2"
OpName %t "t"
OpName %gl_FragColor "gl_FragColor" 
OpName %dead "dead"
OpName %BaseColor "BaseColor"
OpName %param "param" 
OpName %param_0 "param"
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%17 = OpTypeFunction %v4float %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %13
%20 = OpLabel
%dead = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %dead
OpStore %param %21
%22 = OpLoad %v4float %BaseColor
OpStore %param_0 %22
%23 = OpFunctionCall %v4float %foo_vf4_vf4_ %param %param_0
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;"
OpName %v1 "v1"
OpName %v2 "v2"
OpName %gl_FragColor "gl_FragColor"
OpName %dead "dead"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %param_0 "param"
%void = OpTypeVoid
%13 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%17 = OpTypeFunction %v4float %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %13
%20 = OpLabel
%dead = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%21 = OpLoad %v4float %dead
OpStore %param %21
%22 = OpLoad %v4float %BaseColor
OpStore %param_0 %22
%23 = OpFunctionCall %v4float %foo_vf4_vf4_ %param %param_0
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string func_before =
      R"(%foo_vf4_vf4_ = OpFunction %v4float None %17
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%24 = OpLabel
%t = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %v1 
%26 = OpFNegate %v4float %25
OpStore %t %26 
%27 = OpLoad %v4float %v2 
OpReturnValue %27
OpFunctionEnd
)";

  const std::string func_after =
      R"(%foo_vf4_vf4_ = OpFunction %v4float None %17
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%24 = OpLabel
%27 = OpLoad %v4float %v2
OpReturnValue %27
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      defs_before + func_before, defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, ElimOpaque) {
  // SPIR-V not representable from GLSL; not generatable from HLSL
  // for the moment.

  const std::string defs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpMemberName %S_t 2 "smp"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %s0 "s0"
OpName %texCoords "texCoords"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%14 = OpTypeImage %float 2D 0 0 0 1 Unknown
%15 = OpTypeSampledImage %14
%S_t = OpTypeStruct %v2float %v2float %15
%_ptr_Function_S_t = OpTypePointer Function %S_t
%17 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%_ptr_Function_15 = OpTypePointer Function %15
%sampler15 = OpVariable %_ptr_UniformConstant_15 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string defs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %outColor %texCoords
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpMemberName %S_t 2 "smp"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %texCoords "texCoords"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%14 = OpTypeImage %float 2D 0 0 0 1 Unknown
%15 = OpTypeSampledImage %14
%S_t = OpTypeStruct %v2float %v2float %15
%_ptr_Function_S_t = OpTypePointer Function %S_t
%17 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_15 = OpTypePointer UniformConstant %15
%_ptr_Function_15 = OpTypePointer Function %15
%sampler15 = OpVariable %_ptr_UniformConstant_15 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function 
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0 
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%31 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %31
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%26 = OpLoad %v2float %texCoords
%29 = OpLoad %15 %sampler15
%31 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      defs_before + func_before, defs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoParamStoreElim) {
  // Should not eliminate stores to params
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void foo(in vec4 v1, out vec4 v2)
  // {
  //     v2 = -v1;
  // }
  //
  // void main()
  // {
  //     foo(BaseColor, OutColor);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_vf4_vf4_ "foo(vf4;vf4;"
OpName %v1 "v1"
OpName %v2 "v2"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpName %param "param"
OpName %param_0 "param"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%15 = OpTypeFunction %void %_ptr_Function_v4float %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %11
%18 = OpLabel
%param = OpVariable %_ptr_Function_v4float Function
%param_0 = OpVariable %_ptr_Function_v4float Function
%19 = OpLoad %v4float %BaseColor
OpStore %param %19
%20 = OpFunctionCall %void %foo_vf4_vf4_ %param %param_0
%21 = OpLoad %v4float %param_0
OpStore %OutColor %21
OpReturn
OpFunctionEnd
%foo_vf4_vf4_ = OpFunction %void None %15
%v1 = OpFunctionParameter %_ptr_Function_v4float
%v2 = OpFunctionParameter %_ptr_Function_v4float
%22 = OpLabel
%23 = OpLoad %v4float %v1
%24 = OpFNegate %v4float %23
OpStore %v2 %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, PrivateStoreElimInEntryNoCalls) {
  // Eliminate stores to private in entry point with no calls
  // Note: Not legal GLSL
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 1) in vec4 Dead;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 dv;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     dv = Dead;
  //     OutColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Dead %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %dv "dv"
OpName %Dead "Dead"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %Dead Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%Dead = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%dv = OpVariable %_ptr_Private_v4float Private
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string main_before =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%18 = OpLoad %v4float %Dead
OpStore %dv %18
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  const std::string main_after =
      R"(%main = OpFunction %void None %9
%16 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%17 = OpLoad %v4float %BaseColor
OpStore %v %17
%19 = OpLoad %v4float %v
%20 = OpFNegate %v4float %19
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs + main_before, predefs + main_after, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimIfLoad) {
  // Should not eliminate stores to private when there is a load
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 pv;
  //
  // void main()
  // {
  //     pv = BaseColor;
  //     OutColor = pv;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %pv "pv"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%pv = OpVariable %_ptr_Private_v4float Private
%main = OpFunction %void None %7
%13 = OpLabel
%14 = OpLoad %v4float %BaseColor
OpStore %pv %14
%15 = OpLoad %v4float %pv
%16 = OpFNegate %v4float %15
OpStore %OutColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimWithCall) {
  // Should not eliminate stores to private when function contains call
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 v1;
  //
  // void foo()
  // {
  //     OutColor = -v1;
  // }
  //
  // void main()
  // {
  //     v1 = BaseColor;
  //     foo();
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
OpName %OutColor "OutColor"
OpName %v1 "v1"
OpName %BaseColor "BaseColor"
OpDecorate %OutColor Location 0
OpDecorate %BaseColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%v1 = OpVariable %_ptr_Private_v4float Private
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %8
%14 = OpLabel
%15 = OpLoad %v4float %BaseColor
OpStore %v1 %15
%16 = OpFunctionCall %void %foo_
OpReturn
OpFunctionEnd
%foo_ = OpFunction %void None %8
%17 = OpLabel
%18 = OpLoad %v4float %v1
%19 = OpFNegate %v4float %18
OpStore %OutColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoPrivateStoreElimInNonEntry) {
  // Should not eliminate stores to private when function is not entry point
  // Note: Not legal GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // private vec4 v1;
  //
  // void foo()
  // {
  //     v1 = BaseColor;
  // }
  //
  // void main()
  // {
  //     foo();
  //     OutColor = -v1;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
OpName %v1 "v1"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Private_v4float = OpTypePointer Private %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%v1 = OpVariable %_ptr_Private_v4float Private
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%14 = OpLabel
%15 = OpFunctionCall %void %foo_
%16 = OpLoad %v4float %v1
%17 = OpFNegate %v4float %16
OpStore %OutColor %17
OpReturn
OpFunctionEnd
%foo_ = OpFunction %void None %8
%18 = OpLabel
%19 = OpLoad %v4float %BaseColor
OpStore %v1 %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThenElse) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     else
  //       d = BaseColor.z;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%21 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%21 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%22 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%23 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%24 = OpLoad %float %23
%25 = OpFOrdEqual %bool %24 %float_0 
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %28
%27 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%30 = OpLoad %float %29
OpStore %d %30
OpBranch %26
%28 = OpLabel
%31 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%32 = OpLoad %float %31
OpStore %d %32
OpBranch %26
%26 = OpLabel
OpStore %OutColor %21
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%22 = OpLabel
OpBranch %26
%26 = OpLabel
OpStore %OutColor %21
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThen) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%20 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%20 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%21 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%22 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%23 = OpLoad %float %22
%24 = OpFOrdEqual %bool %23 %float_0 
OpSelectionMerge %25 None
OpBranchConditional %24 %26 %25
%26 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %d %28
OpBranch %25
%25 = OpLabel
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%21 = OpLabel
OpBranch %25
%25 = OpLabel
OpStore %OutColor %20
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

// This test fails. OpSwitch is not handled by ADCE.
// (https://github.com/KhronosGroup/SPIRV-Tools/issues/1021).
TEST_F(AggressiveDCETest, DISABLED_EliminateDeadSwitch) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 1) in flat int x;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     switch (x) {
  //       case 0:
  //         d = BaseColor.y;
  //     }
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }
  const std::string before =
      R"(OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %x %BaseColor %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %d "d"
               OpName %BaseColor "BaseColor"
               OpName %OutColor "OutColor"
               OpDecorate %x Flat
               OpDecorate %x Location 1
               OpDecorate %BaseColor Location 0
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
          %x = OpVariable %_ptr_Input_int Input
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
  %BaseColor = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %OutColor = OpVariable %_ptr_Output_v4float Output
    %float_1 = OpConstant %float 1
         %27 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %d = OpVariable %_ptr_Function_float Function
          %9 = OpLoad %int %x
               OpSelectionMerge %11 None
               OpSwitch %9 %11 0 %10
         %10 = OpLabel
         %21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
         %22 = OpLoad %float %21
               OpStore %d %22
               OpBranch %11
         %11 = OpLabel
               OpStore %OutColor %27
               OpReturn
               OpFunctionEnd)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %x %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %x "x"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %x Flat
OpDecorate %x Location 1
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%8 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Input_int = OpTypePointer Input %int
%x = OpVariable %_ptr_Input_int Input
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%20 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%main = OpFunction %void None %8
%21 = OpLabel
OpBranch %11
%11 = OpLabel
OpStore %OutColor %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(before, after, true, true);
}

TEST_F(AggressiveDCETest, EliminateDeadIfThenElseNested) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       if (BaseColor.y == 0)
  //         d = 0.0;
  //       else
  //         d = 0.25;
  //     else
  //       if (BaseColor.y == 0)
  //         d = 0.5;
  //       else
  //         d = 0.75;
  //     OutColor = vec4(1.0,1.0,1.0,1.0);
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_1 = OpConstant %float 1
%23 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%24 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%25 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%26 = OpLoad %float %25
%27 = OpFOrdEqual %bool %26 %float_0 
OpSelectionMerge %28 None
OpBranchConditional %27 %29 %30
%29 = OpLabel
%31 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%32 = OpLoad %float %31
%33 = OpFOrdEqual %bool %32 %float_0 
OpSelectionMerge %34 None
OpBranchConditional %33 %35 %36
%35 = OpLabel
OpStore %d %float_0
OpBranch %34
%36 = OpLabel
OpStore %d %float_0_25 
OpBranch %34
%34 = OpLabel
OpBranch %28
%30 = OpLabel
%37 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%38 = OpLoad %float %37
%39 = OpFOrdEqual %bool %38 %float_0 
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %42
%41 = OpLabel
OpStore %d %float_0_5
OpBranch %40
%42 = OpLabel
OpStore %d %float_0_75 
OpBranch %40
%40 = OpLabel
OpBranch %28
%28 = OpLabel
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%24 = OpLabel
OpBranch %28
%28 = OpLabel
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveIfThenElse) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       t = BaseColor.y;
  //     else
  //       t = BaseColor.z;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %t "t"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%20 = OpLabel
%t = OpVariable %_ptr_Function_float Function
%21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%22 = OpLoad %float %21
%23 = OpFOrdEqual %bool %22 %float_0
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %t %28
OpBranch %24
%26 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%30 = OpLoad %float %29
OpStore %t %30
OpBranch %24
%24 = OpLabel
%31 = OpLoad %float %t
%32 = OpCompositeConstruct %v4float %31 %31 %31 %31
OpStore %OutColor %32
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateLiveIfThenElseNested) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       if (BaseColor.y == 0)
  //         t = 0.0;
  //       else
  //         t = 0.25;
  //     else
  //       if (BaseColor.y == 0)
  //         t = 0.5;
  //       else
  //         t = 0.75;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %t "t"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %7
%22 = OpLabel
%t = OpVariable %_ptr_Function_float Function
%23 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%24 = OpLoad %float %23
%25 = OpFOrdEqual %bool %24 %float_0
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %28
%27 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%30 = OpLoad %float %29
%31 = OpFOrdEqual %bool %30 %float_0
OpSelectionMerge %32 None
OpBranchConditional %31 %33 %34
%33 = OpLabel
OpStore %t %float_0
OpBranch %32
%34 = OpLabel
OpStore %t %float_0_25
OpBranch %32
%32 = OpLabel
OpBranch %26
%28 = OpLabel
%35 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%36 = OpLoad %float %35
%37 = OpFOrdEqual %bool %36 %float_0
OpSelectionMerge %38 None
OpBranchConditional %37 %39 %40
%39 = OpLabel
OpStore %t %float_0_5
OpBranch %38
%40 = OpLabel
OpStore %t %float_0_75
OpBranch %38
%38 = OpLabel
OpBranch %26
%26 = OpLabel
%41 = OpLoad %float %t
%42 = OpCompositeConstruct %v4float %41 %41 %41 %41
OpStore %OutColor %42
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfWithPhi) {
  // Note: Assembly hand-optimized from GLSL
  //
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float t;
  //     if (BaseColor.x == 0)
  //       t = 0.0;
  //     else
  //       t = 1.0;
  //     OutColor = vec4(t);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %6
%18 = OpLabel
%19 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%20 = OpLoad %float %19
%21 = OpFOrdEqual %bool %20 %float_0
OpSelectionMerge %22 None
OpBranchConditional %21 %23 %24
%23 = OpLabel
OpBranch %22
%24 = OpLabel
OpBranch %22
%22 = OpLabel
%25 = OpPhi %float %float_0 %23 %float_1 %24
%26 = OpCompositeConstruct %v4float %25 %25 %25 %25
OpStore %OutColor %26
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, NoEliminateIfBreak) {
  // Note: Assembly optimized from GLSL
  //
  // #version 450 
  // 
  // layout(location=0) in vec4 InColor;
  // layout(location=0) out vec4 OutColor;
  // 
  // void main()
  // {
  //     float f = 0.0;
  //     for (;;) {
  //         f += 2.0;
  //         if (f > 20.0)
  //             break;
  //     }
  // 
  //     OutColor = InColor / f;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %InColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %f "f"
OpName %OutColor "OutColor"
OpName %InColor "InColor"
OpDecorate %OutColor Location 0
OpDecorate %InColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%float_2 = OpConstant %float 2
%float_20 = OpConstant %float 20
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%InColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %7
%17 = OpLabel
%f = OpVariable %_ptr_Function_float Function
OpStore %f %float_0
OpBranch %18
%18 = OpLabel
OpLoopMerge %19 %20 None
OpBranch %21
%21 = OpLabel
%22 = OpLoad %float %f
%23 = OpFAdd %float %22 %float_2
OpStore %f %23
%24 = OpLoad %float %f
%25 = OpFOrdGreaterThan %bool %24 %float_20
OpSelectionMerge %26 None
OpBranchConditional %25 %27 %26
%27 = OpLabel
OpBranch %19
%26 = OpLabel
OpBranch %20
%20 = OpLabel
OpBranch %18
%19 = OpLabel
%28 = OpLoad %v4float %InColor
%29 = OpLoad %float %f
%30 = OpCompositeConstruct %v4float %29 %29 %29 %29
%31 = OpFDiv %v4float %28 %30
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(assembly, assembly, true, true);
}

TEST_F(AggressiveDCETest, EliminateEntireFunctionBody) {
  // #version 450
  //
  // layout(location = 0) in vec4 BaseColor;
  // layout(location = 0) out vec4 OutColor;
  //
  // void main()
  // {
  //     float d;
  //     if (BaseColor.x == 0)
  //       d = BaseColor.y;
  //     else
  //       d = BaseColor.z;
  // }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %d "d"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %OutColor "OutColor"
OpDecorate %BaseColor Location 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string func_before =
      R"(%main = OpFunction %void None %7
%20 = OpLabel
%d = OpVariable %_ptr_Function_float Function
%21 = OpAccessChain %_ptr_Input_float %BaseColor %uint_0
%22 = OpLoad %float %21
%23 = OpFOrdEqual %bool %22 %float_0 
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %26
%25 = OpLabel
%27 = OpAccessChain %_ptr_Input_float %BaseColor %uint_1
%28 = OpLoad %float %27
OpStore %d %28
OpBranch %24
%26 = OpLabel
%29 = OpAccessChain %_ptr_Input_float %BaseColor %uint_2
%30 = OpLoad %float %29
OpStore %d %30
OpBranch %24
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string func_after =
      R"(%main = OpFunction %void None %7
%20 = OpLabel
OpBranch %24
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      predefs_before + func_before, predefs_after + func_after, true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Check that logical addressing required
//    Check that function calls inhibit optimization
//    Others?

}  // anonymous namespace
