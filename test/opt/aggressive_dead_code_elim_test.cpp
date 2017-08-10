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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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
      predefs1 + names_after + predefs2 + func_after, 
      true, true);
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

  SinglePassRunAndCheck<opt::AggressiveDCEPass>(
      assembly, assembly, true, true);
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


// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Check that logical addressing required
//    Check that function calls inhibit optimization
//    Others?

}  // anonymous namespace
