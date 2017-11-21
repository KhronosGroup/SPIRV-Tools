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

using LocalAccessChainConvertTest = PassTest<::testing::Test>;

TEST_F(LocalAccessChainConvertTest, StructOfVecsOfFloatConverted) {

  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //  
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%22 = OpLoad %S_t %s0
%23 = OpCompositeInsert %S_t %18 %22 1
OpStore %s0 %23
%24 = OpLoad %S_t %s0
%25 = OpCompositeExtract %v4float %24 1
OpStore %gl_FragColor %25
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      predefs_before + before, predefs_after + after, true, true);
}

TEST_F(LocalAccessChainConvertTest, InBoundsAccessChainsConverted) {

  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //  
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpInBoundsAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpInBoundsAccessChain %_ptr_Function_v4float %s0 %int_1
%21 = OpLoad %v4float %20
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%22 = OpLoad %S_t %s0
%23 = OpCompositeInsert %S_t %18 %22 1
OpStore %s0 %23
%24 = OpLoad %S_t %s0
%25 = OpCompositeExtract %v4float %24 1
OpStore %gl_FragColor %25
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      predefs_before + before, predefs_after + after, true, true);
}

TEST_F(LocalAccessChainConvertTest, TwoUsesofSingleChainConverted) {

  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //  
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %19 %18
%20 = OpLoad %v4float %19
OpStore %gl_FragColor %20
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%21 = OpLoad %S_t %s0
%22 = OpCompositeInsert %S_t %18 %21 1
OpStore %s0 %22
%23 = OpLoad %S_t %s0
%24 = OpCompositeExtract %v4float %23 1
OpStore %gl_FragColor %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      predefs_before + before, predefs_after + after, true, true);
}

TEST_F(LocalAccessChainConvertTest, OpaqueConverted) {
  // SPIR-V not representable in GLSL; not generatable from HLSL
  // at the moment

  const std::string predefs =
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
OpName %foo_struct_S_t_vf2_vf21_ "foo(struct-S_t-vf2-vf21;"
OpName %s "s"
OpName %outColor "outColor"
OpName %sampler15 "sampler15"
OpName %s0 "s0"
OpName %texCoords "texCoords"
OpName %param "param"
OpDecorate %sampler15 DescriptorSet 0
%void = OpTypeVoid
%12 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outColor = OpVariable %_ptr_Output_v4float Output
%17 = OpTypeImage %float 2D 0 0 0 1 Unknown
%18 = OpTypeSampledImage %17
%S_t = OpTypeStruct %v2float %v2float %18
%_ptr_Function_S_t = OpTypePointer Function %S_t
%20 = OpTypeFunction %void %_ptr_Function_S_t
%_ptr_UniformConstant_18 = OpTypePointer UniformConstant %18
%_ptr_Function_18 = OpTypePointer Function %18
%sampler15 = OpVariable %_ptr_UniformConstant_18 UniformConstant
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_2 = OpConstant %int 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%texCoords = OpVariable %_ptr_Input_v2float Input
)";

  const std::string before =
      R"(%main = OpFunction %void None %12
%28 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function 
%param = OpVariable %_ptr_Function_S_t Function
%29 = OpLoad %v2float %texCoords
%30 = OpAccessChain %_ptr_Function_v2float %s0 %int_0
OpStore %30 %29
%31 = OpLoad %18 %sampler15
%32 = OpAccessChain %_ptr_Function_18 %s0 %int_2
OpStore %32 %31
%33 = OpLoad %S_t %s0 
OpStore %param %33
%34 = OpAccessChain %_ptr_Function_18 %param %int_2
%35 = OpLoad %18 %34
%36 = OpAccessChain %_ptr_Function_v2float %param %int_0
%37 = OpLoad %v2float %36
%38 = OpImageSampleImplicitLod %v4float %35 %37
OpStore %outColor %38
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %12
%28 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%param = OpVariable %_ptr_Function_S_t Function
%29 = OpLoad %v2float %texCoords
%45 = OpLoad %S_t %s0
%46 = OpCompositeInsert %S_t %29 %45 0
OpStore %s0 %46
%31 = OpLoad %18 %sampler15
%47 = OpLoad %S_t %s0
%48 = OpCompositeInsert %S_t %31 %47 2
OpStore %s0 %48
%33 = OpLoad %S_t %s0
OpStore %param %33
%49 = OpLoad %S_t %param
%50 = OpCompositeExtract %18 %49 2
%51 = OpLoad %S_t %param
%52 = OpCompositeExtract %v2float %51 0
%38 = OpImageSampleImplicitLod %v4float %50 %52
OpStore %outColor %38
OpReturn
OpFunctionEnd
)";

  const std::string remain =
      R"(%foo_struct_S_t_vf2_vf21_ = OpFunction %void None %20
%s = OpFunctionParameter %_ptr_Function_S_t
%39 = OpLabel
%40 = OpAccessChain %_ptr_Function_18 %s %int_2
%41 = OpLoad %18 %40
%42 = OpAccessChain %_ptr_Function_v2float %s %int_0
%43 = OpLoad %v2float %42
%44 = OpImageSampleImplicitLod %v4float %41 %43
OpStore %outColor %44
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      predefs + before + remain, predefs + after + remain, true, true);
}

TEST_F(LocalAccessChainConvertTest, 
       NestedStructsConverted) {

  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  
  //  struct S1_t {
  //      vec4 v1;
  //  };
  //  
  //  struct S2_t {
  //      vec4 v2;
  //      S1_t s1;
  //  };
  //  
  //  void main()
  //  {
  //      S2_t s2;
  //      s2.s1.v1 = BaseColor;
  //      gl_FragColor = s2.s1.v1;
  //  }

  const std::string predefs_before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S1_t "S1_t"
OpMemberName %S1_t 0 "v1"
OpName %S2_t "S2_t"
OpMemberName %S2_t 0 "v2"
OpMemberName %S2_t 1 "s1"
OpName %s2 "s2"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S1_t = OpTypeStruct %v4float
%S2_t = OpTypeStruct %v4float %S1_t
%_ptr_Function_S2_t = OpTypePointer Function %S2_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string predefs_after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S1_t "S1_t"
OpMemberName %S1_t 0 "v1"
OpName %S2_t "S2_t"
OpMemberName %S2_t 0 "v2"
OpMemberName %S2_t 1 "s1"
OpName %s2 "s2"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S1_t = OpTypeStruct %v4float
%S2_t = OpTypeStruct %v4float %S1_t
%_ptr_Function_S2_t = OpTypePointer Function %S2_t
%int = OpTypeInt 32 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%s2 = OpVariable %_ptr_Function_S2_t Function
%20 = OpLoad %v4float %BaseColor
%21 = OpAccessChain %_ptr_Function_v4float %s2 %int_1 %int_0
OpStore %21 %20
%22 = OpAccessChain %_ptr_Function_v4float %s2 %int_1 %int_0
%23 = OpLoad %v4float %22
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%s2 = OpVariable %_ptr_Function_S2_t Function
%20 = OpLoad %v4float %BaseColor
%24 = OpLoad %S2_t %s2
%25 = OpCompositeInsert %S2_t %20 %24 1 0
OpStore %s2 %25
%26 = OpLoad %S2_t %s2
%27 = OpCompositeExtract %v4float %26 1 0
OpStore %gl_FragColor %27
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      predefs_before + before , predefs_after + after, true, true);
}

TEST_F(LocalAccessChainConvertTest, 
       DynamicallyIndexedVarNotConverted) {

  //  #version 140
  //  
  //  in vec4 BaseColor;
  //  flat in int Idx;
  //  in float Bi;
  //
  //  struct S_t {
  //      vec4 v0;
  //      vec4 v1;
  //  };
  //  
  //  void main()
  //  {
  //      S_t s0;
  //      s0.v1 = BaseColor;
  //      s0.v1[Idx] = Bi;
  //      gl_FragColor = s0.v1;
  //  }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %Idx %Bi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %S_t "S_t"
OpMemberName %S_t 0 "v0"
OpMemberName %S_t 1 "v1"
OpName %s0 "s0"
OpName %BaseColor "BaseColor"
OpName %Idx "Idx"
OpName %Bi "Bi"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %Idx Flat
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%S_t = OpTypeStruct %v4float %v4float
%_ptr_Function_S_t = OpTypePointer Function %S_t
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_int = OpTypePointer Input %int
%Idx = OpVariable %_ptr_Input_int Input
%_ptr_Input_float = OpTypePointer Input %float
%Bi = OpVariable %_ptr_Input_float Input
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %v4float %BaseColor
%24 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
OpStore %24 %23
%25 = OpLoad %int %Idx
%26 = OpLoad %float %Bi
%27 = OpAccessChain %_ptr_Function_float %s0 %int_1 %25
OpStore %27 %26
%28 = OpAccessChain %_ptr_Function_v4float %s0 %int_1
%29 = OpLoad %v4float %28
OpStore %gl_FragColor %29
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalAccessChainConvertPass>(
      assembly, assembly, false, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Assorted vector and matrix types
//    Assorted struct array types
//    Assorted scalar types
//    Assorted non-target types
//    OpInBoundsAccessChain
//    Others?

}  // anonymous namespace
