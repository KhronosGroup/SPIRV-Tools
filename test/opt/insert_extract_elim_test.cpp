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

using InsertExtractElimTest = PassTest<::testing::Test>;

TEST_F(InsertExtractElimTest, Simple) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
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

  const std::string before =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
%21 = OpCompositeExtract %v4float %20 1
OpStore %gl_FragColor %21
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%17 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%18 = OpLoad %v4float %BaseColor
%19 = OpLoad %S_t %s0
%20 = OpCompositeInsert %S_t %18 %19 1
OpStore %s0 %20
OpStore %gl_FragColor %18
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, OptimizeAcrossNonConflictingInsert) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v0[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string predefs =
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
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 0 2
OpStore %s0 %22
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, OptimizeOpaque) {
  // SPIR-V not representable in GLSL; not generatable from HLSL
  // for the moment.

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

  const std::string before =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function 
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0 
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%31 = OpCompositeExtract %15 %30 2
%32 = OpCompositeExtract %v2float %30 0
%33 = OpImageSampleImplicitLod %v4float %31 %32
OpStore %outColor %33
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%25 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%26 = OpLoad %v2float %texCoords
%27 = OpLoad %S_t %s0
%28 = OpCompositeInsert %S_t %26 %27 0
%29 = OpLoad %15 %sampler15
%30 = OpCompositeInsert %S_t %29 %28 2
OpStore %s0 %30
%33 = OpImageSampleImplicitLod %v4float %29 %26
OpStore %outColor %33
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, OptimizeNestedStruct) {
  // The following HLSL has been pre-optimized to get the SPIR-V:
  // struct S0
  // {
  //     int x;
  //     SamplerState ss;
  // };
  //
  // struct S1
  // {
  //     float b;
  //     S0 s0;
  // };
  //
  // struct S2
  // {
  //     int a1;
  //     S1 resources;
  // };
  //
  // SamplerState samp;
  // Texture2D tex;
  //
  // float4 main(float4 vpos : VPOS) : COLOR0
  // {
  //     S1 s1;
  //     S2 s2;
  //     s1.s0.ss = samp;
  //     s2.resources = s1;
  //     return tex.Sample(s2.resources.s0.ss, float2(0.5));
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %_entryPointOutput
OpExecutionMode %main OriginUpperLeft
OpSource HLSL 500
OpName %main "main"
OpName %S0 "S0"
OpMemberName %S0 0 "x"
OpMemberName %S0 1 "ss"
OpName %S1 "S1"
OpMemberName %S1 0 "b"
OpMemberName %S1 1 "s0"
OpName %samp "samp"
OpName %S2 "S2"
OpMemberName %S2 0 "a1"
OpMemberName %S2 1 "resources"
OpName %tex "tex"
OpName %_entryPointOutput "@entryPointOutput"
OpDecorate %samp DescriptorSet 0
OpDecorate %tex DescriptorSet 0
OpDecorate %_entryPointOutput Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %v4float %_ptr_Function_v4float
%int = OpTypeInt 32 1
%16 = OpTypeSampler
%S0 = OpTypeStruct %int %16
%S1 = OpTypeStruct %float %S0
%_ptr_Function_S1 = OpTypePointer Function %S1
%int_1 = OpConstant %int 1
%_ptr_UniformConstant_16 = OpTypePointer UniformConstant %16
%samp = OpVariable %_ptr_UniformConstant_16 UniformConstant
%_ptr_Function_16 = OpTypePointer Function %16
%S2 = OpTypeStruct %int %S1
%_ptr_Function_S2 = OpTypePointer Function %S2
%22 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_22 = OpTypePointer UniformConstant %22
%tex = OpVariable %_ptr_UniformConstant_22 UniformConstant
%24 = OpTypeSampledImage %22
%v2float = OpTypeVector %float 2
%float_0_5 = OpConstant %float 0.5
%27 = OpConstantComposite %v2float %float_0_5 %float_0_5
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%30 = OpLabel
%31 = OpVariable %_ptr_Function_S1 Function
%32 = OpVariable %_ptr_Function_S2 Function
%33 = OpLoad %16 %samp
%34 = OpLoad %S1 %31
%35 = OpCompositeInsert %S1 %33 %34 1 1
OpStore %31 %35
%36 = OpLoad %S2 %32
%37 = OpCompositeInsert %S2 %35 %36 1
OpStore %32 %37
%38 = OpLoad %22 %tex
%39 = OpCompositeExtract %16 %37 1 1 1
%40 = OpSampledImage %24 %38 %39
%41 = OpImageSampleImplicitLod %v4float %40 %27
OpStore %_entryPointOutput %41
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%30 = OpLabel
%31 = OpVariable %_ptr_Function_S1 Function
%32 = OpVariable %_ptr_Function_S2 Function
%33 = OpLoad %16 %samp
%34 = OpLoad %S1 %31
%35 = OpCompositeInsert %S1 %33 %34 1 1
OpStore %31 %35
%36 = OpLoad %S2 %32
%37 = OpCompositeInsert %S2 %35 %36 1
OpStore %32 %37
%38 = OpLoad %22 %tex
%40 = OpSampledImage %24 %38 %33
%41 = OpImageSampleImplicitLod %v4float %40 %27
OpStore %_entryPointOutput %41
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1 = BaseColor;
  //     s0.v1[2] = 0.0;
  //     gl_FragColor = s0.v1;
  // }

  const std::string assembly =
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
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %8
%18 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%19 = OpLoad %v4float %BaseColor
%20 = OpLoad %S_t %s0
%21 = OpCompositeInsert %S_t %19 %20 1
%22 = OpCompositeInsert %S_t %float_0 %21 1 2
OpStore %s0 %22
%23 = OpCompositeExtract %v4float %22 1
OpStore %gl_FragColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(assembly, assembly, true,
                                                    true);
}

TEST_F(InsertExtractElimTest, ConflictingInsertPreventsOptimization2) {
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // struct S_t {
  //     vec4 v0;
  //     vec4 v1;
  // };
  //
  // void main()
  // {
  //     S_t s0;
  //     s0.v1[1] = 1.0; // dead
  //     s0.v1 = Baseline;
  //     gl_FragColor = vec4(s0.v1[1], 0.0, 0.0, 0.0);
  // }

  const std::string before_predefs =
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
%float_1 = OpConstant %float 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string after_predefs =
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
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string before =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %S_t %s0
%24 = OpCompositeInsert %S_t %float_1 %23 1 1
%25 = OpLoad %v4float %BaseColor
%26 = OpCompositeInsert %S_t %25 %24 1
%27 = OpCompositeExtract %float %26 1 1
%28 = OpCompositeConstruct %v4float %27 %float_0 %float_0 %float_0
OpStore %gl_FragColor %28
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %8
%22 = OpLabel
%s0 = OpVariable %_ptr_Function_S_t Function
%23 = OpLoad %S_t %s0
%25 = OpLoad %v4float %BaseColor
%26 = OpCompositeInsert %S_t %25 %23 1
%27 = OpCompositeExtract %float %26 1 1
%28 = OpCompositeConstruct %v4float %27 %float_0 %float_0 %float_0
OpStore %gl_FragColor %28
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      before_predefs + before, after_predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, MixWithConstants) {
  // Extract component of FMix with 0.0 or 1.0 as the a-value.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=2) in float m;
  // layout (location=3) in float m2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 bcv2 = vec4(bc2, bc, 1.0, 0.0);
  //     vec4 v = mix(bcv, bcv2, vec4(0.0,1.0,m,m2));
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %m %m2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %m "m"
OpName %m2 "m2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %m Location 2
OpDecorate %m2 Location 3
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%m = OpVariable %_ptr_Input_float Input
%m2 = OpVariable %_ptr_Input_float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%20 = OpLoad %float %bc
%21 = OpLoad %float %bc2
%22 = OpCompositeConstruct %v4float %20 %21 %float_0 %float_1
%23 = OpLoad %float %bc2
%24 = OpLoad %float %bc
%25 = OpCompositeConstruct %v4float %23 %24 %float_1 %float_0
%26 = OpLoad %float %m
%27 = OpLoad %float %m2
%28 = OpCompositeConstruct %v4float %float_0 %float_1 %26 %27
%29 = OpExtInst %v4float %1 FMix %22 %25 %28
%30 = OpCompositeExtract %float %29 1
%31 = OpCompositeConstruct %v4float %30 %30 %30 %30
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%20 = OpLoad %float %bc
%21 = OpLoad %float %bc2
%22 = OpCompositeConstruct %v4float %20 %21 %float_0 %float_1
%23 = OpLoad %float %bc2
%24 = OpLoad %float %bc
%25 = OpCompositeConstruct %v4float %23 %24 %float_1 %float_0
%26 = OpLoad %float %m
%27 = OpLoad %float %m2
%28 = OpCompositeConstruct %v4float %float_0 %float_1 %26 %27
%29 = OpExtInst %v4float %1 FMix %22 %25 %28
%31 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %OutColor %31
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, VectorShuffle1) {
  // Extract component from first vector in VectorShuffle
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 v = bcv.zwxy;
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 3 0 1
%22 = OpCompositeExtract %float %21 1
%23 = OpCompositeConstruct %v4float %22 %22 %22 %22
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 3 0 1
%23 = OpCompositeConstruct %v4float %float_1 %float_1 %float_1 %float_1
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, VectorShuffle2) {
  // Extract component from second vector in VectorShuffle
  // Identical to test VectorShuffle1 except for the vector
  // shuffle index of 7.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float bc;
  // layout (location=1) in float bc2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec4 bcv = vec4(bc, bc2, 0.0, 1.0);
  //     vec4 v = bcv.zwxy;
  //     OutColor = vec4(v.y);
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %bc %bc2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %bc "bc"
OpName %bc2 "bc2"
OpName %OutColor "OutColor"
OpDecorate %bc Location 0
OpDecorate %bc2 Location 1
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_float = OpTypePointer Input %float
%bc = OpVariable %_ptr_Input_float Input
%bc2 = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 7 0 1
%22 = OpCompositeExtract %float %21 1
%23 = OpCompositeConstruct %v4float %22 %22 %22 %22
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%17 = OpLabel
%18 = OpLoad %float %bc
%19 = OpLoad %float %bc2
%20 = OpCompositeConstruct %v4float %18 %19 %float_0 %float_1
%21 = OpVectorShuffle %v4float %20 %20 2 7 0 1
%23 = OpCompositeConstruct %v4float %float_1 %float_1 %float_1 %float_1
OpStore %OutColor %23
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      predefs + before, predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, InsertAfterInsertElim) {
  // With two insertions to the same offset, the first is dead.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in float In0;
  // layout (location=1) in float In1;
  // layout (location=2) in vec2 In2;
  // layout (location=0) out vec4 OutColor;
  //
  // void main()
  // {
  //     vec2 v = In2;
  //     v.x = In0 + In1; // dead
  //     v.x = 0.0;
  //     OutColor = v.xyxy;
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In2 %In0 %In1 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In2 "In2"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %OutColor "OutColor"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_n"
OpName %_ ""
OpDecorate %In2 Location 2
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %OutColor Location 0
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%In2 = OpVariable %_ptr_Input_v2float Input
%_ptr_Input_float = OpTypePointer Input %float
%In0 = OpVariable %_ptr_Input_float Input
%In1 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%int = OpTypeInt 32 1
%_Globals_ = OpTypeStruct %uint %int
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In2 %In0 %In1 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In2 "In2"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %OutColor "OutColor"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_n"
OpName %_ ""
OpDecorate %In2 Location 2
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %OutColor Location 0
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Input_v2float = OpTypePointer Input %v2float
%In2 = OpVariable %_ptr_Input_v2float Input
%_ptr_Input_float = OpTypePointer Input %float
%In0 = OpVariable %_ptr_Input_float Input
%In1 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%int = OpTypeInt 32 1
%_Globals_ = OpTypeStruct %uint %int
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%25 = OpLabel
%26 = OpLoad %v2float %In2
%27 = OpLoad %float %In0
%28 = OpLoad %float %In1
%29 = OpFAdd %float %27 %28
%35 = OpCompositeInsert %v2float %29 %26 0
%37 = OpCompositeInsert %v2float %float_0 %35 0
%33 = OpVectorShuffle %v4float %37 %37 0 1 0 1
OpStore %OutColor %33
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%23 = OpLabel
%24 = OpLoad %v2float %In2
%29 = OpCompositeInsert %v2float %float_0 %24 0
%30 = OpVectorShuffle %v4float %29 %29 0 1 0 1
OpStore %OutColor %30
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      before_predefs + before, after_predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, DeadInsertInChainWithPhi) {
  // Dead insert eliminated with phi in insertion chain.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in vec4 In0;
  // layout (location=1) in float In1;
  // layout (location=2) in float In2;
  // layout (location=0) out vec4 OutColor;
  //
  // layout(std140, binding = 0 ) uniform _Globals_
  // {
  //     bool g_b;
  // };
  //
  // void main()
  // {
  //     vec4 v = In0;
  //     v.z = In1 + In2;
  //     if (g_b) v.w = 1.0;
  //     OutColor = vec4(v.x,v.y,0.0,v.w);
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%_Globals_ = OpTypeStruct %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_ptr_Function_float = OpTypePointer Function %float
%_Globals_ = OpTypeStruct %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
)";

  const std::string before =
      R"(%main = OpFunction %void None %11
%31 = OpLabel
%32 = OpLoad %v4float %In0
%33 = OpLoad %float %In1
%34 = OpLoad %float %In2
%35 = OpFAdd %float %33 %34
%51 = OpCompositeInsert %v4float %35 %32 2
%37 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%38 = OpLoad %uint %37
%39 = OpINotEqual %bool %38 %uint_0
OpSelectionMerge %40 None
OpBranchConditional %39 %41 %40
%41 = OpLabel
%53 = OpCompositeInsert %v4float %float_1 %51 3
OpBranch %40
%40 = OpLabel
%60 = OpPhi %v4float %51 %31 %53 %41
%55 = OpCompositeExtract %float %60 0
%57 = OpCompositeExtract %float %60 1
%59 = OpCompositeExtract %float %60 3
%49 = OpCompositeConstruct %v4float %55 %57 %float_0 %59
OpStore %OutColor %49
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%27 = OpLabel
%28 = OpLoad %v4float %In0
%33 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%34 = OpLoad %uint %33
%35 = OpINotEqual %bool %34 %uint_0
OpSelectionMerge %36 None
OpBranchConditional %35 %37 %36
%37 = OpLabel
%38 = OpCompositeInsert %v4float %float_1 %28 3
OpBranch %36
%36 = OpLabel
%39 = OpPhi %v4float %28 %27 %38 %37
%40 = OpCompositeExtract %float %39 0
%41 = OpCompositeExtract %float %39 1
%42 = OpCompositeExtract %float %39 3
%43 = OpCompositeConstruct %v4float %40 %41 %float_0 %42
OpStore %OutColor %43
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      before_predefs + before, after_predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, DeadInsertTwoPasses) {
  // Dead insert which requires two passes to eliminate
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in vec4 In0;
  // layout (location=1) in float In1;
  // layout (location=2) in float In2;
  // layout (location=0) out vec4 OutColor;
  //
  // layout(std140, binding = 0 ) uniform _Globals_
  // {
  //     bool g_b;
  //     bool g_b2;
  // };
  //
  // void main()
  // {
  //     vec4 v1, v2;
  //     v1 = In0;
  //     v1.y = In1 + In2; // dead, second pass
  //     if (g_b) v1.x = 1.0;
  //     v2.x = v1.x;
  //     v2.y = v1.y; // dead, first pass
  //     if (g_b2) v2.x = 0.0;
  //     OutColor = vec4(v2.x,v2.x,0.0,1.0);
  // }

  const std::string before_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_b2"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_Globals_ = OpTypeStruct %uint %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%27 = OpUndef %v4float
)";

  const std::string after_predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %In0 %In1 %In2 %OutColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_b"
OpMemberName %_Globals_ 1 "g_b2"
OpName %_ ""
OpName %OutColor "OutColor"
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
OpMemberDecorate %_Globals_ 0 Offset 0
OpMemberDecorate %_Globals_ 1 Offset 4
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%uint = OpTypeInt 32 0
%_Globals_ = OpTypeStruct %uint %uint
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%27 = OpUndef %v4float
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%28 = OpLabel
%29 = OpLoad %v4float %In0
%30 = OpLoad %float %In1
%31 = OpLoad %float %In2
%32 = OpFAdd %float %30 %31
%33 = OpCompositeInsert %v4float %32 %29 1
%34 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%35 = OpLoad %uint %34
%36 = OpINotEqual %bool %35 %uint_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
%39 = OpCompositeInsert %v4float %float_1 %33 0
OpBranch %37
%37 = OpLabel
%40 = OpPhi %v4float %33 %28 %39 %38
%41 = OpCompositeExtract %float %40 0
%42 = OpCompositeInsert %v4float %41 %27 0
%43 = OpCompositeExtract %float %40 1
%44 = OpCompositeInsert %v4float %43 %42 1
%45 = OpAccessChain %_ptr_Uniform_uint %_ %int_1
%46 = OpLoad %uint %45
%47 = OpINotEqual %bool %46 %uint_0
OpSelectionMerge %48 None
OpBranchConditional %47 %49 %48
%49 = OpLabel
%50 = OpCompositeInsert %v4float %float_0 %44 0
OpBranch %48
%48 = OpLabel
%51 = OpPhi %v4float %44 %37 %50 %49
%52 = OpCompositeExtract %float %51 0
%53 = OpCompositeExtract %float %51 0
%54 = OpCompositeConstruct %v4float %52 %53 %float_0 %float_1
OpStore %OutColor %54
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%28 = OpLabel
%29 = OpLoad %v4float %In0
%34 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%35 = OpLoad %uint %34
%36 = OpINotEqual %bool %35 %uint_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
%39 = OpCompositeInsert %v4float %float_1 %29 0
OpBranch %37
%37 = OpLabel
%40 = OpPhi %v4float %29 %28 %39 %38
%41 = OpCompositeExtract %float %40 0
%42 = OpCompositeInsert %v4float %41 %27 0
%45 = OpAccessChain %_ptr_Uniform_uint %_ %int_1
%46 = OpLoad %uint %45
%47 = OpINotEqual %bool %46 %uint_0
OpSelectionMerge %48 None
OpBranchConditional %47 %49 %48
%49 = OpLabel
%50 = OpCompositeInsert %v4float %float_0 %42 0
OpBranch %48
%48 = OpLabel
%51 = OpPhi %v4float %42 %37 %50 %49
%52 = OpCompositeExtract %float %51 0
%53 = OpCompositeExtract %float %51 0
%54 = OpCompositeConstruct %v4float %52 %53 %float_0 %float_1
OpStore %OutColor %54
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(
      before_predefs + before, after_predefs + after, true, true);
}

TEST_F(InsertExtractElimTest, DeadInsertInCycleToDo) {
  // Dead insert in chain with cycle. Demonstrates analysis can handle
  // cycles in chains.
  //
  // TODO(greg-lunarg): Improve algorithm to remove dead insert into v.y. Will
  // likely require similar logic to ADCE.
  //
  // Note: The SPIR-V assembly has had store/load elimination
  // performed to allow the inserts and extracts to directly
  // reference each other.
  //
  // #version 450
  //
  // layout (location=0) in vec4 In0;
  // layout (location=1) in float In1;
  // layout (location=2) in float In2;
  // layout (location=0) out vec4 OutColor;
  //
  // layout(std140, binding = 0 ) uniform _Globals_
  // {
  //     int g_n  ;
  // };
  //
  // void main()
  // {
  //     vec2 v = vec2(0.0, 1.0);
  //     for (int i = 0; i < g_n; i++) {
  //       v.x = v.x + 1;
  //       v.y = v.y * 0.9; // dead
  //     }
  //     OutColor = vec4(v.x);
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %OutColor %In0 %In1 %In2
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %_Globals_ "_Globals_"
OpMemberName %_Globals_ 0 "g_n"
OpName %_ ""
OpName %OutColor "OutColor"
OpName %In0 "In0"
OpName %In1 "In1"
OpName %In2 "In2"
OpMemberDecorate %_Globals_ 0 Offset 0
OpDecorate %_Globals_ Block
OpDecorate %_ DescriptorSet 0
OpDecorate %_ Binding 0
OpDecorate %OutColor Location 0
OpDecorate %In0 Location 0
OpDecorate %In1 Location 1
OpDecorate %In2 Location 2
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%16 = OpConstantComposite %v2float %float_0 %float_1
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%_Globals_ = OpTypeStruct %int
%_ptr_Uniform__Globals_ = OpTypePointer Uniform %_Globals_
%_ = OpVariable %_ptr_Uniform__Globals_ Uniform
%_ptr_Uniform_int = OpTypePointer Uniform %int
%bool = OpTypeBool
%float_0_9 = OpConstant %float 0.9
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%OutColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%In0 = OpVariable %_ptr_Input_v4float Input
%_ptr_Input_float = OpTypePointer Input %float
%In1 = OpVariable %_ptr_Input_float Input
%In2 = OpVariable %_ptr_Input_float Input
%main = OpFunction %void None %10
%29 = OpLabel
OpBranch %30
%30 = OpLabel
%31 = OpPhi %v2float %16 %29 %32 %33
%34 = OpPhi %int %int_0 %29 %35 %33
OpLoopMerge %36 %33 None
OpBranch %37
%37 = OpLabel
%38 = OpAccessChain %_ptr_Uniform_int %_ %int_0
%39 = OpLoad %int %38
%40 = OpSLessThan %bool %34 %39
OpBranchConditional %40 %41 %36
%41 = OpLabel
%42 = OpCompositeExtract %float %31 0
%43 = OpFAdd %float %42 %float_1
%44 = OpCompositeInsert %v2float %43 %31 0
%45 = OpCompositeExtract %float %44 1
%46 = OpFMul %float %45 %float_0_9
%32 = OpCompositeInsert %v2float %46 %44 1
OpBranch %33
%33 = OpLabel
%35 = OpIAdd %int %34 %int_1
OpBranch %30
%36 = OpLabel
%47 = OpCompositeExtract %float %31 0
%48 = OpCompositeConstruct %v4float %47 %47 %47 %47
OpStore %OutColor %48
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::InsertExtractElimPass>(assembly, assembly, true,
                                                    true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//

}  // anonymous namespace
