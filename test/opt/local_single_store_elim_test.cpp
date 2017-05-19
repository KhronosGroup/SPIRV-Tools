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

using LocalSingleStoreElimTest = PassTest<::testing::Test>;


TEST_F(LocalSingleStoreElimTest, PositiveAndNegative) {
  // Single store to v is optimized. Multiple store to
  // f is not optimized.
  //
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float fi;
  // 
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     float f = fi;
  //     if (f < 0)
  //         f = 0.0;
  //     gl_FragColor = v + f;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %fi %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %f "f"
OpName %fi "fi"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%9 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f = OpVariable %_ptr_Function_float Function
%20 = OpLoad %v4float %BaseColor
OpStore %v %20
%21 = OpLoad %float %fi
OpStore %f %21
%22 = OpLoad %float %f
%23 = OpFOrdLessThan %bool %22 %float_0
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %24
%25 = OpLabel
OpStore %f %float_0
OpBranch %24
%24 = OpLabel
%26 = OpLoad %v4float %v
%27 = OpLoad %float %f
%28 = OpCompositeConstruct %v4float %27 %27 %27 %27
%29 = OpFAdd %v4float %26 %28
OpStore %gl_FragColor %29
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %9
%19 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%f = OpVariable %_ptr_Function_float Function
%20 = OpLoad %v4float %BaseColor
%21 = OpLoad %float %fi
OpStore %f %21
%22 = OpLoad %float %f
%23 = OpFOrdLessThan %bool %22 %float_0
OpSelectionMerge %24 None
OpBranchConditional %23 %25 %24
%25 = OpLabel
OpStore %f %float_0
OpBranch %24
%24 = OpLabel
%27 = OpLoad %float %f
%28 = OpCompositeConstruct %v4float %27 %27 %27 %27
%29 = OpFAdd %v4float %20 %28
OpStore %gl_FragColor %29
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleStoreElimPass>(predefs + before, 
      predefs + after, true, true);
}

TEST_F(LocalSingleStoreElimTest, NoOptIfStoreNotDominating) {
  // Single store to f not optimized because it does not dominate
  // the load.
  // 
  // #version 140
  // 
  // in vec4 BaseColor;
  // in float fi;
  // 
  // void main()
  // {
  //     float f;
  //     if (fi < 0)
  //         f = 0.5;
  //     if (fi < 0)
  //         gl_FragColor = BaseColor * f;
  //     else
  //         gl_FragColor = BaseColor;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %fi %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %fi "fi"
OpName %f "f"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
%fi = OpVariable %_ptr_Input_float Input
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Function_float = OpTypePointer Function %float
%float_0_5 = OpConstant %float 0.5
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %8
%18 = OpLabel
%f = OpVariable %_ptr_Function_float Function
%19 = OpLoad %float %fi
%20 = OpFOrdLessThan %bool %19 %float_0
OpSelectionMerge %21 None
OpBranchConditional %20 %22 %21
%22 = OpLabel
OpStore %f %float_0_5
OpBranch %21
%21 = OpLabel
%23 = OpLoad %float %fi
%24 = OpFOrdLessThan %bool %23 %float_0
OpSelectionMerge %25 None
OpBranchConditional %24 %26 %27
%26 = OpLabel
%28 = OpLoad %v4float %BaseColor
%29 = OpLoad %float %f
%30 = OpVectorTimesScalar %v4float %28 %29
OpStore %gl_FragColor %30
OpBranch %25
%27 = OpLabel
%31 = OpLoad %v4float %BaseColor
OpStore %gl_FragColor %31
OpBranch %25
%25 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LocalSingleStoreElimPass>(assembly, assembly,
      true, true);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Other types
//    Multiple loads optimized
//    Non-dominating store not optimized
//    Others?

}  // anonymous namespace
