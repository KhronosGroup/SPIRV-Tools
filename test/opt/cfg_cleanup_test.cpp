// Copyright (c) 2017 Google Inc.
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

using CFGCleanupTest = PassTest<::testing::Test>;

TEST_F(CFGCleanupTest, RemoveUnreachableBlocks) {
  const std::string declarations = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %inf %outf4
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpName %main "main"
OpName %inf "inf"
OpName %outf4 "outf4"
OpDecorate %inf Location 0
OpDecorate %outf4 Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Input_float = OpTypePointer Input %float
%inf = OpVariable %_ptr_Input_float Input
%float_2 = OpConstant %float 2
%bool = OpTypeBool
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%outf4 = OpVariable %_ptr_Output_v4float Output
%float_n0_5 = OpConstant %float -0.5
)";

  const std::string body_before = R"(%main = OpFunction %void None %6
%14 = OpLabel
OpSelectionMerge %17 None
OpBranch %18
%19 = OpLabel
%20 = OpLoad %float %inf
%21 = OpCompositeConstruct %v4float %20 %20 %20 %20
OpStore %outf4 %21
OpBranch %17
%18 = OpLabel
%22 = OpLoad %float %inf
%23 = OpFAdd %float %22 %float_n0_5
%24 = OpCompositeConstruct %v4float %23 %23 %23 %23
OpStore %outf4 %24
OpBranch %17
%17 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string body_after = R"(%main = OpFunction %void None %6
%14 = OpLabel
OpSelectionMerge %15 None
OpBranch %16
%16 = OpLabel
%20 = OpLoad %float %inf
%21 = OpFAdd %float %20 %float_n0_5
%22 = OpCompositeConstruct %v4float %21 %21 %21 %21
OpStore %outf4 %22
OpBranch %15
%15 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CFGCleanupPass>(
      declarations + body_before, declarations + body_after, true, true);
}

TEST_F(CFGCleanupTest, RemoveDecorations) {
  const std::string before = R"(
                       OpCapability Shader
                  %1 = OpExtInstImport "GLSL.std.450"
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint Fragment %main "main"
                       OpName %main "main"
                       OpName %x "x"
                       OpName %dead "dead"
                       OpDecorate %x RelaxedPrecision
                       OpDecorate %dead RelaxedPrecision
               %void = OpTypeVoid
                  %6 = OpTypeFunction %void
              %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
            %float_2 = OpConstant %float 2
            %float_4 = OpConstant %float 4

               %main = OpFunction %void None %6
                 %14 = OpLabel
                  %x = OpVariable %_ptr_Function_float Function
                       OpSelectionMerge %17 None
                       OpBranch %18
                 %19 = OpLabel
               %dead = OpVariable %_ptr_Function_float Function
                       OpStore %dead %float_2
                       OpBranch %17
                 %18 = OpLabel
                       OpStore %x %float_4
                       OpBranch %17
                 %17 = OpLabel
                       OpReturn
                       OpFunctionEnd
)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpName %main "main"
OpName %x "x"
OpDecorate %x RelaxedPrecision
%void = OpTypeVoid
%6 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_2 = OpConstant %float 2
%float_4 = OpConstant %float 4
%main = OpFunction %void None %6
%11 = OpLabel
%x = OpVariable %_ptr_Function_float Function
OpSelectionMerge %12 None
OpBranch %13
%13 = OpLabel
OpStore %x %float_4
OpBranch %12
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CFGCleanupPass>(before, after, true, true);
}


TEST_F(CFGCleanupTest, UpdatePhis) {
  const std::string before = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %y %outparm
               OpName %main "main"
               OpName %y "y"
               OpName %outparm "outparm"
               OpDecorate %y Flat
               OpDecorate %y Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
          %y = OpVariable %_ptr_Input_int Input
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
     %int_42 = OpConstant %int 42
     %int_23 = OpConstant %int 23
      %int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
         %11 = OpLoad %int %y
               OpBranch %21
         %16 = OpLabel
         %20 = OpIAdd %int %11 %int_42
               OpBranch %17
         %21 = OpLabel
         %24 = OpISub %int %11 %int_23
               OpBranch %17
         %17 = OpLabel
         %31 = OpPhi %int %20 %16 %24 %21
         %27 = OpIAdd %int %31 %int_5
               OpStore %outparm %27
               OpReturn
               OpFunctionEnd
)";

  const std::string after = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %y %outparm
OpName %main "main"
OpName %y "y"
OpName %outparm "outparm"
OpDecorate %y Flat
OpDecorate %y Location 0
OpDecorate %outparm Location 0
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
%y = OpVariable %_ptr_Input_int Input
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%int_42 = OpConstant %int 42
%int_23 = OpConstant %int 23
%int_5 = OpConstant %int 5
%_ptr_Output_int = OpTypePointer Output %int
%outparm = OpVariable %_ptr_Output_int Output
%main = OpFunction %void None %6
%16 = OpLabel
%17 = OpLoad %int %y
OpBranch %18
%18 = OpLabel
%22 = OpISub %int %17 %int_23
OpBranch %21
%21 = OpLabel
%23 = OpPhi %int %22 %18
%24 = OpIAdd %int %23 %int_5
OpStore %outparm %24
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::CFGCleanupPass>(before, after, true, true);
}
}  // anonymous namespace
