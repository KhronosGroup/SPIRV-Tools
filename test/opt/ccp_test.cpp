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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pass_fixture.h"
#include "pass_utils.h"

#include "opt/ccp_pass.h"

namespace {

using namespace spvtools;

using CCPTest = PassTest<::testing::Test>;

// TODO(dneto): Add Effcee as required dependency, and make this unconditional.
#ifdef SPIRV_EFFCEE
TEST_F(CCPTest, PropagateThroughPhis) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %x %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %outparm "outparm"
               OpDecorate %x Flat
               OpDecorate %x Location 0
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Input_int = OpTypePointer Input %int
          %x = OpVariable %_ptr_Input_int Input
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %4 = OpLabel
          %5 = OpLoad %int %x
          %9 = OpIAdd %int %int_1 %int_3
          %6 = OpSGreaterThan %bool %5 %int_3
               OpSelectionMerge %25 None
               OpBranchConditional %6 %22 %23
         %22 = OpLabel

; CHECK: OpCopyObject %int %int_4
          %7 = OpCopyObject %int %9

               OpBranch %25
         %23 = OpLabel
          %8 = OpCopyObject %int %int_4
               OpBranch %25
         %25 = OpLabel

; %int_4 should have propagated to both OpPhi operands.
; CHECK: OpPhi %int %int_4 {{%\d+}} %int_4 {{%\d+}}
         %35 = OpPhi %int %7 %22 %8 %23

; This function always returns 4. DCE should get rid of everything else.
; CHECK OpStore %outparm %int_4
               OpStore %outparm %35
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifyConditionals) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %4 = OpLabel
          %9 = OpIAdd %int %int_4 %int_3
          %6 = OpSGreaterThan %bool %9 %int_3
               OpSelectionMerge %25 None
; CHECK: OpBranchConditional %true [[bb_taken:%\d+]] [[bb_not_taken:%\d+]]
               OpBranchConditional %6 %22 %23
; CHECK: [[bb_taken]] = OpLabel
         %22 = OpLabel
; CHECK: OpCopyObject %int %int_7
          %7 = OpCopyObject %int %9
               OpBranch %25
; CHECK: [[bb_not_taken]] = OpLabel
         %23 = OpLabel
; CHECK: [[id_not_evaluated:%\d+]] = OpCopyObject %int %int_4
          %8 = OpCopyObject %int %int_4
               OpBranch %25
         %25 = OpLabel

; %int_7 should have propagated to the first OpPhi operand. But the else branch
; is not executable (conditional is always true), so no values should be
; propagated there and the value of the OpPhi should always be %int_7.
; CHECK: OpPhi %int %int_7 [[bb_taken]] [[id_not_evaluated]] [[bb_not_taken]]
         %35 = OpPhi %int %7 %22 %8 %23

; Only the true path of the conditional is ever executed. The output of this
; function is always %int_7.
; CHECK: OpStore %outparm %int_7
               OpStore %outparm %35
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifySwitches) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_23 = OpConstant %int 23
     %int_42 = OpConstant %int 42
     %int_14 = OpConstant %int 14
     %int_15 = OpConstant %int 15
      %int_4 = OpConstant %int 4
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %6
         %15 = OpLabel
               OpSelectionMerge %17 None
               OpSwitch %int_23 %17 10 %18 13 %19 23 %20
         %18 = OpLabel
               OpBranch %17
         %19 = OpLabel
               OpBranch %17
         %20 = OpLabel
               OpBranch %17
         %17 = OpLabel
         %24 = OpPhi %int %int_23 %15 %int_42 %18 %int_14 %19 %int_15 %20

; The switch will always jump to label %20, which carries the value %int_15.
; CHECK: OpIAdd %int %int_15 %int_4
         %22 = OpIAdd %int %24 %int_4

; Consequently, the return value will always be %int_19.
; CHECK: OpStore %outparm %int_19
               OpStore %outparm %22
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifySwitchesDefaultBranch) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %6 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_42 = OpConstant %int 42
      %int_4 = OpConstant %int 4
      %int_1 = OpConstant %int 1
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %6
         %13 = OpLabel
         %15 = OpIAdd %int %int_42 %int_4
               OpSelectionMerge %16 None

; CHECK: OpSwitch %int_46 {{%\d+}} 10 {{%\d+}}
               OpSwitch %15 %17 10 %18
         %18 = OpLabel
               OpBranch %16
         %17 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %22 = OpPhi %int %int_42 %18 %int_1 %17

; The switch will always jump to the default label %17.  This carries the value
; %int_1.
; CHECK: OpIAdd %int %int_1 %int_4
         %20 = OpIAdd %int %22 %int_4

; Resulting in a return value of %int_5.
; CHECK: OpStore %outparm %int_5
               OpStore %outparm %20
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, SimplifyIntVector) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %v "v"
               OpName %OutColor "OutColor"
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %v4int = OpTypeVector %int 4
%_ptr_Function_v4int = OpTypePointer Function %v4int
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
      %int_4 = OpConstant %int 4
         %14 = OpConstantComposite %v4int %int_1 %int_2 %int_3 %int_4
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Output_v4int = OpTypePointer Output %v4int
   %OutColor = OpVariable %_ptr_Output_v4int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4int Function
               OpStore %v %14
         %18 = OpAccessChain %_ptr_Function_int %v %uint_0
         %19 = OpLoad %int %18

; The constant folder does not see through access chains. To get this, the
; vector would have to be scalarized.
; CHECK: [[result_id:%\d+]] = OpIAdd %int {{%\d+}} %int_1
         %20 = OpIAdd %int %19 %int_1
         %21 = OpAccessChain %_ptr_Function_int %v %uint_0

; CHECK: OpStore {{%\d+}} [[result_id]]
               OpStore %21 %20
         %24 = OpLoad %v4int %v
               OpStore %OutColor %24
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, BadSimplifyFloatVector) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %OutColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %v "v"
               OpName %OutColor "OutColor"
               OpDecorate %OutColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
    %float_3 = OpConstant %float 3
    %float_4 = OpConstant %float 4
         %14 = OpConstantComposite %v4float %float_1 %float_2 %float_3 %float_4
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %OutColor = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %v = OpVariable %_ptr_Function_v4float Function
               OpStore %v %14
         %18 = OpAccessChain %_ptr_Function_float %v %uint_0
         %19 = OpLoad %float %18

; NOTE: This test should start failing once floating point folding is
;       implemented (https://github.com/KhronosGroup/SPIRV-Tools/issues/943).
;       This should be checking that we are adding %float_1 + %float_1.
; CHECK: [[result_id:%\d+]] = OpFAdd %float {{%\d+}} %float_1
         %20 = OpFAdd %float %19 %float_1
         %21 = OpAccessChain %_ptr_Function_float %v %uint_0

; This should be checkint that we are storing %float_2 instead of result_it.
; CHECK: OpStore {{%\d+}} [[result_id]]
               OpStore %21 %20
         %24 = OpLoad %v4float %v
               OpStore %OutColor %24
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, NoLoadStorePropagation) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outparm
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %x "x"
               OpName %outparm "outparm"
               OpDecorate %outparm Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
     %int_23 = OpConstant %int 23
%_ptr_Output_int = OpTypePointer Output %int
    %outparm = OpVariable %_ptr_Output_int Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %x = OpVariable %_ptr_Function_int Function
               OpStore %x %int_23

; int_23 should not propagate into this load.
; CHECK: OpLoad %int %x
         %12 = OpLoad %int %x

; Likewise here.
; CHECK: OpStore %outparm {{%\d+}}
               OpStore %outparm %12
               OpReturn
               OpFunctionEnd
               )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

TEST_F(CCPTest, HandleAbortInstructions) {
  const std::string spv_asm = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource HLSL 500
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
; CHECK: %true = OpConstantTrue %bool
      %int_3 = OpConstant %int 3
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %4 = OpLabel
          %9 = OpIAdd %int %int_3 %int_1
          %6 = OpSGreaterThan %bool %9 %int_3
               OpSelectionMerge %23 None
; CHECK: OpBranchConditional %true {{%\d+}} {{%\d+}}
               OpBranchConditional %6 %22 %23
         %22 = OpLabel
               OpKill
         %23 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<opt::CCPPass>(spv_asm, true);
}

#endif

}  // namespace
