// Copyright (c) 2018 Google LLC.
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

#include <string>

#include "gmock/gmock.h"
#include "test/opt/pass_fixture.h"

#include "effcee/effcee.h"

namespace spvtools {
namespace opt {
namespace {

using UnswitchTest = PassTest<::testing::Test>;

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 450 core
uniform vec4 c;
void main() {
  int i = 0;
  int j = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++, j++) {
    if (cond) {
      i++;
    }
    else {
      j++;
    }
  }
}
*/
TEST_F(UnswitchTest, SimpleUnswitch) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpFOrdEqual
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false.
; CHECK: [[loop_f]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: [[phi_j:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_j:%\w+]] [[continue]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=1 and j+=2.
; CHECK: [[phi_j:%\w+]] = OpIAdd %int [[phi_j]] %int_1
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_j]] = OpIAdd %int [[phi_j]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: [[phi_j:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_j:%\w+]] [[continue]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=2 and j+=1.
; CHECK: [[phi_i:%\w+]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_j]] = OpIAdd %int [[phi_j]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginLowerLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %c DescriptorSet 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_UniformConstant_v4float = OpTypePointer UniformConstant %v4float
          %c = OpVariable %_ptr_UniformConstant_v4float UniformConstant
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_UniformConstant_float = OpTypePointer UniformConstant %float
    %float_0 = OpConstant %float 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %21 = OpAccessChain %_ptr_UniformConstant_float %c %uint_0
         %22 = OpLoad %float %21
         %24 = OpFOrdEqual %bool %22 %float_0
               OpBranch %25
         %25 = OpLabel
         %46 = OpPhi %int %int_0 %5 %43 %28
         %47 = OpPhi %int %int_0 %5 %45 %28
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
         %32 = OpSLessThan %bool %46 %int_10
               OpBranchConditional %32 %26 %27
         %26 = OpLabel
               OpSelectionMerge %35 None
               OpBranchConditional %24 %34 %39
         %34 = OpLabel
         %38 = OpIAdd %int %46 %int_1
               OpBranch %35
         %39 = OpLabel
         %41 = OpIAdd %int %47 %int_1
               OpBranch %35
         %35 = OpLabel
         %48 = OpPhi %int %38 %34 %46 %39
         %49 = OpPhi %int %47 %34 %41 %39
               OpBranch %28
         %28 = OpLabel
         %43 = OpIAdd %int %48 %int_1
         %45 = OpIAdd %int %49 %int_1
               OpBranch %25
         %27 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  int i = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++) {
    if (cond) {
      i++;
    }
    else {
      return;
    }
  }
}
*/
TEST_F(UnswitchTest, UnswitchExit) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpFOrdEqual
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false.
; CHECK: [[loop_f]] = OpLabel
; CHECK: OpReturn

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=2.
; CHECK: [[phi_i:%\w+]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %23 Uniform
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %20 = OpAccessChain %_ptr_Input_float %c %uint_0
         %21 = OpLoad %float %20
         %23 = OpFOrdEqual %bool %21 %float_0
               OpBranch %24
         %24 = OpLabel
         %42 = OpPhi %int %int_0 %5 %41 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %31 = OpSLessThan %bool %42 %int_10
               OpBranchConditional %31 %25 %26
         %25 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %23 %33 %38
         %33 = OpLabel
         %37 = OpIAdd %int %42 %int_1
               OpBranch %34
         %38 = OpLabel
               OpReturn
         %34 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %41 = OpIAdd %int %37 %int_1
               OpBranch %24
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  int i = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++) {
    if (cond) {
      continue;
    }
    else {
      i++;
    }
  }
}
*/
TEST_F(UnswitchTest, UnswitchContinue) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpFOrdEqual
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false.
; CHECK: [[loop_f]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=2.
; CHECK: [[phi_i:%\w+]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=1.
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %23 Uniform
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %20 = OpAccessChain %_ptr_Input_float %c %uint_0
         %21 = OpLoad %float %20
         %23 = OpFOrdEqual %bool %21 %float_0
               OpBranch %24
         %24 = OpLabel
         %42 = OpPhi %int %int_0 %5 %41 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %31 = OpSLessThan %bool %42 %int_10
               OpBranchConditional %31 %25 %26
         %25 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %23 %33 %36
         %33 = OpLabel
               OpBranch %27
         %36 = OpLabel
         %39 = OpIAdd %int %42 %int_1
               OpBranch %34
         %34 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %43 = OpPhi %int %42 %33 %39 %34
         %41 = OpIAdd %int %43 %int_1
               OpBranch %24
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  int i = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++) {
    if (cond) {
      i++;
    }
    else {
      break;
    }
  }
}
*/
TEST_F(UnswitchTest, UnswitchKillLoop) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpFOrdEqual
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false.
; CHECK: [[loop_f]] = OpLabel
; CHECK: OpBranch [[if_merge]]

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=2.
; CHECK: [[phi_i:%\w+]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %23 Uniform
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %20 = OpAccessChain %_ptr_Input_float %c %uint_0
         %21 = OpLoad %float %20
         %23 = OpFOrdEqual %bool %21 %float_0
               OpBranch %24
         %24 = OpLabel
         %42 = OpPhi %int %int_0 %5 %41 %27
               OpLoopMerge %26 %27 None
               OpBranch %28
         %28 = OpLabel
         %31 = OpSLessThan %bool %42 %int_10
               OpBranchConditional %31 %25 %26
         %25 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %23 %33 %38
         %33 = OpLabel
         %37 = OpIAdd %int %42 %int_1
               OpBranch %34
         %38 = OpLabel
               OpBranch %26
         %34 = OpLabel
               OpBranch %27
         %27 = OpLabel
         %41 = OpIAdd %int %37 %int_1
               OpBranch %24
         %26 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  int i = 0;
  int cond = int(c[0]);
  for (; i < 10; i++) {
    switch (cond) {
      case 0:
        return;
      case 1:
        discard;
      case 2:
        break;
      default:
        break;
    }
  }
  bool cond2 = i == 9;
}
*/
TEST_F(UnswitchTest, UnswitchSwitch) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpConvertFToS
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpSwitch [[cst_cond]] [[default:%\w+]] 0 [[loop_0:%\w+]] 1 [[loop_1:%\w+]] 2 [[loop_2:%\w+]]

; Loop specialized for 2.
; CHECK: [[loop_2]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_2]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=1.
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: OpBranch [[loop]]

; Loop specialized for 1.
; CHECK: [[loop_1]] = OpLabel
; CHECK: OpKill

; Loop specialized for 0.
; CHECK: [[loop_0]] = OpLabel
; CHECK: OpReturn

; Loop specialized for the default case.
; CHECK: [[default]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[default]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have i+=1.
; CHECK: [[phi_i:%\w+]] = OpIAdd %int [[phi_i]] %int_1
; CHECK: OpBranch [[loop]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %20 Uniform
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
      %int_1 = OpConstant %int 1
%_ptr_Function_bool = OpTypePointer Function %bool
       %main = OpFunction %void None %3
          %5 = OpLabel
         %18 = OpAccessChain %_ptr_Input_float %c %uint_0
         %19 = OpLoad %float %18
         %20 = OpConvertFToS %int %19
               OpBranch %21
         %21 = OpLabel
         %49 = OpPhi %int %int_0 %5 %43 %24
               OpLoopMerge %23 %24 None
               OpBranch %25
         %25 = OpLabel
         %29 = OpSLessThan %bool %49 %int_10
               OpBranchConditional %29 %22 %23
         %22 = OpLabel
               OpSelectionMerge %35 None
               OpSwitch %20 %34 0 %31 1 %32 2 %33
         %34 = OpLabel
               OpBranch %35
         %31 = OpLabel
               OpReturn
         %32 = OpLabel
               OpKill
         %33 = OpLabel
               OpBranch %35
         %35 = OpLabel
               OpBranch %24
         %24 = OpLabel
         %43 = OpIAdd %int %49 %int_1
               OpBranch %21
         %23 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 440 core
layout(location = 0)in vec4 c;
void main() {
  int i = 0;
  int j = 0;
  int k = 0;
  bool cond = c[0] == 0;
  for (; i < 10; i++) {
    for (; j < 10; j++) {
      if (cond) {
        i++;
      } else {
        j++;
      }
    }
  }
  for (; k < 10; k++) {
    if (cond) {
      k++;
    }
  }
}
*/
TEST_F(UnswitchTest, UnSwitchNested) {
  const std::string text = R"(
; CHECK: [[cst_cond:%\w+]] = OpFOrdEqual
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false, one loop is killed, j won't change anymore.
; CHECK: [[loop_f]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: [[phi_j:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_j:%\w+]] [[continue]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[iv_i]] = OpIAdd %int [[phi_i]] %int_1
; CHECK-NEXT: OpBranch [[loop]]
; CHECK: OpReturn

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_i:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_i:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: [[phi_j:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_j:%\w+]] [[continue]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_i]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] [[pre_loop_inner:%\w+]] [[merge]]

; CHECK: [[pre_loop_inner]] = OpLabel
; CHECK-NEXT: OpBranch [[loop_inner:%\w+]]
; CHECK-NEXT: [[loop_inner]] = OpLabel
; CHECK-NEXT: [[phi2_i:%\w+]] = OpPhi %int [[phi_i]] [[pre_loop_inner]] [[iv2_i:%\w+]] [[continue2:%\w+]]
; CHECK-NEXT: [[phi2_j:%\w+]] = OpPhi %int [[phi_j]] [[pre_loop_inner]] [[iv2_j:%\w+]] [[continue2]]
; CHECK-NEXT: OpLoopMerge [[merge2:%\w+]] [[continue2]] None

; CHECK: OpBranch [[continue2]]
; CHECK: [[merge2]] = OpLabel
; CHECK: OpBranch [[continue]]
; CHECK: [[merge]] = OpLabel

; Unswitched double nested loop is done. Test the single remaining one.

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[if_merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional [[cst_cond]] [[loop_t:%\w+]] [[loop_f:%\w+]]

; Loop specialized for false.
; CHECK: [[loop_f]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_k:%\w+]] = OpPhi %int %int_0 [[loop_f]] [[iv_k:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_k]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have k+=1
; CHECK: [[iv_k]] = OpIAdd %int [[phi_k]] %int_1
; CHECK: OpBranch [[loop]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; Loop specialized for true.
; CHECK: [[loop_t]] = OpLabel
; CHECK-NEXT: OpBranch [[loop:%\w+]]
; CHECK: [[loop]] = OpLabel
; CHECK-NEXT: [[phi_k:%\w+]] = OpPhi %int %int_0 [[loop_t]] [[iv_k:%\w+]] [[continue:%\w+]]
; CHECK-NEXT: OpLoopMerge [[merge:%\w+]] [[continue]] None
; CHECK: [[loop_exit:%\w+]] = OpSLessThan {{%\w+}} [[phi_k]] {{%\w+}}
; CHECK-NEXT: OpBranchConditional [[loop_exit]] {{%\w+}} [[merge]]
; Check that we have k+=2.
; CHECK: [[tmp_k:%\w+]] = OpIAdd %int [[phi_k]] %int_1
; CHECK: [[iv_k]] = OpIAdd %int [[tmp_k]] %int_1
; CHECK: OpBranch [[loop]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpBranch [[if_merge]]

; CHECK: [[if_merge]] = OpLabel
; CHECK-NEXT: OpReturn

               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 440
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
               OpDecorate %25 Uniform
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %22 = OpAccessChain %_ptr_Input_float %c %uint_0
         %23 = OpLoad %float %22
         %25 = OpFOrdEqual %bool %23 %float_0
               OpBranch %26
         %26 = OpLabel
         %67 = OpPhi %int %int_0 %5 %52 %29
         %68 = OpPhi %int %int_0 %5 %70 %29
               OpLoopMerge %28 %29 None
               OpBranch %30
         %30 = OpLabel
         %33 = OpSLessThan %bool %67 %int_10
               OpBranchConditional %33 %27 %28
         %27 = OpLabel
               OpBranch %34
         %34 = OpLabel
         %69 = OpPhi %int %67 %27 %46 %37
         %70 = OpPhi %int %68 %27 %50 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %40 = OpSLessThan %bool %70 %int_10
               OpBranchConditional %40 %35 %36
         %35 = OpLabel
               OpSelectionMerge %43 None
               OpBranchConditional %25 %42 %47
         %42 = OpLabel
         %46 = OpIAdd %int %69 %int_1
               OpBranch %43
         %47 = OpLabel
               OpReturn
         %43 = OpLabel
               OpBranch %37
         %37 = OpLabel
         %50 = OpIAdd %int %70 %int_1
               OpBranch %34
         %36 = OpLabel
               OpBranch %29
         %29 = OpLabel
         %52 = OpIAdd %int %69 %int_1
               OpBranch %26
         %28 = OpLabel
               OpBranch %53
         %53 = OpLabel
         %71 = OpPhi %int %int_0 %28 %66 %56
               OpLoopMerge %55 %56 None
               OpBranch %57
         %57 = OpLabel
         %59 = OpSLessThan %bool %71 %int_10
               OpBranchConditional %59 %54 %55
         %54 = OpLabel
               OpSelectionMerge %62 None
               OpBranchConditional %25 %61 %62
         %61 = OpLabel
         %64 = OpIAdd %int %71 %int_1
               OpBranch %62
         %62 = OpLabel
         %72 = OpPhi %int %71 %54 %64 %61
               OpBranch %56
         %56 = OpLabel
         %66 = OpIAdd %int %72 %int_1
               OpBranch %53
         %55 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<LoopUnswitchPass>(text, true);
}

/*
Generated from the following GLSL + --eliminate-local-multi-store

#version 330 core
in vec4 c;
void main() {
  bool cond = false;
  if (c[0] == 0) {
     cond = c[1] == 0;
  } else {
     cond = c[2] == 0;
  }
  for (int i = 0; i < 10; i++) {
    if (cond) {
      i++;
    }
  }
}
*/
TEST_F(UnswitchTest, UnswitchNotUniform) {
  // Check that the unswitch is not triggered (condition loop invariant but not
  // uniform)
  const std::string text = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %c
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 330
               OpName %main "main"
               OpName %c "c"
               OpDecorate %c Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %bool = OpTypeBool
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
          %c = OpVariable %_ptr_Input_v4float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_0 = OpConstant %float 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %17 = OpAccessChain %_ptr_Input_float %c %uint_0
         %18 = OpLoad %float %17
         %20 = OpFOrdEqual %bool %18 %float_0
               OpSelectionMerge %22 None
               OpBranchConditional %20 %21 %27
         %21 = OpLabel
         %24 = OpAccessChain %_ptr_Input_float %c %uint_1
         %25 = OpLoad %float %24
         %26 = OpFOrdEqual %bool %25 %float_0
               OpBranch %22
         %27 = OpLabel
         %29 = OpAccessChain %_ptr_Input_float %c %uint_2
         %30 = OpLoad %float %29
         %31 = OpFOrdEqual %bool %30 %float_0
               OpBranch %22
         %22 = OpLabel
         %52 = OpPhi %bool %26 %21 %31 %27
               OpBranch %36
         %36 = OpLabel
         %53 = OpPhi %int %int_0 %22 %51 %39
               OpLoopMerge %38 %39 None
               OpBranch %40
         %40 = OpLabel
         %43 = OpSLessThan %bool %53 %int_10
               OpBranchConditional %43 %37 %38
         %37 = OpLabel
               OpSelectionMerge %46 None
               OpBranchConditional %52 %45 %46
         %45 = OpLabel
         %49 = OpIAdd %int %53 %int_1
               OpBranch %46
         %46 = OpLabel
         %54 = OpPhi %int %53 %37 %49 %45
               OpBranch %39
         %39 = OpLabel
         %51 = OpIAdd %int %54 %int_1
               OpBranch %36
         %38 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  auto result =
      SinglePassRunAndDisassemble<LoopUnswitchPass>(text, true, false);

  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
