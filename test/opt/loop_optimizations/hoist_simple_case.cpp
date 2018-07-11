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

#include <gmock/gmock.h>

#include "../pass_fixture.h"
#include "opt/licm_pass.h"

namespace {

using namespace spvtools;
using ::testing::UnorderedElementsAre;

using PassClassTest = PassTest<::testing::Test>;

/*
  A simple test for the LICM pass

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 440 core
void main(){
  int a = 1;
  int b = 2;
  int hoist = 0;
  for (int i = 0; i < 10; i++) {
    // invariant
    hoist = a + b;
  }
}
*/
TEST_F(PassClassTest, SimpleHoist) {
  const std::string before_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%main = OpFunction %void None %4
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%14 = OpPhi %int %int_0 %12 %15 %16
%17 = OpPhi %int %int_0 %12 %18 %16
OpLoopMerge %19 %16 None
OpBranch %20
%20 = OpLabel
%21 = OpSLessThan %bool %17 %int_10
OpBranchConditional %21 %22 %19
%22 = OpLabel
%15 = OpIAdd %int %int_1 %int_2
OpBranch %16
%16 = OpLabel
%18 = OpIAdd %int %17 %int_1
OpBranch %13
%19 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 440
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_0 = OpConstant %int 0
%int_10 = OpConstant %int 10
%bool = OpTypeBool
%main = OpFunction %void None %4
%12 = OpLabel
%15 = OpIAdd %int %int_1 %int_2
OpBranch %13
%13 = OpLabel
%14 = OpPhi %int %int_0 %12 %15 %16
%17 = OpPhi %int %int_0 %12 %18 %16
OpLoopMerge %19 %16 None
OpBranch %20
%20 = OpLabel
%21 = OpSLessThan %bool %17 %int_10
OpBranchConditional %21 %22 %19
%22 = OpLabel
OpBranch %16
%16 = OpLabel
%18 = OpIAdd %int %17 %int_1
OpBranch %13
%19 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<opt::LICMPassToken>(before_hoist, after_hoist, true);
}

}  // namespace
