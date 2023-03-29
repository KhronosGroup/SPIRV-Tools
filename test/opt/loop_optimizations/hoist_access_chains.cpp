// Copyright (c) 2023 The Khronos Group Inc.
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
#include "source/opt/licm_pass.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using PassClassTest = PassTest<::testing::Test>;

/*
  Tests for the LICM pass to check it handles access chains correctly

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 460
void main() {
    // LICMPass previously generated incorrect SPIRV on the below loop by hoisting
    // the OpAccessChain of do_not_hoist1 without hoisting the OpStore of vec2(0.0f).
    // The access chain is no longer hoisted with the AreInOperandsInvariantInLoop() check.
    // However
    // - do_not_hoist1 may be hoistable if OpStore of a constant was considered movable
    // - do_not_hoist2 may be hoistable if OpStore is considered movable when its input
    //   operands are loop invariant and its output is only modified once
    for (uint i = 0; i < 123u; ++i) {
        vec2 do_not_hoist1 = vec2(0.0f);
        float do_not_hoist2 = do_not_hoist1.x;
    }
}
*/

TEST_F(PassClassTest, HoistAccessChains) {
  const std::string spirv = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 460
OpName %main "main"
OpName %i "i"
OpName %do_not_hoist1 "do_not_hoist1"
OpName %do_not_hoist2 "do_not_hoist2"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_0 = OpConstant %uint 0
%uint_123 = OpConstant %uint 123
%bool = OpTypeBool
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%float_0 = OpConstant %float 0
%17 = OpConstantComposite %v2float %float_0 %float_0
%_ptr_Function_float = OpTypePointer Function %float
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%main = OpFunction %void None %7
%21 = OpLabel
%i = OpVariable %_ptr_Function_uint Function
%do_not_hoist1 = OpVariable %_ptr_Function_v2float Function
%do_not_hoist2 = OpVariable %_ptr_Function_float Function
OpStore %i %uint_0
OpBranch %22
%22 = OpLabel
%23 = OpPhi %uint %uint_0 %21 %24 %25
OpLoopMerge %26 %25 None
OpBranch %27
%27 = OpLabel
%28 = OpULessThan %bool %23 %uint_123
OpBranchConditional %28 %29 %26
%29 = OpLabel
OpStore %do_not_hoist1 %17
%30 = OpAccessChain %_ptr_Function_float %do_not_hoist1 %uint_0
%31 = OpLoad %float %30
OpStore %do_not_hoist2 %31
OpBranch %25
%25 = OpLabel
%24 = OpIAdd %uint %23 %int_1
OpStore %i %24
OpBranch %22
%26 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LICMPass>(spirv, spirv, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
