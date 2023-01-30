// Copyright (c) 2022 The Khronos Group Inc.
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

using ::testing::UnorderedElementsAre;
using PassClassTest = PassTest<::testing::Test>;

/*
  Tests for the LICM pass to ensure it is not hoisting variables
  modified more than once or which are modified in a loop and
  referenced outside of it.

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 460
void main() {
    float invariant_a;
    float variant_b;
    float variant_c = 0.0f;
    for (uint i = 0; i < 2u; ++i) {
      if (i > 1) {
        invariant_a = 0.0f;
        variant_b = 0.0f;
        variant_c = 1.0f;
      }
    }
    variant_b *= 3.0f;
}
*/

TEST_F(PassClassTest, HoistUseDependent) {
  const std::string before_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 460
OpName %main "main"
OpName %variant_c "variant_c"
OpName %i "i"
OpName %invariant_a "invariant_a"
OpName %variant_b "variant_b"
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%float_1 = OpConstant %float 1
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_3 = OpConstant %float 3
%40 = OpUndef %float
%main = OpFunction %void None %3
%5 = OpLabel
%variant_c = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_uint Function
%invariant_a = OpVariable %_ptr_Function_float Function
%variant_b = OpVariable %_ptr_Function_float Function
OpStore %variant_c %float_0
OpStore %i %uint_0
OpBranch %14
%14 = OpLabel
%39 = OpPhi %float %40 %5 %42 %17
%38 = OpPhi %uint %uint_0 %5 %34 %17
OpLoopMerge %16 %17 None
OpBranch %18
%18 = OpLabel
%22 = OpULessThan %bool %38 %uint_2
OpBranchConditional %22 %15 %16
%15 = OpLabel
%25 = OpUGreaterThan %bool %38 %uint_1
OpSelectionMerge %27 None
OpBranchConditional %25 %26 %27
%26 = OpLabel
OpStore %invariant_a %float_0
OpStore %variant_b %float_0
OpStore %variant_c %float_1
OpBranch %27
%27 = OpLabel
%42 = OpPhi %float %39 %15 %float_0 %26
OpBranch %17
%17 = OpLabel
%34 = OpIAdd %uint %38 %int_1
OpStore %i %34
OpBranch %14
%16 = OpLabel
%37 = OpFMul %float %39 %float_3
OpStore %variant_b %37
OpReturn
OpFunctionEnd
)";

  const std::string after_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 460
OpName %main "main"
OpName %variant_c "variant_c"
OpName %i "i"
OpName %invariant_a "invariant_a"
OpName %variant_b "variant_b"
%void = OpTypeVoid
%8 = OpTypeFunction %void
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%bool = OpTypeBool
%uint_1 = OpConstant %uint 1
%float_1 = OpConstant %float 1
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_3 = OpConstant %float 3
%22 = OpUndef %float
%main = OpFunction %void None %8
%23 = OpLabel
%variant_c = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_uint Function
%invariant_a = OpVariable %_ptr_Function_float Function
%variant_b = OpVariable %_ptr_Function_float Function
OpStore %variant_c %float_0
OpStore %i %uint_0
OpStore %invariant_a %float_0
OpBranch %24
%24 = OpLabel
%25 = OpPhi %float %22 %23 %26 %27
%28 = OpPhi %uint %uint_0 %23 %29 %27
OpLoopMerge %30 %27 None
OpBranch %31
%31 = OpLabel
%32 = OpULessThan %bool %28 %uint_2
OpBranchConditional %32 %33 %30
%33 = OpLabel
%34 = OpUGreaterThan %bool %28 %uint_1
OpSelectionMerge %35 None
OpBranchConditional %34 %36 %35
%36 = OpLabel
OpStore %variant_b %float_0
OpStore %variant_c %float_1
OpBranch %35
%35 = OpLabel
%26 = OpPhi %float %25 %33 %float_0 %36
OpBranch %27
%27 = OpLabel
%29 = OpIAdd %uint %28 %int_1
OpStore %i %29
OpBranch %24
%30 = OpLabel
%37 = OpFMul %float %25 %float_3
OpStore %variant_b %37
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LICMPass>(before_hoist, after_hoist, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
