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
  Tests for the LICM pass to check it handles access chains correctly

  Generated from the following GLSL fragment shader
--eliminate-local-multi-store has also been run on the spv binary
#version 460
layout(constant_id = 0) const uint sc = 0;
void main() {
    for (uint i = 0; i < 2u; ++i) {
        vec2 invariant_a = vec2(0.0f);
        float invariant_b = invariant_a.x;
    }
    for (uint i = 0; i < 2u; ++i) {
        vec2 invariant_c = vec2(0.0f);
        float variant_d = invariant_c[i];
    }
    for (uint i = 0; i < 2u; ++i) {
        vec2 variant_e = vec2(i);
        float variant_f = variant_e.x;
    }
    for (uint i = 0; i < 2u; ++i) {
        vec2 variant_g = vec2(0.0f);
        variant_g.x = 1.0f;
    }
    for (uint i = 0; i < 2u; ++i) {
        vec2 invariant_h = vec2(0.0f);
        float invariant_i = invariant_h[sc];
    }
}
*/

TEST_F(PassClassTest, HoistAccessChains) {
  const std::string before_hoist = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 460
OpName %main "main"
OpName %i "i"
OpName %invariant_a "invariant_a"
OpName %invariant_b "invariant_b"
OpName %i_0 "i"
OpName %invariant_c "invariant_c"
OpName %variant_d "variant_d"
OpName %i_1 "i"
OpName %variant_e "variant_e"
OpName %variant_f "variant_f"
OpName %i_2 "i"
OpName %variant_g "variant_g"
OpName %i_3 "i"
OpName %invariant_h "invariant_h"
OpName %invariant_i "invariant_i"
OpName %sc "sc"
OpDecorate %sc SpecId 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%bool = OpTypeBool
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%float_0 = OpConstant %float 0
%24 = OpConstantComposite %v2float %float_0 %float_0
%_ptr_Function_float = OpTypePointer Function %float
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_1 = OpConstant %float 1
%sc = OpSpecConstant %uint 0
%main = OpFunction %void None %3
%5 = OpLabel
%i = OpVariable %_ptr_Function_uint Function
%invariant_a = OpVariable %_ptr_Function_v2float Function
%invariant_b = OpVariable %_ptr_Function_float Function
%i_0 = OpVariable %_ptr_Function_uint Function
%invariant_c = OpVariable %_ptr_Function_v2float Function
%variant_d = OpVariable %_ptr_Function_float Function
%i_1 = OpVariable %_ptr_Function_uint Function
%variant_e = OpVariable %_ptr_Function_v2float Function
%variant_f = OpVariable %_ptr_Function_float Function
%i_2 = OpVariable %_ptr_Function_uint Function
%variant_g = OpVariable %_ptr_Function_v2float Function
%i_3 = OpVariable %_ptr_Function_uint Function
%invariant_h = OpVariable %_ptr_Function_v2float Function
%invariant_i = OpVariable %_ptr_Function_float Function
OpStore %i %uint_0
OpBranch %10
%10 = OpLabel
%93 = OpPhi %uint %uint_0 %5 %32 %13
OpLoopMerge %12 %13 None
OpBranch %14
%14 = OpLabel
%18 = OpULessThan %bool %93 %uint_2
OpBranchConditional %18 %11 %12
%11 = OpLabel
OpStore %invariant_a %24
%27 = OpAccessChain %_ptr_Function_float %invariant_a %uint_0
%28 = OpLoad %float %27
OpStore %invariant_b %28
OpBranch %13
%13 = OpLabel
%32 = OpIAdd %uint %93 %int_1
OpStore %i %32
OpBranch %10
%12 = OpLabel
OpStore %i_0 %uint_0
OpBranch %34
%34 = OpLabel
%94 = OpPhi %uint %uint_0 %12 %47 %37
OpLoopMerge %36 %37 None
OpBranch %38
%38 = OpLabel
%40 = OpULessThan %bool %94 %uint_2
OpBranchConditional %40 %35 %36
%35 = OpLabel
OpStore %invariant_c %24
%44 = OpAccessChain %_ptr_Function_float %invariant_c %94
%45 = OpLoad %float %44
OpStore %variant_d %45
OpBranch %37
%37 = OpLabel
%47 = OpIAdd %uint %94 %int_1
OpStore %i_0 %47
OpBranch %34
%36 = OpLabel
OpStore %i_1 %uint_0
OpBranch %49
%49 = OpLabel
%95 = OpPhi %uint %uint_0 %36 %64 %52
OpLoopMerge %51 %52 None
OpBranch %53
%53 = OpLabel
%55 = OpULessThan %bool %95 %uint_2
OpBranchConditional %55 %50 %51
%50 = OpLabel
%58 = OpConvertUToF %float %95
%59 = OpCompositeConstruct %v2float %58 %58
OpStore %variant_e %59
%61 = OpAccessChain %_ptr_Function_float %variant_e %uint_0
%62 = OpLoad %float %61
OpStore %variant_f %62
OpBranch %52
%52 = OpLabel
%64 = OpIAdd %uint %95 %int_1
OpStore %i_1 %64
OpBranch %49
%51 = OpLabel
OpStore %i_2 %uint_0
OpBranch %66
%66 = OpLabel
%96 = OpPhi %uint %uint_0 %51 %77 %69
OpLoopMerge %68 %69 None
OpBranch %70
%70 = OpLabel
%72 = OpULessThan %bool %96 %uint_2
OpBranchConditional %72 %67 %68
%67 = OpLabel
OpStore %variant_g %24
%75 = OpAccessChain %_ptr_Function_float %variant_g %uint_0
OpStore %75 %float_1
OpBranch %69
%69 = OpLabel
%77 = OpIAdd %uint %96 %int_1
OpStore %i_2 %77
OpBranch %66
%68 = OpLabel
OpStore %i_3 %uint_0
OpBranch %79
%79 = OpLabel
%97 = OpPhi %uint %uint_0 %68 %92 %82
OpLoopMerge %81 %82 None
OpBranch %83
%83 = OpLabel
%85 = OpULessThan %bool %97 %uint_2
OpBranchConditional %85 %80 %81
%80 = OpLabel
OpStore %invariant_h %24
%89 = OpAccessChain %_ptr_Function_float %invariant_h %sc
%90 = OpLoad %float %89
OpStore %invariant_i %90
OpBranch %82
%82 = OpLabel
%92 = OpIAdd %uint %97 %int_1
OpStore %i_3 %92
OpBranch %79
%81 = OpLabel
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
OpName %i "i"
OpName %invariant_a "invariant_a"
OpName %invariant_b "invariant_b"
OpName %i_0 "i"
OpName %invariant_c "invariant_c"
OpName %variant_d "variant_d"
OpName %i_1 "i"
OpName %variant_e "variant_e"
OpName %variant_f "variant_f"
OpName %i_2 "i"
OpName %variant_g "variant_g"
OpName %i_3 "i"
OpName %invariant_h "invariant_h"
OpName %invariant_i "invariant_i"
OpName %sc "sc"
OpDecorate %sc SpecId 0
%void = OpTypeVoid
%19 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%bool = OpTypeBool
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
%float_0 = OpConstant %float 0
%29 = OpConstantComposite %v2float %float_0 %float_0
%_ptr_Function_float = OpTypePointer Function %float
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%float_1 = OpConstant %float 1
%sc = OpSpecConstant %uint 0
%main = OpFunction %void None %19
%34 = OpLabel
%i = OpVariable %_ptr_Function_uint Function
%invariant_a = OpVariable %_ptr_Function_v2float Function
%invariant_b = OpVariable %_ptr_Function_float Function
%i_0 = OpVariable %_ptr_Function_uint Function
%invariant_c = OpVariable %_ptr_Function_v2float Function
%variant_d = OpVariable %_ptr_Function_float Function
%i_1 = OpVariable %_ptr_Function_uint Function
%variant_e = OpVariable %_ptr_Function_v2float Function
%variant_f = OpVariable %_ptr_Function_float Function
%i_2 = OpVariable %_ptr_Function_uint Function
%variant_g = OpVariable %_ptr_Function_v2float Function
%i_3 = OpVariable %_ptr_Function_uint Function
%invariant_h = OpVariable %_ptr_Function_v2float Function
%invariant_i = OpVariable %_ptr_Function_float Function
OpStore %i %uint_0
OpStore %invariant_a %29
%43 = OpAccessChain %_ptr_Function_float %invariant_a %uint_0
%44 = OpLoad %float %43
OpStore %invariant_b %44
OpBranch %35
%35 = OpLabel
%36 = OpPhi %uint %uint_0 %34 %37 %38
OpLoopMerge %39 %38 None
OpBranch %40
%40 = OpLabel
%41 = OpULessThan %bool %36 %uint_2
OpBranchConditional %41 %42 %39
%42 = OpLabel
OpBranch %38
%38 = OpLabel
%37 = OpIAdd %uint %36 %int_1
OpStore %i %37
OpBranch %35
%39 = OpLabel
OpStore %i_0 %uint_0
OpStore %invariant_c %29
OpBranch %45
%45 = OpLabel
%46 = OpPhi %uint %uint_0 %39 %47 %48
OpLoopMerge %49 %48 None
OpBranch %50
%50 = OpLabel
%51 = OpULessThan %bool %46 %uint_2
OpBranchConditional %51 %52 %49
%52 = OpLabel
%53 = OpAccessChain %_ptr_Function_float %invariant_c %46
%54 = OpLoad %float %53
OpStore %variant_d %54
OpBranch %48
%48 = OpLabel
%47 = OpIAdd %uint %46 %int_1
OpStore %i_0 %47
OpBranch %45
%49 = OpLabel
OpStore %i_1 %uint_0
OpBranch %55
%55 = OpLabel
%56 = OpPhi %uint %uint_0 %49 %57 %58
OpLoopMerge %59 %58 None
OpBranch %60
%60 = OpLabel
%61 = OpULessThan %bool %56 %uint_2
OpBranchConditional %61 %62 %59
%62 = OpLabel
%63 = OpConvertUToF %float %56
%64 = OpCompositeConstruct %v2float %63 %63
OpStore %variant_e %64
%65 = OpAccessChain %_ptr_Function_float %variant_e %uint_0
%66 = OpLoad %float %65
OpStore %variant_f %66
OpBranch %58
%58 = OpLabel
%57 = OpIAdd %uint %56 %int_1
OpStore %i_1 %57
OpBranch %55
%59 = OpLabel
OpStore %i_2 %uint_0
OpBranch %67
%67 = OpLabel
%68 = OpPhi %uint %uint_0 %59 %69 %70
OpLoopMerge %71 %70 None
OpBranch %72
%72 = OpLabel
%73 = OpULessThan %bool %68 %uint_2
OpBranchConditional %73 %74 %71
%74 = OpLabel
OpStore %variant_g %29
%75 = OpAccessChain %_ptr_Function_float %variant_g %uint_0
OpStore %75 %float_1
OpBranch %70
%70 = OpLabel
%69 = OpIAdd %uint %68 %int_1
OpStore %i_2 %69
OpBranch %67
%71 = OpLabel
OpStore %i_3 %uint_0
OpStore %invariant_h %29
%84 = OpAccessChain %_ptr_Function_float %invariant_h %sc
%85 = OpLoad %float %84
OpStore %invariant_i %85
OpBranch %76
%76 = OpLabel
%77 = OpPhi %uint %uint_0 %71 %78 %79
OpLoopMerge %80 %79 None
OpBranch %81
%81 = OpLabel
%82 = OpULessThan %bool %77 %uint_2
OpBranchConditional %82 %83 %80
%83 = OpLabel
OpBranch %79
%79 = OpLabel
%78 = OpIAdd %uint %77 %int_1
OpStore %i_3 %78
OpBranch %76
%80 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<LICMPass>(before_hoist, after_hoist, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
