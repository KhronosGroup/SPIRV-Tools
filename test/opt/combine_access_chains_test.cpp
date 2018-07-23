// Copyright (c) 2018 Google LLC
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

#include "assembly_builder.h"
#include "gmock/gmock.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using CombineAccessChainsTest = PassTest<::testing::Test>;

#ifdef SPIRV_EFFCEE
TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint %var %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChainsPass>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromInBoundsAccessChainConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpInBoundsAccessChain [[ptr_int]] [[var]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpInBoundsAccessChain %ptr_Workgroup_uint %var %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChainsPass>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainCombineConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int2]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint %var %uint_1
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChainsPass>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainNonConstant) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[ld1:%\w+]] = OpLoad
; CHECK: [[ld2:%\w+]] = OpLoad
; CHECK: [[add:%\w+]] = OpIAdd [[int]] [[ld1]] [[ld2]]
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[add]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%local_var = OpVariable %ptr_Function_uint Function
%ld1 = OpLoad %uint %local_var
%gep = OpAccessChain %ptr_Workgroup_uint %var %ld1
%ld2 = OpLoad %uint %local_var
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %ld2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChainsPass>(text, true);
}
#endif  // SPIRV_EFFCEE

}  // namespace
}  // namespace opt
}  // namespace spvtools
