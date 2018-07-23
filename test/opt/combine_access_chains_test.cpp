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

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
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

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
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

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
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

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, PtrAccessChainFromAccessChainExtraIndices) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%uint_array_4_array_4_array_4 = OpTypeArray %uint_array_4_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%ptr_Workgroup_uint_array_4_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1 %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_2 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessChainCombineElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int6]] [[int3]]
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
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessChainOnlyElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int4:%\w+]] = OpConstant [[int]] 4
; CHECK: [[array:%\w+]] = OpTypeArray [[int]] [[int4]]
; CHECK: [[ptr_array:%\w+]] = OpTypePointer Workgroup [[array]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: [[int6:%\w+]] = OpConstant [[int]] 6
; CHECK: OpPtrAccessChain [[ptr_array]] [[var]] [[int6]]
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
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       PtrAccessChainFromPtrAccessCombineNonElementIndex) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int3]] [[int3]] [[int3]]
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
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Function_uint = OpTypePointer Function %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3 %uint_0
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint %gep %uint_3 %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest,
       AccessChainFromPtrAccessChainOnlyElementOperand) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int3]] [[int3]]
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
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_3
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, AccessChainFromPtrAccessChainAppend) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[int3:%\w+]] = OpConstant [[int]] 3
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpPtrAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]] [[int3]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%ptr_gep = OpPtrAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1 %uint_2
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_3
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}

TEST_F(CombineAccessChainsTest, AccessChainFromAccessChainAppend) {
  const std::string text = R"(
; CHECK: [[int:%\w+]] = OpTypeInt 32 0
; CHECK: [[int1:%\w+]] = OpConstant [[int]] 1
; CHECK: [[int2:%\w+]] = OpConstant [[int]] 2
; CHECK: [[ptr_int:%\w+]] = OpTypePointer Workgroup [[int]]
; CHECK: [[var:%\w+]] = OpVariable {{%\w+}} Workgroup
; CHECK: OpAccessChain [[ptr_int]] [[var]] [[int1]] [[int2]]
OpCapability Shader
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_array_4 = OpTypeArray %uint %uint_4
%uint_array_4_array_4 = OpTypeArray %uint_array_4 %uint_4
%ptr_Workgroup_uint = OpTypePointer Workgroup %uint
%ptr_Workgroup_uint_array_4 = OpTypePointer Workgroup %uint_array_4
%ptr_Workgroup_uint_array_4_array_4 = OpTypePointer Workgroup %uint_array_4_array_4
%var = OpVariable %ptr_Workgroup_uint_array_4_array_4 Workgroup
%void_func = OpTypeFunction %void
%main = OpFunction %void None %void_func
%main_lab = OpLabel
%ptr_gep = OpAccessChain %ptr_Workgroup_uint_array_4 %var %uint_1
%gep = OpAccessChain %ptr_Workgroup_uint %ptr_gep %uint_2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<CombineAccessChains>(text, true);
}
#endif  // SPIRV_EFFCEE

}  // namespace
}  // namespace opt
}  // namespace spvtools
