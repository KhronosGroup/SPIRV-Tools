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

namespace {

using namespace spvtools;

using UpgradeMemoryModelTest = PassTest<::testing::Test>;

#ifdef SPIRV_EFFCEE
TEST_F(UpgradeMemoryModelTest, InvalidMemoryModelOpenCL) {
  const std::string text = R"(
; CHECK: OpMemoryModel Logical OpenCL
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Logical OpenCL
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, InvalidMemoryModelVulkanKHR) {
  const std::string text = R"(
; CHECK: OpMemoryModel Logical VulkanKHR
OpCapability Shader
OpCapability Linkage
OpCapability VulkanMemoryModelKHR
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical VulkanKHR
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, JustMemoryModel) {
  const std::string text = R"(
; CHECK: OpCapability VulkanMemoryModelKHR
; CHECK: OpExtension "SPV_KHR_vulkan_memory_model"
; CHECK: OpMemoryModel Logical VulkanKHR
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, RemoveDecorations) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
OpDecorate %var Coherent
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, WorkgroupVariable) {
  const std::string text = R"(
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%var = OpVariable %ptr_int_Workgroup Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
%st = OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, WorkgroupFunctionParameter) {
  const std::string text = R"(
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%func_ty = OpTypeFunction %void %ptr_int_Workgroup
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Workgroup
%1 = OpLabel
%ld = OpLoad %int %param
%st = OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariable) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameter) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_int_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Uniform
%1 = OpLabel
%ld = OpLoad %int %param
OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableOnlyVolatile) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK-NOT: OpConstant
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%ld = OpLoad %int %var
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableCopied) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_int_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%copy = OpCopyObject %ptr_int_Uniform %var
%ld = OpLoad %int %copy
OpStore %copy %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameterCopied) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_int_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_Uniform
%1 = OpLabel
%copy = OpCopyObject %ptr_int_Uniform %param
%ld = OpLoad %int %copy
%copy2 = OpCopyObject %ptr_int_Uniform %param
OpStore %copy2 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformVariableAccessChain) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int3 = OpConstant %int 3
%int_array_3 = OpTypeArray %int %int3
%ptr_intarray_Uniform = OpTypePointer Uniform %int_array_3
%ptr_int_Uniform = OpTypePointer Uniform %int
%var = OpVariable %ptr_intarray_Uniform Uniform
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%gep = OpAccessChain %ptr_int_Uniform %var %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, SimpleUniformFunctionParameterAccessChain) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int3 = OpConstant %int 3
%int_array_3 = OpTypeArray %int %int3
%ptr_intarray_Uniform = OpTypePointer Uniform %int_array_3
%ptr_int_Uniform = OpTypePointer Uniform %int
%func_ty = OpTypeFunction %void %ptr_intarray_Uniform
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_intarray_Uniform
%1 = OpLabel
%ld_gep = OpAccessChain %ptr_int_Uniform %param %int0
%ld = OpLoad %int %ld_gep
%st_gep = OpAccessChain %ptr_int_Uniform %param %int0
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerSelect) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %var Coherent
OpDecorate %var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%null = OpConstantNull %ptr_int_StorageBuffer
%var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%select = OpSelect %ptr_int_StorageBuffer %true %var %null
%ld = OpLoad %int %select
OpStore %var %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerSelectConservative) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %var1 Coherent
OpDecorate %var2 Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%var1 = OpVariable %ptr_int_StorageBuffer StorageBuffer
%var2 = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
%select = OpSelect %ptr_int_StorageBuffer %true %var1 %var2
%ld = OpLoad %int %select
OpStore %select %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, VariablePointerIncrement) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate {{%\w+}} Coherent
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpCapability VariablePointers
OpExtension "SPV_KHR_variable_pointers"
OpMemoryModel Logical GLSL450
OpDecorate %param Coherent
OpDecorate %param ArrayStride 4
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%int10 = OpConstant %int 10
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_int_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_int_StorageBuffer
%1 = OpLabel
OpBranch %2
%2 = OpLabel
%phi = OpPhi %ptr_int_StorageBuffer %param %1 %ptr_next %2
%iv = OpPhi %int %int0 %1 %inc %2
%inc = OpIAdd %int %iv %int1
%ptr_next = OpPtrAccessChain %ptr_int_StorageBuffer %phi %int1
%cmp = OpIEqual %bool %iv %int10
OpLoopMerge %3 %2 None
OpBranchConditional %cmp %3 %2
%3 = OpLabel
%ld = OpLoad %int %phi
OpStore %phi %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentStructElement) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%struct = OpTypeStruct %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%gep = OpAccessChain %ptr_int_StorageBuffer %param %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentElementFullStructAccess) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%struct = OpTypeStruct %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%ld = OpLoad %struct %param
OpStore %param %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentElementNotAccessed) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %struct 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%struct = OpTypeStruct %int %int
%ptr_struct_StorageBuffer = OpTypePointer StorageBuffer %struct
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_struct_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_struct_StorageBuffer
%1 = OpLabel
%gep = OpAccessChain %ptr_int_StorageBuffer %param %int0
%ld = OpLoad %int %gep
OpStore %gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, MultiIndexAccessCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int0 %int0 %int1
%ld = OpLoad %int %ld_gep
%st_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int1 %int0 %int1
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, MultiIndexAccessNonCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int0 %int0 %int0
%ld = OpLoad %int %ld_gep
%st_gep = OpInBoundsAccessChain %ptr_int_StorageBuffer %param %int1 %int0 %int0
OpStore %st_gep %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ConsecutiveAccessChainCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, ConsecutiveAccessChainNonCoherent) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK-NOT: MakePointerAvailableKHR
; CHECK-NOT: NonPrivatePointerKHR
; CHECK-NOT: MakePointerVisibleKHR
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %inner 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int0
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int0
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CoherentStructElementAccess) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpLoad {{%\w+}} {{%\w+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %middle 0 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, NonCoherentLoadCoherentStore) {
  const std::string text = R"(
; CHECK-NOT: OpMemberDecorate
; CHECK: [[scope:%\w+]] = OpConstant {{%\w+}} 1
; CHECK-NOT: MakePointerAvailableKHR
; CHECK: OpStore {{%\w+}} {{%\w+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpMemberDecorate %outer 1 Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%int1 = OpConstant %int 1
%inner = OpTypeStruct %int %int
%middle = OpTypeStruct %inner
%outer = OpTypeStruct %middle %middle
%ptr_outer_StorageBuffer = OpTypePointer StorageBuffer %outer
%ptr_middle_StorageBuffer = OpTypePointer StorageBuffer %middle
%ptr_inner_StorageBuffer = OpTypePointer StorageBuffer %inner
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_outer_StorageBuffer
%func = OpFunction %void None %func_ty
%param = OpFunctionParameter %ptr_outer_StorageBuffer
%1 = OpLabel
%ld_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int0
%ld_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %ld_gep1 %int0
%ld_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %ld_gep2 %int1
%ld = OpLoad %int %ld_gep3
%st_gep1 = OpInBoundsAccessChain %ptr_middle_StorageBuffer %param %int1
%st_gep2 = OpInBoundsAccessChain %ptr_inner_StorageBuffer %st_gep1 %int0
%st_gep3 = OpInBoundsAccessChain %ptr_int_StorageBuffer %st_gep2 %int1
OpStore %st_gep3 %ld
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemory) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[device:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpCopyMemory {{%\w+}} {{%\w+}} Volatile|MakePointerVisibleKHR|NonPrivatePointerKHR [[device]]
; CHECK-NOT: [[device]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %in_var Coherent
OpDecorate %out_var Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%in_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%out_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpCopyMemory %out_var %in_var
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemorySized) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK: [[device:%\w+]] = OpConstant {{%\w+}} 1
; CHECK: OpCopyMemorySized {{%\w+}} {{%\w+}} {{%\w+}} Volatile|MakePointerAvailableKHR|NonPrivatePointerKHR [[device]]
; CHECK-NOT: [[device]]
OpCapability Shader
OpCapability Linkage
OpCapability Addresses
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %out_param Coherent
OpDecorate %in_param Volatile
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int4 = OpConstant %int 4
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%func_ty = OpTypeFunction %void %ptr_int_StorageBuffer %ptr_int_StorageBuffer
%func = OpFunction %void None %func_ty
%in_param = OpFunctionParameter %ptr_int_StorageBuffer
%out_param = OpFunctionParameter %ptr_int_StorageBuffer
%1 = OpLabel
OpCopyMemorySized %out_param %in_param %int4
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}

TEST_F(UpgradeMemoryModelTest, CopyMemoryTwoScopes) {
  const std::string text = R"(
; CHECK-NOT: OpDecorate
; CHECK-DAG: [[device:%\w+]] = OpConstant {{%\w+}} 1
; CHECK-DAG: [[workgroup:%\w+]] = OpConstant {{%\w+}} 2
; CHECK: OpCopyMemory {{%\w+}} {{%\w+}} MakePointerAvailableKHR|MakePointerVisibleKHR|NonPrivatePointerKHR [[device]] [[workgroup]]
OpCapability Shader
OpCapability Linkage
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpDecorate %in_var Coherent
OpDecorate %out_var Coherent
%void = OpTypeVoid
%int = OpTypeInt 32 0
%ptr_int_Workgroup = OpTypePointer Workgroup %int
%ptr_int_StorageBuffer = OpTypePointer StorageBuffer %int
%in_var = OpVariable %ptr_int_StorageBuffer StorageBuffer
%out_var = OpVariable %ptr_int_Workgroup Workgroup
%func_ty = OpTypeFunction %void
%func = OpFunction %void None %func_ty
%1 = OpLabel
OpCopyMemory %out_var %in_var
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<opt::UpgradeMemoryModel>(text, true);
}
#endif

}  // namespace
