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
; CHECK: [[scope:%\w+]] = OpConstant {{%w\+}} 2
; CHECK: OpLoad {{%w\+}} {{%w\+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%w\+}} {{%w\+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
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
; CHECK: [[scope:%\w+]] = OpConstant {{%w\+}} 2
; CHECK: OpLoad {{%w\+}} {{%w\+}} MakePointerAvailableKHR|NonPrivatePointerKHR [[scope]]
; CHECK: OpStore {{%w\+}} {{%w\+}} MakePointerVisibleKHR|NonPrivatePointerKHR [[scope]]
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
#endif

}  // namespace
