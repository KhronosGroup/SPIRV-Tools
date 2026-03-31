// Copyright (c) 2026 Google Inc.
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

#include <gtest/gtest.h>

#include <tuple>

#include "source/opt/trim_variable_pointers_capabilities_pass.h"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using TrimVariablePointersCapabilitiesPassTest = PassTest<::testing::Test>;

TEST_F(TrimVariablePointersCapabilitiesPassTest,
       VariablePointers_RemovedWhenUnused) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability VariablePointers
; CHECK:       OpCapability Shader
; CHECK-NOT:   OpCapability VariablePointers
; CHECK-NOT:   OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %buf
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
  %_struct_4 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_4 = OpTypePointer StorageBuffer %_struct_4
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
        %buf = OpVariable %_ptr_StorageBuffer__struct_4 StorageBuffer
       %main = OpFunction %void None %3
          %9 = OpLabel
         %10 = OpAccessChain %_ptr_StorageBuffer_uint %buf %uint_0 %uint_0
         %11 = OpLoad %uint %10
               OpReturn
               OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimVariablePointersCapabilitiesPass>(
          kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(TrimVariablePointersCapabilitiesPassTest,
       VariablePointers_RemainsForWorkgroupSelect) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability VariablePointers
; CHECK:       OpCapability Shader
; CHECK:       OpCapability VariablePointers
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %bool = OpTypeBool
      %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
        %var = OpVariable %_ptr_Workgroup_uint Workgroup
       %main = OpFunction %void None %3
          %8 = OpLabel
          %9 = OpSelect %_ptr_Workgroup_uint %true %var %var
               OpReturn
               OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimVariablePointersCapabilitiesPass>(
          kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimVariablePointersCapabilitiesPassTest,
       VariablePointers_RemainsWhenItIsTheOnlyExplicitCapabilityForStorageBuffer) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability VariablePointers
; CHECK:       OpCapability Shader
; CHECK:       OpCapability VariablePointers
; CHECK-NOT:   OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %buf
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
  %_struct_4 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_4 = OpTypePointer StorageBuffer %_struct_4
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
          %8 = OpTypeFunction %uint %_ptr_StorageBuffer_uint
        %buf = OpVariable %_ptr_StorageBuffer__struct_4 StorageBuffer
       %main = OpFunction %void None %3
         %10 = OpLabel
         %11 = OpAccessChain %_ptr_StorageBuffer_uint %buf %uint_0 %uint_0
         %12 = OpFunctionCall %uint %callee %11
               OpReturn
               OpFunctionEnd
     %callee = OpFunction %uint None %8
         %15 = OpFunctionParameter %_ptr_StorageBuffer_uint
         %16 = OpLabel
               OpReturnValue %uint_0
               OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimVariablePointersCapabilitiesPass>(
          kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimVariablePointersCapabilitiesPassTest,
       VariablePointersStorageBuffer_RemainsForFunctionCallParameter) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
; CHECK:       OpCapability Shader
; CHECK:       OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %buf
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
  %_struct_4 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_4 = OpTypePointer StorageBuffer %_struct_4
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
          %8 = OpTypeFunction %uint %_ptr_StorageBuffer_uint
        %buf = OpVariable %_ptr_StorageBuffer__struct_4 StorageBuffer
       %main = OpFunction %void None %3
         %10 = OpLabel
         %11 = OpAccessChain %_ptr_StorageBuffer_uint %buf %uint_0 %uint_0
         %12 = OpFunctionCall %uint %callee %11
               OpReturn
               OpFunctionEnd
     %callee = OpFunction %uint None %8
         %15 = OpFunctionParameter %_ptr_StorageBuffer_uint
         %16 = OpLabel
               OpReturnValue %uint_0
               OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimVariablePointersCapabilitiesPass>(
          kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithoutChange);
}

TEST_F(TrimVariablePointersCapabilitiesPassTest,
       VariablePointers_RemovedWhenStorageBufferCapabilityIsAlsoDeclared) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability VariablePointers
               OpCapability VariablePointersStorageBuffer
; CHECK:       OpCapability Shader
; CHECK-NOT:   OpCapability VariablePointers
; CHECK:       OpCapability VariablePointersStorageBuffer
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %buf
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %_struct_4 Block
               OpMemberDecorate %_struct_4 0 Offset 0
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
  %_struct_4 = OpTypeStruct %uint
%_ptr_StorageBuffer__struct_4 = OpTypePointer StorageBuffer %_struct_4
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
          %8 = OpTypeFunction %uint %_ptr_StorageBuffer_uint
        %buf = OpVariable %_ptr_StorageBuffer__struct_4 StorageBuffer
       %main = OpFunction %void None %3
         %10 = OpLabel
         %11 = OpAccessChain %_ptr_StorageBuffer_uint %buf %uint_0 %uint_0
         %12 = OpFunctionCall %uint %callee %11
               OpReturn
               OpFunctionEnd
     %callee = OpFunction %uint None %8
         %15 = OpFunctionParameter %_ptr_StorageBuffer_uint
         %16 = OpLabel
               OpReturnValue %uint_0
               OpFunctionEnd
  )";
  const auto result =
      SinglePassRunAndMatch<TrimVariablePointersCapabilitiesPass>(
          kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
