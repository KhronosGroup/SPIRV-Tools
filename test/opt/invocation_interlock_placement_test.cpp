// Copyright (c) 2023 Google Inc.
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

#include <algorithm>
#include <queue>
#include <set>
#include <tuple>

#include "spirv-tools/optimizer.hpp"
#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InterlockInvocationPlacementTest = PassTest<::testing::Test>;

TEST_F(InterlockInvocationPlacementTest, CheckUnchangedIfNotFragment) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
       %main = OpFunction %void None %1
          %2 = OpLabel
               OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
	             OpEndInvocationInterlockEXT
	             OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  EXPECT_EQ(
      Pass::Status::SuccessWithoutChange,
      std::get<1>(SinglePassRunAndDisassemble<InvocationInterlockPlacementPass>(
          kTest, /* skip_nop= */ false, /* do_validation= */ false)));
}

TEST_F(InterlockInvocationPlacementTest, CheckUnchangedWithoutCapability) {
  const std::string kTest = R"(
               OpCapability Shader
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
       %main = OpFunction %void None %1
          %2 = OpLabel
               OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
	             OpEndInvocationInterlockEXT
	             OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  EXPECT_EQ(
      Pass::Status::SuccessWithoutChange,
      std::get<1>(SinglePassRunAndDisassemble<InvocationInterlockPlacementPass>(
          kTest, /* skip_nop= */ false, /* do_validation= */ false)));
}

TEST_F(InterlockInvocationPlacementTest, CheckSingleBasicBlock) {
  // We're using OpNoLine as a generic standin for any other instruction, to
  // test that begin and end aren't moved.
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
       %main = OpFunction %void None %1
; CHECK: OpLabel
          %2 = OpLabel
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
	             OpEndInvocationInterlockEXT
	             OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckFunctionCallExtractionBegin) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
        %foo = OpFunction %void None %1
; CHECK: OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
          %2 = OpLabel
               OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd
       %main = OpFunction %void None %1
; CHECK: OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpFunctionCall
          %4 = OpFunctionCall %void %foo
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckFunctionCallExtractionEnd) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
        %foo = OpFunction %void None %1
; CHECK: OpLabel
; CHECK-NOT: OpEndInvocationInterlockEXT
          %2 = OpLabel
               OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd
       %main = OpFunction %void None %1
; CHECK: OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpFunctionCall
          %4 = OpFunctionCall %void %foo
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest,
       CheckFunctionCallExtractionRepeatedCall) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
        %foo = OpFunction %void None %1
; CHECK: OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
          %2 = OpLabel
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd
       %main = OpFunction %void None %1
; CHECK: OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpFunctionCall
          %4 = OpFunctionCall %void %foo
; CHECK-NEXT: OpFunctionCall
          %5 = OpFunctionCall %void %foo
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest,
       CheckFunctionCallExtractionNestedCall) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
          %1 = OpTypeFunction %void
        %foo = OpFunction %void None %1
; CHECK: OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
          %2 = OpLabel
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd
        %bar = OpFunction %void None %1
; CHECK: OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
          %3 = OpLabel
          %4 = OpFunctionCall %void %foo
               OpReturn
; CHECK: OpFunctionEnd
               OpFunctionEnd
       %main = OpFunction %void None %1
; CHECK: OpLabel
          %5 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpFunctionCall
          %6 = OpFunctionCall %void %bar
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckLoopExtraction) {
  // Tests that any begin or end instructions in a loop are moved outside of the
  // loop.
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

          %2 = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpBranch %3

          %3 = OpLabel
               OpLoopMerge %3 %4 None
; CHECK: OpBranchConditional
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpBranchConditional %true %4 %5

          %4 = OpLabel
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK: OpBranch
               OpBranch %3

; CHECK-NEXT: OpLabel
          %5 = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckAddBeginToElse) {
  // Test that if there is a begin in a single branch of a conditional, begin
  // will be added to the other branch.
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
	             OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

          %2 = OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
               OpSelectionMerge %5 None
; CHECK: OpBranchConditional
               OpBranchConditional %true %3 %4

; CHECK-NEXT: OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
               OpBranch %5

          %4 = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpBranch
               OpBranch %5

; CHECK-NEXT: OpLabel
          %5 = OpLabel
               OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckAddEndToElse) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
	             OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

          %2 = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpSelectionMerge %5 None
; CHECK: OpBranchConditional
               OpBranchConditional %true %3 %4

; CHECK-NEXT: OpLabel
          %3 = OpLabel
               OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
               OpBranch %5

          %4 = OpLabel
; CHECK: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
               OpBranch %5

; CHECK-NEXT: OpLabel
          %5 = OpLabel
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckSplitIfWithoutElseBegin) {
  // Test that if there is a begin in the then branch of a conditional, and no
  // else branch, an else branch with a begin will created.
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
	             OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

          %2 = OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
               OpSelectionMerge %5 None
; CHECK: OpBranchConditional
               OpBranchConditional %true %3 %5

; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpBranch

; CHECK-NEXT: OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpBranch %5

; CHECK: OpLabel
          %5 = OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckSplitIfWithoutElseEnd) {
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
	             OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

          %2 = OpLabel

; CHECK: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpSelectionMerge [[merge:%\d+]]
               OpSelectionMerge %5 None
; CHECK-NEXT: OpBranchConditional %true [[then:%\d+]] [[else:%\d+]]
               OpBranchConditional %true %3 %5

; CHECK-NEXT: [[else]] = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[merge]]

; CHECK-NEXT: [[then]] = OpLabel
          %3 = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[merge]]
               OpBranch %5

; CHECK-NEXT: [[merge]] = OpLabel
          %5 = OpLabel
; CHECK-NEXT: OpReturn
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(InterlockInvocationPlacementTest, CheckSplitSwitch) {
  // Test that if there is a begin or end in a single branch of a switch, begin
  // or end will be added to all the other branches.
  const std::string kTest = R"(
               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
	             OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockOrderedEXT
               OpName %main "main"
       %void = OpTypeVoid
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
           %1 = OpTypeFunction %void
       %main = OpFunction %void None %1

; CHECK: OpLabel
          %2 = OpLabel
; CHECK-NEXT: OpSelectionMerge [[merge:%\d+]]
               OpSelectionMerge %8 None
; CHECK-NEXT: OpSwitch %uint_1 [[default:%\d+]] 0 [[case_0:%\d+]] 1 [[case_1:%\d+]] 2 [[case_2:%\d+]]
               OpSwitch %uint_1 %8 0 %4 1 %5 2 %8

; CHECK-NEXT: [[case_2]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[merge]]

; CHECK-NEXT: [[default]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[merge]]

; CHECK-NEXT: [[case_0]] = OpLabel
          %4 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpBranch [[merge]]
               OpBranch %8

; CHECK-NEXT: [[case_1]] = OpLabel
          %5 = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NOT: OpEndInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpNoLine
               OpNoLine
; CHECK-NEXT: OpBranch [[merge]]
               OpBranch %8

; CHECK-NEXT: [[merge]] = OpLabel
          %8 = OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
  )";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  const auto result = SinglePassRunAndMatch<InvocationInterlockPlacementPass>(
      kTest, /* skip_nop= */ false);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

const std::string kEarlyExitHeader = R"(
               OpCapability Shader
               OpCapability FragmentShaderPixelInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %rov
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main PixelInterlockOrderedEXT
               OpDecorate %rov DescriptorSet 0
               OpDecorate %rov Binding 0
)";

const std::string kEarlyExitTypes = R"(
       %void = OpTypeVoid
       %bool = OpTypeBool
       %cond = OpUndef %bool
       %uint = OpTypeInt 32 0
       %zero = OpConstant %uint 0
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
      %value = OpConstantNull %v4float
       %func = OpTypeFunction %void
)";

const std::string kEarlyExitImage = kEarlyExitTypes + R"(
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
      %coord = OpConstantNull %v2int
      %image = OpTypeImage %float 2D 0 0 0 2 Rgba32f
  %image_ptr = OpTypePointer UniformConstant %image
        %rov = OpVariable %image_ptr UniformConstant
)";

const std::string kEarlyExitBuffer = R"(
               OpDecorate %array ArrayStride 16
               OpMemberDecorate %buffer 0 Offset 0
               OpDecorate %buffer Block
)" + kEarlyExitTypes + R"(
      %array = OpTypeRuntimeArray %v4float
     %buffer = OpTypeStruct %array
%buffer_ptr = OpTypePointer StorageBuffer %buffer
  %data_ptr = OpTypePointer StorageBuffer %v4float
        %rov = OpVariable %buffer_ptr StorageBuffer
)";

// Checks that every path through each function executes exactly one begin
// followed by exactly one end, and that every access is between them. Loads,
// stores, image reads and image writes in these modules are all ROV accesses.
// Explores (block, begin count, end count) states rather than unrolling loops.
void CheckExactlyOneInterlockPerPath(IRContext* context) {
  for (auto& function : *context->module()) {
    using State = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::queue<State> pending;
    std::set<State> visited;
    pending.emplace(function.entry()->id(), 0, 0);
    while (!pending.empty()) {
      const auto state = pending.front();
      pending.pop();
      if (!visited.insert(state).second) continue;
      auto block = context->cfg()->block(std::get<0>(state));
      uint32_t begins = std::get<1>(state);
      uint32_t ends = std::get<2>(state);
      SCOPED_TRACE(block->id());
      for (auto& inst : *block) {
        switch (inst.opcode()) {
          case spv::Op::OpBeginInvocationInterlockEXT:
            EXPECT_EQ(begins, 0u);
            EXPECT_EQ(ends, 0u);
            begins = std::min(begins + 1, 2u);
            break;
          case spv::Op::OpEndInvocationInterlockEXT:
            EXPECT_EQ(begins, 1u);
            EXPECT_EQ(ends, 0u);
            ends = std::min(ends + 1, 2u);
            break;
          case spv::Op::OpLoad:
          case spv::Op::OpStore:
          case spv::Op::OpImageRead:
          case spv::Op::OpImageWrite:
            EXPECT_EQ(begins, 1u);
            EXPECT_EQ(ends, 0u);
            break;
          default:
            break;
        }
      }
      if (block->tail()->IsReturnOrAbort()) {
        EXPECT_EQ(begins, 1u);
        EXPECT_EQ(ends, 1u);
      }
      block->ForEachSuccessorLabel([&](uint32_t successor) {
        pending.emplace(successor, begins, ends);
      });
    }
  }
}

TEST_F(InterlockInvocationPlacementTest, EarlyReturnBeforeImageAccess) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpSelectionMerge %exit None
               OpSwitch %zero %test
       %test = OpLabel
               OpSelectionMerge %read None
; CHECK: OpBranchConditional %{{\w+}} [[early:%\w+]] [[read:%\w+]]
               OpBranchConditional %cond %early %read
; CHECK: [[early]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
      %early = OpLabel
               OpBranch %exit
       %read = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
; CHECK: OpImageRead
          %d = OpImageRead %v4float %img1 %coord
               OpEndInvocationInterlockEXT
               OpBranch %write
      %write = OpLabel
               OpBeginInvocationInterlockEXT
       %img2 = OpLoad %image %rov
; CHECK: OpImageWrite
               OpImageWrite %img2 %coord %d
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpBranch %exit
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, EarlyReturnBetweenImageAccesses) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
          %d = OpImageRead %v4float %img1 %coord
               OpEndInvocationInterlockEXT
               OpSelectionMerge %exit None
; CHECK: OpBranchConditional %{{\w+}} [[early:%\w+]] [[write:%\w+]]
               OpBranchConditional %cond %early %write
; CHECK: [[early]] = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
      %early = OpLabel
               OpBranch %exit
      %write = OpLabel
               OpBeginInvocationInterlockEXT
       %img2 = OpLoad %image %rov
               OpImageWrite %img2 %coord %d
               OpEndInvocationInterlockEXT
               OpBranch %exit
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, EarlyReturnBeforeBufferAccess) {
  const std::string text = kEarlyExitHeader + kEarlyExitBuffer + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpSelectionMerge %exit None
               OpSwitch %zero %test
       %test = OpLabel
               OpSelectionMerge %access None
; CHECK: OpBranchConditional %{{\w+}} [[early:%\w+]] [[access:%\w+]]
               OpBranchConditional %cond %early %access
; CHECK: [[early]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
      %early = OpLabel
               OpBranch %exit
     %access = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
               OpBeginInvocationInterlockEXT
        %ptr = OpAccessChain %data_ptr %rov %zero %zero
          %d = OpLoad %v4float %ptr
               OpEndInvocationInterlockEXT
               OpBranch %write
      %write = OpLabel
               OpBeginInvocationInterlockEXT
; CHECK: OpStore
               OpStore %ptr %d
; CHECK-NEXT: OpEndInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpBranch %exit
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, EarlyReturnBetweenBufferAccesses) {
  const std::string text = kEarlyExitHeader + kEarlyExitBuffer + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpBeginInvocationInterlockEXT
        %ptr = OpAccessChain %data_ptr %rov %zero %zero
          %d = OpLoad %v4float %ptr
               OpEndInvocationInterlockEXT
               OpSelectionMerge %exit None
; CHECK: OpBranchConditional %{{\w+}} [[early:%\w+]] [[write:%\w+]]
               OpBranchConditional %cond %early %write
; CHECK: [[early]] = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
      %early = OpLabel
               OpBranch %exit
      %write = OpLabel
               OpBeginInvocationInterlockEXT
               OpStore %ptr %d
               OpEndInvocationInterlockEXT
               OpBranch %exit
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, DiscardBeforeImageAccess) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpSelectionMerge %read None
; CHECK: OpBranchConditional %{{\w+}} [[discard:%\w+]] [[read:%\w+]]
               OpBranchConditional %cond %discard %read
; CHECK: [[discard]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpKill
    %discard = OpLabel
               OpKill
       %read = OpLabel
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
          %d = OpImageRead %v4float %img1 %coord
               OpEndInvocationInterlockEXT
               OpBranch %write
      %write = OpLabel
               OpBeginInvocationInterlockEXT
       %img2 = OpLoad %image %rov
               OpImageWrite %img2 %coord %d
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, LoopBreakBetweenImageAccesses) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
               OpBranch %header
     %header = OpLabel
               OpLoopMerge %exit %continue None
               OpBranch %read
       %read = OpLabel
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
          %d = OpImageRead %v4float %img1 %coord
               OpEndInvocationInterlockEXT
               OpSelectionMerge %write None
               OpBranchConditional %cond %break %write
      %break = OpLabel
; CHECK: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
               OpBranch %exit
      %write = OpLabel
               OpBeginInvocationInterlockEXT
       %img2 = OpLoad %image %rov
               OpImageWrite %img2 %coord %d
               OpEndInvocationInterlockEXT
               OpBranch %continue
   %continue = OpLabel
               OpBranch %header
       %exit = OpLabel
; CHECK: OpReturn
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, SkipEdgeToMergeIsSplit) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpSelectionMerge %exit None
; CHECK: OpBranchConditional %{{\w+}} [[access:%\w+]] [[skip:%\w+]]
               OpBranchConditional %cond %access %exit
; CHECK-NEXT: [[skip]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[exit:%\w+]]
; CHECK-NEXT: [[access]] = OpLabel
     %access = OpLabel
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
          %d = OpImageRead %v4float %img1 %coord
               OpEndInvocationInterlockEXT
               OpBranch %exit
; CHECK: [[exit]] = OpLabel
; CHECK-NEXT: OpReturn
       %exit = OpLabel
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, BeginInBranchEndInMerge) {
  const std::string text = kEarlyExitHeader + kEarlyExitImage + R"(
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpSelectionMerge %merge None
; CHECK: OpBranchConditional %{{\w+}} [[read:%\w+]] [[other:%\w+]]
               OpBranchConditional %cond %read %other
       %read = OpLabel
               OpBeginInvocationInterlockEXT
       %img1 = OpLoad %image %rov
          %d = OpImageRead %v4float %img1 %coord
               OpBranch %merge
; CHECK: [[other]] = OpLabel
; CHECK-NEXT: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpBranch [[merge:%\w+]]
      %other = OpLabel
               OpBranch %merge
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK-NOT: OpBeginInvocationInterlockEXT
; CHECK: OpImageWrite
; CHECK-NEXT: OpEndInvocationInterlockEXT
      %merge = OpLabel
       %img2 = OpLoad %image %rov
               OpImageWrite %img2 %coord %value
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

TEST_F(InterlockInvocationPlacementTest, SecondEntryPointAfterSplit) {
  // Placing the end for %main1 splits an edge out of a block with a begin.
  // %main2 must be analyzed without walking into that new block.
  const std::string text = R"(
               OpCapability Shader
               OpCapability FragmentShaderPixelInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main1 "main1"
               OpEntryPoint Fragment %main2 "main2"
               OpExecutionMode %main1 OriginUpperLeft
               OpExecutionMode %main1 PixelInterlockOrderedEXT
               OpExecutionMode %main2 OriginUpperLeft
               OpExecutionMode %main2 PixelInterlockOrderedEXT
       %void = OpTypeVoid
       %bool = OpTypeBool
       %cond = OpUndef %bool
       %func = OpTypeFunction %void
      %main1 = OpFunction %void None %func
      %entry = OpLabel
               OpBeginInvocationInterlockEXT
               OpSelectionMerge %merge None
; CHECK: OpBranchConditional %{{\w+}} [[then:%\w+]] [[skip:%\w+]]
               OpBranchConditional %cond %then %merge
; CHECK-NEXT: [[skip]] = OpLabel
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpBranch
       %then = OpLabel
               OpEndInvocationInterlockEXT
               OpBranch %merge
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
      %main2 = OpFunction %void None %func
     %entry2 = OpLabel
; CHECK: OpBeginInvocationInterlockEXT
; CHECK-NEXT: OpEndInvocationInterlockEXT
; CHECK-NEXT: OpReturn
               OpBeginInvocationInterlockEXT
               OpEndInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<InvocationInterlockPlacementPass>(text, true);
  CheckExactlyOneInterlockPerPath(context());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
