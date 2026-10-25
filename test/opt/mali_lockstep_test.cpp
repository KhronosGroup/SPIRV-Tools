// Copyright (c) 2026 Lee Gao
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
#include <tuple>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using MaliLockstepTest = PassTest<::testing::Test>;

size_t CountOccurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + 1)) {
    ++count;
  }
  return count;
}

const char* kLockstepModule = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_0 = OpConstant %uint 0
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpULessThanEqual %bool %uint_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

TEST_F(MaliLockstepTest, MakesStridedLoopUniform) {
  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      kLockstepModule, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpPhi"));
}

TEST_F(MaliLockstepTest, LeavesLoopWithoutWorkgroupAccess) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_0 = OpConstant %uint 0
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_private_uint = OpTypePointer Private %uint
         %x = OpVariable %ptr_private_uint Private
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %x
               OpStore %x %val
       %cond = OpULessThanEqual %bool %uint_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliLockstepTest, UsesUGreaterThanEqual) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_0 = OpConstant %uint 0
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpUGreaterThanEqual %bool %i %uint_64
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpPhi"));
}

TEST_F(MaliLockstepTest, HandlesSixtyFourBitInduction) {
  const std::string text = R"(
               OpCapability Shader
               OpCapability Int64
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
      %ulong = OpTypeInt 64 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
    %ulong_32 = OpConstant %ulong 32
    %ulong_64 = OpConstant %ulong 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
      %tid64 = OpUConvert %ulong %tid
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %ulong %tid64 %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpULessThanEqual %bool %ulong_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %ulong %i %ulong_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpPhi"));
}

TEST_F(MaliLockstepTest, IgnoresWorkgroupAccessOutsideLoop) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
     %outside = OpLoad %uint %wg
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
       %cond = OpULessThanEqual %bool %uint_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliLockstepTest, LeavesNonStridedIncrement) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_1 = OpConstant %uint 1
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpULessThanEqual %bool %uint_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_1
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliLockstepTest, PreservesOriginalComparisonUses) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpULessThanEqual %bool %uint_64 %i
       %side = OpSelect %uint %cond %uint_1 %uint_2
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  // The original comparison stays for %side, and a new one feeds the branch.
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpULessThanEqual"));
  EXPECT_NE(std::string::npos, disassembly.find("OpSelect"));
}

TEST_F(MaliLockstepTest, DetectsAtomicWorkgroupAccess) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 32 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
    %uint_32 = OpConstant %uint 32
    %uint_64 = OpConstant %uint 64
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
     %atomic = OpAtomicIAdd %uint %wg %uint_2 %uint_0 %uint_1
       %cond = OpULessThanEqual %bool %uint_64 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_32
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpPhi"));
}

TEST_F(MaliLockstepTest, LeavesSingleInvocationWorkgroup) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationId
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %gl_LocalInvocationId BuiltIn LocalInvocationId
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%ptr_input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationId = OpVariable %ptr_input_v3uint Input
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
%ptr_wg_uint = OpTypePointer Workgroup %uint
         %wg = OpVariable %ptr_wg_uint Workgroup
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %liid = OpLoad %v3uint %gl_LocalInvocationId
        %tid = OpCompositeExtract %uint %liid 0
               OpBranch %header
     %header = OpLabel
          %i = OpPhi %uint %tid %entry %inext %continue
               OpLoopMerge %merge %continue None
               OpBranch %body
       %body = OpLabel
        %val = OpLoad %uint %wg
               OpStore %wg %val
       %cond = OpULessThanEqual %bool %uint_2 %i
               OpSelectionMerge %exit None
               OpBranchConditional %cond %exit %continue
       %exit = OpLabel
               OpBranch %merge
   %continue = OpLabel
      %inext = OpIAdd %uint %i %uint_1
               OpBranch %header
      %merge = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliLockstepPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
