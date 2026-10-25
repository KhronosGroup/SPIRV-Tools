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

using MaliOptimizationBarrierTest = PassTest<::testing::Test>;

TEST_F(MaliOptimizationBarrierTest, AfterConstantShift) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %label = OpLabel
      %shift = OpShiftLeftLogical %uint %uint_4 %uint_2
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpBitFieldInsert"), std::string::npos);
}

TEST_F(MaliOptimizationBarrierTest, LeavesVariableShift) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
  %ptr_uint = OpTypePointer Function %uint
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %label = OpLabel
        %var = OpVariable %ptr_uint Function
        %val = OpLoad %uint %var
        %amt = OpLoad %uint %var
      %shift = OpShiftLeftLogical %uint %val %amt
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliOptimizationBarrierTest, AddsBarrierToVectorShift) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
     %uint_4 = OpConstant %uint 4
     %uint_2 = OpConstant %uint 2
    %const_a = OpConstantComposite %v2uint %uint_4 %uint_4
    %const_b = OpConstantComposite %v2uint %uint_2 %uint_2
       %main = OpFunction %void None %fn
      %label = OpLabel
      %shift = OpShiftRightLogical %v2uint %const_a %const_b
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpBitFieldInsert"), std::string::npos);
}

TEST_F(MaliOptimizationBarrierTest, AddsBarrierToSixtyFourBitShift) {
  const std::string text = R"(
               OpCapability Shader
               OpCapability Int64
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %ulong = OpTypeInt 64 0
       %uint = OpTypeInt 32 0
    %ulong_4 = OpConstant %ulong 4
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %label = OpLabel
      %shift = OpShiftLeftLogical %ulong %ulong_4 %uint_2
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpBitFieldInsert"), std::string::npos);
}

TEST_F(MaliOptimizationBarrierTest, AddsBarrierToSignedShift) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
      %int_4 = OpConstant %int -4
     %uint_1 = OpConstant %uint 1
       %main = OpFunction %void None %fn
      %label = OpLabel
      %shift = OpShiftRightArithmetic %int %int_4 %uint_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpBitFieldInsert"), std::string::npos);
}

TEST_F(MaliOptimizationBarrierTest, LeavesNonShift) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_4 = OpConstant %uint 4
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %label = OpLabel
        %add = OpIAdd %uint %uint_4 %uint_2
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliOptimizationBarrierPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(std::get<0>(result).find("OpBitFieldInsert"), std::string::npos);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
