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

using FixMaliSpecConstantCompositeTest = PassTest<::testing::Test>;

TEST_F(FixMaliSpecConstantCompositeTest, PromotesBooleanVector) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
      %v2bool = OpTypeVector %bool 2
      %false = OpConstantFalse %bool
   %spec_true = OpSpecConstantTrue %bool
  %composite = OpConstantComposite %v2bool %false %spec_true
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<FixMaliSpecConstantCompositePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpSpecConstantComposite"),
            std::string::npos);
}

TEST_F(FixMaliSpecConstantCompositeTest, LeavesPlainConstants) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
      %v2bool = OpTypeVector %bool 2
      %false = OpConstantFalse %bool
       %true = OpConstantTrue %bool
  %composite = OpConstantComposite %v2bool %false %true
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<FixMaliSpecConstantCompositePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(FixMaliSpecConstantCompositeTest, PromotesMixedOperands) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
     %v3bool = OpTypeVector %bool 3
      %false = OpConstantFalse %bool
       %true = OpConstantTrue %bool
 %spec_false = OpSpecConstantFalse %bool
  %composite = OpConstantComposite %v3bool %false %true %spec_false
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<FixMaliSpecConstantCompositePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpSpecConstantComposite"),
            std::string::npos);
}

TEST_F(FixMaliSpecConstantCompositeTest, PromotesSpecConstantOp) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
     %v2bool = OpTypeVector %bool 2
      %false = OpConstantFalse %bool
 %spec_false = OpSpecConstantFalse %bool
    %spec_op = OpSpecConstantOp %bool LogicalNot %spec_false
  %composite = OpConstantComposite %v2bool %false %spec_op
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<FixMaliSpecConstantCompositePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpSpecConstantComposite"),
            std::string::npos);
}

TEST_F(FixMaliSpecConstantCompositeTest, LeavesIntegerVector) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
  %composite = OpConstantComposite %v2uint %uint_1 %uint_2
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<FixMaliSpecConstantCompositePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
