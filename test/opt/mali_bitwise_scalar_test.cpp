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

using MaliBitwiseScalarTest = PassTest<::testing::Test>;

size_t CountOccurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + 1)) {
    ++count;
  }
  return count;
}

TEST_F(MaliBitwiseScalarTest, ScalarizesVec3BitwiseAnd) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
      %vec_a = OpConstantComposite %v3uint %uint_1 %uint_3 %uint_1
      %vec_b = OpConstantComposite %v3uint %uint_3 %uint_1 %uint_3
       %main = OpFunction %void None %fn
      %label = OpLabel
        %res = OpBitwiseAnd %v3uint %vec_a %vec_b
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpCompositeExtract"), std::string::npos);
  EXPECT_NE(std::get<0>(result).find("OpCompositeConstruct"), std::string::npos);
}

TEST_F(MaliBitwiseScalarTest, ScalarizesVec2BitwiseNot) {
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
      %vec_a = OpConstantComposite %v2uint %uint_1 %uint_1
       %main = OpFunction %void None %fn
      %label = OpLabel
        %res = OpNot %v2uint %vec_a
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::get<0>(result).find("OpCompositeExtract"), std::string::npos);
  EXPECT_NE(std::get<0>(result).find("OpCompositeConstruct"), std::string::npos);
}

TEST_F(MaliBitwiseScalarTest, LeavesScalarBitwiseUnchanged) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
       %main = OpFunction %void None %fn
      %label = OpLabel
        %res = OpBitwiseXor %uint %uint_1 %uint_3
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliBitwiseScalarTest, ScalarisesConstructedOperand) {
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
       %main = OpFunction %void None %fn
      %label = OpLabel
          %v = OpCompositeConstruct %v2uint %uint_1 %uint_2
      %splat = OpCompositeConstruct %v2uint %uint_1 %uint_1
          %r = OpBitwiseOr %v2uint %v %splat
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_NE(std::string::npos, disassembly.find("OpCompositeExtract"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpBitwiseOr"));
}

TEST_F(MaliBitwiseScalarTest, LeavesLoadedOperands) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
%ptr_v2uint = OpTypePointer Private %v2uint
      %a_ptr = OpVariable %ptr_v2uint Private
      %b_ptr = OpVariable %ptr_v2uint Private
       %main = OpFunction %void None %fn
      %label = OpLabel
          %a = OpLoad %v2uint %a_ptr
          %b = OpLoad %v2uint %b_ptr
          %r = OpBitwiseAnd %v2uint %a %b
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliBitwiseScalarTest, RewriteAllTakesLoadedOperands) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
%ptr_v2uint = OpTypePointer Private %v2uint
      %a_ptr = OpVariable %ptr_v2uint Private
      %b_ptr = OpVariable %ptr_v2uint Private
       %main = OpFunction %void None %fn
      %label = OpLabel
          %a = OpLoad %v2uint %a_ptr
          %b = OpLoad %v2uint %b_ptr
          %r = OpBitwiseAnd %v2uint %a %b
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true,
      /*rewrite_all=*/true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpBitwiseAnd"));
}

TEST_F(MaliBitwiseScalarTest, ScalarisesShiftLeft) {
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
       %main = OpFunction %void None %fn
      %label = OpLabel
          %v = OpCompositeConstruct %v2uint %uint_1 %uint_2
      %shift = OpShiftLeftLogical %v2uint %v %v
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpShiftLeftLogical"));
  // Two source lanes per operand, two operands.
  EXPECT_EQ(4u, CountOccurrences(disassembly, "OpCompositeExtract"));
}

TEST_F(MaliBitwiseScalarTest, ScalarisesNot) {
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
       %main = OpFunction %void None %fn
      %label = OpLabel
          %v = OpCompositeConstruct %v2uint %uint_1 %uint_2
          %r = OpNot %v2uint %v
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpNot"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpCompositeExtract"));
}

TEST_F(MaliBitwiseScalarTest, LooksThroughCopyObject) {
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
       %main = OpFunction %void None %fn
      %label = OpLabel
          %v = OpCompositeConstruct %v2uint %uint_1 %uint_2
       %copy = OpCopyObject %v2uint %v
          %r = OpBitwiseXor %v2uint %copy %v
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpBitwiseXor"));
}

TEST_F(MaliBitwiseScalarTest, ScalarisesVectorShuffleOperand) {
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
       %main = OpFunction %void None %fn
      %label = OpLabel
          %v = OpCompositeConstruct %v2uint %uint_1 %uint_2
       %shuf = OpVectorShuffle %v2uint %v %v 1 0
          %r = OpBitwiseAnd %v2uint %shuf %v
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpBitwiseAnd"));
}

TEST_F(MaliBitwiseScalarTest, ScalarisesConstantVectors) {
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
     %uint_3 = OpConstant %uint 3
          %a = OpConstantComposite %v2uint %uint_1 %uint_2
          %b = OpConstantComposite %v2uint %uint_3 %uint_2
       %main = OpFunction %void None %fn
      %label = OpLabel
          %r = OpBitwiseOr %v2uint %a %b
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpBitwiseOr"));
}

TEST_F(MaliBitwiseScalarTest, LeavesScalarOperations) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %fn
      %label = OpLabel
          %r = OpBitwiseOr %uint %uint_1 %uint_2
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliBitwiseScalarPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(0u, CountOccurrences(std::get<0>(result), "OpCompositeExtract"));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
