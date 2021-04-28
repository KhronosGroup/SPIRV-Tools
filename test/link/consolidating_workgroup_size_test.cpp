// Copyright (c) 2021 Pierre Moreau
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

#include "gmock/gmock.h"
#include "test/link/linker_fixture.h"

namespace spvtools {
namespace {

using ::testing::HasSubstr;
using ConsolidatingWorkgroupSize = spvtest::LinkerTest;

TEST_F(ConsolidatingWorkgroupSize, Default) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint GLCompute %1 "main"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 None %3
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body}, &linked_binary))
      << GetErrorMessage();

  const std::string expected_res =
      R"(OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpModuleProcessed "Linked by SPIR-V Tools Linker"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %2 None %3
OpFunctionEnd
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  EXPECT_EQ(SPV_SUCCESS, Disassemble(linked_binary, &res_body))
      << GetErrorMessage();
  EXPECT_EQ(expected_res, res_body);
}

TEST_F(ConsolidatingWorkgroupSize, NoLocalSize) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpDecorate %2 BuiltIn WorkgroupSize
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 0
%6 = OpTypeVector %5 3
%7 = OpConstant %5 7
%8 = OpConstant %5 8
%9 = OpConstant %5 9
%2 = OpConstantComposite %6 %7 %8 %9
%1 = OpFunction %3 None %4
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body}, &linked_binary))
      << GetErrorMessage();

  const std::string expected_res =
      R"(OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 7 8 9
OpModuleProcessed "Linked by SPIR-V Tools Linker"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%4 = OpTypeInt 32 0
%5 = OpTypeVector %4 3
%6 = OpConstant %4 7
%7 = OpConstant %4 8
%8 = OpConstant %4 9
%9 = OpConstantComposite %5 %6 %7 %8
%1 = OpFunction %2 None %3
OpFunctionEnd
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  EXPECT_EQ(SPV_SUCCESS, Disassemble(linked_binary, &res_body))
      << GetErrorMessage();
  EXPECT_EQ(expected_res, res_body);
}

TEST_F(ConsolidatingWorkgroupSize, ConflictWithBuiltin) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 1 2 3
OpDecorate %2 BuiltIn WorkgroupSize
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 0
%6 = OpTypeVector %5 3
%7 = OpConstant %5 7
%8 = OpConstant %5 8
%9 = OpConstant %5 9
%2 = OpConstantComposite %6 %7 %8 %9
%1 = OpFunction %3 None %4
OpFunctionEnd
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, AssembleAndLink({body}, &linked_binary));
  EXPECT_THAT(
      GetErrorMessage(),
      HasSubstr(
          "entry point %1 already has a specified LocalSize, (1,2,3), which "
          "differs from the specified constant WorkgroupSize, (7,8,9)"));
}

TEST_F(ConsolidatingWorkgroupSize, SameAsBuiltin) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 7 8 9
OpDecorate %2 BuiltIn WorkgroupSize
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 0
%6 = OpTypeVector %5 3
%7 = OpConstant %5 7
%8 = OpConstant %5 8
%9 = OpConstant %5 9
%2 = OpConstantComposite %6 %7 %8 %9
%1 = OpFunction %3 None %4
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body}, &linked_binary))
      << GetErrorMessage();

  const std::string expected_res = R"(OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpExecutionMode %1 LocalSize 7 8 9
OpModuleProcessed "Linked by SPIR-V Tools Linker"
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%4 = OpTypeInt 32 0
%5 = OpTypeVector %4 3
%6 = OpConstant %4 7
%7 = OpConstant %4 8
%8 = OpConstant %4 9
%9 = OpConstantComposite %5 %6 %7 %8
%1 = OpFunction %2 None %3
OpFunctionEnd
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  EXPECT_EQ(SPV_SUCCESS, Disassemble(linked_binary, &res_body))
      << GetErrorMessage();
  EXPECT_EQ(expected_res, res_body);
}

TEST_F(ConsolidatingWorkgroupSize, MultiEntryPointsWithoutBuiltIn) {
  const std::string body = R"(
OpCapability Shader
OpCapability Kernel
OpEntryPoint GLCompute %1 "main"
OpEntryPoint Kernel %2 "main"
OpExecutionMode %1 LocalSize 7 8 9
OpExecutionMode %2 LocalSize 1 2 3
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%1 = OpFunction %3 None %4
OpFunctionEnd
%2 = OpFunction %3 None %4
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_SUCCESS, AssembleAndLink({body}, &linked_binary))
      << GetErrorMessage();
}

TEST_F(ConsolidatingWorkgroupSize, MultiEntryPointsWithBuiltIn) {
  const std::string body = R"(
OpCapability Shader
OpCapability Kernel
OpEntryPoint GLCompute %1 "main"
OpEntryPoint Kernel %2 "main"
OpExecutionMode %1 LocalSize 7 8 9
OpDecorate %3 BuiltIn WorkgroupSize
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 0
%7 = OpTypeVector %6 3
%8 = OpConstant %6 7
%9 = OpConstant %6 8
%10 = OpConstant %6 9
%3 = OpConstantComposite %7 %8 %9 %10
%1 = OpFunction %4 None %5
OpFunctionEnd
%2 = OpFunction %4 None %5
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, AssembleAndLink({body}, &linked_binary));
  EXPECT_THAT(
      GetErrorMessage(),
      HasSubstr("More than one GLCompute or Kernel entry point was found"));
}

TEST_F(ConsolidatingWorkgroupSize, MultiWorkgroupSizes) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint GLCompute %1 "main"
OpDecorate %2 BuiltIn WorkgroupSize
OpDecorate %3 BuiltIn WorkgroupSize
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%6 = OpTypeInt 32 0
%7 = OpTypeVector %6 3
%8 = OpConstant %6 7
%9 = OpConstant %6 8
%10 = OpConstant %6 9
%2 = OpConstantComposite %7 %8 %9 %10
%3 = OpConstantComposite %7 %10 %8 %9
%1 = OpFunction %4 None %5
OpFunctionEnd
)";
  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, AssembleAndLink({body}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("More than one WorkgroupSize constant was found"));
}

TEST_F(ConsolidatingWorkgroupSize, NoGLComputeNorKernel) {
  const std::string body = R"(
OpCapability Shader
OpEntryPoint Vertex %1 "main"
OpDecorate %2 BuiltIn WorkgroupSize
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%5 = OpTypeInt 32 0
%6 = OpTypeVector %5 3
%7 = OpConstant %5 7
%8 = OpConstant %5 8
%9 = OpConstant %5 9
%2 = OpConstantComposite %6 %7 %8 %9
%1 = OpFunction %3 None %4
OpFunctionEnd
)";

  spvtest::Binary linked_binary;
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, AssembleAndLink({body}, &linked_binary));
  EXPECT_THAT(GetErrorMessage(),
              HasSubstr("No GLCompute or Kernel entry points found"));
}

}  // namespace
}  // namespace spvtools
