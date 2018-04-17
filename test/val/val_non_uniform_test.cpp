// Copyright (c) 2018 Google LLC.
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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using ::testing::HasSubstr;
using ::testing::Combine;
using ::testing::Values;
using ::testing::ValuesIn;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "GLCompute") {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformVote
)";

  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0

%true = OpConstantTrue %bool
%false = OpConstantFalse %bool

%cross_device = OpConstant %u32 0
%device = OpConstant %u32 1
%workgroup = OpConstant %u32 2
%subgroup = OpConstant %u32 3
%invocation = OpConstant %u32 4

%reduce = OpConstant %u32 0
%inclusive_scan = OpConstant %u32 1
%exclusive_scan = OpConstant %u32 2
%clustered_reduce = OpConstant %u32 3

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

using VulkanGroupNonUniformScope = spvtest::ValidateBase<
    std::tuple<std::string, std::string, std::string, std::string>>;

TEST_P(VulkanGroupNonUniformScope, Scopes) {
  std::string opcode = std::get<0>(GetParam());
  std::string type = std::get<1>(GetParam());
  std::string execution_scope = std::get<2>(GetParam());
  std::string args = std::get<3>(GetParam());

  std::ostringstream sstr;
  sstr << "%result = " << opcode << " ";
  sstr << type << " ";
  sstr << execution_scope << " ";
  sstr << args << "\n";

  CompileSuccessfully(GenerateShaderCode(sstr.str()), SPV_ENV_VULKAN_1_1);
  spv_result_t result = ValidateInstructions(SPV_ENV_VULKAN_1_1);
  if (execution_scope == "%subgroup") {
    ASSERT_EQ(SPV_SUCCESS, result);
  } else {
    ASSERT_EQ(SPV_ERROR_INVALID_DATA, result);
    EXPECT_THAT(
        getDiagnosticString(),
        HasSubstr(
            "in Vulkan environment Execution scope is limited to Subgroup"));
  }
}

INSTANTIATE_TEST_CASE_P(GroupNonUniformElect, VulkanGroupNonUniformScope,
                        Combine(Values("OpGroupNonUniformElect"),
                                Values("%bool"),
                                Values("%cross_device", "%device", "%workgroup",
                                       "%subgroup", "%invocation"),
                                Values("")));

INSTANTIATE_TEST_CASE_P(GroupNonUniformVote, VulkanGroupNonUniformScope,
                        Combine(Values("OpGroupNonUniformAll",
                                       "OpGroupNonUniformAny",
                                       "OpGroupNonUniformAllEqual"),
                                Values("%bool"),
                                Values("%cross_device", "%device", "%workgroup",
                                       "%subgroup", "%invocation"),
                                Values("%true")));

}  // anonymous namespace
