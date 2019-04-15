// Copyright (c) 2019 Google LLC.
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

#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using DecomposeInitializedVariablesTest = PassTest<::testing::Test>;

void operator+=(std::vector<const char*>& lhs,
                const std::vector<const char*> rhs) {
  for (auto elem : rhs) lhs.push_back(elem);
}

std::vector<const char*> header = {
    "OpCapability Shader",
    "OpCapability VulkanMemoryModelKHR",
    "OpExtension \"SPV_KHR_vulkan_memory_model\"",
    "OpMemoryModel Logical VulkanKHR",
    "OpEntryPoint Vertex %1 \"shader\"",
    "%uint = OpTypeInt 32 0",
    "%3 = OpConstantNull %uint",
    "%void = OpTypeVoid",
    "%5 = OpTypeFunction %void"};

std::string GetFunctionTest(std::vector<const char*> body) {
  auto result = header;
  result += {"%_ptr_Function_uint = OpTypePointer Function %uint",
             "%1 = OpFunction %void None %5", "%7 = OpLabel"};
  result += body;
  result += {"OpReturn", "OpFunctionEnd"};
  return JoinAllInsts(result);
}

TEST_F(DecomposeInitializedVariablesTest, FunctionChanged) {
  std::string input =
      GetFunctionTest({"%8 = OpVariable %_ptr_Function_uint Function %3"});
  std::string expected = GetFunctionTest(
      {"%8 = OpVariable %_ptr_Function_uint Function", "OpStore %8 %3"});

  SinglePassRunAndCheck<DecomposeInitializedVariablesPass>(
      input, expected, /* skip_nop = */ false);
}

TEST_F(DecomposeInitializedVariablesTest, FunctionUnchanged) {
  std::string input =
      GetFunctionTest({"%8 = OpVariable %_ptr_Function_uint Function"});

  SinglePassRunAndCheck<DecomposeInitializedVariablesPass>(
      input, input, /* skip_nop = */ false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
