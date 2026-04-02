// Copyright (c) 2020 Google LLC
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

// Tests for OpExtension validator rules.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_target_env.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::HasSubstr;
using ::testing::Values;
using ::testing::ValuesIn;

using ValidateSpvKHRAbort = spvtest::ValidateBase<bool>;

TEST_F(ValidateSpvKHRAbort, Valid) {
  const std::string str = R"(
OpCapability Shader
OpCapability AbortKHR
OpExtension "SPV_KHR_abort"
OpMemoryModel Logical Simple
OpEntryPoint GLCompute %main "main"
%msg = OpString "abort message"
%void    = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpAbortKHR %msg
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateSpvKHRAbort, RequiresCapability) {
  const std::string str = R"(
OpCapability Shader
OpExtension "SPV_KHR_abort"
OpMemoryModel Logical Simple
OpEntryPoint GLCompute %main "main"
%msg = OpString "abort message"
%void    = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpAbortKHR %msg
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Opcode AbortKHR requires one of these capabilities: AbortKHR"));
}

TEST_F(ValidateSpvKHRAbort, RequiresExtention) {
  const std::string str = R"(
OpCapability Shader
OpCapability AbortKHR
OpMemoryModel Logical Simple
OpEntryPoint GLCompute %main "main"
%msg = OpString "abort message"
%void    = OpTypeVoid
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpAbortKHR %msg
OpReturn
OpFunctionEnd
)";
  CompileSuccessfully(str.c_str());
  EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("1st operand of Capability: operand AbortKHR(5120) "
                        "requires one of these extensions: SPV_KHR_abort"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
