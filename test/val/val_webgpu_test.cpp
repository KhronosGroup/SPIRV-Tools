// Copyright (c) 2018 Google Inc.
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

// Validation tests for WebGPU env specific checks

#include <string>

#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using testing::HasSubstr;

using ValidateWebGPU = spvtest::ValidateBase<bool>;

TEST_F(ValidateWebGPU, OpUndefIsDisallowed) {
  std::string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    %float = OpTypeFloat 32
    %1 = OpUndef %float
)";

  CompileSuccessfully(spirv);

  // Control case: OpUndef is allowed in SPIR-V 1.3
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));

  // Control case: OpUndef is disallowed in the WebGPU env
  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(), HasSubstr("OpUndef is disallowed"));
}

TEST_F(ValidateWebGPU, OpNameIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpName %1 "foo"
%1 = OpTypeFloat 32
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpName %foo \"foo\"\n"));
}

TEST_F(ValidateWebGPU, OpMemberNameIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpMemberName %2 0 "foo"
%1 = OpTypeFloat 32
%2 = OpTypeStruct %1
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpMemberName %_struct_1 0 "
                        "\"foo\"\n"));
}

TEST_F(ValidateWebGPU, OpSourceIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpSource GLSL 450
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpSource GLSL 450\n"));
}

// OpSourceContinued does not have a test case, because it requires being
// preceded by OpSource, which will cause a validation error.

TEST_F(ValidateWebGPU, OpSourceExtensionIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpSourceExtension "bar"
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpSourceExtension "
                        "\"bar\"\n"));
}

TEST_F(ValidateWebGPU, OpStringIsDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
%1 = OpString "foo"
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  %1 = OpString \"foo\"\n"));
}

// OpLine does not have a test case, because it requires being preceded by
// OpString, which will cause a validation error.

TEST_F(ValidateWebGPU, OpNoLineDisallowed) {
  std::string spirv = R"(
     OpCapability Shader
     OpCapability Linkage
     OpMemoryModel Logical GLSL450
     OpNoLine
)";

  CompileSuccessfully(spirv);

  EXPECT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_WEBGPU_0));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Debugging instructions are not allowed in the WebGPU "
                        "execution environment.\n  OpNoLine\n"));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
