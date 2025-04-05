// Copyright (c) 2025 Google Inc.
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

// Tests for invalid types.

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
using ::testing::Not;
using ::testing::Values;

using ValidateInvalidType = spvtest::ValidateBase<bool>;

std::string GenerateBFloatCode(const std::string& main_body) {
  const std::string prefix =
      R"(
OpCapability Shader
OpCapability BFloat16TypeKHR
OpCapability AtomicFloat16AddEXT
OpCapability GroupNonUniformShuffle
OpExtension "SPV_EXT_shader_atomic_float16_add"
OpExtension "SPV_KHR_bfloat16"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 450
OpName %main "main"
%void = OpTypeVoid
%bfloat16 = OpTypeFloat 16 BFloat16KHR
%func = OpTypeFunction %void
%u32 = OpTypeInt 32 0
%u1 = OpConstant %u32 1
%u0 = OpConstant %u32 0
%u3 = OpConstant %u32 3
%bf16_1 = OpConstant %bfloat16 1
%_ptr_Function_bfloat16 = OpTypePointer Function %bfloat16
%v2bfloat16 = OpTypeVector %bfloat16 2
%_ptr_Function_v2bfloat16 = OpTypePointer Function %v2bfloat16
%bf16_ptr = OpTypePointer Workgroup %bfloat16
%bf16_var = OpVariable %bf16_ptr Workgroup
%main = OpFunction %void None %func
%main_entry = OpLabel)";

  const std::string suffix =
      R"(
OpReturn
OpFunctionEnd)";

  return prefix + main_body + suffix;
}

TEST_F(ValidateInvalidType, Bfloat16InvalidArithmeticInstruction) {
  const std::string body = R"(
%v1 = OpVariable %_ptr_Function_bfloat16 Function
%v2 = OpVariable %_ptr_Function_bfloat16 Function
%12 = OpLoad %bfloat16 %v1
%14 = OpLoad %bfloat16 %v2
%15 = OpFMul %bfloat16 %12 %14
)";

  CompileSuccessfully(GenerateBFloatCode(body).c_str(), SPV_ENV_VULKAN_1_3);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_6));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("FMul doesn't support BFloat16 type."));
}

TEST_F(ValidateInvalidType, Bfloat16InvalidAtomicInstruction) {
  const std::string body = R"(
%val1 = OpAtomicFAddEXT %bfloat16 %bf16_var %u1 %u0 %bf16_1
)";

  CompileSuccessfully(GenerateBFloatCode(body).c_str(), SPV_ENV_VULKAN_1_3);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_6));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFAddEXT doesn't support BFloat16 type."));
}

TEST_F(ValidateInvalidType, Bfloat16InvalidGroupNonUniformShuffle) {
  const std::string body = R"(
%val1 = OpGroupNonUniformShuffle %bfloat16 %u3 %bf16_1 %u0
)";

  CompileSuccessfully(GenerateBFloatCode(body).c_str(), SPV_ENV_VULKAN_1_3);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA,
            ValidateInstructions(SPV_ENV_UNIVERSAL_1_6));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("GroupNonUniformShuffle doesn't support BFloat16 type."));
}

}  // namespace
}  // namespace val
}  // namespace spvtools
