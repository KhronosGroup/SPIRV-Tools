// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "../TestFixture.h"
#include "UnitSPIRV.h"

#include "spirv/1.0/spirv.h"
#include "gmock/gmock.h"

using std::vector;
using spvtest::MakeInstruction;
using spvtest::Concatenate;
using spvtest::ScopedContext;

spv_target_env env = SPV_ENV_UNIVERSAL_1_0;

spv_result_t TestApi(const vector<uint32_t>& bytecode) {
  spv_diagnostic const_binary_diagnostic;
  spv_diagnostic separate_diagnostic;
  std::unique_ptr<spv_const_binary_t> bin(
      new spv_const_binary_t{bytecode.data(), bytecode.size()});

  spv_result_t const_binary_result = spvValidate(
      ScopedContext(env).context, bin.get(), &const_binary_diagnostic);

  spv_result_t separate_result =
      spvValidateBinary(ScopedContext(env).context,
                        bytecode.data(), bytecode.size(), &separate_diagnostic);
  EXPECT_EQ(const_binary_result, separate_result);
  if (const_binary_diagnostic) {
    EXPECT_STREQ(const_binary_diagnostic->error, separate_diagnostic->error);
    EXPECT_EQ(const_binary_diagnostic->position.column,
              separate_diagnostic->position.column);
    EXPECT_EQ(const_binary_diagnostic->position.index,
              separate_diagnostic->position.index);
    EXPECT_EQ(const_binary_diagnostic->position.line,
              separate_diagnostic->position.line);
    EXPECT_EQ(const_binary_diagnostic->isTextSource,
              separate_diagnostic->isTextSource);
  }
  return const_binary_result;
}

TEST(ValidateAPI, BinaryAPISuccess) {
  uint32_t bound = 0;

  auto bytecode = Concatenate(
      {{SpvMagicNumber, SpvVersion,
        SPV_GENERATOR_WORD(SPV_GENERATOR_KHRONOS_ASSEMBLER, 0), bound, 0},
       MakeInstruction(SpvOpCapability, {SpvCapabilityKernel}),
       MakeInstruction(SpvOpCapability, {SpvCapabilityAddresses}),
       MakeInstruction(SpvOpMemoryModel,
                       {SpvAddressingModelPhysical64, SpvMemoryModelOpenCL})});
  ASSERT_EQ(SPV_SUCCESS, TestApi(bytecode));
}

TEST(ValidateAPI, BinaryAPIBad) {
  char str[] = R"(
          OpCapability Shader
          OpMemoryModel Logical GLSL450
          OpName %missing "missing"
%voidt  = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%func   = OpFunction %vfunct None %missing
%flabel = OpLabel
          OpReturn
          OpFunctionEnd
    )";

  spv_diagnostic diagnostic = nullptr;
  spv_binary binary;
  EXPECT_EQ(SPV_SUCCESS,
            spvTextToBinary(ScopedContext(env).context, str,
                            sizeof(str), &binary, &diagnostic));

  vector<uint32_t> words(binary->code, binary->code + binary->wordCount);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, TestApi(words));
}
