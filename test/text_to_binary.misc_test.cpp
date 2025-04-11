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

// Assembler tests for instructions in the "Miscellaneous" section of the
// SPIR-V spec.

#include "test/unit_spirv.h"

#include "gmock/gmock.h"
#include "test/test_fixture.h"

namespace spvtools {
namespace {

using SpirvVector = spvtest::TextToBinaryTest::SpirvVector;
using spvtest::MakeInstruction;
using ::testing::Eq;
using TextToBinaryMisc = spvtest::TextToBinaryTest;

TEST_F(TextToBinaryMisc, OpNop) {
  EXPECT_THAT(CompiledInstructions("OpNop"),
              Eq(MakeInstruction(spv::Op::OpNop, {})));
}

TEST_F(TextToBinaryMisc, OpUndef) {
  const SpirvVector code = CompiledInstructions(R"(%f32 = OpTypeFloat 32
                                                   %u = OpUndef %f32)");
  const uint32_t typeID = 1;
  EXPECT_THAT(code[1], Eq(typeID));
  EXPECT_THAT(Subvector(code, 3),
              Eq(MakeInstruction(spv::Op::OpUndef, {typeID, 2})));
}

TEST_F(TextToBinaryMisc, OpWrong) {
  EXPECT_THAT(CompileFailure(" OpWrong %1 %2"),
              Eq("Invalid Opcode name 'OpWrong'"));
}

TEST_F(TextToBinaryMisc, OpWrongAfterRight) {
  const auto assembly = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpXYZ
)";
  EXPECT_THAT(CompileFailure(assembly), Eq("Invalid Opcode name 'OpXYZ'"));
}

TEST_F(TextToBinaryMisc, OpAbortKHR) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  const auto assembly = R"(
OpExtension "SPV_KHR_abort"
OpCapability AbortKHR
             OpCapability ConstantDataKHR

             OpDecorate %string1_t UTFCodePointsKHR
             OpDecorate %string2_t UTFCodePointsKHR

             OpDecorate %string1_x UTFCodePointsKHR
             OpDecorate %string1_x ArrayStride 1
             OpDecorate %string2_x UTFCodePointsKHR
             OpDecorate %string2_x ArrayStride 1
             OpMemberDecorate %message_x 0 Offset 0
             OpMemberDecorate %message_x 1 Offset 6
             OpMemberDecorate %message_x 2 Offset 8

   %char_t = OpTypeInt 8 0
 %uint32_t = OpTypeInt 32 0
  %str1len = OpConstant %uint32_t 6
%string1_t = OpTypeArray %char_t %str1len
  %string1 = OpConstantDataKHR %string_t "test: "
  %str2len = OpSpecConstant %uint32_t 2
%string2_t = OpTypeArray %char_t %str2len
  %string2 = OpSpecConstantDataKHR %string_t "%u"
%message_t = OpTypeStruct %string1_t %string2_t %uint32_t

%string1_x = OpTypeArray %char_t %str1len
%string2_x = OpTypeArray %char_t %str2len
%message_x = OpTypeStruct %string1_t %string2_t %uint32_t

    %abort = OpLabel
  %message = OpCompositeConstruct %message_t %string1 %string2 %uintval
             OpAbortKHR %message_x %message
)";

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, assembly, strlen(assembly),
                                         &binary, &diagnostic));
  EXPECT_NE(nullptr, binary);
  if (binary) {
    EXPECT_NE(nullptr, binary->code);
    EXPECT_NE(0u, binary->wordCount);
  }
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    ASSERT_TRUE(false);
  }

  spvContextDestroy(context);
}

TEST_F(TextToBinaryMisc, OpAbortKHRInvalidIdName) {
  const auto assembly = R"(
OpAbortKHR "aaa"
)";
  EXPECT_THAT(CompileFailure(assembly), Eq("Expected id to start with %."));
}

TEST_F(TextToBinaryMisc, OpConstantDataKHR) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
  const auto assembly = R"(
             OpCapability ConstantDataKHR

             OpDecorate %string1_t UTFCodePointsKHR
             OpDecorate %string2_t UTFCodePointsKHR

             OpDecorate %string1_x UTFCodePointsKHR
             OpDecorate %string1_x ArrayStride 1
             OpDecorate %string2_x UTFCodePointsKHR
             OpDecorate %string2_x ArrayStride 1
             OpMemberDecorate %message_x 0 Offset 0
             OpMemberDecorate %message_x 1 Offset 6
             OpMemberDecorate %message_x 2 Offset 8

   %char_t = OpTypeInt 8 0
 %uint32_t = OpTypeInt 32 0
  %str1len = OpConstant %uint32_t 6
%string1_t = OpTypeArray %char_t %str1len
  %string1 = OpConstantDataKHR %string_t "test: "
  %str2len = OpSpecConstant %uint32_t 2
%string2_t = OpTypeArray %char_t %str2len
  %string2 = OpSpecConstantDataKHR %string_t "%u"
%message_t = OpTypeStruct %string1_t %string2_t %uint32_t

%string1_x = OpTypeArray %char_t %str1len
%string2_x = OpTypeArray %char_t %str2len
%message_x = OpTypeStruct %string1_t %string2_t %uint32_t

    %abort = OpLabel
  %message = OpCompositeConstruct %message_t %string1 %string2 %uintval
)";

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, assembly, strlen(assembly),
                                         &binary, &diagnostic));
  EXPECT_NE(nullptr, binary);
  if (binary) {
    EXPECT_NE(nullptr, binary->code);
    EXPECT_NE(0u, binary->wordCount);
  }
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    ASSERT_TRUE(false);
  }

  spvContextDestroy(context);
}

}  // namespace
}  // namespace spvtools
