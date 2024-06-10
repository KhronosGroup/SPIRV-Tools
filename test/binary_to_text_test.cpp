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

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "source/spirv_constant.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::AutoText;
using spvtest::ScopedContext;
using spvtest::TextToBinaryTest;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::HasSubstr;

class BinaryToText : public ::testing::Test {
 public:
  BinaryToText()
      : context(spvContextCreate(SPV_ENV_UNIVERSAL_1_0)), binary(nullptr) {}
  ~BinaryToText() override {
    spvBinaryDestroy(binary);
    spvContextDestroy(context);
  }

  void SetUp() override {
    const char* textStr = R"(
      OpSource OpenCL_C 12
      OpMemoryModel Physical64 OpenCL
      OpSourceExtension "PlaceholderExtensionName"
      OpEntryPoint Kernel %1 "foo"
      OpExecutionMode %1 LocalSizeHint 1 1 1
 %2 = OpTypeVoid
 %3 = OpTypeBool
 %4 = OpTypeInt 8 0
 %5 = OpTypeInt 8 1
 %6 = OpTypeInt 16 0
 %7 = OpTypeInt 16 1
 %8 = OpTypeInt 32 0
 %9 = OpTypeInt 32 1
%10 = OpTypeInt 64 0
%11 = OpTypeInt 64 1
%12 = OpTypeFloat 16
%13 = OpTypeFloat 32
%14 = OpTypeFloat 64
%15 = OpTypeVector %4 2
)";
    spv_text_t text = {textStr, strlen(textStr)};
    spv_diagnostic diagnostic = nullptr;
    spv_result_t error =
        spvTextToBinary(context, text.str, text.length, &binary, &diagnostic);
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }

  void TearDown() override {
    spvBinaryDestroy(binary);
    binary = nullptr;
  }

  // Compiles the given assembly text, and saves it into 'binary'.
  void CompileSuccessfully(std::string text) {
    spvBinaryDestroy(binary);
    binary = nullptr;
    spv_diagnostic diagnostic = nullptr;
    EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(context, text.c_str(), text.size(),
                                           &binary, &diagnostic));
  }

  spv_context context;
  spv_binary binary;
};

TEST_F(BinaryToText, Default) {
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(
      SPV_SUCCESS,
      spvBinaryToText(context, binary->code, binary->wordCount,
                      SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
  printf("%s", text->str);
  spvTextDestroy(text);
}

TEST_F(BinaryToText, Print) {
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(
      SPV_SUCCESS,
      spvBinaryToText(context, binary->code, binary->wordCount,
                      SPV_BINARY_TO_TEXT_OPTION_PRINT, &text, &diagnostic));
  ASSERT_EQ(text, nullptr);
  spvTextDestroy(text);
}

TEST_F(BinaryToText, MissingModule) {
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryToText(context, nullptr, 42, SPV_BINARY_TO_TEXT_OPTION_NONE,
                      &text, &diagnostic));
  EXPECT_THAT(diagnostic->error, Eq(std::string("Missing module.")));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, TruncatedModule) {
  // Make a valid module with zero instructions.
  CompileSuccessfully("");
  EXPECT_EQ(SPV_INDEX_INSTRUCTION, binary->wordCount);

  for (size_t length = 0; length < SPV_INDEX_INSTRUCTION; length++) {
    spv_text text = nullptr;
    spv_diagnostic diagnostic = nullptr;
    EXPECT_EQ(
        SPV_ERROR_INVALID_BINARY,
        spvBinaryToText(context, binary->code, length,
                        SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
    ASSERT_NE(nullptr, diagnostic);
    std::stringstream expected;
    expected << "Module has incomplete header: only " << length
             << " words instead of " << SPV_INDEX_INSTRUCTION;
    EXPECT_THAT(diagnostic->error, Eq(expected.str()));
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, InvalidMagicNumber) {
  CompileSuccessfully("");
  std::vector<uint32_t> damaged_binary(binary->code,
                                       binary->code + binary->wordCount);
  damaged_binary[SPV_INDEX_MAGIC_NUMBER] ^= 123;

  spv_diagnostic diagnostic = nullptr;
  spv_text text;
  EXPECT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryToText(context, damaged_binary.data(), damaged_binary.size(),
                      SPV_BINARY_TO_TEXT_OPTION_NONE, &text, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  std::stringstream expected;
  expected << "Invalid SPIR-V magic number '" << std::hex
           << damaged_binary[SPV_INDEX_MAGIC_NUMBER] << "'.";
  EXPECT_THAT(diagnostic->error, Eq(expected.str()));
  spvDiagnosticDestroy(diagnostic);
}

struct FailedDecodeCase {
  std::string source_text;
  std::vector<uint32_t> appended_instruction;
  std::string expected_error_message;
};

using BinaryToTextFail =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<FailedDecodeCase>>;

TEST_P(BinaryToTextFail, EncodeSuccessfullyDecodeFailed) {
  EXPECT_THAT(EncodeSuccessfullyDecodeFailed(GetParam().source_text,
                                             GetParam().appended_instruction),
              Eq(GetParam().expected_error_message));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidIds, BinaryToTextFail,
    ::testing::ValuesIn(std::vector<FailedDecodeCase>{
        {"", spvtest::MakeInstruction(spv::Op::OpTypeVoid, {0}),
         "Error: Result Id is 0"},
        {"", spvtest::MakeInstruction(spv::Op::OpConstant, {0, 1, 42}),
         "Error: Type Id is 0"},
        {"%1 = OpTypeVoid", spvtest::MakeInstruction(spv::Op::OpTypeVoid, {1}),
         "Id 1 is defined more than once"},
        {"%1 = OpTypeVoid\n"
         "%2 = OpNot %1 %foo",
         spvtest::MakeInstruction(spv::Op::OpNot, {1, 2, 3}),
         "Id 2 is defined more than once"},
        {"%1 = OpTypeVoid\n"
         "%2 = OpNot %1 %foo",
         spvtest::MakeInstruction(spv::Op::OpNot, {1, 1, 3}),
         "Id 1 is defined more than once"},
        // The following are the two failure cases for
        // Parser::setNumericTypeInfoForType.
        {"", spvtest::MakeInstruction(spv::Op::OpConstant, {500, 1, 42}),
         "Type Id 500 is not a type"},
        {"%1 = OpTypeInt 32 0\n"
         "%2 = OpTypeVector %1 4",
         spvtest::MakeInstruction(spv::Op::OpConstant, {2, 3, 999}),
         "Type Id 2 is not a scalar numeric type"},
    }));

INSTANTIATE_TEST_SUITE_P(
    InvalidIdsCheckedDuringLiteralCaseParsing, BinaryToTextFail,
    ::testing::ValuesIn(std::vector<FailedDecodeCase>{
        {"", spvtest::MakeInstruction(spv::Op::OpSwitch, {1, 2, 3, 4}),
         "Invalid OpSwitch: selector id 1 has no type"},
        {"%1 = OpTypeVoid\n",
         spvtest::MakeInstruction(spv::Op::OpSwitch, {1, 2, 3, 4}),
         "Invalid OpSwitch: selector id 1 is a type, not a value"},
        {"%1 = OpConstantTrue !500",
         spvtest::MakeInstruction(spv::Op::OpSwitch, {1, 2, 3, 4}),
         "Type Id 500 is not a type"},
        {"%1 = OpTypeFloat 32\n%2 = OpConstant %1 1.5",
         spvtest::MakeInstruction(spv::Op::OpSwitch, {2, 3, 4, 5}),
         "Invalid OpSwitch: selector id 2 is not a scalar integer"},
    }));

TEST_F(TextToBinaryTest, OneInstruction) {
  const std::string input = "OpSource OpenCL_C 12\n";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

// Exercise the case where an operand itself has operands.
// This could detect problems in updating the expected-set-of-operands
// list.
TEST_F(TextToBinaryTest, OperandWithOperands) {
  const std::string input = R"(OpEntryPoint Kernel %1 "foo"
OpExecutionMode %1 LocalSizeHint 100 200 300
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%1 = OpFunction %1 None %3
)";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

using RoundTripInstructionsTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::tuple<spv_target_env, std::string>>>;

TEST_P(RoundTripInstructionsTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(std::get<1>(GetParam()),
                                          SPV_BINARY_TO_TEXT_OPTION_NONE,
                                          std::get<0>(GetParam())),
              Eq(std::get<1>(GetParam())));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    NumericLiterals, RoundTripInstructionsTest,
    // This test is independent of environment, so just test the one.
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%1 = OpTypeInt 12 0\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 12 1\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 12 1\n%2 = OpConstant %1 -1867\n",
                "%1 = OpTypeInt 32 0\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 32 1\n%2 = OpConstant %1 1867\n",
                "%1 = OpTypeInt 32 1\n%2 = OpConstant %1 -1867\n",
                "%1 = OpTypeInt 64 0\n%2 = OpConstant %1 18446744073709551615\n",
                "%1 = OpTypeInt 64 1\n%2 = OpConstant %1 9223372036854775807\n",
                "%1 = OpTypeInt 64 1\n%2 = OpConstant %1 -9223372036854775808\n",
                // 16-bit floats print as hex floats.
                "%1 = OpTypeFloat 16\n%2 = OpConstant %1 0x1.ff4p+16\n",
                "%1 = OpTypeFloat 16\n%2 = OpConstant %1 -0x1.d2cp-10\n",
                // 32-bit floats
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -3.125\n",
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1.8p+128\n", // NaN
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1.0002p+128\n", // NaN
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 0x1p+128\n", // Inf
                "%1 = OpTypeFloat 32\n%2 = OpConstant %1 -0x1p+128\n", // -Inf
                // 64-bit floats
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -3.125\n",
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.ffffffffffffap-1023\n", // small normal
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.ffffffffffffap-1023\n",
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1.8p+1024\n", // NaN
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1.0002p+1024\n", // NaN
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 0x1p+1024\n", // Inf
                "%1 = OpTypeFloat 64\n%2 = OpConstant %1 -0x1p+1024\n", // -Inf
            })));
// clang-format on

INSTANTIATE_TEST_SUITE_P(
    MemoryAccessMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpStore %1 %2\n",       // 3 words long.
                "OpStore %1 %2 None\n",  // 4 words long, explicit final 0.
                "OpStore %1 %2 Volatile\n",
                "OpStore %1 %2 Aligned 8\n",
                "OpStore %1 %2 Nontemporal\n",
                // Combinations show the names from LSB to MSB
                "OpStore %1 %2 Volatile|Aligned 16\n",
                "OpStore %1 %2 Volatile|Nontemporal\n",
                "OpStore %1 %2 Volatile|Aligned|Nontemporal 32\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    FPFastMathModeMasks, RoundTripInstructionsTest,
    Combine(
        ::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                          SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
        ::testing::ValuesIn(std::vector<std::string>{
            "OpDecorate %1 FPFastMathMode None\n",
            "OpDecorate %1 FPFastMathMode NotNaN\n",
            "OpDecorate %1 FPFastMathMode NotInf\n",
            "OpDecorate %1 FPFastMathMode NSZ\n",
            "OpDecorate %1 FPFastMathMode AllowRecip\n",
            "OpDecorate %1 FPFastMathMode Fast\n",
            // Combinations show the names from LSB to MSB
            "OpDecorate %1 FPFastMathMode NotNaN|NotInf\n",
            "OpDecorate %1 FPFastMathMode NSZ|AllowRecip\n",
            "OpDecorate %1 FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast\n",
        })));

INSTANTIATE_TEST_SUITE_P(
    LoopControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_2),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpLoopMerge %1 %2 None\n",
                "OpLoopMerge %1 %2 Unroll\n",
                "OpLoopMerge %1 %2 DontUnroll\n",
                "OpLoopMerge %1 %2 Unroll|DontUnroll\n",
            })));

INSTANTIATE_TEST_SUITE_P(LoopControlMasksV11, RoundTripInstructionsTest,
                         Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_1,
                                                   SPV_ENV_UNIVERSAL_1_2,
                                                   SPV_ENV_UNIVERSAL_1_3),
                                 ::testing::ValuesIn(std::vector<std::string>{
                                     "OpLoopMerge %1 %2 DependencyInfinite\n",
                                     "OpLoopMerge %1 %2 DependencyLength 8\n",
                                 })));

INSTANTIATE_TEST_SUITE_P(
    SelectionControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_3, SPV_ENV_UNIVERSAL_1_2),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpSelectionMerge %1 None\n",
                "OpSelectionMerge %1 Flatten\n",
                "OpSelectionMerge %1 DontFlatten\n",
                "OpSelectionMerge %1 Flatten|DontFlatten\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    FunctionControlMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%2 = OpFunction %1 None %3\n",
                "%2 = OpFunction %1 Inline %3\n",
                "%2 = OpFunction %1 DontInline %3\n",
                "%2 = OpFunction %1 Pure %3\n",
                "%2 = OpFunction %1 Const %3\n",
                "%2 = OpFunction %1 Inline|Pure|Const %3\n",
                "%2 = OpFunction %1 DontInline|Const %3\n",
            })));

INSTANTIATE_TEST_SUITE_P(
    ImageMasks, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_0, SPV_ENV_UNIVERSAL_1_1,
                              SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "%2 = OpImageFetch %1 %3 %4\n",
                "%2 = OpImageFetch %1 %3 %4 None\n",
                "%2 = OpImageFetch %1 %3 %4 Bias %5\n",
                "%2 = OpImageFetch %1 %3 %4 Lod %5\n",
                "%2 = OpImageFetch %1 %3 %4 Grad %5 %6\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffset %5\n",
                "%2 = OpImageFetch %1 %3 %4 Offset %5\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffsets %5\n",
                "%2 = OpImageFetch %1 %3 %4 Sample %5\n",
                "%2 = OpImageFetch %1 %3 %4 MinLod %5\n",
                "%2 = OpImageFetch %1 %3 %4 Bias|Lod|Grad %5 %6 %7 %8\n",
                "%2 = OpImageFetch %1 %3 %4 ConstOffset|Offset|ConstOffsets"
                " %5 %6 %7\n",
                "%2 = OpImageFetch %1 %3 %4 Sample|MinLod %5 %6\n",
                "%2 = OpImageFetch %1 %3 %4"
                " Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample|MinLod"
                " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"})));

INSTANTIATE_TEST_SUITE_P(
    NewInstructionsInSPIRV1_2, RoundTripInstructionsTest,
    Combine(::testing::Values(SPV_ENV_UNIVERSAL_1_2, SPV_ENV_UNIVERSAL_1_3),
            ::testing::ValuesIn(std::vector<std::string>{
                "OpExecutionModeId %1 SubgroupsPerWorkgroupId %2\n",
                "OpExecutionModeId %1 LocalSizeId %2 %3 %4\n",
                "OpExecutionModeId %1 LocalSizeHintId %2 %3 %4\n",
                "OpDecorateId %1 AlignmentId %2\n",
                "OpDecorateId %1 MaxByteOffsetId %2\n",
            })));

using MaskSorting = TextToBinaryTest;

TEST_F(MaskSorting, MasksAreSortedFromLSBToMSB) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "OpStore %1 %2 Nontemporal|Aligned|Volatile 32"),
              Eq("OpStore %1 %2 Volatile|Aligned|Nontemporal 32\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(
          "OpDecorate %1 FPFastMathMode NotInf|Fast|AllowRecip|NotNaN|NSZ"),
      Eq("OpDecorate %1 FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpLoopMerge %1 %2 DontUnroll|Unroll"),
      Eq("OpLoopMerge %1 %2 Unroll|DontUnroll\n"));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpSelectionMerge %1 DontFlatten|Flatten"),
      Eq("OpSelectionMerge %1 Flatten|DontFlatten\n"));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "%2 = OpFunction %1 DontInline|Const|Pure|Inline %3"),
              Eq("%2 = OpFunction %1 Inline|DontInline|Pure|Const %3\n"));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "%2 = OpImageFetch %1 %3 %4"
                  " MinLod|Sample|Offset|Lod|Grad|ConstOffsets|ConstOffset|Bias"
                  " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"),
              Eq("%2 = OpImageFetch %1 %3 %4"
                 " Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample|MinLod"
                 " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"));
}

using OperandTypeTest = TextToBinaryTest;

TEST_F(OperandTypeTest, OptionalTypedLiteralNumber) {
  const std::string input =
      "%1 = OpTypeInt 32 0\n"
      "%2 = OpConstant %1 42\n"
      "OpSwitch %2 %3 100 %4\n";
  EXPECT_EQ(input, EncodeAndDecodeSuccessfully(input));
}

using IndentTest = spvtest::TextToBinaryTest;

TEST_F(IndentTest, Sample) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits
%11 = OpConstant %1 42
OpStore %2 %3 Aligned|Volatile 4 ; bogus, but not indented
)";
  const std::string expected =
      R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
          %1 = OpTypeInt 32 0
          %2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10
         %11 = OpConstant %1 42
               OpStore %2 %3 Volatile|Aligned 4
)";
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(input, SPV_BINARY_TO_TEXT_OPTION_INDENT),
      expected);
}

using FriendlyNameDisassemblyTest = spvtest::TextToBinaryTest;

TEST_F(FriendlyNameDisassemblyTest, Sample) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %3 %4 %5 %6 %7 %8 %9 %10 ; force IDs into double digits
%11 = OpConstant %1 42
)";
  const std::string expected =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%_struct_2 = OpTypeStruct %uint %3 %4 %5 %6 %7 %8 %9 %10
%uint_42 = OpConstant %uint 42
)";
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  input, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES),
              expected);
}

TEST_F(TextToBinaryTest, ShowByteOffsetsWhenRequested) {
  const std::string input = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeInt 32 0
%2 = OpTypeVoid
)";
  const std::string expected =
      R"(OpCapability Shader                                 ; 0x00000014
OpMemoryModel Logical GLSL450                       ; 0x0000001c
%1 = OpTypeInt 32 0                                 ; 0x00000028
%2 = OpTypeVoid                                     ; 0x00000038
)";
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  input, SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET),
              expected);
}

TEST_F(TextToBinaryTest, Comments) {
  const std::string input = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %63 "main" %4 %22
OpExecutionMode %63 OriginUpperLeft
OpSource GLSL 450
OpName %4 "_ue"
OpName %8 "_uf"
OpName %11 "_ug"
OpName %12 "_uA"
OpMemberName %12 0 "_ux"
OpName %14 "_uc"
OpName %15 "_uB"
OpMemberName %15 0 "_ux"
OpName %20 "_ud"
OpName %22 "_ucol"
OpName %26 "ANGLEDepthRangeParams"
OpMemberName %26 0 "near"
OpMemberName %26 1 "far"
OpMemberName %26 2 "diff"
OpMemberName %26 3 "reserved"
OpName %27 "ANGLEUniformBlock"
OpMemberName %27 0 "viewport"
OpMemberName %27 1 "clipDistancesEnabled"
OpMemberName %27 2 "xfbActiveUnpaused"
OpMemberName %27 3 "xfbVerticesPerInstance"
OpMemberName %27 4 "numSamples"
OpMemberName %27 5 "xfbBufferOffsets"
OpMemberName %27 6 "acbBufferOffsets"
OpMemberName %27 7 "depthRange"
OpName %29 "ANGLEUniforms"
OpName %33 "_uc"
OpName %32 "_uh"
OpName %49 "_ux"
OpName %50 "_uy"
OpName %48 "_ui"
OpName %63 "main"
OpName %65 "param"
OpName %68 "param"
OpName %73 "param"
OpDecorate %4 Location 0
OpDecorate %8 RelaxedPrecision
OpDecorate %8 DescriptorSet 0
OpDecorate %8 Binding 0
OpDecorate %11 DescriptorSet 0
OpDecorate %11 Binding 1
OpMemberDecorate %12 0 Offset 0
OpMemberDecorate %12 0 RelaxedPrecision
OpDecorate %12 Block
OpDecorate %14 DescriptorSet 0
OpDecorate %14 Binding 2
OpMemberDecorate %15 0 Offset 0
OpMemberDecorate %15 0 RelaxedPrecision
OpDecorate %15 BufferBlock
OpDecorate %20 DescriptorSet 0
OpDecorate %20 Binding 3
OpDecorate %22 RelaxedPrecision
OpDecorate %22 Location 0
OpMemberDecorate %26 0 Offset 0
OpMemberDecorate %26 1 Offset 4
OpMemberDecorate %26 2 Offset 8
OpMemberDecorate %26 3 Offset 12
OpMemberDecorate %27 0 Offset 0
OpMemberDecorate %27 1 Offset 16
OpMemberDecorate %27 2 Offset 20
OpMemberDecorate %27 3 Offset 24
OpMemberDecorate %27 4 Offset 28
OpMemberDecorate %27 5 Offset 32
OpMemberDecorate %27 6 Offset 48
OpMemberDecorate %27 7 Offset 64
OpMemberDecorate %27 2 RelaxedPrecision
OpMemberDecorate %27 4 RelaxedPrecision
OpDecorate %27 Block
OpDecorate %29 DescriptorSet 0
OpDecorate %29 Binding 4
OpDecorate %32 RelaxedPrecision
OpDecorate %33 RelaxedPrecision
OpDecorate %36 RelaxedPrecision
OpDecorate %37 RelaxedPrecision
OpDecorate %38 RelaxedPrecision
OpDecorate %39 RelaxedPrecision
OpDecorate %41 RelaxedPrecision
OpDecorate %42 RelaxedPrecision
OpDecorate %43 RelaxedPrecision
OpDecorate %48 RelaxedPrecision
OpDecorate %49 RelaxedPrecision
OpDecorate %50 RelaxedPrecision
OpDecorate %52 RelaxedPrecision
OpDecorate %53 RelaxedPrecision
OpDecorate %54 RelaxedPrecision
OpDecorate %55 RelaxedPrecision
OpDecorate %56 RelaxedPrecision
OpDecorate %57 RelaxedPrecision
OpDecorate %58 RelaxedPrecision
OpDecorate %59 RelaxedPrecision
OpDecorate %60 RelaxedPrecision
OpDecorate %67 RelaxedPrecision
OpDecorate %68 RelaxedPrecision
OpDecorate %72 RelaxedPrecision
OpDecorate %73 RelaxedPrecision
OpDecorate %75 RelaxedPrecision
OpDecorate %76 RelaxedPrecision
OpDecorate %77 RelaxedPrecision
OpDecorate %80 RelaxedPrecision
OpDecorate %81 RelaxedPrecision
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%5 = OpTypeImage %1 2D 0 0 0 1 Unknown
%6 = OpTypeSampledImage %5
%9 = OpTypeImage %1 2D 0 0 0 2 Rgba8
%12 = OpTypeStruct %2
%15 = OpTypeStruct %2
%16 = OpTypeInt 32 0
%17 = OpConstant %16 2
%18 = OpTypeArray %15 %17
%23 = OpTypeInt 32 1
%24 = OpTypeVector %23 4
%25 = OpTypeVector %16 4
%26 = OpTypeStruct %1 %1 %1 %1
%27 = OpTypeStruct %2 %16 %16 %23 %23 %24 %25 %26
%35 = OpTypeVector %1 2
%40 = OpTypeVector %23 2
%61 = OpTypeVoid
%69 = OpConstant %16 0
%78 = OpConstant %16 1
%3 = OpTypePointer Input %2
%7 = OpTypePointer UniformConstant %6
%10 = OpTypePointer UniformConstant %9
%13 = OpTypePointer Uniform %12
%19 = OpTypePointer Uniform %18
%21 = OpTypePointer Output %2
%28 = OpTypePointer Uniform %27
%30 = OpTypePointer Function %2
%70 = OpTypePointer Uniform %2
%31 = OpTypeFunction %2 %30
%47 = OpTypeFunction %2 %30 %30
%62 = OpTypeFunction %61
%4 = OpVariable %3 Input
%8 = OpVariable %7 UniformConstant
%11 = OpVariable %10 UniformConstant
%14 = OpVariable %13 Uniform
%20 = OpVariable %19 Uniform
%22 = OpVariable %21 Output
%29 = OpVariable %28 Uniform
%32 = OpFunction %2 None %31
%33 = OpFunctionParameter %30
%34 = OpLabel
%36 = OpLoad %6 %8
%37 = OpLoad %2 %33
%38 = OpVectorShuffle %35 %37 %37 0 1
%39 = OpImageSampleImplicitLod %2 %36 %38
%41 = OpLoad %2 %33
%42 = OpVectorShuffle %35 %41 %41 2 3
%43 = OpConvertFToS %40 %42
%44 = OpLoad %9 %11
%45 = OpImageRead %2 %44 %43
%46 = OpFAdd %2 %39 %45
OpReturnValue %46
OpFunctionEnd
%48 = OpFunction %2 None %47
%49 = OpFunctionParameter %30
%50 = OpFunctionParameter %30
%51 = OpLabel
%52 = OpLoad %2 %49
%53 = OpVectorShuffle %35 %52 %52 0 1
%54 = OpLoad %2 %50
%55 = OpVectorShuffle %35 %54 %54 2 3
%56 = OpCompositeExtract %1 %53 0
%57 = OpCompositeExtract %1 %53 1
%58 = OpCompositeExtract %1 %55 0
%59 = OpCompositeExtract %1 %55 1
%60 = OpCompositeConstruct %2 %56 %57 %58 %59
OpReturnValue %60
OpFunctionEnd
%63 = OpFunction %61 None %62
%64 = OpLabel
%65 = OpVariable %30 Function
%68 = OpVariable %30 Function
%73 = OpVariable %30 Function
%66 = OpLoad %2 %4
OpStore %65 %66
%67 = OpFunctionCall %2 %32 %65
%71 = OpAccessChain %70 %14 %69
%72 = OpLoad %2 %71
OpStore %68 %72
%74 = OpAccessChain %70 %20 %69 %69
%75 = OpLoad %2 %74
OpStore %73 %75
%76 = OpFunctionCall %2 %48 %68 %73
%77 = OpFAdd %2 %67 %76
%79 = OpAccessChain %70 %20 %78 %69
%80 = OpLoad %2 %79
%81 = OpFAdd %2 %77 %80
OpStore %22 %81
OpReturn
OpFunctionEnd
)";
  const std::string expected = R"(               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %1 "main" %2 %3
               OpExecutionMode %1 OriginUpperLeft

               ; Debug Information
               OpSource GLSL 450
               OpName %2 "_ue"                      ; id %2
               OpName %4 "_uf"                      ; id %4
               OpName %5 "_ug"                      ; id %5
               OpName %6 "_uA"                      ; id %6
               OpMemberName %6 0 "_ux"
               OpName %7 "_uc"                      ; id %7
               OpName %8 "_uB"                      ; id %8
               OpMemberName %8 0 "_ux"
               OpName %9 "_ud"                      ; id %9
               OpName %3 "_ucol"                    ; id %3
               OpName %10 "ANGLEDepthRangeParams"   ; id %10
               OpMemberName %10 0 "near"
               OpMemberName %10 1 "far"
               OpMemberName %10 2 "diff"
               OpMemberName %10 3 "reserved"
               OpName %11 "ANGLEUniformBlock"       ; id %11
               OpMemberName %11 0 "viewport"
               OpMemberName %11 1 "clipDistancesEnabled"
               OpMemberName %11 2 "xfbActiveUnpaused"
               OpMemberName %11 3 "xfbVerticesPerInstance"
               OpMemberName %11 4 "numSamples"
               OpMemberName %11 5 "xfbBufferOffsets"
               OpMemberName %11 6 "acbBufferOffsets"
               OpMemberName %11 7 "depthRange"
               OpName %12 "ANGLEUniforms"           ; id %12
               OpName %13 "_uc"                     ; id %13
               OpName %14 "_uh"                     ; id %14
               OpName %15 "_ux"                     ; id %15
               OpName %16 "_uy"                     ; id %16
               OpName %17 "_ui"                     ; id %17
               OpName %1 "main"                     ; id %1
               OpName %18 "param"                   ; id %18
               OpName %19 "param"                   ; id %19
               OpName %20 "param"                   ; id %20

               ; Annotations
               OpDecorate %2 Location 0
               OpDecorate %4 RelaxedPrecision
               OpDecorate %4 DescriptorSet 0
               OpDecorate %4 Binding 0
               OpDecorate %5 DescriptorSet 0
               OpDecorate %5 Binding 1
               OpMemberDecorate %6 0 Offset 0
               OpMemberDecorate %6 0 RelaxedPrecision
               OpDecorate %6 Block
               OpDecorate %7 DescriptorSet 0
               OpDecorate %7 Binding 2
               OpMemberDecorate %8 0 Offset 0
               OpMemberDecorate %8 0 RelaxedPrecision
               OpDecorate %8 BufferBlock
               OpDecorate %9 DescriptorSet 0
               OpDecorate %9 Binding 3
               OpDecorate %3 RelaxedPrecision
               OpDecorate %3 Location 0
               OpMemberDecorate %10 0 Offset 0
               OpMemberDecorate %10 1 Offset 4
               OpMemberDecorate %10 2 Offset 8
               OpMemberDecorate %10 3 Offset 12
               OpMemberDecorate %11 0 Offset 0
               OpMemberDecorate %11 1 Offset 16
               OpMemberDecorate %11 2 Offset 20
               OpMemberDecorate %11 3 Offset 24
               OpMemberDecorate %11 4 Offset 28
               OpMemberDecorate %11 5 Offset 32
               OpMemberDecorate %11 6 Offset 48
               OpMemberDecorate %11 7 Offset 64
               OpMemberDecorate %11 2 RelaxedPrecision
               OpMemberDecorate %11 4 RelaxedPrecision
               OpDecorate %11 Block
               OpDecorate %12 DescriptorSet 0
               OpDecorate %12 Binding 4
               OpDecorate %14 RelaxedPrecision
               OpDecorate %13 RelaxedPrecision
               OpDecorate %21 RelaxedPrecision
               OpDecorate %22 RelaxedPrecision
               OpDecorate %23 RelaxedPrecision
               OpDecorate %24 RelaxedPrecision
               OpDecorate %25 RelaxedPrecision
               OpDecorate %26 RelaxedPrecision
               OpDecorate %27 RelaxedPrecision
               OpDecorate %17 RelaxedPrecision
               OpDecorate %15 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %29 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %31 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %33 RelaxedPrecision
               OpDecorate %34 RelaxedPrecision
               OpDecorate %35 RelaxedPrecision
               OpDecorate %36 RelaxedPrecision
               OpDecorate %37 RelaxedPrecision
               OpDecorate %19 RelaxedPrecision
               OpDecorate %38 RelaxedPrecision
               OpDecorate %20 RelaxedPrecision
               OpDecorate %39 RelaxedPrecision
               OpDecorate %40 RelaxedPrecision
               OpDecorate %41 RelaxedPrecision
               OpDecorate %42 RelaxedPrecision
               OpDecorate %43 RelaxedPrecision

               ; Types, variables and constants
         %44 = OpTypeFloat 32
         %45 = OpTypeVector %44 4
         %46 = OpTypeImage %44 2D 0 0 0 1 Unknown
         %47 = OpTypeSampledImage %46
         %48 = OpTypeImage %44 2D 0 0 0 2 Rgba8
          %6 = OpTypeStruct %45                     ; Block
          %8 = OpTypeStruct %45                     ; BufferBlock
         %49 = OpTypeInt 32 0
         %50 = OpConstant %49 2
         %51 = OpTypeArray %8 %50
         %52 = OpTypeInt 32 1
         %53 = OpTypeVector %52 4
         %54 = OpTypeVector %49 4
         %10 = OpTypeStruct %44 %44 %44 %44
         %11 = OpTypeStruct %45 %49 %49 %52 %52 %53 %54 %10     ; Block
         %55 = OpTypeVector %44 2
         %56 = OpTypeVector %52 2
         %57 = OpTypeVoid
         %58 = OpConstant %49 0
         %59 = OpConstant %49 1
         %60 = OpTypePointer Input %45
         %61 = OpTypePointer UniformConstant %47
         %62 = OpTypePointer UniformConstant %48
         %63 = OpTypePointer Uniform %6
         %64 = OpTypePointer Uniform %51
         %65 = OpTypePointer Output %45
         %66 = OpTypePointer Uniform %11
         %67 = OpTypePointer Function %45
         %68 = OpTypePointer Uniform %45
         %69 = OpTypeFunction %45 %67
         %70 = OpTypeFunction %45 %67 %67
         %71 = OpTypeFunction %57
          %2 = OpVariable %60 Input                 ; Location 0
          %4 = OpVariable %61 UniformConstant       ; RelaxedPrecision, DescriptorSet 0, Binding 0
          %5 = OpVariable %62 UniformConstant       ; DescriptorSet 0, Binding 1
          %7 = OpVariable %63 Uniform               ; DescriptorSet 0, Binding 2
          %9 = OpVariable %64 Uniform               ; DescriptorSet 0, Binding 3
          %3 = OpVariable %65 Output                ; RelaxedPrecision, Location 0
         %12 = OpVariable %66 Uniform               ; DescriptorSet 0, Binding 4

               ; Function 14
         %14 = OpFunction %45 None %69              ; RelaxedPrecision
         %13 = OpFunctionParameter %67              ; RelaxedPrecision
         %72 = OpLabel
         %21 = OpLoad %47 %4                        ; RelaxedPrecision
         %22 = OpLoad %45 %13                       ; RelaxedPrecision
         %23 = OpVectorShuffle %55 %22 %22 0 1      ; RelaxedPrecision
         %24 = OpImageSampleImplicitLod %45 %21 %23     ; RelaxedPrecision
         %25 = OpLoad %45 %13                           ; RelaxedPrecision
         %26 = OpVectorShuffle %55 %25 %25 2 3          ; RelaxedPrecision
         %27 = OpConvertFToS %56 %26                    ; RelaxedPrecision
         %73 = OpLoad %48 %5
         %74 = OpImageRead %45 %73 %27
         %75 = OpFAdd %45 %24 %74
               OpReturnValue %75
               OpFunctionEnd

               ; Function 17
         %17 = OpFunction %45 None %70              ; RelaxedPrecision
         %15 = OpFunctionParameter %67              ; RelaxedPrecision
         %16 = OpFunctionParameter %67              ; RelaxedPrecision
         %76 = OpLabel
         %28 = OpLoad %45 %15                       ; RelaxedPrecision
         %29 = OpVectorShuffle %55 %28 %28 0 1      ; RelaxedPrecision
         %30 = OpLoad %45 %16                       ; RelaxedPrecision
         %31 = OpVectorShuffle %55 %30 %30 2 3      ; RelaxedPrecision
         %32 = OpCompositeExtract %44 %29 0         ; RelaxedPrecision
         %33 = OpCompositeExtract %44 %29 1         ; RelaxedPrecision
         %34 = OpCompositeExtract %44 %31 0         ; RelaxedPrecision
         %35 = OpCompositeExtract %44 %31 1         ; RelaxedPrecision
         %36 = OpCompositeConstruct %45 %32 %33 %34 %35     ; RelaxedPrecision
               OpReturnValue %36
               OpFunctionEnd

               ; Function 1
          %1 = OpFunction %57 None %71
         %77 = OpLabel
         %18 = OpVariable %67 Function
         %19 = OpVariable %67 Function              ; RelaxedPrecision
         %20 = OpVariable %67 Function              ; RelaxedPrecision
         %78 = OpLoad %45 %2
               OpStore %18 %78
         %37 = OpFunctionCall %45 %14 %18           ; RelaxedPrecision
         %79 = OpAccessChain %68 %7 %58
         %38 = OpLoad %45 %79                       ; RelaxedPrecision
               OpStore %19 %38
         %80 = OpAccessChain %68 %9 %58 %58
         %39 = OpLoad %45 %80                       ; RelaxedPrecision
               OpStore %20 %39
         %40 = OpFunctionCall %45 %17 %19 %20       ; RelaxedPrecision
         %41 = OpFAdd %45 %37 %40                   ; RelaxedPrecision
         %81 = OpAccessChain %68 %9 %59 %58
         %42 = OpLoad %45 %81                       ; RelaxedPrecision
         %43 = OpFAdd %45 %41 %42                   ; RelaxedPrecision
               OpStore %3 %43
               OpReturn
               OpFunctionEnd
)";

  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(input, SPV_BINARY_TO_TEXT_OPTION_COMMENT |
                                             SPV_BINARY_TO_TEXT_OPTION_INDENT),
      expected);
}

// Test version string.
TEST_F(TextToBinaryTest, VersionString) {
  auto words = CompileSuccessfully("");
  spv_text decoded_text = nullptr;
  EXPECT_THAT(spvBinaryToText(ScopedContext().context, words.data(),
                              words.size(), SPV_BINARY_TO_TEXT_OPTION_NONE,
                              &decoded_text, &diagnostic),
              Eq(SPV_SUCCESS));
  EXPECT_EQ(nullptr, diagnostic);

  EXPECT_THAT(decoded_text->str, HasSubstr("Version: 1.0\n"))
      << EncodeAndDecodeSuccessfully("");
  spvTextDestroy(decoded_text);
}

// Test generator string.

// A test case for the generator string.  This allows us to
// test both of the 16-bit components of the generator word.
struct GeneratorStringCase {
  uint16_t generator;
  uint16_t misc;
  std::string expected;
};

using GeneratorStringTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<GeneratorStringCase>>;

TEST_P(GeneratorStringTest, Sample) {
  auto words = CompileSuccessfully("");
  EXPECT_EQ(2u, SPV_INDEX_GENERATOR_NUMBER);
  words[SPV_INDEX_GENERATOR_NUMBER] =
      SPV_GENERATOR_WORD(GetParam().generator, GetParam().misc);

  spv_text decoded_text = nullptr;
  EXPECT_THAT(spvBinaryToText(ScopedContext().context, words.data(),
                              words.size(), SPV_BINARY_TO_TEXT_OPTION_NONE,
                              &decoded_text, &diagnostic),
              Eq(SPV_SUCCESS));
  EXPECT_THAT(diagnostic, Eq(nullptr));
  EXPECT_THAT(std::string(decoded_text->str), HasSubstr(GetParam().expected));
  spvTextDestroy(decoded_text);
}

INSTANTIATE_TEST_SUITE_P(GeneratorStrings, GeneratorStringTest,
                         ::testing::ValuesIn(std::vector<GeneratorStringCase>{
                             {SPV_GENERATOR_KHRONOS, 12, "Khronos; 12"},
                             {SPV_GENERATOR_LUNARG, 99, "LunarG; 99"},
                             {SPV_GENERATOR_VALVE, 1, "Valve; 1"},
                             {SPV_GENERATOR_CODEPLAY, 65535, "Codeplay; 65535"},
                             {SPV_GENERATOR_NVIDIA, 19, "NVIDIA; 19"},
                             {SPV_GENERATOR_ARM, 1000, "ARM; 1000"},
                             {SPV_GENERATOR_KHRONOS_LLVM_TRANSLATOR, 38,
                              "Khronos LLVM/SPIR-V Translator; 38"},
                             {SPV_GENERATOR_KHRONOS_ASSEMBLER, 2,
                              "Khronos SPIR-V Tools Assembler; 2"},
                             {SPV_GENERATOR_KHRONOS_GLSLANG, 1,
                              "Khronos Glslang Reference Front End; 1"},
                             {1000, 18, "Unknown(1000); 18"},
                             {65535, 32767, "Unknown(65535); 32767"},
                         }));

// TODO(dneto): Test new instructions and enums in SPIR-V 1.3

}  // namespace
}  // namespace spvtools
