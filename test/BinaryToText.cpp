//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

using ::testing::Eq;

namespace {

class BinaryToText : public ::testing::Test {
 public:
  BinaryToText() : binary(), opcodeTable(nullptr), operandTable(nullptr) {}

  virtual void SetUp() {
    ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
    ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
    ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

    const char* textStr = R"(
      OpSource OpenCL 12
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
        spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                        extInstTable, &binary, &diagnostic);
    if (error) {
      spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
      ASSERT_EQ(SPV_SUCCESS, error);
    }
  }

  virtual void TearDown() { spvBinaryDestroy(binary); }

  spv_binary binary;
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
};

TEST_F(BinaryToText, Default) {
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvBinaryToText(binary->code, binary->wordCount,
                            SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            operandTable, extInstTable, &text, &diagnostic));
  printf("%s", text->str);
  spvTextDestroy(text);
}

TEST_F(BinaryToText, InvalidCode) {
  spv_binary_t binary = {nullptr, 42};
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(
      SPV_ERROR_INVALID_BINARY,
      spvBinaryToText(nullptr, 42, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                      operandTable, extInstTable, &text, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, InvalidTable) {
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvBinaryToText(binary->code, binary->wordCount, 0, nullptr,
                            operandTable, extInstTable, &text, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvBinaryToText(binary->code, binary->wordCount,
                            SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            nullptr, extInstTable, &text, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvBinaryToText(binary->code, binary->wordCount,
                            SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            operandTable, nullptr, &text, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, InvalidDiagnostic) {
  spv_text text;
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC,
            spvBinaryToText(binary->code, binary->wordCount,
                            SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            operandTable, extInstTable, &text, nullptr));
}

TEST(BinaryToTextSmall, OneInstruction) {
  // TODO(dneto): This test could/should be refactored.
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;
  const char* input = "OpSource OpenCL 12";
  spv_result_t error =
      spvTextToBinary(input, strlen(input), opcodeTable, operandTable,
                      extInstTable, &binary, &diagnostic);
  ASSERT_EQ(SPV_SUCCESS, error);
  spv_text text = nullptr;
  error = spvBinaryToText(binary->code, binary->wordCount,
                          SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                          operandTable, extInstTable, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, error);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
  spvTextDestroy(text);
}

// Exercise the case where an operand itself has operands.
// This could detect problems in updating the expected-set-of-operands
// list.
TEST(BinaryToTextSmall, OperandWithOperands) {
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;

  AutoText input(R"(OpEntryPoint Kernel %fn "foo"
                 OpExecutionMode %fn LocalSizeHint 100 200 300
                 %void = OpTypeVoid
                 %fnType = OpTypeFunction %void
                 %fn = OpFunction %void None %fnType
                 )");
  spv_result_t error =
      spvTextToBinary(input.str.c_str(), input.str.length(), opcodeTable,
                      operandTable, extInstTable, &binary, &diagnostic);
  ASSERT_EQ(SPV_SUCCESS, error);
  spv_text text = nullptr;
  error = spvBinaryToText(binary->code, binary->wordCount,
                          SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                          operandTable, extInstTable, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, error);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
  spvTextDestroy(text);
}

TEST(BinaryToTextSmall, LiteralInt64) {
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;

  AutoText input("%1 = OpTypeInt 64 0\n%2 = OpConstant %1 123456789021\n");
  spv_result_t error =
      spvTextToBinary(input.str.c_str(), input.str.length(), opcodeTable,
                      operandTable, extInstTable, &binary, &diagnostic);
  ASSERT_EQ(SPV_SUCCESS, error);
  spv_text text = nullptr;
  error = spvBinaryToText(binary->code, binary->wordCount,
                          SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                          operandTable, extInstTable, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, error);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
  const std::string header =
      "; SPIR-V\n; Version: 99\n; Generator: Khronos\n; "
      "Bound: 3\n; Schema: 0\n";
  EXPECT_EQ(header + input.str, text->str);
  spvTextDestroy(text);
}

TEST(BinaryToTextSmall, LiteralDouble) {
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;

  // Pi: 3.1415926535897930 => 0x400921fb54442d18 => 4614256656552045848
  AutoText input(
      "%1 = OpTypeFloat 64\n%2 = OpSpecConstant %1 3.1415926535897930");
  spv_result_t error =
      spvTextToBinary(input.str.c_str(), input.str.length(), opcodeTable,
                      operandTable, extInstTable, &binary, &diagnostic);
  ASSERT_EQ(SPV_SUCCESS, error);
  spv_text text = nullptr;
  error = spvBinaryToText(binary->code, binary->wordCount,
                          SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                          operandTable, extInstTable, &text, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, error);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
  const std::string output =
      R"(; SPIR-V
; Version: 99
; Generator: Khronos
; Bound: 3
; Schema: 0
%1 = OpTypeFloat 64
%2 = OpSpecConstant %1 4614256656552045848
)";
  EXPECT_EQ(output, text->str);
  spvTextDestroy(text);
}

using RoundTripInstructionsTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<std::string>>;

TEST_P(RoundTripInstructionsTest, Sample) {
  EXPECT_THAT(EncodeAndDecodeSuccessfully(GetParam()),
              Eq(GetParam()));
};

// clang-format off
INSTANTIATE_TEST_CASE_P(
    MemoryAccessMasks, RoundTripInstructionsTest,
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
    }));
// clang-format on

INSTANTIATE_TEST_CASE_P(
    FPFastMathModeMasks, RoundTripInstructionsTest,
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
    }));

INSTANTIATE_TEST_CASE_P(LoopControlMasks, RoundTripInstructionsTest,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "OpLoopMerge %1 %2 None\n",
                            "OpLoopMerge %1 %2 Unroll\n",
                            "OpLoopMerge %1 %2 DontUnroll\n",
                            "OpLoopMerge %1 %2 Unroll|DontUnroll\n",
                        }));

INSTANTIATE_TEST_CASE_P(SelectionControlMasks, RoundTripInstructionsTest,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "OpSelectionMerge %1 None\n",
                            "OpSelectionMerge %1 Flatten\n",
                            "OpSelectionMerge %1 DontFlatten\n",
                            "OpSelectionMerge %1 Flatten|DontFlatten\n",
                        }));

// clang-format off
INSTANTIATE_TEST_CASE_P(
    FunctionControlMasks, RoundTripInstructionsTest,
    ::testing::ValuesIn(std::vector<std::string>{
        "%2 = OpFunction %1 None %3\n",
        "%2 = OpFunction %1 Inline %3\n",
        "%2 = OpFunction %1 DontInline %3\n",
        "%2 = OpFunction %1 Pure %3\n",
        "%2 = OpFunction %1 Const %3\n",
        "%2 = OpFunction %1 Inline|Pure|Const %3\n",
        "%2 = OpFunction %1 DontInline|Const %3\n",
    }));
// clang-format on

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ImageMasks, RoundTripInstructionsTest,
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
              " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"}));
// clang-format on

using MaskSorting = spvtest::TextToBinaryTest;

TEST_F(MaskSorting, MasksAreSortedFromLSBToMSB) {
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(
          "OpStore %1 %2 Nontemporal|Aligned|Volatile 32"),
      Eq(std::string("OpStore %1 %2 Volatile|Aligned|Nontemporal 32\n")));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(
          "OpDecorate %1 FPFastMathMode NotInf|Fast|AllowRecip|NotNaN|NSZ"),
      Eq(std::string(
          "OpDecorate %1 FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast\n")));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpLoopMerge %1 %2 DontUnroll|Unroll"),
      Eq(std::string("OpLoopMerge %1 %2 Unroll|DontUnroll\n")));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully("OpSelectionMerge %1 DontFlatten|Flatten"),
      Eq(std::string("OpSelectionMerge %1 Flatten|DontFlatten\n")));
  EXPECT_THAT(
      EncodeAndDecodeSuccessfully(
          "%2 = OpFunction %1 DontInline|Const|Pure|Inline %3"),
      Eq(std::string("%2 = OpFunction %1 Inline|DontInline|Pure|Const %3\n")));
  EXPECT_THAT(EncodeAndDecodeSuccessfully(
                  "%2 = OpImageFetch %1 %3 %4"
                  " MinLod|Sample|Offset|Lod|Grad|ConstOffsets|ConstOffset|Bias"
                  " %5 %6 %7 %8 %9 %10 %11 %12 %13\n"),
              Eq(std::string(
                  "%2 = OpImageFetch %1 %3 %4"
                  " Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample|MinLod"
                  " %5 %6 %7 %8 %9 %10 %11 %12 %13\n")));
}

}  // anonymous namespace
