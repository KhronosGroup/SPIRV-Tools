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
%15 = OpTypeVector 4 2
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

}  // anonymous namespace
