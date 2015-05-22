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

TEST(TextDestroy, Default) {
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));

  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));

  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

  char textStr[] =
      "OpSource OpenCL 12\n"
      "OpMemoryModel Physical64 OpenCL1.2\n"
      "OpSourceExtension \"PlaceholderExtensionName\"\n"
      "OpEntryPoint Kernel 0\n"
      "OpExecutionMode 0 LocalSizeHint 1 1 1\n"
      "OpTypeVoid 1\n"
      "OpTypeBool 2\n"
      "OpTypeInt 3 8 0\n"
      "OpTypeInt 4 8 1\n"
      "OpTypeInt 5 16 0\n"
      "OpTypeInt 6 16 1\n"
      "OpTypeInt 7 32 0\n"
      "OpTypeInt 8 32 1\n"
      "OpTypeInt 9 64 0\n"
      "OpTypeInt 10 64 1\n"
      "OpTypeFloat 11 16\n"
      "OpTypeFloat 12 32\n"
      "OpTypeFloat 13 64\n"
      "OpTypeVector 14 3 2\n";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  EXPECT_NE(nullptr, binary);
  EXPECT_NE(nullptr, binary->code);
  EXPECT_NE(0, binary->wordCount);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    ASSERT_TRUE(false);
  }

  spv_text resultText = nullptr;
  EXPECT_EQ(SPV_SUCCESS,
            spvBinaryToText(binary, 0, opcodeTable, operandTable, extInstTable,
                            &resultText, &diagnostic));
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_TRUE(false);
  }
  EXPECT_NE(nullptr, text.str);
  EXPECT_NE(0, text.length);
  spvTextDestroy(resultText);
}
