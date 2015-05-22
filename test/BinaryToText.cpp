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

class BinaryToText : public ::testing::Test {
 public:
  BinaryToText() : binary(), opcodeTable(nullptr), operandTable(nullptr) {}

  virtual void SetUp() {
    ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
    ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
    ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

    const char *textStr = R"(
OpSource OpenCL 12
OpMemoryModel Physical64 OpenCL1.2
OpSourceExtension "PlaceholderExtensionName"
OpEntryPoint Kernel $1
OpExecutionMode $1 LocalSizeHint 1 1 1
OpTypeVoid %2
OpTypeBool %3
OpTypeInt %4 8 0
OpTypeInt %5 8 1
OpTypeInt %6 16 0
OpTypeInt %7 16 1
OpTypeInt %8 32 0
OpTypeInt %9 32 1
OpTypeInt %10 64 0
OpTypeInt %11 64 1
OpTypeFloat %12 16
OpTypeFloat %13 32
OpTypeFloat %14 64
OpTypeVector %15 4 2
)";
    spv_text_t text = {textStr, strlen(textStr)};
    spv_diagnostic diagnostic = nullptr;
    spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
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
            spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
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
      spvBinaryToText(&binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
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
            spvBinaryToText(binary, 0, nullptr, operandTable, extInstTable,
                            &text, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            nullptr, extInstTable, &text, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            operandTable, nullptr, &text, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(BinaryToText, InvalidDiagnostic) {
  spv_text text;
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC,
            spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                            operandTable, extInstTable, &text, nullptr));
}

TEST(BinaryToTextExtInst, Default) {
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  const char *spirv = R"(
OpCapability Shader
OpExtInstImport %glsl450 "GLSL.std.450"
OpMemoryModel Logical Simple
OpEntryPoint Vertex $main "main"
OpTypeVoid %void
OpTypeFloat %float 32
OpConstant $float %const1.5 1.5
OpTypeFunction %fnMain %void
OpFunction $void %main None $fnMain
OpLabel %lbMain
OpExtInst $float %result $glsl450 round $const1.5
OpReturn
OpFunctionEnd
)";
  spv_text_t text = {spirv, strlen(spirv)};
  spv_binary binary;
  spv_diagnostic diagnostic;
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
                                       extInstTable, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }

  error = spvBinaryToText(
      binary, SPV_BINARY_TO_TEXT_OPTION_COLOR | SPV_BINARY_TO_TEXT_OPTION_PRINT,
      opcodeTable, operandTable, extInstTable, nullptr, &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
}
