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

class Validate : public ::testing::Test {
 public:
  Validate() : binary(), opcodeTable(nullptr), operandTable(nullptr) {}

  virtual void SetUp() {
    ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
    ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
    ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  }

  virtual void TearDown() { spvBinaryDestroy(binary); }

  spv_binary binary;
  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
};

TEST_F(Validate, DISABLED_Default) {
  char str[] = R"(
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute 1
OpExecutionMode 1 LocalSize 1 1 1
OpTypeVoid 2
OpTypeFunction 3 2
OpFunction 2 1 NoControl 3
OpLabel 4
OpReturn
OpFunctionEnd
)";
  spv_text_t text = {str, strlen(str)};
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  ASSERT_EQ(SPV_SUCCESS,
            spvValidate(binary, opcodeTable, operandTable, extInstTable,
                        SPV_VALIDATE_ALL, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
  }
}

TEST_F(Validate, DISABLED_InvalidIdUndefined) {
  char str[] = R"(
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute 1
OpExecutionMode 5 LocalSize 1 1 1
OpTypeVoid 2
OpTypeFunction 3 2
OpFunction 2 1 NoControl 3
OpLabel 4
OpReturn
OpFunctionEnd
)";
  spv_text_t text = {str, strlen(str)};
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_ID,
            spvValidate(binary, opcodeTable, operandTable, extInstTable,
                        SPV_VALIDATE_ALL, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
}

TEST_F(Validate, DISABLED_InvalidIdRedefined) {
  char str[] = R"(
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute 1
OpExecutionMode 1 LocalSize 1 1 1
OpTypeVoid 2
OpTypeFunction 2 2
OpFunction 2 1 NoControl 3
OpLabel 4
OpReturn
OpFunctionEnd
)";
  spv_text_t text = {str, strlen(str)};
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  // TODO: Fix setting of bound in spvTextTo, then remove this!
  ASSERT_EQ(SPV_ERROR_INVALID_ID,
            spvValidate(binary, opcodeTable, operandTable, extInstTable,
                        SPV_VALIDATE_ALL, &diagnostic));
  ASSERT_NE(nullptr, diagnostic);
  spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
}
