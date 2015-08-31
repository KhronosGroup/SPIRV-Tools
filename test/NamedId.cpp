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

TEST(NamedId, Default) {
  const char *spirv = R"(
          OpCapability Shader
          OpMemoryModel Logical Simple
          OpEntryPoint Vertex %main
  %void = OpTypeVoid
%fnMain = OpTypeFunction %void
  %main = OpFunction %void None %fnMain
%lbMain = OpLabel
          OpReturn
          OpFunctionEnd)";
  spv_text_t text;
  text.str = spirv;
  text.length = strlen(spirv);
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  spv_binary binary;
  spv_diagnostic diagnostic;
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
                                       extInstTable, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvBinaryDestroy(binary);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
  error = spvBinaryToText(
      binary, SPV_BINARY_TO_TEXT_OPTION_PRINT | SPV_BINARY_TO_TEXT_OPTION_COLOR,
      opcodeTable, operandTable, extInstTable, nullptr, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvBinaryDestroy(binary);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
  spvBinaryDestroy(binary);
}

}  // anonymous namespace
