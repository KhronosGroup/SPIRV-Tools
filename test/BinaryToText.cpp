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

    const char* textStr = R"(
      OpSource OpenCL 12
      OpMemoryModel Physical64 OpenCL
      OpSourceExtension "PlaceholderExtensionName"
      OpEntryPoint Kernel %1
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

struct InstValue {
  const char* inst;
  uint32_t value;
};

class BinaryToTextGLExtSingleFloatInst
    : public ::testing::TestWithParam<InstValue> {};

TEST_P(BinaryToTextGLExtSingleFloatInst, Default) {
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  const std::string spirv = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Simple
OpEntryPoint Vertex %2 "main"
%3 = OpTypeVoid
%4 = OpTypeFloat 32
%5 = OpConstant %4 1
%6 = OpTypeFunction %3
%2 = OpFunction %3 None %6
%8 = OpLabel
%9 = OpExtInst %4 %1 )" + std::string(GetParam().inst) +
                            R"( %5
OpReturn
OpFunctionEnd
)";
  const std::string spirv_header =
      R"(; SPIR-V
; Version: 99
; Generator: Khronos
; Bound: 10
; Schema: 0)";
  spv_text_t text = {spirv.c_str(), spirv.size()};
  spv_binary binary;
  spv_diagnostic diagnostic;
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
                                       extInstTable, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error) << "Source was: " << std::endl
                                  << spirv << std::endl
                                  << "Test case for : " << GetParam().inst
                                  << std::endl;
  }

  spv_text output_text;
  error =
      spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                      operandTable, extInstTable, &output_text, &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
  EXPECT_EQ(spirv_header + spirv, output_text->str);
  spvTextDestroy(output_text);
}

INSTANTIATE_TEST_CASE_P(
    SingleElementFloatingParams, BinaryToTextGLExtSingleFloatInst,
    ::testing::ValuesIn(std::vector<InstValue>({
        {"Round", 1}, {"RoundEven", 2}, {"Trunc", 3}, {"FAbs", 4}, {"SAbs", 5},
        {"FSign", 6}, {"SSign", 7}, {"Floor", 8}, {"Ceil", 9}, {"Fract", 10},
        {"Radians", 11}, {"Degrees", 12}, {"Sin", 13}, {"Cos", 14}, {"Tan", 15},
        {"Asin", 16}, {"Acos", 17}, {"Atan", 18}, {"Sinh", 19}, {"Cosh", 20},
        {"Tanh", 21}, {"Asinh", 22}, {"Acosh", 23}, {"Atanh", 24}})));
