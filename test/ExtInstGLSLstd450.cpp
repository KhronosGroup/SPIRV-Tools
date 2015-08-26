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

#include "TestFixture.h"
#include "UnitSPIRV.h"
#include <algorithm>
#include <vector>

struct InstValue {
  const char* inst;
  uint32_t value;
};

class GLExtSingleFloatTextToBinTest
    : public TextToBinaryTestBase<::testing::TestWithParam<InstValue>> {};

TEST_P(GLExtSingleFloatTextToBinTest, GLSLExtSingleFloatParamTest) {
  const std::string spirv = R"(
            OpCapability Shader
 %glsl450 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical Simple
            OpEntryPoint Vertex %main "main"
    %void = OpTypeVoid
   %float = OpTypeFloat 32
%const1.5 = OpConstant %float 1.5
  %fnMain = OpTypeFunction %void
    %main = OpFunction %void None %fnMain
  %lbMain = OpLabel
  %result = OpExtInst %float %glsl450 )" +
                            std::string(GetParam().inst) + R"( %const1.5
            OpReturn
            OpFunctionEnd
)";

  this->text.str = spirv.c_str();
  this->text.length = spirv.size();
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&this->text, this->opcodeTable,
                                         this->operandTable, this->extInstTable,
                                         &this->binary, &this->diagnostic))
      << "Source was: " << std::endl
      << spirv << std::endl
      << "Test case for : " << GetParam().inst << std::endl;
  std::vector<uint32_t> expected_contains(
      {12 /*OpExtInst*/ | 6 << 16, 4 /*%float*/, 8 /*%result*/, 1 /*%glsl450*/,
       GetParam().value, 5 /*const1.5*/});
  EXPECT_TRUE(std::search(this->binary->code,
                          this->binary->code + this->binary->wordCount,
                          expected_contains.begin(), expected_contains.end()) !=
              this->binary->code + this->binary->wordCount);
  if (this->binary) {
    spvBinaryDestroy(this->binary);
  }
  if (this->diagnostic) {
    spvDiagnosticPrint(this->diagnostic);
  }
}

INSTANTIATE_TEST_CASE_P(
    SingleElementFloatingParams, GLExtSingleFloatTextToBinTest,
    ::testing::ValuesIn(std::vector<InstValue>({
        {"Round", 1}, {"RoundEven", 2}, {"Trunc", 3}, {"FAbs", 4}, {"SAbs", 5},
        {"FSign", 6}, {"SSign", 7}, {"Floor", 8}, {"Ceil", 9}, {"Fract", 10},
        {"Radians", 11}, {"Degrees", 12}, {"Sin", 13}, {"Cos", 14}, {"Tan", 15},
        {"Asin", 16}, {"Acos", 17}, {"Atan", 18}, {"Sinh", 19}, {"Cosh", 20},
        {"Tanh", 21}, {"Asinh", 22}, {"Acosh", 23}, {"Atanh", 24},
        // TODO(deki): tests for two-argument functions.
        /*{"Atan2", 25}, {"Pow", 26},*/ {"Exp", 27}, {"Log", 28},
        {"Exp2", 29}, {"Log2", 30}, {"Sqrt", 31}, {"Inversesqrt", 32},
        {"Determinant", 33}, {"Inverse", 34}})));

class GLExtSingleFloatRoundTripTest
    : public ::testing::TestWithParam<InstValue> {};

TEST_P(GLExtSingleFloatRoundTripTest, Default) {
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
    SingleElementFloatingParams, GLExtSingleFloatRoundTripTest,
    ::testing::ValuesIn(std::vector<InstValue>({
        {"Round", 1}, {"RoundEven", 2}, {"Trunc", 3}, {"FAbs", 4}, {"SAbs", 5},
        {"FSign", 6}, {"SSign", 7}, {"Floor", 8}, {"Ceil", 9}, {"Fract", 10},
        {"Radians", 11}, {"Degrees", 12}, {"Sin", 13}, {"Cos", 14}, {"Tan", 15},
        {"Asin", 16}, {"Acos", 17}, {"Atan", 18}, {"Sinh", 19}, {"Cosh", 20},
        {"Tanh", 21}, {"Asinh", 22}, {"Acosh", 23}, {"Atanh", 24},
        // TODO(deki): tests for two-argument functions.
        /*{"Atan2", 25}, {"Pow", 26},*/ {"Exp", 27}, {"Log", 28},
        {"Exp2", 29}, {"Log2", 30}, {"Sqrt", 31}, {"Inversesqrt", 32},
        {"Determinant", 33}, {"Inverse", 34}})));
