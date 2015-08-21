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
#include <iomanip>
#include <utility>
#include <vector>

union char_word_t {
  char cs[4];
  uint32_t u;
};

TEST(TextToBinary, Default) {
  // TODO: Ensure that on big endian systems that this converts the word to
  // little endian for encoding comparison!
  spv_endianness_t endian = SPV_ENDIANNESS_LITTLE;

  const char *textStr = R"(
OpSource OpenCL 12
OpMemoryModel Physical64 OpenCL
OpSourceExtension "PlaceholderExtensionName"
OpEntryPoint Kernel $1
OpExecutionMode $1 LocalSizeHint 1 1 1
OpTypeVoid %2
OpTypeBool %3
; commment
OpTypeInt %4 8 0 ; comment
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

  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));

  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));

  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
                                       extInstTable, &binary, &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }

  struct bin {
    bin(spv_binary binary) : binary(binary) {}
    ~bin() { spvBinaryDestroy(binary); }
    spv_binary binary;
  } bin(binary);

  EXPECT_NE(nullptr, text.str);
  EXPECT_NE(0, text.length);

  // TODO: Verify binary
  ASSERT_EQ(SPV_MAGIC_NUMBER, binary->code[SPV_INDEX_MAGIC_NUMBER]);
  ASSERT_EQ(SPV_VERSION_NUMBER, binary->code[SPV_INDEX_VERSION_NUMBER]);
  ASSERT_EQ(SPV_GENERATOR_KHRONOS, binary->code[SPV_INDEX_GENERATOR_NUMBER]);
  ASSERT_EQ(16, binary->code[SPV_INDEX_BOUND]);  // TODO: Bound?
  ASSERT_EQ(0, binary->code[SPV_INDEX_SCHEMA]);  // Reserved: schema

  uint64_t instIndex = SPV_INDEX_INSTRUCTION;

  ASSERT_EQ(spvOpcodeMake(3, OpSource), binary->code[instIndex++]);
  ASSERT_EQ(SourceLanguageOpenCL, binary->code[instIndex++]);
  ASSERT_EQ(12, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(3, OpMemoryModel), binary->code[instIndex++]);
  ASSERT_EQ(AddressingModelPhysical64, binary->code[instIndex++]);
  ASSERT_EQ(MemoryModelOpenCL, binary->code[instIndex++]);

  uint16_t sourceExtensionWordCount =
      (uint16_t)((strlen("PlaceholderExtensionName") / sizeof(uint32_t)) + 2);
  ASSERT_EQ(spvOpcodeMake(sourceExtensionWordCount, OpSourceExtension),
            binary->code[instIndex++]);
  // TODO: This only works on little endian systems!
  char_word_t cw = {{'P', 'l', 'a', 'c'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  cw = {{'e', 'h', 'o', 'l'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  cw = {{'d', 'e', 'r', 'E'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  cw = {{'x', 't', 'e', 'n'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  cw = {{'s', 'i', 'o', 'n'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  cw = {{'N', 'a', 'm', 'e'}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);
  ASSERT_EQ(0, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(3, OpEntryPoint), binary->code[instIndex++]);
  ASSERT_EQ(ExecutionModelKernel, binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(6, OpExecutionMode), binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);
  ASSERT_EQ(ExecutionModeLocalSizeHint, binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(2, OpTypeVoid), binary->code[instIndex++]);
  ASSERT_EQ(2, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(2, OpTypeBool), binary->code[instIndex++]);
  ASSERT_EQ(3, binary->code[instIndex++]);

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(4, binary->code[instIndex++]);
  ASSERT_EQ(8, binary->code[instIndex++]);  // NOTE: 8 bits wide
  ASSERT_EQ(0, binary->code[instIndex++]);  // NOTE: Unsigned

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(5, binary->code[instIndex++]);
  ASSERT_EQ(8, binary->code[instIndex++]);  // NOTE: 8 bits wide
  ASSERT_EQ(1, binary->code[instIndex++]);  // NOTE: Signed

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(6, binary->code[instIndex++]);
  ASSERT_EQ(16, binary->code[instIndex++]);  // NOTE: 16 bits wide
  ASSERT_EQ(0, binary->code[instIndex++]);   // NOTE: Unsigned

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(7, binary->code[instIndex++]);
  ASSERT_EQ(16, binary->code[instIndex++]);  // NOTE: 16 bits wide
  ASSERT_EQ(1, binary->code[instIndex++]);   // NOTE: Signed

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(8, binary->code[instIndex++]);
  ASSERT_EQ(32, binary->code[instIndex++]);  // NOTE: 32 bits wide
  ASSERT_EQ(0, binary->code[instIndex++]);   // NOTE: Unsigned

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(9, binary->code[instIndex++]);
  ASSERT_EQ(32, binary->code[instIndex++]);  // NOTE: 32 bits wide
  ASSERT_EQ(1, binary->code[instIndex++]);   // NOTE: Signed

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(10, binary->code[instIndex++]);
  ASSERT_EQ(64, binary->code[instIndex++]);  // NOTE: 64 bits wide
  ASSERT_EQ(0, binary->code[instIndex++]);   // NOTE: Unsigned

  ASSERT_EQ(spvOpcodeMake(4, OpTypeInt), binary->code[instIndex++]);
  ASSERT_EQ(11, binary->code[instIndex++]);
  ASSERT_EQ(64, binary->code[instIndex++]);  // NOTE: 64 bits wide
  ASSERT_EQ(1, binary->code[instIndex++]);   // NOTE: Signed

  ASSERT_EQ(spvOpcodeMake(3, OpTypeFloat), binary->code[instIndex++]);
  ASSERT_EQ(12, binary->code[instIndex++]);
  ASSERT_EQ(16, binary->code[instIndex++]);  // NOTE: 16 bits wide

  ASSERT_EQ(spvOpcodeMake(3, OpTypeFloat), binary->code[instIndex++]);
  ASSERT_EQ(13, binary->code[instIndex++]);
  ASSERT_EQ(32, binary->code[instIndex++]);  // NOTE: 32 bits wide

  ASSERT_EQ(spvOpcodeMake(3, OpTypeFloat), binary->code[instIndex++]);
  ASSERT_EQ(14, binary->code[instIndex++]);
  ASSERT_EQ(64, binary->code[instIndex++]);  // NOTE: 64 bits wide

  ASSERT_EQ(spvOpcodeMake(4, OpTypeVector), binary->code[instIndex++]);
  ASSERT_EQ(15, binary->code[instIndex++]);
  ASSERT_EQ(4, binary->code[instIndex++]);
  ASSERT_EQ(2, binary->code[instIndex++]);
}

TEST_F(TextToBinaryTest, InvalidText) {
  spv_text_t text = {nullptr, 0};
  spv_binary binary;
  ASSERT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(&text, opcodeTable, operandTable, extInstTable,
                            &binary, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidTable) {
  SetText("OpEntryPoint Kernel 0\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(&text, nullptr, operandTable, extInstTable, &binary,
                            &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(&text, opcodeTable, nullptr, extInstTable, &binary,
                            &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(&text, opcodeTable, operandTable, nullptr, &binary,
                            &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidPointer) {
  SetText("OpEntryPoint Kernel 0\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER,
            spvTextToBinary(&text, opcodeTable, operandTable, extInstTable,
                            nullptr, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidDiagnostic) {
  SetText("OpEntryPoint Kernel 0\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  spv_binary binary;
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC,
            spvTextToBinary(&text, opcodeTable, operandTable, extInstTable,
                            &binary, nullptr));
}

TEST_F(TextToBinaryTest, InvalidPrefix) {
  SetText("Invalid");
  ASSERT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(&text, opcodeTable, operandTable, extInstTable,
                            &binary, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, ImmediateIntOpCode) {
  SetText("!0x00FF00FF");
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[5]);
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, ImmediateIntOperand) {
  SetText("OpCapability !0x00FF00FF");
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[6]);
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

struct InstValue {
  const char* inst;
  uint32_t value;
};
class GLSingleFloatTest
    : public TextToBinaryTestBase<
          ::testing::TestWithParam<InstValue>> {};

TEST_P(GLSingleFloatTest, GLSLExtSingleFloatParamTest) {
  const std::string spirv = R"(
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
OpExtInst $float %result $glsl450 )" +
                            std::string(GetParam().inst) + R"( $const1.5
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
  std::vector<uint32_t> expected_contains({
    12/*OpExtInst*/ | 6 << 16, 4/*%float*/, 9 /*%result*/, 1 /*%glsl450*/,
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
    SingleElementFloatingParams, GLSingleFloatTest,
    ::testing::ValuesIn(std::vector<InstValue>({
        {"Round", 1}, {"RoundEven", 2}, {"Trunc", 3}, {"FAbs", 4}, {"SAbs", 5},
        {"FSign", 6}, {"SSign", 7}, {"Floor", 8}, {"Ceil", 9}, {"Fract", 10},
        {"Radians", 11}, {"Degrees", 12}, {"Sin", 13}, {"Cos", 14}, {"Tan", 15},
        {"Asin", 16}, {"Acos", 17}, {"Atan", 18}, {"Sinh", 19}, {"Cosh", 20},
        {"Tanh", 21}, {"Asinh", 22}, {"Acosh", 23}, {"Atanh", 24}})));

TEST_F(TextToBinaryTest, StringSpace) {
  SetText("OpSourceExtension \"string with spaces\"");
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  if (binary) {
    spvBinaryDestroy(binary);
  }
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}
