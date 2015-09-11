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

namespace {

using test_fixture::TextToBinaryTest;

TEST(GetWord, Simple) {
  EXPECT_EQ("", spvGetWord(""));
  EXPECT_EQ("", spvGetWord("\0a"));
  EXPECT_EQ("", spvGetWord(" a"));
  EXPECT_EQ("", spvGetWord("\ta"));
  EXPECT_EQ("", spvGetWord("\va"));
  EXPECT_EQ("", spvGetWord("\ra"));
  EXPECT_EQ("", spvGetWord("\na"));
  EXPECT_EQ("abc", spvGetWord("abc"));
  EXPECT_EQ("abc", spvGetWord("abc "));
  EXPECT_EQ("abc", spvGetWord("abc\t"));
  EXPECT_EQ("abc", spvGetWord("abc\r"));
  EXPECT_EQ("abc", spvGetWord("abc\v"));
  EXPECT_EQ("abc", spvGetWord("abc\n"));
}

// TODO(dneto): Aliasing like this relies on undefined behaviour. Fix this.
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
      OpEntryPoint Kernel %1 "foo"
      OpExecutionMode %1 LocalSizeHint 1 1 1
 %2 = OpTypeVoid
 %3 = OpTypeBool
 ; commment
 %4 = OpTypeInt 8 0 ; comment
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

  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));

  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));

  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;
  spv_result_t error =
      spvTextToBinary(textStr, strlen(textStr), opcodeTable, operandTable,
                      extInstTable, &binary, &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }

  EXPECT_NE(nullptr, binary->code);
  EXPECT_NE(0, binary->wordCount);

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

  ASSERT_EQ(spvOpcodeMake(4, OpEntryPoint), binary->code[instIndex++]);
  ASSERT_EQ(ExecutionModelKernel, binary->code[instIndex++]);
  ASSERT_EQ(1, binary->code[instIndex++]);
  cw = {{'f', 'o', 'o', 0}};
  ASSERT_EQ(spvFixWord(cw.u, endian), binary->code[instIndex++]);

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
  spv_binary binary;
  ASSERT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(nullptr, 0, opcodeTable, operandTable, extInstTable,
                            &binary, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidTable) {
  SetText(
      "OpEntryPoint Kernel 0 \"\"\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(text.str, text.length, nullptr, operandTable,
                            extInstTable, &binary, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(text.str, text.length, opcodeTable, nullptr,
                            extInstTable, &binary, &diagnostic));
  ASSERT_EQ(SPV_ERROR_INVALID_TABLE,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            nullptr, &binary, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidPointer) {
  SetText(
      "OpEntryPoint Kernel 0 \"\"\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  ASSERT_EQ(SPV_ERROR_INVALID_POINTER,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, nullptr, &diagnostic));
}

TEST_F(TextToBinaryTest, InvalidDiagnostic) {
  SetText(
      "OpEntryPoint Kernel 0 \"\"\nOpExecutionMode 0 LocalSizeHint 1 1 1\n");
  spv_binary binary;
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, nullptr));
}

TEST_F(TextToBinaryTest, InvalidPrefix) {
  SetText("Invalid");
  ASSERT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, StringSpace) {
  SetText("OpSourceExtension \"string with spaces\"");
  EXPECT_EQ(SPV_SUCCESS,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, UnknownBeginningOfInstruction) {
  SetText(R"(
     OpSource OpenCL 12
     OpMemoryModel Physical64 OpenCL
Google
)");

  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(4, diagnostic->position.line + 1);
  EXPECT_EQ(1, diagnostic->position.column + 1);
  EXPECT_STREQ(
      "Expected <opcode> or <result-id> at the beginning of an instruction, "
      "found 'Google'.",
      diagnostic->error);
}

TEST_F(TextToBinaryTest, NoEqualSign) {
  SetText(R"(
     OpSource OpenCL 12
     OpMemoryModel Physical64 OpenCL
%2
)");

  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(5, diagnostic->position.line + 1);
  EXPECT_EQ(1, diagnostic->position.column + 1);
  EXPECT_STREQ("Expected '=', found end of stream.", diagnostic->error);
}

TEST_F(TextToBinaryTest, NoOpCode) {
  SetText(R"(
     OpSource OpenCL 12
     OpMemoryModel Physical64 OpenCL
%2 =
)");

  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(5, diagnostic->position.line + 1);
  EXPECT_EQ(1, diagnostic->position.column + 1);
  EXPECT_STREQ("Expected opcode, found end of stream.", diagnostic->error);
}

TEST_F(TextToBinaryTest, WrongOpCode) {
  SetText(R"(
     OpSource OpenCL 12
     OpMemoryModel Physical64 OpenCL
%2 = Wahahaha
)");

  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(4, diagnostic->position.line + 1);
  EXPECT_EQ(6, diagnostic->position.column + 1);
  EXPECT_STREQ("Invalid Opcode prefix 'Wahahaha'.", diagnostic->error);
}

TEST_F(TextToBinaryTest, GoodSwitch) {
  const SpirvVector code = CompileSuccessfully(R"(
%i32      = OpTypeInt 32 0
%fortytwo = OpConstant %i32 42
%twelve   = OpConstant %i32 12
%entry    = OpLabel
            OpSwitch %fortytwo %default 42 %go42 12 %go12
%go42     = OpLabel
            OpBranch %default
%go12     = OpLabel
            OpBranch %default
%default  = OpLabel
)");

  // Minimal check: The OpSwitch opcode word is correct.
  EXPECT_EQ(int(spv::OpSwitch) || (7 << 16), code[14]);
}

TEST_F(TextToBinaryTest, GoodSwitchZeroCasesOneDefault) {
  const SpirvVector code = CompileSuccessfully(R"(
%i32      = OpTypeInt 32 0
%fortytwo = OpConstant %i32 42
%entry    = OpLabel
            OpSwitch %fortytwo %default
%default  = OpLabel
)");

  // Minimal check: The OpSwitch opcode word is correct.
  EXPECT_EQ(int(spv::OpSwitch) || (3 << 16), code[10]);
}

TEST_F(TextToBinaryTest, BadSwitchTruncatedCase) {
  SetText(R"(
%i32      = OpTypeInt 32 0
%fortytwo = OpConstant %i32 42
%entry    = OpLabel
            OpSwitch %fortytwo %default 42 ; missing target!
%default  = OpLabel
)");

  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(6, diagnostic->position.line + 1);
  EXPECT_EQ(1, diagnostic->position.column + 1);
  EXPECT_STREQ("Expected operand, found next instruction instead.",
               diagnostic->error);
}

using TextToBinaryFloatValueTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<std::pair<std::string, uint32_t>>>;

TEST_P(TextToBinaryFloatValueTest, NormalValues) {
  const std::string assembly = "%1 = OpTypeFloat 32\n%2 = OpConstant %1 ";
  const std::string input_string = assembly + GetParam().first;
  const std::string expected_string =
      "; SPIR-V\n; Version: 99\n; Generator: Khronos\n; "
      "Bound: 3\n; Schema: 0\n" +
      assembly + std::to_string(GetParam().second) + "\n";
  const std::string decoded_string = EncodeAndDecodeSuccessfully(input_string);
  EXPECT_EQ(expected_string, decoded_string);
}

INSTANTIATE_TEST_CASE_P(
    FloatValues, TextToBinaryFloatValueTest,
    ::testing::ValuesIn(std::vector<std::pair<std::string, uint32_t>>{
        {"0.0", 0x00000000},          // +0
        {"!0x00000001", 0x00000001},  // +denorm
        {"!0x00800000", 0x00800000},  // +norm
        {"1.5", 0x3fc00000},
        {"!0x7f800000", 0x7f800000},  // +inf
        {"!0x7f800001", 0x7f800001},  // NaN

        {"-0.0", 0x80000000},         // -0
        {"!0x80000001", 0x80000001},  // -denorm
        {"!0x80800000", 0x80800000},  // -norm
        {"-2.5", 0xc0200000},
        {"!0xff800000", 0xff800000},  // -inf
        {"!0xff800001", 0xff800001},  // NaN
    }));

}  // anonymous namespace
