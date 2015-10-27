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
#include <utility>
#include <vector>

namespace {

using libspirv::AssemblyContext;
using libspirv::AssemblyGrammar;
using spvtest::TextToBinaryTest;
using spvtest::AutoText;

TEST(GetWord, Simple) {
  EXPECT_EQ("", AssemblyContext(AutoText(""), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText("\0a"), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText(" a"), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText("\ta"), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText("\va"), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText("\ra"), nullptr).getWord());
  EXPECT_EQ("", AssemblyContext(AutoText("\na"), nullptr).getWord());
  EXPECT_EQ("abc", AssemblyContext(AutoText("abc"), nullptr).getWord());
  EXPECT_EQ("abc", AssemblyContext(AutoText("abc "), nullptr).getWord());
  EXPECT_EQ("abc",
            AssemblyContext(AutoText("abc\t"), nullptr).getWord());
  EXPECT_EQ("abc",
            AssemblyContext(AutoText("abc\r"), nullptr).getWord());
  EXPECT_EQ("abc",
            AssemblyContext(AutoText("abc\v"), nullptr).getWord());
  EXPECT_EQ("abc",
            AssemblyContext(AutoText("abc\n"), nullptr).getWord());
}

// An mask parsing test case.
struct MaskCase {
  spv_operand_type_t which_enum;
  uint32_t expected_value;
  const char* expression;
};

using GoodMaskParseTest = ::testing::TestWithParam<MaskCase>;

TEST_P(GoodMaskParseTest, GoodMaskExpressions) {
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));

  uint32_t value;
  EXPECT_EQ(SPV_SUCCESS, AssemblyGrammar(operandTable, nullptr, nullptr)
                             .parseMaskOperand(GetParam().which_enum,
                                               GetParam().expression, &value));
  EXPECT_EQ(GetParam().expected_value, value);
}

INSTANTIATE_TEST_CASE_P(
    ParseMask, GoodMaskParseTest,
    ::testing::ValuesIn(std::vector<MaskCase>{
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 0, "None"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 1, "NotNaN"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 2, "NotInf"},
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotNaN|NotInf"},
        // Mask experssions are symmetric.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotInf|NotNaN"},
        // Repeating a value has no effect.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 3, "NotInf|NotNaN|NotInf"},
        // Using 3 operands still works.
        {SPV_OPERAND_TYPE_FP_FAST_MATH_MODE, 0x13, "NotInf|NotNaN|Fast"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 1, "Flatten"},
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 2, "DontFlatten"},
        // Weirdly, you can specify to flatten and don't flatten a selection.
        {SPV_OPERAND_TYPE_SELECTION_CONTROL, 3, "Flatten|DontFlatten"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 1, "Unroll"},
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 2, "DontUnroll"},
        // Weirdly, you can specify to unroll and don't unroll a loop.
        {SPV_OPERAND_TYPE_LOOP_CONTROL, 3, "Unroll|DontUnroll"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 0, "None"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 1, "Inline"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 2, "DontInline"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 4, "Pure"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 8, "Const"},
        {SPV_OPERAND_TYPE_FUNCTION_CONTROL, 0xd, "Inline|Const|Pure"},
    }));

using BadFPFastMathMaskParseTest = ::testing::TestWithParam<const char*>;

TEST_P(BadFPFastMathMaskParseTest, BadMaskExpressions) {
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));

  uint32_t value;
  EXPECT_NE(SPV_SUCCESS,
            AssemblyGrammar(operandTable, nullptr, nullptr)
                .parseMaskOperand(SPV_OPERAND_TYPE_FP_FAST_MATH_MODE,
                                  GetParam(), &value));
}

INSTANTIATE_TEST_CASE_P(ParseMask, BadFPFastMathMaskParseTest,
                        ::testing::ValuesIn(std::vector<const char*>{
                            nullptr, "", "NotValidEnum", "|", "NotInf|",
                            "|NotInf", "NotInf||NotNaN",
                            "Unroll"  // A good word, but for the wrong enum
                        }));

// TODO(dneto): Aliasing like this relies on undefined behaviour. Fix this.
union char_word_t {
  char cs[4];
  uint32_t u;
};

TEST(TextToBinary, Default) {
  // TODO: Ensure that on big endian systems that this converts the word to
  // little endian for encoding comparison!
  spv_endianness_t endian = SPV_ENDIANNESS_LITTLE;

  const char* textStr = R"(
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
%15 = OpTypeVector %4 2
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

using TextToBinaryFloatValueTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<std::pair<std::string, uint32_t>>>;

TEST_P(TextToBinaryFloatValueTest, NormalValues) {
  const std::string assembly = "%1 = OpTypeFloat 32\n%2 = OpConstant %1 ";
  const std::string input_string = assembly + GetParam().first;
  const std::string expected_string =
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

TEST(AssemblyContextParseNarrowSignedIntegers, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  int16_t i16;

  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &i16, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &i16, ""));

  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &i16, ""));
  EXPECT_EQ(0, i16);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("32767", ec, &i16, ""));
  EXPECT_EQ(32767, i16);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-32768", ec, &i16, ""));
  EXPECT_EQ(-32768, i16);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-0", ec, &i16, ""));
  EXPECT_EQ(0, i16);

  // These are out of range, so they should return an error.
  // The error code depends on whether this is an optional value.
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("32768", ec, &i16, ""));
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            context.parseNumber("65535", SPV_ERROR_INVALID_TEXT, &i16, ""));

  // Check hex parsing.
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0x7fff", ec, &i16, ""));
  EXPECT_EQ(32767, i16);
  // This is out of range.
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0xffff", ec, &i16, ""));
}

TEST(AssemblyContextParseNarrowUnsignedIntegers, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  uint16_t u16;

  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &u16, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &u16, ""));

  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &u16, ""));
  EXPECT_EQ(0, u16);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("65535", ec, &u16, ""));
  EXPECT_EQ(65535, u16);
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("65536", ec, &u16, ""));

  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-0", ec, &u16, ""));
  EXPECT_EQ(0, u16);
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("-1", ec, &u16, ""));
  EXPECT_EQ(0, u16);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0xffff", ec, &u16, ""));
  EXPECT_EQ(0xffff, u16);
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0x10000", ec, &u16, ""));
}

TEST(AssemblyContextParseWideSignedIntegers, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  int64_t i64;
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &i64, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &i64, ""));
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &i64, ""));
  EXPECT_EQ(0, i64);
  EXPECT_EQ(SPV_SUCCESS,
            context.parseNumber("0x7fffffffffffffff", ec, &i64, ""));
  EXPECT_EQ(0x7fffffffffffffff, i64);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-0", ec, &i64, ""));
  EXPECT_EQ(0, i64);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-1", ec, &i64, ""));
  EXPECT_EQ(-1, i64);
}

TEST(AssemblyContextParseWideUnsignedIntegers, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  uint64_t u64;
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &u64, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &u64, ""));
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &u64, ""));
  EXPECT_EQ(0, u64);
  EXPECT_EQ(SPV_SUCCESS,
            context.parseNumber("0xffffffffffffffff", ec, &u64, ""));
  EXPECT_EQ(0xffffffffffffffffULL, u64);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-0", ec, &u64, ""));
  EXPECT_EQ(0, u64);
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("-1", ec, &u64, ""));
}

TEST(AssemblyContextParseFloat, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  float f;

  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &f, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &f, ""));

  // These values are exactly representatble.
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &f, ""));
  EXPECT_EQ(0.0f, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("42", ec, &f, ""));
  EXPECT_EQ(42.0f, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("2.5", ec, &f, ""));
  EXPECT_EQ(2.5f, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-32.5", ec, &f, ""));
  EXPECT_EQ(-32.5f, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("1e38", ec, &f, ""));
  EXPECT_EQ(1e38f, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-1e38", ec, &f, ""));
  EXPECT_EQ(-1e38f, f);

  // Out of range.
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("1e40", ec, &f, ""));
}

TEST(AssemblyContextParseDouble, Sample) {
  AssemblyContext context(AutoText(""), nullptr);
  const spv_result_t ec = SPV_FAILED_MATCH;
  double f;

  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("", ec, &f, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("0=", ec, &f, ""));

  // These values are exactly representatble.
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("0", ec, &f, ""));
  EXPECT_EQ(0.0, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("42", ec, &f, ""));
  EXPECT_EQ(42.0, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("2.5", ec, &f, ""));
  EXPECT_EQ(2.5, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-32.5", ec, &f, ""));
  EXPECT_EQ(-32.5, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("1e38", ec, &f, ""));
  EXPECT_EQ(1e38, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-1e38", ec, &f, ""));
  EXPECT_EQ(-1e38, f);
  // These are out of range for 32-bit float, but in range for 64-bit float.
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("1e40", ec, &f, ""));
  EXPECT_EQ(1e40, f);
  EXPECT_EQ(SPV_SUCCESS, context.parseNumber("-1e40", ec, &f, ""));
  EXPECT_EQ(-1e40, f);

  // Out of range.
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("1e400", ec, &f, ""));
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("-1e400", ec, &f, ""));
}

TEST(AssemblyContextParseMessages, Errors) {
  spv_diagnostic diag = nullptr;
  const spv_result_t ec = SPV_FAILED_MATCH;
  AssemblyContext context(AutoText(""), &diag);
  int16_t i16;

  // No message is generated for a failure to parse an optional value.
  EXPECT_EQ(SPV_FAILED_MATCH, context.parseNumber("abc", ec, &i16, "bad narrow int: "));
  EXPECT_EQ(nullptr, diag);

  // For a required value, use the message fragment.
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            context.parseNumber("abc", SPV_ERROR_INVALID_TEXT, &i16,
                                "bad narrow int: "));
  ASSERT_NE(nullptr, diag);
  EXPECT_EQ("bad narrow int: abc", std::string(diag->error));
  // Don't leak.
  spvDiagnosticDestroy(diag);
}

}  // anonymous namespace
