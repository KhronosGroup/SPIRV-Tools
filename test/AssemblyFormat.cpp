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

namespace {

using test_fixture::TextToBinaryTest;

TEST_F(TextToBinaryTest, EncodeAAFTextAsAAF) {
  SetText("%2 = OpConstant %1 1000");
  EXPECT_EQ(SPV_SUCCESS,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, EncodeAAFTextAsCAF) {
  SetText("%2 = OpConstant %1 1000");
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_CANONICAL,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  ASSERT_TRUE(diagnostic);
  EXPECT_STREQ(
      "Expected <opcode> at the beginning of an instruction, found '%2'.",
      diagnostic->error);
  EXPECT_EQ(0, diagnostic->position.line);
}

TEST_F(TextToBinaryTest, EncodeCAFTextAsCAF) {
  SetText("OpConstant %1 %2 1000");
  EXPECT_EQ(SPV_SUCCESS,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_CANONICAL,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, EncodeCAFTextAsAAF) {
  SetText("OpConstant %1 %2 1000");
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  ASSERT_TRUE(diagnostic);
  EXPECT_STREQ(
      "Expected <result-id> at the beginning of an instruction, found "
      "'OpConstant'.",
      diagnostic->error);
  EXPECT_EQ(0, diagnostic->position.line);
}

TEST_F(TextToBinaryTest, EncodeMixedTextAsAAF) {
  SetText("OpConstant %1 %2 1000\n%3 = OpConstant %1 2000");
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_ASSIGNMENT,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  ASSERT_TRUE(diagnostic);
  EXPECT_STREQ(
      "Expected <result-id> at the beginning of an instruction, found "
      "'OpConstant'.",
      diagnostic->error);
  EXPECT_EQ(0, diagnostic->position.line);
}

TEST_F(TextToBinaryTest, EncodeMixedTextAsCAF) {
  SetText("OpConstant %1 %2 1000\n%3 = OpConstant %1 2000");
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT,
            spvTextWithFormatToBinary(
                text.str, text.length, SPV_ASSEMBLY_SYNTAX_FORMAT_CANONICAL,
                opcodeTable, operandTable, extInstTable, &binary, &diagnostic));
  ASSERT_TRUE(diagnostic);
  EXPECT_STREQ(
      "Expected <opcode> at the beginning of an instruction, found '%3'.",
      diagnostic->error);
  EXPECT_EQ(1, diagnostic->position.line);
}

}  // anonymous namespace
