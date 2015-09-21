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

// Assembler tests for literal numbers and literal strings.

#include "TestFixture.h"

namespace {

using spvtest::TextToBinaryTest;

TEST_F(TextToBinaryTest, LiteralStringInPlaceOfLiteralNumber) {
  EXPECT_EQ(
      R"(Expected literal number, found literal string '"I shouldn't be a string"'.)",
      CompileFailure(R"(OpSource GLSL "I shouldn't be a string")"));
}

TEST_F(TextToBinaryTest, GarbageInPlaceOfLiteralString) {
  EXPECT_EQ(
      R"(Invalid literal string 'nice-source-code'.)",
      CompileFailure(R"(OpSourceExtension nice-source-code)"));
}

TEST_F(TextToBinaryTest, LiteralNumberInPlaceOfLiteralString) {
  EXPECT_EQ(
      R"(Expected literal string, found literal number '1000'.)",
      CompileFailure(R"(OpSourceExtension 1000)"));
}

// TODO(antiagainst): libspirv.h defines SPV_LIMIT_INSTRUCTION_WORD_COUNT_MAX
// to be 0x108. Lift that limit and enable the following test.
TEST_F(TextToBinaryTest, DISABLED_LiteralStringTooLong) {
  const std::string code =
      "OpSourceExtension \"" + std::string(65534, 'o') + "\"\n";
  EXPECT_EQ(code, EncodeAndDecodeSuccessfully(code));
}

}  // anonymous namespace
