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
#ifndef _TEST_FIXTURE_H_
#define _TEST_FIXTURE_H_

#include "UnitSPIRV.h"

// Common setup for TextToBinary tests. SetText() should be called to populate
// the actual test text.
template<typename T>
class TextToBinaryTestBase : public T {
 public:
  TextToBinaryTestBase()
      : opcodeTable(nullptr),
        operandTable(nullptr),
        diagnostic(nullptr),
        text(),
        binary(nullptr) {}

  virtual void SetUp() {
    ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
    ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
    ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
    char textStr[] = "substitute the text member variable with your test";
    text = {textStr, strlen(textStr)};
  }

  virtual void TearDown() {
    if (diagnostic) spvDiagnosticDestroy(diagnostic);
  }

  void SetText(const std::string& code) {
    textString = code;
    text.str = textString.c_str();
    text.length = textString.size();
  }

  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  spv_diagnostic diagnostic;

  std::string textString;
  spv_text_t text;
  spv_binary binary;
};

class TextToBinaryTest : public TextToBinaryTestBase<::testing::Test> {};

#endif// _TEXT_FIXTURE_H_
