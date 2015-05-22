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

TEST(TextAdvance, LeadingNewLines) {
  char textStr[] = "\n\nWord";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t position = {};
  ASSERT_EQ(SPV_SUCCESS, spvTextAdvance(&text, &position));
  ASSERT_EQ(0, position.column);
  ASSERT_EQ(2, position.line);
  ASSERT_EQ(2, position.index);
}

TEST(TextAdvance, LeadingSpaces) {
  char textStr[] = "    Word";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t position = {};
  ASSERT_EQ(SPV_SUCCESS, spvTextAdvance(&text, &position));
  ASSERT_EQ(4, position.column);
  ASSERT_EQ(0, position.line);
  ASSERT_EQ(4, position.index);
}

TEST(TextAdvance, LeadingTabs) {
  char textStr[] = "\t\t\tWord";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t position = {};
  ASSERT_EQ(SPV_SUCCESS, spvTextAdvance(&text, &position));
  ASSERT_EQ(3, position.column);
  ASSERT_EQ(0, position.line);
  ASSERT_EQ(3, position.index);
}

TEST(TextAdvance, LeadingNewLinesSpacesAndTabs) {
  char textStr[] = "\n\n\t  Word";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t position = {};
  ASSERT_EQ(SPV_SUCCESS, spvTextAdvance(&text, &position));
  ASSERT_EQ(3, position.column);
  ASSERT_EQ(2, position.line);
  ASSERT_EQ(5, position.index);
}

TEST(TextAdvance, NullTerminator) {
  char textStr[] = "";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t position = {};
  ASSERT_EQ(SPV_END_OF_STREAM, spvTextAdvance(&text, &position));
}
