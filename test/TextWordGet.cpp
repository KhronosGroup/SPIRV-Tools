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

TEST(TextWordGet, NullTerminator) {
  char textStr[] = "Word";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t startPosition = {};
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(&text, &startPosition, word, &endPosition));
  ASSERT_EQ(4, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(4, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, TabTerminator) {
  char textStr[] = "Word\t";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t startPosition = {};
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(&text, &startPosition, word, &endPosition));
  ASSERT_EQ(4, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(4, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, SpaceTerminator) {
  char textStr[] = "Word ";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t startPosition = {};
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(&text, &startPosition, word, &endPosition));
  ASSERT_EQ(4, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(4, endPosition.index);
  ASSERT_STREQ("Word", word.c_str());
}

TEST(TextWordGet, SemicolonTerminator) {
  char textStr[] = "Wo;rd ";
  spv_text_t text = {textStr, strlen(textStr)};
  spv_position_t startPosition = {};
  std::string word;
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(&text, &startPosition, word, &endPosition));
  ASSERT_EQ(2, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(2, endPosition.index);
  ASSERT_STREQ("Wo", word.c_str());
}

TEST(TextWordGet, MultipleWords) {
  char textStr[] = "Words in a sentence";
  spv_text_t text = {textStr, strlen(textStr)};
  const char *words[] = {"Words", "in", "a", "sentence"};

  spv_position_t startPosition = {};
  spv_position_t endPosition = {};

  std::string word;
  for (uint32_t wordIndex = 0; wordIndex < 4; ++wordIndex) {
    ASSERT_EQ(SPV_SUCCESS,
              spvTextWordGet(&text, &startPosition, word, &endPosition));
    ASSERT_EQ(strlen(words[wordIndex]),
              endPosition.column - startPosition.column);
    ASSERT_EQ(0, endPosition.line);
    ASSERT_EQ(strlen(words[wordIndex]),
              endPosition.index - startPosition.index);
    ASSERT_STREQ(words[wordIndex], word.c_str());

    startPosition = endPosition;
    if (3 != wordIndex) {
      ASSERT_EQ(SPV_SUCCESS, spvTextAdvance(&text, &startPosition));
    } else {
      ASSERT_EQ(SPV_END_OF_STREAM, spvTextAdvance(&text, &startPosition));
    }
  }
}
