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

namespace {

#define TAB "\t"
#define NEWLINE "\n"
#define BACKSLASH R"(\)"
#define QUOTE R"(")"

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

TEST(TextWordGet, QuotesAreKept) {
  AutoText input(R"("quotes" "around words")");
  const char *expected[] = {R"("quotes")", R"("around words")"};

  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  EXPECT_EQ(8, endPosition.column);
  EXPECT_EQ(0, endPosition.line);
  EXPECT_EQ(8, endPosition.index);
  EXPECT_STREQ(expected[0], word.c_str());

  // Move to the next word.
  startPosition = endPosition;
  startPosition.index++;
  startPosition.column++;

  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  EXPECT_EQ(23, endPosition.column);
  EXPECT_EQ(0, endPosition.line);
  EXPECT_EQ(23, endPosition.index);
  EXPECT_STREQ(expected[1], word.c_str());
}

TEST(TextWordGet, QuotesBetweenWordsActLikeGlue) {
  AutoText input(R"(quotes" "between words)");
  const char *expected[] = {R"(quotes" "between)", "words"};

  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  EXPECT_EQ(16, endPosition.column);
  EXPECT_EQ(0, endPosition.line);
  EXPECT_EQ(16, endPosition.index);
  EXPECT_STREQ(expected[0], word.c_str());

  // Move to the next word.
  startPosition = endPosition;
  startPosition.index++;
  startPosition.column++;

  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  EXPECT_EQ(22, endPosition.column);
  EXPECT_EQ(0, endPosition.line);
  EXPECT_EQ(22, endPosition.index);
  EXPECT_STREQ(expected[1], word.c_str());
}

TEST(TextWordGet, QuotingWhitespace) {
  // Whitespace surrounded by quotes acts like glue.
  AutoText input(QUOTE "white " NEWLINE TAB " space" QUOTE);
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  EXPECT_EQ(input.str.length(), endPosition.column);
  EXPECT_EQ(0, endPosition.line);
  EXPECT_EQ(input.str.length(), endPosition.index);
  EXPECT_EQ(input.str, word);
}

TEST(TextWordGet, QuoteAlone) {
  AutoText input(QUOTE);
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  ASSERT_EQ(1, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(1, endPosition.index);
  ASSERT_STREQ(QUOTE, word.c_str());
}

TEST(TextWordGet, EscapeAlone) {
  AutoText input(BACKSLASH);
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  ASSERT_EQ(1, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(1, endPosition.index);
  ASSERT_STREQ(BACKSLASH, word.c_str());
}

TEST(TextWordGet, EscapeAtEndOfInput) {
  AutoText input("word" BACKSLASH);
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  ASSERT_EQ(5, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(5, endPosition.index);
  ASSERT_STREQ("word" BACKSLASH, word.c_str());
}

TEST(TextWordGet, Escaping) {
  AutoText input("w" BACKSLASH QUOTE "o" BACKSLASH NEWLINE "r" BACKSLASH ";d");
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  ASSERT_EQ(10, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(10, endPosition.index);
  ASSERT_EQ(input.str, word);
}

TEST(TextWordGet, EscapingEscape) {
  AutoText input("word" BACKSLASH BACKSLASH " abc");
  std::string word;
  spv_position_t startPosition = {};
  spv_position_t endPosition = {};
  ASSERT_EQ(SPV_SUCCESS,
            spvTextWordGet(input, &startPosition, word, &endPosition));
  ASSERT_EQ(6, endPosition.column);
  ASSERT_EQ(0, endPosition.line);
  ASSERT_EQ(6, endPosition.index);
  ASSERT_STREQ("word" BACKSLASH BACKSLASH, word.c_str());
}

}  // anonymous namespace
