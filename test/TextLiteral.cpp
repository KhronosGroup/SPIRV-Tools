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

#include <string>

TEST(TextLiteral, GoodI32) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_32, l.type);
  EXPECT_EQ(0, l.value.i32);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-2147483648", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_32, l.type);
  EXPECT_EQ(-2147483648, l.value.i32);
}

TEST(TextLiteral, GoodU32) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_32, l.type);
  EXPECT_EQ(0, l.value.i32);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("4294967295", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_32, l.type);
  EXPECT_EQ(4294967295, l.value.u32);
}

TEST(TextLiteral, GoodI64) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-2147483649", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_INT_64, l.type);
  EXPECT_EQ(-2147483649, l.value.i64);
}

TEST(TextLiteral, GoodU64) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("4294967296", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_UINT_64, l.type);
  EXPECT_EQ(4294967296, l.value.u64);
}

TEST(TextLiteral, GoodFloat) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("1.0", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(1.0, l.value.f);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("1.5", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(1.5, l.value.f);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("-.25", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_FLOAT_32, l.type);
  EXPECT_EQ(-.25, l.value.f);
}

TEST(TextLiteral, BadString) {
  spv_literal_t l;

  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("-", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("--", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("1-2", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("123a", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("12.2.3", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("\"", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("\"z", &l));
  EXPECT_EQ(SPV_FAILED_MATCH, spvTextToLiteral("a\"", &l));
}

TEST(TextLiteral, GoodString) {
  spv_literal_t l;

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("\"-\"", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ("-", l.value.str);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("\"--\"", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ("--", l.value.str);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("\"1-2\"", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ("1-2", l.value.str);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("\"123a\"", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ("123a", l.value.str);

  ASSERT_EQ(SPV_SUCCESS, spvTextToLiteral("\"12.2.3\"", &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ("12.2.3", l.value.str);

  // TODO(dneto): escaping in strings is not supported yet.
}

TEST(TextLiteral, StringTooLong) {
  spv_literal_t l;
  std::string too_long = std::string("\"") +
                         std::string(SPV_LIMIT_LITERAL_STRING_MAX - 2, 'a') +
                         "\"";
  EXPECT_EQ(SPV_ERROR_OUT_OF_MEMORY, spvTextToLiteral(too_long.data(), &l));
}

TEST(TextLiteral, GoodLongString) {
  spv_literal_t l;
  std::string unquoted(SPV_LIMIT_LITERAL_STRING_MAX - 3, 'a');
  std::string good_long = std::string("\"") + unquoted + "\"";
  EXPECT_EQ(SPV_SUCCESS, spvTextToLiteral(good_long.data(), &l));
  EXPECT_EQ(SPV_LITERAL_TYPE_STRING, l.type);
  EXPECT_STREQ(unquoted.data(), l.value.str);
}
