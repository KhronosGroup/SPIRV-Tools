// Copyright (c) 2016 Google Inc.
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

#include <gtest/gtest.h>

#include "pass_utils.h"

namespace {

using namespace spvtools;

TEST(JoinAllInsts, Cases) {
  EXPECT_EQ("", JoinAllInsts({}));
  EXPECT_EQ("a\n", JoinAllInsts({"a"}));
  EXPECT_EQ("a\nb\n", JoinAllInsts({"a", "b"}));
  EXPECT_EQ("a\nb\nc\n", JoinAllInsts({"a", "b", "c"}));
  EXPECT_EQ("hello,\nworld!\n\n\n", JoinAllInsts({"hello,", "world!", "\n"}));
}

TEST(JoinNonDebugInsts, Cases) {
  EXPECT_EQ("", JoinNonDebugInsts({}));
  EXPECT_EQ("a\n", JoinNonDebugInsts({"a"}));
  EXPECT_EQ("", JoinNonDebugInsts({"OpName"}));
  EXPECT_EQ("a\nb\n", JoinNonDebugInsts({"a", "b"}));
  EXPECT_EQ("", JoinNonDebugInsts({"OpName", "%1 = OpString \"42\""}));
  EXPECT_EQ("Opstring\n", JoinNonDebugInsts({"OpName", "Opstring"}));
  EXPECT_EQ("the only remaining string\n",
            JoinNonDebugInsts(
                {"OpSourceContinued", "OpSource", "OpSourceExtension",
                 "lgtm OpName", "hello OpMemberName", "this is a OpString",
                 "lonely OpLine", "happy OpNoLine", "OpModuleProcessed",
                 "the only remaining string"}));
}

}  // anonymous namespace
