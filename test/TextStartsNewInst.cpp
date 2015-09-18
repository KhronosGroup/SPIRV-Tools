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

TEST(TextStartsWithOp, YesAtStart) {
  spv_position_t startPosition = {};
  EXPECT_TRUE(spvTextIsStartOfNewInst(AutoText("OpFoo"), &startPosition));
  EXPECT_TRUE(spvTextIsStartOfNewInst(AutoText("OpFoo"), &startPosition));
  EXPECT_TRUE(spvTextIsStartOfNewInst(AutoText("OpEnCL"), &startPosition));
}

TEST(TextStartsWithOp, YesAtMiddle) {
  spv_position_t startPosition = {};
  startPosition.index = 2;
  EXPECT_TRUE(spvTextIsStartOfNewInst(AutoText("  OpFoo"), &startPosition));
  EXPECT_TRUE(spvTextIsStartOfNewInst(AutoText("    OpFoo"), &startPosition));
}

TEST(TextStartsWithOp, NoIfTooFar) {
  spv_position_t startPosition = {};
  startPosition.index = 3;
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("  OpFoo"), &startPosition));
}

TEST(TextStartsWithOp, NoRegular) {
  spv_position_t startPosition = {};
  EXPECT_FALSE(
      spvTextIsStartOfNewInst(AutoText("Fee Fi Fo Fum"), &startPosition));
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("123456"), &startPosition));
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("123456"), &startPosition));
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("OpenCL"), &startPosition));
}

TEST(TextStartsWithOp, YesForValueGenerationForm) {
  spv_position_t startPosition = {};
  EXPECT_TRUE(
      spvTextIsStartOfNewInst(AutoText("%foo = OpAdd"), &startPosition));
  EXPECT_TRUE(
      spvTextIsStartOfNewInst(AutoText("%foo  =  OpAdd"), &startPosition));
}

TEST(TextStartsWithOp, NoForNearlyValueGeneration) {
  spv_position_t startPosition = {};
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("%foo = "), &startPosition));
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("%foo "), &startPosition));
  EXPECT_FALSE(spvTextIsStartOfNewInst(AutoText("%foo"), &startPosition));
}
