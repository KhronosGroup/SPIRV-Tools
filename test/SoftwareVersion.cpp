// Copyright (c) 2015-2016 Google Inc.
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

#include "gmock/gmock.h"

using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Ge;

namespace {

TEST(SoftwareVersion, CorrectForm) {
  const std::string version(spvSoftwareVersionString());
  std::istringstream s(version);
  char v = 'x';
  int year = -1;
  char period = 'x';
  int index = -1;
  s >> v >> year >> period >> index;
  EXPECT_THAT(v, Eq('v'));
  EXPECT_THAT(year, Ge(2016));
  EXPECT_THAT(period, Eq('.'));
  EXPECT_THAT(index, Ge(0));
  EXPECT_TRUE(s.good() || s.eof());

  std::string rest;
  s >> rest;
  EXPECT_THAT(rest, AnyOf("", "wip"));
}

}  // anonymous namespace
