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

#include <vector>

#include "opt/module.h"

#include "gtest/gtest.h"

namespace {

using spvtools::ir::Module;

uint32_t GetIdBound(const Module& m) {
  std::vector<uint32_t> binary;
  m.ToBinary(&binary, false);
  // The 5-word header must always exist.
  EXPECT_GE(5u, binary.size());
  // The bound is the fourth word.
  return binary[3];
}

TEST(ModuleTest, SetIdBound) {
  Module m;
  // It's initialized to 0.
  EXPECT_EQ(0u, GetIdBound(m));

  m.SetIdBound(19);
  EXPECT_EQ(19u, GetIdBound(m));

  m.SetIdBound(102);
  EXPECT_EQ(102u, GetIdBound(m));
}

}  // anonymous namespace
