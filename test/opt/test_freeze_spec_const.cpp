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

#include "pass_fixture.h"
#include "pass_utils.h"

#include <tuple>

namespace {

using namespace spvtools;

using FreezeSpecConstantValueTest = PassTest<::testing::TestWithParam<
    std::tuple<const char*, const char*, const char*>>>;

TEST_P(FreezeSpecConstantValueTest, PrimaryType) {
  auto& test_case = GetParam();
  std::vector<const char*> text = {
      "OpCapability Shader", "OpMemoryModel Logical GLSL450",
      std::get<0>(test_case), std::get<1>(test_case)};
  std::vector<const char*> expected = {
      "OpCapability Shader", "OpMemoryModel Logical GLSL450",
      std::get<0>(test_case), std::get<2>(test_case)};
  SinglePassRunAndCheck<opt::FreezeSpecConstantValuePass>(
      JoinAllInsts(text), JoinAllInsts(expected));
}

// Test each primary type.
INSTANTIATE_TEST_CASE_P(
    PrimaryTypeSpecConst, FreezeSpecConstantValueTest,
    ::testing::ValuesIn(
        std::vector<std::tuple<const char*, const char*, const char*>>({
            // Type, original spec constant definition, expected frozen spec
            // constants.
            std::make_tuple("%1 = OpTypeInt 32 1", "%2 = OpSpecConstant %1 1",
                            "%2 = OpConstant %1 1"),
            std::make_tuple("%1 = OpTypeInt 32 0", "%2 = OpSpecConstant %1 1",
                            "%2 = OpConstant %1 1"),
            std::make_tuple("%1 = OpTypeFloat 32",
                            "%2 = OpSpecConstant %1 3.14",
                            "%2 = OpConstant %1 3.14"),
            std::make_tuple("%1 = OpTypeFloat 64",
                            "%2 = OpSpecConstant %1 3.1415926",
                            "%2 = OpConstant %1 3.1415926"),
            std::make_tuple("%1 = OpTypeBool", "%2 = OpSpecConstantTrue %1",
                            "%2 = OpConstantTrue %1"),
        })));

}  // anonymous namespace
