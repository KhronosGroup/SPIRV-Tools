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

// Assembler tests for instructions in the "Debug" section of the
// SPIR-V spec.

#include "UnitSPIRV.h"

#include <string>

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using spvtest::MakeVector;
using ::testing::Eq;

// Test OpSource

// A single test case for OpSource
struct LanguageCase {
  uint32_t get_language_value() const {
    return static_cast<uint32_t>(language_value);
  }
  const char* language_name;
  spv::SourceLanguage language_value;
  uint32_t version;
};

// clang-format off
// The list of OpSource cases to use.
const LanguageCase kLanguageCases[] = {
#define CASE(NAME, VERSION) \
  { #NAME, spv::SourceLanguage##NAME, VERSION }
  CASE(Unknown, 0),
  CASE(Unknown, 999),
  CASE(ESSL, 310),
  CASE(GLSL, 450),
  CASE(OpenCL, 210),
#undef CASE
};
// clang-format on

using OpSourceTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<LanguageCase>>;

TEST_P(OpSourceTest, AnyLanguage) {
  std::string input = std::string("OpSource ") + GetParam().language_name +
                      " " + std::to_string(GetParam().version);
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpSource, {GetParam().get_language_value(),
                                                 GetParam().version})));
}

INSTANTIATE_TEST_CASE_P(TextToBinaryTestDebug, OpSourceTest,
                        ::testing::ValuesIn(kLanguageCases));

// Test OpSourceExtension

using OpSourceExtensionTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<const char*>>;

TEST_P(OpSourceExtensionTest, AnyExtension) {
  // TODO(dneto): utf-8, quoting, escaping
  std::string input = std::string("OpSourceExtension \"") + GetParam() + "\"";

  const std::vector<uint32_t> encoded_string = MakeVector(GetParam());
  std::vector<uint32_t> expected_instruction{
      spvOpcodeMake(encoded_string.size() + 1, spv::OpSourceExtension)};
  expected_instruction.insert(expected_instruction.end(),
                              encoded_string.begin(), encoded_string.end());
  EXPECT_THAT(CompiledInstructions(input), Eq(expected_instruction));
}

// TODO(dneto): utf-8, quoting, escaping
INSTANTIATE_TEST_CASE_P(TextToBinaryTestDebug, OpSourceExtensionTest,
                        ::testing::ValuesIn(std::vector<const char*>{
                            "", "foo bar this and that"}));

// TODO(dneto): OpName
// TODO(dneto): OpMemberName
// TODO(dneto): OpString
// TODO(dneto): OpLine.  OpLine is significantly different after Rev31.

}  // anonymous namespace
