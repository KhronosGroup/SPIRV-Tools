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

#include "gmock/gmock.h"

#include "TestFixture.h"
#include "UnitSPIRV.h"

#include "source/name_mapper.h"

using libspirv::NameMapper;
using libspirv::FriendlyNameMapper;
using spvtest::ScopedContext;
using ::testing::Eq;

namespace {

TEST(TrivialNameTest, Samples) {
  auto mapper = libspirv::GetTrivialNameMapper();
  EXPECT_THAT(mapper(1), "1");
  EXPECT_THAT(mapper(1999), "1999");
  EXPECT_THAT(mapper(1024), "1024");
}

// A test case for the name mappers that actually look at an assembled module.
struct NameIdCase {
  std::string assembly;  // Input assembly text
  uint32_t id;
  std::string expected_name;
};

using FriendlyNameTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<NameIdCase>>;

TEST_P(FriendlyNameTest, SingleMapping) {
  ScopedContext context;
  auto words = CompileSuccessfully(GetParam().assembly);
  auto friendly_mapper =
      FriendlyNameMapper(context.context, words.data(), words.size());
  NameMapper mapper = friendly_mapper.GetNameMapper();
  EXPECT_THAT(mapper(GetParam().id), Eq(GetParam().expected_name))
      << GetParam().assembly << std::endl
      << " for id " << GetParam().id;
}

INSTANTIATE_TEST_CASE_P(ScalarType, FriendlyNameTest,
                        ::testing::ValuesIn(std::vector<NameIdCase>{
                            {"%1 = OpTypeVoid", 1, "void"},
                            // Verify uniqueness heuristics
                            {"%1 = OpTypeVoid %2 = OpTypeVoid", 1, "void"},
                            {"%1 = OpTypeVoid %2 = OpTypeVoid", 2, "void_0"},
                            {"%1 = OpTypeVoid %2 = OpTypeVoid %3 = OpTypeVoid",
                             3, "void_1"},
                            // TODO(dneto): Fill out others
                        }), );

}  // anonymous namespace
