// Copyright (c) 2017 Pierre Moreau
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gmock/gmock.h"
#include "linker_test.h"

namespace {

using ::testing::HasSubstr;

class MatchingImportsToExports : public spvtest::LinkerTest {
 public:
  MatchingImportsToExports() { }

  virtual void SetUp() { }
  virtual void TearDown() { }
};

TEST_F(MatchingImportsToExports, Default) {
  const std::string body1 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Import
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";
  const std::string body2 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Export
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";

  spvtest::Binary linked_binary;
  ASSERT_EQ(SPV_SUCCESS, Link({ body1, body2 }, linked_binary))
    << GetErrorMessage();
}

TEST_F(MatchingImportsToExports, NotALibraryExtraExports) {
  const std::string body = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Export
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";

  spvtest::Binary linked_binary;
  ASSERT_EQ(SPV_SUCCESS, Link({ body }, linked_binary))
    << GetErrorMessage();

  const std::string expected_res = R"(%1 = OpTypeFloat 32
%2 = OpVariable %1 Uniform
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  ASSERT_EQ(SPV_SUCCESS, Disassemble(linked_binary, res_body))
    << GetErrorMessage();
  ASSERT_EQ(expected_res, res_body);
}

TEST_F(MatchingImportsToExports, LibraryExtraExports) {
  const std::string body = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Export
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";

  spvtest::Binary linked_binary;
  spvtools::LinkerOptions options;
  options.SetCreateLibrary(true);
  ASSERT_EQ(SPV_SUCCESS, Link({ body }, linked_binary, options))
    << GetErrorMessage();

  const std::string expected_res = R"(OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Export
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";
  std::string res_body;
  SetDisassembleOptions(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER);
  ASSERT_EQ(SPV_SUCCESS, Disassemble(linked_binary, res_body))
    << GetErrorMessage();
  ASSERT_EQ(expected_res, res_body);
}

TEST_F(MatchingImportsToExports, UnresolvedImports) {
  const std::string body1 = R"(
OpCapability Linkage
OpDecorate %1 LinkageAttributes "foo" Import
%2 = OpTypeFloat 32
%1 = OpVariable %2 Uniform
)";
  const std::string body2 = R"()";

  spvtest::Binary linked_binary;
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, Link({ body1, body2 }, linked_binary));
  EXPECT_THAT(GetErrorMessage(), HasSubstr("No export linkage was found for \"foo\"."));
}

}  // anonymous namespace
