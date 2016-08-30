// Copyright (c) 2016 Google Inc.
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

#include "pass_fixture.h"

namespace {
using namespace spvtools;

using SpecIdToValueStrMap =
    opt::SetSpecConstantDefaultValuePass::SpecIdToValueStrMap;

struct SetSpecConstantDefaultValueTestCase {
  const char* code;
  SpecIdToValueStrMap default_values;
  const char* expected;
};

using SetSpecConstantDefaultValueParamTest =
    PassTest<::testing::TestWithParam<SetSpecConstantDefaultValueTestCase>>;

TEST_P(SetSpecConstantDefaultValueParamTest, TestCase) {
  const auto& tc = GetParam();
  SinglePassRunAndCheck<opt::SetSpecConstantDefaultValuePass>(
      tc.code, tc.expected, /* skip_nop = */ false, tc.default_values);
}

INSTANTIATE_TEST_CASE_P(
    ValidCases, SetSpecConstantDefaultValueParamTest,
    ::testing::ValuesIn(std::vector<SetSpecConstantDefaultValueTestCase>{
        // 0. Empty.
        {"", SpecIdToValueStrMap{}, ""},
        // 1. Empty with non-empty values to set.
        {"", SpecIdToValueStrMap{{1, "100"}, {2, "200"}}, ""},
        // 2. Bool type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueStrMap{{100, "false"}, {101, "true"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantFalse %bool\n"
            "%2 = OpSpecConstantTrue %bool\n",
        },
        // 3. 32-bit int type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %int 11\n"
            "%3 = OpSpecConstant %int 11\n",
            // default values
            SpecIdToValueStrMap{
                {100, "2147483647"}, {101, "0xffffffff"}, {102, "-42"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "OpDecorate %3 SpecId 102\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 2147483647\n"
            "%2 = OpSpecConstant %int -1\n"
            "%3 = OpSpecConstant %int -42\n",
        },
        // 4. 64-bit uint type.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueStrMap{{100, "18446744073709551614"}, {101, "0x100"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %ulong 18446744073709551614\n"
            "%2 = OpSpecConstant %ulong 256\n",
        },
        // 5. 32-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float 200\n"
            "%2 = OpSpecConstant %float 201\n",
            // default values
            SpecIdToValueStrMap{{101, "-0x1.fffffep+128"}, {102, "2.5"}},
            // expected
            "OpDecorate %1 SpecId 101\n"
            "OpDecorate %2 SpecId 102\n"
            "%float = OpTypeFloat 32\n"
            "%1 = OpSpecConstant %float -0x1.fffffep+128\n"
            "%2 = OpSpecConstant %float 2.5\n",
        },
        // 6. 64-bit float type.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %double 0.142857\n",
            // default values
            SpecIdToValueStrMap{{201, "0x1.fffffffffffffp+1024"},
                                {202, "-32.5"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %double -32.5\n",
        },
        // 7. SpecId not found, expect no modification.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
            // default values
            SpecIdToValueStrMap{{8888, "0.0"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n",
        },
        // 8. Multiple types of spec constants.
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 3.14159265358979\n"
            "%2 = OpSpecConstant %int 1024\n"
            "%3 = OpSpecConstantTrue %bool\n",
            // default values
            SpecIdToValueStrMap{
                {201, "0x1.fffffffffffffp+1024"}, {202, "2048"}, {203, "false"},
            },
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "OpDecorate %3 SpecId 203\n"
            "%bool = OpTypeBool\n"
            "%int = OpTypeInt 32 1\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %double 0x1.fffffffffffffp+1024\n"
            "%2 = OpSpecConstant %int 2048\n"
            "%3 = OpSpecConstantFalse %bool\n",
        },
        // 9. Ignore other decorations.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
        },
        // 10. Distinguish from other decorations.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}, {100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpSpecConstant %int -1\n",
        },
        // 11. Decorate through decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 12. Ignore other decorations in decoration group.
        {
            // code
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{4, "0x7fffffff"}},
            // expected
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
        },
        // 13. Distinguish from other decorations in decoration group.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}, {4, "0x00000001"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %1 ArrayStride 4\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 2147483647\n",
        },
        // 14. Unchanged bool default value
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
            // default values
            SpecIdToValueStrMap{{100, "true"}, {101, "false"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%bool = OpTypeBool\n"
            "%1 = OpSpecConstantTrue %bool\n"
            "%2 = OpSpecConstantFalse %bool\n",
        },
        // 15. Unchanged int default values
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
            // default values
            SpecIdToValueStrMap{{100, "10"}, {101, "11"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "OpDecorate %2 SpecId 101\n"
            "%int = OpTypeInt 32 1\n"
            "%ulong = OpTypeInt 64 0\n"
            "%1 = OpSpecConstant %int 10\n"
            "%2 = OpSpecConstant %ulong 11\n",
        },
        // 16. Unchanged float default values
        {
            // code
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.14159\n"
            "%2 = OpSpecConstant %double 0.142857\n",
            // default values
            SpecIdToValueStrMap{{201, "3.14159"}, {202, "0.142857"}},
            // expected
            "OpDecorate %1 SpecId 201\n"
            "OpDecorate %2 SpecId 202\n"
            "%float = OpTypeFloat 32\n"
            "%double = OpTypeFloat 64\n"
            "%1 = OpSpecConstant %float 3.14159\n"
            "%2 = OpSpecConstant %double 0.142857\n",
        },
        // 17. OpGroupDecorate may have multiple target ids defined by the same
        // eligible spec constant
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %2 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int -1\n",
        },
    }));

INSTANTIATE_TEST_CASE_P(
    InvalidCases, SetSpecConstantDefaultValueParamTest,
    ::testing::ValuesIn(std::vector<SetSpecConstantDefaultValueTestCase>{
        // 0. Do not crash when decoration group is not used.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 100\n",
        },
        // 1. Do not crash when target does not exist.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n",
        },
        // 2. Do nothing when SpecId decoration is not attached to a
        // non-spec-contant instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpConstant %int 101\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%1 = OpConstant %int 101\n",
        },
        // 3. Do nothing when SpecId decoration is not attached to a
        // OpSpecConstant{|True|False} instruction.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
            // default values
            SpecIdToValueStrMap{{100, "0x7fffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%int = OpTypeInt 32 1\n"
            "%3 = OpSpecConstant %int 101\n"
            "%1 = OpSpecConstantOp %int IAdd %3 %3\n",
        },
        // 4. Do not crash and do nothing when SpecId decoration is applied to
        // multiple spec constants.
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2 %3 %4\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpSpecConstant %int 100\n"
            "%3 = OpSpecConstant %int 200\n"
            "%4 = OpSpecConstant %int 300\n",
        },
        // 5. Do not crash and do nothing when SpecId decoration is attached to
        // non-spec-constants (invalid case).
        {
            // code
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpConstant %int 100\n",
            // default values
            SpecIdToValueStrMap{{100, "0xffffffff"}},
            // expected
            "OpDecorate %1 SpecId 100\n"
            "%1 = OpDecorationGroup\n"
            "OpGroupDecorate %1 %2\n"
            "%int = OpTypeInt 32 1\n"
            "%2 = OpConstant %int 100\n",
        },
    }));

}  // anonymous namespace
