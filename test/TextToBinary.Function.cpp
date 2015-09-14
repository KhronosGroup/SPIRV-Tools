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

// Assembler tests for instructions in the "Function" section of the
// SPIR-V spec.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;

// An example case for an enumerated value.
template <typename E>
struct EnumCase {
  E value;
  std::string name;
};

// Test OpFunction

using OpFunctionControlTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::FunctionControlMask>>>;

TEST_P(OpFunctionControlTest, AnySingleFunctionControlMask) {
  std::string input = "%result_id = OpFunction %result_type " +
                      GetParam().name + " %function_type ";
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(spv::OpFunction, {1, 2, GetParam().value, 3})));
}

// clang-format off
#define CASE(VALUE,NAME) { spv::FunctionControl##VALUE, NAME}
INSTANTIATE_TEST_CASE_P(TextToBinaryFunctionTest, OpFunctionControlTest,
                        ::testing::ValuesIn(std::vector<EnumCase<spv::FunctionControlMask>>{
                            CASE(MaskNone, "None"),
                            CASE(InlineMask, "Inline"),
                            CASE(DontInlineMask, "DontInline"),
                            CASE(PureMask, "Pure"),
                            CASE(ConstMask, "Const"),
                        }));
#undef CASE
// clang-format on

// TODO(dneto): Combination of function control masks.

// TODO(dneto): OpFunctionParameter
// TODO(dneto): OpFunctionEnd
// TODO(dneto): OpFunctionCall

}  // anonymous namespace
