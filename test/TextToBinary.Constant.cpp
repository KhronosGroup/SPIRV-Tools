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

// Assembler tests for instructions in the "Group Instrucions" section of the
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

// Test Sampler Addressing Mode enum values

using SamplerAddressingModeTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::SamplerAddressingMode>>>;

TEST_P(SamplerAddressingModeTest, AnySamplerAddressingMode) {
  std::string input =
      "%result = OpConstantSampler %type " + GetParam().name + " 0 Nearest";
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpConstantSampler,
                                 {1, 2, GetParam().value, 0, 0})));
}

// clang-format off
#define CASE(NAME) { spv::SamplerAddressingMode##NAME, #NAME }
INSTANTIATE_TEST_CASE_P(
    TextToBinarySamplerAddressingMode, SamplerAddressingModeTest,
    ::testing::ValuesIn(std::vector<EnumCase<spv::SamplerAddressingMode>>{
        CASE(None),
        CASE(ClampToEdge),
        CASE(Clamp),
        CASE(Repeat),
        CASE(RepeatMirrored),
    }));
#undef CASE
// clang-format on

// Test Sampler Filter Mode enum values

using SamplerFilterModeTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::SamplerFilterMode>>>;

TEST_P(SamplerFilterModeTest, AnySamplerFilterMode) {
  std::string input =
      "%result = OpConstantSampler %type Clamp 0 " + GetParam().name;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpConstantSampler,
                                 {1, 2, 2, 0, GetParam().value})));
}

// clang-format off
#define CASE(NAME) { spv::SamplerFilterMode##NAME, #NAME}
INSTANTIATE_TEST_CASE_P(
    TextToBinarySamplerFilterMode, SamplerFilterModeTest,
    ::testing::ValuesIn(std::vector<EnumCase<spv::SamplerFilterMode>>{
        CASE(Nearest),
        CASE(Linear),
    }));
#undef CASE
// clang-format on

// TODO(dneto): OpConstantTrue
// TODO(dneto): OpConstantFalse
// TODO(dneto): OpConstant
// TODO(dneto): OpConstantComposite
// TODO(dneto): OpConstantSampler: other variations Param is 0 or 1
// TODO(dneto): OpConstantNull
// TODO(dneto): OpSpecConstantTrue
// TODO(dneto): OpSpecConstantFalse
// TODO(dneto): OpSpecConstant
// TODO(dneto): OpSpecConstantComposite
// TODO(dneto): OpSpecConstantOp

}  // anonymous namespace
