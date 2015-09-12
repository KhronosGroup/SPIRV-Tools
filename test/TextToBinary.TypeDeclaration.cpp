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

// Assembler tests for instructions in the "Type-Declaration" section of the
// SPIR-V spec.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;

// Test OpTypePipe

// An example case for OpTypePipe
struct TypePipeCase {
  spv::AccessQualifier value;
  std::string name;
};

using OpTypePipeTest =
    test_fixture::TextToBinaryTestBase<::testing::TestWithParam<TypePipeCase>>;

TEST_P(OpTypePipeTest, AnyAccessQualifier) {
  // TODO(dneto): In Rev31 and later, pipes are opaque, and so the %2, which
  // is the type-of-element operand, should be dropped.
  std::string input = "%1 = OpTypePipe %2 " + GetParam().name;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpTypePipe, {1, 2, GetParam().value})));
}

// clang-format off
#define CASE(NAME) {spv::AccessQualifier##NAME, #NAME}
INSTANTIATE_TEST_CASE_P(TextToBinaryTypePipe, OpTypePipeTest,
                        ::testing::ValuesIn(std::vector<TypePipeCase>{
                            CASE(ReadOnly),
                            CASE(WriteOnly),
                            CASE(ReadWrite),
                        }));
#undef CASE
// clang-format on

// TODO(dneto): OpTypeVoid
// TODO(dneto): OpTypeBool
// TODO(dneto): OpTypeInt
// TODO(dneto): OpTypeFloat
// TODO(dneto): OpTypeVector
// TODO(dneto): OpTypeMatrix
// TODO(dneto): OpTypeImage
// TODO(dneto): OpTypeSampler
// TODO(dneto): OpTypeSampledImage
// TODO(dneto): OpTypeArray
// TODO(dneto): OpTypeRuntimeArray
// TODO(dneto): OpTypeStruct
// TODO(dneto): OpTypeOpaque
// TODO(dneto): OpTypePointer
// TODO(dneto): OpTypeFunction
// TODO(dneto): OpTypeEvent
// TODO(dneto): OpTypeDeviceEvent
// TODO(dneto): OpTypeReserveId
// TODO(dneto): OpTypeQueue
// TODO(dneto): OpTypeForwardPointer

}  // anonymous namespace
