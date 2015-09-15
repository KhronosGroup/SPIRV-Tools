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

// Assembler tests for instructions in the "Memory Instructions" section of
// the SPIR-V spec.

#include "UnitSPIRV.h"

#include <sstream>

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;

// An example case for an enumerated value.
template <typename E>
struct EnumCaseWithOperands {
  E value;
  std::string name;
  std::vector<uint32_t> operands;
};

// Test assembly of Memory Access masks

using MemoryAccessTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCaseWithOperands<spv::MemoryAccessMask>>>;

TEST_P(MemoryAccessTest, AnySingleMemoryAccessMask) {
  std::stringstream input;
  input << "OpStore %ptr %value " << GetParam().name;
  for (auto operand : GetParam().operands) input << " " << operand;
  std::vector<uint32_t> expected_operands{1, 2, GetParam().value};
  expected_operands.insert(expected_operands.end(), GetParam().operands.begin(),
                           GetParam().operands.end());
  EXPECT_THAT(CompiledInstructions(input.str()),
              Eq(MakeInstruction(spv::OpStore, expected_operands)));
}

// clang-format off
INSTANTIATE_TEST_CASE_P(TextToBinaryMemoryAccessTest, MemoryAccessTest,
                        ::testing::ValuesIn(std::vector<EnumCaseWithOperands<spv::MemoryAccessMask>>{
                          {spv::MemoryAccessMaskNone, "None", {}},
                          {spv::MemoryAccessVolatileMask, "Volatile", {}},
                          {spv::MemoryAccessAlignedMask, "Aligned", {16}},
                        }));
#undef CASE
// clang-format on

// Test Storage Class enum values

using StorageClassTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCaseWithOperands<spv::StorageClass>>>;

TEST_P(StorageClassTest, AnyStorageClass) {
  std::string input = "%1 = OpVariable %2 " + GetParam().name;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpVariable, {2, 1, GetParam().value})));
}

// clang-format off
#define CASE(NAME) { spv::StorageClass##NAME, #NAME, {} }
INSTANTIATE_TEST_CASE_P(TextToBinaryStorageClassTest, StorageClassTest,
                        ::testing::ValuesIn(std::vector<EnumCaseWithOperands<spv::StorageClass>>{
                          // TODO(dneto): There are more storage classes in Rev32 and later.
                          CASE(UniformConstant),
                          CASE(Input),
                          CASE(Uniform),
                          CASE(Output),
                          CASE(WorkgroupLocal),
                          CASE(WorkgroupGlobal),
                          CASE(PrivateGlobal),
                          CASE(Function),
                          CASE(Generic),
                          CASE(AtomicCounter),
                          CASE(Image),
                        }));
#undef CASE
// clang-format on

// TODO(dneto): Combination of memory access masks.

// TODO(dneto): OpVariable with initializers
// TODO(dneto): OpImageTexelPointer
// TODO(dneto): OpLoad
// TODO(dneto): OpStore
// TODO(dneto): OpCopyMemory
// TODO(dneto): OpCopyMemorySized
// TODO(dneto): OpAccessChain
// TODO(dneto): OpInBoundsAccessChain
// TODO(dneto): OpPtrAccessChain
// TODO(dneto): OpArrayLength
// TODO(dneto): OpGenercPtrMemSemantics

}  // anonymous namespace
