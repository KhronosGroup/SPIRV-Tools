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

// Assembler tests for instructions in the "Control Flow" section of the
// SPIR-V spec.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using spvtest::TextToBinaryTest;
using ::testing::Eq;

// Test OpSelectionMerge

using OpSelectionMergeTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::SelectionControlMask>>>;

TEST_P(OpSelectionMergeTest, AnySingleSelectionControlMask) {
  std::string input = "OpSelectionMerge %1 " + GetParam().name();
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(spv::OpSelectionMerge, {1, GetParam().value()})));
}

// clang-format off
#define CASE(VALUE,NAME) { spv::SelectionControl##VALUE, NAME}
INSTANTIATE_TEST_CASE_P(TextToBinarySelectionMerge, OpSelectionMergeTest,
                        ::testing::ValuesIn(std::vector<EnumCase<spv::SelectionControlMask>>{
                            CASE(MaskNone, "None"),
                            CASE(FlattenMask, "Flatten"),
                            CASE(DontFlattenMask, "DontFlatten"),
                        }));
#undef CASE
// clang-format on

TEST_F(OpSelectionMergeTest, CombinedSelectionControlMask) {
  const std::string input = "OpSelectionMerge %1 Flatten|DontFlatten";
  const uint32_t expected_mask =
      spv::SelectionControlFlattenMask | spv::SelectionControlDontFlattenMask;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpSelectionMerge, {1, expected_mask})));
}

// Test OpLoopMerge

using OpLoopMergeTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::LoopControlMask>>>;

TEST_P(OpLoopMergeTest, AnySingleLoopControlMask) {
  std::string input = "OpLoopMerge %merge %continue " + GetParam().name();
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(spv::OpLoopMerge, {1, 2, GetParam().value()})));
}

// clang-format off
#define CASE(VALUE,NAME) { spv::LoopControl##VALUE, NAME}
INSTANTIATE_TEST_CASE_P(TextToBinaryLoopMerge, OpLoopMergeTest,
                        ::testing::ValuesIn(std::vector<EnumCase<spv::LoopControlMask>>{
                            CASE(MaskNone, "None"),
                            CASE(UnrollMask, "Unroll"),
                            CASE(DontUnrollMask, "DontUnroll"),
                        }));
#undef CASE
// clang-format on

TEST_F(OpLoopMergeTest, CombinedLoopControlMask) {
  const std::string input = "OpLoopMerge %merge %continue Unroll|DontUnroll";
  const uint32_t expected_mask =
      spv::LoopControlUnrollMask | spv::LoopControlDontUnrollMask;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpLoopMerge, {1, 2, expected_mask})));
}

// Test OpSwitch

TEST_F(TextToBinaryTest, SwitchGoodZeroTargets) {
  EXPECT_THAT(CompiledInstructions("OpSwitch %selector %default"),
              Eq(MakeInstruction(spv::OpSwitch, {1, 2})));
}

TEST_F(TextToBinaryTest, SwitchGoodOneTarget) {
  EXPECT_THAT(CompiledInstructions("OpSwitch %selector %default 12 %target0"),
              Eq(MakeInstruction(spv::OpSwitch, {1, 2, 12, 3})));
}

TEST_F(TextToBinaryTest, SwitchGoodTwoTargets) {
  EXPECT_THAT(CompiledInstructions(
                  "OpSwitch %selector %default 12 %target0 42 %target1"),
              Eq(MakeInstruction(spv::OpSwitch, {1, 2, 12, 3, 42, 4})));
}

TEST_F(TextToBinaryTest, SwitchBadMissingSelector) {
  EXPECT_THAT(CompileFailure("OpSwitch"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(TextToBinaryTest, SwitchBadInvalidSelector) {
  EXPECT_THAT(CompileFailure("OpSwitch 12"),
              Eq("Expected id to start with %."));
}

TEST_F(TextToBinaryTest, SwitchBadMissingDefault) {
  EXPECT_THAT(CompileFailure("OpSwitch %selector"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(TextToBinaryTest, SwitchBadInvalidDefault) {
  EXPECT_THAT(CompileFailure("OpSwitch %selector 12"),
              Eq("Expected id to start with %."));
}

TEST_F(TextToBinaryTest, SwitchBadInvalidLiteralDefaultFormat) {
  // The assembler recognizes "OpSwitch %selector %default" as a complete
  // instruction.  Then it tries to parse "%abc" as the start of an
  // assignment form instruction, but can't since it hits the end
  // of stream.
  EXPECT_THAT(CompileFailure("OpSwitch %selector %default %abc"),
              Eq("Expected '=', found end of stream."));
}

TEST_F(TextToBinaryTest, SwitchBadInvalidLiteralCanonicalFormat) {
  EXPECT_THAT(CompileWithFormatFailure("OpSwitch %selector %default %abc",
                                       SPV_ASSEMBLY_SYNTAX_FORMAT_CANONICAL),
              Eq("Expected <opcode> at the beginning of an instruction, found "
                 "'%abc'."));
}

TEST_F(TextToBinaryTest, SwitchBadMissingTarget) {
  EXPECT_THAT(CompileFailure("OpSwitch %selector %default 12"),
              Eq("Expected operand, found end of stream."));
}


// TODO(dneto): OpPhi
// TODO(dneto): OpLoopMerge
// TODO(dneto): OpLabel
// TODO(dneto): OpBranch
// TODO(dneto): OpSwitch
// TODO(dneto): OpKill
// TODO(dneto): OpReturn
// TODO(dneto): OpReturnValue
// TODO(dneto): OpUnreachable
// TODO(dneto): OpLifetimeStart
// TODO(dneto): OpLifetimeStop

}  // anonymous namespace
