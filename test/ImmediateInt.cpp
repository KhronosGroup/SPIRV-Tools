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

#include <cassert>
#include <vector>

#include <gmock/gmock.h>

#include "TestFixture.h"

namespace {

using ::testing::ElementsAre;
using test_fixture::TextToBinaryTest;

TEST_F(TextToBinaryTest, ImmediateIntOpCode) {
  SetText("!0x00FF00FF");
  ASSERT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[5]);
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, ImmediateIntOperand) {
  SetText("OpCapability !0x00FF00FF");
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&text, opcodeTable, operandTable,
                                         extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[6]);
  spvBinaryDestroy(binary);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

using ImmediateIntTest = TextToBinaryTest;

TEST_F(ImmediateIntTest, AnyWordInSimpleStatement) {
  SpirvVector original = CompileSuccessfully("OpConstant %1 %2 123");
  // TODO(deki): uncomment assertions below and make them pass.
  // EXPECT_EQ(original, CompileSuccessfully("!0x0004002B %1 %2 123"));
  EXPECT_EQ(original, CompileSuccessfully("OpConstant !1 %2 123"));
  // EXPECT_EQ(original, CompileSuccessfully("OpConstant %1 !2 123"));
  EXPECT_EQ(original, CompileSuccessfully("OpConstant %1 %2 !123"));
  // EXPECT_EQ(original, CompileSuccessfully("!0x0004002B %1 !2 123"));
  EXPECT_EQ(original, CompileSuccessfully("OpConstant !1 %2 !123"));
  // EXPECT_EQ(original, CompileSuccessfully("!0x0004002B !1 !2 !123"));
}

TEST_F(ImmediateIntTest, AnyWordInAssignmentStatement) {
  SpirvVector original = CompileSuccessfully("%2 = OpArrayLength %12 %1 123");
  // TODO(deki): uncomment assertions below and make them pass.
  // EXPECT_EQ(original, CompileSuccessfully("!2 = OpArrayLength %12 %1 123"));
  // EXPECT_EQ(original, CompileSuccessfully("%2 = !0x00040044 %12 %1 123"));
  // EXPECT_EQ(original, CompileSuccessfully("%2 = OpArrayLength !12 %1 123"));
  EXPECT_EQ(original, CompileSuccessfully("%2 = OpArrayLength %12 !1 123"));
  EXPECT_EQ(original, CompileSuccessfully("%2 = OpArrayLength %12 %1 !123"));
  // Instead of checking all possible multiple-! combinations, only probe a few.
  EXPECT_EQ(original, CompileSuccessfully("%2 = OpArrayLength %12 !1 !123"));
  // EXPECT_EQ(original, CompileSuccessfully("%2 = !0x00040044 !12 !1 !123"));
  // EXPECT_EQ(original, CompileSuccessfully("!2 = !0x00040044 %12 %1 123"));
}

// Literal integers after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IntegerFollowingImmediate) {
  SpirvVector original = CompileSuccessfully("OpTypeInt %1 8 1");
  // TODO(deki): uncomment assertions below and make them pass.
  // EXPECT_EQ(original, CompileSuccessfully("!0x00040015 1 8 1"));
  // EXPECT_EQ(original, CompileSuccessfully("OpTypeInt !1 8 1"));

  // 64-bit integer literal.
  EXPECT_EQ(CompileSuccessfully("OpConstant %10 %1 5000000000"),
            CompileSuccessfully("OpConstant %10 !1 5000000000"));

  // Negative integer.
  EXPECT_EQ(CompileSuccessfully("OpConstant %10 %1 -123"),
            CompileSuccessfully("OpConstant %10 !1 -123"));
}

// Literal floats after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, FloatFollowingImmediate) {
  EXPECT_EQ(CompileSuccessfully("OpConstant %10 %1 0.123"),
            CompileSuccessfully("OpConstant %10 !1 0.123"));
  EXPECT_EQ(CompileSuccessfully("OpConstant %10 %1 -0.5"),
            CompileSuccessfully("OpConstant %10 !1 -0.5"));
  // 64-bit float.
  EXPECT_EQ(
      CompileSuccessfully(
          "OpConstant %10 %1 9999999999999999999999999999999999999999.9"),
      CompileSuccessfully(
          "OpConstant %10 !1 9999999999999999999999999999999999999999.9"));
}

// Literal strings after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, StringFollowingImmediate) {
  // Try a variety of strings, including empty and single-character.
  for (std::string name : {"", "s", "longish"}) {
    SpirvVector original =
        CompileSuccessfully("OpMemberName %10 4 \"" + name + "\"");
    EXPECT_EQ(original,
              CompileSuccessfully("OpMemberName %10 !4 \"" + name + "\""));
    // TODO(deki): uncomment assertions below and make them pass.
    // EXPECT_EQ(original,
    //           CompileSuccessfully("!0x00040006 !10 4 \"" + name + "\""));
  }
}

// IDs after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IdFollowingImmediate) {
// TODO(deki): uncomment assertions below and make them pass.
#if 0
  EXPECT_EQ(CompileSuccessfully("OpDecorationGroup %123"),
            CompileSuccessfully("!0x00020049 %123"));
  EXPECT_EQ(CompileSuccessfully("OpDecorationGroup %group"),
            CompileSuccessfully("!0x00020049 %group"));
#endif
}

// !<integer> after !<integer> is handled correctly.
TEST_F(ImmediateIntTest, ImmediateFollowingImmediate) {
  SpirvVector original = CompileSuccessfully("OpTypeMatrix %11 %10 7");
  EXPECT_EQ(original, CompileSuccessfully("OpTypeMatrix %11 !10 !7"));
  // TODO(deki): uncomment assertions below and make them pass.
  // EXPECT_EQ(original, CompileSuccessfully("!0x00040018 %11 !10 !7"));
}

TEST_F(ImmediateIntTest, InvalidStatement) {
  EXPECT_THAT(Subvector(CompileSuccessfully("!4 !3 !2 !1"), kFirstInstruction),
              ElementsAre(4, 3, 2, 1));
}

TEST_F(ImmediateIntTest, InvalidStatementBetweenValidOnes) {
  EXPECT_THAT(
      Subvector(CompileSuccessfully("OpTypeFloat %10 32 !5 !6 !7 OpEmitVertex"),
                kFirstInstruction),
      ElementsAre(spvOpcodeMake(3, spv::OpTypeFloat), 10, 32, 5, 6, 7,
                  spvOpcodeMake(1, spv::OpEmitVertex)));
}

TEST_F(ImmediateIntTest, NextOpcodeRecognized) {
  SpirvVector original = CompileSuccessfully(R"(
OpLoad %10 %1 %2 Volatile
OpCompositeInsert %11 %4 %1 %3 0 1 2
)");
  SpirvVector alternate = CompileSuccessfully(R"(
OpLoad %10 %1 %2 !1
OpCompositeInsert %11 %4 %1 %3 0 1 2
)");
  EXPECT_EQ(original, alternate);
}

TEST_F(ImmediateIntTest, WrongLengthButNextOpcodeStillRecognized) {
  SpirvVector original = CompileSuccessfully(R"(
OpLoad %10 %1 %2 Volatile
OpCopyMemorySized %3 %4 %1
)");
// TODO(deki): uncomment assertions below and make them pass.
#if 0
  SpirvVector alternate = CompileSuccessfully(R"(
!0x0002003D %10 %1 %2 !1
OpCopyMemorySized %3 %4 %1
)");
  EXPECT_EQ(0x0002003D, alternate[kFirstInstruction]);
  EXPECT_EQ(Subvector(original, kFirstInstruction + 1),
            Subvector(alternate, kFirstInstruction + 1));
#endif
}

// Like NextOpcodeRecognized, but next statement is in assignment form.
TEST_F(ImmediateIntTest, NextAssignmentRecognized) {
  SpirvVector original = CompileSuccessfully(R"(
OpLoad %10 %1 %2 None
%4 = OpFunctionCall %10 %3 123
)");
  SpirvVector alternate = CompileSuccessfully(R"(
OpLoad %10 !1 %2 !0
%4 = OpFunctionCall %10 %3 123
)");
  EXPECT_EQ(original, alternate);
}

// Two instructions in a row each have !<integer> opcode.
TEST_F(ImmediateIntTest, ConsecutiveImmediateOpcodes) {
  SpirvVector original = CompileSuccessfully(R"(
OpConstantSampler %10 %1 Clamp 78 Linear
OpFRem %11 %4 %3 %2
%5 = OpIsValidEvent %12 %2
)");
// TODO(deki): uncomment assertions below and make them pass.
#if 0
  SpirvVector alternate = CompileSuccessfully(R"(
!0x0006002D %10 %1 !2 78 !1
!0x0005008C %11 %4 %3 %2
%5 = OpIsValidEvent %12 %2
)");
  EXPECT_EQ(original, alternate);
#endif
}

// TODO(deki): implement all tests below.

TEST_F(ImmediateIntTest, ForbiddenOperands) {
  // !<integer> followed by, eg, an enum or '=' or a random bareword.
}

}  // anonymous namespace
