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

// TODO(deki): uncomment assertions below and make them pass.
TEST_F(ImmediateIntTest, AnyWordInSimpleStatement) {
  SpirvVector original = CompileSuccessfully("OpConstant %1 %2 123");
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
  EXPECT_EQ(original, CompileSuccessfully("%2 = OpArrayLength %12 %1 123"));
}

TEST_F(ImmediateIntTest, InvalidStatement) {
  EXPECT_THAT(Subvector(CompileSuccessfully("!4 !3 !2 !1"), kFirstInstruction),
              ElementsAre(4, 3, 2, 1));
}

TEST_F(ImmediateIntTest, InvalidStatementBetweenValidOnes) {
  EXPECT_THAT(
      Subvector(CompileSuccessfully("OpTypeFloat %10 32 !5 !6 !7 OpEmitVertex"),
                kFirstInstruction),
      ElementsAre(0x00030016, 10, 32, 5, 6, 7, 0x000100DA));
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

// TODO(deki): implement.
TEST_F(ImmediateIntTest, NextAssignmentRecognized) {
  // Like NextOpcodeRecognized, but next statement is in assignment form.
}

TEST_F(ImmediateIntTest, ConsecutiveImmediateOpcodes) {
  // Two instructions in a row each have !<integer> opcode.
}

TEST_F(ImmediateIntTest, LiteralOperands) {
  // !<integer> followed by a literal-number operand.
  // !<integer> followed by a literal-string operand.
  // Combos thereof.
}

TEST_F(ImmediateIntTest, IdOperands) {
  // !<integer> followed by ID operand(s).
}

TEST_F(ImmediateIntTest, ImmediateOperands) {
  // !<integer> followed by !<integer> operand(s).
}

TEST_F(ImmediateIntTest, ForbiddenOperands) {
  // !<integer> followed by, eg, an enum or '=' or a random bareword.
}

// NB: when/if these cases are handled, it will require reworking the
// description in readme.md, which currently dictates that each word past
// !<integer> must be a literal, an ID, or another immediate (ie, not a '=').

TEST_F(ImmediateIntTest, AssignmentLHS) {
  // !<integer> = OpIAdd %i32 %op0 %op1
}

TEST_F(ImmediateIntTest, AssignmentLHSAndOpCode) {
  // !<integer> = !<integer> %i32 %op0 %op1
}

}  // anonymous namespace
