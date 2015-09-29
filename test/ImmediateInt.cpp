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
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using spvtest::TextToBinaryTest;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::StrEq;

const auto kCAF = SPV_ASSEMBLY_SYNTAX_FORMAT_CANONICAL;

TEST_F(TextToBinaryTest, ImmediateIntOpCode) {
  SetText("!0x00FF00FF");
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[5]);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

TEST_F(TextToBinaryTest, ImmediateIntOperand) {
  SetText("OpCapability !0x00FF00FF");
  EXPECT_EQ(SPV_SUCCESS,
            spvTextToBinary(text.str, text.length, opcodeTable, operandTable,
                            extInstTable, &binary, &diagnostic));
  EXPECT_EQ(0x00FF00FF, binary->code[6]);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
  }
}

using ImmediateIntTest = TextToBinaryTest;

TEST_F(ImmediateIntTest, AnyWordInSimpleStatement) {
  EXPECT_THAT(CompiledInstructions("!0x0004002B %a %b 123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("OpConstant !1 %b 123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 1, 123})));
  EXPECT_THAT(CompiledInstructions("OpConstant %1 !2 123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("OpConstant  %a %b !123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("!0x0004002B %1 !2 123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("OpConstant !1 %b !123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 1, 123})));
  EXPECT_THAT(CompiledInstructions("!0x0004002B !1 !2 !123", kCAF),
              Eq(MakeInstruction(spv::OpConstant, {1, 2, 123})));
}

TEST_F(ImmediateIntTest, AnyWordAfterEqualsAndOpCode) {
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 %c 123"),
              Eq(MakeInstruction(spv::OpArrayLength, {2, 1, 2, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b !3 123"),
              Eq(MakeInstruction(spv::OpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b %c !123"),
              Eq(MakeInstruction(spv::OpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength %b !3 !123"),
              Eq(MakeInstruction(spv::OpArrayLength, {1, 2, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 !3 123"),
              Eq(MakeInstruction(spv::OpArrayLength, {2, 1, 3, 123})));
  EXPECT_THAT(CompiledInstructions("%a = OpArrayLength !2 !3 !123"),
              Eq(MakeInstruction(spv::OpArrayLength, {2, 1, 3, 123})));
}

TEST_F(ImmediateIntTest, ResultIdInAssignment) {
  EXPECT_EQ("!2 not allowed before =.",
            CompileFailure("!2 = OpArrayLength %12 %1 123"));
  EXPECT_EQ("!2 not allowed before =.",
            CompileFailure("!2 = !0x00040044 %12 %1 123"));
}

TEST_F(ImmediateIntTest, OpCodeInAssignment) {
  EXPECT_EQ("Invalid Opcode prefix '!0x00040044'.",
            CompileFailure("%2 = !0x00040044 %12 %1 123"));
}

// Literal integers after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IntegerFollowingImmediate) {
  const SpirvVector original = CompiledInstructions("OpTypeInt %1 8 1", kCAF);
  EXPECT_EQ(original, CompiledInstructions("!0x00040015 1 8 1", kCAF));
  EXPECT_EQ(original, CompiledInstructions("OpTypeInt !1 8 1", kCAF));

  // 64-bit integer literal.
  EXPECT_EQ(CompiledInstructions("OpConstant %10 %2 5000000000", kCAF),
            CompiledInstructions("OpConstant %10 !2 5000000000", kCAF));

  // Negative integer.
  EXPECT_EQ(CompiledInstructions("OpConstant %10 %2 -123", kCAF),
            CompiledInstructions("OpConstant %10 !2 -123", kCAF));

  // TODO(deki): uncomment assertions below and make them pass.
  // Hex value(s).
  // EXPECT_EQ(CompileSuccessfully("OpConstant %10 %1 0x12345678", kCAF),
  //           CompileSuccessfully("OpConstant %10 !1 0x12345678", kCAF));
  // EXPECT_EQ(
  //     CompileSuccessfully("OpConstant %10 %1 0x12345678 0x87654321", kCAF),
  //     CompileSuccessfully("OpConstant %10 !1 0x12345678 0x87654321", kCAF));
}

// Literal floats after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, FloatFollowingImmediate) {
  EXPECT_EQ(CompiledInstructions("OpConstant %10 %2 0.123", kCAF),
            CompiledInstructions("OpConstant %10 !2 0.123", kCAF));
  EXPECT_EQ(CompiledInstructions("OpConstant %10 %2 -0.5", kCAF),
            CompiledInstructions("OpConstant %10 !2 -0.5", kCAF));
  // 64-bit float.
  EXPECT_EQ(
      CompiledInstructions(
          "OpConstant %10 %2 9999999999999999999999999999999999999999.9", kCAF),
      CompiledInstructions(
          "OpConstant %10 !2 9999999999999999999999999999999999999999.9",
          kCAF));
}

// Literal strings after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, StringFollowingImmediate) {
  // Try a variety of strings, including empty and single-character.
  for (std::string name : {"", "s", "longish", "really looooooooooooooooong"}) {
    const SpirvVector original =
        CompiledInstructions("OpMemberName %10 4 \"" + name + "\"", kCAF);
    EXPECT_EQ(original, CompiledInstructions(
                            "OpMemberName %10 !4 \"" + name + "\"", kCAF))
        << name;
    EXPECT_EQ(original,
              CompiledInstructions("OpMemberName !1 !4 \"" + name + "\"", kCAF))
        << name;
    const uint32_t wordCount = 4 + name.size() / 4;
    const uint32_t firstWord = spvOpcodeMake(wordCount, spv::OpMemberName);
    EXPECT_EQ(original, CompiledInstructions("!" + std::to_string(firstWord) +
                                                 " %10 !4 \"" + name + "\"",
                                             kCAF))
        << name;
  }
}

// IDs after !<integer> are handled correctly.
TEST_F(ImmediateIntTest, IdFollowingImmediate) {
  EXPECT_EQ(CompileSuccessfully("OpDecorationGroup %123", kCAF),
            CompileSuccessfully("!0x00020049 %123", kCAF));
  EXPECT_EQ(CompileSuccessfully("OpDecorationGroup %group", kCAF),
            CompileSuccessfully("!0x00020049 %group", kCAF));
}

// !<integer> after !<integer> is handled correctly.
TEST_F(ImmediateIntTest, ImmediateFollowingImmediate) {
  const SpirvVector original =
      CompiledInstructions("OpTypeMatrix %a %b 7", kCAF);
  EXPECT_EQ(original, CompiledInstructions("OpTypeMatrix %a !2 !7", kCAF));
  EXPECT_EQ(original, CompiledInstructions("!0x00040018 %a !2 !7", kCAF));
}

TEST_F(ImmediateIntTest, InvalidStatement) {
  EXPECT_THAT(
      Subvector(CompileSuccessfully("!4 !3 !2 !1", kCAF), kFirstInstruction),
      ElementsAre(4, 3, 2, 1));
}

TEST_F(ImmediateIntTest, InvalidStatementBetweenValidOnes) {
  EXPECT_THAT(Subvector(CompileSuccessfully(
                            "OpTypeFloat %10 32 !5 !6 !7 OpEmitVertex", kCAF),
                        kFirstInstruction),
              ElementsAre(spvOpcodeMake(3, spv::OpTypeFloat), 1, 32, 5, 6, 7,
                          spvOpcodeMake(1, spv::OpEmitVertex)));
}

TEST_F(ImmediateIntTest, NextOpcodeRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
OpLoad %10 %1 %2 Volatile
OpCompositeInsert %11 %4 %1 %3 0 1 2
)",
                                                   kCAF);
  const SpirvVector alternate = CompileSuccessfully(R"(
OpLoad %10 %1 %2 !1
OpCompositeInsert %11 %4 %1 %3 0 1 2
)",
                                                    kCAF);
  EXPECT_EQ(original, alternate);
}

TEST_F(ImmediateIntTest, WrongLengthButNextOpcodeStillRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
OpLoad %10 %1 %2 Volatile
OpCopyMemorySized %3 %4 %1
)",
                                                   kCAF);
  const SpirvVector alternate = CompileSuccessfully(R"(
!0x0002003D %10 %1 %2 !1
OpCopyMemorySized %3 %4 %1
)",
                                                    kCAF);
  EXPECT_EQ(0x0002003D, alternate[kFirstInstruction]);
  EXPECT_EQ(Subvector(original, kFirstInstruction + 1),
            Subvector(alternate, kFirstInstruction + 1));
}

// Like NextOpcodeRecognized, but next statement is in assignment form.
TEST_F(ImmediateIntTest, NextAssignmentRecognized) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 None
%4 = OpFunctionCall %10 %3 %123
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
%1 = OpLoad %10 %2 !0
%4 = OpFunctionCall %10 %3 %123
)");
  EXPECT_EQ(original, alternate);
}

// Two instructions in a row each have !<integer> opcode.
TEST_F(ImmediateIntTest, ConsecutiveImmediateOpcodes) {
  const SpirvVector original = CompileSuccessfully(R"(
%1 = OpConstantSampler %10 Clamp 78 Linear
%4 = OpFRem %11 %3 %2
%5 = OpIsValidEvent %12 %2
)");
  const SpirvVector alternate = CompileSuccessfully(R"(
!0x0006002D %10 %1 !2 78 !1
!0x0005008C %11 %4 %3 %2
%5 = OpIsValidEvent %12 %2
)");
  EXPECT_EQ(original, alternate);
}

// !<integer> followed by, eg, an enum or '=' or a random bareword.
TEST_F(ImmediateIntTest, ForbiddenOperands) {
  EXPECT_THAT(CompileFailure("OpMemoryModel !0 OpenCL"), HasSubstr("OpenCL"));
  EXPECT_THAT(CompileFailure("!1 %0 = !2"), HasSubstr("="));
  // Immediate integers longer than one 32-bit word.
  EXPECT_THAT(CompileFailure("!5000000000"), HasSubstr("5000000000"));
  EXPECT_THAT(CompileFailure("!999999999999999999"),
              HasSubstr("999999999999999999"));
  EXPECT_THAT(CompileFailure("!0x00020049 !5000000000"),
              HasSubstr("5000000000"));
  EXPECT_THAT(CompileFailure("OpMemoryModel !0 random_bareword"),
              HasSubstr("random_bareword"));
}

TEST_F(ImmediateIntTest, NotInteger) {
  EXPECT_THAT(CompileFailure("!abc"),
              StrEq("Invalid immediate integer '!abc'."));
  EXPECT_THAT(CompileFailure("!12.3"),
              StrEq("Invalid immediate integer '!12.3'."));
  EXPECT_THAT(CompileFailure("!12K"),
              StrEq("Invalid immediate integer '!12K'."));
}

}  // anonymous namespace
