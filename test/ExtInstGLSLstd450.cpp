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

#include "TestFixture.h"
#include "UnitSPIRV.h"
#include <algorithm>
#include <vector>

/// Context for an extended instruction.
///
/// Information about a GLSL extended instruction (including its opname, return
/// type, etc.) and related instructions used to generate the return type and
/// constant as the operands. Used in generating extended instruction tests.
struct ExtInstContext {
  const char* typeGenInst;
  const char* constGenInst;
  const char* extInstRetType;
  const char* extInstOpName;
  const char* extInstOperandVars;
  /// The following fields are used to check the SPIR-V binary representation
  /// of this instruction.
  uint32_t extInstOpcode;  ///< Opcode value for this extended instruction.
  uint32_t extInstLength;  ///< Wordcount of this extended instruction.
  std::vector<uint32_t> extInstOperandIds;  ///< Ids for operands.
};

using ExtInstGLSLstd450RoundTripTest = ::testing::TestWithParam<ExtInstContext>;

TEST_P(ExtInstGLSLstd450RoundTripTest, ParameterizedExtInst) {
  spv_opcode_table opcodeTable;
  ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
  spv_operand_table operandTable;
  ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
  spv_ext_inst_table extInstTable;
  ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));

  const std::string spirv = R"(
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Simple
OpEntryPoint Vertex %2 "main"
%3 = OpTypeVoid
)" + std::string(GetParam().typeGenInst) +
                            "\n" + std::string(GetParam().constGenInst) + R"(
%6 = OpTypeFunction %3
%2 = OpFunction %3 None %6
%8 = OpLabel
%9 = OpExtInst )" + GetParam().extInstRetType +
                            " %1 " + GetParam().extInstOpName + " " +
                            GetParam().extInstOperandVars + R"(
OpReturn
OpFunctionEnd
)";
  const std::string spirv_header =
      R"(; SPIR-V
; Version: 99
; Generator: Khronos
; Bound: 10
; Schema: 0)";
  spv_text_t text = {spirv.c_str(), spirv.size()};
  spv_binary binary;
  spv_diagnostic diagnostic;
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,
                                       extInstTable, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error)
        << "Source was: " << std::endl
        << spirv << std::endl
        << "Test case for : " << GetParam().extInstOpName << std::endl;
  }

  // Check we do have the extended instruction's corresponding binary code in
  // the generated SPIR-V binary.
  std::vector<uint32_t> expected_contains(
      {12 /*OpExtInst*/ | GetParam().extInstLength << 16, 4 /*return type*/,
       9 /*result id*/, 1 /*glsl450 import*/, GetParam().extInstOpcode});
  for (uint32_t operand : GetParam().extInstOperandIds) {
    expected_contains.push_back(operand);
  }
  EXPECT_NE(binary->code + binary->wordCount,
            std::search(binary->code, binary->code + binary->wordCount,
                        expected_contains.begin(), expected_contains.end()))
      << "Cannot find\n" << expected_contains << "in\n" << *binary;

  // Check round trip gives the same text.
  spv_text output_text;
  error =
      spvBinaryToText(binary, SPV_BINARY_TO_TEXT_OPTION_NONE, opcodeTable,
                      operandTable, extInstTable, &output_text, &diagnostic);

  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    ASSERT_EQ(SPV_SUCCESS, error);
  }
  EXPECT_EQ(spirv_header + spirv, output_text->str);
  spvTextDestroy(output_text);
}

static const char* kF32Type = R"(%4 = OpTypeFloat 32)";
static const char* kF32Const = R"(%5 = OpConstant %4 1)";
static const char* kU32Type = R"(%4 = OpTypeInt 32 0)";
static const char* kS32Type = R"(%4 = OpTypeInt 32 1)";
static const char* kI32Const = R"(%5 = OpConstant %4 1)";

INSTANTIATE_TEST_CASE_P(
    ExtInstParameters, ExtInstGLSLstd450RoundTripTest,
    ::testing::ValuesIn(std::vector<ExtInstContext>({
        {kF32Type, kF32Const, "%4", "Round", "%5", 1, 6, {5}},
        {kF32Type, kF32Const, "%4", "RoundEven", "%5", 2, 6, {5}},
        {kF32Type, kF32Const, "%4", "Trunc", "%5", 3, 6, {5}},
        {kF32Type, kF32Const, "%4", "FAbs", "%5", 4, 6, {5}},
        {kF32Type, kF32Const, "%4", "SAbs", "%5", 5, 6, {5}},
        {kF32Type, kF32Const, "%4", "FSign", "%5", 6, 6, {5}},
        {kF32Type, kF32Const, "%4", "SSign", "%5", 7, 6, {5}},
        {kF32Type, kF32Const, "%4", "Floor", "%5", 8, 6, {5}},
        {kF32Type, kF32Const, "%4", "Ceil", "%5", 9, 6, {5}},
        {kF32Type, kF32Const, "%4", "Fract", "%5", 10, 6, {5}},
        {kF32Type, kF32Const, "%4", "Radians", "%5", 11, 6, {5}},
        {kF32Type, kF32Const, "%4", "Degrees", "%5", 12, 6, {5}},
        {kF32Type, kF32Const, "%4", "Sin", "%5", 13, 6, {5}},
        {kF32Type, kF32Const, "%4", "Cos", "%5", 14, 6, {5}},
        {kF32Type, kF32Const, "%4", "Tan", "%5", 15, 6, {5}},
        {kF32Type, kF32Const, "%4", "Asin", "%5", 16, 6, {5}},
        {kF32Type, kF32Const, "%4", "Acos", "%5", 17, 6, {5}},
        {kF32Type, kF32Const, "%4", "Atan", "%5", 18, 6, {5}},
        {kF32Type, kF32Const, "%4", "Sinh", "%5", 19, 6, {5}},
        {kF32Type, kF32Const, "%4", "Cosh", "%5", 20, 6, {5}},
        {kF32Type, kF32Const, "%4", "Tanh", "%5", 21, 6, {5}},
        {kF32Type, kF32Const, "%4", "Asinh", "%5", 22, 6, {5}},
        {kF32Type, kF32Const, "%4", "Acosh", "%5", 23, 6, {5}},
        {kF32Type, kF32Const, "%4", "Atanh", "%5", 24, 6, {5}},
        {kF32Type, kF32Const, "%4", "Atan2", "%5 %5", 25, 7, {5, 5}},
        {kF32Type, kF32Const, "%4", "Pow", "%5 %5", 26, 7, {5, 5}},
        {kF32Type, kF32Const, "%4", "Exp", "%5", 27, 6, {5}},
        {kF32Type, kF32Const, "%4", "Log", "%5", 28, 6, {5}},
        {kF32Type, kF32Const, "%4", "Exp2", "%5", 29, 6, {5}},
        {kF32Type, kF32Const, "%4", "Log2", "%5", 30, 6, {5}},
        {kF32Type, kF32Const, "%4", "Sqrt", "%5", 31, 6, {5}},
        {kF32Type, kF32Const, "%4", "Inversesqrt", "%5", 32, 6, {5}},
        {kF32Type, kF32Const, "%4", "Determinant", "%5", 33, 6, {5}},
        {kF32Type, kF32Const, "%4", "Inverse", "%5", 34, 6, {5}},
        /* Modf */
        /* ModfStruct */
        {kF32Type, kF32Const, "%4", "FMin", "%5 %5", 37, 7, {5, 5}},
        {kU32Type, kI32Const, "%4", "UMin", "%5 %5", 38, 7, {5, 5}},
        {kS32Type, kI32Const, "%4", "SMin", "%5 %5", 39, 7, {5, 5}},
        {kF32Type, kF32Const, "%4", "FMax", "%5 %5", 40, 7, {5, 5}},
        {kU32Type, kI32Const, "%4", "UMax", "%5 %5", 41, 7, {5, 5}},
        {kS32Type, kI32Const, "%4", "SMax", "%5 %5", 42, 7, {5, 5}},
        {kF32Type, kF32Const, "%4", "FClamp", "%5 %5 %5", 43, 8, {5, 5, 5}},
        {kU32Type, kI32Const, "%4", "UClamp", "%5 %5 %5", 44, 8, {5, 5, 5}},
        {kS32Type, kI32Const, "%4", "SClamp", "%5 %5 %5", 45, 8, {5, 5, 5}},
        {kF32Type, kF32Const, "%4", "Mix", "%5 %5 %5", 46, 8, {5, 5, 5}},
        {kF32Type, kF32Const, "%4", "Step", "%5 %5", 47, 7, {5, 5}},
        {kF32Type, kF32Const, "%4", "Smoothstep", "%5 %5 %5", 48, 8, {5, 5, 5}},
    })));
