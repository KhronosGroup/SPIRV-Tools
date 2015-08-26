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
  uint32_t extInstOpcode;
};

/// Context for an extended instruction with corresponding binary code for some
/// fields.
///
/// Information about a GLSL extended instruction (including its opname, return
/// type, etc.) and related instructions used to generate the return type and
/// constant as the operands. Also includes the corresponding binary code for
/// some fields. Used in generating extended instruction tests.
struct ExtInstBinContext {
  const char* typeGenInst;
  const char* constGenInst;
  const char* extInstRetType;
  const char* extInstOpName;
  const char* extInstOperandVars;
  uint32_t extInstOpcode;
  uint32_t extInstLength;
  std::vector<uint32_t> extInstOperandIds;
};

using ExtInstGLSLstd450TextToBinTest =
    TextToBinaryTestBase<::testing::TestWithParam<ExtInstBinContext>>;

TEST_P(ExtInstGLSLstd450TextToBinTest, ParamterizedExtInst) {
  const std::string spirv = R"(
            OpCapability Shader
 %glsl450 = OpExtInstImport "GLSL.std.450"
            OpMemoryModel Logical Simple
            OpEntryPoint Vertex %main "main"
    %void = OpTypeVoid
)" + std::string(GetParam().typeGenInst) +
                            "\n" + std::string(GetParam().constGenInst) + R"(
  %fnMain = OpTypeFunction %void
    %main = OpFunction %void None %fnMain
  %lbMain = OpLabel
  %result = OpExtInst )" + GetParam().extInstRetType +
                            " %glsl450 " + GetParam().extInstOpName + " " +
                            GetParam().extInstOperandVars + R"(
            OpReturn
            OpFunctionEnd
)";

  this->text.str = spirv.c_str();
  this->text.length = spirv.size();
  EXPECT_EQ(SPV_SUCCESS, spvTextToBinary(&this->text, this->opcodeTable,
                                         this->operandTable, this->extInstTable,
                                         &this->binary, &this->diagnostic))
      << "Source was: " << std::endl
      << spirv << std::endl
      << "Test case for : " << GetParam().extInstOpName << std::endl;
  std::vector<uint32_t> expected_contains(
      {12 /*OpExtInst*/ | GetParam().extInstLength << 16, 4 /*%flt*/,
       8 /*%result*/, 1 /*%glsl450*/, GetParam().extInstOpcode});
  for (uint32_t operand : GetParam().extInstOperandIds) {
    expected_contains.push_back(operand);
  }
  EXPECT_TRUE(std::search(this->binary->code,
                          this->binary->code + this->binary->wordCount,
                          expected_contains.begin(), expected_contains.end()) !=
              this->binary->code + this->binary->wordCount)
      << "Cannot find\n" << expected_contains << "in\n" << *this->binary;
  if (this->binary) {
    spvBinaryDestroy(this->binary);
  }
  if (this->diagnostic) {
    spvDiagnosticPrint(this->diagnostic);
  }
}

static const char* kF32TypeSym = R"(%flt = OpTypeFloat 32)";
static const char* kF32ConstSym = R"(%c1.5 = OpConstant %flt 1.5)";
static const char* kU32TypeSym = R"(%int = OpTypeInt 32 0)";
static const char* kS32TypeSym = R"(%int = OpTypeInt 32 1)";
static const char* kI32ConstSym = R"(%c1 = OpConstant %int 1)";

INSTANTIATE_TEST_CASE_P(
    ExtInstParameters, ExtInstGLSLstd450TextToBinTest,
    ::testing::ValuesIn(std::vector<ExtInstBinContext>({
        // clang-format off
        {kF32TypeSym, kF32ConstSym, "%flt", "Round", "%c1.5", 1, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "RoundEven", "%c1.5", 2, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Trunc", "%c1.5", 3, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "FAbs", "%c1.5", 4, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "SAbs", "%c1.5", 5, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "FSign", "%c1.5", 6, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "SSign", "%c1.5", 7, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Floor", "%c1.5", 8, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Ceil", "%c1.5", 9, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Fract", "%c1.5", 10, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Radians", "%c1.5", 11, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Degrees", "%c1.5", 12, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Sin", "%c1.5", 13, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Cos", "%c1.5", 14, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Tan", "%c1.5", 15, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Asin", "%c1.5", 16, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Acos", "%c1.5", 17, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Atan", "%c1.5", 18, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Sinh", "%c1.5", 19, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Cosh", "%c1.5", 20, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Tanh", "%c1.5", 21, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Asinh", "%c1.5", 22, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Acosh", "%c1.5", 23, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Atanh", "%c1.5", 24, 6, {5}},
        /* {"Atan2", 25}, {"Pow", 26} */
        {kF32TypeSym, kF32ConstSym, "%flt", "Exp", "%c1.5", 27, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Log", "%c1.5", 28, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Exp2", "%c1.5", 29, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Log2", "%c1.5", 30, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Sqrt", "%c1.5", 31, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Inversesqrt", "%c1.5", 32, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Determinant", "%c1.5", 33, 6, {5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Inverse", "%c1.5", 34, 6, {5}},
        /* Modf */
        /* ModfStruct */
        {kF32TypeSym, kF32ConstSym, "%flt", "FMin", "%c1.5 %c1.5", 37, 7, {5, 5}},
        {kU32TypeSym, kI32ConstSym, "%int", "UMin", "%c1 %c1", 38, 7, {5, 5}},
        {kS32TypeSym, kI32ConstSym, "%int", "SMin", "%c1 %c1", 39, 7, {5, 5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "FMax", "%c1.5 %c1.5", 40, 7, {5, 5}},
        {kU32TypeSym, kI32ConstSym, "%int", "UMax", "%c1 %c1", 41, 7, {5, 5}},
        {kS32TypeSym, kI32ConstSym, "%int", "SMax", "%c1 %c1", 42, 7, {5, 5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "FClamp", "%c1.5 %c1.5 %c1.5", 43, 8, {5, 5, 5}},
        {kU32TypeSym, kI32ConstSym, "%int", "UClamp", "%c1 %c1 %c1", 44, 8, {5, 5, 5}},
        {kS32TypeSym, kI32ConstSym, "%int", "SClamp", "%c1 %c1 %c1", 45, 8, {5, 5, 5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Mix", "%c1.5 %c1.5 %c1.5", 46, 8, {5, 5, 5}},
        {kF32TypeSym, kF32ConstSym, "%flt", "Step", "%c1.5 %c1.5", 47, 7, {5, 5}},
        /* SmoothStep */
        // clang-format on
    })));

using ExtInstGLSLstd450RoundTripTest = ::testing::TestWithParam<ExtInstContext>;

TEST_P(ExtInstGLSLstd450RoundTripTest, ParamterizedExtInst) {
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

static const char* kF32TypeNum = R"(%4 = OpTypeFloat 32)";
static const char* kF32ConstNum = R"(%5 = OpConstant %4 1)";
static const char* kU32TypeNum = R"(%4 = OpTypeInt 32 0)";
static const char* kS32TypeNum = R"(%4 = OpTypeInt 32 1)";
static const char* kI32ConstNum = R"(%5 = OpConstant %4 1)";

INSTANTIATE_TEST_CASE_P(
    ExtInstParameters, ExtInstGLSLstd450RoundTripTest,
    ::testing::ValuesIn(std::vector<ExtInstContext>({
        {kF32TypeNum, kF32ConstNum, "%4", "Round", "%5", 1},
        {kF32TypeNum, kF32ConstNum, "%4", "RoundEven", "%5", 2},
        {kF32TypeNum, kF32ConstNum, "%4", "Trunc", "%5", 3},
        {kF32TypeNum, kF32ConstNum, "%4", "FAbs", "%5", 4},
        {kF32TypeNum, kF32ConstNum, "%4", "SAbs", "%5", 5},
        {kF32TypeNum, kF32ConstNum, "%4", "FSign", "%5", 6},
        {kF32TypeNum, kF32ConstNum, "%4", "SSign", "%5", 7},
        {kF32TypeNum, kF32ConstNum, "%4", "Floor", "%5", 8},
        {kF32TypeNum, kF32ConstNum, "%4", "Ceil", "%5", 9},
        {kF32TypeNum, kF32ConstNum, "%4", "Fract", "%5", 10},
        {kF32TypeNum, kF32ConstNum, "%4", "Radians", "%5", 11},
        {kF32TypeNum, kF32ConstNum, "%4", "Degrees", "%5", 12},
        {kF32TypeNum, kF32ConstNum, "%4", "Sin", "%5", 13},
        {kF32TypeNum, kF32ConstNum, "%4", "Cos", "%5", 14},
        {kF32TypeNum, kF32ConstNum, "%4", "Tan", "%5", 15},
        {kF32TypeNum, kF32ConstNum, "%4", "Asin", "%5", 16},
        {kF32TypeNum, kF32ConstNum, "%4", "Acos", "%5", 17},
        {kF32TypeNum, kF32ConstNum, "%4", "Atan", "%5", 18},
        {kF32TypeNum, kF32ConstNum, "%4", "Sinh", "%5", 19},
        {kF32TypeNum, kF32ConstNum, "%4", "Cosh", "%5", 20},
        {kF32TypeNum, kF32ConstNum, "%4", "Tanh", "%5", 21},
        {kF32TypeNum, kF32ConstNum, "%4", "Asinh", "%5", 22},
        {kF32TypeNum, kF32ConstNum, "%4", "Acosh", "%5", 23},
        {kF32TypeNum, kF32ConstNum, "%4", "Atanh", "%5", 24},
        {kU32TypeNum, kI32ConstNum, "%4", "UMin", "%5 %5", 38},
        {kS32TypeNum, kI32ConstNum, "%4", "SMin", "%5 %5", 39},
        {kU32TypeNum, kI32ConstNum, "%4", "UMax", "%5 %5", 41},
        {kS32TypeNum, kI32ConstNum, "%4", "SMax", "%5 %5", 42},
        /* {"Atan2", 25}, {"Pow", 26} */
        {kF32TypeNum, kF32ConstNum, "%4", "Exp", "%5", 27},
        {kF32TypeNum, kF32ConstNum, "%4", "Log", "%5", 28},
        {kF32TypeNum, kF32ConstNum, "%4", "Exp2", "%5", 29},
        {kF32TypeNum, kF32ConstNum, "%4", "Log2", "%5", 30},
        {kF32TypeNum, kF32ConstNum, "%4", "Sqrt", "%5", 31},
        {kF32TypeNum, kF32ConstNum, "%4", "Inversesqrt", "%5", 32},
        {kF32TypeNum, kF32ConstNum, "%4", "Determinant", "%5", 33},
        {kF32TypeNum, kF32ConstNum, "%4", "Inverse", "%5", 34},
        /* Modf */
        /* ModfStruct */
        {kF32TypeNum, kF32ConstNum, "%4", "FMin", "%5 %5", 37},
        {kU32TypeNum, kI32ConstNum, "%4", "UMin", "%5 %5", 38},
        {kS32TypeNum, kI32ConstNum, "%4", "SMin", "%5 %5", 39},
        {kF32TypeNum, kF32ConstNum, "%4", "FMax", "%5 %5", 40},
        {kU32TypeNum, kI32ConstNum, "%4", "UMax", "%5 %5", 41},
        {kS32TypeNum, kI32ConstNum, "%4", "SMax", "%5 %5", 42},
        {kF32TypeNum, kF32ConstNum, "%4", "FClamp", "%5 %5 %5", 43},
        {kU32TypeNum, kI32ConstNum, "%4", "UClamp", "%5 %5 %5", 44},
        {kS32TypeNum, kI32ConstNum, "%4", "SClamp", "%5 %5 %5", 45},
        {kF32TypeNum, kF32ConstNum, "%4", "Mix", "%5 %5 %5", 46},
        {kF32TypeNum, kF32ConstNum, "%4", "Step", "%5 %5", 47},
        /* SmoothStep */
    })));
