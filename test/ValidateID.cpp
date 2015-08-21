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

#include "UnitSPIRV.h"

// NOTE: The tests in this file are ONLY testing ID usage, there for the input
// SPIR-V does not follow the logical layout rules from the spec in all cases in
// order to makes the tests smaller. Validation of the whole module is handled
// in stages, ID validation is only one of these stages. All validation stages
// are stand alone.

class ValidateID : public ::testing::Test {
 public:
  ValidateID() : opcodeTable(nullptr), operandTable(nullptr), binary() {}

  virtual void SetUp() {
    ASSERT_EQ(SPV_SUCCESS, spvOpcodeTableGet(&opcodeTable));
    ASSERT_EQ(SPV_SUCCESS, spvOperandTableGet(&operandTable));
    ASSERT_EQ(SPV_SUCCESS, spvExtInstTableGet(&extInstTable));
  }

  virtual void TearDown() { spvBinaryDestroy(binary); }

  spv_opcode_table opcodeTable;
  spv_operand_table operandTable;
  spv_ext_inst_table extInstTable;
  spv_binary binary;
};

#define CHECK(str, expected)                                                \
  spv_text_t text = {str, strlen(str)};                                     \
  spv_diagnostic diagnostic;                                                \
  spv_result_t error = spvTextToBinary(&text, opcodeTable, operandTable,    \
                                       extInstTable, &binary, &diagnostic); \
  if (error) {                                                              \
    spvDiagnosticPrint(diagnostic);                                         \
    spvDiagnosticDestroy(diagnostic);                                       \
    ASSERT_EQ(SPV_SUCCESS, error);                                          \
  }                                                                         \
  spv_result_t result =                                                     \
      spvValidate(binary, opcodeTable, operandTable, extInstTable,          \
                  SPV_VALIDATE_ID_BIT, &diagnostic);                        \
  if (SPV_SUCCESS != result) {                                              \
    spvDiagnosticPrint(diagnostic);                                         \
    spvDiagnosticDestroy(diagnostic);                                       \
  }                                                                         \
  ASSERT_EQ(expected, result);

// TODO: OpUndef

TEST_F(ValidateID, OpName) {
  const char *spirv = R"(
     OpName %2 "name"
%1 = OpTypeInt 32 0
%2 = OpTypePointer UniformConstant %1
%3 = OpVariable %2 UniformConstant)";
  CHECK(spirv, SPV_SUCCESS);
}

TEST_F(ValidateID, OpMemberNameGood) {
  const char *spirv = R"(
     OpMemberName %2 0 "foo"
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpMemberNameTypeBad) {
  const char *spirv = R"(
     OpMemberName %1 0 "foo"
%1 = OpTypeInt 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpMemberNameMemberBad) {
  const char *spirv = R"(
     OpMemberName %2 1 "foo"
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpLineGood) {
  const char *spirv = R"(
%1 = OpString "/path/to/source.file"
     OpLine %4 %1 0 0
%2 = OpTypeInt 32 0
%3 = OpTypePointer Generic %2
%4 = OpVariable %3 Generic)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpLineFileBad) {
  const char *spirv = R"(
     OpLine %4 %2 0 0
%2 = OpTypeInt 32 0
%3 = OpTypePointer Generic %2
%4 = OpVariable %3 Generic)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpDecorateGood) {
  const char *spirv = R"(
     OpDecorate %2 GLSLShared
%1 = OpTypeInt 64 0
%2 = OpTypeStruct %1 %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpDecorateBad) {
  const char *spirv = R"(
OpDecorate %1 GLSLShared)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpMemberDecorateGood) {
  const char *spirv = R"(
     OpMemberDecorate %2 0 Uniform
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpMemberDecorateBad) {
  const char *spirv = R"(
     OpMemberDecorate %1 0 Uniform
%1 = OpTypeInt 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpMemberDecorateMemberBad) {
  const char *spirv = R"(
     OpMemberDecorate %2 3 Uniform
%1 = OpTypeInt 32 0
%2 = OpTypeStruct %1 %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpGroupDecorateGood) {
  const char *spirv = R"(
%1 = OpDecorationGroup
     OpDecorate %1 Uniform
     OpDecorate %1 GLSLShared
     OpGroupDecorate %1 %3 %4
%2 = OpTypeInt 32 0
%3 = OpConstant %2 42
%4 = OpConstant %2 23)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpGroupDecorateDecorationGroupBad) {
  const char *spirv = R"(
     OpGroupDecorate %2 %3 %4
%2 = OpTypeInt 32 0
%3 = OpConstant %2 42
%4 = OpConstant %2 23)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpGroupDecorateTargetBad) {
  const char *spirv = R"(
%1 = OpDecorationGroup
     OpDecorate %1 Uniform
     OpDecorate %1 GLSLShared
     OpGroupDecorate %1 %3
%2 = OpTypeInt 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpGroupMemberDecorate
// TODO: OpExtInst

TEST_F(ValidateID, OpEntryPointGood) {
  const char *spirv = R"(
     OpEntryPoint GLCompute %3
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpEntryPointFunctionBad) {
  const char *spirv = R"(
     OpEntryPoint GLCompute %1
%1 = OpTypeVoid)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpEntryPointParameterCountBad) {
  const char *spirv = R"(
     OpEntryPoint GLCompute %3
%1 = OpTypeVoid
%2 = OpTypeFunction %1 %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpEntryPointReturnTypeBad) {
  const char *spirv = R"(
     OpEntryPoint GLCompute %3
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpExecutionModeGood) {
  const char *spirv = R"(
     OpEntryPoint GLCompute %3
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpExecutionModeEntryPointBad) {
  const char *spirv = R"(
     OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeVectorGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeVectorComponentTypeBad) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypePointer UniformConstant %1
%3 = OpTypeVector %2 4)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeMatrixGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeVector %1 2
%3 = OpTypeMatrix %2 3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeMatrixColumnTypeBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeMatrix %1 3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeSamplerGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeSampler %1 2D 0 0 0 0)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeSamplerSampledTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeSampler %1 2D 0 0 0 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeArrayGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1
%3 = OpTypeArray %1 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeArrayElementTypeBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1
%3 = OpTypeArray %2 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpTypeArrayLengthBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypeArray %1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeRuntimeArrayGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeRuntimeArray %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeRuntimeArrayBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypeRuntimeArray %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
// TODO: Object of this type can only be created with OpVariable using the
// Unifrom Storage Class

TEST_F(ValidateID, OpTypeStructGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFloat 64
%3 = OpTypePointer Generic %1
%4 = OpTypeStruct %1 %2 %3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeStructMemberTypeBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeFloat 64
%3 = OpConstant %2 0.0
%4 = OpTypeStruct %1 %2 %3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypePointerGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypePointer Generic %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypePointerBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypePointer Generic %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeFunctionGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeFunction %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeFunctionReturnTypeBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 0
%3 = OpTypeFunction %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpTypeFunctionParameterBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpConstant %2 0
%4 = OpTypeFunction %1 %2 %3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypePipeGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 16
%3 = OpTypePipe %2 ReadOnly)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypePipeBad) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpConstant %1 0
%3 = OpTypePipe %2 ReadOnly)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantTrueGood) {
  const char *spirv = R"(
%1 = OpTypeBool
%2 = OpConstantTrue %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantTrueBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpConstantTrue %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantFalseGood) {
  const char *spirv = R"(
OpTypeBool %1
%2 = OpConstantTrue %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantFalseBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpConstantFalse %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpConstant %1 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantCompositeVectorGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpConstant %1 3.14
%4 = OpConstantComposite %2 %3 %3 %3 %3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeVectorResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%3 = OpConstant %1 3.14
%4 = OpConstantComposite %1 %3 %3 %3 %3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeVectorConstituentBad) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeVector %1 4
%4 = OpTypeInt 32 0
%3 = OpConstant %1 3.14
%5 = OpConstant %4 42
%6 = OpConstantComposite %2 %3 %5 %3 %3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeMatrixGood) {
  const char *spirv = R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 %5 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpConstantComposite %2 %5 %5 %5 %4
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeMatrixConstituentBad) {
  const char *spirv = R"(
 %1 = OpTypeFloat 32
 %2 = OpTypeVector %1 4
%11 = OpTypeVector %1 3
 %3 = OpTypeMatrix %2 4
 %4 = OpConstant %1 1.0
 %5 = OpConstant %1 0.0
 %6 = OpConstantComposite %2 %4 %5 %5 %5
 %7 = OpConstantComposite %2 %5 %4 %5 %5
 %8 = OpConstantComposite %2 %5 %5 %4 %5
 %9 = OpConstantComposite %11 %5 %5 %5
%10 = OpConstantComposite %3 %6 %7 %8 %9)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeMatrixColumnTypeBad) {
  const char *spirv = R"(
 %1 = OpTypeInt 32 0
 %2 = OpTypeFloat 32
 %3 = OpTypeVector %1 2
 %4 = OpTypeVector %3 2
 %5 = OpTypeMatrix %2 2
 %6 = OpConstant %1 42
 %7 = OpConstant %2 3.14
 %8 = OpConstantComposite %3 %6 %6
 %9 = OpConstantComposite %4 %7 %7
%10 = OpConstantComposite %5 %8 %9)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeArrayGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpConstantComposite %3 %2 %2 %2 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeArrayConstConstituentBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%4 = OpConstantComposite %3 %2 %2 %2 %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeArrayConstituentBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpConstant %1 4
%3 = OpTypeArray %1 %2
%5 = OpTypeFloat 32
%6 = OpConstant %5 3.14
%4 = OpConstantComposite %3 %2 %2 %2 %6)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeStructGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 1
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpConstant %2 4300000000
%6 = OpConstantComposite %3 %4 %4 %5)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeStructMemberBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 0
%2 = OpTypeInt 64 1
%3 = OpTypeStruct %1 %1 %2
%4 = OpConstant %1 42
%5 = OpConstant %2 4300000000
%6 = OpConstantComposite %3 %4 %5 %4)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantSamplerGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpTypeSampler %1 2D 1 0 1 0
%3 = OpConstantSampler %2 ClampToEdge 0 Nearest)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantSamplerResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpConstantSampler %1 Clamp 0 Nearest)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantNullGood) {
  const char *spirv = R"(
 %1 = OpTypeBool
 %2 = OpConstantNull %1
 %3 = OpTypeInt 32 0
 %4 = OpConstantNull %3
 %5 = OpTypeFloat 32
 %6 = OpConstantNull %5
 %7 = OpTypePointer UniformConstant %3
 %8 = OpConstantNull %7
 %9 = OpTypeEvent
%10 = OpConstantNull %9
%11 = OpTypeDeviceEvent
%12 = OpConstantNull %11
%13 = OpTypeReserveId
%14 = OpConstantNull %13
%15 = OpTypeQueue
%16 = OpConstantNull %15
%17 = OpTypeVector %3 2
%18 = OpConstantNull %17
%19 = OpTypeMatrix %17 2
%20 = OpConstantNull %19
%25 = OpConstant %3 8
%21 = OpTypeArray %3 %25
%22 = OpConstantNull %21
%23 = OpTypeStruct %3 %5 %1
%24 = OpConstantNull %23
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantNullBasicBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpConstantNull %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantNullArrayBad) {
  const char *spirv = R"(
%1 = OpTypeInt 8 0
%2 = OpTypeInt 32 0
%3 = OpTypeSampler %1 2D 0 0 0 0
%4 = OpConstant %2 4
%5 = OpTypeArray %3 %4
%6 = OpConstantNull %5)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantNullStructBad) {
  const char *spirv = R"(
%1 = OpTypeInt 8 0
%2 = OpTypeSampler %1 2D 0 0 0 0
%3 = OpTypeStruct %2 %2
%4 = OpConstantNull %3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantTrueGood) {
  const char *spirv = R"(
%1 = OpTypeBool
%2 = OpSpecConstantTrue %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantTrueBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpSpecConstantTrue %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantFalseGood) {
  const char *spirv = R"(
%1 = OpTypeBool
%2 = OpSpecConstantFalse %1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantFalseBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpSpecConstantFalse %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantGood) {
  const char *spirv = R"(
%1 = OpTypeFloat 32
%2 = OpSpecConstant %1 42)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpSpecConstant %1 3.14)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpSpecConstantComposite
// TODO: OpSpecConstantOp

TEST_F(ValidateID, OpVariableGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 1
%2 = OpTypePointer Generic %1
%3 = OpVariable %2 Generic)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpVariableInitializerGood) {
  const char *spirv = R"(
%1 = OpTypeInt 32 1
%2 = OpTypePointer Generic %1
%3 = OpConstant %1 42
%4 = OpVariable %2 Generic %3)";
  CHECK(spirv, SPV_SUCCESS);
}
// TODO: Positive test OpVariable with OpConstantNull of OpTypePointer
TEST_F(ValidateID, OpVariableResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 1
%2 = OpVariable %1 Generic)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpVariableInitializerBad) {
  const char *spirv = R"(
%1 = OpTypeInt 32 1
%2 = OpTypePointer Generic %1
%3 = OpVariable %2 Generic %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpLoadGood) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 1
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypeFunction %1
 %5 = OpVariable %3 UniformConstant
 %6 = OpFunction %1 None %4
 %7 = OpLabel
 %8 = OpLoad %3 %5
 %9 = OpReturn
%10 = OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpLoadResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpVariable %3 UniformConstant
%6 = OpFunction %1 None %4
%7 = OpLabel
%8 = OpLoad %2 %5
     OpReturn
     OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpLoadPointerBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%9 = OpTypeFloat 32
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%6 = OpFunction %1 None %4
%7 = OpLabel
%8 = OpLoad %9 %3
     OpReturn
     OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpStoreGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 UniformConstant
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %5
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpStorePointerBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 UniformConstant
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %3 %5
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpStoreObjectGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42
%6 = OpVariable %3 UniformConstant
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %7
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpStoreTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%9 = OpTypeFloat 32
%3 = OpTypePointer UniformConstant %2
%4 = OpTypeFunction %1
%5 = OpConstant %9 3.14
%6 = OpVariable %3 UniformConstant
%7 = OpFunction %1 None %4
%8 = OpLabel
     OpStore %6 %5
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpCopyMemoryGood) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpConstant %2 42
 %5 = OpVariable %3 UniformConstant %4
 %6 = OpTypePointer Function %2
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %6 Function
      OpCopyMemory %10 %5 None
      OpReturn
      OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpCopyMemoryBad) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpConstant %2 42
 %5 = OpVariable %3 UniformConstant %4
%11 = OpTypeFloat 32
 %6 = OpTypePointer Function %11
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %6 Function
      OpCopyMemory %10 %5 None
      OpReturn
      OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpCopyMemorySized
TEST_F(ValidateID, OpCopyMemorySizedGood) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %5 None
      OpReturn
      OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpCopyMemorySizedTargetBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypePointer Function %2
%5 = OpConstant %2 4
%6 = OpVariable %3 UniformConstant %5
%7 = OpTypeFunction %1
%8 = OpFunction %1 None %7
%9 = OpLabel
     OpCopyMemorySized %9 %6 %5 None
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSourceBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypePointer UniformConstant %2
%4 = OpTypePointer Function %2
%5 = OpConstant %2 4
%6 = OpTypeFunction %1
%7 = OpFunction %1 None %6
%8 = OpLabel
%9 = OpVariable %4 Function
     OpCopyMemorySized %9 %6 %5 None
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSizeBad) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %6 None
      OpReturn
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSizeTypeBad) {
  const char *spirv = R"(
 %1 = OpTypeVoid
 %2 = OpTypeInt 32 0
 %3 = OpTypePointer UniformConstant %2
 %4 = OpTypePointer Function %2
 %5 = OpConstant %2 4
 %6 = OpVariable %3 UniformConstant %5
 %7 = OpTypeFunction %1
%11 = OpTypeFloat 32
%12 = OpConstant %11 1.0
 %8 = OpFunction %1 None %7
 %9 = OpLabel
%10 = OpVariable %4 Function
      OpCopyMemorySized %10 %6 %12 None
      OpReturn
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpAccessChain
// TODO: OpInBoundsAccessChain
// TODO: OpArrayLength
// TODO: OpImagePointer
// TODO: OpGenericPtrMemSemantics

TEST_F(ValidateID, OpFunctionGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%3 = OpTypeFunction %1 %2 %2
%4 = OpFunction %1 None %3
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%5 = OpConstant %2 42
%3 = OpTypeFunction %1 %2 %2
%4 = OpFunction %2 None %3
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionFunctionTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 1
%4 = OpFunction %1 None %2
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpFunctionParameterGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2
%4 = OpFunction %1 None %3
%5 = OpFunctionParameter %2
%6 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionParameterResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2
%4 = OpFunction %1 None %3
%5 = OpFunctionParameter %1
%6 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionParameterOrderBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %1 %2
%7 = OpTypePointer Function %2
%4 = OpFunction %1 None %3
%8 = OpVariable %7 Function
%5 = OpFunctionParameter %2
%6 = OpLabel
     OpReturn
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpFunctionCallGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpLoad %2 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %2 %6 %5
      OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionCallResultTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpLoad %2 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %1 %6 %5
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionCallFunctionBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %2 %5 %5
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionCallArgumentTypeBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42

%13 = OpTypeFloat 32
%14 = OpConstant %13 3.14

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpLoad %2 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %2 %6 %14
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
#if 0
TEST_F(ValidateID, OpFunctionCallArgumentCountBar) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpTypeFunction %1
%5 = OpConstant %2 42 ;21

%6 = OpFunction %2 None %3
%7 = OpFunctionParameter %2
%8 = OpLabel
%9 = OpLoad %2 %7
     OpReturnValue %9
     OpFunctionEnd

%10 = OpFunction %1 None %4
%11 = OpLabel
      OpReturn
%12 = OpFunctionCall %2 %6 %5
      OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
#endif

// TODO: OpSampler
// TODO: OpTextureSample
// TODO: OpTextureSampleDref
// TODO: OpTextureSampleLod
// TODO: OpTextureSampleProj
// TODO: OpTextureSampleGrad
// TODO: OpTextureSampleOffset
// TODO: OpTextureSampleProjLod
// TODO: OpTextureSampleProjGrad
// TODO: OpTextureSampleLodOffset
// TODO: OpTextureSampleProjOffset
// TODO: OpTextureSampleGradOffset
// TODO: OpTextureSampleProjLodOffset
// TODO: OpTextureSampleProjGradOffset
// TODO: OpTextureFetchTexelLod
// TODO: OpTextureFetchTexelOffset
// TODO: OpTextureFetchSample
// TODO: OpTextureFetchTexel
// TODO: OpTextureGather
// TODO: OpTextureGatherOffset
// TODO: OpTextureGatherOffsets
// TODO: OpTextureQuerySizeLod
// TODO: OpTextureQuerySize
// TODO: OpTextureQueryLevels
// TODO: OpTextureQuerySamples
// TODO: OpConvertUToF
// TODO: OpConvertFToS
// TODO: OpConvertSToF
// TODO: OpConvertUToF
// TODO: OpUConvert
// TODO: OpSConvert
// TODO: OpFConvert
// TODO: OpConvertPtrToU
// TODO: OpConvertUToPtr
// TODO: OpPtrCastToGeneric
// TODO: OpGenericCastToPtr
// TODO: OpBitcast
// TODO: OpGenericCastToPtrExplicit
// TODO: OpSatConvertSToU
// TODO: OpSatConvertUToS
// TODO: OpVectorExtractDynamic
// TODO: OpVectorInsertDynamic
// TODO: OpVectorShuffle
// TODO: OpCompositeConstruct
// TODO: OpCompositeExtract
// TODO: OpCompositeInsert
// TODO: OpCopyObject
// TODO: OpTranspose
// TODO: OpSNegate
// TODO: OpFNegate
// TODO: OpNot
// TODO: OpIAdd
// TODO: OpFAdd
// TODO: OpISub
// TODO: OpFSub
// TODO: OpIMul
// TODO: OpFMul
// TODO: OpUDiv
// TODO: OpSDiv
// TODO: OpFDiv
// TODO: OpUMod
// TODO: OpSRem
// TODO: OpSMod
// TODO: OpFRem
// TODO: OpFMod
// TODO: OpVectorTimesScalar
// TODO: OpMatrixTimesScalar
// TODO: OpVectorTimesMatrix
// TODO: OpMatrixTimesVector
// TODO: OpMatrixTimesMatrix
// TODO: OpOuterProduct
// TODO: OpDot
// TODO: OpShiftRightLogical
// TODO: OpShiftRightArithmetic
// TODO: OpShiftLeftLogical
// TODO: OpBitwiseOr
// TODO: OpBitwiseXor
// TODO: OpBitwiseAnd
// TODO: OpAny
// TODO: OpAll
// TODO: OpIsNan
// TODO: OpIsInf
// TODO: OpIsFinite
// TODO: OpIsNormal
// TODO: OpSignBitSet
// TODO: OpLessOrGreater
// TODO: OpOrdered
// TODO: OpUnordered
// TODO: OpLogicalOr
// TODO: OpLogicalXor
// TODO: OpLogicalAnd
// TODO: OpSelect
// TODO: OpIEqual
// TODO: OpFOrdEqual
// TODO: OpFUnordEqual
// TODO: OpINotEqual
// TODO: OpFOrdNotEqual
// TODO: OpFUnordNotEqual
// TODO: OpULessThan
// TODO: OpSLessThan
// TODO: OpFOrdLessThan
// TODO: OpFUnordLessThan
// TODO: OpUGreaterThan
// TODO: OpSGreaterThan
// TODO: OpFOrdGreaterThan
// TODO: OpFUnordGreaterThan
// TODO: OpULessThanEqual
// TODO: OpSLessThanEqual
// TODO: OpFOrdLessThanEqual
// TODO: OpFUnordLessThanEqual
// TODO: OpUGreaterThanEqual
// TODO: OpSGreaterThanEqual
// TODO: OpFOrdGreaterThanEqual
// TODO: OpFUnordGreaterThanEqual
// TODO: OpDPdx
// TODO: OpDPdy
// TODO: OpFWidth
// TODO: OpDPdxFine
// TODO: OpDPdyFine
// TODO: OpFwidthFine
// TODO: OpDPdxCoarse
// TODO: OpDPdyCoarse
// TODO: OpFwidthCoarse
// TODO: OpPhi
// TODO: OpLoopMerge
// TODO: OpSelectionMerge
// TODO: OpBranch
// TODO: OpBranchConditional
// TODO: OpSwitch

TEST_F(ValidateID, OpReturnValueConstantGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpConstant %2 42
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %4
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpReturnValueVariableGood) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0 ;10
%3 = OpTypeFunction %2 %2 ;14
%8 = OpTypePointer Function %2 ;18
%4 = OpConstant %2 42 ;22
%5 = OpFunction %2 None %3 ;27
%6 = OpLabel ;29
%7 = OpVariable %8 Function %4 ;34
     OpReturnValue %7 ;36
     OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpReturnValueBad) {
  const char *spirv = R"(
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFunction %2 %2
%4 = OpConstant %2 42
%5 = OpFunction %2 None %3
%6 = OpLabel
     OpReturnValue %1
     OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpLifetimeStart
// TODO: OpLifetimeStop
// TODO: OpAtomicInit
// TODO: OpAtomicLoad
// TODO: OpAtomicStore
// TODO: OpAtomicExchange
// TODO: OpAtomicCompareExchange
// TODO: OpAtomicCompareExchangeWeak
// TODO: OpAtomicIIncrement
// TODO: OpAtomicIDecrement
// TODO: OpAtomicIAdd
// TODO: OpAtomicISub
// TODO: OpAtomicUMin
// TODO: OpAtomicUMax
// TODO: OpAtomicAnd
// TODO: OpAtomicOr
// TODO: OpAtomicXor
// TODO: OpAtomicIMin
// TODO: OpAtomicIMax
// TODO: OpEmitStreamVertex
// TODO: OpEndStreamPrimitive
// TODO: OpAsyncGroupCopy
// TODO: OpWaitGroupEvents
// TODO: OpGroupAll
// TODO: OpGroupAny
// TODO: OpGroupBroadcast
// TODO: OpGroupIAdd
// TODO: OpGroupFAdd
// TODO: OpGroupFMin
// TODO: OpGroupUMin
// TODO: OpGroupSMin
// TODO: OpGroupFMax
// TODO: OpGroupUMax
// TODO: OpGroupSMax
// TODO: OpEnqueueMarker
// TODO: OpEnqueueKernel
// TODO: OpGetKernelNDrangeSubGroupCount
// TODO: OpGetKernelNDrangeMaxSubGroupSize
// TODO: OpGetKernelWorkGroupSize
// TODO: OpGetKernelPreferredWorkGroupSizeMultiple
// TODO: OpRetainEvent
// TODO: OpReleaseEvent
// TODO: OpCreateUserEvent
// TODO: OpIsValidEvent
// TODO: OpSetUserEventStatus
// TODO: OpCaptureEventProfilingInfo
// TODO: OpGetDefaultQueue
// TODO: OpBuildNDRange
// TODO: OpReadPipe
// TODO: OpWritePipe
// TODO: OpReservedReadPipe
// TODO: OpReservedWritePipe
// TODO: OpReserveReadPipePackets
// TODO: OpReserveWritePipePackets
// TODO: OpCommitReadPipe
// TODO: OpCommitWritePipe
// TODO: OpIsValidReserveId
// TODO: OpGetNumPipePackets
// TODO: OpGetMaxPipePackets
// TODO: OpGroupReserveReadPipePackets
// TODO: OpGroupReserveWritePipePackets
// TODO: OpGroupCommitReadPipe
// TODO: OpGroupCommitWritePipe
