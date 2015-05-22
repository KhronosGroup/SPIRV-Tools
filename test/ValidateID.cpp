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
OpName $2 "name"
OpTypeInt %1 32 0
OpTypePointer %2 UniformConstant $1
OpVariable $2 %3 UniformConstant)";
  CHECK(spirv, SPV_SUCCESS);
}

TEST_F(ValidateID, OpMemberNameGood) {
  const char *spirv = R"(
OpMemberName $2 0 "foo"
OpTypeInt %1 32 0
OpTypeStruct %2 $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpMemberNameTypeBad) {
  const char *spirv = R"(
OpMemberName $1 0 "foo"
OpTypeInt %1 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpMemberNameMemberBad) {
  const char *spirv = R"(
OpMemberName $2 1 "foo"
OpTypeInt %1 32 0
OpTypeStruct %2 $1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpLineGood) {
  const char *spirv = R"(
OpString %1 "/path/to/source.file"
OpLine $4 $1 0 0
OpTypeInt %2 32 0
OpTypePointer %3 Generic $2
OpVariable $3 %4 Generic)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpLineFileBad) {
  const char *spirv = R"(
OpLine $4 $2 0 0
OpTypeInt %2 32 0
OpTypePointer %3 Generic $2
OpVariable $3 %4 Generic)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpDecorateGood) {
  const char *spirv = R"(
OpDecorate $2 GLSLShared
OpTypeInt %1 64 0
OpTypeStruct %2 $1 $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpDecorateBad) {
  const char *spirv = R"(
OpDecorate $1 GLSLShared)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpMemberDecorateGood) {
  const char *spirv = R"(
OpMemberDecorate $2 0 Uniform
OpTypeInt %1 32 0
OpTypeStruct %2 $1 $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpMemberDecorateBad) {
  const char *spirv = R"(
OpMemberDecorate $1 0 Uniform
OpTypeInt %1 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpMemberDecorateMemberBad) {
  const char *spirv = R"(
OpMemberDecorate $2 3 Uniform
OpTypeInt %1 32 0
OpTypeStruct %2 $1 $1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpGroupDecorateGood) {
  const char *spirv = R"(
OpDecorationGroup %1
OpDecorate $1 Uniform
OpDecorate $1 GLSLStd430
OpGroupDecorate $1 $3 $4
OpTypeInt %2 32 0
OpConstant $2 %3 42
OpConstant $2 %4 23)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpGroupDecorateDecorationGroupBad) {
  const char *spirv = R"(
OpGroupDecorate $2 $3 $4
OpTypeInt %2 32 0
OpConstant $2 %3 42
OpConstant $2 %4 23)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpGroupDecorateTargetBad) {
  const char *spirv = R"(
OpDecorationGroup %1
OpDecorate $1 Uniform
OpDecorate $1 GLSLStd430
OpGroupDecorate $1 $3
OpTypeInt %2 32 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpGroupMemberDecorate
// TODO: OpExtInst

TEST_F(ValidateID, OpEntryPointGood) {
  const char *spirv = R"(
OpEntryPoint GLCompute $3
OpTypeVoid %1
OpTypeFunction %2 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpEntryPointFunctionBad) {
  const char *spirv = R"(
OpEntryPoint GLCompute $1
OpTypeVoid %1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpEntryPointParameterCountBad) {
  const char *spirv = R"(
OpEntryPoint GLCompute $3
OpTypeVoid %1
OpTypeFunction %2 $1 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpEntryPointReturnTypeBad) {
  const char *spirv = R"(
OpEntryPoint GLCompute $3
OpTypeInt %1 32 0
OpTypeFunction %2 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpExecutionModeGood) {
  const char *spirv = R"(
OpEntryPoint GLCompute $3
OpExecutionMode $3 LocalSize 1 1 1
OpTypeVoid %1
OpTypeFunction %2 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpExecutionModeEntryPointBad) {
  const char *spirv = R"(
OpExecutionMode $3 LocalSize 1 1 1
OpTypeVoid %1
OpTypeFunction %2 $1
OpFunction $1 %3 None $2
OpLabel %4
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeVectorGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeVectorComponentTypeBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypePointer %2 UniformConstant $1
OpTypeVector %3 $2 4)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeMatrixGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeVector %2 $1 2
OpTypeMatrix %3 $2 3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeMatrixColumnTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeMatrix %2 $1 3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeSamplerGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeSampler %2 $1 2D 0 0 0 0)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeSamplerSampledTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeSampler %2 $1 2D 0 0 0 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeArrayGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 1
OpTypeArray %3 $1 $2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeArrayElementTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 1
OpTypeArray %3 $2 $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpTypeArrayLengthBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 0
OpTypeArray %3 $1 $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeRuntimeArrayGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeRuntimeArray %2 $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeRuntimeArrayBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 0
OpTypeRuntimeArray %3 $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
// TODO: Object of this type can only be created with OpVariable using the
// Unifrom Storage Class

TEST_F(ValidateID, OpTypeStructGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeFloat %2 64
OpTypePointer %3 Generic $1
OpTypeStruct %4 $1 $2 $3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeStructMemberTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeFloat %2 64
OpConstant $2 %3 0.0
OpTypeStruct %4 $1 $2 $3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypePointerGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypePointer %2 Generic $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypePointerBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 0
OpTypePointer %3 Generic $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypeFunctionGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeFunction %2 $1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypeFunctionReturnTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 0
OpTypeFunction %3 $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpTypeFunctionParameterBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpConstant $2 %3 0
OpTypeFunction %4 $1 $2 $3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpTypePipeGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 16
OpTypePipe %3 $2 ReadOnly)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpTypePipeBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpConstant $1 %2 0
OpTypePipe %3 $2 ReadOnly)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantTrueGood) {
  const char *spirv = R"(
OpTypeBool %1
OpConstantTrue $1 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantTrueBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpConstantTrue $1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantFalseGood) {
  const char *spirv = R"(
OpTypeBool %1
OpConstantFalse $1 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantFalseBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpConstantFalse $1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 1)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpConstant $1 %2 0)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantCompositeVectorGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4
OpConstant $1 %3 3.14
OpConstantComposite $2 %4 $3 $3 $3 $3)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeVectorResultTypeBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4
OpConstant $1 %3 3.14
OpConstantComposite $1 %4 $3 $3 $3 $3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeVectorConstituentBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4
OpTypeInt %4 32 0
OpConstant $1 %3 3.14
OpConstant $4 %5 42
OpConstantComposite $2 %6 $3 $5 $3 $3)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeMatrixGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4
OpTypeMatrix %3 $2 4
OpConstant $1 %4 1.0
OpConstant $1 %5 0.0
OpConstantComposite $2 %6 $4 $5 $5 $5
OpConstantComposite $2 %7 $5 $4 $5 $5
OpConstantComposite $2 %8 $5 $5 $4 $5
OpConstantComposite $2 %9 $5 $5 $5 $4
OpConstantComposite $3 %10 $6 $7 $8 $9)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeMatrixConstituentBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeVector %2 $1 4
OpTypeVector %11 $1 3
OpTypeMatrix %3 $2 4
OpConstant $1 %4 1.0
OpConstant $1 %5 0.0
OpConstantComposite $2 %6 $4 $5 $5 $5
OpConstantComposite $2 %7 $5 $4 $5 $5
OpConstantComposite $2 %8 $5 $5 $4 $5
OpConstantComposite $11 %9 $5 $5 $5
OpConstantComposite $3 %10 $6 $7 $8 $9)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeMatrixColumnTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeFloat %2 32
OpTypeVector %3 $1 2
OpTypeVector %4 $3 2
OpTypeMatrix %5 $2 2
OpConstant $1 %6 42
OpConstant $2 %7 3.14
OpConstantComposite $3 %8 $6 $6
OpConstantComposite $4 %9 $7 $7
OpConstantComposite $5 %10 $8 $9)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeArrayGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 4
OpTypeArray %3 $1 $2
OpConstantComposite $3 %4 $2 $2 $2 $2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeArrayConstConstituentBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 4
OpTypeArray %3 $1 $2
OpConstantComposite $3 %4 $2 $2 $2 $1)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeArrayConstituentBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpConstant $1 %2 4
OpTypeArray %3 $1 $2
OpTypeFloat %5 32
OpConstant $5 %6 3.14
OpConstantComposite $3 %4 $2 $2 $2 $6)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantCompositeStructGood) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeInt %2 64 1
OpTypeStruct %3 $1 $1 $2
OpConstant $1 %4 42
OpConstant $2 %5 4300000000
OpConstantComposite $3 %6 $4 $4 $5)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantCompositeStructMemberBad) {
  const char *spirv = R"(
OpTypeInt %1 32 0
OpTypeInt %2 64 1
OpTypeStruct %3 $1 $1 $2
OpConstant $1 %4 42
OpConstant $2 %5 4300000000
OpConstantComposite $3 %6 $4 $5 $4)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantSamplerGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpTypeSampler %2 $1 2D 1 0 1 0
OpConstantSampler $2 %3 ClampToEdge 0 Nearest)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantSamplerResultTypeBad) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpConstantSampler $1 %2 Clamp 0 Nearest)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpConstantNullGood) {
  const char *spirv = R"(
OpTypeBool %1
OpConstantNull $1 %2
OpTypeInt %3 32 0
OpConstantNull $3 %4
OpTypeFloat %5 32
OpConstantNull $5 %6
OpTypePointer %7 UniformConstant $3
OpConstantNull $7 %8
OpTypeEvent %9
OpConstantNull $9 %10
OpTypeDeviceEvent %11
OpConstantNull $11 %12
OpTypeReserveId %13
OpConstantNull $13 %14
OpTypeQueue %15
OpConstantNull $15 %16
OpTypeVector %17 $3 2
OpConstantNull $17 %18
OpTypeMatrix %19 $17 2
OpConstantNull $19 %20
OpConstant $3 %25 8
OpTypeArray %21 $3 $25
OpConstantNull $21 %22
OpTypeStruct %23 $3 $5 $1
OpConstantNull $23 %24
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpConstantNullBasicBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpConstantNull $1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantNullArrayBad) {
  const char *spirv = R"(
OpTypeInt %1 8 0
OpTypeInt %2 32 0
OpTypeSampler %3 $1 2D 0 0 0 0
OpConstant $2 %4 4
OpTypeArray %5 $3 $4
OpConstantNull $5 %6)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpConstantNullStructBad) {
  const char *spirv = R"(
OpTypeInt %1 8 0
OpTypeSampler %2 $1 2D 0 0 0 0
OpTypeStruct %3 $2 $2
OpConstantNull $3 %4)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantTrueGood) {
  const char *spirv = R"(
OpTypeBool %1
OpSpecConstantTrue $1 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantTrueBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpSpecConstantTrue $1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantFalseGood) {
  const char *spirv = R"(
OpTypeBool %1
OpSpecConstantFalse $1 %2)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantFalseBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpSpecConstantFalse $1 %2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpSpecConstantGood) {
  const char *spirv = R"(
OpTypeFloat %1 32
OpSpecConstant $1 %2 42)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpSpecConstantBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpSpecConstant $1 %2 3.14)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpSpecConstantComposite
// TODO: OpSpecConstantOp

TEST_F(ValidateID, OpVariableGood) {
  const char *spirv = R"(
OpTypeInt %1 32 1
OpTypePointer %2 Generic $1
OpVariable $2 %3 Generic)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpVariableInitializerGood) {
  const char *spirv = R"(
OpTypeInt %1 32 1
OpTypePointer %2 Generic $1
OpConstant $1 %3 42
OpVariable $2 %4 Generic $3)";
  CHECK(spirv, SPV_SUCCESS);
}
// TODO: Positive test OpVariable with OpConstantNull of OpTypePointer
TEST_F(ValidateID, OpVariableResultTypeBad) {
  const char *spirv = R"(
OpTypeInt %1 32 1
OpVariable $1 %2 Generic)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpVariableInitializerBad) {
  const char *spirv = R"(
OpTypeInt %1 32 1
OpTypePointer %2 Generic $1
OpVariable $2 %3 Generic $2)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpLoadGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpVariable $3 %5 UniformConstant
OpFunction $1 %6 None $4
OpLabel %7
OpLoad $3 %8 $5
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpLoadResultTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpVariable $3 %5 UniformConstant
OpFunction $1 %6 None $4
OpLabel %7
OpLoad $2 %8 $5
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpLoadPointerBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypeFloat %9 32
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpFunction $1 %6 None $4
OpLabel %7
OpLoad $9 %8 $3
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpStoreGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpConstant $2 %5 42
OpVariable $3 %6 UniformConstant
OpFunction $1 %7 None $4
OpLabel %8
OpStore $6 $5
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpStorePointerBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpConstant $2 %5 42
OpVariable $3 %6 UniformConstant
OpFunction $1 %7 None $4
OpLabel %8
OpStore $3 $5
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpStoreObjectGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpConstant $2 %5 42
OpVariable $3 %6 UniformConstant
OpFunction $1 %7 None $4
OpLabel %8
OpStore $6 $7
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpStoreTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypeFloat %9 32
OpTypePointer %3 UniformConstant $2
OpTypeFunction %4 $1
OpConstant $9 %5 3.14
OpVariable $3 %6 UniformConstant
OpFunction $1 %7 None $4
OpLabel %8
OpStore $6 $5
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpCopyMemoryGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpConstant $2 %4 42
OpVariable $3 %5 UniformConstant $4
OpTypePointer %6 Function $2
OpTypeFunction %7 $1
OpFunction $1 %8 None $7
OpLabel %9
OpVariable $6 %10 Function
OpCopyMemory $10 $5 None
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpCopyMemoryBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpConstant $2 %4 42
OpVariable $3 %5 UniformConstant $4
OpTypeFloat %11 32
OpTypePointer %6 Function $11
OpTypeFunction %7 $1
OpFunction $1 %8 None $7
OpLabel %9
OpVariable $6 %10 Function
OpCopyMemory $10 $5 None
OpReturn
OpFunctionEnd
)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

// TODO: OpCopyMemorySized
TEST_F(ValidateID, OpCopyMemorySizedGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpTypePointer %4 Function $2
OpConstant $2 %5 4
OpVariable $3 %6 UniformConstant $5
OpTypeFunction %7 $1
OpFunction $1 %8 None $7
OpLabel %9
OpVariable $4 %10 Function
OpCopyMemorySized $10 $6 $5 None
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpCopyMemorySizedTargetBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpTypePointer %4 Function $2
OpConstant $2 %5 4
OpVariable $3 %6 UniformConstant $5
OpTypeFunction %7 $1
OpFunction $1 %8 None $7
OpLabel %9
OpCopyMemorySized $9 $6 $5 None
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSourceBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpTypePointer %4 Function $2
OpConstant $2 %5 4
OpTypeFunction %6 $1
OpFunction $1 %7 None $6
OpLabel %8
OpVariable $4 %9 Function
OpCopyMemorySized $9 $6 $5 None
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSizeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpTypePointer %4 Function $2
OpConstant $2 %5 4
OpVariable $3 %6 UniformConstant $5
OpTypeFunction %7 $1
OpFunction $1 %8 None $7
OpLabel %9
OpVariable $4 %10 Function
OpCopyMemorySized $10 $6 $6 None
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpCopyMemorySizedSizeTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypePointer %3 UniformConstant $2
OpTypePointer %4 Function $2
OpConstant $2 %5 4
OpVariable $3 %6 UniformConstant $5
OpTypeFunction %7 $1
OpTypeFloat %11 32
OpConstant $11 %12 1.0
OpFunction $1 %8 None $7
OpLabel %9
OpVariable $4 %10 Function
OpCopyMemorySized $10 $6 $12 None
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
OpTypeVoid %1
OpTypeInt %2 32 1
OpTypeFunction %3 $1 $2 $2
OpFunction $1 %4 None $3
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionResultTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpConstant $2 %5 42
OpTypeFunction %3 $1 $2 $2
OpFunction $2 %4 None $3
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionFunctionTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 1
OpFunction $1 %4 None $2
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpFunctionParameterGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $1 $2
OpFunction $1 %4 None $3
OpFunctionParameter $2 %5
OpLabel %6
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionParameterResultTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $1 $2
OpFunction $1 %4 None $3
OpFunctionParameter $1 %5
OpLabel %6
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionParameterOrderBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $1 $2
OpTypePointer %7 Function $2
OpFunction $1 %4 None $3
OpVariable $7 %8 Function
OpFunctionParameter $2 %5
OpLabel %6
OpReturn
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}

TEST_F(ValidateID, OpFunctionCallGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpTypeFunction %4 $1
OpConstant $2 %5 42 ;21

OpFunction $2 %6 None $3
OpFunctionParameter $2 %7
OpLabel %8
OpLoad $2 %9 $7
OpReturnValue $9
OpFunctionEnd

OpFunction $1 %10 None $4
OpLabel %11
OpReturn
OpFunctionCall $2 %12 $6 $5
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpFunctionCallResultTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpTypeFunction %4 $1
OpConstant $2 %5 42 ;21

OpFunction $2 %6 None $3
OpFunctionParameter $2 %7
OpLabel %8
OpLoad $2 %9 $7
OpReturnValue $9
OpFunctionEnd

OpFunction $1 %10 None $4
OpLabel %11
OpReturn
OpFunctionCall $1 %12 $6 $5
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionCallFunctionBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpTypeFunction %4 $1
OpConstant $2 %5 42 ;21

OpFunction $1 %10 None $4
OpLabel %11
OpReturn
OpFunctionCall $2 %12 $5 $5
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
TEST_F(ValidateID, OpFunctionCallArgumentTypeBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpTypeFunction %4 $1
OpConstant $2 %5 42

OpTypeFloat %13 32
OpConstant $13 %14 3.14

OpFunction $2 %6 None $3
OpFunctionParameter $2 %7
OpLabel %8
OpLoad $2 %9 $7
OpReturnValue $9
OpFunctionEnd

OpFunction $1 %10 None $4
OpLabel %11
OpReturn
OpFunctionCall $2 %12 $6 $14
OpFunctionEnd)";
  CHECK(spirv, SPV_ERROR_INVALID_ID);
}
#if 0
TEST_F(ValidateID, OpFunctionCallArgumentCountBar) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpTypeFunction %4 $1
OpConstant $2 %5 42 ;21

OpFunction $2 %6 None $3
OpFunctionParameter $2 %7
OpLabel %8
OpLoad $2 %9 $7
OpReturnValue $9
OpFunctionEnd

OpFunction $1 %10 None $4
OpLabel %11
OpReturn
OpFunctionCall $2 %12 $6 $5
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
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpConstant $2 %4 42
OpFunction $2 %5 None $3
OpLabel %6
OpReturnValue $4
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpReturnValueVariableGood) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0 ;10
OpTypeFunction %3 $2 $2 ;14
OpTypePointer %8 Function $2 ;18
OpConstant $2 %4 42 ;22
OpFunction $2 %5 None $3 ;27
OpLabel %6 ;29
OpVariable $8 %7 Function $4 ;34
OpReturnValue $7 ;36
OpFunctionEnd)";
  CHECK(spirv, SPV_SUCCESS);
}
TEST_F(ValidateID, OpReturnValueBad) {
  const char *spirv = R"(
OpTypeVoid %1
OpTypeInt %2 32 0
OpTypeFunction %3 $2 $2
OpConstant $2 %4 42
OpFunction $2 %5 None $3
OpLabel %6
OpReturnValue $1
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
