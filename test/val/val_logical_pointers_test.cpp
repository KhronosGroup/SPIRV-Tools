// Copyright (c) 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Combine;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Values;
using ::testing::ValuesIn;

using ValidateLogicalPointersTest = spvtest::ValidateBase<bool>;

enum class MatrixTrace : uint32_t {
  kNotAMatrix,
  kColumn,
  kComponent,
};

const MatrixTrace traces[] = {MatrixTrace::kNotAMatrix, MatrixTrace::kColumn,
                              MatrixTrace::kComponent};

using MatrixTraceTypedTest = spvtest::ValidateBase<MatrixTrace>;

TEST_P(MatrixTraceTypedTest, PhiLoopOp1) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
OpBranch %loop
%loop = OpLabel
%phi = OpPhi )" + type + R"( %gep %entry %copy %continue
OpLoopMerge %merge %continue None
OpBranchConditional %bool_cond %merge %continue
%continue = OpLabel
%copy = OpCopyObject )" + type +
                            R"( %phi
OpBranch %loop
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, PhiLoopOp2) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
OpBranch %loop
%loop = OpLabel
%phi = OpPhi )" + type + R"( %copy %continue %gep %entry
OpLoopMerge %merge %continue None
OpBranchConditional %bool_cond %merge %continue
%continue = OpLabel
%copy = OpCopyObject )" + type +
                            R"( %phi
OpBranch %loop
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, SelectOp1) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
%copy = OpCopyObject )" + type +
                            R"( %gep
%sel = OpSelect )" + type + R"( %bool_cond %copy %null
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, SelectOp2) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
%copy = OpCopyObject )" + type +
                            R"( %gep
%sel = OpSelect )" + type + R"( %bool_cond %null %copy
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionVariable) {
  const auto trace_type = GetParam();
  std::string gep, type, ld_type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_func_wg_v2float";
      ld_type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var_mat2x2 %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_func_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_mat2x2 %index %index";
      break;
    default:
      type = "%ptr_func_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_float";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_float = OpTypePointer Workgroup %float
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_mat2x2 = OpTypePointer Workgroup %mat2x2
%ptr_func_wg_float = OpTypePointer Function %ptr_wg_float
%ptr_func_wg_v2float = OpTypePointer Function %ptr_wg_v2float
%var_mat2x2 = OpVariable %ptr_wg_mat2x2 Workgroup
%var_float = OpVariable %ptr_wg_float Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%func_var = OpVariable )" + type +
                            R"( Function
%gep = )" + gep + R"(
OpStore %func_var %gep
%ld = OpLoad )" + ld_type + R"( %func_var
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, PrivateVariable) {
  const auto trace_type = GetParam();
  std::string gep, type, ld_type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_priv_wg_v2float";
      ld_type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var_mat2x2 %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_priv_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_mat2x2 %index %index";
      break;
    default:
      type = "%ptr_priv_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_float";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_float = OpTypePointer Workgroup %float
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_mat2x2 = OpTypePointer Workgroup %mat2x2
%ptr_priv_wg_float = OpTypePointer Private %ptr_wg_float
%ptr_priv_wg_v2float = OpTypePointer Private %ptr_wg_v2float
%var_mat2x2 = OpVariable %ptr_wg_mat2x2 Workgroup
%var_float = OpVariable %ptr_wg_float Workgroup
%priv_var = OpVariable )" + type +
                            R"( Private
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
OpStore %priv_var %gep
%ld = OpLoad )" + ld_type + R"( %priv_var
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionVariableAggregate) {
  const auto trace_type = GetParam();
  std::string gep, var_type, gep_type, ld_type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      var_type = "%ptr_func_wg_struct_v2float";
      gep_type = "%ptr_func_wg_v2float";
      ld_type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var_mat2x2 %index";
      break;
    case MatrixTrace::kComponent:
      var_type = "%ptr_func_wg_struct_float";
      gep_type = "%ptr_func_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_mat2x2 %index %index";
      break;
    default:
      var_type = "%ptr_func_wg_struct_float";
      gep_type = "%ptr_func_wg_float";
      ld_type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var_float";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_float = OpTypePointer Workgroup %float
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_mat2x2 = OpTypePointer Workgroup %mat2x2
%func_array_float = OpTypeArray %ptr_wg_float %int_1
%func_array_v2float = OpTypeArray %ptr_wg_v2float %int_1
%func_struct_float = OpTypeStruct %func_array_float
%func_struct_v2float = OpTypeStruct %func_array_v2float
%ptr_func_wg_struct_float = OpTypePointer Function %func_struct_float
%ptr_func_wg_struct_v2float = OpTypePointer Function %func_struct_v2float
%ptr_func_wg_float = OpTypePointer Function %ptr_wg_float
%ptr_func_wg_v2float = OpTypePointer Function %ptr_wg_v2float
%var_mat2x2 = OpVariable %ptr_wg_mat2x2 Workgroup
%var_float = OpVariable %ptr_wg_float Workgroup
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
%func_var = OpVariable )" + var_type +
                            R"( Function
%gep = )" + gep + R"(
%store_gep = OpAccessChain )" +
                            gep_type + R"( %func_var %int_0 %index
%store_gep_copy = OpCopyObject )" +
                            gep_type + R"( %store_gep
OpStore %store_gep_copy %gep
%ld_gep = OpAccessChain )" + gep_type +
                            R"( %func_var %int_0 %index
%ld = OpLoad )" + ld_type + R"( %ld_gep
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionCallParam1) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%foo_fn = OpTypeFunction %void )" +
                            type + R"( %bool
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
%copy = OpCopyObject )" + type +
                            R"( %gep
%call = OpFunctionCall %void %foo %copy %bool_cond
OpReturn
OpFunctionEnd
%foo = OpFunction %void None %foo_fn
%ptr_param = OpFunctionParameter )" +
                            type + R"(
%bool_param = OpFunctionParameter %bool
%foo_entry = OpLabel
%sel = OpSelect )" + type + R"( %bool_param %ptr_param %null
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionCallParam2) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%foo_fn = OpTypeFunction %void %bool )" +
                            type + R"(
%main = OpFunction %void None %void_fn
%entry = OpLabel
%gep = )" + gep + R"(
%copy = OpCopyObject )" + type +
                            R"( %gep
%call = OpFunctionCall %void %foo %bool_cond %copy
OpReturn
OpFunctionEnd
%foo = OpFunction %void None %foo_fn
%bool_param = OpFunctionParameter %bool
%ptr_param = OpFunctionParameter )" +
                            type + R"(
%foo_entry = OpLabel
%sel = OpSelect )" + type + R"( %bool_param %ptr_param %null
OpReturn
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionCall) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%foo_ty = OpTypeFunction )" +
                            type + R"(
%main = OpFunction %void None %void_fn
%entry = OpLabel
%call = OpFunctionCall )" + type +
                            R"( %foo
OpReturn
OpFunctionEnd
%foo = OpFunction )" + type +
                            R"( None %foo_ty
%foo_entry = OpLabel
%gep = )" + gep + R"(
OpReturnValue %gep
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionCallMultiReturn1) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%foo_ty = OpTypeFunction )" +
                            type + R"(
%main = OpFunction %void None %void_fn
%entry = OpLabel
%call = OpFunctionCall )" + type +
                            R"( %foo
OpReturn
OpFunctionEnd
%foo = OpFunction )" + type +
                            R"( None %foo_ty
%foo_entry = OpLabel
%gep = )" + gep + R"(
OpSelectionMerge %merge None
OpBranchConditional %bool_cond %then %merge
%then = OpLabel
OpReturnValue %gep
%merge = OpLabel
OpReturnValue %null
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

TEST_P(MatrixTraceTypedTest, FunctionCallMultiReturn2) {
  const auto trace_type = GetParam();
  std::string gep, type;
  switch (trace_type) {
    case MatrixTrace::kColumn:
      type = "%ptr_wg_v2float";
      gep = "OpAccessChain %ptr_wg_v2float %var %int_0 %index %index";
      break;
    case MatrixTrace::kComponent:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_0 %index %index %index";
      break;
    default:
      type = "%ptr_wg_float";
      gep = "OpAccessChain %ptr_wg_float %var %int_1";
      break;
  }

  const std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointers
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%bool_cond = OpUndef %bool
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%index = OpUndef %int
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%array = OpTypeArray %mat2x2 %int_2
%struct = OpTypeStruct %array %float
%ptr_wg_struct = OpTypePointer Workgroup %struct
%ptr_wg_v2float = OpTypePointer Workgroup %v2float
%ptr_wg_float = OpTypePointer Workgroup %float
%null = OpConstantNull )" + type +
                            R"(
%var = OpVariable %ptr_wg_struct Workgroup
%void_fn = OpTypeFunction %void
%foo_ty = OpTypeFunction )" +
                            type + R"(
%main = OpFunction %void None %void_fn
%entry = OpLabel
%call = OpFunctionCall )" + type +
                            R"( %foo
OpReturn
OpFunctionEnd
%foo = OpFunction )" + type +
                            R"( None %foo_ty
%foo_entry = OpLabel
%gep = )" + gep + R"(
OpSelectionMerge %merge None
OpBranchConditional %bool_cond %then %merge
%then = OpLabel
OpReturnValue %null
%merge = OpLabel
OpReturnValue %gep
OpFunctionEnd
)";

  const auto expected = trace_type == MatrixTrace::kNotAMatrix
                            ? SPV_SUCCESS
                            : SPV_ERROR_INVALID_DATA;
  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(expected, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  if (expected) {
    EXPECT_THAT(getDiagnosticString(),
                HasSubstr("Variable pointer must not point to a column or a "
                          "component of a column of a matrix"));
  }
}

INSTANTIATE_TEST_SUITE_P(ValidateLogicalPointersMatrixTraceTyped,
                         MatrixTraceTypedTest, ValuesIn(traces));
}  // namespace
}  // namespace val
}  // namespace spvtools
