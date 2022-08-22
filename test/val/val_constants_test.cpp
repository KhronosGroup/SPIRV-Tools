// Copyright (c) 2019 Google LLC
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

// Test validation of constants.
//
// This file contains newer tests.  Older tests may be in other files such as
// val_id_test.cpp.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_code_generator.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Combine;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Values;
using ::testing::ValuesIn;

using ValidateConstant = spvtest::ValidateBase<bool>;

#define kBasicTypes                                        \
  "%bool = OpTypeBool "                                    \
  "%uint = OpTypeInt 32 0 "                                \
  "%uint64 = OpTypeInt 64 0 "                              \
  "%uint2 = OpTypeVector %uint 2 "                         \
  "%float = OpTypeFloat 32 "                               \
  "%float64 = OpTypeFloat 64 "                             \
  "%_ptr_uint = OpTypePointer Workgroup %uint "            \
  "%uint_0 = OpConstantNull %uint "                        \
  "%uint64_0 = OpConstantNull %uint64 "                    \
  "%uint2_0 = OpConstantComposite %uint2 %uint_0 %uint_0 " \
  "%float_0 = OpConstantNull %float "                      \
  "%float64_0 = OpConstantNull %float64 "                  \
  "%false = OpConstantFalse %bool "                        \
  "%true = OpConstantTrue %bool "                          \
  "%null = OpConstantNull %_ptr_uint "

#define kShaderPreamble    \
  "OpCapability Shader\n"  \
  "OpCapability Linkage\n" \
  "OpCapability Int64\n"   \
  "OpCapability Float64\n" \
  "OpMemoryModel Logical Simple\n"

#define kKernelPreamble           \
  "OpCapability Kernel\n"         \
  "OpCapability Linkage\n"        \
  "OpCapability Int64\n"          \
  "OpCapability Float64\n"        \
  "OpCapability GenericPointer\n" \
  "OpCapability Addresses\n"      \
  "OpMemoryModel Physical32 OpenCL\n"

struct ConstantOpCase {
  spv_target_env env;
  std::string assembly;
  bool expect_success;
  std::string expect_err;
};

using ValidateConstantOp = spvtest::ValidateBase<ConstantOpCase>;

TEST_P(ValidateConstantOp, Samples) {
  const auto env = GetParam().env;
  CompileSuccessfully(GetParam().assembly, env);
  const auto result = ValidateInstructions(env);
  if (GetParam().expect_success) {
    EXPECT_EQ(SPV_SUCCESS, result);
    EXPECT_THAT(getDiagnosticString(), Eq(""));
  } else {
    EXPECT_EQ(SPV_ERROR_INVALID_ID, result);
    EXPECT_THAT(getDiagnosticString(), HasSubstr(GetParam().expect_err));
  }
}

#define GOOD_SHADER_10(STR) \
  { SPV_ENV_UNIVERSAL_1_0, kShaderPreamble kBasicTypes STR, true, "" }
#define GOOD_KERNEL_10(STR) \
  { SPV_ENV_UNIVERSAL_1_0, kKernelPreamble kBasicTypes STR, true, "" }
INSTANTIATE_TEST_SUITE_P(
    UniversalInShader, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SConvert %uint64_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %float FConvert %float64_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SNegate %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint Not %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint IAdd %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint ISub %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint IMul %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint UDiv %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SDiv %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint UMod %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SRem %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint SMod %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftRightLogical %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftRightArithmetic %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint ShiftLeftLogical %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %uint BitwiseOr %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint BitwiseXor %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint2 VectorShuffle %uint2_0 %uint2_0 1 3"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint CompositeExtract %uint2_0 1"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint2 CompositeInsert %uint_0 %uint2_0 1"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalOr %true %false"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalNot %true"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalAnd %true %false"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool LogicalEqual %true %false"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool LogicalNotEqual %true %false"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %uint Select %true %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool IEqual %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool INotEqual %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool ULessThan %uint_0 %uint_0"),
        GOOD_SHADER_10("%v = OpSpecConstantOp %bool SLessThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool ULessThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SLessThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool UGreaterThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool UGreaterThanEqual %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SGreaterThan %uint_0 %uint_0"),
        GOOD_SHADER_10(
            "%v = OpSpecConstantOp %bool SGreaterThanEqual %uint_0 %uint_0"),
    }));

INSTANTIATE_TEST_SUITE_P(
    UniversalInKernel, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SConvert %uint64_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FConvert %float64_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SNegate %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint Not %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint IAdd %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ISub %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint IMul %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UDiv %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SDiv %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UMod %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SRem %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint SMod %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftRightLogical %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftRightArithmetic %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint ShiftLeftLogical %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint BitwiseOr %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint BitwiseXor %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint2 VectorShuffle %uint2_0 %uint2_0 1 3"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint CompositeExtract %uint2_0 1"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint2 CompositeInsert %uint_0 %uint2_0 1"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalOr %true %false"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalNot %true"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalAnd %true %false"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool LogicalEqual %true %false"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool LogicalNotEqual %true %false"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %uint Select %true %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool IEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool INotEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool ULessThan %uint_0 %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %bool SLessThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool ULessThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SLessThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool UGreaterThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool UGreaterThanEqual %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SGreaterThan %uint_0 %uint_0"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %bool SGreaterThanEqual %uint_0 %uint_0"),
    }));

INSTANTIATE_TEST_SUITE_P(
    UConvert, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        {SPV_ENV_UNIVERSAL_1_0,
         kKernelPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
        {SPV_ENV_UNIVERSAL_1_1,
         kKernelPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
        {SPV_ENV_UNIVERSAL_1_3,
         kKernelPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
        {SPV_ENV_UNIVERSAL_1_3,
         kKernelPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
        {SPV_ENV_UNIVERSAL_1_4,
         kKernelPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
        {SPV_ENV_UNIVERSAL_1_0,
         kShaderPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         false,
         "Prior to SPIR-V 1.4, specialization constant operation "
         "UConvert requires Kernel capability"},
        {SPV_ENV_UNIVERSAL_1_1,
         kShaderPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         false,
         "Prior to SPIR-V 1.4, specialization constant operation "
         "UConvert requires Kernel capability"},
        {SPV_ENV_UNIVERSAL_1_3,
         kShaderPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         false,
         "Prior to SPIR-V 1.4, specialization constant operation "
         "UConvert requires Kernel capability"},
        {SPV_ENV_UNIVERSAL_1_3,
         kShaderPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         false,
         "Prior to SPIR-V 1.4, specialization constant operation "
         "UConvert requires Kernel capability"},
        {SPV_ENV_UNIVERSAL_1_4,
         kShaderPreamble kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
    }));

INSTANTIATE_TEST_SUITE_P(
    KernelInKernel, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ConvertFToS %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float ConvertSToF %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint ConvertFToU %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float ConvertUToF %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint UConvert %uint64_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %uint Bitcast %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FNegate %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FAdd %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FSub %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FMul %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FDiv %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FRem %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %float FMod %float_0 %float_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %_ptr_uint AccessChain %null"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint InBoundsAccessChain %null"),
        GOOD_KERNEL_10(
            "%v = OpSpecConstantOp %_ptr_uint PtrAccessChain %null %uint_0"),
        GOOD_KERNEL_10("%v = OpSpecConstantOp %_ptr_uint "
                       "InBoundsPtrAccessChain %null %uint_0"),
    }));

#define BAD_SHADER_10(STR, NAME)                                   \
  {                                                                \
    SPV_ENV_UNIVERSAL_1_0, kShaderPreamble kBasicTypes STR, false, \
        "Specialization constant operation " NAME                  \
        " requires Kernel capability"                              \
  }
INSTANTIATE_TEST_SUITE_P(
    KernelInShader, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // Don't need to test GenericCastToPtr or PtrCastToGeneric as a valid
        // module can't have a Generic storage class with Shader capability
        BAD_SHADER_10("%v = OpSpecConstantOp %uint ConvertFToS %float_0",
                      "ConvertFToS"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float ConvertSToF %uint_0",
                      "ConvertSToF"),
        BAD_SHADER_10("%v = OpSpecConstantOp %uint ConvertFToU %float_0",
                      "ConvertFToU"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float ConvertUToF %uint_0",
                      "ConvertUToF"),
        BAD_SHADER_10("%v = OpSpecConstantOp %uint Bitcast %uint_0", "Bitcast"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FNegate %float_0",
                      "FNegate"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FAdd %float_0 %float_0",
                      "FAdd"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FSub %float_0 %float_0",
                      "FSub"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FMul %float_0 %float_0",
                      "FMul"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FDiv %float_0 %float_0",
                      "FDiv"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FRem %float_0 %float_0",
                      "FRem"),
        BAD_SHADER_10("%v = OpSpecConstantOp %float FMod %float_0 %float_0",
                      "FMod"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint AccessChain %null",
                      "AccessChain"),
        BAD_SHADER_10(
            "%v = OpSpecConstantOp %_ptr_uint InBoundsAccessChain %null",
            "InBoundsAccessChain"),
        BAD_SHADER_10(
            "%v = OpSpecConstantOp %_ptr_uint PtrAccessChain %null %uint_0",
            "PtrAccessChain"),
        BAD_SHADER_10("%v = OpSpecConstantOp %_ptr_uint "
                      "InBoundsPtrAccessChain %null %uint_0",
                      "InBoundsPtrAccessChain"),
    }));

INSTANTIATE_TEST_SUITE_P(
    UConvertInAMD_gpu_shader_int16, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // SPV_AMD_gpu_shader_int16 should enable UConvert for OpSpecConstantOp
        // https://github.com/KhronosGroup/glslang/issues/848
        {SPV_ENV_UNIVERSAL_1_0,
         "OpCapability Shader "
         "OpCapability Int64 "
         "OpCapability Float64 "
         "OpCapability Linkage ; So we don't need to define a function\n"
         "OpExtension \"SPV_AMD_gpu_shader_int16\" "
         "OpMemoryModel Logical Simple " kBasicTypes
         "%v = OpSpecConstantOp %uint UConvert %uint64_0",
         true, ""},
    }));

TEST_F(ValidateConstant, SpecConstantUConvert1p3Binary1p4EnvBad) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%int = OpTypeInt 32 0
%int0 = OpConstant %int 0
%const = OpSpecConstantOp %int UConvert %int0
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_4));
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "Prior to SPIR-V 1.4, specialization constant operation UConvert "
          "requires Kernel capability or extension SPV_AMD_gpu_shader_int16"));
}

using SmallStorageConstants = spvtest::ValidateBase<std::string>;

CodeGenerator GetSmallStorageCodeGenerator() {
  CodeGenerator generator;
  generator.capabilities_ = R"(
OpCapability Shader
OpCapability Linkage
OpCapability UniformAndStorageBuffer16BitAccess
OpCapability StoragePushConstant16
OpCapability StorageInputOutput16
OpCapability UniformAndStorageBuffer8BitAccess
OpCapability StoragePushConstant8
)";
  generator.extensions_ = R"(
OpExtension "SPV_KHR_16bit_storage"
OpExtension "SPV_KHR_8bit_storage"
)";
  generator.memory_model_ = "OpMemoryModel Logical GLSL450\n";
  generator.types_ = R"(
%short = OpTypeInt 16 0
%short2 = OpTypeVector %short 2
%char = OpTypeInt 8 0
%char2 = OpTypeVector %char 2
%half = OpTypeFloat 16
%half2 = OpTypeVector %half 2
%int = OpTypeInt 32 0
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
)";
  return generator;
}

TEST_P(SmallStorageConstants, SmallConstant) {
  std::string constant = GetParam();
  CodeGenerator generator = GetSmallStorageCodeGenerator();
  generator.after_types_ += constant + "\n";
  CompileSuccessfully(generator.Build(), SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Cannot form constants of 8- or 16-bit types"));
}

// Constant composites would be caught through scalar constants.
INSTANTIATE_TEST_SUITE_P(
    SmallConstants, SmallStorageConstants,
    Values("%c = OpConstant %char 0", "%c = OpConstantNull %char2",
           "%c = OpConstant %short 0", "%c = OpConstantNull %short",
           "%c = OpConstant %half 0", "%c = OpConstantNull %half",
           "%c = OpSpecConstant %char 0", "%c = OpSpecConstant %short 0",
           "%c = OpSpecConstant %half 0",
           "%c = OpSpecConstantOp %char SConvert %int_0",
           "%c = OpSpecConstantOp %short SConvert %int_0",
           "%c = OpSpecConstantOp %half FConvert %float_0"));

TEST_F(ValidateConstant, NullPointerTo16BitStorageOk) {
  std::string spirv = R"(
OpCapability Shader
OpCapability VariablePointersStorageBuffer
OpCapability UniformAndStorageBuffer16BitAccess
OpCapability Linkage
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
%half = OpTypeFloat 16
%ptr_ssbo_half = OpTypePointer StorageBuffer %half
%null_ptr = OpConstantNull %ptr_ssbo_half
)";

  CompileSuccessfully(spirv, SPV_ENV_UNIVERSAL_1_3);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions(SPV_ENV_UNIVERSAL_1_3));
}

TEST_F(ValidateConstant, NullMatrix) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%mat2x2 = OpTypeMatrix %v2float 2
%null_vector = OpConstantNull %v2float
%null_matrix = OpConstantComposite %mat2x2 %null_vector %null_vector
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateConstant, NullPhysicalStorageBuffer) {
  std::string spirv = R"(
OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpCapability Linkage
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpName %ptr "ptr"
%int = OpTypeInt 32 0
%ptr = OpTypePointer PhysicalStorageBuffer %int
%null = OpConstantNull %ptr
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpConstantNull Result Type <id> '1[%ptr]' cannot have "
                        "a null value"));
}

TEST_F(ValidateConstant, BadShaderOperandsQuantizeToF16) {
  std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%uint_1 = OpConstant %uint 1
%float_1 = OpConstant %float 1
%good = OpSpecConstantOp %float QuantizeToF16 %float_1
%bad1 = OpSpecConstantOp %uint QuantizeToF16 %float_1
%bad2 = OpSpecConstantOp %float QuantizeToF16 %uint_1
)";

  CompileSuccessfully(spirv);
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("OpSpecConstantOp opcode QuantizeToF16 has invalid "
                        "types or operands."));
}

#define BAD_KERNEL_OPERANDS(STR, NAME)                                    \
  {                                                                       \
    SPV_ENV_UNIVERSAL_1_0, kKernelPreamble kBasicTypes STR, false,        \
        "OpSpecConstantOp opcode " NAME " has invalid types or operands." \
  }

INSTANTIATE_TEST_SUITE_P(
    BadOperandsKernel, ValidateConstantOp,
    ValuesIn(std::vector<ConstantOpCase>{
        // 2 of each, first has bad return type, second has bad operand
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float SConvert %uint_0",
                            "SConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint SConvert %uint_0",
                            "SConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint FConvert %float_0",
                            "FConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float FConvert %float_0",
                            "FConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float SNegate %uint_0",
                            "SNegate"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint SNegate %float_0",
                            "SNegate"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float Not %uint_0", "Not"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint Not %float_0", "Not"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float IAdd %uint_0 %uint_0",
                            "IAdd"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint IAdd %uint_0 %float_0",
                            "IAdd"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float ISub %uint_0 %uint_0",
                            "ISub"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint ISub %uint_0 %float_0",
                            "ISub"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float IMul %uint_0 %uint_0",
                            "IMul"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint IMul %uint_0 %float_0",
                            "IMul"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float UDiv %uint_0 %uint_0",
                            "UDiv"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint UDiv %uint_0 %float_0",
                            "UDiv"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float SDiv %uint_0 %uint_0",
                            "SDiv"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint SDiv %uint_0 %float_0",
                            "SDiv"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float UMod %uint_0 %uint_0",
                            "UMod"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint UMod %uint_0 %float_0",
                            "UMod"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float SRem %uint_0 %uint_0",
                            "SRem"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint SRem %uint_0 %float_0",
                            "SRem"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float SMod %uint_0 %uint_0",
                            "SMod"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint SMod %uint_0 %float_0",
                            "SMod"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float ShiftRightLogical %uint_0 %uint_0",
            "ShiftRightLogical"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint ShiftRightLogical %uint_0 %float_0",
            "ShiftRightLogical"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float ShiftRightArithmetic %uint_0 %uint_0",
            "ShiftRightArithmetic"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint ShiftRightArithmetic %uint_0 %float_0",
            "ShiftRightArithmetic"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float ShiftLeftLogical %uint_0 %uint_0",
            "ShiftLeftLogical"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint ShiftLeftLogical %uint_0 %float_0",
            "ShiftLeftLogical"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float BitwiseOr %uint_0 %uint_0",
            "BitwiseOr"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint BitwiseOr %uint_0 %float_0",
            "BitwiseOr"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float BitwiseXor %uint_0 %uint_0",
            "BitwiseXor"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint BitwiseXor %uint_0 %float_0",
            "BitwiseXor"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float VectorShuffle %uint2_0 %uint2_0 1 3",
            "VectorShuffle"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint2 VectorShuffle %uint2_0 %uint_0 1 3",
            "VectorShuffle"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float CompositeExtract %uint2_0 1",
            "CompositeExtract"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint CompositeExtract %uint_0 1",
            "CompositeExtract"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float CompositeInsert %uint_0 %uint2_0 1",
            "CompositeInsert"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint2 CompositeInsert %uint_0 %uint_0 1",
            "CompositeInsert"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float LogicalOr %true %false", "LogicalOr"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool LogicalOr %true %uint_0", "LogicalOr"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float LogicalNot %true",
                            "LogicalNot"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %bool LogicalNot %uint_0",
                            "LogicalNot"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float LogicalAnd %true %false",
            "LogicalAnd"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool LogicalAnd %true %uint_0",
            "LogicalAnd"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float LogicalEqual %true %false",
            "LogicalEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool LogicalEqual %true %uint_0",
            "LogicalEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float LogicalNotEqual %true %false",
            "LogicalNotEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool LogicalNotEqual %uint_0 %false",
            "LogicalNotEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float Select %true %uint_0 %uint_0",
            "Select"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint Select %uint_0 %uint_0 %uint_0",
            "Select"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float IEqual %uint_0 %uint_0", "IEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool IEqual %uint_0 %float_0", "IEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float INotEqual %uint_0 %uint_0",
            "INotEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool INotEqual %uint_0 %float_0",
            "INotEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float ULessThan %uint_0 %uint_0",
            "ULessThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool ULessThan %uint_0 %float_0",
            "ULessThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float SLessThan %uint_0 %uint_0",
            "SLessThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool SLessThan %uint_0 %float_0",
            "SLessThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float ULessThanEqual %uint_0 %uint_0",
            "ULessThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool ULessThanEqual %uint_0 %float_0",
            "ULessThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float SLessThanEqual %uint_0 %uint_0",
            "SLessThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool SLessThanEqual %uint_0 %float_0",
            "SLessThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float UGreaterThan %uint_0 %uint_0",
            "UGreaterThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool UGreaterThan %uint_0 %float_0",
            "UGreaterThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float UGreaterThanEqual %uint_0 %uint_0",
            "UGreaterThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool UGreaterThanEqual %uint_0 %float_0",
            "UGreaterThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float SGreaterThan %uint_0 %uint_0",
            "SGreaterThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool SGreaterThan %uint_0 %float_0",
            "SGreaterThan"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float SGreaterThanEqual %uint_0 %uint_0",
            "SGreaterThanEqual"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %bool SGreaterThanEqual %uint_0 %float_0",
            "SGreaterThanEqual"),
        // // Kernel only
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float ConvertFToS %float_0",
                            "ConvertFToS"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint ConvertFToS %uint2_0",
                            "ConvertFToS"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint ConvertSToF %uint_0",
                            "ConvertSToF"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float ConvertSToF %float_0",
                            "ConvertSToF"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float ConvertFToU %float_0",
                            "ConvertFToU"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint ConvertFToU %uint2_0",
                            "ConvertFToU"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint ConvertUToF %uint_0",
                            "ConvertUToF"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float ConvertUToF %float_0",
                            "ConvertUToF"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float UConvert %uint_0",
                            "UConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint UConvert %uint_0",
                            "UConvert"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %bool Bitcast %uint_0",
                            "Bitcast"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint Bitcast %true",
                            "Bitcast"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint FNegate %float_0",
                            "FNegate"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %float FNegate %uint_0",
                            "FNegate"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FAdd %float_0 %float_0", "FAdd"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FAdd %float_0 %uint_0", "FAdd"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FSub %float_0 %float_0", "FSub"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FSub %float_0 %uint_0", "FSub"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FMul %float_0 %float_0", "FMul"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FMul %float_0 %uint_0", "FMul"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FDiv %float_0 %float_0", "FDiv"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FDiv %float_0 %uint_0", "FDiv"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FRem %float_0 %float_0", "FRem"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FRem %float_0 %uint_0", "FRem"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint FMod %float_0 %float_0", "FMod"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %float FMod %float_0 %uint_0", "FMod"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %uint AccessChain %null",
                            "AccessChain"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %_ptr_uint AccessChain %null %float_0",
            "AccessChain"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint InBoundsAccessChain %null",
            "InBoundsAccessChain"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %_ptr_uint "
                            "InBoundsAccessChain %null %float_0",
                            "InBoundsAccessChain"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint PtrAccessChain %null %uint_0",
            "PtrAccessChain"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %_ptr_uint PtrAccessChain %float_0 %float_0",
            "PtrAccessChain"),
        BAD_KERNEL_OPERANDS(
            "%v = OpSpecConstantOp %uint InBoundsPtrAccessChain %null %uint_0",
            "InBoundsPtrAccessChain"),
        BAD_KERNEL_OPERANDS("%v = OpSpecConstantOp %_ptr_uint "
                            "InBoundsPtrAccessChain %float_0 %float_0",
                            "InBoundsPtrAccessChain"),
    }));

}  // namespace
}  // namespace val
}  // namespace spvtools
