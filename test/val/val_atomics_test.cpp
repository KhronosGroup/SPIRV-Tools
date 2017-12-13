// Copyright (c) 2017 Google Inc.
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

#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using ::testing::HasSubstr;
using ::testing::Not;

using ValidateAtomics = spvtest::ValidateBase<bool>;

std::string GenerateKernelCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "") {
  std::ostringstream ss;
  ss << R"(
OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpCapability Int64
)";

  ss << capabilities_and_extensions;
  ss << R"(
OpMemoryModel Physical32 OpenCL
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%f32vec4 = OpTypeVector %f32 4

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u64_1 = OpConstant %u64 1
%f32vec4_0000 = OpConstantComposite %f32vec4 %f32_0 %f32_0 %f32_0 %f32_0

%scope = OpConstant %u32 1
%memory_semantics = OpConstant %u32 1

%f32_ptr = OpTypePointer Workgroup %f32
%f32_var = OpVariable %f32_ptr Workgroup

%u32_ptr = OpTypePointer Workgroup %u32
%u32_var = OpVariable %u32_ptr Workgroup

%u64_ptr = OpTypePointer Workgroup %u64
%u64_var = OpVariable %u64_ptr Workgroup

%f32vec4_ptr = OpTypePointer Workgroup %f32vec4
%f32vec4_var = OpVariable %f32vec4_ptr Workgroup

%f32_ptr_function = OpTypePointer Function %f32

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_F(ValidateAtomics, AtomicLoadSuccess) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %scope %memory_semantics
%val2 = OpAtomicLoad %u32 %u32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicLoadWrongResultType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32vec4 %f32vec4_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongPointerType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_ptr %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %u32 %f32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongScopeType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %f32_1 %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicLoad: expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicLoadWrongMemorySemanticsType) {
  const std::string body = R"(
%val1 = OpAtomicLoad %f32 %f32_var %scope %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicLoad: expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicStoreSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicStore %u32_var %scope %memory_semantics %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerType) {
  const std::string body = R"(
OpAtomicStore %f32_1 %scope %memory_semantics %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerDataType) {
  const std::string body = R"(
OpAtomicStore %f32vec4_var %scope %memory_semantics %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: "
                "expected Pointer to be a pointer to int or float scalar "
                "type"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongPointerStorageType) {
  const std::string body = R"(
%f32_var_function = OpVariable %f32_ptr_function Function
OpAtomicStore %f32_var_function %scope %memory_semantics %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Pointer Storage Class to be Uniform, "
                "Workgroup, CrossWorkgroup, Generic, AtomicCounter, Image or "
                "StorageBuffer"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %f32_1 %memory_semantics %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicStore: expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %f32_1 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicStoreWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicStore: "
                "expected Value type and the type pointed to by Pointer to "
                "be the same"));
}

TEST_F(ValidateAtomics, AtomicExchangeSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %scope %memory_semantics %f32_0
%val3 = OpAtomicStore %u32_var %scope %memory_semantics %u32_1
%val4 = OpAtomicExchange %u32 %u32_var %scope %memory_semantics %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicExchangeWrongResultType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicExchange %f32vec4 %f32vec4_var %scope %memory_semantics %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongPointerType) {
  const std::string body = R"(
%val2 = OpAtomicExchange %f32 %f32vec4_ptr %scope %memory_semantics %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicExchange: expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicExchange %f32 %f32vec4_var %scope %memory_semantics %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicExchange: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %f32_1 %memory_semantics %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %scope %f32_1 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicExchange: expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicExchangeWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicExchange %f32 %f32_var %scope %memory_semantics %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicExchange: "
                        "expected Value to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeSuccess) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %scope %memory_semantics %memory_semantics %f32_0 %f32_1
%val3 = OpAtomicStore %u32_var %scope %memory_semantics %u32_1
%val4 = OpAtomicCompareExchange %u32 %u32_var %scope %memory_semantics %memory_semantics %u32_0 %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongResultType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicCompareExchange %f32vec4 %f32vec4_var %scope %memory_semantics %memory_semantics %f32vec4_0000 %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Result Type to be int or float scalar type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongPointerType) {
  const std::string body = R"(
%val2 = OpAtomicCompareExchange %f32 %f32vec4_ptr %scope %memory_semantics %memory_semantics %f32vec4_0000 %f32vec4_0000
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: expected Pointer to be of type "
                        "OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongPointerDataType) {
  const std::string body = R"(
%val1 = OpStore %f32vec4_var %f32vec4_0000
%val2 = OpAtomicCompareExchange %f32 %f32vec4_var %scope %memory_semantics %memory_semantics %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicCompareExchange: "
                "expected Pointer to point to a value of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongScopeType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %f32_1 %memory_semantics %memory_semantics %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicCompareExchange: expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongMemorySemanticsType1) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %scope %f32_1 %memory_semantics %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicCompareExchange: expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongMemorySemanticsType2) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %scope %memory_semantics %f32_1 %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr(
          "AtomicCompareExchange: expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongValueType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %scope %memory_semantics %memory_semantics %u32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Value to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWrongComparatorType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchange %f32 %f32_var %scope %memory_semantics %memory_semantics %f32_0 %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchange: "
                        "expected Comparator to be of type Result Type"));
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWeakSuccess) {
  const std::string body = R"(
%val3 = OpAtomicStore %u32_var %scope %memory_semantics %u32_1
%val4 = OpAtomicCompareExchangeWeak %u32 %u32_var %scope %memory_semantics %memory_semantics %u32_0 %u32_0
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicCompareExchangeWeakWrongResultType) {
  const std::string body = R"(
OpAtomicStore %f32_var %scope %memory_semantics %f32_1
%val2 = OpAtomicCompareExchangeWeak %f32 %f32_var %scope %memory_semantics %memory_semantics %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicCompareExchangeWeak: "
                        "expected Result Type to be int scalar type"));
}

TEST_F(ValidateAtomics, AtomicArithmeticsSuccess) {
  const std::string body = R"(
OpAtomicStore %u32_var %scope %memory_semantics %u32_1
%val1 = OpAtomicIIncrement %u32 %u32_var %scope %memory_semantics
%val2 = OpAtomicIDecrement %u32 %u32_var %scope %memory_semantics
%val3 = OpAtomicIAdd %u32 %u32_var %scope %memory_semantics %u32_1
%val4 = OpAtomicISub %u32 %u32_var %scope %memory_semantics %u32_1
%val5 = OpAtomicUMin %u32 %u32_var %scope %memory_semantics %u32_1
%val6 = OpAtomicUMax %u32 %u32_var %scope %memory_semantics %u32_1
%val7 = OpAtomicSMin %u32 %u32_var %scope %memory_semantics %u32_1
%val8 = OpAtomicSMax %u32 %u32_var %scope %memory_semantics %u32_1
%val9 = OpAtomicAnd %u32 %u32_var %scope %memory_semantics %u32_1
%val10 = OpAtomicOr %u32 %u32_var %scope %memory_semantics %u32_1
%val11 = OpAtomicXor %u32 %u32_var %scope %memory_semantics %u32_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicFlagsSuccess) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %scope %memory_semantics
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongResultType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %u32 %u32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Result Type to be bool scalar type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotPointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_1 %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotIntPointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %f32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagTestAndSet: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetNotInt32Pointer) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u64_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagTestAndSet: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongScopeType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %u64_1 %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicFlagTestAndSetWrongMemorySemanticsType) {
  const std::string body = R"(
%val1 = OpAtomicFlagTestAndSet %bool %u32_var %scope %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagTestAndSet: "
                        "expected Memory Semantics to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotPointer) {
  const std::string body = R"(
OpAtomicFlagClear %u32_1 %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagClear: "
                        "expected Pointer to be of type OpTypePointer"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotIntPointer) {
  const std::string body = R"(
OpAtomicFlagClear %f32_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagClear: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagClearNotInt32Pointer) {
  const std::string body = R"(
OpAtomicFlagClear %u64_var %scope %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagClear: "
                "expected Pointer to point to a value of 32-bit int type"));
}

TEST_F(ValidateAtomics, AtomicFlagClearWrongScopeType) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %u64_1 %memory_semantics
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("AtomicFlagClear: expected Scope to be 32-bit int"));
}

TEST_F(ValidateAtomics, AtomicFlagClearWrongMemorySemanticsType) {
  const std::string body = R"(
OpAtomicFlagClear %u32_var %scope %u64_1
)";

  CompileSuccessfully(GenerateKernelCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("AtomicFlagClear: expected Memory Semantics to be 32-bit int"));
}

}  // anonymous namespace
