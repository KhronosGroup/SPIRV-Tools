// Copyright (c) 2019 Google LLC.
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

// Validation tests for misc instructions

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

using ValidateMisc = spvtest::ValidateBase<bool>;

TEST_F(ValidateMisc, UndefRestrictedShort) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability StorageBuffer16BitAccess
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
%short = OpTypeInt 16 0
%undef = OpUndef %short
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Cannot create undefined values with 8- or 16-bit types"));
}

TEST_F(ValidateMisc, UndefRestrictedChar) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability StorageBuffer8BitAccess
OpExtension "SPV_KHR_8bit_storage"
OpMemoryModel Logical GLSL450
%char = OpTypeInt 8 0
%undef = OpUndef %char
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Cannot create undefined values with 8- or 16-bit types"));
}

TEST_F(ValidateMisc, UndefRestrictedHalf) {
  const std::string spirv = R"(
OpCapability Shader
OpCapability Linkage
OpCapability StorageBuffer16BitAccess
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
%half = OpTypeFloat 16
%undef = OpUndef %half
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Cannot create undefined values with 8- or 16-bit types"));
}

const std::string ShaderClockSpriv = R"(
OpCapability Shader
OpCapability Int64
OpCapability ShaderClockKHR
OpExtension "SPV_KHR_shader_clock"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 450
OpSourceExtension "GL_ARB_gpu_shader_int64"
OpSourceExtension "GL_ARB_shader_clock"
OpSourceExtension "GL_EXT_shader_realtime_clock"
OpName %main "main"
OpName %time1 "time1"
%void = OpTypeVoid
)";

TEST_F(ValidateMisc, ShaderClockInt64) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
%uint_3 = OpConstant %uint 3
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_uint Function
%11 = OpReadClockKHR %uint %uint_3
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("or 64bit unsigned integer"));
}

TEST_F(ValidateMisc, ShaderClockVec2) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%v2uint = OpTypeVector %ulong 2
%_ptr_Function_v2uint = OpTypePointer Function %v2uint
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_v2uint Function
%15 = OpReadClockKHR %v2uint %uint_3
OpStore %time1 %15
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("vector of two components"));
}

TEST_F(ValidateMisc, ShaderClockInvalidScopeValue) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%uint = OpTypeInt 32 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%uint_10 = OpConstant %uint 10
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_ulong Function
%11 = OpReadClockKHR %ulong %uint_10
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr("Invalid scope value"));
}

TEST_F(ValidateMisc, ShaderClockSubgroupScope) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%uint = OpTypeInt 32 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%subgroup = OpConstant %uint 3
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_ulong Function
%11 = OpReadClockKHR %ulong %subgroup
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMisc, ShaderClockDeviceScope) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%uint = OpTypeInt 32 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%device = OpConstant %uint 1
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_ulong Function
%11 = OpReadClockKHR %ulong %device
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateMisc, ShaderClockWorkgroupScope) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%uint = OpTypeInt 32 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%workgroup = OpConstant %uint 2
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_ulong Function
%11 = OpReadClockKHR %ulong %workgroup
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Scope must be Subgroup or Device"));
}

TEST_F(ValidateMisc, VulkanShaderClockWorkgroupScope) {
  const std::string spirv = ShaderClockSpriv + R"(
%3 = OpTypeFunction %void
%ulong = OpTypeInt 64 0
%uint = OpTypeInt 32 0
%_ptr_Function_ulong = OpTypePointer Function %ulong
%workgroup = OpConstant %uint 2
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%time1 = OpVariable %_ptr_Function_ulong Function
%11 = OpReadClockKHR %ulong %workgroup
OpStore %time1 %11
OpReturn
OpFunctionEnd)";

  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              AnyVUID("VUID-StandaloneSpirv-OpReadClockKHR-04652"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Scope must be Subgroup or Device"));
}

TEST_F(ValidateMisc, UndefVoid) {
  const std::string spirv = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 320
          %2 = OpTypeVoid
         %10 = OpUndef %2
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Cannot create undefined values with void type"));
}

TEST_F(ValidateMisc, VulkanInvalidStorageClass) {
  const std::string spirv = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %func "shader"
%int = OpTypeInt 32 0
%ptr = OpTypePointer CrossWorkgroup %int
%var = OpVariable %ptr CrossWorkgroup
%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv, SPV_ENV_VULKAN_1_0);
  ASSERT_EQ(SPV_ERROR_INVALID_BINARY, ValidateInstructions(SPV_ENV_VULKAN_1_0));
  EXPECT_THAT(getDiagnosticString(),
              AnyVUID("VUID-StandaloneSpirv-None-04643"));
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Invalid storage class for target environment"));
}

struct UndefCase {
  std::string model;
  std::string storage_class;
  bool variable_pointers;
  bool variable_pointers_storage_buffer;
  std::string expected_error;  // empty if validation should pass.
};
std::ostream& operator<<(std::ostream& os, const UndefCase& uc) {
  os << "UndefCase(" << uc.model << " " << uc.storage_class
     << (uc.variable_pointers ? " vp" : "")
     << (uc.variable_pointers_storage_buffer ? " vpsb" : "") << ")";
  return os;
}

using ValidateMiscUndef = spvtest::ValidateBase<UndefCase>;

std::string Preamble(const UndefCase& undef_case) {
  const auto addresses_cap = std::string(
      (undef_case.model == "Physical32" || undef_case.model == "Physical64")
          ? "OpCapability Addresses\n"
          : "");

  const auto storage_buffer_ext =
      std::string((undef_case.storage_class == "StorageBuffer")
                      ? "OpExtension \"SPV_KHR_storage_buffer_storage_class\"\n"
                      : "");

  const auto physical_buffer_cap =
      std::string(undef_case.model == "PhysicalStorageBuffer64"
                      ? "OpCapability PhysicalStorageBufferAddresses\n"
                      : "");
  const auto physical_buffer_ext =
      std::string(undef_case.model == "PhysicalStorageBuffer64"
                      ? "OpExtension \"SPV_KHR_physical_storage_buffer\"\n"
                      : "");

  const auto var_ptr_cap =
      std::string(undef_case.variable_pointers
                      ? "OpCapability VariablePointers\n"
                      : "") +
      std::string(undef_case.variable_pointers_storage_buffer
                      ? "OpCapability VariablePointersStorageBuffer\n"
                      : "");
  const auto var_ptr_ext = std::string(
      !var_ptr_cap.empty() ? "OpExtension \"SPV_KHR_variable_pointers\"\n"
                           : "");

  return addresses_cap + physical_buffer_cap + var_ptr_cap +
         "OpCapability Shader\n" + storage_buffer_ext + physical_buffer_ext +
         var_ptr_ext +

         R"(
OpMemoryModel )" +
         undef_case.model + R"( Simple
)";
}

TEST_P(ValidateMiscUndef, Undef_ModuleScope) {
  const std::string spirv = Preamble(GetParam()) + R"(
OpEntryPoint Vertex %func "shader"
%int = OpTypeInt 32 0
%ptr = OpTypePointer )" + GetParam().storage_class +
                            R"( %int
%undef = OpUndef %ptr

%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  if (GetParam().expected_error.empty()) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
    EXPECT_EQ(getDiagnosticString(), "") << spirv;
  } else {
    EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(), HasSubstr(GetParam().expected_error));
  }
}

TEST_P(ValidateMiscUndef, Undef_FunctionScope) {
  const std::string spirv = Preamble(GetParam()) + R"(
OpEntryPoint Vertex %func "shader"
%int = OpTypeInt 32 0
%ptr = OpTypePointer )" + 
                            GetParam().storage_class +
                            R"( %int

%void   = OpTypeVoid
%void_f = OpTypeFunction %void
%func   = OpFunction %void None %void_f
%label  = OpLabel
%undef = OpUndef %ptr
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(spirv);
  if (GetParam().expected_error.empty()) {
    EXPECT_EQ(SPV_SUCCESS, ValidateInstructions());
    EXPECT_EQ(getDiagnosticString(), "") << spirv;
  } else {
    EXPECT_NE(SPV_SUCCESS, ValidateInstructions());
    EXPECT_THAT(getDiagnosticString(), HasSubstr(GetParam().expected_error));
  }
}

INSTANTIATE_TEST_SUITE_P(
    WithoutVariablePointers, ValidateMiscUndef,
    ::testing::ValuesIn(std::vector<UndefCase>{
        {"Physical32", "Private", false, false, ""},
        {"Physical64", "Private", false, false, ""},
        // PhysicalStorageBuffer64 addressing model
        {"PhysicalStorageBuffer64", "PhysicalStorageBuffer", false, false,
         "Cannot create undefined ponter for PhysicalStorageBuffer storage "
         "class"},
        {"PhysicalStorageBuffer64", "Function", false, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Private", false, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "StorageBuffer", false, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Uniform", false, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "UniformConstant", false, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Workgroup", false, false,
         "Cannot create undefined logical pointer"},
        // Logical addressing model
        {"Logical", "Function", false, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Private", false, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "StorageBuffer", false, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Uniform", false, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "UniformConstant", false, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Workgroup", false, false,
         "Cannot create undefined logical pointer"},
    }));

INSTANTIATE_TEST_SUITE_P(
    VariablePointers, ValidateMiscUndef,
    ::testing::ValuesIn(std::vector<UndefCase>{
        // PhysicalStorageBuffer64 addressing model
        {"PhysicalStorageBuffer64", "PhysicalStorageBuffer", false, false,
         "Cannot create undefined ponter for PhysicalStorageBuffer storage "
         "class"},
        {"PhysicalStorageBuffer64", "Function", true, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Private", true, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "StorageBuffer", true, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Uniform", true, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "UniformConstant", true, false,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Workgroup", true, false,
         "Cannot create undefined logical pointer"},
        // Logical addressing model
        {"Logical", "Function", true, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Private", true, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "StorageBuffer", true, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Uniform", true, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "UniformConstant", true, false,
         "Cannot create undefined logical pointer"},
        {"Logical", "Workgroup", true, false,
         "Cannot create undefined logical pointer"},
    }));

INSTANTIATE_TEST_SUITE_P(
    VariablePointersStorageBuffer, ValidateMiscUndef,
    ::testing::ValuesIn(std::vector<UndefCase>{
        // PhysicalStorageBuffer64 addressing model
        {"PhysicalStorageBuffer64", "PhysicalStorageBuffer", false, false,
         "Cannot create undefined ponter for PhysicalStorageBuffer storage "
         "class"},
        {"PhysicalStorageBuffer64", "Function", false, true,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Private", false, true,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "StorageBuffer", false, true,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Uniform", false, true,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "UniformConstant", false, true,
         "Cannot create undefined logical pointer"},
        {"PhysicalStorageBuffer64", "Workgroup", false, true,
         "Cannot create undefined logical pointer"},
        // Logical addressing model
        {"Logical", "Function", false, true,
         "Cannot create undefined logical pointer"},
        {"Logical", "Private", false, true,
         "Cannot create undefined logical pointer"},
        {"Logical", "StorageBuffer", false, true,
         "Cannot create undefined logical pointer"},
        {"Logical", "Uniform", false, true,
         "Cannot create undefined logical pointer"},
        {"Logical", "UniformConstant", false, true,
         "Cannot create undefined logical pointer"},
        {"Logical", "Workgroup", false, true,
         "Cannot create undefined logical pointer"},
    }));

}  // namespace
}  // namespace val
}  // namespace spvtools
