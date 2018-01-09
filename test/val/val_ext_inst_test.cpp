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

using ValidateExtInst = spvtest::ValidateBase<bool>;
using ValidateGlslStd450SqrtLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450FMinLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450FClampLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450SAbsLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450UMinLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450UClampLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450SinLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450PowLike = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450Pack = spvtest::ValidateBase<std::string>;
using ValidateGlslStd450Unpack = spvtest::ValidateBase<std::string>;

// Returns number of components in Pack/Unpack extended instructions.
// |ext_inst_name| is expected to be of the format "PackHalf2x16".
// Number of components is assumed to be single-digit.
uint32_t GetPackedNumComponents(const std::string& ext_inst_name) {
  const size_t x_index = ext_inst_name.find_last_of('x');
  const std::string num_components_str =
      ext_inst_name.substr(x_index - 1, x_index);
  return uint32_t(std::stoul(num_components_str));
}

// Returns packed bit width in Pack/Unpack extended instructions.
// |ext_inst_name| is expected to be of the format "PackHalf2x16".
uint32_t GetPackedBitWidth(const std::string& ext_inst_name) {
  const size_t x_index = ext_inst_name.find_last_of('x');
  const std::string packed_bit_width_str = ext_inst_name.substr(x_index + 1);
  return uint32_t(std::stoul(packed_bit_width_str));
}

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "Fragment") {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability Float16
OpCapability Float64
OpCapability Int16
OpCapability Int64
)";

  ss << capabilities_and_extensions;
  ss << "%extinst = OpExtInstImport \"GLSL.std.450\"\n";
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f16 = OpTypeFloat 16
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%u64 = OpTypeInt 64 0
%s64 = OpTypeInt 64 1
%u16 = OpTypeInt 16 0
%s16 = OpTypeInt 16 1
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%f64vec2 = OpTypeVector %f64 2
%f64vec3 = OpTypeVector %f64 3
%f64vec4 = OpTypeVector %f64 4
%u32vec2 = OpTypeVector %u32 2
%u32vec3 = OpTypeVector %u32 3
%s32vec2 = OpTypeVector %s32 2
%u32vec4 = OpTypeVector %u32 4
%s32vec4 = OpTypeVector %s32 4
%u64vec2 = OpTypeVector %u64 2
%s64vec2 = OpTypeVector %s64 2
%f64mat22 = OpTypeMatrix %f64vec2 2
%f32mat22 = OpTypeMatrix %f32vec2 2
%f32mat23 = OpTypeMatrix %f32vec2 3
%f32mat32 = OpTypeMatrix %f32vec3 2
%f32mat33 = OpTypeMatrix %f32vec3 3

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32_4 = OpConstant %f32 4
%f32_h = OpConstant %f32 0.5
%f32vec2_01 = OpConstantComposite %f32vec2 %f32_0 %f32_1
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec3_012 = OpConstantComposite %f32vec3 %f32_0 %f32_1 %f32_2
%f32vec3_123 = OpConstantComposite %f32vec3 %f32_1 %f32_2 %f32_3
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
%f32vec4_1234 = OpConstantComposite %f32vec4 %f32_1 %f32_2 %f32_3 %f32_4

%f64_0 = OpConstant %f64 0
%f64_1 = OpConstant %f64 1
%f64_2 = OpConstant %f64 2
%f64_3 = OpConstant %f64 3
%f64vec2_01 = OpConstantComposite %f64vec2 %f64_0 %f64_1
%f64vec3_012 = OpConstantComposite %f64vec3 %f64_0 %f64_1 %f64_2
%f64vec4_0123 = OpConstantComposite %f64vec4 %f64_0 %f64_1 %f64_2 %f64_3

%f16_0 = OpConstant %f16 0
%f16_1 = OpConstant %f16 1

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3

%s32_0 = OpConstant %s32 0
%s32_1 = OpConstant %s32 1
%s32_2 = OpConstant %s32 2
%s32_3 = OpConstant %s32 3

%u64_0 = OpConstant %u64 0
%u64_1 = OpConstant %u64 1
%u64_2 = OpConstant %u64 2
%u64_3 = OpConstant %u64 3

%s64_0 = OpConstant %s64 0
%s64_1 = OpConstant %s64 1
%s64_2 = OpConstant %s64 2
%s64_3 = OpConstant %s64 3

%s32vec2_01 = OpConstantComposite %s32vec2 %s32_0 %s32_1
%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1

%s32vec2_12 = OpConstantComposite %s32vec2 %s32_1 %s32_2
%u32vec2_12 = OpConstantComposite %u32vec2 %u32_1 %u32_2

%s32vec4_0123 = OpConstantComposite %s32vec4 %s32_0 %s32_1 %s32_2 %s32_3
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3

%s64vec2_01 = OpConstantComposite %s64vec2 %s64_0 %s64_1
%u64vec2_01 = OpConstantComposite %u64vec2 %u64_0 %u64_1

%f32mat22_1212 = OpConstantComposite %f32mat22 %f32vec2_12 %f32vec2_12
%f32mat23_121212 = OpConstantComposite %f32mat23 %f32vec2_12 %f32vec2_12 %f32vec2_12

%f32_ptr_output = OpTypePointer Output %f32
%f32vec2_ptr_output = OpTypePointer Output %f32vec2

%u32_ptr_output = OpTypePointer Output %u32
%u32vec2_ptr_output = OpTypePointer Output %u32vec2

%u64_ptr_output = OpTypePointer Output %u64

%f32_output = OpVariable %f32_ptr_output Output
%f32vec2_output = OpVariable %f32vec2_ptr_output Output

%u32_output = OpVariable %u32_ptr_output Output
%u32vec2_output = OpVariable %u32vec2_ptr_output Output

%u64_output = OpVariable %u64_ptr_output Output

%f32_ptr_input = OpTypePointer Input %f32
%f32vec2_ptr_input = OpTypePointer Input %f32vec2

%u32_ptr_input = OpTypePointer Input %u32
%u32vec2_ptr_input = OpTypePointer Input %u32vec2

%u64_ptr_input = OpTypePointer Input %u64

%f32_input = OpVariable %f32_ptr_input Input
%f32vec2_input = OpVariable %f32vec2_ptr_input Input

%u32_input = OpVariable %u32_ptr_input Input
%u32vec2_input = OpVariable %u32vec2_ptr_input Input

%u64_input = OpVariable %u64_ptr_input Input

%struct_f32_f32 = OpTypeStruct %f32 %f32
%struct_f32_f32_f32 = OpTypeStruct %f32 %f32 %f32
%struct_f32_u32 = OpTypeStruct %f32 %u32
%struct_f32_u32_f32 = OpTypeStruct %f32 %u32 %f32
%struct_u32_f32 = OpTypeStruct %u32 %f32
%struct_u32_u32 = OpTypeStruct %u32 %u32
%struct_f32_f64 = OpTypeStruct %f32 %f64
%struct_f32vec2_f32vec2 = OpTypeStruct %f32vec2 %f32vec2
%struct_f32vec2_u32vec2 = OpTypeStruct %f32vec2 %u32vec2

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_P(ValidateGlslStd450SqrtLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %f32 %extinst " << ext_inst_name << " %f32_0\n";
  ss << "%val2 = OpExtInst %f32vec2 %extinst " << ext_inst_name
     << " %f32vec2_01\n";
  ss << "%val3 = OpExtInst %f64 %extinst " << ext_inst_name << " %f64_0\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450SqrtLike, IntResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %u32 %extinst " + ext_inst_name + " %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a float scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450SqrtLike, IntOperand) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllSqrtLike, ValidateGlslStd450SqrtLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "Round",
                            "RoundEven",
                            "FAbs",
                            "Trunc",
                            "FSign",
                            "Floor",
                            "Ceil",
                            "Fract",
                            "Sqrt",
                            "InverseSqrt",
                            "Normalize",
                        }), );

TEST_P(ValidateGlslStd450FMinLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %f32 %extinst " << ext_inst_name
     << " %f32_0 %f32_1\n";
  ss << "%val2 = OpExtInst %f32vec2 %extinst " << ext_inst_name
     << " %f32vec2_01 %f32vec2_12\n";
  ss << "%val3 = OpExtInst %f64 %extinst " << ext_inst_name
     << " %f64_0 %f64_0\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450FMinLike, IntResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %u32 %extinst " + ext_inst_name + " %f32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a float scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450FMinLike, IntOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450FMinLike, IntOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %f32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllFMinLike, ValidateGlslStd450FMinLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "FMin",
                            "FMax",
                            "Step",
                            "Reflect",
                            "NMin",
                            "NMax",
                        }), );

TEST_P(ValidateGlslStd450FClampLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %f32 %extinst " << ext_inst_name
     << " %f32_0 %f32_1 %f32_2\n";
  ss << "%val2 = OpExtInst %f32vec2 %extinst " << ext_inst_name
     << " %f32vec2_01 %f32vec2_01 %f32vec2_12\n";
  ss << "%val3 = OpExtInst %f64 %extinst " << ext_inst_name
     << " %f64_0 %f64_0 %f64_1\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450FClampLike, IntResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %u32 %extinst " + ext_inst_name +
                           " %f32_0 %f32_1 %f32_2\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a float scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450FClampLike, IntOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %f32 %extinst " + ext_inst_name +
                           " %u32_0 %f32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450FClampLike, IntOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %f32 %extinst " + ext_inst_name +
                           " %f32_0 %u32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450FClampLike, IntOperand3) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %f32 %extinst " + ext_inst_name +
                           " %f32_1 %f32_0 %u32_2\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllFClampLike, ValidateGlslStd450FClampLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "FClamp",
                            "FMix",
                            "SmoothStep",
                            "Fma",
                            "FaceForward",
                            "NClamp",
                        }), );

TEST_P(ValidateGlslStd450SAbsLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %s32 %extinst " << ext_inst_name << " %u32_1\n";
  ss << "%val2 = OpExtInst %s32 %extinst " << ext_inst_name << " %s32_1\n";
  ss << "%val3 = OpExtInst %u32 %extinst " << ext_inst_name << " %u32_1\n";
  ss << "%val4 = OpExtInst %u32 %extinst " << ext_inst_name << " %s32_1\n";
  ss << "%val5 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01\n";
  ss << "%val6 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01\n";
  ss << "%val7 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01\n";
  ss << "%val8 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450SAbsLike, FloatResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be an int scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450SAbsLike, FloatOperand) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s32 %extinst " + ext_inst_name + " %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450SAbsLike, WrongDimOperand) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s32 %extinst " + ext_inst_name + " %s32vec2_01\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450SAbsLike, WrongBitWidthOperand) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s64 %extinst " + ext_inst_name + " %s32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllSAbsLike, ValidateGlslStd450SAbsLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "SAbs",
                            "SSign",
                            "FindILsb",
                            "FindUMsb",
                            "FindSMsb",
                        }), );

TEST_F(ValidateExtInst, FindUMsbNot32Bit) {
  const std::string body = R"(
%val1 = OpExtInst %s64 %extinst FindUMsb %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FindUMsb: this instruction is currently "
                        "limited to 32-bit width components"));
}

TEST_F(ValidateExtInst, FindSMsbNot32Bit) {
  const std::string body = R"(
%val1 = OpExtInst %s64 %extinst FindSMsb %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FindSMsb: this instruction is currently "
                        "limited to 32-bit width components"));
}

TEST_P(ValidateGlslStd450UMinLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %s32 %extinst " << ext_inst_name
     << " %u32_1 %s32_2\n";
  ss << "%val2 = OpExtInst %s32 %extinst " << ext_inst_name
     << " %s32_1 %u32_2\n";
  ss << "%val3 = OpExtInst %u32 %extinst " << ext_inst_name
     << " %u32_1 %s32_2\n";
  ss << "%val4 = OpExtInst %u32 %extinst " << ext_inst_name
     << " %s32_1 %u32_2\n";
  ss << "%val5 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01 %u32vec2_01\n";
  ss << "%val6 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01 %s32vec2_01\n";
  ss << "%val7 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01 %u32vec2_01\n";
  ss << "%val8 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01 %s32vec2_01\n";
  ss << "%val9 = OpExtInst %s64 %extinst " << ext_inst_name
     << " %u64_1 %s64_0\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450UMinLike, FloatResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0 %u32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be an int scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450UMinLike, FloatOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s32 %extinst " + ext_inst_name + " %f32_0 %u32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450UMinLike, FloatOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s32 %extinst " + ext_inst_name + " %u32_0 %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450UMinLike, WrongDimOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %s32vec2_01 %s32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UMinLike, WrongDimOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %s32_0 %s32vec2_01\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UMinLike, WrongBitWidthOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s64 %extinst " + ext_inst_name + " %s32_0 %s64_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UMinLike, WrongBitWidthOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %s64 %extinst " + ext_inst_name + " %s64_0 %s32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllUMinLike, ValidateGlslStd450UMinLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "UMin",
                            "SMin",
                            "UMax",
                            "SMax",
                        }), );

TEST_P(ValidateGlslStd450UClampLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %s32 %extinst " << ext_inst_name
     << " %s32_0 %u32_1 %s32_2\n";
  ss << "%val2 = OpExtInst %s32 %extinst " << ext_inst_name
     << " %u32_0 %s32_1 %u32_2\n";
  ss << "%val3 = OpExtInst %u32 %extinst " << ext_inst_name
     << " %s32_0 %u32_1 %s32_2\n";
  ss << "%val4 = OpExtInst %u32 %extinst " << ext_inst_name
     << " %u32_0 %s32_1 %u32_2\n";
  ss << "%val5 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01 %u32vec2_01 %u32vec2_12\n";
  ss << "%val6 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01 %s32vec2_01 %s32vec2_12\n";
  ss << "%val7 = OpExtInst %u32vec2 %extinst " << ext_inst_name
     << " %s32vec2_01 %u32vec2_01 %u32vec2_12\n";
  ss << "%val8 = OpExtInst %s32vec2 %extinst " << ext_inst_name
     << " %u32vec2_01 %s32vec2_01 %s32vec2_12\n";
  ss << "%val9 = OpExtInst %s64 %extinst " << ext_inst_name
     << " %u64_1 %s64_0 %s64_1\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450UClampLike, FloatResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %f32 %extinst " + ext_inst_name +
                           " %u32_0 %u32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be an int scalar "
                        "or vector type"));
}

TEST_P(ValidateGlslStd450UClampLike, FloatOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %f32_0 %u32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450UClampLike, FloatOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %u32_0 %f32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450UClampLike, FloatOperand3) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %u32_0 %u32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to be int scalars or "
                        "vectors"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongDimOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %s32vec2_01 %s32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongDimOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %s32_0 %s32vec2_01 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongDimOperand3) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s32 %extinst " + ext_inst_name +
                           " %s32_0 %u32_1 %s32vec2_01\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same dimension as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongBitWidthOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s64 %extinst " + ext_inst_name +
                           " %s32_0 %s64_0 %s64_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongBitWidthOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s64 %extinst " + ext_inst_name +
                           " %s64_0 %s32_0 %s64_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450UClampLike, WrongBitWidthOperand3) {
  const std::string ext_inst_name = GetParam();
  const std::string body = "%val1 = OpExtInst %s64 %extinst " + ext_inst_name +
                           " %s64_0 %s64_0 %s32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected all operands to have the same bit width as "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllUClampLike, ValidateGlslStd450UClampLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "UClamp",
                            "SClamp",
                        }), );

TEST_P(ValidateGlslStd450SinLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %f32 %extinst " << ext_inst_name << " %f32_0\n";
  ss << "%val2 = OpExtInst %f32vec2 %extinst " << ext_inst_name
     << " %f32vec2_01\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450SinLike, IntResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %u32 %extinst " + ext_inst_name + " %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a 16 or 32-bit scalar "
                        "or vector float type"));
}

TEST_P(ValidateGlslStd450SinLike, F64ResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f64 %extinst " + ext_inst_name + " %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a 16 or 32-bit scalar "
                        "or vector float type"));
}

TEST_P(ValidateGlslStd450SinLike, IntOperand) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllSinLike, ValidateGlslStd450SinLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "Radians",
                            "Degrees",
                            "Sin",
                            "Cos",
                            "Tan",
                            "Asin",
                            "Acos",
                            "Atan",
                            "Sinh",
                            "Cosh",
                            "Tanh",
                            "Asinh",
                            "Acosh",
                            "Atanh",
                            "Exp",
                            "Exp2",
                            "Log",
                            "Log2",
                        }), );

TEST_P(ValidateGlslStd450PowLike, Success) {
  const std::string ext_inst_name = GetParam();
  std::ostringstream ss;
  ss << "%val1 = OpExtInst %f32 %extinst " << ext_inst_name
     << " %f32_1 %f32_1\n";
  ss << "%val2 = OpExtInst %f32vec2 %extinst " << ext_inst_name
     << " %f32vec2_01 %f32vec2_12\n";
  CompileSuccessfully(GenerateShaderCode(ss.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450PowLike, IntResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %u32 %extinst " + ext_inst_name + " %f32_1 %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a 16 or 32-bit scalar "
                        "or vector float type"));
}

TEST_P(ValidateGlslStd450PowLike, F64ResultType) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f64 %extinst " + ext_inst_name + " %f32_1 %f32_0\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected Result Type to be a 16 or 32-bit scalar "
                        "or vector float type"));
}

TEST_P(ValidateGlslStd450PowLike, IntOperand1) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %u32_0 %f32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

TEST_P(ValidateGlslStd450PowLike, IntOperand2) {
  const std::string ext_inst_name = GetParam();
  const std::string body =
      "%val1 = OpExtInst %f32 %extinst " + ext_inst_name + " %f32_0 %u32_1\n";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 " + ext_inst_name +
                        ": expected types of all operands to be equal to "
                        "Result Type"));
}

INSTANTIATE_TEST_CASE_P(AllPowLike, ValidateGlslStd450PowLike,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "Atan2",
                            "Pow",
                        }), );

TEST_F(ValidateExtInst, GlslStd450DeterminantSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Determinant %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450DeterminantIncompatibleResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst Determinant %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Determinant: "
                        "expected operand X component type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450DeterminantNotMatrix) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Determinant %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Determinant: "
                        "expected operand X to be a square matrix"));
}

TEST_F(ValidateExtInst, GlslStd450DeterminantMatrixNotSquare) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Determinant %f32mat23_121212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Determinant: "
                        "expected operand X to be a square matrix"));
}

TEST_F(ValidateExtInst, GlslStd450MatrixInverseSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32mat22 %extinst MatrixInverse %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450MatrixInverseIncompatibleResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f32mat33 %extinst MatrixInverse %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 MatrixInverse: "
                        "expected operand X type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450MatrixInverseNotMatrix) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst MatrixInverse %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 MatrixInverse: "
                        "expected Result Type to be a square matrix"));
}

TEST_F(ValidateExtInst, GlslStd450MatrixInverseMatrixNotSquare) {
  const std::string body = R"(
%val1 = OpExtInst %f32mat23 %extinst MatrixInverse %f32mat23_121212
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 MatrixInverse: "
                        "expected Result Type to be a square matrix"));
}

TEST_F(ValidateExtInst, GlslStd450ModfSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Modf %f32_h %f32_output
%val2 = OpExtInst %f32vec2 %extinst Modf %f32vec2_01 %f32vec2_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450ModfIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst Modf %f32_h %f32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Modf: "
                        "expected Result Type to be a scalar or vector "
                        "float type"));
}

TEST_F(ValidateExtInst, GlslStd450ModfXNotOfResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Modf %f64_0 %f32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Modf: "
                        "expected operand X type to be equal to Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450ModfINotPointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Modf %f32_h %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Modf: "
                        "expected operand I to be a pointer"));
}

TEST_F(ValidateExtInst, GlslStd450ModfIDataNotOfResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Modf %f32_h %f32vec2_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Modf: "
                        "expected operand I data type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450ModfStructSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_f32 %extinst ModfStruct %f32_h
%val2 = OpExtInst %struct_f32vec2_f32vec2 %extinst ModfStruct %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450ModfStructResultTypeNotStruct) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst ModfStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 ModfStruct: "
                        "expected Result Type to be a struct with two "
                        "identical scalar or vector float type members"));
}

TEST_F(ValidateExtInst, GlslStd450ModfStructResultTypeStructWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_f32_f32 %extinst ModfStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 ModfStruct: "
                        "expected Result Type to be a struct with two "
                        "identical scalar or vector float type members"));
}

TEST_F(ValidateExtInst, GlslStd450ModfStructResultTypeStructWrongFirstMember) {
  const std::string body = R"(
%val1 = OpExtInst %struct_u32_f32 %extinst ModfStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 ModfStruct: "
                        "expected Result Type to be a struct with two "
                        "identical scalar or vector float type members"));
}

TEST_F(ValidateExtInst, GlslStd450ModfStructResultTypeStructMembersNotEqual) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_f64 %extinst ModfStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 ModfStruct: "
                        "expected Result Type to be a struct with two "
                        "identical scalar or vector float type members"));
}

TEST_F(ValidateExtInst, GlslStd450ModfStructXWrongType) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_f32 %extinst ModfStruct %f64_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 ModfStruct: "
                        "expected operand X type to be equal to members of "
                        "Result Type struct"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Frexp %f32_h %u32_output
%val2 = OpExtInst %f32vec2 %extinst Frexp %f32vec2_01 %u32vec2_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450FrexpIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst Frexp %f32_h %u32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Frexp: "
                        "expected Result Type to be a scalar or vector "
                        "float type"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpWrongXType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Frexp %u32_1 %u32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Frexp: "
                        "expected operand X type to be equal to Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpExpNotPointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Frexp %f32_1 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Frexp: "
                        "expected operand Exp to be a pointer"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpExpNotInt32Pointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Frexp %f32_1 %f32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Frexp: "
                        "expected operand Exp data type to be a 32-bit int "
                        "scalar or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpExpWrongComponentNumber) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Frexp %f32vec2_01 %u32_output
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Frexp: "
                        "expected operand Exp data type to have the same "
                        "component number as Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450LdexpSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Ldexp %f32_h %u32_2
%val2 = OpExtInst %f32vec2 %extinst Ldexp %f32vec2_01 %u32vec2_12
%val3 = OpExtInst %f32 %extinst Ldexp %f32_h %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450LdexpIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst Ldexp %f32_h %u32_2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Ldexp: "
                        "expected Result Type to be a scalar or vector "
                        "float type"));
}

TEST_F(ValidateExtInst, GlslStd450LdexpWrongXType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Ldexp %u32_1 %u32_2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Ldexp: "
                        "expected operand X type to be equal to Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450LdexpFloatExp) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Ldexp %f32_1 %f32_2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Ldexp: "
                        "expected operand Exp to be a 32-bit int scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450LdexpExpWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Ldexp %f32vec2_12 %u32_2
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Ldexp: "
                        "expected operand Exp to have the same component "
                        "number as Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_u32 %extinst FrexpStruct %f32_h
%val2 = OpExtInst %struct_f32vec2_u32vec2 %extinst FrexpStruct %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructResultTypeNotStruct) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst FrexpStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FrexpStruct: "
                        "expected Result Type to be a struct with two members, "
                        "first member a float scalar or vector, second member "
                        "a 32-bit int scalar or vector with the same number of "
                        "components as the first member"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructResultTypeStructWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_u32_f32 %extinst FrexpStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FrexpStruct: "
                        "expected Result Type to be a struct with two members, "
                        "first member a float scalar or vector, second member "
                        "a 32-bit int scalar or vector with the same number of "
                        "components as the first member"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructResultTypeStructWrongMember1) {
  const std::string body = R"(
%val1 = OpExtInst %struct_u32_u32 %extinst FrexpStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FrexpStruct: "
                        "expected Result Type to be a struct with two members, "
                        "first member a float scalar or vector, second member "
                        "a 32-bit int scalar or vector with the same number of "
                        "components as the first member"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructResultTypeStructWrongMember2) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_f32 %extinst FrexpStruct %f32_h
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FrexpStruct: "
                        "expected Result Type to be a struct with two members, "
                        "first member a float scalar or vector, second member "
                        "a 32-bit int scalar or vector with the same number of "
                        "components as the first member"));
}

TEST_F(ValidateExtInst, GlslStd450FrexpStructXWrongType) {
  const std::string body = R"(
%val1 = OpExtInst %struct_f32_u32 %extinst FrexpStruct %f64_0
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 FrexpStruct: "
                        "expected operand X type to be equal to the first "
                        "member of Result Type struct"));
}

TEST_P(ValidateGlslStd450Pack, Success) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 2 ? " %f32vec2_01\n" : " %f32vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %u" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;
  body << "%val2 = OpExtInst %s" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;
  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450Pack, Float32ResultType) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 2 ? " %f32vec2_01\n" : " %f32vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %f" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be " << total_bit_width
           << "-bit int scalar type";

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Pack, Int16ResultType) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 2 ? " %f32vec2_01\n" : " %f32vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %u16 %extinst " << ext_inst_name << vec_str;

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be " << total_bit_width
           << "-bit int scalar type";

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Pack, VNotVector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;

  std::ostringstream body;
  body << "%val1 = OpExtInst %u" << total_bit_width << " %extinst "
       << ext_inst_name << " %f32_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand V to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Pack, VNotFloatVector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 2 ? " %u32vec2_01\n" : " %u32vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %u" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand V to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Pack, VNotFloat32Vector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 2 ? " %f64vec2_01\n" : " %f64vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %u" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand V to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Pack, VWrongSizeVector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string vec_str =
      num_components == 4 ? " %f32vec2_01\n" : " %f32vec4_0123\n";

  std::ostringstream body;
  body << "%val1 = OpExtInst %u" << total_bit_width << " %extinst "
       << ext_inst_name << vec_str;

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand V to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

INSTANTIATE_TEST_CASE_P(AllPack, ValidateGlslStd450Pack,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "PackSnorm4x8",
                            "PackUnorm4x8",
                            "PackSnorm2x16",
                            "PackUnorm2x16",
                            "PackHalf2x16",
                        }), );

TEST_F(ValidateExtInst, PackDouble2x32Success) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst PackDouble2x32 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, PackDouble2x32Float32ResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst PackDouble2x32 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected Result Type to "
                        "be 64-bit float scalar type"));
}

TEST_F(ValidateExtInst, PackDouble2x32Int64ResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u64 %extinst PackDouble2x32 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected Result Type to "
                        "be 64-bit float scalar type"));
}

TEST_F(ValidateExtInst, PackDouble2x32VNotVector) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst PackDouble2x32 %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected operand V to be "
                        "a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, PackDouble2x32VNotIntVector) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst PackDouble2x32 %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected operand V to be "
                        "a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, PackDouble2x32VNotInt32Vector) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst PackDouble2x32 %u64vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected operand V to be "
                        "a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, PackDouble2x32VWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst PackDouble2x32 %u32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 PackDouble2x32: expected operand V to be "
                        "a 32-bit int vector of size 2"));
}

TEST_P(ValidateGlslStd450Unpack, Success) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str =
      num_components == 2 ? "%f32vec2" : " %f32vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << total_bit_width << "_1\n";
  body << "%val2 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %s" << total_bit_width << "_1\n";
  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_P(ValidateGlslStd450Unpack, ResultTypeNotVector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str = "%f32";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << total_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Unpack, ResultTypeNotFloatVector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str =
      num_components == 2 ? "%u32vec2" : " %u32vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << total_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Unpack, ResultTypeNotFloat32Vector) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str =
      num_components == 2 ? "%f64vec2" : " %f64vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << total_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Unpack, ResultTypeWrongSize) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str =
      num_components == 4 ? "%f32vec2" : " %f32vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << total_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected Result Type to be a 32-bit float vector of size "
           << num_components;

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Unpack, ResultPNotInt) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const std::string result_type_str =
      num_components == 2 ? "%f32vec2" : " %f32vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %f" << total_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand P to be a " << total_bit_width
           << "-bit int scalar";

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

TEST_P(ValidateGlslStd450Unpack, ResultPWrongBitWidth) {
  const std::string ext_inst_name = GetParam();
  const uint32_t num_components = GetPackedNumComponents(ext_inst_name);
  const uint32_t packed_bit_width = GetPackedBitWidth(ext_inst_name);
  const uint32_t total_bit_width = num_components * packed_bit_width;
  const uint32_t wrong_bit_width = total_bit_width == 32 ? 64 : 32;
  const std::string result_type_str =
      num_components == 2 ? "%f32vec2" : " %f32vec4";

  std::ostringstream body;
  body << "%val1 = OpExtInst " << result_type_str << " %extinst "
       << ext_inst_name << " %u" << wrong_bit_width << "_1\n";

  std::ostringstream expected;
  expected << "GLSL.std.450 " << ext_inst_name
           << ": expected operand P to be a " << total_bit_width
           << "-bit int scalar";

  CompileSuccessfully(GenerateShaderCode(body.str()));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(), HasSubstr(expected.str()));
}

INSTANTIATE_TEST_CASE_P(AllUnpack, ValidateGlslStd450Unpack,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "UnpackSnorm4x8",
                            "UnpackUnorm4x8",
                            "UnpackSnorm2x16",
                            "UnpackUnorm2x16",
                            "UnpackHalf2x16",
                        }), );

TEST_F(ValidateExtInst, UnpackDouble2x32Success) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec2 %extinst UnpackDouble2x32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, UnpackDouble2x32ResultTypeNotVector) {
  const std::string body = R"(
%val1 = OpExtInst %u64 %extinst UnpackDouble2x32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected Result Type "
                        "to be a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, UnpackDouble2x32ResultTypeNotIntVector) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst UnpackDouble2x32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected Result Type "
                        "to be a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, UnpackDouble2x32ResultTypeNotInt32Vector) {
  const std::string body = R"(
%val1 = OpExtInst %u64vec2 %extinst UnpackDouble2x32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected Result Type "
                        "to be a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, UnpackDouble2x32ResultTypeWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec4 %extinst UnpackDouble2x32 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected Result Type "
                        "to be a 32-bit int vector of size 2"));
}

TEST_F(ValidateExtInst, UnpackDouble2x32VNotFloat) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec2 %extinst UnpackDouble2x32 %u64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected operand V to "
                        "be a 64-bit float scalar"));
}

TEST_F(ValidateExtInst, UnpackDouble2x32VNotFloat64) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec2 %extinst UnpackDouble2x32 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 UnpackDouble2x32: expected operand V to "
                        "be a 64-bit float scalar"));
}

TEST_F(ValidateExtInst, GlslStd450LengthSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Length %f32_1
%val2 = OpExtInst %f32 %extinst Length %f32vec2_01
%val3 = OpExtInst %f32 %extinst Length %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450LengthIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst Length %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Length: "
                        "expected Result Type to be a float scalar type"));
}

TEST_F(ValidateExtInst, GlslStd450LengthIntX) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Length %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Length: "
                        "expected operand X to be of float scalar or "
                        "vector type"));
}

TEST_F(ValidateExtInst, GlslStd450LengthDifferentType) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst Length %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Length: "
                        "expected operand X component type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %f32_0 %f32_1
%val2 = OpExtInst %f32 %extinst Distance %f32vec2_01 %f32vec2_12
%val3 = OpExtInst %f32 %extinst Distance %f32vec4_0123 %f32vec4_1234
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450DistanceIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst Distance %f32vec2_01 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected Result Type to be a float scalar type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceIntP0) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %u32_0 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected operand P0 to be of float scalar or "
                        "vector type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceF64VectorP0) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %f64vec2_01 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected operand P0 component type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceIntP1) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %f32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected operand P1 to be of float scalar or "
                        "vector type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceF64VectorP1) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %f32vec2_12 %f64vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected operand P1 component type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450DistanceDifferentSize) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Distance %f32vec2_01 %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Distance: "
                        "expected operands P0 and P1 to have the same number "
                        "of components"));
}

TEST_F(ValidateExtInst, GlslStd450CrossSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec3 %extinst Cross %f32vec3_012 %f32vec3_123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450CrossIntVectorResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec3 %extinst Cross %f32vec3_012 %f32vec3_123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Cross: "
                        "expected Result Type to be a float vector type"));
}

TEST_F(ValidateExtInst, GlslStd450CrossResultTypeWrongSize) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Cross %f32vec3_012 %f32vec3_123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Cross: "
                        "expected Result Type to have 3 components"));
}

TEST_F(ValidateExtInst, GlslStd450CrossXWrongType) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec3 %extinst Cross %f64vec3_012 %f32vec3_123
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Cross: "
                        "expected operand X type to be equal to Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450CrossYWrongType) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec3 %extinst Cross %f32vec3_123 %f64vec3_012
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Cross: "
                        "expected operand Y type to be equal to Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450RefractSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst Refract %f32_1 %f32_1 %f32_1
%val2 = OpExtInst %f32vec2 %extinst Refract %f32vec2_01 %f32vec2_01 %f16_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450RefractIntVectorResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32vec2 %extinst Refract %f32vec2_01 %f32vec2_01 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Refract: "
                        "expected Result Type to be a float scalar or "
                        "vector type"));
}

TEST_F(ValidateExtInst, GlslStd450RefractIntVectorI) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Refract %u32vec2_01 %f32vec2_01 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Refract: "
                        "expected operand I to be of type equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450RefractIntVectorN) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Refract %f32vec2_01 %u32vec2_01 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Refract: "
                        "expected operand N to be of type equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450RefractIntEta) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Refract %f32vec2_01 %f32vec2_01 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Refract: "
                        "expected operand Eta to be a 16 or 32-bit "
                        "float scalar"));
}

TEST_F(ValidateExtInst, GlslStd450RefractFloat64Eta) {
  const std::string body = R"(
%val1 = OpExtInst %f32vec2 %extinst Refract %f32vec2_01 %f32vec2_01 %f64_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 Refract: "
                        "expected operand Eta to be a 16 or 32-bit "
                        "float scalar"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32_input
%val2 = OpExtInst %f32vec2 %extinst InterpolateAtCentroid %f32vec2_input
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidNoCapability) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32_input
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid requires "
                        "capability InterpolationFunction"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst InterpolateAtCentroid %f32_input
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidF64ResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst InterpolateAtCentroid %f32_input
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidNotPointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid: "
                        "expected Interpolant to be a pointer"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidWrongDataType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32vec2_input
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid: "
                        "expected Interpolant data type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidWrongStorageClass) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32_output
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid: "
                        "expected Interpolant storage class to be Input"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtCentroidWrongExecutionModel) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtCentroid %f32_input
)";

  CompileSuccessfully(GenerateShaderCode(
      body, "OpCapability InterpolationFunction\n", "Vertex"));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtCentroid requires "
                        "Fragment execution model"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_input %u32_1
%val2 = OpExtInst %f32vec2 %extinst InterpolateAtSample %f32vec2_input %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleNoCapability) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_input %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample requires "
                        "capability InterpolationFunction"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst InterpolateAtSample %f32_input %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleF64ResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst InterpolateAtSample %f32_input %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleNotPointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_1 %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Interpolant to be a pointer"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleWrongDataType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32vec2_input %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Interpolant data type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleWrongStorageClass) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_output %u32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Interpolant storage class to be Input"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleFloatSample) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_input %f32_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Sample to be 32-bit integer"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleU64Sample) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_input %u64_1
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample: "
                        "expected Sample to be 32-bit integer"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtSampleWrongExecutionModel) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtSample %f32_input %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(
      body, "OpCapability InterpolationFunction\n", "Vertex"));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtSample requires "
                        "Fragment execution model"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetSuccess) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f32vec2_01
%val2 = OpExtInst %f32vec2 %extinst InterpolateAtOffset %f32vec2_input %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetNoCapability) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body));
  ASSERT_EQ(SPV_ERROR_INVALID_CAPABILITY, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset requires "
                        "capability InterpolationFunction"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetIntResultType) {
  const std::string body = R"(
%val1 = OpExtInst %u32 %extinst InterpolateAtOffset %f32_input %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetF64ResultType) {
  const std::string body = R"(
%val1 = OpExtInst %f64 %extinst InterpolateAtOffset %f32_input %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Result Type to be a 32-bit float scalar "
                        "or vector type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetNotPointer) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_1 %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Interpolant to be a pointer"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetWrongDataType) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32vec2_input %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Interpolant data type to be equal to "
                        "Result Type"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetWrongStorageClass) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_output %f32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Interpolant storage class to be Input"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetOffsetNotVector) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f32_0
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Offset to be a vector of 2 32-bit floats"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetOffsetNotVector2) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f32vec3_012
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Offset to be a vector of 2 32-bit floats"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetOffsetNotFloatVector) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %u32vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Offset to be a vector of 2 32-bit floats"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetOffsetNotFloat32Vector) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f64vec2_01
)";

  CompileSuccessfully(
      GenerateShaderCode(body, "OpCapability InterpolationFunction\n"));
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset: "
                        "expected Offset to be a vector of 2 32-bit floats"));
}

TEST_F(ValidateExtInst, GlslStd450InterpolateAtOffsetWrongExecutionModel) {
  const std::string body = R"(
%val1 = OpExtInst %f32 %extinst InterpolateAtOffset %f32_input %f32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(
      body, "OpCapability InterpolationFunction\n", "Vertex"));
  ASSERT_EQ(SPV_ERROR_INVALID_ID, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("GLSL.std.450 InterpolateAtOffset requires "
                        "Fragment execution model"));
}

}  // anonymous namespace
