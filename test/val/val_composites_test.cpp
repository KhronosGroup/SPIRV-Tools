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

using ValidateComposites = spvtest::ValidateBase<bool>;

std::string GenerateShaderCode(
    const std::string& body,
    const std::string& capabilities_and_extensions = "",
    const std::string& execution_model = "Fragment") {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability Float64
)";

  ss << capabilities_and_extensions;
  ss << "OpMemoryModel Logical GLSL450\n";
  ss << "OpEntryPoint " << execution_model << " %main \"main\"\n";

  ss << R"(
%void = OpTypeVoid
%func = OpTypeFunction %void
%bool = OpTypeBool
%f32 = OpTypeFloat 32
%f64 = OpTypeFloat 64
%u32 = OpTypeInt 32 0
%s32 = OpTypeInt 32 1
%f32vec2 = OpTypeVector %f32 2
%f32vec3 = OpTypeVector %f32 3
%f32vec4 = OpTypeVector %f32 4
%f64vec2 = OpTypeVector %f64 2
%u32vec2 = OpTypeVector %u32 2
%u32vec4 = OpTypeVector %u32 4
%f64mat22 = OpTypeMatrix %f64vec2 2
%f32mat22 = OpTypeMatrix %f32vec2 2
%f32mat23 = OpTypeMatrix %f32vec2 3
%f32mat32 = OpTypeMatrix %f32vec3 2

%f32_0 = OpConstant %f32 0
%f32_1 = OpConstant %f32 1
%f32_2 = OpConstant %f32 2
%f32_3 = OpConstant %f32 3
%f32vec2_12 = OpConstantComposite %f32vec2 %f32_1 %f32_2
%f32vec4_0123 = OpConstantComposite %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3

%u32_0 = OpConstant %u32 0
%u32_1 = OpConstant %u32 1
%u32_2 = OpConstant %u32 2
%u32_3 = OpConstant %u32 3

%u32vec2_01 = OpConstantComposite %u32vec2 %u32_0 %u32_1
%u32vec4_0123 = OpConstantComposite %u32vec4 %u32_0 %u32_1 %u32_2 %u32_3

%f32mat22_1212 = OpConstantComposite %f32mat22 %f32vec2_12 %f32vec2_12
%f32mat23_121212 = OpConstantComposite %f32mat23 %f32vec2_12 %f32vec2_12 %f32vec2_12

%f32vec2arr3 = OpTypeArray %f32vec2 %u32_3

%f32u32struct = OpTypeStruct %f32 %u32

%main = OpFunction %void None %func
%main_entry = OpLabel
)";

  ss << body;

  ss << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

TEST_F(ValidateComposites, VectorExtractDynamicSuccess) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongResultType) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32vec4 %f32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorExtractDynamic: "
                        "expected Result Type to be a scalar type"));
}

TEST_F(ValidateComposites, VectorExtractDynamicNotVector) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32mat22_1212 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorExtractDynamic: "
                        "expected Vector type to be OpTypeVector"));
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongVectorComponent) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %u32vec4_0123 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("VectorExtractDynamic: "
                "expected Vector component type to be equal to Result Type"));
}

TEST_F(ValidateComposites, VectorExtractDynamicWrongIndexType) {
  const std::string body = R"(
%val1 = OpVectorExtractDynamic %f32 %f32vec4_0123 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorExtractDynamic: "
                        "expected Index to be int scalar"));
}

TEST_F(ValidateComposites, VectorInsertDynamicSuccess) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongResultType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32 %f32vec4_0123 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorInsertDynamic: "
                        "expected Result Type to be OpTypeVector"));
}

TEST_F(ValidateComposites, VectorInsertDynamicNotVector) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32mat22_1212 %f32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorInsertDynamic: "
                        "expected Vector type to be equal to Result Type"));
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongComponentType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %u32_1 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorInsertDynamic: "
                        "expected Component type to be equal to Result Type "
                        "component type"));
}

TEST_F(ValidateComposites, VectorInsertDynamicWrongIndexType) {
  const std::string body = R"(
%val1 = OpVectorInsertDynamic %f32vec4 %f32vec4_0123 %f32_1 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("VectorInsertDynamic: "
                        "expected Index to be int scalar"));
}

TEST_F(ValidateComposites, CompositeConstructNotComposite) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("CompositeConstruct: "
                        "expected Result Type to be a composite type"));
}

TEST_F(ValidateComposites, CompositeConstructVectorSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32vec2_12
%val2 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32_0 %f32_0
%val3 = OpCompositeConstruct %f32vec4 %f32_0 %f32_0 %f32vec2_12
%val4 = OpCompositeConstruct %f32vec4 %f32_0 %f32_1 %f32_2 %f32_3
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructVectorOnlyOneConstituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("CompositeConstruct: "
                        "expected number of constituents to be at least 2"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected Constituents to be scalars or vectors of the same "
                "type as Result Type components"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected Constituents to be scalars or vectors of the same "
                "type as Result Type components"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent3) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %u32_0 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected Constituents to be scalars or vectors of the same "
                "type as Result Type components"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongComponentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of given components to be equal to the "
                "size of Result Type vector"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongComponentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec4 %f32vec2_12 %f32vec2_12 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of given components to be equal to the "
                "size of Result Type vector"));
}

TEST_F(ValidateComposites, CompositeConstructMatrixSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %f32vec2_12
%val2 = OpCompositeConstruct %f32mat23 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of columns of Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of columns of Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructVectorWrongConsituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32mat22 %f32vec2_12 %u32vec2_01
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected Constituent type to be equal to the column type "
                "Result Type matrix"));
}

TEST_F(ValidateComposites, CompositeConstructArraySuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of elements of Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %f32vec2_12 %f32vec2_12 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of elements of Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructArrayWrongConsituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32vec2arr3 %f32vec2_12 %u32vec2_01 %f32vec2_12
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected Constituent type to be equal to the column type "
                "Result Type array"));
}

TEST_F(ValidateComposites, CompositeConstructStructSuccess) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituentNumber1) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of members of Result Type struct"));
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituentNumber2) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %u32_1 %u32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CompositeConstruct: "
                "expected total number of Constituents to be equal to the "
                "number of members of Result Type struct"));
}

TEST_F(ValidateComposites, CompositeConstructStructWrongConstituent) {
  const std::string body = R"(
%val1 = OpCompositeConstruct %f32u32struct %f32_0 %f32_1
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("CompositeConstruct: "
                        "expected Constituent type to be equal to the "
                        "corresponding member type of Result Type struct"));
}

TEST_F(ValidateComposites, CopyObjectSuccess) {
  const std::string body = R"(
%val1 = OpCopyObject %f32 %f32_0
%val2 = OpCopyObject %f32vec4 %f32vec4_0123
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// TODO(atgoo@github.com) Reenable this after this check passes Vulkan CTS.
// A change to Vulkan CTS has been sent for review.
TEST_F(ValidateComposites, DISABLED_CopyObjectResultTypeNotType) {
  const std::string body = R"(
%val1 = OpCopyObject %f32_0 %f32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("CopyObject: expected Result Type to be a type"));
}

// TODO(atgoo@github.com) Reenable this after this check passes Vulkan CTS.
// A change to Vulkan CTS has been sent for review.
TEST_F(ValidateComposites, DISABLED_CopyObjectWrongOperandType) {
  const std::string body = R"(
%val1 = OpCopyObject %f32 %u32_0
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("CopyObject: "
                "expected Result Type and Operand type to be the same"));
}

TEST_F(ValidateComposites, TransposeSuccess) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat32 %f32mat23_121212
%val2 = OpTranspose %f32mat22 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateComposites, TransposeResultTypeNotMatrix) {
  const std::string body = R"(
%val1 = OpTranspose %f32vec4 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(getDiagnosticString(),
              HasSubstr("Transpose: expected Result Type to be a matrix type"));
}

TEST_F(ValidateComposites, TransposeDifferentComponentTypes) {
  const std::string body = R"(
%val1 = OpTranspose %f64mat22 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Transpose: "
                "expected component types of Matrix and Result Type to be "
                "identical"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions1) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat23 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Transpose: expected number of columns and the column size "
                "of Matrix to be the reverse of those of Result Type"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions2) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat32 %f32mat22_1212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Transpose: expected number of columns and the column size "
                "of Matrix to be the reverse of those of Result Type"));
}

TEST_F(ValidateComposites, TransposeIncompatibleDimensions3) {
  const std::string body = R"(
%val1 = OpTranspose %f32mat23 %f32mat23_121212
)";

  CompileSuccessfully(GenerateShaderCode(body).c_str());
  ASSERT_EQ(SPV_ERROR_INVALID_DATA, ValidateInstructions());
  EXPECT_THAT(
      getDiagnosticString(),
      HasSubstr("Transpose: expected number of columns and the column size "
                "of Matrix to be the reverse of those of Result Type"));
}

}  // anonymous namespace
