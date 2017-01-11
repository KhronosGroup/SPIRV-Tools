// Copyright (c) 2016 Google Inc.
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

// Common validation fixtures for unit tests

#include "gmock/gmock.h"
#include "unit_spirv.h"
#include "val_fixtures.h"

namespace {

using std::string;
using ::testing::HasSubstr;

using ValidateDecorations = spvtest::ValidateBase<bool>;

TEST_F(ValidateDecorations, ValidateOpDecorateRegistration) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpDecorate %1 ArrayStride 4
    OpDecorate %1 Uniform
    %2 = OpTypeFloat 32
    %1 = OpTypeRuntimeArray %2
    ; Since %1 is used first in Decoration, it gets id 1.
)";
  const uint32_t id = 1;
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  // Must have 2 decorations.
  EXPECT_EQ(size_t(2), vstate_->id_decorations(id).size());
  // First decoration.
  EXPECT_EQ(SpvDecorationArrayStride,
            vstate_->id_decorations(id)[0].dec_type());
  EXPECT_EQ(size_t(1), vstate_->id_decorations(id)[0].params().size());
  EXPECT_EQ(unsigned(4), vstate_->id_decorations(id)[0].params()[0]);
  // Second decoration.
  EXPECT_EQ(SpvDecorationUniform, vstate_->id_decorations(id)[1].dec_type());
  EXPECT_EQ(size_t(0), vstate_->id_decorations(id)[1].params().size());
}

TEST_F(ValidateDecorations, ValidateOpMemberDecorateRegistration) {
  string spirv = R"(
    OpCapability Shader
    OpCapability Linkage
    OpMemoryModel Logical GLSL450
    OpDecorate %_arr_double_uint_6 ArrayStride 4
    OpMemberDecorate %_struct_115 2 NonReadable
    OpMemberDecorate %_struct_115 2 Offset 2
    OpDecorate %_struct_115 BufferBlock
    %float = OpTypeFloat 32
    %uint = OpTypeInt 32 0
    %uint_6 = OpConstant %uint 6
    %_arr_double_uint_6 = OpTypeArray %float %uint_6
    %_struct_115 = OpTypeStruct %float %float %_arr_double_uint_6
)";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());

  // The array must have 1 decoration.
  const uint32_t arr_id = 1;
  EXPECT_EQ(size_t(1), vstate_->id_decorations(arr_id).size());
  EXPECT_EQ(SpvDecorationArrayStride,
            vstate_->id_decorations(arr_id)[0].dec_type());
  EXPECT_EQ(-1, vstate_->id_decorations(arr_id)[0].struct_member_index());
  EXPECT_EQ(size_t(1), vstate_->id_decorations(arr_id)[0].params().size());
  EXPECT_EQ(unsigned(4), vstate_->id_decorations(arr_id)[0].params()[0]);

  // The struct must have 3 decorations.
  const uint32_t struct_id = 2;
  EXPECT_EQ(size_t(3), vstate_->id_decorations(struct_id).size());
  // First decoration:
  EXPECT_EQ(SpvDecorationNonReadable,
            vstate_->id_decorations(struct_id)[0].dec_type());
  EXPECT_EQ(2, vstate_->id_decorations(struct_id)[0].struct_member_index());
  EXPECT_EQ(size_t(0), vstate_->id_decorations(struct_id)[0].params().size());
  // Second decoration:
  EXPECT_EQ(SpvDecorationOffset,
            vstate_->id_decorations(struct_id)[1].dec_type());
  EXPECT_EQ(2, vstate_->id_decorations(struct_id)[1].struct_member_index());
  EXPECT_EQ(size_t(1), vstate_->id_decorations(struct_id)[1].params().size());
  EXPECT_EQ(unsigned(2), vstate_->id_decorations(struct_id)[1].params()[0]);
  // Third decoration:
  EXPECT_EQ(SpvDecorationBufferBlock,
            vstate_->id_decorations(struct_id)[2].dec_type());
  EXPECT_EQ(-1, vstate_->id_decorations(struct_id)[2].struct_member_index());
  EXPECT_EQ(size_t(0), vstate_->id_decorations(struct_id)[2].params().size());
}

TEST_F(ValidateDecorations, ValidateGroupDecorateRegistration) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %1 DescriptorSet 0
               OpDecorate %1 NonWritable
               OpDecorate %1 Restrict
          %1 = OpDecorationGroup
               OpGroupDecorate %1 %2 %3
               OpGroupDecorate %1 %4
  %float = OpTypeFloat 32
%_runtimearr_float = OpTypeRuntimeArray %float
  %_struct_9 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_9 = OpTypePointer Uniform %_struct_9
         %2 = OpVariable %_ptr_Uniform__struct_9 Uniform
 %_struct_10 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_10 = OpTypePointer Uniform %_struct_10
         %3 = OpVariable %_ptr_Uniform__struct_10 Uniform
 %_struct_11 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__struct_11 = OpTypePointer Uniform %_struct_11
         %4 = OpVariable %_ptr_Uniform__struct_11 Uniform
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());

  // Decoration group has 3 decorations.
  auto check_decorations_for_id = [this](const uint32_t obj_id) {
    EXPECT_EQ(size_t(3), vstate_->id_decorations(obj_id).size());
    // First decoration:
    EXPECT_EQ(SpvDecorationDescriptorSet,
              vstate_->id_decorations(obj_id)[0].dec_type());
    EXPECT_EQ(-1, vstate_->id_decorations(obj_id)[0].struct_member_index());
    EXPECT_EQ(size_t(1), vstate_->id_decorations(obj_id)[0].params().size());
    EXPECT_EQ(unsigned(0), vstate_->id_decorations(obj_id)[0].params()[0]);
    // Second decoration:
    EXPECT_EQ(SpvDecorationNonWritable,
              vstate_->id_decorations(obj_id)[1].dec_type());
    EXPECT_EQ(-1, vstate_->id_decorations(obj_id)[1].struct_member_index());
    EXPECT_EQ(size_t(0), vstate_->id_decorations(obj_id)[1].params().size());
    // Third decoration:
    EXPECT_EQ(SpvDecorationRestrict,
              vstate_->id_decorations(obj_id)[2].dec_type());
    EXPECT_EQ(-1, vstate_->id_decorations(obj_id)[2].struct_member_index());
    EXPECT_EQ(size_t(0), vstate_->id_decorations(obj_id)[2].params().size());
  };

  // Decoration group is applied to id 1, 2, 3, and 4. Note that id 1 (which is
  // the decoration group id) also has all the decorations.
  check_decorations_for_id(1);
  check_decorations_for_id(2);
  check_decorations_for_id(3);
  check_decorations_for_id(4);
}

TEST_F(ValidateDecorations, ValidateGroupMemberDecorateRegistration) {
  string spirv = R"(
               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450
               OpDecorate %1 Offset 3
          %1 = OpDecorationGroup
               OpGroupMemberDecorate %1 %_struct_1 3 %_struct_2 3 %_struct_3 3
      %float = OpTypeFloat 32
%_runtimearr = OpTypeRuntimeArray %float
  %_struct_1 = OpTypeStruct %float %float %float %_runtimearr
  %_struct_2 = OpTypeStruct %float %float %float %_runtimearr
  %_struct_3 = OpTypeStruct %float %float %float %_runtimearr
  )";
  CompileSuccessfully(spirv);
  EXPECT_EQ(SPV_SUCCESS, ValidateAndRetrieveValidationState());
  // Decoration group has 1 decoration.
  auto check_decorations_for_id = [this](const uint32_t obj_id) {
    EXPECT_EQ(size_t(1), vstate_->id_decorations(obj_id).size());
    EXPECT_EQ(SpvDecorationOffset,
              vstate_->id_decorations(obj_id)[0].dec_type());
    EXPECT_EQ(3, vstate_->id_decorations(obj_id)[0].struct_member_index());
    EXPECT_EQ(size_t(1), vstate_->id_decorations(obj_id)[0].params().size());
    EXPECT_EQ(unsigned(3), vstate_->id_decorations(obj_id)[0].params()[0]);
  };

  // Decoration group is applied to id 1, 2, 3, and 4. Note that id 1 (which is
  // the decoration group id) also has all the decorations.
  check_decorations_for_id(2);
  check_decorations_for_id(3);
  check_decorations_for_id(4);
}

}  // anonymous namespace

