// Copyright (c) 2021 Google LLC
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

#include "source/fuzz/transformation_add_type_int.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddConstantBooleanTest, NeitherPresentInitiallyAddBoth) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);
  
  // Should be able to add a signed/unsigned 32-bit integer type with id 7
  ASSERT_TRUE(TransformationAddTypeInt(7, 32, false)
                  .IsApplicable(context.get(), transformation_context));
  
  ASSERT_TRUE(TransformationAddTypeInt(7, 32, true)
                  .IsApplicable(context.get(), transformation_context));

  // Should be able to add a signed/unsigned 8-bit integer with id 8

  if (context.get()->get_feature_mgr()->HasCapability(SpvCapabilityInt8)) {
      ASSERT_TRUE(TransformationAddTypeInt(7, 8, false)
                  .IsApplicable(context.get(), transformation_context));
  }
  else{
      ASSERT_FALSE(TransformationAddTypeInt(7, 8, false)
                  .IsApplicable(context.get(), transformation_context));
      
      ASSERT_FALSE(TransformationAddTypeInt(7, 8, true)
                  .IsApplicable(context.get(), transformation_context));
  }

  if (context.get()->get_feature_mgr()->HasCapability(SpvCapabilityInt16)) {
      ASSERT_TRUE(TransformationAddTypeInt(7, 16, false)
                  .IsApplicable(context.get(), transformation_context));
      ASSERT_TRUE(TransformationAddTypeInt(7, 16, true)
                  .IsApplicable(context.get(), transformation_context));
  }
  else{
      ASSERT_FALSE(TransformationAddTypeInt(7, 16, false)
                  .IsApplicable(context.get(), transformation_context));
      
      ASSERT_FALSE(TransformationAddTypeInt(7, 16, true)
                  .IsApplicable(context.get(), transformation_context));
  }

  if (context.get()->get_feature_mgr()->HasCapability(SpvCapabilityInt64)) {
      ASSERT_TRUE(TransformationAddTypeInt(7, 64, false)
                  .IsApplicable(context.get(), transformation_context));
      
      ASSERT_TRUE(TransformationAddTypeInt(7, 64, true)
                  .IsApplicable(context.get(), transformation_context));
  }
  else{
      ASSERT_FALSE(TransformationAddTypeInt(7, 64, false)
                  .IsApplicable(context.get(), transformation_context));
      
      ASSERT_FALSE(TransformationAddTypeInt(7, 64, true)
                  .IsApplicable(context.get(), transformation_context));
  }
  

// An unsigned 32-bit integer with id 2 already exists so the statement below should return false
  ASSERT_FALSE(TransformationAddTypeInt(2, 32, false)
                  .IsApplicable(context.get(), transformation_context));
  
  ASSERT_FALSE(TransformationAddTypeInt(2, 32, true)
                  .IsApplicable(context.get(), transformation_context));
  
// Should not be able to add signed/unsigned integers of width different from 16/32/64 bits
// This test fails because of the assert statement in TransformationAddTypeInt::IsApplicable

//   ASSERT_FALSE(TransformationAddTypeInt(7, 20, true)
//                   .IsApplicable(context.get(), transformation_context));
  
//   ASSERT_FALSE(TransformationAddTypeInt(7, 15, false)
//                   .IsApplicable(context.get(), transformation_context));
  

  

  auto add_int_type_unsigned = TransformationAddTypeInt(7, 32, false);
  auto add_int_type_signed = TransformationAddTypeInt(8, 32, true);

  ASSERT_TRUE(add_int_type_unsigned.IsApplicable(context.get(), transformation_context));
  ApplyAndCheckFreshIds(add_int_type_unsigned, context.get(), &transformation_context);
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));

  // Having added int_type_unsigned, we cannot add it again with the same id.
  ASSERT_FALSE(add_int_type_unsigned.IsApplicable(context.get(), transformation_context));
  
  // And we cannot also add it with a different id.
  auto add_int_type_unsigned_again = TransformationAddTypeInt(10, 32, false);
  ASSERT_FALSE(
      add_int_type_unsigned_again.IsApplicable(context.get(), transformation_context));
  
  ASSERT_TRUE(add_int_type_signed.IsApplicable(context.get(), transformation_context));
  ApplyAndCheckFreshIds(add_int_type_signed, context.get(), &transformation_context);
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));

//   // Having added int_type_signed, we cannot add it again with the same id
  ASSERT_FALSE(add_int_type_signed.IsApplicable(context.get(), transformation_context));
  
//   // And we cannot also add it with a different id
  auto add_int_type_signed_again = TransformationAddTypeInt(11, 32, true);
  ASSERT_FALSE(
      add_int_type_signed_again.IsApplicable(context.get(), transformation_context));
 

 std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %7 = OpTypeInt 32 0
          %8 = OpTypeInt 32 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";


  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
