// Copyright (c) 2021 Google, LLC
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

#include "source/fuzz/transformation_swap_two_functions.h"

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"


namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationSwapTwoFunctionsTest, SimpleTest) {


// layout(location=0) in float value;
// void main() {
// 
// }
//
// float multiplyBy2(in float value) {
//   return value*2.0;
// }
//
// float multiplyBy4(in float value) {
//   return multiplyBy2(value)*2.0;
// }

// float multiplyBy8(in float value) {
//   return multiplyBy2(value)*multiplyBy4(value);
// }
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %8
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %8 "value"
               OpDecorate %8 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Input %6
          %8 = OpVariable %7 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd

  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3; 
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Function 1 should not swap with itself
  auto same_func_swap = TransformationSwapTwoFunctions(1, 1);
  ASSERT_FALSE(
     same_func_swap.IsApplicable(context.get(), transformation_context)); 

  // Function 1 is reachable, Function with id 5 is not reachable (not in range)
  auto swap_1_and_5 = TransformationSwapTwoFunctions(1, 5); 
  ASSERT_FALSE(
    swap_1_and_5.IsApplicable(context.get(), transformation_context));

  // Function 5 is not reachable, function 2 is.
  auto swap_5_and_2 = TransformationSwapTwoFunctions(5,2)
  ASSERT_FALSE(
    swap_5_and_2.IsApplicable(context.get(), transformation_context));  

  // Both function 5 and 6 are not reachable. 
  auto swap_5_and_6 = TransformationSwapTwoFunctions(5,6); 
  ASSERT_FALSE(
    swap_5_and_6.IsApplicable(context.get(), transformation_context));   

  //Function 2 and 3 should swap successfully. 
  ASSERT_TRUE(
    TransformationSwapTwoFunctions(2,3).IsApplicable(context.get(), transformation_context));  

  //Function 1 and 4 should swap successfully. 
   ASSERT_TRUE(                                                                                                       TransformationSwapTwoFunctions(1,4).IsApplicable(context.get(), transformation_context));

  std::string after_transformation = R"(



  )";
  // Final check to make sure the serious transformation above is correct. 
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

} // namespace 
} // namespace fuzz
} // namespace spvtools
