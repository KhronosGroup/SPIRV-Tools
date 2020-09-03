// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_add_loop_to_create_int_constant_synonym.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddLoopToCreateIntConstantSynonymTest,
     ConstantsNotSuitable) {
  std::string shader = R"(
               OpCapability Shader
               OpCapability Int64
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpConstant %5 -1
          %7 = OpConstant %5 0
          %8 = OpConstant %5 1
          %9 = OpConstant %5 2
         %10 = OpConstant %5 5
         %11 = OpConstant %5 10
         %12 = OpConstant %5 20
         %13 = OpConstant %5 33
         %14 = OpTypeVector %5 2
         %15 = OpConstantComposite %14 %10 %11
         %16 = OpConstantComposite %14 %12 %12
         %17 = OpTypeVector %5 3
         %18 = OpConstantComposite %17 %11 %7 %11
         %19 = OpTypeInt 64 1
         %20 = OpConstant %19 0
         %21 = OpConstant %19 10
         %22 = OpTypeVector %19 2
         %23 = OpConstantComposite %22 %21 %20
         %24 = OpTypeFloat 32
         %25 = OpConstant %24 0
         %26 = OpConstant %24 5
         %27 = OpConstant %24 10
         %28 = OpConstant %24 20
         %29 = OpTypeVector %24 3
         %30 = OpConstantComposite %29 %26 %27 %26
         %31 = OpConstantComposite %29 %28 %28 %28
         %32 = OpConstantComposite %29 %27 %25 %27
          %2 = OpFunction %3 None %4
         %33 = OpLabel
         %34 = OpCopyObject %5 %11
               OpBranch %35
         %35 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Reminder: the first four parameters of the constructor are the constants
  // with values for C, I, S, N respectively.

  // %70 does not correspond to an id in the module.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   70, 12, 10, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %35 is not a constant.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   35, 12, 10, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %27, %28 and %26 are not integer constants, but scalar floats.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   27, 28, 26, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %32, %31 and %30 are not integer constants, but vector floats.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   32, 31, 30, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %18=(10, 0, 10) has 3 components, while %16=(20, 20) and %15=(5, 10)
  // have 2.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   18, 16, 15, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %21 has bit width 64, while the width of %12 and %10 is 32.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   21, 12, 10, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %13 has component width 64, while the component width of %16 and %15 is 32.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   13, 16, 15, 9, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %21 (N) is a 64-bit integer, not 32-bit.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   7, 7, 7, 21, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %7 (N) has value 0, so N <= 0.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   7, 7, 7, 7, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %6 (N) has value -1, so N <= 1.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   7, 7, 7, 6, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // %13 (N) has value 33, so N > 32.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   7, 7, 7, 6, 13, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // C(%11)=10, I(%12)=20, S(%10)=5, N(%8)=1, so C=I-S*N does not hold, as
  // 20-5*1=15.
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   11, 12, 10, 8, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));

  // C(%15)=(5, 10), I(%16)=(20, 20), S(%15)=(5, 10), N(%8)=1, so C=I-S*N does
  // not hold, as (20, 20)-1*(5, 10) = (15, 10).
  ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                   15, 16, 15, 8, 35, 100, 101, 102, 103, 104, 105, 106, 107)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddLoopToCreateIntConstantSynonymTest, MissingConstants) {
  {
    // The shader is missing a 32-bit integer 0 constant.
    std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpConstant %5 1
          %7 = OpConstant %5 2
          %8 = OpConstant %5 5
          %9 = OpConstant %5 10
         %10 = OpConstant %5 20
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

    const auto env = SPV_ENV_UNIVERSAL_1_5;
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, shader, kFuzzAssembleOption);
    ASSERT_TRUE(IsValid(env, context.get()));

    FactManager fact_manager;
    spvtools::ValidatorOptions validator_options;
    TransformationContext transformation_context(&fact_manager,
                                                 validator_options);

    ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                     9, 10, 8, 7, 12, 100, 101, 102, 103, 104, 105, 106, 107)
                     .IsApplicable(context.get(), transformation_context));
  }
  {
    // The shader is missing a 32-bit integer 1 constant.
    std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpConstant %5 0
          %7 = OpConstant %5 2
          %8 = OpConstant %5 5
          %9 = OpConstant %5 10
         %10 = OpConstant %5 20
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpBranch %12
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
)";

    const auto env = SPV_ENV_UNIVERSAL_1_5;
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, shader, kFuzzAssembleOption);
    ASSERT_TRUE(IsValid(env, context.get()));

    FactManager fact_manager;
    spvtools::ValidatorOptions validator_options;
    TransformationContext transformation_context(&fact_manager,
                                                 validator_options);

    ASSERT_FALSE(TransformationAddLoopToCreateIntConstantSynonym(
                     9, 10, 8, 7, 12, 100, 101, 102, 103, 104, 105, 106, 107)
                     .IsApplicable(context.get(), transformation_context));
  }
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
