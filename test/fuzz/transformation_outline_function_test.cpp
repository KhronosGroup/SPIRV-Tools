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

#include "source/fuzz/transformation_outline_function.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationOutlineFunctionTest, TrivialOutline) {
  // This tests outlining of a single, empty basic block.

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
               OpBranch %6
          %6 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 6, /* not relevant */ 200,
                                               100, 101, 102, 103,
                                               /* not relevant */ 201, {}, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

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
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %103 = OpFunctionCall %2 %101
               OpReturn
               OpFunctionEnd
        %101 = OpFunction %2 None %3
        %102 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest, OutlineInterestingControlFlowNoState) {
  // This tests outlining of some non-trivial control flow, but such that the
  // basic blocks in the control flow do not actually do anything.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeBool
         %21 = OpConstantTrue %20
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %21 %8 %9
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %12 %11 None
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %21 %11 %12
         %11 = OpLabel
               OpBranch %9
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 12, /* not relevant */
                                               200, 100, 101, 102, 103,
                                               /* not relevant */ 201, {}, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeBool
         %21 = OpConstantTrue %20
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %103 = OpFunctionCall %2 %101
               OpReturn
               OpFunctionEnd
        %101 = OpFunction %2 None %3
        %102 = OpLabel
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %21 %8 %9
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %12 %11 None
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %21 %11 %12
         %11 = OpLabel
               OpBranch %9
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest,
     OutlineInterestingControlEndingWithControlFlow) {
  // This tests outlining of some non-trivial control flow that itself ends
  // with some control flow.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeBool
         %21 = OpConstantTrue %20
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %21 %8 %9
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %12 %11 None
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %21 %11 %12
         %11 = OpLabel
               OpBranch %9
         %12 = OpLabel
               OpSelectionMerge %15 None
               OpBranchConditional %21 %13 %14
         %13 = OpLabel
               OpBranch %15
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 12, /* not relevant */
                                               200, 100, 101, 102, 103,
                                               /* not relevant */ 201, {}, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeBool
         %21 = OpConstantTrue %20
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %103 = OpFunctionCall %2 %101
               OpSelectionMerge %15 None
               OpBranchConditional %21 %13 %14
         %13 = OpLabel
               OpBranch %15
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpReturn
               OpFunctionEnd
        %101 = OpFunction %2 None %3
        %102 = OpLabel
               OpBranch %7
          %7 = OpLabel
               OpSelectionMerge %9 None
               OpBranchConditional %21 %8 %9
          %8 = OpLabel
               OpBranch %9
          %9 = OpLabel
               OpLoopMerge %12 %11 None
               OpBranch %10
         %10 = OpLabel
               OpBranchConditional %21 %11 %12
         %11 = OpLabel
               OpBranch %9
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest, OutlineCodeThatGeneratesUnusedIds) {
  // This tests outlining of a single basic block that does some computation,
  // but that does not use nor generate ids required outside of the outlined
  // region.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
          %7 = OpCopyObject %20 %21
          %8 = OpCopyObject %20 %21
          %9 = OpIAdd %20 %7 %8
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 6, /* not relevant */ 200,
                                               100, 101, 102, 103,
                                               /* not relevant */ 201, {}, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %103 = OpFunctionCall %2 %101
               OpReturn
               OpFunctionEnd
        %101 = OpFunction %2 None %3
        %102 = OpLabel
          %7 = OpCopyObject %20 %21
          %8 = OpCopyObject %20 %21
          %9 = OpIAdd %20 %7 %8
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest, OutlineCodeThatGeneratesSingleUsedId) {
  // This tests outlining of a block that generates an id that is used in a
  // later block.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
          %7 = OpCopyObject %20 %21
          %8 = OpCopyObject %20 %21
          %9 = OpIAdd %20 %7 %8
               OpBranch %10
         %10 = OpLabel
         %11 = OpCopyObject %20 %9
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 6, 99, 100, 101, 102, 103,
                                               105, {}, {{9, 104}});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
         %99 = OpTypeStruct %20
        %100 = OpTypeFunction %99
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %103 = OpFunctionCall %99 %101
          %9 = OpCompositeExtract %20 %103 0
               OpBranch %10
         %10 = OpLabel
         %11 = OpCopyObject %20 %9
               OpReturn
               OpFunctionEnd
        %101 = OpFunction %99 None %100
        %102 = OpLabel
          %7 = OpCopyObject %20 %21
          %8 = OpCopyObject %20 %21
        %104 = OpIAdd %20 %7 %8
        %105 = OpCompositeConstruct %99 %104
               OpReturnValue %105
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest, OutlineDiamondThatGeneratesSeveralIds) {
  // This tests outlining of several blocks that generate a number of ids that
  // are used in later blocks.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
         %22 = OpTypeBool
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
          %7 = OpCopyObject %20 %21
          %8 = OpCopyObject %20 %21
          %9 = OpSLessThan %22 %7 %8
               OpSelectionMerge %12 None
               OpBranchConditional %9 %10 %11
         %10 = OpLabel
         %13 = OpIAdd %20 %7 %8
               OpBranch %12
         %11 = OpLabel
         %14 = OpIAdd %20 %7 %7
               OpBranch %12
         %12 = OpLabel
         %15 = OpPhi %20 %13 %10 %14 %11
               OpBranch %16
         %16 = OpLabel
         %17 = OpCopyObject %20 %15
         %18 = OpCopyObject %22 %9
         %19 = OpIAdd %20 %7 %8
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(
      6, 12, 100, 101, 102, 103, 104, 105, {},
      {{15, 106}, {9, 107}, {7, 108}, {8, 109}});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
         %22 = OpTypeBool
          %3 = OpTypeFunction %2
        %100 = OpTypeStruct %20 %20 %22 %20
        %101 = OpTypeFunction %100
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %6
          %6 = OpLabel
        %104 = OpFunctionCall %100 %102
          %7 = OpCompositeExtract %20 %104 0
          %8 = OpCompositeExtract %20 %104 1
          %9 = OpCompositeExtract %22 %104 2
         %15 = OpCompositeExtract %20 %104 3
               OpBranch %16
         %16 = OpLabel
         %17 = OpCopyObject %20 %15
         %18 = OpCopyObject %22 %9
         %19 = OpIAdd %20 %7 %8
               OpReturn
               OpFunctionEnd
        %102 = OpFunction %100 None %101
        %103 = OpLabel
        %108 = OpCopyObject %20 %21
        %109 = OpCopyObject %20 %21
        %107 = OpSLessThan %22 %108 %109
               OpSelectionMerge %12 None
               OpBranchConditional %107 %10 %11
         %10 = OpLabel
         %13 = OpIAdd %20 %108 %109
               OpBranch %12
         %11 = OpLabel
         %14 = OpIAdd %20 %108 %108
               OpBranch %12
         %12 = OpLabel
        %106 = OpPhi %20 %13 %10 %14 %11
        %105 = OpCompositeConstruct %100 %108 %109 %107 %106
               OpReturnValue %105
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

TEST(TransformationOutlineFunctionTest, OutlineCodeThatUsesASingleId) {
  // This tests outlining of a block that uses an id defined earlier.

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %7 = OpCopyObject %20 %21
               OpBranch %6
          %6 = OpLabel
          %8 = OpCopyObject %20 %7
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  TransformationOutlineFunction transformation(6, 6, 100, 101, 102, 103, 104,
                                               105, {{7, 106}}, {});
  ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
  transformation.Apply(context.get(), &fact_manager);
  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
         %20 = OpTypeInt 32 1
         %21 = OpConstant %20 5
          %3 = OpTypeFunction %2
        %101 = OpTypeFunction %2 %20
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %7 = OpCopyObject %20 %21
               OpBranch %6
          %6 = OpLabel
        %104 = OpFunctionCall %2 %102 %7
               OpBranch %10
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
        %102 = OpFunction %2 None %101
        %106 = OpFunctionParameter %20
        %103 = OpLabel
          %8 = OpCopyObject %20 %106
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
