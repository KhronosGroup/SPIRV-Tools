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

#include "source/fuzz/transformation_add_opphi_synonym.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

protobufs::Fact MakeSynonymFact(uint32_t first, uint32_t second) {
  protobufs::FactDataSynonym data_synonym_fact;
  *data_synonym_fact.mutable_data1() = MakeDataDescriptor(first, {});
  *data_synonym_fact.mutable_data2() = MakeDataDescriptor(second, {});
  protobufs::Fact result;
  *result.mutable_data_synonym_fact() = data_synonym_fact;
  return result;
}

// Adds synonym facts to the fact manager.
void SetUpIdSynonyms(FactManager* fact_manager, opt::IRContext* context) {
  fact_manager->AddFact(MakeSynonymFact(11, 9), context);
  fact_manager->AddFact(MakeSynonymFact(13, 9), context);
  fact_manager->AddFact(MakeSynonymFact(14, 9), context);
  fact_manager->AddFact(MakeSynonymFact(19, 9), context);
  fact_manager->AddFact(MakeSynonymFact(20, 9), context);
  fact_manager->AddFact(MakeSynonymFact(10, 21), context);
}

TEST(TransformationAddOpPhiSynonymTest, Inapplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpTypeInt 32 0
          %9 = OpConstant %7 1
         %10 = OpConstant %7 2
         %11 = OpConstant %8 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %13 = OpCopyObject %7 %9
         %14 = OpCopyObject %8 %11
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %6 %17 %18
         %17 = OpLabel
         %19 = OpCopyObject %7 %13
         %20 = OpCopyObject %8 %14
         %21 = OpCopyObject %7 %10
               OpBranch %16
         %18 = OpLabel
               OpBranch %16
         %16 = OpLabel
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

  SetUpIdSynonyms(&fact_manager, context.get());

  // %13 is not a block label.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(13, {}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // Block %12 does not have a predecessor.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(12, {}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // Not all predecessors of %16 (%17 and %18) are considered in the map.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 19}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %30 does not exist in the module.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{30, 19}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %20 is not a block label.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{20, 19}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %15 is not the id of one of the predecessors of the block.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{15, 19}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %30 does not exist in the module.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 30}, {18, 13}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %19 and %10 are not synonymous.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 19}, {18, 10}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %19 and %14 do not have the same type.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 19}, {18, 14}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %19 is not available at the end of %18.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 9}, {18, 19}}, 100)
                   .IsApplicable(context.get(), transformation_context));

  // %21 is not a fresh id.
  ASSERT_FALSE(TransformationAddOpPhiSynonym(16, {{17, 9}, {18, 9}}, 21)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddOpPhiSynonymTest, Apply) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpTypeInt 32 0
          %9 = OpConstant %7 1
         %10 = OpConstant %7 2
         %11 = OpConstant %8 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %13 = OpCopyObject %7 %9
         %14 = OpCopyObject %8 %11
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %6 %17 %18
         %17 = OpLabel
         %19 = OpCopyObject %7 %13
         %20 = OpCopyObject %8 %14
         %21 = OpCopyObject %7 %10
               OpBranch %16
         %18 = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpBranch %22
         %22 = OpLabel
               OpLoopMerge %23 %24 None
               OpBranchConditional %6 %25 %23
         %25 = OpLabel
               OpSelectionMerge %26 None
               OpBranchConditional %6 %27 %26
         %27 = OpLabel
         %28 = OpCopyObject %7 %13
               OpBranch %23
         %26 = OpLabel
               OpSelectionMerge %29 None
               OpBranchConditional %6 %29 %24
         %29 = OpLabel
         %30 = OpCopyObject %7 %13
               OpBranch %23
         %24 = OpLabel
               OpBranch %22
         %23 = OpLabel
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

  SetUpIdSynonyms(&fact_manager, context.get());

  // Add some further synonym facts.
  fact_manager.AddFact(MakeSynonymFact(28, 9), context.get());
  fact_manager.AddFact(MakeSynonymFact(30, 9), context.get());

  auto transformation1 = TransformationAddOpPhiSynonym(17, {{15, 13}}, 100);
  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));
  transformation1.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(100, {}),
                                        MakeDataDescriptor(9, {})));

  auto transformation2 =
      TransformationAddOpPhiSynonym(16, {{17, 19}, {18, 13}}, 101);
  ASSERT_TRUE(
      transformation2.IsApplicable(context.get(), transformation_context));
  transformation2.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(101, {}),
                                        MakeDataDescriptor(9, {})));

  auto transformation3 =
      TransformationAddOpPhiSynonym(23, {{22, 13}, {27, 28}, {29, 30}}, 102);
  ASSERT_TRUE(
      transformation3.IsApplicable(context.get(), transformation_context));
  transformation3.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(102, {}),
                                        MakeDataDescriptor(9, {})));

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string after_transformations = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpTypeInt 32 1
          %8 = OpTypeInt 32 0
          %9 = OpConstant %7 1
         %10 = OpConstant %7 2
         %11 = OpConstant %8 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %13 = OpCopyObject %7 %9
         %14 = OpCopyObject %8 %11
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %6 %17 %18
         %17 = OpLabel
        %100 = OpPhi %7 %13 %15
         %19 = OpCopyObject %7 %13
         %20 = OpCopyObject %8 %14
         %21 = OpCopyObject %7 %10
               OpBranch %16
         %18 = OpLabel
               OpBranch %16
         %16 = OpLabel
        %101 = OpPhi %7 %19 %17 %13 %18
               OpBranch %22
         %22 = OpLabel
               OpLoopMerge %23 %24 None
               OpBranchConditional %6 %25 %23
         %25 = OpLabel
               OpSelectionMerge %26 None
               OpBranchConditional %6 %27 %26
         %27 = OpLabel
         %28 = OpCopyObject %7 %13
               OpBranch %23
         %26 = OpLabel
               OpSelectionMerge %29 None
               OpBranchConditional %6 %29 %24
         %29 = OpLabel
         %30 = OpCopyObject %7 %13
               OpBranch %23
         %24 = OpLabel
               OpBranch %22
         %23 = OpLabel
        %102 = OpPhi %7 %13 %22 %28 %27 %30 %29
               OpReturn
               OpFunctionEnd
)";

  ASSERT_TRUE(IsEqual(env, after_transformations, context.get()));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
