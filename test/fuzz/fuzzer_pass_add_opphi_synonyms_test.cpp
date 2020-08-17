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

#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"
#include "source/fuzz/pseudo_random_generator.h"
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

bool EquivalenceClassesMatch(const std::vector<std::set<uint32_t>>& classes1,
                             const std::vector<std::set<uint32_t>>& classes2) {
  std::set<std::set<uint32_t>> set1;
  for (auto equivalence_class : classes1) {
    set1.emplace(equivalence_class);
  }

  std::set<std::set<uint32_t>> set2;
  for (auto equivalence_class : classes2) {
    set2.emplace(equivalence_class);
  }

  return set1 == set2;
}

TEST(FuzzerPassAddOpPhiSynonymsTest, GetEquivalenceClasses) {
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
         %22 = OpTypePointer Function %7
          %9 = OpConstant %7 1
         %10 = OpConstant %7 2
         %11 = OpConstant %8 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %23 = OpVariable %22 Function
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
         %24 = OpCopyObject %22 %23
         %25 = OpCopyObject %7 %10
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

  PseudoRandomGenerator prng(0);
  FuzzerContext fuzzer_context(&prng, 100);
  protobufs::TransformationSequence transformation_sequence;

  FuzzerPassAddOpPhiSynonyms fuzzer_pass(context.get(), &transformation_context,
                                         &fuzzer_context,
                                         &transformation_sequence);

  SetUpIdSynonyms(&fact_manager, context.get());
  fact_manager.AddFact(MakeSynonymFact(23, 24), context.get());

  std::vector<std::set<uint32_t>> expected_equivalence_classes = {
      {9, 13, 19}, {11, 14, 20}, {10, 21}, {6}, {25}};

  ASSERT_TRUE(EquivalenceClassesMatch(fuzzer_pass.GetIdEquivalenceClasses(),
                                      expected_equivalence_classes));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools
