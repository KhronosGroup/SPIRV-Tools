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

#include "source/fuzz/transformation_copy_object.h"
#include "source/fuzz/data_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationCopyObjectTest, CopyBooleanConstants) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %6 = OpTypeBool
          %7 = OpConstantTrue %6
          %8 = OpConstantFalse %6
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  ASSERT_EQ(0, fact_manager.GetIdsForWhichSynonymsAreKnown().size());

  TransformationCopyObject copy_true(100, 7, 5, 0);
  ASSERT_TRUE(copy_true.IsApplicable(context.get(), fact_manager));
  copy_true.Apply(context.get(), &fact_manager);

  const std::set<uint32_t>& ids_for_which_synonyms_are_known =
      fact_manager.GetIdsForWhichSynonymsAreKnown();
  ASSERT_EQ(1, ids_for_which_synonyms_are_known.size());
  ASSERT_TRUE(ids_for_which_synonyms_are_known.find(7) !=
              ids_for_which_synonyms_are_known.end());
  ASSERT_EQ(1, fact_manager.GetSynonymsForId(7).size());
  protobufs::DataDescriptor descriptor_100 = MakeDataDescriptor(100, {});
  ASSERT_TRUE(DataDescriptorEquals()(&descriptor_100,
                                     &fact_manager.GetSynonymsForId(7)[0]));

  TransformationCopyObject copy_false(101, 8, 100, 0);
  ASSERT_TRUE(copy_false.IsApplicable(context.get(), fact_manager));
  copy_false.Apply(context.get(), &fact_manager);
  ASSERT_EQ(2, ids_for_which_synonyms_are_known.size());
  ASSERT_TRUE(ids_for_which_synonyms_are_known.find(8) !=
              ids_for_which_synonyms_are_known.end());
  ASSERT_EQ(1, fact_manager.GetSynonymsForId(8).size());
  protobufs::DataDescriptor descriptor_101 = MakeDataDescriptor(101, {});
  ASSERT_TRUE(DataDescriptorEquals()(&descriptor_101,
                                     &fact_manager.GetSynonymsForId(8)[0]));

  TransformationCopyObject copy_false_again(102, 101, 5, 2);
  ASSERT_TRUE(copy_false_again.IsApplicable(context.get(), fact_manager));
  copy_false_again.Apply(context.get(), &fact_manager);
  ASSERT_EQ(3, ids_for_which_synonyms_are_known.size());
  ASSERT_TRUE(ids_for_which_synonyms_are_known.find(101) !=
              ids_for_which_synonyms_are_known.end());
  ASSERT_EQ(1, fact_manager.GetSynonymsForId(101).size());
  protobufs::DataDescriptor descriptor_102 = MakeDataDescriptor(102, {});
  ASSERT_TRUE(DataDescriptorEquals()(&descriptor_102,
                                     &fact_manager.GetSynonymsForId(101)[0]));

  TransformationCopyObject copy_true_again(103, 7, 102, 0);
  ASSERT_TRUE(copy_true_again.IsApplicable(context.get(), fact_manager));
  copy_true_again.Apply(context.get(), &fact_manager);
  // This does re-uses an id for which synonyms are already known, so the count
  // of such ids does not change.
  ASSERT_EQ(3, ids_for_which_synonyms_are_known.size());
  ASSERT_TRUE(ids_for_which_synonyms_are_known.find(7) !=
              ids_for_which_synonyms_are_known.end());
  ASSERT_EQ(2, fact_manager.GetSynonymsForId(7).size());
  protobufs::DataDescriptor descriptor_103 = MakeDataDescriptor(103, {});
  ASSERT_TRUE(DataDescriptorEquals()(&descriptor_103,
                                     &fact_manager.GetSynonymsForId(7)[0]) ||
              DataDescriptorEquals()(&descriptor_103,
                                     &fact_manager.GetSynonymsForId(7)[1]));

  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %6 = OpTypeBool
          %7 = OpConstantTrue %6
          %8 = OpConstantFalse %6
          %3 = OpTypeFunction %2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
        %100 = OpCopyObject %6 %7
        %101 = OpCopyObject %6 %8
        %102 = OpCopyObject %6 %101
        %103 = OpCopyObject %6 %7
               OpReturn
               OpFunctionEnd
  )";
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
