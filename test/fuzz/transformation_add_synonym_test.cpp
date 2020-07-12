// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_add_synonym.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddSynonymTest, NotApplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %8 RelaxedPrecision
               OpDecorate %22 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %10 = OpTypeFloat 32
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 4.5
         %14 = OpTypeVector %10 2
         %15 = OpTypePointer Function %14
         %17 = OpConstant %10 3
         %18 = OpConstant %10 4
         %19 = OpConstantComposite %14 %17 %18
         %20 = OpTypeVector %6 2
         %21 = OpTypePointer Function %20
         %23 = OpConstant %6 4
         %24 = OpConstantComposite %20 %9 %23
         %26 = OpConstantNull %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %12 = OpVariable %11 Function
         %16 = OpVariable %15 Function
         %22 = OpVariable %21 Function
               OpStore %8 %9
               OpStore %12 %13
               OpStore %16 %19
               OpStore %22 %24
         %25 = OpUndef %6
         %27 = OpLoad %6 %8
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(22, SpvOpReturn, 0);

#ifndef NDEBUG
  ASSERT_DEATH(
      TransformationAddSynonym(
          9, static_cast<protobufs::TransformationAddSynonym::SynonymType>(-1),
          40, insert_before)
          .IsApplicable(context.get(), transformation_context),
      "Synonym type is invalid");
#endif

  // |synonym_fresh_id| is not fresh.
  ASSERT_FALSE(
      TransformationAddSynonym(9, protobufs::TransformationAddSynonym::ADD_ZERO,
                               9, insert_before)
          .IsApplicable(context.get(), transformation_context));

  // |result_id| is invalid.
  ASSERT_FALSE(
      TransformationAddSynonym(
          40, protobufs::TransformationAddSynonym::ADD_ZERO, 40, insert_before)
          .IsApplicable(context.get(), transformation_context));

  // Instruction with |result_id| has no type id.
  ASSERT_FALSE(
      TransformationAddSynonym(5, protobufs::TransformationAddSynonym::ADD_ZERO,
                               40, insert_before)
          .IsApplicable(context.get(), transformation_context));

  // Instruction with |result_id| is an OpUndef.
  ASSERT_FALSE(
      TransformationAddSynonym(
          25, protobufs::TransformationAddSynonym::ADD_ZERO, 40, insert_before)
          .IsApplicable(context.get(), transformation_context));

  // Instruction with |result_id| is an OpConstantNull.
  ASSERT_FALSE(
      TransformationAddSynonym(
          26, protobufs::TransformationAddSynonym::ADD_ZERO, 40, insert_before)
          .IsApplicable(context.get(), transformation_context));

  // |insert_before| is invalid.
  ASSERT_FALSE(
      TransformationAddSynonym(9, protobufs::TransformationAddSynonym::ADD_ZERO,
                               40, MakeInstructionDescriptor(25, SpvOpStore, 0))
          .IsApplicable(context.get(), transformation_context));

  // Can't insert before |insert_before|.
  ASSERT_FALSE(
      TransformationAddSynonym(9, protobufs::TransformationAddSynonym::ADD_ZERO,
                               40, MakeInstructionDescriptor(5, SpvOpLabel, 0))
          .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddSynonym(
                   9, protobufs::TransformationAddSynonym::ADD_ZERO, 40,
                   MakeInstructionDescriptor(22, SpvOpVariable, 0))
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddSynonym(
                   9, protobufs::TransformationAddSynonym::ADD_ZERO, 40,
                   MakeInstructionDescriptor(25, SpvOpFunctionEnd, 0))
                   .IsApplicable(context.get(), transformation_context));

  // Domination rules are not satisfied.
  ASSERT_FALSE(TransformationAddSynonym(
                   27, protobufs::TransformationAddSynonym::ADD_ZERO, 40,
                   MakeInstructionDescriptor(27, SpvOpLoad, 0))
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddSynonym(
                   27, protobufs::TransformationAddSynonym::ADD_ZERO, 40,
                   MakeInstructionDescriptor(22, SpvOpStore, 1))
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddSynonymTest, AddZeroSubZeroMulOne) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpConstant %6 0
          %8 = OpConstant %6 1
          %9 = OpConstant %6 34
         %10 = OpTypeInt 32 0
         %13 = OpConstant %10 34
         %14 = OpTypeFloat 32
         %15 = OpConstant %14 0
         %16 = OpConstant %14 1
         %17 = OpConstant %14 34
         %18 = OpTypeVector %14 2
         %19 = OpConstantComposite %18 %15 %15
         %20 = OpConstantComposite %18 %16 %16
         %21 = OpConstant %14 3
         %22 = OpConstant %14 4
         %23 = OpConstantComposite %18 %21 %22
         %24 = OpTypeVector %6 2
         %25 = OpConstantComposite %24 %7 %7
         %26 = OpConstantComposite %24 %8 %8
         %27 = OpConstant %6 3
         %28 = OpConstant %6 4
         %29 = OpConstantComposite %24 %27 %28
         %30 = OpTypeVector %10 2
         %33 = OpConstant %10 3
         %34 = OpConstant %10 4
         %35 = OpConstantComposite %30 %33 %34
         %36 = OpTypeBool
         %37 = OpTypeVector %36 2
         %38 = OpConstantTrue %36
         %39 = OpConstantComposite %37 %38 %38
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
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(5, SpvOpReturn, 0);

  uint32_t fresh_id = 50;
  for (auto synonym_type : {protobufs::TransformationAddSynonym::ADD_ZERO,
                            protobufs::TransformationAddSynonym::SUB_ZERO,
                            protobufs::TransformationAddSynonym::MUL_ONE}) {
    ASSERT_TRUE(
        TransformationAddSynonym::IsAdditionalConstantRequired(synonym_type));

    // Can't create a synonym of a scalar or a vector of a wrong (in this case -
    // boolean) type.
    ASSERT_FALSE(
        TransformationAddSynonym(38, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));
    ASSERT_FALSE(
        TransformationAddSynonym(39, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));

    // Required constant is not present in the module.
    ASSERT_FALSE(
        TransformationAddSynonym(13, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));
    ASSERT_FALSE(
        TransformationAddSynonym(35, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));

    for (auto result_id : {9, 17, 23, 29}) {
      TransformationAddSynonym transformation(result_id, synonym_type, fresh_id,
                                              insert_before);
      ASSERT_TRUE(
          transformation.IsApplicable(context.get(), transformation_context));
      transformation.Apply(context.get(), &transformation_context);
      ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(result_id, {}),
                                            MakeDataDescriptor(fresh_id, {})));
      ++fresh_id;
    }
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpConstant %6 0
          %8 = OpConstant %6 1
          %9 = OpConstant %6 34
         %10 = OpTypeInt 32 0
         %13 = OpConstant %10 34
         %14 = OpTypeFloat 32
         %15 = OpConstant %14 0
         %16 = OpConstant %14 1
         %17 = OpConstant %14 34
         %18 = OpTypeVector %14 2
         %19 = OpConstantComposite %18 %15 %15
         %20 = OpConstantComposite %18 %16 %16
         %21 = OpConstant %14 3
         %22 = OpConstant %14 4
         %23 = OpConstantComposite %18 %21 %22
         %24 = OpTypeVector %6 2
         %25 = OpConstantComposite %24 %7 %7
         %26 = OpConstantComposite %24 %8 %8
         %27 = OpConstant %6 3
         %28 = OpConstant %6 4
         %29 = OpConstantComposite %24 %27 %28
         %30 = OpTypeVector %10 2
         %33 = OpConstant %10 3
         %34 = OpConstant %10 4
         %35 = OpConstantComposite %30 %33 %34
         %36 = OpTypeBool
         %37 = OpTypeVector %36 2
         %38 = OpConstantTrue %36
         %39 = OpConstantComposite %37 %38 %38
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %50 = OpIAdd %6 %9 %7
         %51 = OpFAdd %14 %17 %15
         %52 = OpFAdd %18 %23 %19
         %53 = OpIAdd %24 %29 %25
         %54 = OpISub %6 %9 %7
         %55 = OpFSub %14 %17 %15
         %56 = OpFSub %18 %23 %19
         %57 = OpISub %24 %29 %25
         %58 = OpIMul %6 %9 %8
         %59 = OpFMul %14 %17 %16
         %60 = OpFMul %18 %23 %20
         %61 = OpIMul %24 %29 %26
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationAddSynonymTest, LogicalAndLogicalOr) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantFalse %6
          %9 = OpConstantTrue %6
         %10 = OpTypeVector %6 2
         %11 = OpConstantComposite %10 %7 %9
         %12 = OpConstantComposite %10 %7 %7
         %13 = OpConstantComposite %10 %9 %9
         %14 = OpTypeFloat 32
         %17 = OpConstant %14 35
         %18 = OpTypeVector %14 2
         %21 = OpConstant %14 3
         %22 = OpConstant %14 4
         %23 = OpConstantComposite %18 %21 %22
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
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(5, SpvOpReturn, 0);

  uint32_t fresh_id = 50;
  for (auto synonym_type : {protobufs::TransformationAddSynonym::LOGICAL_AND,
                            protobufs::TransformationAddSynonym::LOGICAL_OR}) {
    ASSERT_TRUE(
        TransformationAddSynonym::IsAdditionalConstantRequired(synonym_type));

    // Can't create a synonym of a scalar or a vector of a wrong (in this case -
    // float) type.
    ASSERT_FALSE(
        TransformationAddSynonym(17, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));
    ASSERT_FALSE(
        TransformationAddSynonym(23, synonym_type, fresh_id, insert_before)
            .IsApplicable(context.get(), transformation_context));

    for (auto result_id : {9, 11}) {
      TransformationAddSynonym transformation(result_id, synonym_type, fresh_id,
                                              insert_before);
      ASSERT_TRUE(
          transformation.IsApplicable(context.get(), transformation_context));
      transformation.Apply(context.get(), &transformation_context);
      ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(result_id, {}),
                                            MakeDataDescriptor(fresh_id, {})));
      ++fresh_id;
    }
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantFalse %6
          %9 = OpConstantTrue %6
         %10 = OpTypeVector %6 2
         %11 = OpConstantComposite %10 %7 %9
         %12 = OpConstantComposite %10 %7 %7
         %13 = OpConstantComposite %10 %9 %9
         %14 = OpTypeFloat 32
         %17 = OpConstant %14 35
         %18 = OpTypeVector %14 2
         %21 = OpConstant %14 3
         %22 = OpConstant %14 4
         %23 = OpConstantComposite %18 %21 %22
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %50 = OpLogicalAnd %6 %9 %9
         %51 = OpLogicalAnd %10 %11 %13
         %52 = OpLogicalOr %6 %9 %7
         %53 = OpLogicalOr %10 %11 %12
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationAddSynonymTest, LogicalAndConstantIsNotPresent) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantFalse %6
         %10 = OpTypeVector %6 2
         %12 = OpConstantComposite %10 %7 %7
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
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(5, SpvOpReturn, 0);
  const auto synonym_type = protobufs::TransformationAddSynonym::LOGICAL_AND;

  // Required constant is not present in the module.
  ASSERT_FALSE(TransformationAddSynonym(7, synonym_type, 50, insert_before)
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddSynonym(12, synonym_type, 50, insert_before)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddSynonymTest, LogicalOrConstantIsNotPresent) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpConstantTrue %6
         %10 = OpTypeVector %6 2
         %12 = OpConstantComposite %10 %7 %7
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
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(5, SpvOpReturn, 0);
  const auto synonym_type = protobufs::TransformationAddSynonym::LOGICAL_OR;

  // Required constant is not present in the module.
  ASSERT_FALSE(TransformationAddSynonym(7, synonym_type, 50, insert_before)
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationAddSynonym(12, synonym_type, 50, insert_before)
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddSynonymTest, CopyObject) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %8 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 4
         %10 = OpTypeFloat 32
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 4
         %14 = OpTypeVector %10 2
         %15 = OpTypePointer Function %14
         %17 = OpConstant %10 3.4000001
         %18 = OpConstantComposite %14 %17 %17
         %19 = OpTypeBool
         %20 = OpTypeStruct %19
         %21 = OpTypePointer Function %20
         %23 = OpConstantTrue %19
         %24 = OpConstantComposite %20 %23
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %12 = OpVariable %11 Function
         %16 = OpVariable %15 Function
         %22 = OpVariable %21 Function
               OpStore %8 %9
               OpStore %12 %13
               OpStore %16 %18
               OpStore %22 %24
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto insert_before = MakeInstructionDescriptor(5, SpvOpReturn, 0);
  const auto synonym_type = protobufs::TransformationAddSynonym::COPY_OBJECT;

  ASSERT_FALSE(
      TransformationAddSynonym::IsAdditionalConstantRequired(synonym_type));

  uint32_t fresh_id = 50;
  for (auto result_id : {9, 13, 17, 18, 23, 24, 22}) {
    TransformationAddSynonym transformation(result_id, synonym_type, fresh_id,
                                            insert_before);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(result_id, {}),
                                          MakeDataDescriptor(fresh_id, {})));
    ++fresh_id;
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %8 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 4
         %10 = OpTypeFloat 32
         %11 = OpTypePointer Function %10
         %13 = OpConstant %10 4
         %14 = OpTypeVector %10 2
         %15 = OpTypePointer Function %14
         %17 = OpConstant %10 3.4000001
         %18 = OpConstantComposite %14 %17 %17
         %19 = OpTypeBool
         %20 = OpTypeStruct %19
         %21 = OpTypePointer Function %20
         %23 = OpConstantTrue %19
         %24 = OpConstantComposite %20 %23
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %12 = OpVariable %11 Function
         %16 = OpVariable %15 Function
         %22 = OpVariable %21 Function
               OpStore %8 %9
               OpStore %12 %13
               OpStore %16 %18
               OpStore %22 %24
         %50 = OpCopyObject %6 %9
         %51 = OpCopyObject %10 %13
         %52 = OpCopyObject %10 %17
         %53 = OpCopyObject %14 %18
         %54 = OpCopyObject %19 %23
         %55 = OpCopyObject %20 %24
         %56 = OpCopyObject %21 %22
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
