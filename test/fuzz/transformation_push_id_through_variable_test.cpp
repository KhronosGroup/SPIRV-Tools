// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_push_id_through_variable.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

std::string reference_shader = R"(
             OpCapability Shader
        %1 = OpExtInstImport "GLSL.std.450"
             OpMemoryModel Logical GLSL450
             OpEntryPoint Fragment %4 "main" %92 %52 %53
             OpExecutionMode %4 OriginUpperLeft
             OpSource ESSL 310
             OpDecorate %92 BuiltIn FragCoord
        %2 = OpTypeVoid
        %3 = OpTypeFunction %2
        %6 = OpTypeInt 32 1
        %7 = OpTypeFloat 32
        %8 = OpTypeStruct %6 %7
        %9 = OpTypePointer Function %8
       %10 = OpTypeFunction %6 %9
       %14 = OpConstant %6 0
       %15 = OpTypePointer Function %6
       %51 = OpTypePointer Private %6
       %21 = OpConstant %6 2
       %23 = OpConstant %6 1
       %24 = OpConstant %7 1
       %25 = OpTypePointer Function %7
       %50 = OpTypePointer Private %7
       %34 = OpTypeBool
       %35 = OpConstantFalse %34
       %60 = OpConstantNull %50
       %61 = OpUndef %51
       %52 = OpVariable %50 Private
       %53 = OpVariable %51 Private
       %80 = OpConstantComposite %8 %21 %24
       %90 = OpTypeVector %7 4
       %91 = OpTypePointer Input %90
       %92 = OpVariable %91 Input
       %93 = OpConstantComposite %90 %24 %24 %24 %24
        %4 = OpFunction %2 None %3
        %5 = OpLabel
       %20 = OpVariable %9 Function
       %27 = OpVariable %9 Function
       %22 = OpAccessChain %15 %20 %14
       %44 = OpCopyObject %9 %20
       %26 = OpAccessChain %25 %20 %23
       %29 = OpFunctionCall %6 %12 %27
       %30 = OpAccessChain %15 %20 %14
       %45 = OpCopyObject %15 %30
       %81 = OpCopyObject %9 %27
       %33 = OpAccessChain %15 %20 %14
             OpSelectionMerge %37 None
             OpBranchConditional %35 %36 %37
       %36 = OpLabel
       %38 = OpAccessChain %15 %20 %14
       %40 = OpAccessChain %15 %20 %14
       %43 = OpAccessChain %15 %20 %14
       %82 = OpCopyObject %9 %27
             OpBranch %37
       %37 = OpLabel
             OpReturn
             OpFunctionEnd
       %12 = OpFunction %6 None %10
       %11 = OpFunctionParameter %9
       %13 = OpLabel
       %46 = OpCopyObject %9 %11
       %16 = OpAccessChain %15 %11 %14
       %95 = OpCopyObject %8 %80
             OpReturnValue %21
             OpFunctionEnd
)";

const auto env = SPV_ENV_UNIVERSAL_1_4;
const auto consumer = nullptr;
const auto context =
    BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);

FactManager fact_manager;
spvtools::ValidatorOptions validator_options;
TransformationContext transformation_context(&fact_manager, validator_options);

// Tests the reference shader validity.
TEST(TransformationPushIdThroughVariableTest, ReferenceShaderValidity) {
  ASSERT_TRUE(IsValid(env, context.get()));
}

// Tests |value_synonym_id| is a fresh id.
TEST(TransformationPushIdThroughVariableTest, FreshId) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 21;
  uint32_t variable_id = 16;
  auto instruction_descriptor =
      MakeInstructionDescriptor(95, SpvOpReturnValue, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests |value_synonym_id| is a non-fresh id.
TEST(TransformationPushIdThroughVariableTest, NonFreshId) {
  uint32_t value_synonym_id = 61;
  uint32_t value_id = 80;
  uint32_t variable_id = 27;
  auto instruction_descriptor =
      MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests variable instruction not available.
TEST(TransformationPushIdThroughVariableTest, VariableIdNotAvailable) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 80;
  uint32_t variable_id = 63;
  auto instruction_descriptor =
      MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests variable type instruction is an OpTypePointer instruction.
TEST(TransformationPushIdThroughVariableTest, IsTypePointerInstruction) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 80;
  uint32_t variable_id = 95;
  auto instruction_descriptor =
      MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Attempting to store to a read-only pointer.
TEST(TransformationPushIdThroughVariableTest, TryToStoreToReadOnlyPointer) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 93;
  uint32_t variable_id = 92;
  auto instruction_descriptor =
      MakeInstructionDescriptor(40, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Attempting to store to a null or undefined pointer.
TEST(TransformationPushIdThroughVariableTest, TryToStoreToNullOrUndefPointer) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 80;
  uint32_t variable_id = 60;
  auto instruction_descriptor =
      MakeInstructionDescriptor(40, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  variable_id = 61;
  transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Attempting to insert the store and load instructions
// before an OpVariable instruction.
TEST(TransformationPushIdThroughVariableTest, NotInsertingBeforeInstruction) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 24;
  uint32_t variable_id = 52;
  auto instruction_descriptor = MakeInstructionDescriptor(27, SpvOpVariable, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests value instruction not available.
TEST(TransformationPushIdThroughVariableTest, ValueIdNotAvailable) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 63;
  uint32_t variable_id = 16;
  auto instruction_descriptor =
      MakeInstructionDescriptor(95, SpvOpReturnValue, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests pointer and value with different types.
TEST(TransformationPushIdThroughVariableTest,
     PointerAndValueWithDifferentTypes) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 14;
  uint32_t variable_id = 27;
  auto instruction_descriptor =
      MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests pointer instruction not available before instruction.
TEST(TransformationPushIdThroughVariableTest,
     PointerNotAvailableBeforeInstruction) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 80;
  uint32_t variable_id = 82;
  auto instruction_descriptor = MakeInstructionDescriptor(37, SpvOpReturn, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

// Tests value instruction not available before instruction.
TEST(TransformationPushIdThroughVariableTest,
     ValueNotAvailableBeforeInstruction) {
  uint32_t value_synonym_id = 62;
  uint32_t value_id = 95;
  uint32_t variable_id = 27;
  auto instruction_descriptor =
      MakeInstructionDescriptor(40, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationPushIdThroughVariableTest, Apply) {
  uint32_t value_synonym_id = 100;
  uint32_t value_id = 80;
  uint32_t variable_id = 27;
  auto instruction_descriptor =
      MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  auto transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  value_synonym_id = 101;
  value_id = 21;
  variable_id = 53;
  instruction_descriptor = MakeInstructionDescriptor(38, SpvOpAccessChain, 0);
  transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  value_synonym_id = 102;
  value_id = 95;
  variable_id = 11;
  instruction_descriptor = MakeInstructionDescriptor(95, SpvOpReturnValue, 0);
  transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  value_synonym_id = 103;
  value_id = 80;
  variable_id = 46;
  instruction_descriptor = MakeInstructionDescriptor(95, SpvOpReturnValue, 0);
  transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  value_synonym_id = 104;
  value_id = 21;
  variable_id = 16;
  instruction_descriptor = MakeInstructionDescriptor(95, SpvOpReturnValue, 0);
  transformation = TransformationPushIdThroughVariable(
      value_synonym_id, value_id, variable_id, instruction_descriptor);
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %92 %52 %53
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %92 BuiltIn FragCoord
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeFloat 32
          %8 = OpTypeStruct %6 %7
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %6 %9
         %14 = OpConstant %6 0
         %15 = OpTypePointer Function %6
         %51 = OpTypePointer Private %6
         %21 = OpConstant %6 2
         %23 = OpConstant %6 1
         %24 = OpConstant %7 1
         %25 = OpTypePointer Function %7
         %50 = OpTypePointer Private %7
         %34 = OpTypeBool
         %35 = OpConstantFalse %34
         %60 = OpConstantNull %50
         %61 = OpUndef %51
         %52 = OpVariable %50 Private
         %53 = OpVariable %51 Private
         %80 = OpConstantComposite %8 %21 %24
         %90 = OpTypeVector %7 4
         %91 = OpTypePointer Input %90
         %92 = OpVariable %91 Input
         %93 = OpConstantComposite %90 %24 %24 %24 %24
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %20 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %22 = OpAccessChain %15 %20 %14
         %44 = OpCopyObject %9 %20
         %26 = OpAccessChain %25 %20 %23
         %29 = OpFunctionCall %6 %12 %27
         %30 = OpAccessChain %15 %20 %14
         %45 = OpCopyObject %15 %30
         %81 = OpCopyObject %9 %27
         %33 = OpAccessChain %15 %20 %14
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %37
         %36 = OpLabel
               OpStore %27 %80
        %100 = OpLoad %8 %27
               OpStore %53 %21
        %101 = OpLoad %6 %53
         %38 = OpAccessChain %15 %20 %14
         %40 = OpAccessChain %15 %20 %14
         %43 = OpAccessChain %15 %20 %14
         %82 = OpCopyObject %9 %27
               OpBranch %37
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %6 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %46 = OpCopyObject %9 %11
         %16 = OpAccessChain %15 %11 %14
         %95 = OpCopyObject %8 %80
               OpStore %11 %95
        %102 = OpLoad %8 %11
               OpStore %46 %80
        %103 = OpLoad %8 %46
               OpStore %16 %21
        %104 = OpLoad %6 %16
               OpReturnValue %21
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(80, {}),
                                        MakeDataDescriptor(100, {})));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(21, {}),
                                        MakeDataDescriptor(101, {})));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(95, {}),
                                        MakeDataDescriptor(102, {})));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(80, {}),
                                        MakeDataDescriptor(103, {})));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(21, {}),
                                        MakeDataDescriptor(104, {})));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
