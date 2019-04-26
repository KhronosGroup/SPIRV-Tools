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

#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

using module_navigation::IdUseDescriptor;

TEST(TransformationReplaceConstantWithUniformTest, BasicReplacements) {
  // #version 450
  //
  // uniform blockname {
  //   int a;
  //   int b;
  //   int c;
  // };
  //
  // void main()
  // {
  //   int x;
  //   x = 1;
  //   x = x + 2;
  //   x = 3 + x;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "x"
               OpName %16 "blockname"
               OpMemberName %16 0 "a"
               OpMemberName %16 1 "b"
               OpMemberName %16 2 "c"
               OpName %18 ""
               OpMemberDecorate %16 0 Offset 0
               OpMemberDecorate %16 1 Offset 4
               OpMemberDecorate %16 2 Offset 8
               OpDecorate %16 Block
               OpDecorate %18 DescriptorSet 0
               OpDecorate %18 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %14 = OpConstant %6 3
         %16 = OpTypeStruct %6 %6 %6
         %17 = OpTypePointer Uniform %16
         %18 = OpVariable %17 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
         %10 = OpLoad %6 %8
         %12 = OpIAdd %6 %10 %11
               OpStore %8 %12
         %13 = OpLoad %6 %8
         %15 = OpIAdd %6 %14 %13
               OpStore %8 %15
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);

  FactManager fact_manager;
  FactManager::UniformBufferElementDescriptor blockname_a = {18, {0}};
  FactManager::UniformBufferElementDescriptor blockname_b = {18, {1}};
  FactManager::UniformBufferElementDescriptor blockname_c = {18, {2}};

  fact_manager.AddUniformIntValueFact(32, true, {1}, std::move(blockname_a));
  fact_manager.AddUniformIntValueFact(32, true, {2}, std::move(blockname_b));
  fact_manager.AddUniformIntValueFact(32, true, {3}, std::move(blockname_c));

  // The constant ids are 9, 11 and 14, for 1, 2 and 3 respectively.

  IdUseDescriptor use_of_9_in_store(9, SpvOpStore, 1, 8, 0);
  IdUseDescriptor use_of_11_in_add(11, SpvOpIAdd, 1, 12, 0);
  IdUseDescriptor use_of_14_in_add(14, SpvOpIAdd, 0, 15, 0);

  // These transformatios work: they match the facts.
  ASSERT_TRUE(
      TransformationReplaceConstantWithUniform(use_of_9_in_store, blockname_a)
          .IsApplicable(context.get(), fact_manager));
  ASSERT_TRUE(
      TransformationReplaceConstantWithUniform(use_of_11_in_add, blockname_b)
          .IsApplicable(context.get(), fact_manager));
  ASSERT_TRUE(
      TransformationReplaceConstantWithUniform(use_of_14_in_add, blockname_c)
          .IsApplicable(context.get(), fact_manager));

  // The transformations are not applicable if we change which uniforms are
  // applied to which constants.
  ASSERT_FALSE(
      TransformationReplaceConstantWithUniform(use_of_9_in_store, blockname_b)
          .IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(
      TransformationReplaceConstantWithUniform(use_of_11_in_add, blockname_c)
          .IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(
      TransformationReplaceConstantWithUniform(use_of_14_in_add, blockname_a)
          .IsApplicable(context.get(), fact_manager));

  FactManager::UniformBufferElementDescriptor nonsense_uniform_descriptor1 = {
      19, {0}};
  FactManager::UniformBufferElementDescriptor nonsense_uniform_descriptor2 = {
      18, {5}};
  ASSERT_FALSE(TransformationReplaceConstantWithUniform(
                   use_of_9_in_store, nonsense_uniform_descriptor1)
                   .IsApplicable(context.get(), fact_manager));
  ASSERT_FALSE(TransformationReplaceConstantWithUniform(
                   use_of_9_in_store, nonsense_uniform_descriptor2)
                   .IsApplicable(context.get(), fact_manager));
  IdUseDescriptor nonsense_id_use_descriptor(9, SpvOpIAdd, 0, 15, 0);
  ASSERT_FALSE(TransformationReplaceConstantWithUniform(
                   nonsense_id_use_descriptor, blockname_a)
                   .IsApplicable(context.get(), fact_manager));

  // TODO: now apply some transformations.
}

TEST(TransformationMoveBlockDownTest, ComplexReplacements) {
  // TODO: do something interesting with this shader.

  // #version 450
  //
  // struct T {
  //   int a[5];
  //   ivec4 b;
  //   ivec3 c;
  //   int d;
  //   bool e;
  // };
  //
  // uniform block {
  //   T f;
  //   int g;
  //   ivec2 h;
  // };
  //
  // void main()
  // {
  //   T myT = T(int[](1, 2, 3, 4, 5), ivec4(6, 7, 8, 9), ivec3(10, 11, 12), 13,
  //   true); myT.c.x = 4; myT.a[2] = 5; myT.e = myT.c[1] > 2;
  // }
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
