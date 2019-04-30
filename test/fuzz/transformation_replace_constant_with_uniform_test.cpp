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

TEST(TransformationReplaceConstantWithUniformTest, BasicReplacements) {
  // This test came from the following GLSL:
  //
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
         %50 = OpConstant %6 0
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %14 = OpConstant %6 3
         %16 = OpTypeStruct %6 %6 %6
         %17 = OpTypePointer Uniform %16
         %51 = OpTypePointer Uniform %6
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
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_a =
      FactManager::MakeUniformBufferElementDescriptor(18, {0});
  protobufs::UniformBufferElementDescriptor blockname_b =
      FactManager::MakeUniformBufferElementDescriptor(18, {1});
  protobufs::UniformBufferElementDescriptor blockname_c =
      FactManager::MakeUniformBufferElementDescriptor(18, {2});

  fact_manager.AddUniformIntValueFact(32, true, {1}, blockname_a);
  fact_manager.AddUniformIntValueFact(32, true, {2}, blockname_b);
  fact_manager.AddUniformIntValueFact(32, true, {3}, blockname_c);

  // The constant ids are 9, 11 and 14, for 1, 2 and 3 respectively.

  protobufs::IdUseDescriptor use_of_9_in_store =
      module_navigation::MakeIdUseDescriptor(9, SpvOpStore, 1, 8, 0);
  protobufs::IdUseDescriptor use_of_11_in_add =
      module_navigation::MakeIdUseDescriptor(11, SpvOpIAdd, 1, 12, 0);
  protobufs::IdUseDescriptor use_of_14_in_add =
      module_navigation::MakeIdUseDescriptor(14, SpvOpIAdd, 0, 15, 0);

  // These transformations work: they match the facts.
  auto transformation_use_of_9_in_store =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_a, 100, 101);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_9_in_store,
                                           context.get(), fact_manager));
  auto transformation_use_of_11_in_add =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_add, blockname_b, 102, 103);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_11_in_add,
                                           context.get(), fact_manager));
  auto transformation_use_of_14_in_add =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_14_in_add, blockname_c, 104, 105);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_14_in_add,
                                           context.get(), fact_manager));

  // The transformations are not applicable if we change which uniforms are
  // applied to which constants.
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_b, 101, 102),
      context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_add, blockname_c, 101, 102),
      context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_14_in_add, blockname_a, 101, 102),
      context.get(), fact_manager));

  // The following transformations do not apply because the uniform descriptors
  // are not sensible.
  protobufs::UniformBufferElementDescriptor nonsense_uniform_descriptor1 =
      FactManager::MakeUniformBufferElementDescriptor(19, {0});
  protobufs::UniformBufferElementDescriptor nonsense_uniform_descriptor2 =
      FactManager::MakeUniformBufferElementDescriptor(18, {5});
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, nonsense_uniform_descriptor1, 101, 102),
      context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, nonsense_uniform_descriptor2, 101, 102),
      context.get(), fact_manager));

  // The following transformation does not apply because the id descriptor is
  // not sensible.
  protobufs::IdUseDescriptor nonsense_id_use_descriptor =
      module_navigation::MakeIdUseDescriptor(9, SpvOpIAdd, 0, 15, 0);
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          nonsense_id_use_descriptor, blockname_a, 101, 102),
      context.get(), fact_manager));

  // The following transformations do not apply because the ids are not fresh.
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_add, blockname_b, 15, 103),
      context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_add, blockname_b, 102, 15),
      context.get(), fact_manager));

  // Apply the use of 9 in a store.
  transformation::Apply(transformation_use_of_9_in_store, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  std::string after_replacing_use_of_9_in_store = R"(
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
         %50 = OpConstant %6 0
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %14 = OpConstant %6 3
         %16 = OpTypeStruct %6 %6 %6
         %17 = OpTypePointer Uniform %16
         %51 = OpTypePointer Uniform %6
         %18 = OpVariable %17 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
        %100 = OpAccessChain %51 %18 %50
        %101 = OpLoad %6 %100
               OpStore %8 %101
         %10 = OpLoad %6 %8
         %12 = OpIAdd %6 %10 %11
               OpStore %8 %12
         %13 = OpLoad %6 %8
         %15 = OpIAdd %6 %14 %13
               OpStore %8 %15
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_replacing_use_of_9_in_store, context.get());

  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_11_in_add,
                                           context.get(), fact_manager));
  // Apply the use of 11 in an add.
  transformation::Apply(transformation_use_of_11_in_add, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  std::string after_replacing_use_of_11_in_add = R"(
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
         %50 = OpConstant %6 0
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %14 = OpConstant %6 3
         %16 = OpTypeStruct %6 %6 %6
         %17 = OpTypePointer Uniform %16
         %51 = OpTypePointer Uniform %6
         %18 = OpVariable %17 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
        %100 = OpAccessChain %51 %18 %50
        %101 = OpLoad %6 %100
               OpStore %8 %101
         %10 = OpLoad %6 %8
        %102 = OpAccessChain %51 %18 %9
        %103 = OpLoad %6 %102
         %12 = OpIAdd %6 %10 %103
               OpStore %8 %12
         %13 = OpLoad %6 %8
         %15 = OpIAdd %6 %14 %13
               OpStore %8 %15
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_replacing_use_of_11_in_add, context.get());

  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_14_in_add,
                                           context.get(), fact_manager));
  // Apply the use of 15 in an add.
  transformation::Apply(transformation_use_of_14_in_add, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  std::string after_replacing_use_of_14_in_add = R"(
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
         %50 = OpConstant %6 0
          %9 = OpConstant %6 1
         %11 = OpConstant %6 2
         %14 = OpConstant %6 3
         %16 = OpTypeStruct %6 %6 %6
         %17 = OpTypePointer Uniform %16
         %51 = OpTypePointer Uniform %6
         %18 = OpVariable %17 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
        %100 = OpAccessChain %51 %18 %50
        %101 = OpLoad %6 %100
               OpStore %8 %101
         %10 = OpLoad %6 %8
        %102 = OpAccessChain %51 %18 %9
        %103 = OpLoad %6 %102
         %12 = OpIAdd %6 %10 %103
               OpStore %8 %12
         %13 = OpLoad %6 %8
        %104 = OpAccessChain %51 %18 %11
        %105 = OpLoad %6 %104
         %15 = OpIAdd %6 %105 %13
               OpStore %8 %15
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after_replacing_use_of_14_in_add, context.get());
}

TEST(TransformationReplaceConstantWithUniformTest, NestedStruct) {
  // This test came from the following GLSL:
  //
  // #version 450
  //
  // struct U {
  //   int x; // == 4
  // };
  //
  // struct T {
  //   int x; // == 3
  //   U y;
  // };
  //
  // struct S {
  //   T x;
  //   int y; // == 2
  // };
  //
  // uniform blockname {
  //   int x; // == 1
  //   S y;
  // };
  //
  // void foo(int a) { }
  //
  // void main()
  // {
  //   int x;
  //   x = 1;
  //   x = x + 2;
  //   x = 3 + x;
  //   foo(4);
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %10 "foo(i1;"
               OpName %9 "a"
               OpName %12 "x"
               OpName %21 "param"
               OpName %23 "U"
               OpMemberName %23 0 "x"
               OpName %24 "T"
               OpMemberName %24 0 "x"
               OpMemberName %24 1 "y"
               OpName %25 "S"
               OpMemberName %25 0 "x"
               OpMemberName %25 1 "y"
               OpName %26 "blockname"
               OpMemberName %26 0 "x"
               OpMemberName %26 1 "y"
               OpName %28 ""
               OpMemberDecorate %23 0 Offset 0
               OpMemberDecorate %24 0 Offset 0
               OpMemberDecorate %24 1 Offset 16
               OpMemberDecorate %25 0 Offset 0
               OpMemberDecorate %25 1 Offset 32
               OpMemberDecorate %26 0 Offset 0
               OpMemberDecorate %26 1 Offset 16
               OpDecorate %26 Block
               OpDecorate %28 DescriptorSet 0
               OpDecorate %28 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %50 = OpConstant %6 0
         %13 = OpConstant %6 1
         %15 = OpConstant %6 2
         %17 = OpConstant %6 3
         %20 = OpConstant %6 4
         %23 = OpTypeStruct %6
         %24 = OpTypeStruct %6 %23
         %25 = OpTypeStruct %24 %6
         %26 = OpTypeStruct %6 %25
         %27 = OpTypePointer Uniform %26
         %51 = OpTypePointer Uniform %6
         %28 = OpVariable %27 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpVariable %7 Function
         %21 = OpVariable %7 Function
               OpStore %12 %13
         %14 = OpLoad %6 %12
         %16 = OpIAdd %6 %14 %15
               OpStore %12 %16
         %18 = OpLoad %6 %12
         %19 = OpIAdd %6 %17 %18
               OpStore %12 %19
               OpStore %21 %20
         %22 = OpFunctionCall %2 %10 %21
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_1 =
      FactManager::MakeUniformBufferElementDescriptor(28, {0});
  protobufs::UniformBufferElementDescriptor blockname_2 =
      FactManager::MakeUniformBufferElementDescriptor(28, {1, 1});
  protobufs::UniformBufferElementDescriptor blockname_3 =
      FactManager::MakeUniformBufferElementDescriptor(28, {1, 0, 0});
  protobufs::UniformBufferElementDescriptor blockname_4 =
      FactManager::MakeUniformBufferElementDescriptor(28, {1, 0, 1, 0});

  fact_manager.AddUniformIntValueFact(32, true, {1}, blockname_1);
  fact_manager.AddUniformIntValueFact(32, true, {2}, blockname_2);
  fact_manager.AddUniformIntValueFact(32, true, {3}, blockname_3);
  fact_manager.AddUniformIntValueFact(32, true, {4}, blockname_4);

  // The constant ids are 13, 15, 17 and 20, for 1, 2, 3 and 4 respectively.
  protobufs::IdUseDescriptor use_of_13_in_store =
      module_navigation::MakeIdUseDescriptor(13, SpvOpStore, 1, 21, 0);
  protobufs::IdUseDescriptor use_of_15_in_add =
      module_navigation::MakeIdUseDescriptor(15, SpvOpIAdd, 1, 16, 0);
  protobufs::IdUseDescriptor use_of_17_in_add =
      module_navigation::MakeIdUseDescriptor(17, SpvOpIAdd, 0, 19, 0);
  protobufs::IdUseDescriptor use_of_20_in_store =
      module_navigation::MakeIdUseDescriptor(20, SpvOpStore, 1, 19, 1);

  // These transformations work: they match the facts.
  auto transformation_use_of_13_in_store =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_13_in_store, blockname_1, 100, 101);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                           context.get(), fact_manager));
  auto transformation_use_of_15_in_add =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_15_in_add, blockname_2, 102, 103);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                           context.get(), fact_manager));
  auto transformation_use_of_17_in_add =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_17_in_add, blockname_3, 104, 105);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                           context.get(), fact_manager));
  auto transformation_use_of_20_in_store =
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_20_in_store, blockname_4, 106, 107);
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                           context.get(), fact_manager));

  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                           context.get(), fact_manager));

  transformation::Apply(transformation_use_of_13_in_store, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                            context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                           context.get(), fact_manager));

  transformation::Apply(transformation_use_of_15_in_add, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                            context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                           context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                           context.get(), fact_manager));

  transformation::Apply(transformation_use_of_17_in_add, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                            context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                           context.get(), fact_manager));

  transformation::Apply(transformation_use_of_20_in_store, context.get(),
                        &fact_manager);
  CheckValid(env, context.get());
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_13_in_store,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_15_in_add,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_17_in_add,
                                            context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(transformation_use_of_20_in_store,
                                            context.get(), fact_manager));

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %10 "foo(i1;"
               OpName %9 "a"
               OpName %12 "x"
               OpName %21 "param"
               OpName %23 "U"
               OpMemberName %23 0 "x"
               OpName %24 "T"
               OpMemberName %24 0 "x"
               OpMemberName %24 1 "y"
               OpName %25 "S"
               OpMemberName %25 0 "x"
               OpMemberName %25 1 "y"
               OpName %26 "blockname"
               OpMemberName %26 0 "x"
               OpMemberName %26 1 "y"
               OpName %28 ""
               OpMemberDecorate %23 0 Offset 0
               OpMemberDecorate %24 0 Offset 0
               OpMemberDecorate %24 1 Offset 16
               OpMemberDecorate %25 0 Offset 0
               OpMemberDecorate %25 1 Offset 32
               OpMemberDecorate %26 0 Offset 0
               OpMemberDecorate %26 1 Offset 16
               OpDecorate %26 Block
               OpDecorate %28 DescriptorSet 0
               OpDecorate %28 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %50 = OpConstant %6 0
         %13 = OpConstant %6 1
         %15 = OpConstant %6 2
         %17 = OpConstant %6 3
         %20 = OpConstant %6 4
         %23 = OpTypeStruct %6
         %24 = OpTypeStruct %6 %23
         %25 = OpTypeStruct %24 %6
         %26 = OpTypeStruct %6 %25
         %27 = OpTypePointer Uniform %26
         %51 = OpTypePointer Uniform %6
         %28 = OpVariable %27 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %12 = OpVariable %7 Function
         %21 = OpVariable %7 Function
        %100 = OpAccessChain %51 %28 %50
        %101 = OpLoad %6 %100
               OpStore %12 %101
         %14 = OpLoad %6 %12
        %102 = OpAccessChain %51 %28 %13 %13
        %103 = OpLoad %6 %102
         %16 = OpIAdd %6 %14 %103
               OpStore %12 %16
         %18 = OpLoad %6 %12
        %104 = OpAccessChain %51 %28 %13 %50 %50
        %105 = OpLoad %6 %104
         %19 = OpIAdd %6 %105 %18
               OpStore %12 %19
        %106 = OpAccessChain %51 %28 %13 %50 %13 %50
        %107 = OpLoad %6 %106
               OpStore %21 %107
         %22 = OpFunctionCall %2 %10 %21
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
  )";
  CheckEqual(env, after, context.get());
}

TEST(TransformationReplaceConstantWithUniformTest, NoUniformIntPointerPresent) {
  // This test came from the following GLSL:
  //
  // #version 450
  //
  // uniform blockname {
  //   int x; // == 0
  // };
  //
  // void main()
  // {
  //   int a;
  //   a = 0;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "blockname"
               OpMemberName %10 0 "x"
               OpName %12 ""
               OpMemberDecorate %10 0 Offset 0
               OpDecorate %10 Block
               OpDecorate %12 DescriptorSet 0
               OpDecorate %12 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 0
         %10 = OpTypeStruct %6
         %11 = OpTypePointer Uniform %10
         %12 = OpVariable %11 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_0 =
      FactManager::MakeUniformBufferElementDescriptor(12, {0});

  fact_manager.AddUniformIntValueFact(32, true, {0}, blockname_0);

  // The constant id is 9 for 0.
  protobufs::IdUseDescriptor use_of_9_in_store =
      module_navigation::MakeIdUseDescriptor(9, SpvOpStore, 1, 8, 0);

  // This transformation is not available because no uniform pointer to integer
  // type is present:
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_0, 100, 101),
      context.get(), fact_manager));
}

TEST(TransformationReplaceConstantWithUniformTest, NoConstantPresentForIndex) {
  // This test came from the following GLSL:
  //
  // #version 450
  //
  // uniform blockname {
  //   int x; // == 0
  //   int y; // == 9
  // };
  //
  // void main()
  // {
  //   int a;
  //   a = 9;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "blockname"
               OpMemberName %10 0 "x"
               OpMemberName %10 1 "y"
               OpName %12 ""
               OpMemberDecorate %10 0 Offset 0
               OpMemberDecorate %10 1 Offset 4
               OpDecorate %10 Block
               OpDecorate %12 DescriptorSet 0
               OpDecorate %12 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 9
         %10 = OpTypeStruct %6 %6
         %11 = OpTypePointer Uniform %10
         %50 = OpTypePointer Uniform %6
         %12 = OpVariable %11 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_0 =
      FactManager::MakeUniformBufferElementDescriptor(12, {0});
  protobufs::UniformBufferElementDescriptor blockname_9 =
      FactManager::MakeUniformBufferElementDescriptor(12, {1});

  fact_manager.AddUniformIntValueFact(32, true, {9}, blockname_9);

  // The constant id is 9 for 9.
  protobufs::IdUseDescriptor use_of_9_in_store =
      module_navigation::MakeIdUseDescriptor(9, SpvOpStore, 1, 8, 0);

  // This transformation is not available because no constant is present for the
  // index 1 required to index into the uniform buffer:
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_9, 100, 101),
      context.get(), fact_manager));
}

TEST(TransformationReplaceConstantWithUniformTest,
     NoIntTypePresentToEnableIndexing) {
  // This test came from the following GLSL:
  //
  // #version 450
  //
  // uniform blockname {
  //   float f; // == 9
  // };
  //
  // void main()
  // {
  //   float a;
  //   a = 3.0;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "blockname"
               OpMemberName %10 0 "f"
               OpName %12 ""
               OpMemberDecorate %10 0 Offset 0
               OpDecorate %10 Block
               OpDecorate %12 DescriptorSet 0
               OpDecorate %12 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 3
         %10 = OpTypeStruct %6
         %11 = OpTypePointer Uniform %10
         %12 = OpVariable %11 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
               OpStore %8 %9
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_3 =
      FactManager::MakeUniformBufferElementDescriptor(12, {0});

  uint32_t float_data[1];
  float temp = 3.0;
  memcpy(&float_data[0], &temp, sizeof(float));
  fact_manager.AddUniformFloatValueFact(32, {float_data[0]}, blockname_3);

  // The constant id is 9 for 3.0.
  protobufs::IdUseDescriptor use_of_9_in_store =
      module_navigation::MakeIdUseDescriptor(9, SpvOpStore, 1, 8, 0);

  // This transformation is not available because no integer type is present to
  // allow a constant index to be expressed:
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_3, 100, 101),
      context.get(), fact_manager));
}

TEST(TransformationReplaceConstantWithUniformTest,
     UniformFactsDoNotMatchConstants) {
  // This test came from the following GLSL:
  //
  // #version 450
  //
  // uniform blockname {
  //   int x; // == 9
  //   int y; // == 10
  // };
  //
  // void main()
  // {
  //   int a;
  //   int b;
  //   a = 9;
  //   b = 10;
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %8 "a"
               OpName %10 "b"
               OpName %12 "blockname"
               OpMemberName %12 0 "x"
               OpMemberName %12 1 "y"
               OpName %14 ""
               OpMemberDecorate %12 0 Offset 0
               OpMemberDecorate %12 1 Offset 4
               OpDecorate %12 Block
               OpDecorate %14 DescriptorSet 0
               OpDecorate %14 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 9
         %11 = OpConstant %6 10
         %50 = OpConstant %6 0
         %51 = OpConstant %6 1
         %12 = OpTypeStruct %6 %6
         %13 = OpTypePointer Uniform %12
         %52 = OpTypePointer Uniform %6
         %14 = OpVariable %13 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
               OpStore %8 %9
               OpStore %10 %11
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;
  protobufs::UniformBufferElementDescriptor blockname_9 =
      FactManager::MakeUniformBufferElementDescriptor(14, {0});
  protobufs::UniformBufferElementDescriptor blockname_10 =
      FactManager::MakeUniformBufferElementDescriptor(14, {1});

  fact_manager.AddUniformIntValueFact(32, true, {9}, blockname_9);
  fact_manager.AddUniformIntValueFact(32, true, {10}, blockname_10);

  // The constant ids for 9 and 10 are 9 and 11 respectively
  protobufs::IdUseDescriptor use_of_9_in_store =
      module_navigation::MakeIdUseDescriptor(9, SpvOpStore, 1, 10, 0);
  protobufs::IdUseDescriptor use_of_11_in_store =
      module_navigation::MakeIdUseDescriptor(11, SpvOpStore, 1, 10, 1);

  // These are right:
  ASSERT_TRUE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_9, 100, 101),
      context.get(), fact_manager));
  ASSERT_TRUE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_store, blockname_10, 102, 103),
      context.get(), fact_manager));

  // These are wrong because the constants do not match the facts about
  // uniforms.
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_11_in_store, blockname_9, 100, 101),
      context.get(), fact_manager));
  ASSERT_FALSE(transformation::IsApplicable(
      transformation::MakeTransformationReplaceConstantWithUniform(
          use_of_9_in_store, blockname_10, 102, 103),
      context.get(), fact_manager));
}

TEST(TransformationMoveBlockDownTest, ComplexReplacements) {
  // The following GLSL was the basis for this test:

  // #version 450
  //
  // struct T {
  //   float a[5]; // [1.0, 1.5, 1.75, 1.875, 1.9375]
  //   ivec4 b; // (1, 2, 3, 4)
  //   vec3 c; // (2.0, 2.5, 2.75)
  //   uint d; // 42u
  //   bool e; // Not used in test
  // };
  //
  // uniform block {
  //   T f;
  //   int g; // 22
  //   uvec2 h; // (100u, 200u)
  // };
  //
  // void main()
  // {
  //   T myT;
  //
  //   myT.a[0] = 1.9375;
  //   myT.a[1] = 1.875;
  //   myT.a[2] = 1.75;
  //   myT.a[3] = 1.5;
  //   myT.a[4] = 1.0;
  //
  //   myT.b.x = 4;
  //   myT.b.y = 3;
  //   myT.b.z = 2;
  //   myT.b.w = 1;
  //
  //   myT.b.r = 22;
  //
  //   myT.c[0] = 2.75;
  //   myT.c[0] = 2.5;
  //   myT.c[0] = 2.0;
  //
  //   myT.d = 42u;
  //   myT.d = 100u;
  //   myT.d = 200u;
  //
  //   myT.e = true; // No attempt to replace 'true' by a uniform value
  //
  // }

  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %14 "T"
               OpMemberName %14 0 "a"
               OpMemberName %14 1 "b"
               OpMemberName %14 2 "c"
               OpMemberName %14 3 "d"
               OpMemberName %14 4 "e"
               OpName %16 "myT"
               OpName %61 "T"
               OpMemberName %61 0 "a"
               OpMemberName %61 1 "b"
               OpMemberName %61 2 "c"
               OpMemberName %61 3 "d"
               OpMemberName %61 4 "e"
               OpName %63 "block"
               OpMemberName %63 0 "f"
               OpMemberName %63 1 "g"
               OpMemberName %63 2 "h"
               OpName %65 ""
               OpDecorate %60 ArrayStride 16
               OpMemberDecorate %61 0 Offset 0
               OpMemberDecorate %61 1 Offset 80
               OpMemberDecorate %61 2 Offset 96
               OpMemberDecorate %61 3 Offset 108
               OpMemberDecorate %61 4 Offset 112
               OpMemberDecorate %63 0 Offset 0
               OpMemberDecorate %63 1 Offset 128
               OpMemberDecorate %63 2 Offset 136
               OpDecorate %63 Block
               OpDecorate %65 DescriptorSet 0
               OpDecorate %65 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeInt 32 0
          %8 = OpConstant %7 5
          %9 = OpTypeArray %6 %8
         %10 = OpTypeInt 32 1
         %11 = OpTypeVector %10 4
         %12 = OpTypeVector %6 3
         %13 = OpTypeBool
         %14 = OpTypeStruct %9 %11 %12 %7 %13
         %15 = OpTypePointer Function %14
         %17 = OpConstant %10 0
         %18 = OpConstant %6 1.9375
         %19 = OpTypePointer Function %6
         %21 = OpConstant %10 1
         %22 = OpConstant %6 1.875
         %24 = OpConstant %10 2
         %25 = OpConstant %6 1.75
         %27 = OpConstant %10 3
         %28 = OpConstant %6 1.5
         %30 = OpConstant %10 4
         %31 = OpConstant %6 1
         %33 = OpConstant %7 0
         %34 = OpTypePointer Function %10
         %36 = OpConstant %7 1
         %38 = OpConstant %7 2
         %40 = OpConstant %7 3
         %42 = OpConstant %10 22
         %44 = OpConstant %6 2.75
         %46 = OpConstant %6 2.5
         %48 = OpConstant %6 2
         %50 = OpConstant %7 42
         %51 = OpTypePointer Function %7
         %53 = OpConstant %7 100
         %55 = OpConstant %7 200
         %57 = OpConstantTrue %13
         %58 = OpTypePointer Function %13
         %60 = OpTypeArray %6 %8
         %61 = OpTypeStruct %60 %11 %12 %7 %7
         %62 = OpTypeVector %7 2
         %63 = OpTypeStruct %61 %10 %62
         %64 = OpTypePointer Uniform %63
        %100 = OpTypePointer Uniform %10
        %101 = OpTypePointer Uniform %7
        %102 = OpTypePointer Uniform %6
        %103 = OpTypePointer Uniform %13
         %65 = OpVariable %64 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %16 = OpVariable %15 Function
         %20 = OpAccessChain %19 %16 %17 %17
               OpStore %20 %18
         %23 = OpAccessChain %19 %16 %17 %21
               OpStore %23 %22
         %26 = OpAccessChain %19 %16 %17 %24
               OpStore %26 %25
         %29 = OpAccessChain %19 %16 %17 %27
               OpStore %29 %28
         %32 = OpAccessChain %19 %16 %17 %30
               OpStore %32 %31
         %35 = OpAccessChain %34 %16 %21 %33
               OpStore %35 %30
         %37 = OpAccessChain %34 %16 %21 %36
               OpStore %37 %27
         %39 = OpAccessChain %34 %16 %21 %38
               OpStore %39 %24
         %41 = OpAccessChain %34 %16 %21 %40
               OpStore %41 %21
         %43 = OpAccessChain %34 %16 %21 %33
               OpStore %43 %42
         %45 = OpAccessChain %19 %16 %24 %33
               OpStore %45 %44
         %47 = OpAccessChain %19 %16 %24 %33
               OpStore %47 %46
         %49 = OpAccessChain %19 %16 %24 %33
               OpStore %49 %48
         %52 = OpAccessChain %51 %16 %27
               OpStore %52 %50
         %54 = OpAccessChain %51 %16 %27
               OpStore %54 %53
         %56 = OpAccessChain %51 %16 %27
               OpStore %56 %55
         %59 = OpAccessChain %58 %16 %30
               OpStore %59 %57
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  CheckValid(env, context.get());

  FactManager fact_manager;

  const float float_array_values[5] = {1.0, 1.5, 1.75, 1.875, 1.9375};
  uint32_t float_array_data[5];
  memcpy(&float_array_data, &float_array_values, sizeof(float_array_values));

  const float float_vector_values[3] = {2.0, 2.5, 2.75};
  uint32_t float_vector_data[3];
  memcpy(&float_vector_data, &float_vector_values, sizeof(float_vector_values));

  protobufs::UniformBufferElementDescriptor uniform_f_a_0 =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 0, 0});
  protobufs::UniformBufferElementDescriptor uniform_f_a_1 =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 0, 1});
  protobufs::UniformBufferElementDescriptor uniform_f_a_2 =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 0, 2});
  protobufs::UniformBufferElementDescriptor uniform_f_a_3 =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 0, 3});
  protobufs::UniformBufferElementDescriptor uniform_f_a_4 =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 0, 4});

  protobufs::UniformBufferElementDescriptor uniform_f_b_x =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 1, 0});
  protobufs::UniformBufferElementDescriptor uniform_f_b_y =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 1, 1});
  protobufs::UniformBufferElementDescriptor uniform_f_b_z =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 1, 2});
  protobufs::UniformBufferElementDescriptor uniform_f_b_w =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 1, 3});

  protobufs::UniformBufferElementDescriptor uniform_f_c_x =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 2, 0});
  protobufs::UniformBufferElementDescriptor uniform_f_c_y =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 2, 1});
  protobufs::UniformBufferElementDescriptor uniform_f_c_z =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 2, 2});

  protobufs::UniformBufferElementDescriptor uniform_f_d =
      FactManager::MakeUniformBufferElementDescriptor(65, {0, 3});

  protobufs::UniformBufferElementDescriptor uniform_g =
      FactManager::MakeUniformBufferElementDescriptor(65, {1});

  protobufs::UniformBufferElementDescriptor uniform_h_x =
      FactManager::MakeUniformBufferElementDescriptor(65, {2, 0});
  protobufs::UniformBufferElementDescriptor uniform_h_y =
      FactManager::MakeUniformBufferElementDescriptor(65, {2, 1});

  fact_manager.AddUniformFloatValueFact(32, {float_array_data[0]},
                                        uniform_f_a_0);
  fact_manager.AddUniformFloatValueFact(32, {float_array_data[1]},
                                        uniform_f_a_1);
  fact_manager.AddUniformFloatValueFact(32, {float_array_data[2]},
                                        uniform_f_a_2);
  fact_manager.AddUniformFloatValueFact(32, {float_array_data[3]},
                                        uniform_f_a_3);
  fact_manager.AddUniformFloatValueFact(32, {float_array_data[4]},
                                        uniform_f_a_4);

  fact_manager.AddUniformIntValueFact(32, true, {1}, uniform_f_b_x);
  fact_manager.AddUniformIntValueFact(32, true, {2}, uniform_f_b_y);
  fact_manager.AddUniformIntValueFact(32, true, {3}, uniform_f_b_z);
  fact_manager.AddUniformIntValueFact(32, true, {4}, uniform_f_b_w);

  fact_manager.AddUniformFloatValueFact(32, {float_vector_data[0]},
                                        uniform_f_c_x);
  fact_manager.AddUniformFloatValueFact(32, {float_vector_data[1]},
                                        uniform_f_c_y);
  fact_manager.AddUniformFloatValueFact(32, {float_vector_data[2]},
                                        uniform_f_c_z);

  fact_manager.AddUniformIntValueFact(32, false, {42}, uniform_f_d);

  fact_manager.AddUniformIntValueFact(32, true, {22}, uniform_g);

  fact_manager.AddUniformIntValueFact(32, false, {100}, uniform_h_x);
  fact_manager.AddUniformIntValueFact(32, false, {200}, uniform_h_y);

  std::vector<protobufs::TransformationReplaceConstantWithUniform>
      transformations;

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(18, SpvOpStore, 1, 20, 0),
          uniform_f_a_4, 200, 201));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(22, SpvOpStore, 1, 23, 0),
          uniform_f_a_3, 202, 203));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(25, SpvOpStore, 1, 26, 0),
          uniform_f_a_2, 204, 205));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(28, SpvOpStore, 1, 29, 0),
          uniform_f_a_1, 206, 207));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(31, SpvOpStore, 1, 32, 0),
          uniform_f_a_0, 208, 209));

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(30, SpvOpStore, 1, 35, 0),
          uniform_f_b_w, 210, 211));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(27, SpvOpStore, 1, 37, 0),
          uniform_f_b_z, 212, 213));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(24, SpvOpStore, 1, 39, 0),
          uniform_f_b_y, 214, 215));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(21, SpvOpStore, 1, 41, 0),
          uniform_f_b_x, 216, 217));

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(44, SpvOpStore, 1, 45, 0),
          uniform_f_c_z, 220, 221));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(46, SpvOpStore, 1, 47, 0),
          uniform_f_c_y, 222, 223));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(48, SpvOpStore, 1, 49, 0),
          uniform_f_c_x, 224, 225));

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(50, SpvOpStore, 1, 52, 0),
          uniform_f_d, 226, 227));

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(53, SpvOpStore, 1, 54, 0),
          uniform_h_x, 228, 229));
  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(55, SpvOpStore, 1, 56, 0),
          uniform_h_y, 230, 231));

  transformations.emplace_back(
      transformation::MakeTransformationReplaceConstantWithUniform(
          module_navigation::MakeIdUseDescriptor(42, SpvOpStore, 1, 43, 0),
          uniform_g, 218, 219));

  for (auto& transformation : transformations) {
    ASSERT_TRUE(transformation::IsApplicable(transformation, context.get(),
                                             fact_manager));
    transformation::Apply(transformation, context.get(), &fact_manager);
    CheckValid(env, context.get());
  }

  std::string after = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
               OpName %14 "T"
               OpMemberName %14 0 "a"
               OpMemberName %14 1 "b"
               OpMemberName %14 2 "c"
               OpMemberName %14 3 "d"
               OpMemberName %14 4 "e"
               OpName %16 "myT"
               OpName %61 "T"
               OpMemberName %61 0 "a"
               OpMemberName %61 1 "b"
               OpMemberName %61 2 "c"
               OpMemberName %61 3 "d"
               OpMemberName %61 4 "e"
               OpName %63 "block"
               OpMemberName %63 0 "f"
               OpMemberName %63 1 "g"
               OpMemberName %63 2 "h"
               OpName %65 ""
               OpDecorate %60 ArrayStride 16
               OpMemberDecorate %61 0 Offset 0
               OpMemberDecorate %61 1 Offset 80
               OpMemberDecorate %61 2 Offset 96
               OpMemberDecorate %61 3 Offset 108
               OpMemberDecorate %61 4 Offset 112
               OpMemberDecorate %63 0 Offset 0
               OpMemberDecorate %63 1 Offset 128
               OpMemberDecorate %63 2 Offset 136
               OpDecorate %63 Block
               OpDecorate %65 DescriptorSet 0
               OpDecorate %65 Binding 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeInt 32 0
          %8 = OpConstant %7 5
          %9 = OpTypeArray %6 %8
         %10 = OpTypeInt 32 1
         %11 = OpTypeVector %10 4
         %12 = OpTypeVector %6 3
         %13 = OpTypeBool
         %14 = OpTypeStruct %9 %11 %12 %7 %13
         %15 = OpTypePointer Function %14
         %17 = OpConstant %10 0
         %18 = OpConstant %6 1.9375
         %19 = OpTypePointer Function %6
         %21 = OpConstant %10 1
         %22 = OpConstant %6 1.875
         %24 = OpConstant %10 2
         %25 = OpConstant %6 1.75
         %27 = OpConstant %10 3
         %28 = OpConstant %6 1.5
         %30 = OpConstant %10 4
         %31 = OpConstant %6 1
         %33 = OpConstant %7 0
         %34 = OpTypePointer Function %10
         %36 = OpConstant %7 1
         %38 = OpConstant %7 2
         %40 = OpConstant %7 3
         %42 = OpConstant %10 22
         %44 = OpConstant %6 2.75
         %46 = OpConstant %6 2.5
         %48 = OpConstant %6 2
         %50 = OpConstant %7 42
         %51 = OpTypePointer Function %7
         %53 = OpConstant %7 100
         %55 = OpConstant %7 200
         %57 = OpConstantTrue %13
         %58 = OpTypePointer Function %13
         %60 = OpTypeArray %6 %8
         %61 = OpTypeStruct %60 %11 %12 %7 %7
         %62 = OpTypeVector %7 2
         %63 = OpTypeStruct %61 %10 %62
         %64 = OpTypePointer Uniform %63
        %100 = OpTypePointer Uniform %10
        %101 = OpTypePointer Uniform %7
        %102 = OpTypePointer Uniform %6
        %103 = OpTypePointer Uniform %13
         %65 = OpVariable %64 Uniform
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %16 = OpVariable %15 Function
         %20 = OpAccessChain %19 %16 %17 %17
        %200 = OpAccessChain %102 %65 %17 %17 %30
        %201 = OpLoad %6 %200
               OpStore %20 %201
         %23 = OpAccessChain %19 %16 %17 %21
        %202 = OpAccessChain %102 %65 %17 %17 %27
        %203 = OpLoad %6 %202
               OpStore %23 %203
         %26 = OpAccessChain %19 %16 %17 %24
        %204 = OpAccessChain %102 %65 %17 %17 %24
        %205 = OpLoad %6 %204
               OpStore %26 %205
         %29 = OpAccessChain %19 %16 %17 %27
        %206 = OpAccessChain %102 %65 %17 %17 %21
        %207 = OpLoad %6 %206
               OpStore %29 %207
         %32 = OpAccessChain %19 %16 %17 %30
        %208 = OpAccessChain %102 %65 %17 %17 %17
        %209 = OpLoad %6 %208
               OpStore %32 %209
         %35 = OpAccessChain %34 %16 %21 %33
        %210 = OpAccessChain %100 %65 %17 %21 %27
        %211 = OpLoad %10 %210
               OpStore %35 %211
         %37 = OpAccessChain %34 %16 %21 %36
        %212 = OpAccessChain %100 %65 %17 %21 %24
        %213 = OpLoad %10 %212
               OpStore %37 %213
         %39 = OpAccessChain %34 %16 %21 %38
        %214 = OpAccessChain %100 %65 %17 %21 %21
        %215 = OpLoad %10 %214
               OpStore %39 %215
         %41 = OpAccessChain %34 %16 %21 %40
        %216 = OpAccessChain %100 %65 %17 %21 %17
        %217 = OpLoad %10 %216
               OpStore %41 %217
         %43 = OpAccessChain %34 %16 %21 %33
        %218 = OpAccessChain %100 %65 %21
        %219 = OpLoad %10 %218
               OpStore %43 %219
         %45 = OpAccessChain %19 %16 %24 %33
        %220 = OpAccessChain %102 %65 %17 %24 %24
        %221 = OpLoad %6 %220
               OpStore %45 %221
         %47 = OpAccessChain %19 %16 %24 %33
        %222 = OpAccessChain %102 %65 %17 %24 %21
        %223 = OpLoad %6 %222
               OpStore %47 %223
         %49 = OpAccessChain %19 %16 %24 %33
        %224 = OpAccessChain %102 %65 %17 %24 %17
        %225 = OpLoad %6 %224
               OpStore %49 %225
         %52 = OpAccessChain %51 %16 %27
        %226 = OpAccessChain %101 %65 %17 %27
        %227 = OpLoad %7 %226
               OpStore %52 %227
         %54 = OpAccessChain %51 %16 %27
        %228 = OpAccessChain %101 %65 %24 %17
        %229 = OpLoad %7 %228
               OpStore %54 %229
         %56 = OpAccessChain %51 %16 %27
        %230 = OpAccessChain %101 %65 %24 %21
        %231 = OpLoad %7 %230
               OpStore %56 %231
         %59 = OpAccessChain %58 %16 %30
               OpStore %59 %57
               OpReturn
               OpFunctionEnd
  )";

  CheckEqual(env, after, context.get());
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
