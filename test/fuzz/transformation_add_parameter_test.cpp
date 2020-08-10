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

#include "source/fuzz/transformation_add_parameter.h"
#include "test/fuzz/fuzz_test_util.h"

#include "source/fuzz/fuzzer_pass_add_parameters.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/transformation_add_global_variable.h"

namespace spvtools {
namespace fuzz {
namespace {
/*
TEST(TransformationAddParameterTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
         %11 = OpTypeInt 32 1
         %16 = OpTypeFloat 32
          %3 = OpTypeFunction %2
          %6 = OpTypeFunction %7 %7
          %8 = OpConstant %11 23
         %12 = OpConstantTrue %7
         %15 = OpTypeFunction %2 %16
         %24 = OpTypeFunction %2 %16 %7
         %31 = OpTypeStruct %7 %11
         %32 = OpConstant %16 23
         %33 = OpConstantComposite %31 %12 %8
         %41 = OpTypeStruct %11 %16
         %42 = OpConstantComposite %41 %8 %32
         %43 = OpTypeFunction %2 %41
         %44 = OpTypeFunction %2 %41 %7
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpFunctionCall %7 %9 %12
               OpReturn
               OpFunctionEnd

          ; adjust type of the function in-place
          %9 = OpFunction %7 None %6
         %14 = OpFunctionParameter %7
         %10 = OpLabel
               OpReturnValue %12
               OpFunctionEnd

         ; reuse an existing function type
         %17 = OpFunction %2 None %15
         %18 = OpFunctionParameter %16
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %21 = OpFunctionParameter %16
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
         %25 = OpFunction %2 None %24
         %26 = OpFunctionParameter %16
         %27 = OpFunctionParameter %7
         %28 = OpLabel
               OpReturn
               OpFunctionEnd

         ; create a new function type
         %29 = OpFunction %2 None %3
         %30 = OpLabel
               OpReturn
               OpFunctionEnd

         ; don't adjust the type of the function if it creates a duplicate
         %34 = OpFunction %2 None %43
         %35 = OpFunctionParameter %41
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
         %37 = OpFunction %2 None %44
         %38 = OpFunctionParameter %41
         %39 = OpFunctionParameter %7
         %40 = OpLabel
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

  // Can't modify entry point function.
  ASSERT_FALSE(TransformationAddParameter(4, 60, 12, 61)
                   .IsApplicable(context.get(), transformation_context));

  // There is no function with result id 29.
  ASSERT_FALSE(TransformationAddParameter(60, 60, 8, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 14, 8, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id is not fresh.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 8, 14)
                   .IsApplicable(context.get(), transformation_context));

  // Function type id and parameter type id are equal.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 8, 60)
                   .IsApplicable(context.get(), transformation_context));

  // Parameter's initializer doesn't exist.
  ASSERT_FALSE(TransformationAddParameter(9, 60, 60, 61)
                   .IsApplicable(context.get(), transformation_context));

  // Correct transformations.
  {
    TransformationAddParameter correct(9, 60, 8, 61);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(60));
  }
  {
    TransformationAddParameter correct(17, 62, 12, 63);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(62));
  }
  {
    TransformationAddParameter correct(29, 64, 33, 65);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(64));
  }
  {
    TransformationAddParameter correct(34, 66, 12, 67);
    ASSERT_TRUE(correct.IsApplicable(context.get(), transformation_context));
    correct.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.IdIsIrrelevant(66));
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %7 = OpTypeBool
         %11 = OpTypeInt 32 1
         %16 = OpTypeFloat 32
          %3 = OpTypeFunction %2
          %8 = OpConstant %11 23
         %12 = OpConstantTrue %7
         %15 = OpTypeFunction %2 %16
         %24 = OpTypeFunction %2 %16 %7
         %31 = OpTypeStruct %7 %11
         %32 = OpConstant %16 23
         %33 = OpConstantComposite %31 %12 %8
         %41 = OpTypeStruct %11 %16
         %42 = OpConstantComposite %41 %8 %32
         %44 = OpTypeFunction %2 %41 %7
          %6 = OpTypeFunction %7 %7 %11
         %65 = OpTypeFunction %2 %31
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %13 = OpFunctionCall %7 %9 %12 %8
               OpReturn
               OpFunctionEnd

          ; adjust type of the function in-place
          %9 = OpFunction %7 None %6
         %14 = OpFunctionParameter %7
         %60 = OpFunctionParameter %11
         %10 = OpLabel
               OpReturnValue %12
               OpFunctionEnd

         ; reuse an existing function type
         %17 = OpFunction %2 None %24
         %18 = OpFunctionParameter %16
         %62 = OpFunctionParameter %7
         %19 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %21 = OpFunctionParameter %16
         %22 = OpLabel
               OpReturn
               OpFunctionEnd
         %25 = OpFunction %2 None %24
         %26 = OpFunctionParameter %16
         %27 = OpFunctionParameter %7
         %28 = OpLabel
               OpReturn
               OpFunctionEnd

         ; create a new function type
         %29 = OpFunction %2 None %65
         %64 = OpFunctionParameter %31
         %30 = OpLabel
               OpReturn
               OpFunctionEnd

         ; don't adjust the type of the function if it creates a duplicate
         %34 = OpFunction %2 None %44
         %35 = OpFunctionParameter %41
         %66 = OpFunctionParameter %7
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
         %37 = OpFunction %2 None %44
         %38 = OpFunctionParameter %41
         %39 = OpFunctionParameter %7
         %40 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}*/

TEST(TransformationAddParameterTest, PointerTypeTest) {
  std::string shader = R"(
  OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %6 "nothing("
               OpName %12 "add(i1;"
               OpName %11 "a"
               OpName %21 "i1"
               OpName %24 "f1"
               OpName %28 "u1"
               OpName %30 "i2"
               OpName %31 "param"
               OpDecorate %12 RelaxedPrecision
               OpDecorate %11 RelaxedPrecision
               OpDecorate %15 RelaxedPrecision
               OpDecorate %17 RelaxedPrecision
               OpDecorate %21 RelaxedPrecision
               OpDecorate %28 RelaxedPrecision
               OpDecorate %30 RelaxedPrecision
               OpDecorate %32 RelaxedPrecision
               OpDecorate %33 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %8 %9
         %16 = OpConstant %8 2
         %22 = OpTypeFloat 32
         %23 = OpTypePointer Function %22
         %25 = OpConstant %22 3
         %26 = OpTypeInt 32 0
         %27 = OpTypePointer Function %26
         %29 = OpConstant %26 2
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %21 = OpVariable %9 Function
         %24 = OpVariable %23 Function
         %28 = OpVariable %27 Function
         %30 = OpVariable %9 Function
         %31 = OpVariable %9 Function
         %20 = OpFunctionCall %2 %6
               OpStore %21 %16
               OpStore %24 %25
               OpStore %28 %29
         %32 = OpLoad %8 %21
               OpStore %31 %32
         %33 = OpFunctionCall %8 %12 %31
               OpStore %30 %33
               OpReturn
               OpFunctionEnd
          %6 = OpFunction %2 None %3
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %8 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %15 = OpLoad %8 %11
         %17 = OpIAdd %8 %15 %16
               OpReturnValue %17
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

  auto prng = MakeUnique<PseudoRandomGenerator>(0);
  FuzzerContext fuzzer_context(prng.get(), 100);
  protobufs::TransformationSequence transformation_sequence;

  FuzzerPassAddParameters fuzzer_pass(context.get(), &transformation_context,
                                      &fuzzer_context,
                                      &transformation_sequence);

  for (int i = 0; i < 10; i++) {
    fuzzer_pass.Apply();
  }

  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled;

  /*
  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);
  TransformationAddGlobalVariable add_global_variable(
      60, 22, SpvStorageClassPrivate, 13, true);
  add_global_variable.Apply(context.get(), &transformation_context);

  TransformationAddParameter correct(10, 50, 22, 51);
  correct.Apply(context.get(), &transformation_context);
  std::vector<uint32_t> actual_binary;
  context.get()->module()->ToBinary(&actual_binary, false);
  SpirvTools t(env);
  std::string actual_disassembled;
  t.Disassemble(actual_binary, &actual_disassembled, kFuzzDisassembleOption);
  std::cout << actual_disassembled;
   */
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
