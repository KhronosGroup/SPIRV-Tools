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

#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_access_chain.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAccessChainTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeFloat 32
          %7 = OpTypeVector %6 2
         %50 = OpTypeMatrix %7 2
         %70 = OpTypePointer Function %7
         %71 = OpTypePointer Function %50
          %8 = OpTypeStruct %7 %6
          %9 = OpTypePointer Function %8
         %10 = OpTypeInt 32 1
         %11 = OpTypePointer Function %10
         %12 = OpTypeFunction %10 %9 %11
         %17 = OpConstant %10 0
         %18 = OpTypeInt 32 0
         %19 = OpConstant %18 0
         %20 = OpTypePointer Function %6
         %29 = OpConstant %6 0
         %30 = OpConstant %6 1
         %31 = OpConstantComposite %7 %29 %30
         %32 = OpConstant %6 2
         %33 = OpConstantComposite %8 %31 %32
         %35 = OpConstant %10 10
         %51 = OpConstant %18 10
         %80 = OpConstant %18 0
         %81 = OpConstant %10 1
         %82 = OpConstant %18 2
         %83 = OpConstant %10 3
         %84 = OpConstant %18 4
         %85 = OpConstant %10 5
         %52 = OpTypeArray %50 %51
         %53 = OpTypePointer Private %52
         %45 = OpUndef %9
         %46 = OpConstantNull %9
         %47 = OpTypePointer Private %8
         %48 = OpVariable %47 Private
         %54 = OpVariable %53 Private
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %28 = OpVariable %9 Function
         %34 = OpVariable %11 Function
         %36 = OpVariable %9 Function
         %38 = OpVariable %11 Function
         %44 = OpCopyObject %9 %36
               OpStore %28 %33
               OpStore %34 %35
         %37 = OpLoad %8 %28
               OpStore %36 %37
         %39 = OpLoad %10 %34
               OpStore %38 %39
         %40 = OpFunctionCall %10 %15 %36 %38
         %41 = OpLoad %10 %34
         %42 = OpIAdd %10 %41 %40
               OpStore %34 %42
               OpReturn
               OpFunctionEnd
         %15 = OpFunction %10 None %12
         %13 = OpFunctionParameter %9
         %14 = OpFunctionParameter %11
         %16 = OpLabel
         %21 = OpAccessChain %20 %13 %17 %19
         %43 = OpCopyObject %9 %13
         %22 = OpLoad %6 %21
         %23 = OpConvertFToS %10 %22
         %24 = OpLoad %10 %14
         %25 = OpIAdd %10 %23 %24
               OpReturnValue %25
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  // Types:
  // Ptr | Pointee | Storage class | GLSL for pointee    | Ids of this type
  // ----+---------+---------------+---------------------+------------------
  //  9  |    8    | Function      | struct(vec2, float) | 28, 36, 44, 13, 43
  // 11  |   10    | Function      | int                 | 34, 38, 14
  // 20  |    6    | Function      | float               | -
  // 53  |   52    | Private       | mat2x2[10]          | 54
  // 47  |    8    | Private       | struct(vec2, float) | 48
  // 70  |    7    | Function      | vec2                | -
  // 71  |   59    | Function      | mat2x2              | -

  // Indices 0-5 are in ids 80-85

  FactManager fact_manager;
  fact_manager.AddFactValueOfPointeeIsIrrelevant(54);

  // Bad: id is not fresh

  // Bad: pointer id does not exist

  // Bad: pointer id is not a pointer

  // Bad: index id does not exist

  // Bad: index id is not a constant

  // Bad: too many indices

  // Bad: index id is out of bounds

  // Bad: attempt to insert before variable


  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3179)

  {
    TransformationAccessChain transformation(100, 43, { 80 }, MakeInstructionDescriptor(24, SpvOpLoad, 0));
    ASSERT_TRUE(transformation.IsApplicable(context.get(), fact_manager));
    transformation.Apply(context.get(), &fact_manager);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_FALSE(fact_manager.PointeeValueIsIrrelevant(100));
  }


  std::string after_transformation = R"(
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
  FAIL();  // Remove once test is implemented
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
