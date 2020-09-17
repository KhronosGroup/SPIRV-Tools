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

#include "source/fuzz/transformation_add_copy_memory.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddCopyMemoryTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %19 RelaxedPrecision
               OpMemberDecorate %66 0 RelaxedPrecision
               OpDecorate %69 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpTypePointer Function %6
         %78 = OpTypePointer Private %6
          %8 = OpTypeFunction %6 %7
         %17 = OpTypeInt 32 1
         %18 = OpTypePointer Function %17
         %79 = OpTypePointer Private %17
         %20 = OpConstant %17 0
         %21 = OpTypeFloat 32
         %22 = OpTypePointer Function %21
         %80 = OpTypePointer Private %21
         %24 = OpConstant %21 0
         %25 = OpConstantFalse %6
         %32 = OpConstantTrue %6
         %33 = OpTypeVector %21 4
         %34 = OpTypePointer Function %33
         %81 = OpTypePointer Private %33
         %36 = OpConstantComposite %33 %24 %24 %24 %24
         %37 = OpTypeMatrix %33 4
         %84 = OpConstantComposite %37 %36 %36 %36 %36
         %38 = OpTypePointer Function %37
         %82 = OpTypePointer Private %37
         %44 = OpConstant %21 1
         %66 = OpTypeStruct %17 %21 %6 %33 %37
         %85 = OpConstantComposite %66 %20 %24 %25 %36 %84
         %67 = OpTypePointer Function %66
         %83 = OpTypePointer Private %66
         %86 = OpVariable %79 Private %20
         %87 = OpUndef %79
         %88 = OpConstantNull %79
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %19 = OpVariable %18 Function
         %23 = OpVariable %22 Function
         %26 = OpVariable %7 Function
         %30 = OpVariable %7 Function
         %35 = OpVariable %34 Function
         %39 = OpVariable %38 Function
         %68 = OpVariable %67 Function
               OpStore %19 %20
               OpStore %23 %24
               OpStore %26 %25
         %27 = OpFunctionCall %6 %10 %26
               OpSelectionMerge %29 None
               OpBranchConditional %27 %28 %31
         %28 = OpLabel
               OpBranch %29
         %31 = OpLabel
               OpBranch %29
         %76 = OpLabel
         %77 = OpLogicalEqual %6 %25 %32
               OpBranch %29
         %29 = OpLabel
         %75 = OpPhi %6 %25 %31 %32 %28 %77 %76
               OpStore %30 %75
         %40 = OpLoad %33 %35
         %41 = OpLoad %33 %35
         %42 = OpLoad %33 %35
         %43 = OpLoad %33 %35
         %45 = OpCompositeExtract %21 %40 0
         %46 = OpCompositeExtract %21 %40 1
         %47 = OpCompositeExtract %21 %40 2
         %48 = OpCompositeExtract %21 %40 3
         %49 = OpCompositeExtract %21 %41 0
         %50 = OpCompositeExtract %21 %41 1
         %51 = OpCompositeExtract %21 %41 2
         %52 = OpCompositeExtract %21 %41 3
         %53 = OpCompositeExtract %21 %42 0
         %54 = OpCompositeExtract %21 %42 1
         %55 = OpCompositeExtract %21 %42 2
         %56 = OpCompositeExtract %21 %42 3
         %57 = OpCompositeExtract %21 %43 0
         %58 = OpCompositeExtract %21 %43 1
         %59 = OpCompositeExtract %21 %43 2
         %60 = OpCompositeExtract %21 %43 3
         %61 = OpCompositeConstruct %33 %45 %46 %47 %48
         %62 = OpCompositeConstruct %33 %49 %50 %51 %52
         %63 = OpCompositeConstruct %33 %53 %54 %55 %56
         %64 = OpCompositeConstruct %33 %57 %58 %59 %60
         %65 = OpCompositeConstruct %37 %61 %62 %63 %64
               OpStore %39 %65
         %69 = OpLoad %17 %19
         %70 = OpLoad %21 %23
         %71 = OpLoad %6 %30
         %72 = OpLoad %33 %35
         %73 = OpLoad %37 %39
         %74 = OpCompositeConstruct %66 %69 %70 %71 %72 %73
               OpStore %68 %74
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %6 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
         %13 = OpLoad %6 %9
               OpStore %12 %13
         %14 = OpLoad %6 %12
               OpReturnValue %14
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

  // Target id is not fresh (59).
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 59, 19,
                   SpvStorageClassPrivate, 20)
                   .IsApplicable(context.get(), transformation_context));

  // Instruction descriptor is invalid (id 89 is undefined).
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(89, SpvOpVariable, 0), 89, 19,
                   SpvStorageClassPrivate, 20)
                   .IsApplicable(context.get(), transformation_context));

  // Cannot insert OpCopyMemory before OpPhi.
  ASSERT_FALSE(
      TransformationAddCopyMemory(MakeInstructionDescriptor(75, SpvOpPhi, 0),
                                  89, 19, SpvStorageClassPrivate, 20)
          .IsApplicable(context.get(), transformation_context));

  // Source instruction is invalid.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 89, 76,
                   SpvStorageClassPrivate, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Source instruction's type doesn't exist.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 89, 5,
                   SpvStorageClassPrivate, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Source instruction's type is invalid.
  ASSERT_FALSE(
      TransformationAddCopyMemory(MakeInstructionDescriptor(41, SpvOpLoad, 0),
                                  89, 40, SpvStorageClassPrivate, 0)
          .IsApplicable(context.get(), transformation_context));

  // Source instruction is OpUndef.
  ASSERT_FALSE(
      TransformationAddCopyMemory(MakeInstructionDescriptor(41, SpvOpLoad, 0),
                                  89, 87, SpvStorageClassPrivate, 0)
          .IsApplicable(context.get(), transformation_context));

  // Source instruction is OpConstantNull.
  ASSERT_FALSE(
      TransformationAddCopyMemory(MakeInstructionDescriptor(41, SpvOpLoad, 0),
                                  89, 88, SpvStorageClassPrivate, 0)
          .IsApplicable(context.get(), transformation_context));

  // Storage class is invalid.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 89, 19,
                   SpvStorageClassWorkgroup, 20)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer is 0.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 89, 19,
                   SpvStorageClassPrivate, 0)
                   .IsApplicable(context.get(), transformation_context));

  // Initializer has wrong type.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(27, SpvOpFunctionCall, 0), 89, 19,
                   SpvStorageClassPrivate, 25)
                   .IsApplicable(context.get(), transformation_context));

  // Source and target instructions are in different functions.
  ASSERT_FALSE(
      TransformationAddCopyMemory(MakeInstructionDescriptor(13, SpvOpLoad, 0),
                                  89, 19, SpvStorageClassPrivate, 20)
          .IsApplicable(context.get(), transformation_context));

  // Source instruction doesn't dominate the target instruction.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(77, SpvOpLogicalEqual, 0), 89, 19,
                   SpvStorageClassPrivate, 20)
                   .IsApplicable(context.get(), transformation_context));

  // Source and target instructions are the same.
  ASSERT_FALSE(TransformationAddCopyMemory(
                   MakeInstructionDescriptor(19, SpvOpVariable, 0), 89, 19,
                   SpvStorageClassPrivate, 20)
                   .IsApplicable(context.get(), transformation_context));

  // Correct transformations.
  uint32_t fresh_id = 89;
  auto descriptor = MakeInstructionDescriptor(27, SpvOpFunctionCall, 0);
  std::vector<uint32_t> source_ids = {19, 23, 26, 30, 35, 39, 68, 86};
  std::vector<uint32_t> initializers = {20, 24, 25, 25, 36, 84, 85, 20};
  std::vector<SpvStorageClass> storage_classes = {SpvStorageClassPrivate,
                                                  SpvStorageClassFunction};
  for (size_t i = 0, n = source_ids.size(); i < n; ++i) {
    TransformationAddCopyMemory transformation(
        descriptor, fresh_id, source_ids[i],
        storage_classes[i % storage_classes.size()], initializers[i]);
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
    ASSERT_TRUE(IsValid(env, context.get()));
    ASSERT_TRUE(fact_manager.PointeeValueIsIrrelevant(fresh_id, context.get()));
    fresh_id++;
  }

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %19 RelaxedPrecision
               OpMemberDecorate %66 0 RelaxedPrecision
               OpDecorate %69 RelaxedPrecision
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeBool
          %7 = OpTypePointer Function %6
         %78 = OpTypePointer Private %6
          %8 = OpTypeFunction %6 %7
         %17 = OpTypeInt 32 1
         %18 = OpTypePointer Function %17
         %79 = OpTypePointer Private %17
         %20 = OpConstant %17 0
         %21 = OpTypeFloat 32
         %22 = OpTypePointer Function %21
         %80 = OpTypePointer Private %21
         %24 = OpConstant %21 0
         %25 = OpConstantFalse %6
         %32 = OpConstantTrue %6
         %33 = OpTypeVector %21 4
         %34 = OpTypePointer Function %33
         %81 = OpTypePointer Private %33
         %36 = OpConstantComposite %33 %24 %24 %24 %24
         %37 = OpTypeMatrix %33 4
         %84 = OpConstantComposite %37 %36 %36 %36 %36
         %38 = OpTypePointer Function %37
         %82 = OpTypePointer Private %37
         %44 = OpConstant %21 1
         %66 = OpTypeStruct %17 %21 %6 %33 %37
         %85 = OpConstantComposite %66 %20 %24 %25 %36 %84
         %67 = OpTypePointer Function %66
         %83 = OpTypePointer Private %66
         %86 = OpVariable %79 Private %20
         %87 = OpUndef %79
         %88 = OpConstantNull %79
         %89 = OpVariable %79 Private %20
         %91 = OpVariable %78 Private %25
         %93 = OpVariable %81 Private %36
         %95 = OpVariable %83 Private %85
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %96 = OpVariable %18 Function %20
         %94 = OpVariable %38 Function %84
         %92 = OpVariable %7 Function %25
         %90 = OpVariable %22 Function %24
         %19 = OpVariable %18 Function
         %23 = OpVariable %22 Function
         %26 = OpVariable %7 Function
         %30 = OpVariable %7 Function
         %35 = OpVariable %34 Function
         %39 = OpVariable %38 Function
         %68 = OpVariable %67 Function
               OpStore %19 %20
               OpStore %23 %24
               OpStore %26 %25
               OpCopyMemory %89 %19
               OpCopyMemory %90 %23
               OpCopyMemory %91 %26
               OpCopyMemory %92 %30
               OpCopyMemory %93 %35
               OpCopyMemory %94 %39
               OpCopyMemory %95 %68
               OpCopyMemory %96 %86
         %27 = OpFunctionCall %6 %10 %26
               OpSelectionMerge %29 None
               OpBranchConditional %27 %28 %31
         %28 = OpLabel
               OpBranch %29
         %31 = OpLabel
               OpBranch %29
         %76 = OpLabel
         %77 = OpLogicalEqual %6 %25 %32
               OpBranch %29
         %29 = OpLabel
         %75 = OpPhi %6 %25 %31 %32 %28 %77 %76
               OpStore %30 %75
         %40 = OpLoad %33 %35
         %41 = OpLoad %33 %35
         %42 = OpLoad %33 %35
         %43 = OpLoad %33 %35
         %45 = OpCompositeExtract %21 %40 0
         %46 = OpCompositeExtract %21 %40 1
         %47 = OpCompositeExtract %21 %40 2
         %48 = OpCompositeExtract %21 %40 3
         %49 = OpCompositeExtract %21 %41 0
         %50 = OpCompositeExtract %21 %41 1
         %51 = OpCompositeExtract %21 %41 2
         %52 = OpCompositeExtract %21 %41 3
         %53 = OpCompositeExtract %21 %42 0
         %54 = OpCompositeExtract %21 %42 1
         %55 = OpCompositeExtract %21 %42 2
         %56 = OpCompositeExtract %21 %42 3
         %57 = OpCompositeExtract %21 %43 0
         %58 = OpCompositeExtract %21 %43 1
         %59 = OpCompositeExtract %21 %43 2
         %60 = OpCompositeExtract %21 %43 3
         %61 = OpCompositeConstruct %33 %45 %46 %47 %48
         %62 = OpCompositeConstruct %33 %49 %50 %51 %52
         %63 = OpCompositeConstruct %33 %53 %54 %55 %56
         %64 = OpCompositeConstruct %33 %57 %58 %59 %60
         %65 = OpCompositeConstruct %37 %61 %62 %63 %64
               OpStore %39 %65
         %69 = OpLoad %17 %19
         %70 = OpLoad %21 %23
         %71 = OpLoad %6 %30
         %72 = OpLoad %33 %35
         %73 = OpLoad %37 %39
         %74 = OpCompositeConstruct %66 %69 %70 %71 %72 %73
               OpStore %68 %74
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %6 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
         %13 = OpLoad %6 %9
               OpStore %12 %13
         %14 = OpLoad %6 %12
               OpReturnValue %14
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools