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

#include "source/fuzz/transformation_merge_function_returns.h"

#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

protobufs::ReturnMergingInfo MakeReturnMergingInfo(
    uint32_t merge_block_id, uint32_t is_returning_id,
    uint32_t maybe_return_val_id,
    std::map<uint32_t, uint32_t> opphi_to_suitable_id) {
  protobufs::ReturnMergingInfo result;
  result.set_merge_block_id(merge_block_id);
  result.set_is_returning_id(is_returning_id);
  result.set_maybe_return_val_id(maybe_return_val_id);
  *result.mutable_opphi_to_suitable_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(opphi_to_suitable_id);
  return result;
}

TEST(TransformationMergeFunctionReturnsTest, SimpleInapplicable) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypeFunction %5
          %7 = OpTypeFloat 32
          %8 = OpTypeFunction %7
          %9 = OpTypeBool
         %10 = OpConstantTrue %9
         %11 = OpConstantFalse %9
         %12 = OpConstant %5 0
         %13 = OpConstant %5 1
          %2 = OpFunction %3 None %4
         %14 = OpLabel
         %15 = OpFunctionCall %3 %16
         %17 = OpFunctionCall %3 %18
         %19 = OpFunctionCall %3 %20
         %21 = OpFunctionCall %7 %22
               OpReturn
               OpFunctionEnd
         %16 = OpFunction %3 None %4
         %23 = OpLabel
               OpSelectionMerge %24 None
               OpBranchConditional %10 %25 %26
         %25 = OpLabel
               OpReturn
         %26 = OpLabel
               OpReturn
         %24 = OpLabel
               OpUnreachable
               OpFunctionEnd
         %18 = OpFunction %3 None %4
         %27 = OpLabel
               OpBranch %28
         %28 = OpLabel
               OpLoopMerge %29 %30 None
               OpBranch %31
         %31 = OpLabel
               OpBranchConditional %10 %32 %29
         %32 = OpLabel
               OpReturn
         %30 = OpLabel
               OpBranch %28
         %29 = OpLabel
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %3 None %4
         %33 = OpLabel
               OpBranch %34
         %34 = OpLabel
               OpLoopMerge %35 %36 None
               OpBranch %37
         %37 = OpLabel
               OpBranchConditional %10 %38 %35
         %38 = OpLabel
               OpReturn
         %36 = OpLabel
               OpBranch %34
         %35 = OpLabel
         %39 = OpFunctionCall %3 %18
               OpReturn
               OpFunctionEnd
         %22 = OpFunction %7 None %8
         %40 = OpLabel
               OpBranch %51
         %51 = OpLabel
               OpSelectionMerge %41 None
               OpBranchConditional %10 %42 %41
         %42 = OpLabel
         %43 = OpConvertSToF %7 %12
               OpReturnValue %43
         %41 = OpLabel
         %44 = OpConvertSToF %7 %13
               OpReturnValue %44
               OpFunctionEnd
         %45 = OpFunction %5 None %6
         %46 = OpLabel
               OpBranch %52
         %52 = OpLabel
         %47 = OpConvertSToF %7 %13
               OpSelectionMerge %48 None
               OpBranchConditional %10 %49 %48
         %49 = OpLabel
               OpReturnValue %12
         %48 = OpLabel
         %50 = OpCopyObject %5 %12
               OpReturnValue %13
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

  // Function %1 does not exist.
  ASSERT_FALSE(TransformationMergeFunctionReturns(1, 100, 101, 0, 0, {{}})
                   .IsApplicable(context.get(), transformation_context));

  // The entry block (%22) of function %15 does not branch unconditionally to
  // the following block.
  ASSERT_FALSE(TransformationMergeFunctionReturns(16, 100, 101, 0, 0, {{}})
                   .IsApplicable(context.get(), transformation_context));

  // Block %28 is the merge block of a loop containing a return instruction, but
  // it contains an OpReturn instruction (so, it contains instructions that are
  // not OpLabel, OpPhi or OpBranch).
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          18, 100, 101, 0, 0, {{MakeReturnMergingInfo(29, 102, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Block %34 is the merge block of a loop containing a return instruction, but
  // it contains an OpFunctionCall instruction (so, it contains instructions
  // that are not OpLabel, OpPhi or OpBranch).
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          20, 100, 101, 0, 0, {{MakeReturnMergingInfo(35, 102, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Id %1000 cannot be found in the module and there is no id of the correct
  // type (float) available at the end of the entry block of function %21.
  ASSERT_FALSE(TransformationMergeFunctionReturns(22, 100, 101, 102, 1000, {{}})
                   .IsApplicable(context.get(), transformation_context));

  // Id %47 is of type float, while function %45 has return type int.
  ASSERT_FALSE(TransformationMergeFunctionReturns(45, 100, 101, 102, 47, {{}})
                   .IsApplicable(context.get(), transformation_context));

  // Id %50 is not available at the end of the entry block of function %45.
  ASSERT_FALSE(TransformationMergeFunctionReturns(45, 100, 101, 102, 50, {{}})
                   .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationMergeFunctionReturnsTest, MissingBooleans) {
  {
    // OpConstantTrue is missing.
    std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "A("
               OpDecorate %3 RelaxedPrecision
               OpDecorate %4 RelaxedPrecision
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypeFunction %7
          %9 = OpTypeBool
         %10 = OpConstantFalse %9
         %11 = OpConstant %7 1
         %12 = OpConstant %7 2
          %2 = OpFunction %5 None %6
         %13 = OpLabel
          %4 = OpFunctionCall %7 %3
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %7 None %8
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %10 %17 %16
         %17 = OpLabel
               OpReturnValue %11
         %16 = OpLabel
               OpReturnValue %12
               OpFunctionEnd
)";

    const auto env = SPV_ENV_UNIVERSAL_1_5;
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, shader, kFuzzAssembleOption);
    ASSERT_TRUE(IsValid(env, context.get()));

    FactManager fact_manager;
    spvtools::ValidatorOptions validator_options;
    TransformationContext transformation_context(&fact_manager,
                                                 validator_options);

    ASSERT_FALSE(TransformationMergeFunctionReturns(3, 100, 101, 0, 0, {{}})
                     .IsApplicable(context.get(), transformation_context));
  }
  {
    // OpConstantFalse is missing.
    std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "A("
               OpDecorate %3 RelaxedPrecision
               OpDecorate %4 RelaxedPrecision
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeInt 32 1
          %8 = OpTypeFunction %7
          %9 = OpTypeBool
         %10 = OpConstantTrue %9
         %11 = OpConstant %7 1
         %12 = OpConstant %7 2
          %2 = OpFunction %5 None %6
         %13 = OpLabel
          %4 = OpFunctionCall %7 %3
               OpReturn
               OpFunctionEnd
          %3 = OpFunction %7 None %8
         %14 = OpLabel
               OpBranch %15
         %15 = OpLabel
               OpSelectionMerge %16 None
               OpBranchConditional %10 %17 %16
         %17 = OpLabel
               OpReturnValue %11
         %16 = OpLabel
               OpReturnValue %12
               OpFunctionEnd
)";

    const auto env = SPV_ENV_UNIVERSAL_1_5;
    const auto consumer = nullptr;
    const auto context =
        BuildModule(env, consumer, shader, kFuzzAssembleOption);
    ASSERT_TRUE(IsValid(env, context.get()));

    FactManager fact_manager;
    spvtools::ValidatorOptions validator_options;
    TransformationContext transformation_context(&fact_manager,
                                                 validator_options);

    ASSERT_FALSE(TransformationMergeFunctionReturns(3, 100, 101, 0, 0, {{}})
                     .IsApplicable(context.get(), transformation_context));
  }
}

TEST(TransformationMergeFunctionReturnsTest, InvalidFreshIds) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeInt 32 1
          %6 = OpTypeFunction %5
          %7 = OpTypeBool
          %8 = OpConstantTrue %7
          %9 = OpConstantFalse %7
         %10 = OpConstant %5 0
         %11 = OpConstant %5 1
          %2 = OpFunction %3 None %4
         %12 = OpLabel
         %13 = OpFunctionCall %5 %14
         %15 = OpFunctionCall %3 %16
               OpReturn
               OpFunctionEnd
         %17 = OpFunction %3 None %4
         %18 = OpLabel
               OpBranch %19
         %19 = OpLabel
               OpLoopMerge %20 %21 None
               OpBranch %22
         %22 = OpLabel
               OpBranchConditional %8 %23 %20
         %23 = OpLabel
               OpReturn
         %21 = OpLabel
               OpBranch %19
         %20 = OpLabel
               OpBranch %24
         %24 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %5 None %6
         %25 = OpLabel
               OpBranch %26
         %26 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
               OpBranchConditional %8 %30 %27
         %30 = OpLabel
               OpReturnValue %10
         %28 = OpLabel
               OpBranch %26
         %27 = OpLabel
         %31 = OpPhi %5 %10 %29
         %32 = OpPhi %5 %11 %29
               OpBranch %33
         %33 = OpLabel
               OpReturnValue %11
               OpFunctionEnd
         %16 = OpFunction %3 None %4
         %34 = OpLabel
               OpBranch %35
         %35 = OpLabel
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
               OpBranchConditional %8 %39 %36
         %39 = OpLabel
               OpReturn
         %37 = OpLabel
               OpBranch %35
         %36 = OpLabel
               OpBranch %40
         %40 = OpLabel
         %41 = OpFunctionCall %3 %17
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

  // Fresh id %100 is used twice.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 100, 0, 0, {{MakeReturnMergingInfo(20, 101, 102, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Fresh id %100 is used twice.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 101, 0, 0, {{MakeReturnMergingInfo(20, 100, 102, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 0, 0, 0, {{MakeReturnMergingInfo(20, 101, 102, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // TODO: Continue
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools