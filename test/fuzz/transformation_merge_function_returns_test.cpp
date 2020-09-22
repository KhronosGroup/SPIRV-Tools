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

#include "source/fuzz/counter_overflow_id_source.h"
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

TEST(TransformationMergeFunctionReturnsTest, InvalidIds) {
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
         %42 = OpTypeFloat 32
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
         %43 = OpConvertSToF %42 %10
               OpBranchConditional %8 %39 %36
         %39 = OpLabel
               OpReturn
         %37 = OpLabel
         %44 = OpConvertSToF %42 %10
               OpBranch %35
         %36 = OpLabel
         %31 = OpPhi %42 %43 %38
         %32 = OpPhi %5 %11 %38
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
          17, 100, 100, 0, 0, {{MakeReturnMergingInfo(20, 101, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Fresh id %100 is used twice.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 101, 0, 0, {{MakeReturnMergingInfo(20, 100, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id for the new merge block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 0, 0, 0, {{MakeReturnMergingInfo(20, 101, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id for the new header block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 0, 100, 0, 0, {{MakeReturnMergingInfo(20, 101, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id for the new |is_returning| instruction in an
  // existing merge block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          17, 100, 101, 0, 0, {{MakeReturnMergingInfo(20, 0, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id for the new |return_val| instruction in the new
  // return block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          14, 100, 101, 0, 10, {{MakeReturnMergingInfo(27, 102, 103, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // %0 cannot be a fresh id for the new |maybe_return_val| instruction in an
  // existing merge block, inside a non-void function.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          14, 100, 101, 102, 10, {{MakeReturnMergingInfo(27, 103, 0, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Fresh id %102 is repeated.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          14, 100, 101, 102, 10, {{MakeReturnMergingInfo(27, 102, 104, {{}})}})
          .IsApplicable(context.get(), transformation_context));

  // Id %11 (type int) does not have the correct type (float) for OpPhi
  // instruction %31.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          16, 100, 101, 0, 0,
          {{MakeReturnMergingInfo(36, 103, 104, {{{31, 11}, {32, 11}}})}})
          .IsApplicable(context.get(), transformation_context));

  // Id %11 (type int) does not have the correct type (float) for OpPhi
  // instruction %31.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          16, 100, 101, 0, 0,
          {{MakeReturnMergingInfo(36, 102, 0, {{{31, 11}, {32, 11}}})}})
          .IsApplicable(context.get(), transformation_context));

  // Id %43 is not available at the end of the entry block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          16, 100, 101, 0, 0,
          {{MakeReturnMergingInfo(36, 102, 0, {{{31, 44}, {32, 11}}})}})
          .IsApplicable(context.get(), transformation_context));

  // There is not a mapping for id %31 (float OpPhi instruction in a loop merge
  // block) and no suitable id is available at the end of the entry block.
  ASSERT_FALSE(TransformationMergeFunctionReturns(
                   16, 100, 101, 0, 0,
                   {{MakeReturnMergingInfo(36, 102, 0, {{{32, 11}}})}})
                   .IsApplicable(context.get(), transformation_context));

  // Id %1000 cannot be found in the module and no suitable id for OpPhi %31 is
  // available at the end of the entry block.
  ASSERT_FALSE(
      TransformationMergeFunctionReturns(
          16, 100, 101, 0, 0,
          {{MakeReturnMergingInfo(36, 102, 0, {{{31, 1000}, {32, 11}}})}})
          .IsApplicable(context.get(), transformation_context));
}

TEST(TransformationMergeFunctionReturnsTest, Simple) {
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
          %8 = OpTypeFunction %7 %7
          %9 = OpTypeFunction %7
         %10 = OpTypeBool
         %11 = OpConstantTrue %10
         %40 = OpConstantFalse %10
         %12 = OpConstant %5 1
          %2 = OpFunction %3 None %4
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %3 None %4
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpSelectionMerge %17 None
               OpBranchConditional %11 %18 %17
         %18 = OpLabel
               OpReturn
         %17 = OpLabel
               OpReturn
               OpFunctionEnd
         %19 = OpFunction %5 None %6
         %20 = OpLabel
               OpBranch %21
         %21 = OpLabel
               OpSelectionMerge %22 None
               OpBranchConditional %11 %23 %24
         %23 = OpLabel
               OpReturnValue %12
         %24 = OpLabel
         %25 = OpIAdd %5 %12 %12
               OpReturnValue %25
         %22 = OpLabel
               OpUnreachable
               OpFunctionEnd
         %26 = OpFunction %7 None %8
         %27 = OpFunctionParameter %7
         %28 = OpLabel
               OpBranch %29
         %29 = OpLabel
               OpSelectionMerge %30 None
               OpBranchConditional %11 %31 %30
         %31 = OpLabel
         %32 = OpFAdd %7 %27 %27
               OpReturnValue %32
         %30 = OpLabel
               OpReturnValue %27
               OpFunctionEnd
         %33 = OpFunction %7 None %9
         %34 = OpLabel
         %35 = OpConvertSToF %7 %12
               OpBranch %36
         %36 = OpLabel
               OpSelectionMerge %37 None
               OpBranchConditional %11 %38 %37
         %38 = OpLabel
         %39 = OpFAdd %7 %35 %35
               OpReturnValue %39
         %37 = OpLabel
               OpReturnValue %35
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

  // The 0s are allowed because the function's return type is void.
  auto transformation1 =
      TransformationMergeFunctionReturns(14, 100, 101, 0, 0, {{}});
  ASSERT_TRUE(
      transformation1.IsApplicable(context.get(), transformation_context));

  // %12 is available at the end of the entry block of %19 (it is a global
  // variable).
  ASSERT_TRUE(TransformationMergeFunctionReturns(19, 110, 111, 112, 12, {{}})
                  .IsApplicable(context.get(), transformation_context));

  // %1000 cannot be found in the module, but there is a suitable id available
  // at the end of the entry block (%12).
  auto transformation2 =
      TransformationMergeFunctionReturns(19, 110, 111, 112, 1000, {{}});
  ASSERT_TRUE(
      transformation2.IsApplicable(context.get(), transformation_context));

  // %27 is available at the end of the entry block of %26 (it is a function
  // parameter).
  ASSERT_TRUE(TransformationMergeFunctionReturns(26, 120, 121, 122, 27, {{}})
                  .IsApplicable(context.get(), transformation_context));

  // %1000 cannot be found in the module, but there is a suitable id available
  // at the end of the entry block (%27).
  auto transformation3 =
      TransformationMergeFunctionReturns(26, 120, 121, 122, 1000, {{}});
  ASSERT_TRUE(
      transformation3.IsApplicable(context.get(), transformation_context));

  // %35 is available at the end of the entry block of %33 (it is in the entry
  // block).
  ASSERT_TRUE(TransformationMergeFunctionReturns(26, 120, 121, 122, 27, {{}})
                  .IsApplicable(context.get(), transformation_context));

  // %1000 cannot be found in the module, but there is a suitable id available
  // at the end of the entry block (%35).
  auto transformation4 =
      TransformationMergeFunctionReturns(33, 120, 121, 122, 1000, {{}});
  ASSERT_TRUE(
      transformation3.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationMergeFunctionReturnsTest, NestedLoops) {
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
         %10 = OpConstant %5 2
         %11 = OpConstant %5 1
         %12 = OpConstant %5 3
          %2 = OpFunction %3 None %4
         %13 = OpLabel
               OpReturn
               OpFunctionEnd
         %14 = OpFunction %5 None %6
         %15 = OpLabel
               OpBranch %16
         %16 = OpLabel
               OpLoopMerge %17 %18 None
               OpBranch %19
         %19 = OpLabel
               OpBranchConditional %8 %20 %17
         %20 = OpLabel
               OpSelectionMerge %21 None
               OpBranchConditional %9 %22 %21
         %22 = OpLabel
               OpBranch %23
         %23 = OpLabel
               OpLoopMerge %24 %25 None
               OpBranch %26
         %26 = OpLabel
               OpBranchConditional %9 %27 %24
         %27 = OpLabel
               OpReturnValue %10
         %25 = OpLabel
               OpBranch %23
         %24 = OpLabel
         %28 = OpPhi %5 %11 %26
         %29 = OpPhi %5 %10 %26
               OpBranch %30
         %30 = OpLabel
               OpReturnValue %28
         %21 = OpLabel
               OpBranch %18
         %18 = OpLabel
               OpBranch %16
         %17 = OpLabel
               OpBranch %31
         %31 = OpLabel
               OpReturnValue %12
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

  auto transformation = TransformationMergeFunctionReturns(
      14, 100, 101, 102, 11,
      {{MakeReturnMergingInfo(24, 103, 104, {{{28, 10}, {29, 12}}}),
        MakeReturnMergingInfo(17, 105, 106, {})}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationMergeFunctionReturnsTest, OverflowIds) {
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
         %10 = OpConstant %5 1
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %5 None %6
         %13 = OpLabel
               OpBranch %14
         %14 = OpLabel
         %15 = OpIAdd %5 %10 %10
               OpLoopMerge %16 %17 None
               OpBranch %18
         %18 = OpLabel
               OpBranchConditional %8 %19 %16
         %19 = OpLabel
               OpSelectionMerge %20 None
               OpBranchConditional %9 %21 %20
         %21 = OpLabel
               OpReturnValue %10
         %20 = OpLabel
               OpBranch %17
         %17 = OpLabel
               OpBranchConditional %8 %14 %16
         %16 = OpLabel
         %22 = OpPhi %5 %15 %17 %10 %18
               OpBranch %23
         %23 = OpLabel
               OpReturnValue %22
               OpFunctionEnd
         %24 = OpFunction %3 None %4
         %25 = OpLabel
               OpBranch %26
         %26 = OpLabel
               OpLoopMerge %27 %28 None
               OpBranch %29
         %29 = OpLabel
               OpBranchConditional %8 %30 %27
         %30 = OpLabel
               OpSelectionMerge %31 None
               OpBranchConditional %9 %32 %31
         %32 = OpLabel
               OpReturn
         %31 = OpLabel
               OpBranch %28
         %28 = OpLabel
               OpBranch %26
         %27 = OpLabel
         %33 = OpPhi %5 %10 %29
               OpBranch %34
         %34 = OpLabel
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

  TransformationContext transformation_context_with_overflow_ids(
      &fact_manager, validator_options,
      MakeUnique<CounterOverflowIdSource>(1000));

  // No mapping from merge block %16 to fresh ids is given, so overflow ids are
  // needed.
  auto transformation1 =
      TransformationMergeFunctionReturns(12, 100, 101, 102, 10, {{}});

#ifndef NDEBUG
  ASSERT_DEATH(
      transformation1.IsApplicable(context.get(), transformation_context),
      "Bad attempt to query whether overflow ids are available.");
#endif

  ASSERT_TRUE(transformation1.IsApplicable(
      context.get(), transformation_context_with_overflow_ids));

  // No mapping from merge block %27 to fresh ids is given, so overflow ids are
  // needed.
  auto transformation2 =
      TransformationMergeFunctionReturns(24, 110, 111, 0, 0, {{}});

#ifndef NDEBUG
  ASSERT_DEATH(
      transformation2.IsApplicable(context.get(), transformation_context),
      "Bad attempt to query whether overflow ids are available.");
#endif

  ASSERT_TRUE(transformation2.IsApplicable(
      context.get(), transformation_context_with_overflow_ids));
}

TEST(TransformationMergeFunctionReturnsTest, MissingIdsForOpPhi) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3
          %5 = OpTypeBool
          %6 = OpConstantTrue %5
          %7 = OpConstantFalse %5
          %8 = OpTypeInt 32 1
          %9 = OpTypeFunction %3 %8
         %10 = OpTypeFloat 32
          %2 = OpFunction %3 None %4
         %11 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %3 None %9
         %13 = OpFunctionParameter %8
         %14 = OpLabel
         %15 = OpConvertSToF %10 %13
               OpBranch %16
         %16 = OpLabel
               OpLoopMerge %17 %18 None
               OpBranch %19
         %19 = OpLabel
               OpBranchConditional %6 %20 %17
         %20 = OpLabel
               OpSelectionMerge %21 None
               OpBranchConditional %7 %22 %21
         %22 = OpLabel
               OpReturn
         %21 = OpLabel
               OpBranch %18
         %18 = OpLabel
               OpBranch %16
         %17 = OpLabel
         %23 = OpPhi %8 %13 %19
         %24 = OpPhi %10 %15 %19
         %25 = OpPhi %5 %6 %19
               OpBranch %26
         %26 = OpLabel
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

  // This tests checks whether the transformation is able to find suitable ids
  // to use in existing OpPhi instructions if they are not provided in the
  // corresponding mapping.

  auto transformation = TransformationMergeFunctionReturns(
      12, 101, 102, 0, 0,
      {{MakeReturnMergingInfo(17, 103, 0, {{{25, 7}, {35, 8}}})}});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
}
}  // namespace
}  // namespace fuzz
}  // namespace spvtools