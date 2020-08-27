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

#include "source/fuzz/transformation_duplicate_region_with_selection.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {
TEST(TransformationDuplicateRegionWithSelectionTest, BasicUseTest) {
  // This test handles a case where the ids from the original region are used in
  // subsequent block.

  std::string shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %800
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %900
         %900 = OpLabel
         %901 = OpIAdd %6 %15 %13
         %902 = OpISub %6 %13 %15
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));
  std::string expected_shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %19 %800 %100
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %501
        %100 = OpLabel
        %201 = OpLoad %6 %9
        %202 = OpIAdd %6 %201 %14
               OpStore %12 %202
               OpBranch %501
        %501 = OpLabel
        %301 = OpPhi %6 %13 %800 %201 %100
        %302 = OpPhi %6 %15 %800 %202 %100
               OpBranch %900
        %900 = OpLabel
        %901 = OpIAdd %6 %302 %301
        %902 = OpISub %6 %301 %302
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

TEST(TransformationDuplicateRegionWithSelectionTest, BasicExitBlockTest) {
  // This test handles a case where the exit block of the region is the exit
  // block of the containing function.

  std::string shader = R"(
               OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %800
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpReturn
               OpFunctionEnd
    )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  TransformationDuplicateRegionWithSelection transformation_good_1 =
      TransformationDuplicateRegionWithSelection(
          500, 19, 501, 800, 800, {{800, 100}}, {{13, 201}, {15, 202}},
          {{13, 301}, {15, 302}});

  ASSERT_TRUE(transformation_good_1.IsApplicable(context.get(),
                                                 transformation_context));
  transformation_good_1.Apply(context.get(), &transformation_context);

  ASSERT_TRUE(IsValid(env, context.get()));

  std::string expected_shader = R"(
   OpCapability Shader
               OpCapability VariablePointers
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %10 "fun(i1;"
               OpName %9 "a"
               OpName %12 "b"
               OpName %18 "c"
               OpName %20 "param"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %8 = OpTypeFunction %2 %7
         %14 = OpConstant %6 2
         %16 = OpTypeBool
         %17 = OpTypePointer Function %16
         %19 = OpConstantTrue %16
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %18 = OpVariable %17 Function
         %20 = OpVariable %7 Function
               OpStore %18 %19
               OpStore %20 %14
         %21 = OpFunctionCall %2 %10 %20
               OpReturn
               OpFunctionEnd
         %10 = OpFunction %2 None %8
          %9 = OpFunctionParameter %7
         %11 = OpLabel
         %12 = OpVariable %7 Function
               OpBranch %500
        %500 = OpLabel
               OpSelectionMerge %501 None
               OpBranchConditional %19 %800 %100
        %800 = OpLabel
         %13 = OpLoad %6 %9
         %15 = OpIAdd %6 %13 %14
               OpStore %12 %15
               OpBranch %501
        %100 = OpLabel
        %201 = OpLoad %6 %9
        %202 = OpIAdd %6 %201 %14
               OpStore %12 %202
               OpBranch %501
        %501 = OpLabel
        %301 = OpPhi %6 %13 %800 %201 %100
        %302 = OpPhi %6 %15 %800 %202 %100
               OpReturn
               OpFunctionEnd

  )";
  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
