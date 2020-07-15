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

#include "source/fuzz/transformation_replace_params_with_struct.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceParamsWithStructTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %13 0 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
          %2 = OpTypeVoid
          %6 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
          %7 = OpTypePointer Private %6
          %8 = OpTypeFloat 32
          %9 = OpTypePointer Private %8
         %10 = OpTypeVector %8 2
         %11 = OpTypePointer Private %10
         %12 = OpTypeBool
         %51 = OpTypeFunction %2 %12
         %64 = OpTypeStruct %6
         %63 = OpTypeFunction %2 %64
         %65 = OpTypeFunction %2 %6
         %40 = OpTypePointer Function %12
         %13 = OpTypeStruct %6 %8
         %45 = OpTypeStruct %6 %10 %13
         %46 = OpTypeStruct %12
         %47 = OpTypeStruct %8 %45 %46
         %14 = OpTypePointer Private %13
         %15 = OpTypeFunction %2 %6 %8 %10 %13 %40 %12
         %22 = OpConstant %6 0
         %23 = OpConstant %8 0
         %26 = OpConstantComposite %10 %23 %23
         %27 = OpConstantTrue %12
         %28 = OpConstantComposite %13 %22 %23
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %41 = OpVariable %40 Function %27
         %33 = OpFunctionCall %2 %20 %22 %23 %26 %28 %41 %27
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %16 = OpFunctionParameter %6
         %17 = OpFunctionParameter %8
         %18 = OpFunctionParameter %10
         %19 = OpFunctionParameter %13
         %42 = OpFunctionParameter %40
         %43 = OpFunctionParameter %12
         %21 = OpLabel
               OpReturn
               OpFunctionEnd
         %50 = OpFunction %2 None %51
         %52 = OpFunctionParameter %12
         %53 = OpLabel
               OpReturn
               OpFunctionEnd
         %54 = OpFunction %2 None %51
         %55 = OpFunctionParameter %12
         %56 = OpLabel
               OpReturn
               OpFunctionEnd
         %57 = OpFunction %2 None %63
         %58 = OpFunctionParameter %64
         %59 = OpLabel
               OpReturn
               OpFunctionEnd
         %60 = OpFunction %2 None %65
         %61 = OpFunctionParameter %6
         %62 = OpLabel
               OpReturn
               OpFunctionEnd
         %66 = OpFunction %2 None %65
         %67 = OpFunctionParameter %6
         %68 = OpLabel
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

  // |parameter_id| is empty.
  ASSERT_FALSE(
      TransformationReplaceParamsWithStruct({}, 70, 71, {{33, 80}, {80, 81}})
          .IsApplicable(context.get(), transformation_context));

  // |parameter_id| has duplicates.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 16, 17}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  // |parameter_id| has invalid values.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({21, 16, 17}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 70, 17}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  // Parameter's belong to different functions.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17, 52}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  // Parameter has unsupported type.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17, 42, 43}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  // OpTypeStruct does not exist in the module.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 43}, 70, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  // |caller_id_to_fresh_composite_id| misses values.
  ASSERT_FALSE(
      TransformationReplaceParamsWithStruct({16, 17}, 70, 71, {{80, 81}})
          .IsApplicable(context.get(), transformation_context));

  // All fresh ids must be unique.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 70,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 71,
                                                     {{33, 70}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 71,
                                                     {{33, 72}, {80, 72}})
                   .IsApplicable(context.get(), transformation_context));

  // All 'fresh' ids must be fresh.
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 71,
                                                     {{33, 33}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 71,
                                                     {{33, 33}, {80, 33}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 33, 71,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));
  ASSERT_FALSE(TransformationReplaceParamsWithStruct({16, 17}, 70, 33,
                                                     {{33, 80}, {80, 81}})
                   .IsApplicable(context.get(), transformation_context));

  {
    TransformationReplaceParamsWithStruct transformation({16, 18, 19}, 70, 71,
                                                         {{33, 72}, {70, 73}});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParamsWithStruct transformation({43}, 73, 74,
                                                         {{33, 75}});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParamsWithStruct transformation({17, 71, 74}, 76, 77,
                                                         {{33, 78}});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParamsWithStruct transformation({55}, 79, 80, {});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }
  {
    TransformationReplaceParamsWithStruct transformation({61}, 81, 82, {});
    ASSERT_TRUE(
        transformation.IsApplicable(context.get(), transformation_context));
    transformation.Apply(context.get(), &transformation_context);
  }

  std::string expected_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpMemberDecorate %13 0 RelaxedPrecision
               OpDecorate %16 RelaxedPrecision
          %2 = OpTypeVoid
          %6 = OpTypeInt 32 1
          %3 = OpTypeFunction %2
          %7 = OpTypePointer Private %6
          %8 = OpTypeFloat 32
          %9 = OpTypePointer Private %8
         %10 = OpTypeVector %8 2
         %11 = OpTypePointer Private %10
         %12 = OpTypeBool
         %51 = OpTypeFunction %2 %12
         %64 = OpTypeStruct %6
         %63 = OpTypeFunction %2 %64
         %65 = OpTypeFunction %2 %6
         %40 = OpTypePointer Function %12
         %13 = OpTypeStruct %6 %8
         %45 = OpTypeStruct %6 %10 %13
         %46 = OpTypeStruct %12
         %47 = OpTypeStruct %8 %45 %46
         %14 = OpTypePointer Private %13
         %22 = OpConstant %6 0
         %23 = OpConstant %8 0
         %26 = OpConstantComposite %10 %23 %23
         %27 = OpConstantTrue %12
         %28 = OpConstantComposite %13 %22 %23
         %15 = OpTypeFunction %2 %40 %47
         %79 = OpTypeFunction %2 %46
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %41 = OpVariable %40 Function %27
         %72 = OpCompositeConstruct %45 %22 %26 %28
         %75 = OpCompositeConstruct %46 %27
         %78 = OpCompositeConstruct %47 %23 %72 %75
         %33 = OpFunctionCall %2 %20 %41 %78
               OpReturn
               OpFunctionEnd
         %20 = OpFunction %2 None %15
         %42 = OpFunctionParameter %40
         %77 = OpFunctionParameter %47
         %21 = OpLabel
         %74 = OpCompositeExtract %46 %77 2
         %71 = OpCompositeExtract %45 %77 1
         %17 = OpCompositeExtract %8 %77 0
         %43 = OpCompositeExtract %12 %74 0
         %19 = OpCompositeExtract %13 %71 2
         %18 = OpCompositeExtract %10 %71 1
         %16 = OpCompositeExtract %6 %71 0
               OpReturn
               OpFunctionEnd
         %50 = OpFunction %2 None %51
         %52 = OpFunctionParameter %12
         %53 = OpLabel
               OpReturn
               OpFunctionEnd
         %54 = OpFunction %2 None %79
         %80 = OpFunctionParameter %46
         %56 = OpLabel
         %55 = OpCompositeExtract %12 %80 0
               OpReturn
               OpFunctionEnd
         %57 = OpFunction %2 None %63
         %58 = OpFunctionParameter %64
         %59 = OpLabel
               OpReturn
               OpFunctionEnd
         %60 = OpFunction %2 None %63
         %82 = OpFunctionParameter %64
         %62 = OpLabel
         %61 = OpCompositeExtract %6 %82 0
               OpReturn
               OpFunctionEnd
         %66 = OpFunction %2 None %65
         %67 = OpFunctionParameter %6
         %68 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsEqual(env, expected_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
