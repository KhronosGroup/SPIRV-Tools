// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/transformation_wrap_vector_synonym.h"
#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_composite_construct.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationWrapVectorSynonym, SimpleTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %97
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 10
         %11 = OpConstant %6 -5
         %12 = OpTypeVector %6 2
         %13 = OpTypePointer Function %12
         %18 = OpTypeInt 32 0
         %19 = OpTypePointer Function %18
         %21 = OpConstant %18 8
         %23 = OpConstant %18 2
         %24 = OpTypeVector %18 3
         %25 = OpTypePointer Function %24
         %31 = OpTypeFloat 32
         %32 = OpTypePointer Function %31
         %34 = OpConstant %31 3.29999995
         %36 = OpConstant %31 1.10000002
         %37 = OpTypeVector %31 4
         %38 = OpTypePointer Function %37
         %96 = OpTypePointer Input %31
         %97 = OpVariable %96 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %13 Function
         %20 = OpVariable %19 Function
         %22 = OpVariable %19 Function
         %26 = OpVariable %25 Function
         %33 = OpVariable %32 Function
         %35 = OpVariable %32 Function
         %39 = OpVariable %38 Function
         %47 = OpVariable %7 Function
         %51 = OpVariable %7 Function
         %55 = OpVariable %7 Function
         %59 = OpVariable %7 Function
         %63 = OpVariable %19 Function
         %67 = OpVariable %19 Function
         %71 = OpVariable %19 Function
         %75 = OpVariable %19 Function
         %79 = OpVariable %32 Function
         %83 = OpVariable %32 Function
         %87 = OpVariable %32 Function
         %91 = OpVariable %32 Function
               OpStore %8 %9
               OpStore %10 %11
         %15 = OpLoad %6 %8
         %16 = OpLoad %6 %10
         %17 = OpCompositeConstruct %12 %15 %16
               OpStore %14 %17
               OpStore %20 %21
               OpStore %22 %23
         %27 = OpLoad %18 %20
         %28 = OpLoad %18 %20
         %29 = OpLoad %18 %22
         %30 = OpCompositeConstruct %24 %27 %28 %29
               OpStore %26 %30
               OpStore %33 %34
               OpStore %35 %36
         %40 = OpLoad %31 %33
         %41 = OpLoad %31 %33
         %42 = OpLoad %31 %35
         %43 = OpLoad %31 %35
         %44 = OpCompositeConstruct %37 %40 %41 %42 %43
         %45 = OpLoad %37 %39
         %46 = OpVectorShuffle %37 %45 %44 5 6 7 4
               OpStore %39 %46
         %48 = OpLoad %6 %8
         %49 = OpLoad %6 %10
         %50 = OpIAdd %6 %48 %49
               OpStore %47 %50
         %52 = OpLoad %6 %8
         %53 = OpLoad %6 %10
         %54 = OpISub %6 %52 %53
               OpStore %51 %54
         %56 = OpLoad %6 %8
         %57 = OpLoad %6 %10
         %58 = OpIMul %6 %56 %57
               OpStore %55 %58
         %60 = OpLoad %6 %8
         %61 = OpLoad %6 %10
         %62 = OpSDiv %6 %60 %61
               OpStore %59 %62
         %64 = OpLoad %18 %20
         %65 = OpLoad %18 %22
         %66 = OpIAdd %18 %64 %65
               OpStore %63 %66
         %68 = OpLoad %18 %20
         %69 = OpLoad %18 %22
         %70 = OpISub %18 %68 %69
               OpStore %67 %70
         %72 = OpLoad %18 %20
         %73 = OpLoad %18 %22
         %74 = OpIMul %18 %72 %73
               OpStore %71 %74
         %76 = OpLoad %18 %20
         %77 = OpLoad %18 %22
         %78 = OpUDiv %18 %76 %77
               OpStore %75 %78
         %80 = OpLoad %31 %33
         %81 = OpLoad %31 %35
         %82 = OpFAdd %31 %80 %81
               OpStore %79 %82
         %84 = OpLoad %31 %33
         %85 = OpLoad %31 %35
         %86 = OpFSub %31 %84 %85
               OpStore %83 %86
         %88 = OpLoad %31 %33
         %89 = OpLoad %31 %35
         %90 = OpFMul %31 %88 %89
               OpStore %87 %90
         %92 = OpLoad %31 %33
         %93 = OpLoad %31 %35
         %94 = OpFDiv %31 %92 %93
               OpStore %91 %94
               OpReturn
               OpFunctionEnd
  )";
  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;

  // Check context validity.
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));

  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  // Vec Type Id |   Vector Type  |  Element Type id |   Element Type  |
  // ------------+----------------+------------------+-----------------+
  //     12      |      vec2      |         6        |      int32      |
  //     24      |      vec3      |        18        |     uint32      |
  //     37      |      vec4      |        31        |      float      |

  // Instruction Id | Opcode  | Type Id | constant id 1 | constant id 2 |
  // ---------------+---------+---------+---------------+---------------+
  //       50       | OpIAdd  |    6    |      48       |      49       |
  //       54       | OpISub  |    6    |      52       |      53       |
  //       58       | OpIMul  |    6    |      56       |      56       |
  //       62       | OpSDiv  |    6    |      60       |      61       |
  //       66       | OpIAdd  |    18   |      64       |      65       |
  //       70       | OpISub  |    18   |      68       |      69       |
  //       74       | OpIMul  |    18   |      72       |      73       |
  //       78       | OpUDiv  |    18   |      76       |      77       |
  //       82       | OpFAdd  |    31   |      80       |      81       |
  //       86       | OpFSub  |    31   |      84       |      85       |
  //       90       | OpFMul  |    31   |      88       |      89       |
  //       94       | OpFDiv  |    31   |      92       |      93       |

  TransformationCompositeConstruct add_int_vec1(
      12, {48, 48}, MakeInstructionDescriptor(50, SpvOpIAdd, 0), 100);
  ASSERT_TRUE(add_int_vec1.IsApplicable(context.get(), transformation_context));

  TransformationCompositeConstruct add_int_vec2(
      12, {49, 49}, MakeInstructionDescriptor(50, SpvOpIAdd, 0), 101);
  ASSERT_TRUE(add_int_vec2.IsApplicable(context.get(), transformation_context));
  // Insert vec2 of id 100 with the first value of OpIAdd instruction with
  // id 50.
  ApplyAndCheckFreshIds(add_int_vec1, context.get(), &transformation_context);
  // Insert vec2 of id 101 with the second value of OpIAdd instruction with
  // id 50.
  ApplyAndCheckFreshIds(add_int_vec2, context.get(), &transformation_context);

  transformation_context.GetFactManager()->AddFactDataSynonym(MakeDataDescriptor(100, {1}),
                                                              MakeDataDescriptor(48, {}));
  transformation_context.GetFactManager()->AddFactDataSynonym(MakeDataDescriptor(101, {1}),
                                                              MakeDataDescriptor(49, {}));

  // The following are all invalid use.
  {
    // Bad: Instruction id does not exist.
    TransformationWrapVectorSynonym wrap_add_int_bad1(103, 100, 101, 102, 1);
    ASSERT_FALSE(
        wrap_add_int_bad1.IsApplicable(context.get(), transformation_context));

    // Bad: Instruction id given is not of a valid arithmetic operation typed
    // instruction.
    TransformationWrapVectorSynonym wrap_add_int_bad2(80, 100, 101, 102, 1);
    ASSERT_FALSE(
        wrap_add_int_bad1.IsApplicable(context.get(), transformation_context));

    // Bad: the id for the first vector does not exist.
    TransformationWrapVectorSynonym wrap_add_int_bad3(50, 105, 101, 102, 1);
    ASSERT_FALSE(
        wrap_add_int_bad3.IsApplicable(context.get(), transformation_context));

    // Bad: the id for the second vector does not exist.
    TransformationWrapVectorSynonym wrap_add_int_bad4(50, 100, 105, 102, 1);
    ASSERT_FALSE(
        wrap_add_int_bad4.IsApplicable(context.get(), transformation_context));

    // Bad: vector id is not fresh.
    TransformationWrapVectorSynonym wrap_add_int_bad6(50, 100, 101, 94, 1);
    ASSERT_FALSE(
        wrap_add_int_bad6.IsApplicable(context.get(), transformation_context));

    // Bad: The position goes out of bound for the given vector type.
    TransformationWrapVectorSynonym wrap_add_int_bad8(50, 100, 100, 94, 2);
    ASSERT_FALSE(
        wrap_add_int_bad8.IsApplicable(context.get(), transformation_context));

    // Bad: The original instruction is not a valid scalar operation instruction.
    TransformationWrapVectorSynonym wrap_add_int(27, 100, 101, 102, 1);
    ASSERT_FALSE(wrap_add_int.IsApplicable(context.get(), transformation_context));
  }


  // Good: The following transformation should be applicable.
  TransformationWrapVectorSynonym wrap_add_int(50, 100, 101, 102, 1);
  ASSERT_TRUE(wrap_add_int.IsApplicable(context.get(), transformation_context));
  // Insert an arithmetic instruction of the same type to add two vectors.
  ApplyAndCheckFreshIds(wrap_add_int, context.get(), &transformation_context);

  // |instruction_id| and id at |scalar_position of the result vector should be synonyms.
  ASSERT_TRUE(transformation_context.GetFactManager()->IsSynonymous(
      MakeDataDescriptor(102, {1}), MakeDataDescriptor(50, {})));

  // After applying transformations, three instructions:
  // %100 = OpCompositeConstruct %12 $48 %48
  // %101 = OpCompositeConstruct %12 $49 %49
  // %102 = OpIAdd %12 %100 %101
  //
  // that wraps the variables of the original instruction and perform vector
  // operation, should be added before:
  //
  // %50 = OpIAdd %6 %48 %49
  std::string after_transformation = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %97
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypePointer Function %6
          %9 = OpConstant %6 10
         %11 = OpConstant %6 -5
         %12 = OpTypeVector %6 2
         %13 = OpTypePointer Function %12
         %18 = OpTypeInt 32 0
         %19 = OpTypePointer Function %18
         %21 = OpConstant %18 8
         %23 = OpConstant %18 2
         %24 = OpTypeVector %18 3
         %25 = OpTypePointer Function %24
         %31 = OpTypeFloat 32
         %32 = OpTypePointer Function %31
         %34 = OpConstant %31 3.29999995
         %36 = OpConstant %31 1.10000002
         %37 = OpTypeVector %31 4
         %38 = OpTypePointer Function %37
         %96 = OpTypePointer Input %31
         %97 = OpVariable %96 Input
          %4 = OpFunction %2 None %3
          %5 = OpLabel
          %8 = OpVariable %7 Function
         %10 = OpVariable %7 Function
         %14 = OpVariable %13 Function
         %20 = OpVariable %19 Function
         %22 = OpVariable %19 Function
         %26 = OpVariable %25 Function
         %33 = OpVariable %32 Function
         %35 = OpVariable %32 Function
         %39 = OpVariable %38 Function
         %47 = OpVariable %7 Function
         %51 = OpVariable %7 Function
         %55 = OpVariable %7 Function
         %59 = OpVariable %7 Function
         %63 = OpVariable %19 Function
         %67 = OpVariable %19 Function
         %71 = OpVariable %19 Function
         %75 = OpVariable %19 Function
         %79 = OpVariable %32 Function
         %83 = OpVariable %32 Function
         %87 = OpVariable %32 Function
         %91 = OpVariable %32 Function
               OpStore %8 %9
               OpStore %10 %11
         %15 = OpLoad %6 %8
         %16 = OpLoad %6 %10
         %17 = OpCompositeConstruct %12 %15 %16
               OpStore %14 %17
               OpStore %20 %21
               OpStore %22 %23
         %27 = OpLoad %18 %20
         %28 = OpLoad %18 %20
         %29 = OpLoad %18 %22
         %30 = OpCompositeConstruct %24 %27 %28 %29
               OpStore %26 %30
               OpStore %33 %34
               OpStore %35 %36
         %40 = OpLoad %31 %33
         %41 = OpLoad %31 %33
         %42 = OpLoad %31 %35
         %43 = OpLoad %31 %35
         %44 = OpCompositeConstruct %37 %40 %41 %42 %43
         %45 = OpLoad %37 %39
         %46 = OpVectorShuffle %37 %45 %44 5 6 7 4
               OpStore %39 %46
         %48 = OpLoad %6 %8
         %49 = OpLoad %6 %10
        %100 = OpCompositeConstruct %12 %48 %48
        %101 = OpCompositeConstruct %12 %49 %49
        %102 = OpIAdd %12 %100 %101
         %50 = OpIAdd %6 %48 %49
               OpStore %47 %50
         %52 = OpLoad %6 %8
         %53 = OpLoad %6 %10
         %54 = OpISub %6 %52 %53
               OpStore %51 %54
         %56 = OpLoad %6 %8
         %57 = OpLoad %6 %10
         %58 = OpIMul %6 %56 %57
               OpStore %55 %58
         %60 = OpLoad %6 %8
         %61 = OpLoad %6 %10
         %62 = OpSDiv %6 %60 %61
               OpStore %59 %62
         %64 = OpLoad %18 %20
         %65 = OpLoad %18 %22
         %66 = OpIAdd %18 %64 %65
               OpStore %63 %66
         %68 = OpLoad %18 %20
         %69 = OpLoad %18 %22
         %70 = OpISub %18 %68 %69
               OpStore %67 %70
         %72 = OpLoad %18 %20
         %73 = OpLoad %18 %22
         %74 = OpIMul %18 %72 %73
               OpStore %71 %74
         %76 = OpLoad %18 %20
         %77 = OpLoad %18 %22
         %78 = OpUDiv %18 %76 %77
               OpStore %75 %78
         %80 = OpLoad %31 %33
         %81 = OpLoad %31 %35
         %82 = OpFAdd %31 %80 %81
               OpStore %79 %82
         %84 = OpLoad %31 %33
         %85 = OpLoad %31 %35
         %86 = OpFSub %31 %84 %85
               OpStore %83 %86
         %88 = OpLoad %31 %33
         %89 = OpLoad %31 %35
         %90 = OpFMul %31 %88 %89
               OpStore %87 %90
         %92 = OpLoad %31 %33
         %93 = OpLoad %31 %35
         %94 = OpFDiv %31 %92 %93
               OpStore %91 %94
               OpReturn
               OpFunctionEnd
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
