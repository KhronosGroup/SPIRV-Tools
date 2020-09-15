// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_add_bit_instruction_synonym.h"
#include "source/fuzz/instruction_descriptor.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationAddBitInstructionSynonymTest, IsApplicable) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %37 "main"

; Types
          %2 = OpTypeInt 32 0
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3

; Constants
          %5 = OpConstant %2 0
          %6 = OpConstant %2 1
          %7 = OpConstant %2 2
          %8 = OpConstant %2 3
          %9 = OpConstant %2 4
         %10 = OpConstant %2 5
         %11 = OpConstant %2 6
         %12 = OpConstant %2 7
         %13 = OpConstant %2 8
         %14 = OpConstant %2 9
         %15 = OpConstant %2 10
         %16 = OpConstant %2 11
         %17 = OpConstant %2 12
         %18 = OpConstant %2 13
         %19 = OpConstant %2 14
         %20 = OpConstant %2 15
         %21 = OpConstant %2 16
         %22 = OpConstant %2 17
         %23 = OpConstant %2 18
         %24 = OpConstant %2 19
         %25 = OpConstant %2 20
         %26 = OpConstant %2 21
         %27 = OpConstant %2 22
         %28 = OpConstant %2 23
         %29 = OpConstant %2 24
         %30 = OpConstant %2 25
         %31 = OpConstant %2 26
         %32 = OpConstant %2 27
         %33 = OpConstant %2 28
         %34 = OpConstant %2 29
         %35 = OpConstant %2 30
         %36 = OpConstant %2 31

; main function
         %37 = OpFunction %3 None %4
         %38 = OpLabel
         %39 = OpBitwiseOr %2 %5 %6
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  // Tests undefined bit instruction.
  auto transformation = TransformationAddBitInstructionSynonym(
      40, {41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
           54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
           67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
           80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
           93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105,
           106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
           119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
           132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
           145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
           158, 159, 160, 161, 162, 163, 164, 165, 166, 167});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests false bit instruction.
  transformation = TransformationAddBitInstructionSynonym(
      38, {40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
           53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
           66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
           79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
           92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
           144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
           157, 158, 159, 160, 161, 162, 163, 164, 165, 166});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests the number of fresh ids being different than the necessary.
  transformation = TransformationAddBitInstructionSynonym(
      39,
      {40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
       54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,
       68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
       82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
       96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
       124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
       138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
       152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests non-fresh ids.
  transformation = TransformationAddBitInstructionSynonym(
      39, {38,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
           52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
           65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
           78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
           91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103,
           104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
           117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
           130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
           143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
           156, 157, 158, 159, 160, 161, 162, 163, 164, 165});
  ASSERT_FALSE(
      transformation.IsApplicable(context.get(), transformation_context));

  // Tests applicable transformation.
  transformation = TransformationAddBitInstructionSynonym(
      39, {40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
           53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
           66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
           79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
           92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
           144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
           157, 158, 159, 160, 161, 162, 163, 164, 165, 166});
  ASSERT_TRUE(
      transformation.IsApplicable(context.get(), transformation_context));
}

TEST(TransformationAddBitInstructionSynonymTest, AddBitwiseSynonym) {
  std::string reference_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %37 "main"

; Types
          %2 = OpTypeInt 32 0
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3

; Constants
          %5 = OpConstant %2 0
          %6 = OpConstant %2 1
          %7 = OpConstant %2 2
          %8 = OpConstant %2 3
          %9 = OpConstant %2 4
         %10 = OpConstant %2 5
         %11 = OpConstant %2 6
         %12 = OpConstant %2 7
         %13 = OpConstant %2 8
         %14 = OpConstant %2 9
         %15 = OpConstant %2 10
         %16 = OpConstant %2 11
         %17 = OpConstant %2 12
         %18 = OpConstant %2 13
         %19 = OpConstant %2 14
         %20 = OpConstant %2 15
         %21 = OpConstant %2 16
         %22 = OpConstant %2 17
         %23 = OpConstant %2 18
         %24 = OpConstant %2 19
         %25 = OpConstant %2 20
         %26 = OpConstant %2 21
         %27 = OpConstant %2 22
         %28 = OpConstant %2 23
         %29 = OpConstant %2 24
         %30 = OpConstant %2 25
         %31 = OpConstant %2 26
         %32 = OpConstant %2 27
         %33 = OpConstant %2 28
         %34 = OpConstant %2 29
         %35 = OpConstant %2 30
         %36 = OpConstant %2 31

; main function
         %37 = OpFunction %3 None %4
         %38 = OpLabel
         %39 = OpBitwiseOr %2 %5 %6
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;
  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(&fact_manager,
                                               validator_options);

  auto transformation = TransformationAddBitInstructionSynonym(
      39, {40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
           53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
           66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
           79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
           92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
           105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
           118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
           131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
           144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
           157, 158, 159, 160, 161, 162, 163, 164, 165, 166});
  transformation.Apply(context.get(), &transformation_context);

  std::string variant_shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %37 "main"

; Types
          %2 = OpTypeInt 32 0
          %3 = OpTypeVoid
          %4 = OpTypeFunction %3

; Constants
          %5 = OpConstant %2 0
          %6 = OpConstant %2 1
          %7 = OpConstant %2 2
          %8 = OpConstant %2 3
          %9 = OpConstant %2 4
         %10 = OpConstant %2 5
         %11 = OpConstant %2 6
         %12 = OpConstant %2 7
         %13 = OpConstant %2 8
         %14 = OpConstant %2 9
         %15 = OpConstant %2 10
         %16 = OpConstant %2 11
         %17 = OpConstant %2 12
         %18 = OpConstant %2 13
         %19 = OpConstant %2 14
         %20 = OpConstant %2 15
         %21 = OpConstant %2 16
         %22 = OpConstant %2 17
         %23 = OpConstant %2 18
         %24 = OpConstant %2 19
         %25 = OpConstant %2 20
         %26 = OpConstant %2 21
         %27 = OpConstant %2 22
         %28 = OpConstant %2 23
         %29 = OpConstant %2 24
         %30 = OpConstant %2 25
         %31 = OpConstant %2 26
         %32 = OpConstant %2 27
         %33 = OpConstant %2 28
         %34 = OpConstant %2 29
         %35 = OpConstant %2 30
         %36 = OpConstant %2 31

; main function
         %37 = OpFunction %3 None %4
         %38 = OpLabel

         %40 = OpBitFieldUExtract %2 %5 %5 %6 ; extracts bit 0 from %5
         %41 = OpBitFieldUExtract %2 %6 %5 %6 ; extracts bit 0 from %6
         %42 = OpBitwiseOr %2 %40 %41

         %43 = OpBitFieldUExtract %2 %5 %6 %6 ; extracts bit 1 from %5
         %44 = OpBitFieldUExtract %2 %6 %6 %6 ; extracts bit 1 from %6
         %45 = OpBitwiseOr %2 %43 %44

         %46 = OpBitFieldUExtract %2 %5 %7 %6 ; extracts bit 2 from %5
         %47 = OpBitFieldUExtract %2 %6 %7 %6 ; extracts bit 2 from %6
         %48 = OpBitwiseOr %2 %46 %47

         %49 = OpBitFieldUExtract %2 %5 %8 %6 ; extracts bit 3 from %5
         %50 = OpBitFieldUExtract %2 %6 %8 %6 ; extracts bit 3 from %6
         %51 = OpBitwiseOr %2 %49 %50

         %52 = OpBitFieldUExtract %2 %5 %9 %6 ; extracts bit 4 from %5
         %53 = OpBitFieldUExtract %2 %6 %9 %6 ; extracts bit 4 from %6
         %54 = OpBitwiseOr %2 %52 %53

         %55 = OpBitFieldUExtract %2 %5 %10 %6 ; extracts bit 5 from %5
         %56 = OpBitFieldUExtract %2 %6 %10 %6 ; extracts bit 5 from %6
         %57 = OpBitwiseOr %2 %55 %56

         %58 = OpBitFieldUExtract %2 %5 %11 %6 ; extracts bit 6 from %5
         %59 = OpBitFieldUExtract %2 %6 %11 %6 ; extracts bit 6 from %6
         %60 = OpBitwiseOr %2 %58 %59

         %61 = OpBitFieldUExtract %2 %5 %12 %6 ; extracts bit 7 from %5
         %62 = OpBitFieldUExtract %2 %6 %12 %6 ; extracts bit 7 from %6
         %63 = OpBitwiseOr %2 %61 %62

         %64 = OpBitFieldUExtract %2 %5 %13 %6 ; extracts bit 8 from %5
         %65 = OpBitFieldUExtract %2 %6 %13 %6 ; extracts bit 8 from %6
         %66 = OpBitwiseOr %2 %64 %65

         %67 = OpBitFieldUExtract %2 %5 %14 %6 ; extracts bit 9 from %5
         %68 = OpBitFieldUExtract %2 %6 %14 %6 ; extracts bit 9 from %6
         %69 = OpBitwiseOr %2 %67 %68

         %70 = OpBitFieldUExtract %2 %5 %15 %6 ; extracts bit 10 from %5
         %71 = OpBitFieldUExtract %2 %6 %15 %6 ; extracts bit 10 from %6
         %72 = OpBitwiseOr %2 %70 %71

         %73 = OpBitFieldUExtract %2 %5 %16 %6 ; extracts bit 11 from %5
         %74 = OpBitFieldUExtract %2 %6 %16 %6 ; extracts bit 11 from %6
         %75 = OpBitwiseOr %2 %73 %74

         %76 = OpBitFieldUExtract %2 %5 %17 %6 ; extracts bit 12 from %5
         %77 = OpBitFieldUExtract %2 %6 %17 %6 ; extracts bit 12 from %6
         %78 = OpBitwiseOr %2 %76 %77

         %79 = OpBitFieldUExtract %2 %5 %18 %6 ; extracts bit 13 from %5
         %80 = OpBitFieldUExtract %2 %6 %18 %6 ; extracts bit 13 from %6
         %81 = OpBitwiseOr %2 %79 %80

         %82 = OpBitFieldUExtract %2 %5 %19 %6 ; extracts bit 14 from %5
         %83 = OpBitFieldUExtract %2 %6 %19 %6 ; extracts bit 14 from %6
         %84 = OpBitwiseOr %2 %82 %83

         %85 = OpBitFieldUExtract %2 %5 %20 %6 ; extracts bit 15 from %5
         %86 = OpBitFieldUExtract %2 %6 %20 %6 ; extracts bit 15 from %6
         %87 = OpBitwiseOr %2 %85 %86

         %88 = OpBitFieldUExtract %2 %5 %21 %6 ; extracts bit 16 from %5
         %89 = OpBitFieldUExtract %2 %6 %21 %6 ; extracts bit 16 from %6
         %90 = OpBitwiseOr %2 %88 %89

         %91 = OpBitFieldUExtract %2 %5 %22 %6 ; extracts bit 17 from %5
         %92 = OpBitFieldUExtract %2 %6 %22 %6 ; extracts bit 17 from %6
         %93 = OpBitwiseOr %2 %91 %92

         %94 = OpBitFieldUExtract %2 %5 %23 %6 ; extracts bit 18 from %5
         %95 = OpBitFieldUExtract %2 %6 %23 %6 ; extracts bit 18 from %6
         %96 = OpBitwiseOr %2 %94 %95

         %97 = OpBitFieldUExtract %2 %5 %24 %6 ; extracts bit 19 from %5
         %98 = OpBitFieldUExtract %2 %6 %24 %6 ; extracts bit 19 from %6
         %99 = OpBitwiseOr %2 %97 %98

        %100 = OpBitFieldUExtract %2 %5 %25 %6 ; extracts bit 20 from %5
        %101 = OpBitFieldUExtract %2 %6 %25 %6 ; extracts bit 20 from %6
        %102 = OpBitwiseOr %2 %100 %101

        %103 = OpBitFieldUExtract %2 %5 %26 %6 ; extracts bit 21 from %5
        %104 = OpBitFieldUExtract %2 %6 %26 %6 ; extracts bit 21 from %6
        %105 = OpBitwiseOr %2 %103 %104

        %106 = OpBitFieldUExtract %2 %5 %27 %6 ; extracts bit 22 from %5
        %107 = OpBitFieldUExtract %2 %6 %27 %6 ; extracts bit 22 from %6
        %108 = OpBitwiseOr %2 %106 %107

        %109 = OpBitFieldUExtract %2 %5 %28 %6 ; extracts bit 23 from %5
        %110 = OpBitFieldUExtract %2 %6 %28 %6 ; extracts bit 23 from %6
        %111 = OpBitwiseOr %2 %109 %110

        %112 = OpBitFieldUExtract %2 %5 %29 %6 ; extracts bit 24 from %5
        %113 = OpBitFieldUExtract %2 %6 %29 %6 ; extracts bit 24 from %6
        %114 = OpBitwiseOr %2 %112 %113

        %115 = OpBitFieldUExtract %2 %5 %30 %6 ; extracts bit 25 from %5
        %116 = OpBitFieldUExtract %2 %6 %30 %6 ; extracts bit 25 from %6
        %117 = OpBitwiseOr %2 %115 %116

        %118 = OpBitFieldUExtract %2 %5 %31 %6 ; extracts bit 26 from %5
        %119 = OpBitFieldUExtract %2 %6 %31 %6 ; extracts bit 26 from %6
        %120 = OpBitwiseOr %2 %118 %119

        %121 = OpBitFieldUExtract %2 %5 %32 %6 ; extracts bit 27 from %5
        %122 = OpBitFieldUExtract %2 %6 %32 %6 ; extracts bit 27 from %6
        %123 = OpBitwiseOr %2 %121 %122

        %124 = OpBitFieldUExtract %2 %5 %33 %6 ; extracts bit 28 from %5
        %125 = OpBitFieldUExtract %2 %6 %33 %6 ; extracts bit 28 from %6
        %126 = OpBitwiseOr %2 %124 %125

        %127 = OpBitFieldUExtract %2 %5 %34 %6 ; extracts bit 29 from %5
        %128 = OpBitFieldUExtract %2 %6 %34 %6 ; extracts bit 29 from %6
        %129 = OpBitwiseOr %2 %127 %128

        %130 = OpBitFieldUExtract %2 %5 %35 %6 ; extracts bit 30 from %5
        %131 = OpBitFieldUExtract %2 %6 %35 %6 ; extracts bit 30 from %6
        %132 = OpBitwiseOr %2 %130 %131

        %133 = OpBitFieldUExtract %2 %5 %36 %6 ; extracts bit 31 from %5
        %134 = OpBitFieldUExtract %2 %6 %36 %6 ; extracts bit 31 from %6
        %135 = OpBitwiseOr %2 %133 %134

        %136 = OpBitFieldInsert %2 %42 %45 %6 %6 ; inserts bit 1
        %137 = OpBitFieldInsert %2 %136 %48 %7 %6 ; inserts bit 2
        %138 = OpBitFieldInsert %2 %137 %51 %8 %6 ; inserts bit 3
        %139 = OpBitFieldInsert %2 %138 %54 %9 %6 ; inserts bit 4
        %140 = OpBitFieldInsert %2 %139 %57 %10 %6 ; inserts bit 5
        %141 = OpBitFieldInsert %2 %140 %60 %11 %6 ; inserts bit 6
        %142 = OpBitFieldInsert %2 %141 %63 %12 %6 ; inserts bit 7
        %143 = OpBitFieldInsert %2 %142 %66 %13 %6 ; inserts bit 8
        %144 = OpBitFieldInsert %2 %143 %69 %14 %6 ; inserts bit 9
        %145 = OpBitFieldInsert %2 %144 %72 %15 %6 ; inserts bit 10
        %146 = OpBitFieldInsert %2 %145 %75 %16 %6 ; inserts bit 11
        %147 = OpBitFieldInsert %2 %146 %78 %17 %6 ; inserts bit 12
        %148 = OpBitFieldInsert %2 %147 %81 %18 %6 ; inserts bit 13
        %149 = OpBitFieldInsert %2 %148 %84 %19 %6 ; inserts bit 14
        %150 = OpBitFieldInsert %2 %149 %87 %20 %6 ; inserts bit 15
        %151 = OpBitFieldInsert %2 %150 %90 %21 %6 ; inserts bit 16
        %152 = OpBitFieldInsert %2 %151 %93 %22 %6 ; inserts bit 17
        %153 = OpBitFieldInsert %2 %152 %96 %23 %6 ; inserts bit 18
        %154 = OpBitFieldInsert %2 %153 %99 %24 %6 ; inserts bit 19
        %155 = OpBitFieldInsert %2 %154 %102 %25 %6 ; inserts bit 20
        %156 = OpBitFieldInsert %2 %155 %105 %26 %6 ; inserts bit 21
        %157 = OpBitFieldInsert %2 %156 %108 %27 %6 ; inserts bit 22
        %158 = OpBitFieldInsert %2 %157 %111 %28 %6 ; inserts bit 23
        %159 = OpBitFieldInsert %2 %158 %114 %29 %6 ; inserts bit 24
        %160 = OpBitFieldInsert %2 %159 %117 %30 %6 ; inserts bit 25
        %161 = OpBitFieldInsert %2 %160 %120 %31 %6 ; inserts bit 26
        %162 = OpBitFieldInsert %2 %161 %123 %32 %6 ; inserts bit 27
        %163 = OpBitFieldInsert %2 %162 %126 %33 %6 ; inserts bit 28
        %164 = OpBitFieldInsert %2 %163 %129 %34 %6 ; inserts bit 29
        %165 = OpBitFieldInsert %2 %164 %132 %35 %6 ; inserts bit 30
        %166 = OpBitFieldInsert %2 %165 %135 %36 %6 ; inserts bit 31
         %39 = OpBitwiseOr %2 %5 %6
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
  ASSERT_TRUE(fact_manager.IsSynonymous(MakeDataDescriptor(166, {}),
                                        MakeDataDescriptor(39, {})));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
