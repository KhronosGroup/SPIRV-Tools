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

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);
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

TEST(TransformationAddBitInstructionSynonymTest, AddOpBitwiseOrOpNotSynonym) {
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
         %40 = OpBitwiseXor %2 %7 %8
         %41 = OpBitwiseAnd %2 %9 %10
         %42 = OpNot %2 %11
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_5;
  const auto consumer = nullptr;
  const auto context =
      BuildModule(env, consumer, reference_shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  spvtools::ValidatorOptions validator_options;
  TransformationContext transformation_context(MakeUnique<FactManager>(context.get()), validator_options);

  // Adds OpBitwiseOr synonym.
  auto transformation = TransformationAddBitInstructionSynonym(
      39, {43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
           56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,
           69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
           82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,
           95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
           108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
           121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
           134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
           147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
           160, 161, 162, 163, 164, 165, 166, 167, 168, 169});
  transformation.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(transformation_context.GetFactManager()->IsSynonymous(MakeDataDescriptor(169, {}), MakeDataDescriptor(39, {})));

  // Adds OpBitwiseXor synonym.
  transformation = TransformationAddBitInstructionSynonym(
      40, {170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
           183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
           196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
           209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
           222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
           235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247,
           248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
           261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
           274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286,
           287, 288, 289, 290, 291, 292, 293, 294, 295, 296});
  transformation.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(transformation_context.GetFactManager()->IsSynonymous(MakeDataDescriptor(296, {}), MakeDataDescriptor(40, {})));

  // Adds OpBitwiseAnd synonym.
  transformation = TransformationAddBitInstructionSynonym(
      41, {297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
           310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
           323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
           336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
           349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
           362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374,
           375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
           388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400,
           401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413,
           414, 415, 416, 417, 418, 419, 420, 421, 422, 423});
  transformation.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(transformation_context.GetFactManager()->IsSynonymous(MakeDataDescriptor(423, {}), MakeDataDescriptor(41, {})));

  // Adds OpNot synonym.
  transformation = TransformationAddBitInstructionSynonym(
      42, {424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437,
           438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451,
           452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
           466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
           480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
           494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
           508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518});
  transformation.Apply(context.get(), &transformation_context);
  ASSERT_TRUE(transformation_context.GetFactManager()->IsSynonymous(MakeDataDescriptor(518, {}), MakeDataDescriptor(42, {})));

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

; Add OpBitwiseOr synonym
         %43 = OpBitFieldUExtract %2 %5 %5 %6 ; extracts bit 0 from %5
         %44 = OpBitFieldUExtract %2 %6 %5 %6 ; extracts bit 0 from %6
         %45 = OpBitwiseOr %2 %43 %44

         %46 = OpBitFieldUExtract %2 %5 %6 %6 ; extracts bit 1 from %5
         %47 = OpBitFieldUExtract %2 %6 %6 %6 ; extracts bit 1 from %6
         %48 = OpBitwiseOr %2 %46 %47

         %49 = OpBitFieldUExtract %2 %5 %7 %6 ; extracts bit 2 from %5
         %50 = OpBitFieldUExtract %2 %6 %7 %6 ; extracts bit 2 from %6
         %51 = OpBitwiseOr %2 %49 %50

         %52 = OpBitFieldUExtract %2 %5 %8 %6 ; extracts bit 3 from %5
         %53 = OpBitFieldUExtract %2 %6 %8 %6 ; extracts bit 3 from %6
         %54 = OpBitwiseOr %2 %52 %53

         %55 = OpBitFieldUExtract %2 %5 %9 %6 ; extracts bit 4 from %5
         %56 = OpBitFieldUExtract %2 %6 %9 %6 ; extracts bit 4 from %6
         %57 = OpBitwiseOr %2 %55 %56

         %58 = OpBitFieldUExtract %2 %5 %10 %6 ; extracts bit 5 from %5
         %59 = OpBitFieldUExtract %2 %6 %10 %6 ; extracts bit 5 from %6
         %60 = OpBitwiseOr %2 %58 %59

         %61 = OpBitFieldUExtract %2 %5 %11 %6 ; extracts bit 6 from %5
         %62 = OpBitFieldUExtract %2 %6 %11 %6 ; extracts bit 6 from %6
         %63 = OpBitwiseOr %2 %61 %62

         %64 = OpBitFieldUExtract %2 %5 %12 %6 ; extracts bit 7 from %5
         %65 = OpBitFieldUExtract %2 %6 %12 %6 ; extracts bit 7 from %6
         %66 = OpBitwiseOr %2 %64 %65

         %67 = OpBitFieldUExtract %2 %5 %13 %6 ; extracts bit 8 from %5
         %68 = OpBitFieldUExtract %2 %6 %13 %6 ; extracts bit 8 from %6
         %69 = OpBitwiseOr %2 %67 %68

         %70 = OpBitFieldUExtract %2 %5 %14 %6 ; extracts bit 9 from %5
         %71 = OpBitFieldUExtract %2 %6 %14 %6 ; extracts bit 9 from %6
         %72 = OpBitwiseOr %2 %70 %71

         %73 = OpBitFieldUExtract %2 %5 %15 %6 ; extracts bit 10 from %5
         %74 = OpBitFieldUExtract %2 %6 %15 %6 ; extracts bit 10 from %6
         %75 = OpBitwiseOr %2 %73 %74

         %76 = OpBitFieldUExtract %2 %5 %16 %6 ; extracts bit 11 from %5
         %77 = OpBitFieldUExtract %2 %6 %16 %6 ; extracts bit 11 from %6
         %78 = OpBitwiseOr %2 %76 %77

         %79 = OpBitFieldUExtract %2 %5 %17 %6 ; extracts bit 12 from %5
         %80 = OpBitFieldUExtract %2 %6 %17 %6 ; extracts bit 12 from %6
         %81 = OpBitwiseOr %2 %79 %80

         %82 = OpBitFieldUExtract %2 %5 %18 %6 ; extracts bit 13 from %5
         %83 = OpBitFieldUExtract %2 %6 %18 %6 ; extracts bit 13 from %6
         %84 = OpBitwiseOr %2 %82 %83

         %85 = OpBitFieldUExtract %2 %5 %19 %6 ; extracts bit 14 from %5
         %86 = OpBitFieldUExtract %2 %6 %19 %6 ; extracts bit 14 from %6
         %87 = OpBitwiseOr %2 %85 %86

         %88 = OpBitFieldUExtract %2 %5 %20 %6 ; extracts bit 15 from %5
         %89 = OpBitFieldUExtract %2 %6 %20 %6 ; extracts bit 15 from %6
         %90 = OpBitwiseOr %2 %88 %89

         %91 = OpBitFieldUExtract %2 %5 %21 %6 ; extracts bit 16 from %5
         %92 = OpBitFieldUExtract %2 %6 %21 %6 ; extracts bit 16 from %6
         %93 = OpBitwiseOr %2 %91 %92

         %94 = OpBitFieldUExtract %2 %5 %22 %6 ; extracts bit 17 from %5
         %95 = OpBitFieldUExtract %2 %6 %22 %6 ; extracts bit 17 from %6
         %96 = OpBitwiseOr %2 %94 %95

         %97 = OpBitFieldUExtract %2 %5 %23 %6 ; extracts bit 18 from %5
         %98 = OpBitFieldUExtract %2 %6 %23 %6 ; extracts bit 18 from %6
         %99 = OpBitwiseOr %2 %97 %98

        %100 = OpBitFieldUExtract %2 %5 %24 %6 ; extracts bit 19 from %5
        %101 = OpBitFieldUExtract %2 %6 %24 %6 ; extracts bit 19 from %6
        %102 = OpBitwiseOr %2 %100 %101

        %103 = OpBitFieldUExtract %2 %5 %25 %6 ; extracts bit 20 from %5
        %104 = OpBitFieldUExtract %2 %6 %25 %6 ; extracts bit 20 from %6
        %105 = OpBitwiseOr %2 %103 %104

        %106 = OpBitFieldUExtract %2 %5 %26 %6 ; extracts bit 21 from %5
        %107 = OpBitFieldUExtract %2 %6 %26 %6 ; extracts bit 21 from %6
        %108 = OpBitwiseOr %2 %106 %107

        %109 = OpBitFieldUExtract %2 %5 %27 %6 ; extracts bit 22 from %5
        %110 = OpBitFieldUExtract %2 %6 %27 %6 ; extracts bit 22 from %6
        %111 = OpBitwiseOr %2 %109 %110

        %112 = OpBitFieldUExtract %2 %5 %28 %6 ; extracts bit 23 from %5
        %113 = OpBitFieldUExtract %2 %6 %28 %6 ; extracts bit 23 from %6
        %114 = OpBitwiseOr %2 %112 %113

        %115 = OpBitFieldUExtract %2 %5 %29 %6 ; extracts bit 24 from %5
        %116 = OpBitFieldUExtract %2 %6 %29 %6 ; extracts bit 24 from %6
        %117 = OpBitwiseOr %2 %115 %116

        %118 = OpBitFieldUExtract %2 %5 %30 %6 ; extracts bit 25 from %5
        %119 = OpBitFieldUExtract %2 %6 %30 %6 ; extracts bit 25 from %6
        %120 = OpBitwiseOr %2 %118 %119

        %121 = OpBitFieldUExtract %2 %5 %31 %6 ; extracts bit 26 from %5
        %122 = OpBitFieldUExtract %2 %6 %31 %6 ; extracts bit 26 from %6
        %123 = OpBitwiseOr %2 %121 %122

        %124 = OpBitFieldUExtract %2 %5 %32 %6 ; extracts bit 27 from %5
        %125 = OpBitFieldUExtract %2 %6 %32 %6 ; extracts bit 27 from %6
        %126 = OpBitwiseOr %2 %124 %125

        %127 = OpBitFieldUExtract %2 %5 %33 %6 ; extracts bit 28 from %5
        %128 = OpBitFieldUExtract %2 %6 %33 %6 ; extracts bit 28 from %6
        %129 = OpBitwiseOr %2 %127 %128

        %130 = OpBitFieldUExtract %2 %5 %34 %6 ; extracts bit 29 from %5
        %131 = OpBitFieldUExtract %2 %6 %34 %6 ; extracts bit 29 from %6
        %132 = OpBitwiseOr %2 %130 %131

        %133 = OpBitFieldUExtract %2 %5 %35 %6 ; extracts bit 30 from %5
        %134 = OpBitFieldUExtract %2 %6 %35 %6 ; extracts bit 30 from %6
        %135 = OpBitwiseOr %2 %133 %134

        %136 = OpBitFieldUExtract %2 %5 %36 %6 ; extracts bit 31 from %5
        %137 = OpBitFieldUExtract %2 %6 %36 %6 ; extracts bit 31 from %6
        %138 = OpBitwiseOr %2 %136 %137

        %139 = OpBitFieldInsert %2 %45 %48 %6 %6 ; inserts bit 1
        %140 = OpBitFieldInsert %2 %139 %51 %7 %6 ; inserts bit 2
        %141 = OpBitFieldInsert %2 %140 %54 %8 %6 ; inserts bit 3
        %142 = OpBitFieldInsert %2 %141 %57 %9 %6 ; inserts bit 4
        %143 = OpBitFieldInsert %2 %142 %60 %10 %6 ; inserts bit 5
        %144 = OpBitFieldInsert %2 %143 %63 %11 %6 ; inserts bit 6
        %145 = OpBitFieldInsert %2 %144 %66 %12 %6 ; inserts bit 7
        %146 = OpBitFieldInsert %2 %145 %69 %13 %6 ; inserts bit 8
        %147 = OpBitFieldInsert %2 %146 %72 %14 %6 ; inserts bit 9
        %148 = OpBitFieldInsert %2 %147 %75 %15 %6 ; inserts bit 10
        %149 = OpBitFieldInsert %2 %148 %78 %16 %6 ; inserts bit 11
        %150 = OpBitFieldInsert %2 %149 %81 %17 %6 ; inserts bit 12
        %151 = OpBitFieldInsert %2 %150 %84 %18 %6 ; inserts bit 13
        %152 = OpBitFieldInsert %2 %151 %87 %19 %6 ; inserts bit 14
        %153 = OpBitFieldInsert %2 %152 %90 %20 %6 ; inserts bit 15
        %154 = OpBitFieldInsert %2 %153 %93 %21 %6 ; inserts bit 16
        %155 = OpBitFieldInsert %2 %154 %96 %22 %6 ; inserts bit 17
        %156 = OpBitFieldInsert %2 %155 %99 %23 %6 ; inserts bit 18
        %157 = OpBitFieldInsert %2 %156 %102 %24 %6 ; inserts bit 19
        %158 = OpBitFieldInsert %2 %157 %105 %25 %6 ; inserts bit 20
        %159 = OpBitFieldInsert %2 %158 %108 %26 %6 ; inserts bit 21
        %160 = OpBitFieldInsert %2 %159 %111 %27 %6 ; inserts bit 22
        %161 = OpBitFieldInsert %2 %160 %114 %28 %6 ; inserts bit 23
        %162 = OpBitFieldInsert %2 %161 %117 %29 %6 ; inserts bit 24
        %163 = OpBitFieldInsert %2 %162 %120 %30 %6 ; inserts bit 25
        %164 = OpBitFieldInsert %2 %163 %123 %31 %6 ; inserts bit 26
        %165 = OpBitFieldInsert %2 %164 %126 %32 %6 ; inserts bit 27
        %166 = OpBitFieldInsert %2 %165 %129 %33 %6 ; inserts bit 28
        %167 = OpBitFieldInsert %2 %166 %132 %34 %6 ; inserts bit 29
        %168 = OpBitFieldInsert %2 %167 %135 %35 %6 ; inserts bit 30
        %169 = OpBitFieldInsert %2 %168 %138 %36 %6 ; inserts bit 31
         %39 = OpBitwiseOr %2 %5 %6

; Add OpBitwiseXor synonym
        %170 = OpBitFieldUExtract %2 %7 %5 %6 ; extracts bit 0 from %7
        %171 = OpBitFieldUExtract %2 %8 %5 %6 ; extracts bit 0 from %8
        %172 = OpBitwiseXor %2 %170 %171

        %173 = OpBitFieldUExtract %2 %7 %6 %6 ; extracts bit 1 from %7
        %174 = OpBitFieldUExtract %2 %8 %6 %6 ; extracts bit 1 from %8
        %175 = OpBitwiseXor %2 %173 %174

        %176 = OpBitFieldUExtract %2 %7 %7 %6 ; extracts bit 2 from %7
        %177 = OpBitFieldUExtract %2 %8 %7 %6 ; extracts bit 2 from %8
        %178 = OpBitwiseXor %2 %176 %177

        %179 = OpBitFieldUExtract %2 %7 %8 %6 ; extracts bit 3 from %7
        %180 = OpBitFieldUExtract %2 %8 %8 %6 ; extracts bit 3 from %8
        %181 = OpBitwiseXor %2 %179 %180

        %182 = OpBitFieldUExtract %2 %7 %9 %6 ; extracts bit 4 from %7
        %183 = OpBitFieldUExtract %2 %8 %9 %6 ; extracts bit 4 from %8
        %184 = OpBitwiseXor %2 %182 %183

        %185 = OpBitFieldUExtract %2 %7 %10 %6 ; extracts bit 5 from %7
        %186 = OpBitFieldUExtract %2 %8 %10 %6 ; extracts bit 5 from %8
        %187 = OpBitwiseXor %2 %185 %186

        %188 = OpBitFieldUExtract %2 %7 %11 %6 ; extracts bit 6 from %7
        %189 = OpBitFieldUExtract %2 %8 %11 %6 ; extracts bit 6 from %8
        %190 = OpBitwiseXor %2 %188 %189

        %191 = OpBitFieldUExtract %2 %7 %12 %6 ; extracts bit 7 from %7
        %192 = OpBitFieldUExtract %2 %8 %12 %6 ; extracts bit 7 from %8
        %193 = OpBitwiseXor %2 %191 %192

        %194 = OpBitFieldUExtract %2 %7 %13 %6 ; extracts bit 8 from %7
        %195 = OpBitFieldUExtract %2 %8 %13 %6 ; extracts bit 8 from %8
        %196 = OpBitwiseXor %2 %194 %195

        %197 = OpBitFieldUExtract %2 %7 %14 %6 ; extracts bit 9 from %7
        %198 = OpBitFieldUExtract %2 %8 %14 %6 ; extracts bit 9 from %8
        %199 = OpBitwiseXor %2 %197 %198

        %200 = OpBitFieldUExtract %2 %7 %15 %6 ; extracts bit 10 from %7
        %201 = OpBitFieldUExtract %2 %8 %15 %6 ; extracts bit 10 from %8
        %202 = OpBitwiseXor %2 %200 %201

        %203 = OpBitFieldUExtract %2 %7 %16 %6 ; extracts bit 11 from %7
        %204 = OpBitFieldUExtract %2 %8 %16 %6 ; extracts bit 11 from %8
        %205 = OpBitwiseXor %2 %203 %204

        %206 = OpBitFieldUExtract %2 %7 %17 %6 ; extracts bit 12 from %7
        %207 = OpBitFieldUExtract %2 %8 %17 %6 ; extracts bit 12 from %8
        %208 = OpBitwiseXor %2 %206 %207

        %209 = OpBitFieldUExtract %2 %7 %18 %6 ; extracts bit 13 from %7
        %210 = OpBitFieldUExtract %2 %8 %18 %6 ; extracts bit 13 from %8
        %211 = OpBitwiseXor %2 %209 %210

        %212 = OpBitFieldUExtract %2 %7 %19 %6 ; extracts bit 14 from %7
        %213 = OpBitFieldUExtract %2 %8 %19 %6 ; extracts bit 14 from %8
        %214 = OpBitwiseXor %2 %212 %213

        %215 = OpBitFieldUExtract %2 %7 %20 %6 ; extracts bit 15 from %7
        %216 = OpBitFieldUExtract %2 %8 %20 %6 ; extracts bit 15 from %8
        %217 = OpBitwiseXor %2 %215 %216

        %218 = OpBitFieldUExtract %2 %7 %21 %6 ; extracts bit 16 from %7
        %219 = OpBitFieldUExtract %2 %8 %21 %6 ; extracts bit 16 from %8
        %220 = OpBitwiseXor %2 %218 %219

        %221 = OpBitFieldUExtract %2 %7 %22 %6 ; extracts bit 17 from %7
        %222 = OpBitFieldUExtract %2 %8 %22 %6 ; extracts bit 17 from %8
        %223 = OpBitwiseXor %2 %221 %222

        %224 = OpBitFieldUExtract %2 %7 %23 %6 ; extracts bit 18 from %7
        %225 = OpBitFieldUExtract %2 %8 %23 %6 ; extracts bit 18 from %8
        %226 = OpBitwiseXor %2 %224 %225

        %227 = OpBitFieldUExtract %2 %7 %24 %6 ; extracts bit 19 from %7
        %228 = OpBitFieldUExtract %2 %8 %24 %6 ; extracts bit 19 from %8
        %229 = OpBitwiseXor %2 %227 %228

        %230 = OpBitFieldUExtract %2 %7 %25 %6 ; extracts bit 20 from %7
        %231 = OpBitFieldUExtract %2 %8 %25 %6 ; extracts bit 20 from %8
        %232 = OpBitwiseXor %2 %230 %231

        %233 = OpBitFieldUExtract %2 %7 %26 %6 ; extracts bit 21 from %7
        %234 = OpBitFieldUExtract %2 %8 %26 %6 ; extracts bit 21 from %8
        %235 = OpBitwiseXor %2 %233 %234

        %236 = OpBitFieldUExtract %2 %7 %27 %6 ; extracts bit 22 from %7
        %237 = OpBitFieldUExtract %2 %8 %27 %6 ; extracts bit 22 from %8
        %238 = OpBitwiseXor %2 %236 %237

        %239 = OpBitFieldUExtract %2 %7 %28 %6 ; extracts bit 23 from %7
        %240 = OpBitFieldUExtract %2 %8 %28 %6 ; extracts bit 23 from %8
        %241 = OpBitwiseXor %2 %239 %240

        %242 = OpBitFieldUExtract %2 %7 %29 %6 ; extracts bit 24 from %7
        %243 = OpBitFieldUExtract %2 %8 %29 %6 ; extracts bit 24 from %8
        %244 = OpBitwiseXor %2 %242 %243

        %245 = OpBitFieldUExtract %2 %7 %30 %6 ; extracts bit 25 from %7
        %246 = OpBitFieldUExtract %2 %8 %30 %6 ; extracts bit 25 from %8
        %247 = OpBitwiseXor %2 %245 %246

        %248 = OpBitFieldUExtract %2 %7 %31 %6 ; extracts bit 26 from %7
        %249 = OpBitFieldUExtract %2 %8 %31 %6 ; extracts bit 26 from %8
        %250 = OpBitwiseXor %2 %248 %249

        %251 = OpBitFieldUExtract %2 %7 %32 %6 ; extracts bit 27 from %7
        %252 = OpBitFieldUExtract %2 %8 %32 %6 ; extracts bit 27 from %8
        %253 = OpBitwiseXor %2 %251 %252

        %254 = OpBitFieldUExtract %2 %7 %33 %6 ; extracts bit 28 from %7
        %255 = OpBitFieldUExtract %2 %8 %33 %6 ; extracts bit 28 from %8
        %256 = OpBitwiseXor %2 %254 %255

        %257 = OpBitFieldUExtract %2 %7 %34 %6 ; extracts bit 29 from %7
        %258 = OpBitFieldUExtract %2 %8 %34 %6 ; extracts bit 29 from %8
        %259 = OpBitwiseXor %2 %257 %258

        %260 = OpBitFieldUExtract %2 %7 %35 %6 ; extracts bit 30 from %7
        %261 = OpBitFieldUExtract %2 %8 %35 %6 ; extracts bit 30 from %8
        %262 = OpBitwiseXor %2 %260 %261

        %263 = OpBitFieldUExtract %2 %7 %36 %6 ; extracts bit 31 from %7
        %264 = OpBitFieldUExtract %2 %8 %36 %6 ; extracts bit 31 from %8
        %265 = OpBitwiseXor %2 %263 %264

        %266 = OpBitFieldInsert %2 %172 %175 %6 %6 ; inserts bit 1
        %267 = OpBitFieldInsert %2 %266 %178 %7 %6 ; inserts bit 2
        %268 = OpBitFieldInsert %2 %267 %181 %8 %6 ; inserts bit 3
        %269 = OpBitFieldInsert %2 %268 %184 %9 %6 ; inserts bit 4
        %270 = OpBitFieldInsert %2 %269 %187 %10 %6 ; inserts bit 5
        %271 = OpBitFieldInsert %2 %270 %190 %11 %6 ; inserts bit 6
        %272 = OpBitFieldInsert %2 %271 %193 %12 %6 ; inserts bit 7
        %273 = OpBitFieldInsert %2 %272 %196 %13 %6 ; inserts bit 8
        %274 = OpBitFieldInsert %2 %273 %199 %14 %6 ; inserts bit 9
        %275 = OpBitFieldInsert %2 %274 %202 %15 %6 ; inserts bit 10
        %276 = OpBitFieldInsert %2 %275 %205 %16 %6 ; inserts bit 11
        %277 = OpBitFieldInsert %2 %276 %208 %17 %6 ; inserts bit 12
        %278 = OpBitFieldInsert %2 %277 %211 %18 %6 ; inserts bit 13
        %279 = OpBitFieldInsert %2 %278 %214 %19 %6 ; inserts bit 14
        %280 = OpBitFieldInsert %2 %279 %217 %20 %6 ; inserts bit 15
        %281 = OpBitFieldInsert %2 %280 %220 %21 %6 ; inserts bit 16
        %282 = OpBitFieldInsert %2 %281 %223 %22 %6 ; inserts bit 17
        %283 = OpBitFieldInsert %2 %282 %226 %23 %6 ; inserts bit 18
        %284 = OpBitFieldInsert %2 %283 %229 %24 %6 ; inserts bit 19
        %285 = OpBitFieldInsert %2 %284 %232 %25 %6 ; inserts bit 20
        %286 = OpBitFieldInsert %2 %285 %235 %26 %6 ; inserts bit 21
        %287 = OpBitFieldInsert %2 %286 %238 %27 %6 ; inserts bit 22
        %288 = OpBitFieldInsert %2 %287 %241 %28 %6 ; inserts bit 23
        %289 = OpBitFieldInsert %2 %288 %244 %29 %6 ; inserts bit 24
        %290 = OpBitFieldInsert %2 %289 %247 %30 %6 ; inserts bit 25
        %291 = OpBitFieldInsert %2 %290 %250 %31 %6 ; inserts bit 26
        %292 = OpBitFieldInsert %2 %291 %253 %32 %6 ; inserts bit 27
        %293 = OpBitFieldInsert %2 %292 %256 %33 %6 ; inserts bit 28
        %294 = OpBitFieldInsert %2 %293 %259 %34 %6 ; inserts bit 29
        %295 = OpBitFieldInsert %2 %294 %262 %35 %6 ; inserts bit 30
        %296 = OpBitFieldInsert %2 %295 %265 %36 %6 ; inserts bit 31
         %40 = OpBitwiseXor %2 %7 %8

; Add OpBitwiseAnd synonym
        %297 = OpBitFieldUExtract %2 %9 %5 %6 ; extracts bit 0 from %9
        %298 = OpBitFieldUExtract %2 %10 %5 %6 ; extracts bit 0 from %10
        %299 = OpBitwiseAnd %2 %297 %298

        %300 = OpBitFieldUExtract %2 %9 %6 %6 ; extracts bit 1 from %9
        %301 = OpBitFieldUExtract %2 %10 %6 %6 ; extracts bit 1 from %10
        %302 = OpBitwiseAnd %2 %300 %301

        %303 = OpBitFieldUExtract %2 %9 %7 %6 ; extracts bit 2 from %9
        %304 = OpBitFieldUExtract %2 %10 %7 %6 ; extracts bit 2 from %10
        %305 = OpBitwiseAnd %2 %303 %304

        %306 = OpBitFieldUExtract %2 %9 %8 %6 ; extracts bit 3 from %9
        %307 = OpBitFieldUExtract %2 %10 %8 %6 ; extracts bit 3 from %10
        %308 = OpBitwiseAnd %2 %306 %307

        %309 = OpBitFieldUExtract %2 %9 %9 %6 ; extracts bit 4 from %9
        %310 = OpBitFieldUExtract %2 %10 %9 %6 ; extracts bit 4 from %10
        %311 = OpBitwiseAnd %2 %309 %310

        %312 = OpBitFieldUExtract %2 %9 %10 %6 ; extracts bit 5 from %9
        %313 = OpBitFieldUExtract %2 %10 %10 %6 ; extracts bit 5 from %10
        %314 = OpBitwiseAnd %2 %312 %313

        %315 = OpBitFieldUExtract %2 %9 %11 %6 ; extracts bit 6 from %9
        %316 = OpBitFieldUExtract %2 %10 %11 %6 ; extracts bit 6 from %10
        %317 = OpBitwiseAnd %2 %315 %316

        %318 = OpBitFieldUExtract %2 %9 %12 %6 ; extracts bit 7 from %9
        %319 = OpBitFieldUExtract %2 %10 %12 %6 ; extracts bit 7 from %10
        %320 = OpBitwiseAnd %2 %318 %319

        %321 = OpBitFieldUExtract %2 %9 %13 %6 ; extracts bit 8 from %9
        %322 = OpBitFieldUExtract %2 %10 %13 %6 ; extracts bit 8 from %10
        %323 = OpBitwiseAnd %2 %321 %322

        %324 = OpBitFieldUExtract %2 %9 %14 %6 ; extracts bit 9 from %9
        %325 = OpBitFieldUExtract %2 %10 %14 %6 ; extracts bit 9 from %10
        %326 = OpBitwiseAnd %2 %324 %325

        %327 = OpBitFieldUExtract %2 %9 %15 %6 ; extracts bit 10 from %9
        %328 = OpBitFieldUExtract %2 %10 %15 %6 ; extracts bit 10 from %10
        %329 = OpBitwiseAnd %2 %327 %328

        %330 = OpBitFieldUExtract %2 %9 %16 %6 ; extracts bit 11 from %9
        %331 = OpBitFieldUExtract %2 %10 %16 %6 ; extracts bit 11 from %10
        %332 = OpBitwiseAnd %2 %330 %331

        %333 = OpBitFieldUExtract %2 %9 %17 %6 ; extracts bit 12 from %9
        %334 = OpBitFieldUExtract %2 %10 %17 %6 ; extracts bit 12 from %10
        %335 = OpBitwiseAnd %2 %333 %334

        %336 = OpBitFieldUExtract %2 %9 %18 %6 ; extracts bit 13 from %9
        %337 = OpBitFieldUExtract %2 %10 %18 %6 ; extracts bit 13 from %10
        %338 = OpBitwiseAnd %2 %336 %337

        %339 = OpBitFieldUExtract %2 %9 %19 %6 ; extracts bit 14 from %9
        %340 = OpBitFieldUExtract %2 %10 %19 %6 ; extracts bit 14 from %10
        %341 = OpBitwiseAnd %2 %339 %340

        %342 = OpBitFieldUExtract %2 %9 %20 %6 ; extracts bit 15 from %9
        %343 = OpBitFieldUExtract %2 %10 %20 %6 ; extracts bit 15 from %10
        %344 = OpBitwiseAnd %2 %342 %343

        %345 = OpBitFieldUExtract %2 %9 %21 %6 ; extracts bit 16 from %9
        %346 = OpBitFieldUExtract %2 %10 %21 %6 ; extracts bit 16 from %10
        %347 = OpBitwiseAnd %2 %345 %346

        %348 = OpBitFieldUExtract %2 %9 %22 %6 ; extracts bit 17 from %9
        %349 = OpBitFieldUExtract %2 %10 %22 %6 ; extracts bit 17 from %10
        %350 = OpBitwiseAnd %2 %348 %349

        %351 = OpBitFieldUExtract %2 %9 %23 %6 ; extracts bit 18 from %9
        %352 = OpBitFieldUExtract %2 %10 %23 %6 ; extracts bit 18 from %10
        %353 = OpBitwiseAnd %2 %351 %352

        %354 = OpBitFieldUExtract %2 %9 %24 %6 ; extracts bit 19 from %9
        %355 = OpBitFieldUExtract %2 %10 %24 %6 ; extracts bit 19 from %10
        %356 = OpBitwiseAnd %2 %354 %355

        %357 = OpBitFieldUExtract %2 %9 %25 %6 ; extracts bit 20 from %9
        %358 = OpBitFieldUExtract %2 %10 %25 %6 ; extracts bit 20 from %10
        %359 = OpBitwiseAnd %2 %357 %358

        %360 = OpBitFieldUExtract %2 %9 %26 %6 ; extracts bit 21 from %9
        %361 = OpBitFieldUExtract %2 %10 %26 %6 ; extracts bit 21 from %10
        %362 = OpBitwiseAnd %2 %360 %361

        %363 = OpBitFieldUExtract %2 %9 %27 %6 ; extracts bit 22 from %9
        %364 = OpBitFieldUExtract %2 %10 %27 %6 ; extracts bit 22 from %10
        %365 = OpBitwiseAnd %2 %363 %364

        %366 = OpBitFieldUExtract %2 %9 %28 %6 ; extracts bit 23 from %9
        %367 = OpBitFieldUExtract %2 %10 %28 %6 ; extracts bit 23 from %10
        %368 = OpBitwiseAnd %2 %366 %367

        %369 = OpBitFieldUExtract %2 %9 %29 %6 ; extracts bit 24 from %9
        %370 = OpBitFieldUExtract %2 %10 %29 %6 ; extracts bit 24 from %10
        %371 = OpBitwiseAnd %2 %369 %370

        %372 = OpBitFieldUExtract %2 %9 %30 %6 ; extracts bit 25 from %9
        %373 = OpBitFieldUExtract %2 %10 %30 %6 ; extracts bit 25 from %10
        %374 = OpBitwiseAnd %2 %372 %373

        %375 = OpBitFieldUExtract %2 %9 %31 %6 ; extracts bit 26 from %9
        %376 = OpBitFieldUExtract %2 %10 %31 %6 ; extracts bit 26 from %10
        %377 = OpBitwiseAnd %2 %375 %376

        %378 = OpBitFieldUExtract %2 %9 %32 %6 ; extracts bit 27 from %9
        %379 = OpBitFieldUExtract %2 %10 %32 %6 ; extracts bit 27 from %10
        %380 = OpBitwiseAnd %2 %378 %379

        %381 = OpBitFieldUExtract %2 %9 %33 %6 ; extracts bit 28 from %9
        %382 = OpBitFieldUExtract %2 %10 %33 %6 ; extracts bit 28 from %10
        %383 = OpBitwiseAnd %2 %381 %382

        %384 = OpBitFieldUExtract %2 %9 %34 %6 ; extracts bit 29 from %9
        %385 = OpBitFieldUExtract %2 %10 %34 %6 ; extracts bit 29 from %10
        %386 = OpBitwiseAnd %2 %384 %385

        %387 = OpBitFieldUExtract %2 %9 %35 %6 ; extracts bit 30 from %9
        %388 = OpBitFieldUExtract %2 %10 %35 %6 ; extracts bit 30 from %10
        %389 = OpBitwiseAnd %2 %387 %388

        %390 = OpBitFieldUExtract %2 %9 %36 %6 ; extracts bit 31 from %9
        %391 = OpBitFieldUExtract %2 %10 %36 %6 ; extracts bit 31 from %10
        %392 = OpBitwiseAnd %2 %390 %391

        %393 = OpBitFieldInsert %2 %299 %302 %6 %6 ; inserts bit 1
        %394 = OpBitFieldInsert %2 %393 %305 %7 %6 ; inserts bit 2
        %395 = OpBitFieldInsert %2 %394 %308 %8 %6 ; inserts bit 3
        %396 = OpBitFieldInsert %2 %395 %311 %9 %6 ; inserts bit 4
        %397 = OpBitFieldInsert %2 %396 %314 %10 %6 ; inserts bit 5
        %398 = OpBitFieldInsert %2 %397 %317 %11 %6 ; inserts bit 6
        %399 = OpBitFieldInsert %2 %398 %320 %12 %6 ; inserts bit 7
        %400 = OpBitFieldInsert %2 %399 %323 %13 %6 ; inserts bit 8
        %401 = OpBitFieldInsert %2 %400 %326 %14 %6 ; inserts bit 9
        %402 = OpBitFieldInsert %2 %401 %329 %15 %6 ; inserts bit 10
        %403 = OpBitFieldInsert %2 %402 %332 %16 %6 ; inserts bit 11
        %404 = OpBitFieldInsert %2 %403 %335 %17 %6 ; inserts bit 12
        %405 = OpBitFieldInsert %2 %404 %338 %18 %6 ; inserts bit 13
        %406 = OpBitFieldInsert %2 %405 %341 %19 %6 ; inserts bit 14
        %407 = OpBitFieldInsert %2 %406 %344 %20 %6 ; inserts bit 15
        %408 = OpBitFieldInsert %2 %407 %347 %21 %6 ; inserts bit 16
        %409 = OpBitFieldInsert %2 %408 %350 %22 %6 ; inserts bit 17
        %410 = OpBitFieldInsert %2 %409 %353 %23 %6 ; inserts bit 18
        %411 = OpBitFieldInsert %2 %410 %356 %24 %6 ; inserts bit 19
        %412 = OpBitFieldInsert %2 %411 %359 %25 %6 ; inserts bit 20
        %413 = OpBitFieldInsert %2 %412 %362 %26 %6 ; inserts bit 21
        %414 = OpBitFieldInsert %2 %413 %365 %27 %6 ; inserts bit 22
        %415 = OpBitFieldInsert %2 %414 %368 %28 %6 ; inserts bit 23
        %416 = OpBitFieldInsert %2 %415 %371 %29 %6 ; inserts bit 24
        %417 = OpBitFieldInsert %2 %416 %374 %30 %6 ; inserts bit 25
        %418 = OpBitFieldInsert %2 %417 %377 %31 %6 ; inserts bit 26
        %419 = OpBitFieldInsert %2 %418 %380 %32 %6 ; inserts bit 27
        %420 = OpBitFieldInsert %2 %419 %383 %33 %6 ; inserts bit 28
        %421 = OpBitFieldInsert %2 %420 %386 %34 %6 ; inserts bit 29
        %422 = OpBitFieldInsert %2 %421 %389 %35 %6 ; inserts bit 30
        %423 = OpBitFieldInsert %2 %422 %392 %36 %6 ; inserts bit 31
         %41 = OpBitwiseAnd %2 %9 %10

; Add OpNot synonym
        %424 = OpBitFieldUExtract %2 %11 %5 %6 ; extracts bit 0 from %11
        %425 = OpNot %2 %424

        %426 = OpBitFieldUExtract %2 %11 %6 %6 ; extracts bit 1 from %11
        %427 = OpNot %2 %426

        %428 = OpBitFieldUExtract %2 %11 %7 %6 ; extracts bit 2 from %11
        %429 = OpNot %2 %428

        %430 = OpBitFieldUExtract %2 %11 %8 %6 ; extracts bit 3 from %11
        %431 = OpNot %2 %430

        %432 = OpBitFieldUExtract %2 %11 %9 %6 ; extracts bit 4 from %11
        %433 = OpNot %2 %432

        %434 = OpBitFieldUExtract %2 %11 %10 %6 ; extracts bit 5 from %11
        %435 = OpNot %2 %434

        %436 = OpBitFieldUExtract %2 %11 %11 %6 ; extracts bit 6 from %11
        %437 = OpNot %2 %436

        %438 = OpBitFieldUExtract %2 %11 %12 %6 ; extracts bit 7 from %11
        %439 = OpNot %2 %438

        %440 = OpBitFieldUExtract %2 %11 %13 %6 ; extracts bit 8 from %11
        %441 = OpNot %2 %440

        %442 = OpBitFieldUExtract %2 %11 %14 %6 ; extracts bit 9 from %11
        %443 = OpNot %2 %442

        %444 = OpBitFieldUExtract %2 %11 %15 %6 ; extracts bit 10 from %11
        %445 = OpNot %2 %444

        %446 = OpBitFieldUExtract %2 %11 %16 %6 ; extracts bit 11 from %11
        %447 = OpNot %2 %446

        %448 = OpBitFieldUExtract %2 %11 %17 %6 ; extracts bit 12 from %11
        %449 = OpNot %2 %448

        %450 = OpBitFieldUExtract %2 %11 %18 %6 ; extracts bit 13 from %11
        %451 = OpNot %2 %450

        %452 = OpBitFieldUExtract %2 %11 %19 %6 ; extracts bit 14 from %11
        %453 = OpNot %2 %452

        %454 = OpBitFieldUExtract %2 %11 %20 %6 ; extracts bit 15 from %11
        %455 = OpNot %2 %454

        %456 = OpBitFieldUExtract %2 %11 %21 %6 ; extracts bit 16 from %11
        %457 = OpNot %2 %456

        %458 = OpBitFieldUExtract %2 %11 %22 %6 ; extracts bit 17 from %11
        %459 = OpNot %2 %458

        %460 = OpBitFieldUExtract %2 %11 %23 %6 ; extracts bit 18 from %11
        %461 = OpNot %2 %460

        %462 = OpBitFieldUExtract %2 %11 %24 %6 ; extracts bit 19 from %11
        %463 = OpNot %2 %462

        %464 = OpBitFieldUExtract %2 %11 %25 %6 ; extracts bit 20 from %11
        %465 = OpNot %2 %464

        %466 = OpBitFieldUExtract %2 %11 %26 %6 ; extracts bit 21 from %11
        %467 = OpNot %2 %466

        %468 = OpBitFieldUExtract %2 %11 %27 %6 ; extracts bit 22 from %11
        %469 = OpNot %2 %468

        %470 = OpBitFieldUExtract %2 %11 %28 %6 ; extracts bit 23 from %11
        %471 = OpNot %2 %470

        %472 = OpBitFieldUExtract %2 %11 %29 %6 ; extracts bit 24 from %11
        %473 = OpNot %2 %472

        %474 = OpBitFieldUExtract %2 %11 %30 %6 ; extracts bit 25 from %11
        %475 = OpNot %2 %474

        %476 = OpBitFieldUExtract %2 %11 %31 %6 ; extracts bit 26 from %11
        %477 = OpNot %2 %476

        %478 = OpBitFieldUExtract %2 %11 %32 %6 ; extracts bit 27 from %11
        %479 = OpNot %2 %478

        %480 = OpBitFieldUExtract %2 %11 %33 %6 ; extracts bit 28 from %11
        %481 = OpNot %2 %480

        %482 = OpBitFieldUExtract %2 %11 %34 %6 ; extracts bit 29 from %11
        %483 = OpNot %2 %482

        %484 = OpBitFieldUExtract %2 %11 %35 %6 ; extracts bit 30 from %11
        %485 = OpNot %2 %484

        %486 = OpBitFieldUExtract %2 %11 %36 %6 ; extracts bit 31 from %11
        %487 = OpNot %2 %486

        %488 = OpBitFieldInsert %2 %425 %427 %6 %6 ; inserts bit 1
        %489 = OpBitFieldInsert %2 %488 %429 %7 %6 ; inserts bit 2
        %490 = OpBitFieldInsert %2 %489 %431 %8 %6 ; inserts bit 3
        %491 = OpBitFieldInsert %2 %490 %433 %9 %6 ; inserts bit 4
        %492 = OpBitFieldInsert %2 %491 %435 %10 %6 ; inserts bit 5
        %493 = OpBitFieldInsert %2 %492 %437 %11 %6 ; inserts bit 6
        %494 = OpBitFieldInsert %2 %493 %439 %12 %6 ; inserts bit 7
        %495 = OpBitFieldInsert %2 %494 %441 %13 %6 ; inserts bit 8
        %496 = OpBitFieldInsert %2 %495 %443 %14 %6 ; inserts bit 9
        %497 = OpBitFieldInsert %2 %496 %445 %15 %6 ; inserts bit 10
        %498 = OpBitFieldInsert %2 %497 %447 %16 %6 ; inserts bit 11
        %499 = OpBitFieldInsert %2 %498 %449 %17 %6 ; inserts bit 12
        %500 = OpBitFieldInsert %2 %499 %451 %18 %6 ; inserts bit 13
        %501 = OpBitFieldInsert %2 %500 %453 %19 %6 ; inserts bit 14
        %502 = OpBitFieldInsert %2 %501 %455 %20 %6 ; inserts bit 15
        %503 = OpBitFieldInsert %2 %502 %457 %21 %6 ; inserts bit 16
        %504 = OpBitFieldInsert %2 %503 %459 %22 %6 ; inserts bit 17
        %505 = OpBitFieldInsert %2 %504 %461 %23 %6 ; inserts bit 18
        %506 = OpBitFieldInsert %2 %505 %463 %24 %6 ; inserts bit 19
        %507 = OpBitFieldInsert %2 %506 %465 %25 %6 ; inserts bit 20
        %508 = OpBitFieldInsert %2 %507 %467 %26 %6 ; inserts bit 21
        %509 = OpBitFieldInsert %2 %508 %469 %27 %6 ; inserts bit 22
        %510 = OpBitFieldInsert %2 %509 %471 %28 %6 ; inserts bit 23
        %511 = OpBitFieldInsert %2 %510 %473 %29 %6 ; inserts bit 24
        %512 = OpBitFieldInsert %2 %511 %475 %30 %6 ; inserts bit 25
        %513 = OpBitFieldInsert %2 %512 %477 %31 %6 ; inserts bit 26
        %514 = OpBitFieldInsert %2 %513 %479 %32 %6 ; inserts bit 27
        %515 = OpBitFieldInsert %2 %514 %481 %33 %6 ; inserts bit 28
        %516 = OpBitFieldInsert %2 %515 %483 %34 %6 ; inserts bit 29
        %517 = OpBitFieldInsert %2 %516 %485 %35 %6 ; inserts bit 30
        %518 = OpBitFieldInsert %2 %517 %487 %36 %6 ; inserts bit 31
         %42 = OpNot %2 %11
               OpReturn
               OpFunctionEnd
  )";

  ASSERT_TRUE(IsValid(env, context.get()));
  ASSERT_TRUE(IsEqual(env, variant_shader, context.get()));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
