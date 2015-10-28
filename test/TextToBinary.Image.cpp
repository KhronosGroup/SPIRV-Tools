// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Assembler tests for instructions in the "Image Instructions" section of
// the SPIR-V spec.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;

// An example case for a mask value with operands.
struct ImageOperandsCase {
  std::string image_operands;
  // The expected mask, followed by its operands.
  std::vector<uint32_t> expected_mask_and_operands;
};

// Test all kinds of image operands.

using ImageOperandsTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<ImageOperandsCase>>;

TEST_P(ImageOperandsTest, Sample) {
  const std::string input =
      "%result = OpImageFetch %type %image %coord " + GetParam().image_operands;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpImageFetch, {1, 2, 3, 4},
                                 GetParam().expected_mask_and_operands)));
}

#define MASK(NAME) SpvImageOperands##NAME##Mask
INSTANTIATE_TEST_CASE_P(
    TextToBinaryImageOperandsAny, ImageOperandsTest,
    ::testing::ValuesIn(std::vector<ImageOperandsCase>{
        // TODO(dneto): Rev32 adds many more values, and rearranges their
        // values.
        // Image operands are optional.
        {"", {}},
        // Test each kind, alone.
        {"Bias %5", {MASK(Bias), 5}},
        {"Lod %10", {MASK(Lod), 5}},
        {"Grad %11 %12", {MASK(Grad), 5, 6}},
        {"ConstOffset %13", {MASK(ConstOffset), 5}},
        {"Offset %14", {MASK(Offset), 5}},
        {"ConstOffsets %15", {MASK(ConstOffsets), 5}},
        {"Sample %16", {MASK(Sample), 5}},
        {"MinLod %17", {MASK(MinLod), 5}},
    }));
#undef MASK
#define MASK(NAME) static_cast<uint32_t>(SpvImageOperands##NAME##Mask)
INSTANTIATE_TEST_CASE_P(
    TextToBinaryImageOperandsCombination, ImageOperandsTest,
    ::testing::ValuesIn(std::vector<ImageOperandsCase>{
        // TODO(dneto): Rev32 adds many more values, and rearranges their
        // values.
        // Test adjacent pairs, so we can easily debug the values when it fails.
        {"Bias|Lod %10 %11", {MASK(Bias) | MASK(Lod), 5, 6}},
        {"Lod|Grad %12 %13 %14", {MASK(Lod) | MASK(Grad), 5, 6, 7}},
        {"Grad|ConstOffset %15 %16 %17",
         {MASK(Grad) | MASK(ConstOffset), 5, 6, 7}},
        {"ConstOffset|Offset %18 %19",
         {MASK(ConstOffset) | MASK(Offset), 5, 6}},
        {"Offset|ConstOffsets %20 %21",
         {MASK(Offset) | MASK(ConstOffsets), 5, 6}},
        {"ConstOffsets|Sample %22 %23",
         {MASK(ConstOffsets) | MASK(Sample), 5, 6}},
        // Test all masks together.
        {"Bias|Lod|Grad|ConstOffset|Offset|ConstOffsets|Sample"
         " %5 %10 %11 %12 %13 %14 %15 %16",
         {MASK(Bias) | MASK(Lod) | MASK(Grad) | MASK(ConstOffset) |
              MASK(Offset) | MASK(ConstOffsets) | MASK(Sample),
          5, 6, 7, 8, 9, 10, 11, 12}},
        // The same, but with mask value names reversed.
        {"Sample|ConstOffsets|Offset|ConstOffset|Grad|Lod|Bias"
         " %5 %10 %11 %12 %13 %14 %15 %16",
         {MASK(Bias) | MASK(Lod) | MASK(Grad) | MASK(ConstOffset) |
              MASK(Offset) | MASK(ConstOffsets) | MASK(Sample),
          5, 6, 7, 8, 9, 10, 11, 12}}}));
#undef MASK

TEST_F(ImageOperandsTest, WrongOperand) {
  EXPECT_THAT(CompileFailure("%r = OpImageFetch %t %i %c xxyyzz"),
              Eq("Invalid image operand 'xxyyzz'."));
}

// TODO(dneto): OpSampledImage
// TODO(dneto): OpImageSampleImplicitLod
// TODO(dneto): OpImageSampleExplicitLod
// TODO(dneto): OpImageSampleDrefImplicitLod
// TODO(dneto): OpImageSampleDrefExplicitLod
// TODO(dneto): OpImageSampleProjImplicitLod
// TODO(dneto): OpImageSampleProjExplicitLod
// TODO(dneto): OpImageSampleProjDrefImplicitLod
// TODO(dneto): OpImageSampleProjDrefExplicitLod
// TODO(dneto): OpImageGather
// TODO(dneto): OpImageDrefGather
// TODO(dneto): OpImageRead
// TODO(dneto): OpImageWrite
// TODO(dneto): OpImageQueryFormat
// TODO(dneto): OpImageQueryOrder
// TODO(dneto): OpImageQuerySizeLod
// TODO(dneto): OpImageQuerySize
// TODO(dneto): OpImageQueryLod
// TODO(dneto): OpImageQueryLevels
// TODO(dneto): OpImageQuerySamples
// TODO(dneto): OpImageSparseSampleImplicitLod
// TODO(dneto): OpImageSparseSampleExplicitLod
// TODO(dneto): OpImageSparseSampleDrefImplicitLod
// TODO(dneto): OpImageSparseSampleDrefExplicitLod
// TODO(dneto): OpImageSparseSampleProjImplicitLod
// TODO(dneto): OpImageSparseSampleProjExplicitLod
// TODO(dneto): OpImageSparseSampleProjDrefImplicitLod
// TODO(dneto): OpImageSparseSampleProjDrefExplicitLod
// TODO(dneto): OpImageSparseFetch
// TODO(dneto): OpImageSparseDrefGather
// TODO(dneto): OpImageSparseTexelsResident

}  // anonymous namespace
