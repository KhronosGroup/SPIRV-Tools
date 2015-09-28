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

// Assembler tests for instructions in the "Mode-Setting" section of the
// SPIR-V spec.

#include "UnitSPIRV.h"

#include "gmock/gmock.h"
#include "TestFixture.h"

namespace {

using spvtest::MakeInstruction;
using spvtest::MakeVector;
using ::testing::Eq;

// Test OpMemoryModel

// An example case for OpMemoryModel
struct MemoryModelCase {
  uint32_t get_addressing_value() const {
    return static_cast<uint32_t>(addressing_value);
  }
  uint32_t get_memory_value() const {
    return static_cast<uint32_t>(memory_value);
  }
  spv::AddressingModel addressing_value;
  std::string addressing_name;
  spv::MemoryModel memory_value;
  std::string memory_name;
};

using OpMemoryModelTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<MemoryModelCase>>;

TEST_P(OpMemoryModelTest, AnyMemoryModelCase) {
  std::string input = "OpMemoryModel " + GetParam().addressing_name + " " +
                      GetParam().memory_name;
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(MakeInstruction(spv::OpMemoryModel, {GetParam().get_addressing_value(),
                                              GetParam().get_memory_value()})));
}

#define CASE(ADDRESSING, MEMORY)                                             \
  {                                                                          \
    spv::AddressingModel##ADDRESSING, #ADDRESSING, spv::MemoryModel##MEMORY, \
        #MEMORY                                                              \
  }
// clang-format off
INSTANTIATE_TEST_CASE_P(TextToBinaryMemoryModel, OpMemoryModelTest,
                        ::testing::ValuesIn(std::vector<MemoryModelCase>{
                          // These cases exercise each addressing model, and
                          // each memory model, but not necessarily in
                          // combination.
                            CASE(Logical,Simple),
                            CASE(Logical,GLSL450),
                            CASE(Physical32,OpenCL),
                            CASE(Physical64,OpenCL),
                        }));
#undef CASE
// clang-format on

// Test OpEntryPoint

// An example case for OpEntryPoint
struct EntryPointCase {
  uint32_t get_execution_value() const {
    return static_cast<uint32_t>(execution_value);
  }
  spv::ExecutionModel execution_value;
  std::string execution_name;
  std::string entry_point_name;
};

using OpEntryPointTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EntryPointCase>>;

TEST_P(OpEntryPointTest, AnyEntryPointCase) {
  // TODO(dneto): utf-8, escaping, quoting cases for entry point name.
  std::string input = "OpEntryPoint " + GetParam().execution_name + " %1 \"" +
                      GetParam().entry_point_name + "\"";
  std::vector<uint32_t> expected_operands{GetParam().get_execution_value(), 1};
  std::vector<uint32_t> encoded_entry_point_name =
      MakeVector(GetParam().entry_point_name);
  expected_operands.insert(expected_operands.end(),
                           encoded_entry_point_name.begin(),
                           encoded_entry_point_name.end());
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpEntryPoint, expected_operands)));
}

// clang-format off
#define CASE(NAME) spv::ExecutionModel##NAME, #NAME
INSTANTIATE_TEST_CASE_P(TextToBinaryEntryPoint, OpEntryPointTest,
                        ::testing::ValuesIn(std::vector<EntryPointCase>{
                          { CASE(Vertex), "" },
                          { CASE(TessellationControl), "my tess" },
                          { CASE(TessellationEvaluation), "really fancy" },
                          { CASE(Geometry), "Euclid" },
                          { CASE(Fragment), "FAT32" },
                          { CASE(GLCompute), "cubic" },
                          { CASE(Kernel), "Sanders" },
                        }));
#undef CASE
// clang-format on

// Test OpExecutionMode

using OpExecutionModeTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::ExecutionMode>>>;

TEST_P(OpExecutionModeTest, AnyExecutionMode) {
  // This string should assemble, but should not validate.
  std::stringstream input;
  input << "OpExecutionMode %1 " << GetParam().name();
  for (auto operand : GetParam().operands()) input << " " << operand;
  std::vector<uint32_t> expected_operands{1, GetParam().value()};
  expected_operands.insert(expected_operands.end(), GetParam().operands().begin(),
                           GetParam().operands().end());
  EXPECT_THAT(CompiledInstructions(input.str()),
              Eq(MakeInstruction(spv::OpExecutionMode, expected_operands)));
}

#define CASE(NAME) spv::ExecutionMode##NAME, #NAME
INSTANTIATE_TEST_CASE_P(
    TextToBinaryExecutionMode, OpExecutionModeTest,
    ::testing::ValuesIn(std::vector<EnumCase<spv::ExecutionMode>>{
        // The operand literal values are arbitrarily chosen,
        // but there are the right number of them.
        {CASE(Invocations), {101}},
        {CASE(SpacingEqual), {}},
        {CASE(SpacingFractionalEven), {}},
        {CASE(SpacingFractionalOdd), {}},
        {CASE(VertexOrderCw), {}},
        {CASE(VertexOrderCcw), {}},
        {CASE(PixelCenterInteger), {}},
        {CASE(OriginUpperLeft), {}},
        {CASE(OriginLowerLeft), {}},
        {CASE(EarlyFragmentTests), {}},
        {CASE(PointMode), {}},
        {CASE(Xfb), {}},
        {CASE(DepthReplacing), {}},
        {CASE(DepthGreater), {}},
        {CASE(DepthLess), {}},
        {CASE(DepthUnchanged), {}},
        {CASE(LocalSize), {64, 1, 2}},
        {CASE(LocalSizeHint), {8, 2, 4}},
        {CASE(InputPoints), {}},
        {CASE(InputLines), {}},
        {CASE(InputLinesAdjacency), {}},
        {CASE(InputTriangles), {}},
        {CASE(InputTrianglesAdjacency), {}},
        {CASE(InputQuads), {}},
        {CASE(InputIsolines), {}},
        {CASE(OutputVertices), {21}},
        {CASE(OutputPoints), {}},
        {CASE(OutputLineStrip), {}},
        {CASE(OutputTriangleStrip), {}},
        {CASE(VecTypeHint), {96}},
        {CASE(ContractionOff), {}},
        {CASE(IndependentForwardProgress), {}},
    }));
#undef CASE

// Test OpCapability

using OpCapabilityTest = spvtest::TextToBinaryTestBase<
    ::testing::TestWithParam<EnumCase<spv::Capability>>>;

TEST_P(OpCapabilityTest, AnyCapability) {
  std::string input = "OpCapability " + GetParam().name();
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpCapability, {GetParam().value()})));
}

// clang-format off
#define CASE(NAME) { spv::Capability##NAME, #NAME }
INSTANTIATE_TEST_CASE_P(TextToBinaryCapability, OpCapabilityTest,
                        ::testing::ValuesIn(std::vector<EnumCase<spv::Capability>>{
                            CASE(Matrix),
                            CASE(Shader),
                            CASE(Geometry),
                            CASE(Tessellation),
                            CASE(Addresses),
                            CASE(Linkage),
                            CASE(Kernel),
                            CASE(Vector16),
                            CASE(Float16Buffer),
                            CASE(Float16),
                            CASE(Float64),
                            CASE(Int64),
                            CASE(Int64Atomics),
                            CASE(ImageBasic),
                            CASE(ImageReadWrite),
                            CASE(ImageMipmap),
                            CASE(ImageSRGBWrite),
                            CASE(Pipes),
                            CASE(Groups),
                            CASE(DeviceEnqueue),
                            CASE(LiteralSampler),
                            CASE(AtomicStorage),
                            CASE(Int16),
                            CASE(TessellationPointSize),
                            CASE(GeometryPointSize),
                            CASE(ImageGatherExtended),
                            CASE(StorageImageExtendedFormats),
                            CASE(StorageImageMultisample),
                            CASE(UniformBufferArrayDynamicIndexing),
                            CASE(SampledImageArrayDynamicIndexing),
                            CASE(StorageBufferArrayDynamicIndexing),
                            CASE(StorageImageArrayDynamicIndexing),
                            CASE(ClipDistance),
                            CASE(CullDistance),
                            CASE(ImageCubeArray),
                            CASE(SampleRateShading),
                            CASE(ImageRect),
                            CASE(SampledRect),
                            CASE(GenericPointer),
                            CASE(Int8),
                            CASE(InputTarget),
                            CASE(SparseResidency),
                            CASE(MinLod),
                            CASE(Sampled1D),
                            CASE(Image1D),
                            CASE(SampledCubeArray),
                            CASE(SampledBuffer),
                            CASE(ImageBuffer),
                            CASE(ImageMSArray),
                            CASE(AdvancedFormats),
                            CASE(ImageQuery),
                            CASE(DerivativeControl),
                            CASE(InterpolationFunction),
                            CASE(TransformFeedback),
                        }));
#undef CASE
// clang-format on

using TextToBinaryCapability = spvtest::TextToBinaryTest;

TEST_F(TextToBinaryCapability, BadMissingCapability) {
  EXPECT_THAT(CompileFailure("OpCapability"),
              Eq("Expected operand, found end of stream."));
}

TEST_F(TextToBinaryCapability, BadInvalidCapability) {
  EXPECT_THAT(CompileFailure("OpCapability 123"),
              Eq("Invalid capability '123'."));
}

// TODO(dneto): OpExecutionMode

}  // anonymous namespace
