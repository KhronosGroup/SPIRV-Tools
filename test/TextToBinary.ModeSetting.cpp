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
using ::testing::Eq;

// Test OpCapability

struct CapabilityCase {
  spv::Capability value;
  std::string name;
};

using OpCapabilityTest = test_fixture::TextToBinaryTestBase<
    ::testing::TestWithParam<CapabilityCase>>;

TEST_P(OpCapabilityTest, AnyCapability) {
  std::string input = "OpCapability " + GetParam().name;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(spv::OpCapability, {GetParam().value})));
}

// clang-format off
#define CASE(NAME) { spv::Capability##NAME, #NAME }
INSTANTIATE_TEST_CASE_P(TextToBinaryCapability, OpCapabilityTest,
                        ::testing::ValuesIn(std::vector<CapabilityCase>{
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
                        }));
#undef CASE
// clang-format on

// TODO(dneto): OpMemoryModel
// TODO(dneto): OpMemoryEntryPoint
// TODO(dneto): OpExecutionMode

}  // anonymous namespace
