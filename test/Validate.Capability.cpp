// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Validation tests for Logical Layout

#include <gmock/gmock.h>
#include "UnitSPIRV.h"
#include "ValidateFixtures.h"

#include <functional>
#include <sstream>
#include <string>
#include <utility>

using std::function;
using std::ostream;
using std::ostream_iterator;
using std::pair;
using std::make_pair;
using std::stringstream;
using std::string;
using std::tie;
using std::tuple;
using std::vector;

using ::testing::HasSubstr;

using ValidateCapability =
    spvtest::ValidateBase<tuple<string, pair<string, vector<string>>>,
                          SPV_VALIDATE_INSTRUCTION_BIT>;

TEST_F(ValidateCapability, Default) {
  const char str[] = R"(
            OpCapability Kernel
            OpCapability Matrix
            OpMemoryModel Logical OpenCL
%intt     = OpTypeInt 32 1
%vec3     = OpTypeVector %intt 3
%mat33    = OpTypeMatrix %vec3 3
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

// clang-format off
const vector<string> kAllCapabilities =
  {
    "",
    "Matrix",
    "Shader",
    "Geometry",
    "Tessellation",
    "Addresses",
    "Linkage",
    "Kernel",
    "Vector16",
    "Float16Buffer",
    "Float16",
    "Float64",
    "Int64",
    "Int64Atomics",
    "ImageBasic",
    "ImageReadWrite",
    "ImageMipmap",
    "Pipes",
    "Groups",
    "DeviceEnqueue",
    "LiteralSampler",
    "AtomicStorage",
    "Int16",
    "TessellationPointSize",
    "GeometryPointSize",
    "ImageGatherExtended",
    "StorageImageMultisample",
    "UniformBufferArrayDynamicIndexing",
    "SampledImageArrayDynamicIndexing",
    "StorageBufferArrayDynamicIndexing",
    "StorageImageArrayDynamicIndexing",
    "ClipDistance",
    "CullDistance",
    "ImageCubeArray",
    "SampleRateShading",
    "ImageRect",
    "SampledRect",
    "GenericPointer",
    "Int8",
    "InputAttachment",
    "SparseResidency",
    "MinLod",
    "Sampled1D",
    "Image1D",
    "SampledCubeArray",
    "SampledBuffer",
    "ImageBuffer",
    "ImageMSArray",
    "StorageImageExtendedFormats",
    "ImageQuery",
    "DerivativeControl",
    "InterpolationFunction",
    "TransformFeedback",
    "GeometryStreams",
    "StorageImageReadWithoutFormat",
    "StorageImageWriteWithoutFormat"};

const vector<string> kMatrixDependencies = {
  "Matrix",
  "Shader",
  "Geometry",
  "Tessellation",
  "AtomicStorage",
  "TessellationPointSize",
  "GeometryPointSize",
  "ImageGatherExtended",
  "StorageImageMultisample",
  "UniformBufferArrayDynamicIndexing",
  "SampledImageArrayDynamicIndexing",
  "StorageBufferArrayDynamicIndexing",
  "StorageImageArrayDynamicIndexing",
  "ClipDistance",
  "CullDistance",
  "ImageCubeArray",
  "SampleRateShading",
  "ImageRect",
  "SampledRect",
  "InputAttachment",
  "SparseResidency",
  "MinLod",
  "Sampled1D",
  "Image1D",
  "SampledCubeArray",
  "SampledBuffer",
  "ImageMSArray",
  "ImageBuffer",
  "StorageImageExtendedFormats",
  "ImageQuery",
  "DerivativeControl",
  "InterpolationFunction",
  "TransformFeedback",
  "GeometryStreams",
  "StorageImageReadWithoutFormat",
  "StorageImageWriteWithoutFormat",
};

const vector<string> kShaderDependencies = {
  "Shader",
  "Geometry",
  "Tessellation",
  "AtomicStorage",
  "TessellationPointSize",
  "GeometryPointSize",
  "ImageGatherExtended",
  "StorageImageMultisample",
  "UniformBufferArrayDynamicIndexing",
  "SampledImageArrayDynamicIndexing",
  "StorageBufferArrayDynamicIndexing",
  "StorageImageArrayDynamicIndexing",
  "ClipDistance",
  "CullDistance",
  "ImageCubeArray",
  "SampleRateShading",
  "ImageRect",
  "SampledRect",
  "InputAttachment",
  "SparseResidency",
  "MinLod",
  "Sampled1D",
  "Image1D",
  "SampledCubeArray",
  "SampledBuffer",
  "ImageMSArray",
  "ImageBuffer",
  "StorageImageExtendedFormats",
  "ImageQuery",
  "DerivativeControl",
  "InterpolationFunction",
  "TransformFeedback",
  "GeometryStreams",
  "StorageImageReadWithoutFormat",
  "StorageImageWriteWithoutFormat",
};

const vector<string> kTessellationDependencies = {
  "Tessellation",
  "TessellationPointSize",
};

const vector<string> kGeometryDependencies = {
  "Geometry",
  "GeometryPointSize",
  "GeometryStreams"
};

const vector<string> kGeometryTessellationDependencies = {
  "Tessellation",
  "TessellationPointSize",
  "Geometry",
  "GeometryPointSize",
  "GeometryStreams"
};

const vector<string> kKernelDependencies = {
  "Kernel",
  "Vector16",
  "Float16",
  "Float16Buffer",
  "ImageBasic",
  "ImageReadWrite",
  "ImageMipmap",
  "Pipes",
  "DeviceEnqueue",
  "LiteralSampler",
  "Int8"
};

const vector<string> kAddressesDependencies = {
  "Addresses",
  "GenericPointer"
};

const vector<string> kSampled1DDependencies = {
  "Sampled1D",
  "Image1D"
};

const vector<string> kSampledRectDependencies = {
  "SampledRect",
  "ImageRect",
};

const vector<string> kSampledBufferDependencies = {
  "SampledBuffer",
  "ImageBuffer",
};

INSTANTIATE_TEST_CASE_P(ExecutionModel,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair("OpEntryPoint Vertex %func \"shader\" %var1 %var2\n",                 kShaderDependencies),
make_pair("OpEntryPoint TessellationControl %func \"shader\" %var1 %var2\n",    kTessellationDependencies),
make_pair("OpEntryPoint TessellationEvaluation %func \"shader\" %var1 %var2\n", kTessellationDependencies),
make_pair("OpEntryPoint Geometry %func \"shader\" %var1 %var2\n",               kGeometryDependencies),
make_pair("OpEntryPoint Fragment %func \"shader\" %var1 %var2\n",               kShaderDependencies),
make_pair("OpEntryPoint GLCompute %func \"shader\" %var1 %var2\n",              kShaderDependencies),
make_pair("OpEntryPoint Kernel %func \"shader\" %var1 %var2\n",                 kKernelDependencies)
                                                           )));

INSTANTIATE_TEST_CASE_P(AddressingAndMemoryModel,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair(" OpCapability Shader"
          " OpMemoryModel Logical Simple",     kAllCapabilities),
make_pair(" OpCapability Shader"
          " OpMemoryModel Logical GLSL450",    kAllCapabilities),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Logical OpenCL",     kAllCapabilities),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical32 Simple",  kAddressesDependencies),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical32 GLSL450", kAddressesDependencies),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Physical32 OpenCL",  kAddressesDependencies),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical64 Simple",  kAddressesDependencies),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical64 GLSL450", kAddressesDependencies),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Physical64 OpenCL",  kAddressesDependencies)
                                                           )));

INSTANTIATE_TEST_CASE_P(ExecutionMode,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair("OpExecutionMode %func Invocations 42",          kGeometryDependencies),
make_pair("OpExecutionMode %func SpacingEqual",            kTessellationDependencies),
make_pair("OpExecutionMode %func SpacingFractionalEven",   kTessellationDependencies),
make_pair("OpExecutionMode %func SpacingFractionalOdd",    kTessellationDependencies),
make_pair("OpExecutionMode %func VertexOrderCw",           kTessellationDependencies),
make_pair("OpExecutionMode %func VertexOrderCcw",          kTessellationDependencies),
make_pair("OpExecutionMode %func PixelCenterInteger",      kShaderDependencies),
make_pair("OpExecutionMode %func OriginUpperLeft",         kShaderDependencies),
make_pair("OpExecutionMode %func OriginLowerLeft",         kShaderDependencies),
make_pair("OpExecutionMode %func EarlyFragmentTests",      kShaderDependencies),
make_pair("OpExecutionMode %func PointMode",               kTessellationDependencies),
make_pair("OpExecutionMode %func Xfb",                     vector<string>{"TransformFeedback"}),
make_pair("OpExecutionMode %func DepthReplacing",          kShaderDependencies),
make_pair("OpExecutionMode %func DepthGreater",            kShaderDependencies),
make_pair("OpExecutionMode %func DepthLess",               kShaderDependencies),
make_pair("OpExecutionMode %func DepthUnchanged",          kShaderDependencies),
make_pair("OpExecutionMode %func LocalSize 42 42 42",      kAllCapabilities),
make_pair("OpExecutionMode %func LocalSizeHint 42 42 42",  kKernelDependencies),
make_pair("OpExecutionMode %func InputPoints",             kGeometryDependencies),
make_pair("OpExecutionMode %func InputLines",              kGeometryDependencies),
make_pair("OpExecutionMode %func InputLinesAdjacency",     kGeometryDependencies),
make_pair("OpExecutionMode %func Triangles",               kGeometryTessellationDependencies),
make_pair("OpExecutionMode %func InputTrianglesAdjacency", kGeometryDependencies),
make_pair("OpExecutionMode %func Quads",                   kTessellationDependencies),
make_pair("OpExecutionMode %func Isolines",                kTessellationDependencies),
make_pair("OpExecutionMode %func OutputVertices 42",       kGeometryTessellationDependencies),
make_pair("OpExecutionMode %func OutputPoints",            kGeometryDependencies),
make_pair("OpExecutionMode %func OutputLineStrip",         kGeometryDependencies),
make_pair("OpExecutionMode %func OutputTriangleStrip",     kGeometryDependencies),
make_pair("OpExecutionMode %func VecTypeHint 2",           kKernelDependencies),
make_pair("OpExecutionMode %func ContractionOff",          kKernelDependencies)
)));

INSTANTIATE_TEST_CASE_P(StorageClass,
                        ValidateCapability,
                        ::testing::Combine(
testing::ValuesIn(kAllCapabilities),
testing::Values(
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer UniformConstant %intt\n"
          " %var = OpVariable %ptrt UniformConstant\n",             kAllCapabilities),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Input %intt"
          " %var = OpVariable %ptrt Input\n",                       kShaderDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Uniform %intt\n"
          " %var = OpVariable %ptrt Uniform\n",                     kShaderDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Output %intt\n"
          " %var = OpVariable %ptrt Output\n",                      kShaderDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Workgroup %intt\n"
          " %var = OpVariable %ptrt Workgroup\n",                   kAllCapabilities),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer CrossWorkgroup %intt\n"
          " %var = OpVariable %ptrt CrossWorkgroup\n",              kAllCapabilities),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Private %intt\n"
          " %var = OpVariable %ptrt Private\n",                     kShaderDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Generic %intt\n"
          " %var = OpVariable %ptrt Generic\n",                     kKernelDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer PushConstant %intt\n"
          " %var = OpVariable %ptrt PushConstant\n",                kShaderDependencies),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer AtomicCounter %intt\n"
          " %var = OpVariable %ptrt AtomicCounter\n",               vector<string>{"AtomicStorage"}),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Image %intt\n"
          " %var = OpVariable %ptrt Image\n",                       kAllCapabilities)
  )));

INSTANTIATE_TEST_CASE_P(Dim,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 1D 0 0 0 0 Unknown",       kSampled1DDependencies),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 2D 0 0 0 0 Unknown",       kAllCapabilities),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 3D 0 0 0 0 Unknown",       kAllCapabilities),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Cube 0 0 0 0 Unknown",     kShaderDependencies),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Rect 0 0 0 0 Unknown",     kSampledRectDependencies),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Buffer 0 0 0 0 Unknown",   kSampledBufferDependencies),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt SubpassData 0 0 0 2 Unknown", vector<string>{"InputAttachment"})
                                          )));

// NOTE: All Sampler Address Modes require kernel capabilities but the
// OpConstantSampler requires LiteralSampler which depends on Kernel
INSTANTIATE_TEST_CASE_P(SamplerAddressingMode,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair(" %samplert = OpTypeSampler"
          " %sampler = OpConstantSampler %samplert None 1 Nearest",           vector<string>{"LiteralSampler"}),
make_pair(" %samplert = OpTypeSampler"
          " %sampler = OpConstantSampler %samplert ClampToEdge 1 Nearest",    vector<string>{"LiteralSampler"}),
make_pair(" %samplert = OpTypeSampler"
          " %sampler = OpConstantSampler %samplert Clamp 1 Nearest",          vector<string>{"LiteralSampler"}),
make_pair(" %samplert = OpTypeSampler"
          " %sampler = OpConstantSampler %samplert Repeat 1 Nearest",         vector<string>{"LiteralSampler"}),
make_pair(" %samplert = OpTypeSampler"
          " %sampler = OpConstantSampler %samplert RepeatMirrored 1 Nearest", vector<string>{"LiteralSampler"})
                                        )));

//TODO(umar): Sampler Filter Mode
//TODO(umar): Image Format
//TODO(umar): Image Channel Order
//TODO(umar): Image Channel Data Type
//TODO(umar): Image Operands
//TODO(umar): FP Fast Math Mode
//TODO(umar): FP Rounding Mode
//TODO(umar): Linkage Type
//TODO(umar): Access Qualifier
//TODO(umar): Function Parameter Attribute

INSTANTIATE_TEST_CASE_P(Decoration,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair("OpDecorate %intt RelaxedPrecision\n",                    kShaderDependencies),
make_pair("OpDecorate %intt SpecId 1\n",                            kShaderDependencies),
make_pair("OpDecorate %intt Block\n",                               kShaderDependencies),
make_pair("OpDecorate %intt BufferBlock\n",                         kShaderDependencies),
make_pair("OpDecorate %intt RowMajor\n",                            kMatrixDependencies),
make_pair("OpDecorate %intt ColMajor\n",                            kMatrixDependencies),
make_pair("OpDecorate %intt ArrayStride 1\n",                       kShaderDependencies),
make_pair("OpDecorate %intt MatrixStride 1\n",                      kShaderDependencies),
make_pair("OpDecorate %intt GLSLShared\n",                          kShaderDependencies),
make_pair("OpDecorate %intt GLSLPacked\n",                          kShaderDependencies),
make_pair("OpDecorate %intt CPacked\n",                             kKernelDependencies),
make_pair("OpDecorate %intt NoPerspective\n",                       kShaderDependencies),
make_pair("OpDecorate %intt Flat\n",                                kShaderDependencies),
make_pair("OpDecorate %intt Patch\n",                               kTessellationDependencies),
make_pair("OpDecorate %intt Centroid\n",                            kShaderDependencies),
make_pair("OpDecorate %intt Sample\n",                              kShaderDependencies),
make_pair("OpDecorate %intt Invariant\n",                           kShaderDependencies),
make_pair("OpDecorate %intt Restrict\n",                            kAllCapabilities),
make_pair("OpDecorate %intt Aliased\n",                             kAllCapabilities),
make_pair("OpDecorate %intt Volatile\n",                            kAllCapabilities),
make_pair("OpDecorate %intt Constant\n",                            kKernelDependencies),
make_pair("OpDecorate %intt Coherent\n",                            kAllCapabilities),
make_pair("OpDecorate %intt NonWritable\n",                         kAllCapabilities),
make_pair("OpDecorate %intt NonReadable\n",                         kAllCapabilities),
make_pair("OpDecorate %intt Uniform\n",                             kShaderDependencies),
make_pair("OpDecorate %intt SaturatedConversion\n",                 kKernelDependencies),
make_pair("OpDecorate %intt Stream 0\n",                            vector<string>{"GeometryStreams"}),
make_pair("OpDecorate %intt Location 0\n",                          kShaderDependencies),
make_pair("OpDecorate %intt Component 0\n",                         kShaderDependencies),
make_pair("OpDecorate %intt Index 0\n",                             kShaderDependencies),
make_pair("OpDecorate %intt Binding 0\n",                           kShaderDependencies),
make_pair("OpDecorate %intt DescriptorSet 0\n",                     kShaderDependencies),
make_pair("OpDecorate %intt Offset 0\n",                            kAllCapabilities),
make_pair("OpDecorate %intt XfbBuffer 0\n",                         vector<string>{"TransformFeedback"}),
make_pair("OpDecorate %intt XfbStride 0\n",                         vector<string>{"TransformFeedback"}),
make_pair("OpDecorate %intt FuncParamAttr Zext\n",                  kKernelDependencies),
make_pair("OpDecorate %intt FPRoundingMode RTE\n",                  kKernelDependencies),
make_pair("OpDecorate %intt FPFastMathMode Fast\n",                 kKernelDependencies),
make_pair("OpDecorate %intt LinkageAttributes \"other\" Import\n",  vector<string>{"Linkage"}),
make_pair("OpDecorate %intt NoContraction\n",                       kShaderDependencies),
make_pair("OpDecorate %intt InputAttachmentIndex 0\n",              vector<string>{"InputAttachment"}),
make_pair("OpDecorate %intt Alignment 4\n",                         kKernelDependencies)
  )));


INSTANTIATE_TEST_CASE_P(BuiltIn,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair("OpDecorate %intt BuiltIn Position\n",                  kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn PointSize\n",                 kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn ClipDistance\n",              vector<string>{"ClipDistance"}),
make_pair("OpDecorate %intt BuiltIn CullDistance\n",              vector<string>{"CullDistance"}),
make_pair("OpDecorate %intt BuiltIn VertexId\n",                  kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn InstanceId\n",                kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn PrimitiveId\n",               kGeometryTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn InvocationId\n",              kGeometryTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn Layer\n",                     kGeometryDependencies),
//make_pair("OpDecorate %intt BuiltIn ViewPortIndex\n",           vector<string>{"MultiViewport"}),
make_pair("OpDecorate %intt BuiltIn TessLevelOuter\n",            kTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn TessLevelInner\n",            kTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn TessCoord\n",                 kTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn PatchVertices\n",             kTessellationDependencies),
make_pair("OpDecorate %intt BuiltIn FragCoord\n",                 kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn PointCoord\n",                kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn FrontFacing\n",               kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn SampleId\n",                  vector<string>{"SampleRateShading"}),
make_pair("OpDecorate %intt BuiltIn SamplePosition\n",            vector<string>{"SampleRateShading"}),
make_pair("OpDecorate %intt BuiltIn SampleMask\n",                vector<string>{"SampleRateShading"}),
make_pair("OpDecorate %intt BuiltIn FragDepth\n",                 kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn HelperInvocation\n",          kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn NumWorkgroups\n",             kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn WorkgroupSize\n",             kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn WorkgroupId\n",               kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn LocalInvocationId\n",         kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn GlobalInvocationId\n",        kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn LocalInvocationIndex",        kAllCapabilities),
make_pair("OpDecorate %intt BuiltIn WorkDim\n",                   kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn GlobalSize\n",                kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn EnqueuedWorkgroupSize\n",     kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn GlobalOffset\n",              kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn GlobalLinearId\n",            kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn SubgroupSize\n",              kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn SubgroupMaxSize\n",           kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn NumSubgroups\n",              kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn NumEnqueuedSubgroups\n",      kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn SubgroupId\n",                kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn SubgroupLocalInvocationId\n", kKernelDependencies),
make_pair("OpDecorate %intt BuiltIn VertexIndex\n",               kShaderDependencies),
make_pair("OpDecorate %intt BuiltIn InstanceIndex\n",             kShaderDependencies)
                                                           )));

// TODO(umar): Selection Control
// TODO(umar): Loop Control
// TODO(umar): Function Control
// TODO(umar): Memory Semantics
// TODO(umar): Memory Access
// TODO(umar): Scope
// TODO(umar): Group Operation
// TODO(umar): Kernel Enqueue Flags
// TODO(umar): Kernel Profiling Flags

INSTANTIATE_TEST_CASE_P(MatrixOp,
                        ValidateCapability,
                        ::testing::Combine(
                        testing::ValuesIn(kAllCapabilities),
                        testing::Values(
make_pair(
          "%intt     = OpTypeInt 32 1\n"
          "%vec3     = OpTypeVector %intt 3\n"
          "%mat33    = OpTypeMatrix %vec3 3\n", kMatrixDependencies))));
// clang-format on

// TODO(umar): Instruction capability checks

TEST_P(ValidateCapability, Capability) {
  string capability;
  pair<string, vector<string>> operation;
  std::tie(capability, operation) = GetParam();
  stringstream ss;
  if (capability.empty() == false) {
    ss << "OpCapability " + capability + " ";
  }

  ss << operation.first;

  spv_result_t res = SPV_ERROR_INTERNAL;
  auto& valid_capabilities = operation.second;

  auto it =
      find(begin(valid_capabilities), end(valid_capabilities), capability);
  if (it != end(valid_capabilities)) {
    res = SPV_SUCCESS;
  } else {
    res = SPV_ERROR_INVALID_CAPABILITY;
  }

  CompileSuccessfully(ss.str());
  ASSERT_EQ(res, ValidateInstructions());
}
