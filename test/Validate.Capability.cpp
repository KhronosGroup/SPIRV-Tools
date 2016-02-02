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

#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace {

using std::pair;
using std::make_pair;
using std::stringstream;
using std::string;
using std::tuple;
using std::vector;

using testing::Combine;
using testing::Values;
using testing::ValuesIn;

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
const vector<string>& AllCapabilities() {
  static const auto r = new vector<string>{
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
  return *r;
}

const vector<string>& MatrixDependencies() {
  static const auto r = new vector<string>{
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
  "StorageImageWriteWithoutFormat"};
  return *r;
}

const vector<string>& ShaderDependencies() {
  static const auto r = new vector<string>{
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
  "StorageImageWriteWithoutFormat"};
  return *r;
}

const vector<string>& TessellationDependencies() {
  static const auto r = new vector<string>{
  "Tessellation",
  "TessellationPointSize"};
  return *r;
}

const vector<string>& GeometryDependencies() {
  static const auto r = new vector<string>{
  "Geometry",
  "GeometryPointSize",
  "GeometryStreams"};
  return *r;
}

const vector<string>& GeometryTessellationDependencies() {
  static const auto r = new vector<string>{
  "Tessellation",
  "TessellationPointSize",
  "Geometry",
  "GeometryPointSize",
  "GeometryStreams"};
  return *r;
}

const vector<string>& KernelDependencies() {
  static const auto r = new vector<string>{
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
  "Int8"};
  return *r;
}

const vector<string>& AddressesDependencies() {
  static const auto r = new vector<string>{
  "Addresses",
  "GenericPointer"};
  return *r;
}

const vector<string>& Sampled1DDependencies() {
  static const auto r = new vector<string>{
  "Sampled1D",
  "Image1D"};
  return *r;
}

const vector<string>& SampledRectDependencies() {
  static const auto r = new vector<string>{
  "SampledRect",
  "ImageRect"};
  return *r;
}

const vector<string>& SampledBufferDependencies() {
  static const auto r = new vector<string>{
  "SampledBuffer",
  "ImageBuffer"};
  return *r;
}

INSTANTIATE_TEST_CASE_P(ExecutionModel, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair("OpEntryPoint Vertex %func \"shader\" %var1 %var2\n",                 ShaderDependencies()),
make_pair("OpEntryPoint TessellationControl %func \"shader\" %var1 %var2\n",    TessellationDependencies()),
make_pair("OpEntryPoint TessellationEvaluation %func \"shader\" %var1 %var2\n", TessellationDependencies()),
make_pair("OpEntryPoint Geometry %func \"shader\" %var1 %var2\n",               GeometryDependencies()),
make_pair("OpEntryPoint Fragment %func \"shader\" %var1 %var2\n",               ShaderDependencies()),
make_pair("OpEntryPoint GLCompute %func \"shader\" %var1 %var2\n",              ShaderDependencies()),
make_pair("OpEntryPoint Kernel %func \"shader\" %var1 %var2\n",                 KernelDependencies())
                                                           )));

INSTANTIATE_TEST_CASE_P(AddressingAndMemoryModel, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair(" OpCapability Shader"
          " OpMemoryModel Logical Simple",     AllCapabilities()),
make_pair(" OpCapability Shader"
          " OpMemoryModel Logical GLSL450",    AllCapabilities()),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Logical OpenCL",     AllCapabilities()),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical32 Simple",  AddressesDependencies()),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical32 GLSL450", AddressesDependencies()),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Physical32 OpenCL",  AddressesDependencies()),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical64 Simple",  AddressesDependencies()),
make_pair(" OpCapability Shader"
          " OpMemoryModel Physical64 GLSL450", AddressesDependencies()),
make_pair(" OpCapability Kernel"
          " OpMemoryModel Physical64 OpenCL",  AddressesDependencies())
                                                           )));

INSTANTIATE_TEST_CASE_P(ExecutionMode, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair("OpExecutionMode %func Invocations 42",          GeometryDependencies()),
make_pair("OpExecutionMode %func SpacingEqual",            TessellationDependencies()),
make_pair("OpExecutionMode %func SpacingFractionalEven",   TessellationDependencies()),
make_pair("OpExecutionMode %func SpacingFractionalOdd",    TessellationDependencies()),
make_pair("OpExecutionMode %func VertexOrderCw",           TessellationDependencies()),
make_pair("OpExecutionMode %func VertexOrderCcw",          TessellationDependencies()),
make_pair("OpExecutionMode %func PixelCenterInteger",      ShaderDependencies()),
make_pair("OpExecutionMode %func OriginUpperLeft",         ShaderDependencies()),
make_pair("OpExecutionMode %func OriginLowerLeft",         ShaderDependencies()),
make_pair("OpExecutionMode %func EarlyFragmentTests",      ShaderDependencies()),
make_pair("OpExecutionMode %func PointMode",               TessellationDependencies()),
make_pair("OpExecutionMode %func Xfb",                     vector<string>{"TransformFeedback"}),
make_pair("OpExecutionMode %func DepthReplacing",          ShaderDependencies()),
make_pair("OpExecutionMode %func DepthGreater",            ShaderDependencies()),
make_pair("OpExecutionMode %func DepthLess",               ShaderDependencies()),
make_pair("OpExecutionMode %func DepthUnchanged",          ShaderDependencies()),
make_pair("OpExecutionMode %func LocalSize 42 42 42",      AllCapabilities()),
make_pair("OpExecutionMode %func LocalSizeHint 42 42 42",  KernelDependencies()),
make_pair("OpExecutionMode %func InputPoints",             GeometryDependencies()),
make_pair("OpExecutionMode %func InputLines",              GeometryDependencies()),
make_pair("OpExecutionMode %func InputLinesAdjacency",     GeometryDependencies()),
make_pair("OpExecutionMode %func Triangles",               GeometryTessellationDependencies()),
make_pair("OpExecutionMode %func InputTrianglesAdjacency", GeometryDependencies()),
make_pair("OpExecutionMode %func Quads",                   TessellationDependencies()),
make_pair("OpExecutionMode %func Isolines",                TessellationDependencies()),
make_pair("OpExecutionMode %func OutputVertices 42",       GeometryTessellationDependencies()),
make_pair("OpExecutionMode %func OutputPoints",            GeometryDependencies()),
make_pair("OpExecutionMode %func OutputLineStrip",         GeometryDependencies()),
make_pair("OpExecutionMode %func OutputTriangleStrip",     GeometryDependencies()),
make_pair("OpExecutionMode %func VecTypeHint 2",           KernelDependencies()),
make_pair("OpExecutionMode %func ContractionOff",          KernelDependencies())
)));

INSTANTIATE_TEST_CASE_P(StorageClass, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer UniformConstant %intt\n"
          " %var = OpVariable %ptrt UniformConstant\n",             AllCapabilities()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Input %intt"
          " %var = OpVariable %ptrt Input\n",                       ShaderDependencies()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Uniform %intt\n"
          " %var = OpVariable %ptrt Uniform\n",                     ShaderDependencies()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Output %intt\n"
          " %var = OpVariable %ptrt Output\n",                      ShaderDependencies()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Workgroup %intt\n"
          " %var = OpVariable %ptrt Workgroup\n",                   AllCapabilities()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer CrossWorkgroup %intt\n"
          " %var = OpVariable %ptrt CrossWorkgroup\n",              AllCapabilities()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Private %intt\n"
          " %var = OpVariable %ptrt Private\n",                     ShaderDependencies()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer PushConstant %intt\n"
          " %var = OpVariable %ptrt PushConstant\n",                ShaderDependencies()),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer AtomicCounter %intt\n"
          " %var = OpVariable %ptrt AtomicCounter\n",               vector<string>{"AtomicStorage"}),
make_pair(" %intt = OpTypeInt 32 0\n"
          " %ptrt = OpTypePointer Image %intt\n"
          " %var = OpVariable %ptrt Image\n",                       AllCapabilities())
  )));

INSTANTIATE_TEST_CASE_P(Dim, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 1D 0 0 0 0 Unknown",       Sampled1DDependencies()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 2D 0 0 0 0 Unknown",       AllCapabilities()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt 3D 0 0 0 0 Unknown",       AllCapabilities()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Cube 0 0 0 0 Unknown",     ShaderDependencies()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Rect 0 0 0 0 Unknown",     SampledRectDependencies()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt Buffer 0 0 0 0 Unknown",   SampledBufferDependencies()),
make_pair(" OpCapability ImageBasic"
          " %voidt = OpTypeVoid"
          " %imgt = OpTypeImage %voidt SubpassData 0 0 0 2 Unknown", vector<string>{"InputAttachment"})
                                          )));

// NOTE: All Sampler Address Modes require kernel capabilities but the
// OpConstantSampler requires LiteralSampler which depends on Kernel
INSTANTIATE_TEST_CASE_P(SamplerAddressingMode, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
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

INSTANTIATE_TEST_CASE_P(Decoration, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair("OpDecorate %intt RelaxedPrecision\n",                    ShaderDependencies()),
make_pair("OpDecorate %intt SpecId 1\n",                            ShaderDependencies()),
make_pair("OpDecorate %intt Block\n",                               ShaderDependencies()),
make_pair("OpDecorate %intt BufferBlock\n",                         ShaderDependencies()),
make_pair("OpDecorate %intt RowMajor\n",                            MatrixDependencies()),
make_pair("OpDecorate %intt ColMajor\n",                            MatrixDependencies()),
make_pair("OpDecorate %intt ArrayStride 1\n",                       ShaderDependencies()),
make_pair("OpDecorate %intt MatrixStride 1\n",                      ShaderDependencies()),
make_pair("OpDecorate %intt GLSLShared\n",                          ShaderDependencies()),
make_pair("OpDecorate %intt GLSLPacked\n",                          ShaderDependencies()),
make_pair("OpDecorate %intt CPacked\n",                             KernelDependencies()),
make_pair("OpDecorate %intt NoPerspective\n",                       ShaderDependencies()),
make_pair("OpDecorate %intt Flat\n",                                ShaderDependencies()),
make_pair("OpDecorate %intt Patch\n",                               TessellationDependencies()),
make_pair("OpDecorate %intt Centroid\n",                            ShaderDependencies()),
make_pair("OpDecorate %intt Sample\n",                              ShaderDependencies()),
make_pair("OpDecorate %intt Invariant\n",                           ShaderDependencies()),
make_pair("OpDecorate %intt Restrict\n",                            AllCapabilities()),
make_pair("OpDecorate %intt Aliased\n",                             AllCapabilities()),
make_pair("OpDecorate %intt Volatile\n",                            AllCapabilities()),
make_pair("OpDecorate %intt Constant\n",                            KernelDependencies()),
make_pair("OpDecorate %intt Coherent\n",                            AllCapabilities()),
make_pair("OpDecorate %intt NonWritable\n",                         AllCapabilities()),
make_pair("OpDecorate %intt NonReadable\n",                         AllCapabilities()),
make_pair("OpDecorate %intt Uniform\n",                             ShaderDependencies()),
make_pair("OpDecorate %intt SaturatedConversion\n",                 KernelDependencies()),
make_pair("OpDecorate %intt Stream 0\n",                            vector<string>{"GeometryStreams"}),
make_pair("OpDecorate %intt Location 0\n",                          ShaderDependencies()),
make_pair("OpDecorate %intt Component 0\n",                         ShaderDependencies()),
make_pair("OpDecorate %intt Index 0\n",                             ShaderDependencies()),
make_pair("OpDecorate %intt Binding 0\n",                           ShaderDependencies()),
make_pair("OpDecorate %intt DescriptorSet 0\n",                     ShaderDependencies()),
make_pair("OpDecorate %intt Offset 0\n",                            AllCapabilities()),
make_pair("OpDecorate %intt XfbBuffer 0\n",                         vector<string>{"TransformFeedback"}),
make_pair("OpDecorate %intt XfbStride 0\n",                         vector<string>{"TransformFeedback"}),
make_pair("OpDecorate %intt FuncParamAttr Zext\n",                  KernelDependencies()),
make_pair("OpDecorate %intt FPRoundingMode RTE\n",                  KernelDependencies()),
make_pair("OpDecorate %intt FPFastMathMode Fast\n",                 KernelDependencies()),
make_pair("OpDecorate %intt LinkageAttributes \"other\" Import\n",  vector<string>{"Linkage"}),
make_pair("OpDecorate %intt NoContraction\n",                       ShaderDependencies()),
make_pair("OpDecorate %intt InputAttachmentIndex 0\n",              vector<string>{"InputAttachment"}),
make_pair("OpDecorate %intt Alignment 4\n",                         KernelDependencies())
  )));


INSTANTIATE_TEST_CASE_P(BuiltIn, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair("OpDecorate %intt BuiltIn Position\n",                  ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn PointSize\n",                 ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn ClipDistance\n",              ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn CullDistance\n",              ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn VertexId\n",                  ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn InstanceId\n",                ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn PrimitiveId\n",               GeometryTessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn InvocationId\n",              GeometryTessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn Layer\n",                     GeometryDependencies()),
make_pair("OpDecorate %intt BuiltIn ViewportIndex\n",             GeometryDependencies()),
make_pair("OpDecorate %intt BuiltIn TessLevelOuter\n",            TessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn TessLevelInner\n",            TessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn TessCoord\n",                 TessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn PatchVertices\n",             TessellationDependencies()),
make_pair("OpDecorate %intt BuiltIn FragCoord\n",                 ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn PointCoord\n",                ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn FrontFacing\n",               ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn SampleId\n",                  ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn SamplePosition\n",            ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn SampleMask\n",                ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn FragDepth\n",                 ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn HelperInvocation\n",          ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn VertexIndex\n",               ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn InstanceIndex\n",             ShaderDependencies()),
// Though the remaining builtins don't require Shader, the BuiltIn keyword
// itself currently does require it.
make_pair("OpDecorate %intt BuiltIn NumWorkgroups\n",             ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn WorkgroupSize\n",             ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn WorkgroupId\n",               ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn LocalInvocationId\n",         ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn GlobalInvocationId\n",        ShaderDependencies()),
make_pair("OpDecorate %intt BuiltIn LocalInvocationIndex",        ShaderDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn WorkDim\n",                   KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn GlobalSize\n",                KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn EnqueuedWorkgroupSize\n",     KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn GlobalOffset\n",              KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn GlobalLinearId\n",            KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn SubgroupSize\n",              KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn SubgroupMaxSize\n",           KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn NumSubgroups\n",              KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn NumEnqueuedSubgroups\n",      KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn SubgroupId\n",                KernelDependencies()),
make_pair("OpCapability Shader\n"
          "OpDecorate %intt BuiltIn SubgroupLocalInvocationId\n", KernelDependencies())
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

INSTANTIATE_TEST_CASE_P(MatrixOp, ValidateCapability,
                        Combine(
                            ValuesIn(AllCapabilities()),
                            Values(
make_pair(
          "%intt     = OpTypeInt 32 1\n"
          "%vec3     = OpTypeVector %intt 3\n"
          "%mat33    = OpTypeMatrix %vec3 3\n", MatrixDependencies()))));
// clang-format on

// Creates assembly containing an OpImageFetch instruction using operands for
// the image-operands part.  The assembly defines constants %fzero and %izero
// that can be used for operands where IDs are required.  The assembly is valid,
// apart from not declaring any capabilities required by the operands.
string ImageOperandsTemplate(const string& operands) {
  stringstream ss;
  // clang-format off
  ss << R"(
OpCapability Kernel
OpMemoryModel Logical OpenCL

%i32 = OpTypeInt 32 1
%f32 = OpTypeFloat 32
%v4i32 = OpTypeVector %i32 4
%timg = OpTypeImage %i32 2D 0 0 0 0 Unknown
%pimg = OpTypePointer UniformConstant %timg
%tfun = OpTypeFunction %i32

%vimg = OpVariable %pimg UniformConstant
%izero = OpConstant %i32 0
%fzero = OpConstant %f32 0.

%main = OpFunction %i32 None %tfun
%lbl = OpLabel
%img = OpLoad %timg %vimg
%r1 = OpImageFetch %v4i32 %img %izero )" << operands << R"(
OpReturnValue %izero
OpFunctionEnd
)";
  // clang-format on
  return ss.str();
}

INSTANTIATE_TEST_CASE_P(
    TwoImageOperandsMask, ValidateCapability,
    Combine(
        ValuesIn(AllCapabilities()),
        Values(make_pair(ImageOperandsTemplate("Bias|Lod %fzero %fzero"),
                         ShaderDependencies()),
               make_pair(ImageOperandsTemplate("Lod|Offset %fzero %izero"),
                         vector<string>{"ImageGatherExtended"}),
               make_pair(ImageOperandsTemplate("Sample|MinLod %izero %fzero"),
                         vector<string>{"MinLod"}),
               make_pair(ImageOperandsTemplate("Lod|Sample %fzero %izero"),
                         AllCapabilities()))));

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

}  // namespace anonymous
