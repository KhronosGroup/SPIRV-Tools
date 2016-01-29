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

// Performs validation on instructions that appear inside of a SPIR-V block.

#include <cassert>
#include "validate_passes.h"

using libspirv::ValidationState_t;

namespace {

#define STORAGE_CLASS_CASE(CLASS, CAPABILITY)                                \
  case SpvStorageClass##CLASS:                                               \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {               \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                            \
             << #CLASS " storage class requires " #CAPABILITY " capability"; \
    }                                                                        \
    break

spv_result_t StorageClassCapabilityCheck(ValidationState_t& _,
                                         SpvStorageClass storage_class) {
  switch (storage_class) {
    STORAGE_CLASS_CASE(Input, Shader);
    STORAGE_CLASS_CASE(Uniform, Shader);
    STORAGE_CLASS_CASE(Output, Shader);
    STORAGE_CLASS_CASE(Private, Shader);
    STORAGE_CLASS_CASE(Generic, Kernel);
    STORAGE_CLASS_CASE(PushConstant, Shader);
    STORAGE_CLASS_CASE(AtomicCounter, AtomicStorage);
    default:
      // No capabilities are required for UniformConstant, WorkgroupLocal,
      // WorkgroupGlobal, Function, and Image
      break;
  }
  return SPV_SUCCESS;
}
#undef VARIABLE_STORAGE_CASE

#define DECORATION_CASE(DECORATION, CAPABILITY)                                \
  case SpvDecoration##DECORATION:                                              \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {                 \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                              \
             << #DECORATION " decoration requires " #CAPABILITY " capability"; \
    }                                                                          \
    break

#define BUILTIN_CASE(BUILTIN, CAPABILITY)                                \
  case SpvBuiltIn##BUILTIN:                                              \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {           \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                        \
             << #BUILTIN " builtin requires " #CAPABILITY " capability"; \
    }                                                                    \
    break

#define BUILTIN_CASE2(BUILTIN, CAPABILITY1, CAPABILITY2)                       \
  case SpvBuiltIn##BUILTIN:                                                    \
    if (_.hasCapability(SpvCapability##CAPABILITY1) == false &&                \
        _.hasCapability(SpvCapability##CAPABILITY2) == false) {                \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                              \
             << #BUILTIN " builtin requires " #CAPABILITY1 " or " #CAPABILITY2 \
                         " capabilities";                                      \
    }                                                                          \
    break

spv_result_t DecorationCapabilityCheck(ValidationState_t& _,
                                       SpvDecoration decoration,
                                       SpvBuiltIn optional_builtin) {
  switch (decoration) {
    DECORATION_CASE(RelaxedPrecision, Shader);
    DECORATION_CASE(SpecId, Shader);
    DECORATION_CASE(Block, Shader);
    DECORATION_CASE(BufferBlock, Shader);
    DECORATION_CASE(RowMajor, Matrix);
    DECORATION_CASE(ColMajor, Matrix);
    DECORATION_CASE(ArrayStride, Shader);
    DECORATION_CASE(MatrixStride, Shader);
    DECORATION_CASE(GLSLShared, Shader);
    DECORATION_CASE(GLSLPacked, Shader);
    DECORATION_CASE(CPacked, Kernel);
    DECORATION_CASE(NoPerspective, Shader);
    DECORATION_CASE(Flat, Shader);
    DECORATION_CASE(Patch, Tessellation);
    DECORATION_CASE(Centroid, Shader);
    DECORATION_CASE(Sample, Shader);
    DECORATION_CASE(Invariant, Shader);
    DECORATION_CASE(Constant, Kernel);
    DECORATION_CASE(Uniform, Shader);
    DECORATION_CASE(SaturatedConversion, Kernel);
    DECORATION_CASE(Stream, GeometryStreams);
    DECORATION_CASE(Location, Shader);
    DECORATION_CASE(Component, Shader);
    DECORATION_CASE(Index, Shader);
    DECORATION_CASE(Binding, Shader);
    DECORATION_CASE(DescriptorSet, Shader);
    DECORATION_CASE(XfbBuffer, TransformFeedback);
    DECORATION_CASE(XfbStride, TransformFeedback);
    DECORATION_CASE(FuncParamAttr, Kernel);
    DECORATION_CASE(FPRoundingMode, Kernel);
    DECORATION_CASE(FPFastMathMode, Kernel);
    DECORATION_CASE(LinkageAttributes, Linkage);
    DECORATION_CASE(NoContraction, Shader);
    DECORATION_CASE(Alignment, Kernel);
    DECORATION_CASE(InputAttachmentIndex, InputAttachment);
    case SpvDecorationBuiltIn:
      switch (optional_builtin) {
        BUILTIN_CASE(Position, Shader);
        BUILTIN_CASE(PointSize, Shader);
        BUILTIN_CASE(ClipDistance, ClipDistance);
        BUILTIN_CASE(CullDistance, CullDistance);
        BUILTIN_CASE(VertexId, Shader);
        BUILTIN_CASE(InstanceId, Shader);
        BUILTIN_CASE2(PrimitiveId, Geometry, Tessellation);
        BUILTIN_CASE2(InvocationId, Geometry, Tessellation);
        BUILTIN_CASE(Layer, Geometry);
        case SpvBuiltInViewportIndex:
          assert(
              false &&
              "UNHANDLED");  // TODO(umar): missing SpvCapabilityMultiViewport
          // BUILTIN_CASE(ViewportIndex, MultiViewport);
          BUILTIN_CASE(TessLevelOuter, Tessellation);
          BUILTIN_CASE(TessLevelInner, Tessellation);
          BUILTIN_CASE(TessCoord, Tessellation);
          BUILTIN_CASE(PatchVertices, Tessellation);
          BUILTIN_CASE(FragCoord, Shader);
          BUILTIN_CASE(PointCoord, Shader);
          BUILTIN_CASE(FrontFacing, Shader);
          BUILTIN_CASE(SampleId, SampleRateShading);
          BUILTIN_CASE(SamplePosition, SampleRateShading);
          BUILTIN_CASE(SampleMask, SampleRateShading);
          BUILTIN_CASE(FragDepth, Shader);
          BUILTIN_CASE(HelperInvocation, Shader);
          BUILTIN_CASE(WorkDim, Kernel);
          BUILTIN_CASE(GlobalSize, Kernel);
          BUILTIN_CASE(EnqueuedWorkgroupSize, Kernel);
          BUILTIN_CASE(GlobalOffset, Kernel);
          BUILTIN_CASE(GlobalLinearId, Kernel);
          BUILTIN_CASE(SubgroupSize, Kernel);
          BUILTIN_CASE(SubgroupMaxSize, Kernel);
          BUILTIN_CASE(NumSubgroups, Kernel);
          BUILTIN_CASE(NumEnqueuedSubgroups, Kernel);
          BUILTIN_CASE(SubgroupId, Kernel);
          BUILTIN_CASE(SubgroupLocalInvocationId, Kernel);
          BUILTIN_CASE(VertexIndex, Shader);
          BUILTIN_CASE(InstanceIndex, Shader);
        case SpvBuiltInNumWorkgroups:
        case SpvBuiltInWorkgroupSize:
        case SpvBuiltInWorkgroupId:
        case SpvBuiltInLocalInvocationId:
        case SpvBuiltInGlobalInvocationId:
        case SpvBuiltInLocalInvocationIndex:
          break;
      }
    default:
      // No capabilities are required for Restrict, Aliased, BuiltIn, Volatile,
      // Coherent, NonWritable, NonReadable, and Offset
      break;
  }
#undef DECORATION_CASE
#undef BUILTIN_CASE
#undef BUILTIN_CASE2

  return SPV_SUCCESS;
}

#define EXECUTION_MODEL_CASE(MODEL, CAPABILITY)                                \
  case SpvExecutionModel##MODEL:                                               \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {                 \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                              \
             << #MODEL " execution model requires " #CAPABILITY " capability"; \
    }                                                                          \
    break

spv_result_t ExecutionModelCapabilityCheck(ValidationState_t& _,
                                           SpvExecutionModel execution_model) {
  switch (execution_model) {
    EXECUTION_MODEL_CASE(Vertex, Shader);
    EXECUTION_MODEL_CASE(TessellationControl, Tessellation);
    EXECUTION_MODEL_CASE(TessellationEvaluation, Tessellation);
    EXECUTION_MODEL_CASE(Geometry, Geometry);
    EXECUTION_MODEL_CASE(Fragment, Shader);
    EXECUTION_MODEL_CASE(GLCompute, Shader);
    EXECUTION_MODEL_CASE(Kernel, Kernel);
  }
#undef EXECUTION_MODEL_CASE
  return SPV_SUCCESS;
}

spv_result_t AddressingAndMemoryModelCapabilityCheck(
    ValidationState_t& _, SpvAddressingModel addressing_model,
    SpvMemoryModel memory_model) {
  switch (addressing_model) {
    case SpvAddressingModelPhysical32:
    case SpvAddressingModelPhysical64:
      if (_.hasCapability(SpvCapabilityAddresses) == false) {
        return _.diag(SPV_ERROR_INVALID_CAPABILITY)
               << "Physical32 and Physical64 addressing models require the "
                  "Addresses capability";
      }
      break;
    case SpvAddressingModelLogical:
      break;
  }

  switch (memory_model) {
    case SpvMemoryModelSimple:
    case SpvMemoryModelGLSL450:
      if (_.hasCapability(SpvCapabilityShader) == false) {
        return _.diag(SPV_ERROR_INVALID_CAPABILITY)
               << "Simple and GLSL450 memory models require the Shader "
                  "capability";
      }
      break;
    case SpvMemoryModelOpenCL:
      if (_.hasCapability(SpvCapabilityKernel) == false) {
        return _.diag(SPV_ERROR_INVALID_CAPABILITY)
               << "OpenCL memory model requires the Kernel capability";
      }
      break;
  }
  return SPV_SUCCESS;
}

#define EXECUTION_MODE_CASE(MODE, CAPABILITY)                      \
  case SpvExecutionMode##MODE:                                     \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {     \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                  \
             << #MODE " mode requires " #CAPABILITY " capability"; \
    }                                                              \
    break
#define EXECUTION_MODE_CASE2(MODE, CAPABILITY1, CAPABILITY2)                   \
  case SpvExecutionMode##MODE:                                                 \
    if (_.hasCapability(SpvCapability##CAPABILITY1) == false &&                \
        _.hasCapability(SpvCapability##CAPABILITY2) == false) {                \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY) << #MODE                     \
             " mode requires " #CAPABILITY1 " or " #CAPABILITY2 " capability"; \
    }                                                                          \
    break
spv_result_t ExecutionModeCapabilityCheck(ValidationState_t& _,
                                          SpvExecutionMode execution_mode) {
  switch (execution_mode) {
    EXECUTION_MODE_CASE(Invocations, Geometry);
    EXECUTION_MODE_CASE(SpacingEqual, Tessellation);
    EXECUTION_MODE_CASE(SpacingFractionalEven, Tessellation);
    EXECUTION_MODE_CASE(SpacingFractionalOdd, Tessellation);
    EXECUTION_MODE_CASE(VertexOrderCw, Tessellation);
    EXECUTION_MODE_CASE(VertexOrderCcw, Tessellation);
    EXECUTION_MODE_CASE(PixelCenterInteger, Shader);
    EXECUTION_MODE_CASE(OriginUpperLeft, Shader);
    EXECUTION_MODE_CASE(OriginLowerLeft, Shader);
    EXECUTION_MODE_CASE(EarlyFragmentTests, Shader);
    EXECUTION_MODE_CASE(PointMode, Tessellation);
    EXECUTION_MODE_CASE(Xfb, TransformFeedback);
    EXECUTION_MODE_CASE(DepthReplacing, Shader);
    EXECUTION_MODE_CASE(DepthGreater, Shader);
    EXECUTION_MODE_CASE(DepthLess, Shader);
    EXECUTION_MODE_CASE(DepthUnchanged, Shader);
    EXECUTION_MODE_CASE(LocalSizeHint, Kernel);
    EXECUTION_MODE_CASE(InputPoints, Geometry);
    EXECUTION_MODE_CASE(InputLines, Geometry);
    EXECUTION_MODE_CASE(InputLinesAdjacency, Geometry);
    EXECUTION_MODE_CASE2(Triangles, Geometry, Tessellation);
    EXECUTION_MODE_CASE(InputTrianglesAdjacency, Geometry);
    EXECUTION_MODE_CASE(Quads, Tessellation);
    EXECUTION_MODE_CASE(Isolines, Tessellation);
    EXECUTION_MODE_CASE2(OutputVertices, Geometry, Tessellation);
    EXECUTION_MODE_CASE(OutputPoints, Geometry);
    EXECUTION_MODE_CASE(OutputLineStrip, Geometry);
    EXECUTION_MODE_CASE(OutputTriangleStrip, Geometry);
    EXECUTION_MODE_CASE(VecTypeHint, Kernel);
    EXECUTION_MODE_CASE(ContractionOff, Kernel);
    case SpvExecutionModeLocalSize:
      break;
  }
#undef EXECUTION_MODE_CASE
#undef EXECUTION_MODE_CASE2
  return SPV_SUCCESS;
}

#define DIM_CASE(DIM, CAPABILITY)                                   \
  case SpvDim##DIM:                                                 \
    if (_.hasCapability(SpvCapability##CAPABILITY) == false) {      \
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)                   \
             << "Dim " #DIM " requires " #CAPABILITY " capability"; \
    }                                                               \
    break

spv_result_t DimCapabilityCheck(ValidationState_t& _, SpvDim dim) {
  switch (dim) {
    DIM_CASE(1D, Sampled1D);
    DIM_CASE(Cube, Shader);
    DIM_CASE(Rect, SampledRect);
    DIM_CASE(Buffer, SampledBuffer);
    DIM_CASE(SubpassData, InputAttachment);
    case SpvDim2D:
    case SpvDim3D:
      break;
  }
#undef DIM_CASE
  return SPV_SUCCESS;
}

spv_result_t SamplerAddressingModeCapabilityCheck(
    ValidationState_t& _, SpvSamplerAddressingMode sampler_addressing_mode) {
  std::string mode;
  switch (sampler_addressing_mode) {
    case SpvSamplerAddressingModeNone:
      mode = "None";
      break;
    case SpvSamplerAddressingModeClampToEdge:
      mode = "ClampToEdge";
      break;
    case SpvSamplerAddressingModeClamp:
      mode = "Clamp";
      break;
    case SpvSamplerAddressingModeRepeat:
      mode = "Repeat";
      break;
    case SpvSamplerAddressingModeRepeatMirrored:
      mode = "RepeatMirrored";
      break;
  }
  if (_.hasCapability(SpvCapabilityKernel) == false) {
    return _.diag(SPV_ERROR_INVALID_CAPABILITY)
           << mode + " sample address mode requires Kernel capability";
  }
  return SPV_SUCCESS;
}
}

namespace libspirv {

// clang-format off
spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst) {
  if (_.is_enabled(SPV_VALIDATE_INSTRUCTION_BIT)) {
    SpvOp opcode = inst->opcode;
    switch (opcode) {
      case SpvOpNop: break;
      case SpvOpSourceContinued: break;
      case SpvOpSource: break;
      case SpvOpSourceExtension: break;
      case SpvOpName: break;
      case SpvOpMemberName: break;
      case SpvOpString: break;
      case SpvOpLine: break;
      case SpvOpNoLine: break;
      case SpvOpDecorate: {
        SpvDecoration decoration =
          static_cast<SpvDecoration>(inst->words[inst->operands[1].offset]);
        SpvBuiltIn builtin = static_cast<SpvBuiltIn>(0);
        if(decoration == SpvDecorationBuiltIn) {
          builtin = static_cast<SpvBuiltIn>(inst->words[inst->operands[2].offset]);
        }
        spvCheckReturn(DecorationCapabilityCheck(_, decoration, builtin));
      } break;

      case SpvOpMemberDecorate: {
        SpvDecoration decoration =
          static_cast<SpvDecoration>(inst->words[inst->operands[2].offset]);
        SpvBuiltIn builtin = static_cast<SpvBuiltIn>(0);
        if(decoration == SpvDecorationBuiltIn) {
          builtin = static_cast<SpvBuiltIn>(inst->words[inst->operands[3].offset]);
        }
        spvCheckReturn(DecorationCapabilityCheck(_, decoration, builtin));
      } break;

      case SpvOpDecorationGroup: break;
      case SpvOpGroupDecorate: break;
      case SpvOpGroupMemberDecorate: break;
      case SpvOpExtension: break;
      case SpvOpExtInstImport: break;
      case SpvOpExtInst: break;
      case SpvOpMemoryModel: {
        SpvAddressingModel addressing_model =
          static_cast<SpvAddressingModel>(inst->words[inst->operands[0].offset]);
        SpvMemoryModel memory_model =
          static_cast<SpvMemoryModel>(inst->words[inst->operands[1].offset]);
        spvCheckReturn(AddressingAndMemoryModelCapabilityCheck(_,
                                               addressing_model, memory_model));
      } break;

      case SpvOpEntryPoint: {
        SpvExecutionModel execution_model =
          static_cast<SpvExecutionModel>(inst->words[inst->operands[0].offset]);
        spvCheckReturn(ExecutionModelCapabilityCheck(_, execution_model));
      } break;

      case SpvOpExecutionMode: {
        SpvExecutionMode execution_mode =
          static_cast<SpvExecutionMode>(inst->words[inst->operands[1].offset]);
        spvCheckReturn(ExecutionModeCapabilityCheck(_, execution_mode));
      } break;

      case SpvOpCapability:
        _.registerCapability(
            static_cast<SpvCapability>(inst->words[inst->operands[0].offset]));
        break;

      case SpvOpTypeVoid: break;
      case SpvOpTypeBool: break;
      case SpvOpTypeInt: break;
      case SpvOpTypeFloat: break;
      case SpvOpTypeVector: break;
      case SpvOpTypeMatrix:
        if (_.hasCapability(SpvCapabilityMatrix) == false) {
          return _.diag(SPV_ERROR_INVALID_CAPABILITY)
                 << "Matrix type requires Matrix capability";
        }
        break;
      case SpvOpTypeImage: {
        if (_.hasCapability(SpvCapabilityImageBasic) == false) {
          return _.diag(SPV_ERROR_INVALID_CAPABILITY)
            << "TypeImage requires the ImageBasic capability";
        }
        SpvDim dim =
          static_cast<SpvDim>(inst->words[inst->operands[2].offset]);
        spvCheckReturn(DimCapabilityCheck(_, dim));

      } break;

      case SpvOpTypeSampler: break;
      case SpvOpTypeSampledImage: break;
      case SpvOpTypeArray: break;
      case SpvOpTypeRuntimeArray: break;
      case SpvOpTypeStruct: break;
      case SpvOpTypeOpaque: break;
      case SpvOpTypePointer: {
        const SpvStorageClass storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[1].offset]);
        spvCheckReturn(StorageClassCapabilityCheck(_, storage_class));
      } break;

      case SpvOpTypeFunction: break;
      case SpvOpTypeEvent: break;
      case SpvOpTypeDeviceEvent: break;
      case SpvOpTypeReserveId: break;
      case SpvOpTypeQueue: break;
      case SpvOpTypePipe: break;
      case SpvOpTypeForwardPointer: {
        const SpvStorageClass storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[1].offset]);
        spvCheckReturn(StorageClassCapabilityCheck(_, storage_class));
      } break;

      case SpvOpUndef: break;
      case SpvOpConstantTrue: break;
      case SpvOpConstantFalse: break;
      case SpvOpConstant: break;
      case SpvOpConstantComposite: break;
      case SpvOpConstantSampler: {
        if (_.hasCapability(SpvCapabilityLiteralSampler) == false) {
        return _.diag(SPV_ERROR_INVALID_CAPABILITY)
          << "ConstantSampler requires the LiteralSampler capability";
        }
        const SpvSamplerAddressingMode sampler_addressing_mode =
          static_cast<SpvSamplerAddressingMode>(inst->words[inst->operands[1].offset]);
        spvCheckReturn(SamplerAddressingModeCapabilityCheck(_, sampler_addressing_mode));
      } break;

      case SpvOpConstantNull: break;
      case SpvOpSpecConstantTrue: break;
      case SpvOpSpecConstantFalse: break;
      case SpvOpSpecConstant: break;
      case SpvOpSpecConstantComposite: break;
      case SpvOpSpecConstantOp: break;
      case SpvOpVariable: {
        const SpvStorageClass storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[2].offset]);
        spvCheckReturn(StorageClassCapabilityCheck(_, storage_class));

        if (_.getLayoutSection() == kLayoutFunctionDefinitions) {
          if (storage_class != SpvStorageClassFunction) {
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "Variables must have a function[7] storage class inside"
                      " of a function";
          }
        } else {
          if (storage_class == SpvStorageClassFunction) {
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "Variables can not have a function[7] storage class "
                      "outside of a function";
          }
        }
      } break;

      case SpvOpImageTexelPointer: break;
      case SpvOpLoad: break;
      case SpvOpStore: break;
      case SpvOpCopyMemory: break;
      case SpvOpCopyMemorySized: break;
      case SpvOpAccessChain: break;
      case SpvOpInBoundsAccessChain: break;
      case SpvOpPtrAccessChain: break;
      case SpvOpArrayLength: break;
      case SpvOpGenericPtrMemSemantics: break;
      case SpvOpInBoundsPtrAccessChain: break;
      case SpvOpFunction: break;
      case SpvOpFunctionParameter: break;
      case SpvOpFunctionEnd: break;
      case SpvOpFunctionCall: break;
      case SpvOpSampledImage: break;
      case SpvOpImageSampleImplicitLod: break;
      case SpvOpImageSampleExplicitLod: break;
      case SpvOpImageSampleDrefImplicitLod: break;
      case SpvOpImageSampleDrefExplicitLod: break;
      case SpvOpImageSampleProjImplicitLod: break;
      case SpvOpImageSampleProjExplicitLod: break;
      case SpvOpImageSampleProjDrefImplicitLod: break;
      case SpvOpImageSampleProjDrefExplicitLod: break;
      case SpvOpImageFetch: break;
      case SpvOpImageGather: break;
      case SpvOpImageDrefGather: break;
      case SpvOpImageRead: break;
      case SpvOpImageWrite: break;
      case SpvOpImage: break;
      case SpvOpImageQueryFormat: break;
      case SpvOpImageQueryOrder: break;
      case SpvOpImageQuerySizeLod: break;
      case SpvOpImageQuerySize: break;
      case SpvOpImageQueryLod: break;
      case SpvOpImageQueryLevels: break;
      case SpvOpImageQuerySamples: break;
      case SpvOpImageSparseSampleImplicitLod: break;
      case SpvOpImageSparseSampleExplicitLod: break;
      case SpvOpImageSparseSampleDrefImplicitLod: break;
      case SpvOpImageSparseSampleDrefExplicitLod: break;
      case SpvOpImageSparseSampleProjImplicitLod: break;
      case SpvOpImageSparseSampleProjExplicitLod: break;
      case SpvOpImageSparseSampleProjDrefImplicitLod: break;
      case SpvOpImageSparseSampleProjDrefExplicitLod: break;
      case SpvOpImageSparseFetch: break;
      case SpvOpImageSparseGather: break;
      case SpvOpImageSparseDrefGather: break;
      case SpvOpImageSparseTexelsResident: break;
      case SpvOpConvertFToU: break;
      case SpvOpConvertFToS: break;
      case SpvOpConvertSToF: break;
      case SpvOpConvertUToF: break;
      case SpvOpUConvert: break;
      case SpvOpSConvert: break;
      case SpvOpFConvert: break;
      case SpvOpQuantizeToF16: break;
      case SpvOpConvertPtrToU: break;
      case SpvOpSatConvertSToU: break;
      case SpvOpSatConvertUToS: break;
      case SpvOpConvertUToPtr: break;
      case SpvOpPtrCastToGeneric: break;
      case SpvOpGenericCastToPtr: break;
      case SpvOpGenericCastToPtrExplicit: {
        const SpvStorageClass storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[3].offset]);
        spvCheckReturn(StorageClassCapabilityCheck(_, storage_class));
      } break;

      case SpvOpBitcast: break;
      case SpvOpVectorExtractDynamic: break;
      case SpvOpVectorInsertDynamic: break;
      case SpvOpVectorShuffle: break;
      case SpvOpCompositeConstruct: break;
      case SpvOpCompositeExtract: break;
      case SpvOpCompositeInsert: break;
      case SpvOpCopyObject: break;
      case SpvOpTranspose: break;
      case SpvOpSNegate: break;
      case SpvOpFNegate: break;
      case SpvOpIAdd: break;
      case SpvOpFAdd: break;
      case SpvOpISub: break;
      case SpvOpFSub: break;
      case SpvOpIMul: break;
      case SpvOpFMul: break;
      case SpvOpUDiv: break;
      case SpvOpSDiv: break;
      case SpvOpFDiv: break;
      case SpvOpUMod: break;
      case SpvOpSRem: break;
      case SpvOpSMod: break;
      case SpvOpFRem: break;
      case SpvOpFMod: break;
      case SpvOpVectorTimesScalar: break;
      case SpvOpMatrixTimesScalar: break;
      case SpvOpVectorTimesMatrix: break;
      case SpvOpMatrixTimesVector: break;
      case SpvOpMatrixTimesMatrix: break;
      case SpvOpOuterProduct: break;
      case SpvOpDot: break;
      case SpvOpIAddCarry: break;
      case SpvOpISubBorrow: break;
      case SpvOpUMulExtended: break;
      case SpvOpSMulExtended: break; break;
      case SpvOpShiftRightLogical: break;
      case SpvOpShiftRightArithmetic: break;
      case SpvOpShiftLeftLogical: break;
      case SpvOpBitwiseOr: break;
      case SpvOpBitwiseXor: break;
      case SpvOpBitwiseAnd: break;
      case SpvOpNot: break;
      case SpvOpBitFieldInsert: break;
      case SpvOpBitFieldSExtract: break;
      case SpvOpBitFieldUExtract: break;
      case SpvOpBitReverse: break;
      case SpvOpBitCount: break;
      case SpvOpAny: break;
      case SpvOpAll: break;
      case SpvOpIsNan: break;
      case SpvOpIsInf: break;
      case SpvOpIsFinite: break;
      case SpvOpIsNormal: break;
      case SpvOpSignBitSet: break;
      case SpvOpLessOrGreater: break;
      case SpvOpOrdered: break;
      case SpvOpUnordered: break;
      case SpvOpLogicalEqual: break;
      case SpvOpLogicalNotEqual: break;
      case SpvOpLogicalOr: break;
      case SpvOpLogicalAnd: break;
      case SpvOpLogicalNot: break;
      case SpvOpSelect: break;
      case SpvOpIEqual: break;
      case SpvOpINotEqual: break;
      case SpvOpUGreaterThan: break;
      case SpvOpSGreaterThan: break;
      case SpvOpUGreaterThanEqual: break;
      case SpvOpSGreaterThanEqual: break;
      case SpvOpULessThan: break;
      case SpvOpSLessThan: break;
      case SpvOpULessThanEqual: break;
      case SpvOpSLessThanEqual: break;
      case SpvOpFOrdEqual: break;
      case SpvOpFUnordEqual: break;
      case SpvOpFOrdNotEqual: break;
      case SpvOpFUnordNotEqual: break;
      case SpvOpFOrdLessThan: break;
      case SpvOpFUnordLessThan: break;
      case SpvOpFOrdGreaterThan: break;
      case SpvOpFUnordGreaterThan: break;
      case SpvOpFOrdLessThanEqual: break;
      case SpvOpFUnordLessThanEqual: break;
      case SpvOpFOrdGreaterThanEqual: break;
      case SpvOpFUnordGreaterThanEqual: break;
      case SpvOpDPdx: break;
      case SpvOpDPdy: break;
      case SpvOpFwidth: break;
      case SpvOpDPdxFine: break;
      case SpvOpDPdyFine: break;
      case SpvOpFwidthFine: break;
      case SpvOpDPdxCoarse: break;
      case SpvOpDPdyCoarse: break;
      case SpvOpFwidthCoarse: break;
      case SpvOpPhi: break;
      case SpvOpLoopMerge: break;
      case SpvOpSelectionMerge: break;
      case SpvOpLabel: break;
      case SpvOpBranch: break;
      case SpvOpBranchConditional: break;
      case SpvOpSwitch: break;
      case SpvOpKill: break;
      case SpvOpReturn: break;
      case SpvOpReturnValue: break;
      case SpvOpUnreachable: break;
      case SpvOpLifetimeStart: break;
      case SpvOpLifetimeStop: break;
      case SpvOpAtomicLoad: break;
      case SpvOpAtomicStore: break;
      case SpvOpAtomicExchange: break;
      case SpvOpAtomicCompareExchange: break;
      case SpvOpAtomicCompareExchangeWeak: break;
      case SpvOpAtomicIIncrement: break;
      case SpvOpAtomicIDecrement: break;
      case SpvOpAtomicIAdd: break;
      case SpvOpAtomicISub: break;
      case SpvOpAtomicSMin: break;
      case SpvOpAtomicUMin: break;
      case SpvOpAtomicSMax: break;
      case SpvOpAtomicUMax: break;
      case SpvOpAtomicAnd: break;
      case SpvOpAtomicOr: break;
      case SpvOpAtomicXor: break;
      case SpvOpAtomicFlagTestAndSet: break;
      case SpvOpAtomicFlagClear: break;
      case SpvOpEmitVertex: break;
      case SpvOpEndPrimitive: break;
      case SpvOpEmitStreamVertex: break;
      case SpvOpEndStreamPrimitive: break;
      case SpvOpControlBarrier: break;
      case SpvOpMemoryBarrier: break;
      case SpvOpGroupAsyncCopy: break;
      case SpvOpGroupWaitEvents: break;
      case SpvOpGroupAll: break;
      case SpvOpGroupAny: break;
      case SpvOpGroupBroadcast: break;
      case SpvOpGroupIAdd: break;
      case SpvOpGroupFAdd: break;
      case SpvOpGroupFMin: break;
      case SpvOpGroupUMin: break;
      case SpvOpGroupSMin: break;
      case SpvOpGroupFMax: break;
      case SpvOpGroupUMax: break;
      case SpvOpGroupSMax: break;
      case SpvOpEnqueueMarker: break;
      case SpvOpEnqueueKernel: break;
      case SpvOpGetKernelNDrangeSubGroupCount: break;
      case SpvOpGetKernelNDrangeMaxSubGroupSize: break;
      case SpvOpGetKernelWorkGroupSize: break;
      case SpvOpGetKernelPreferredWorkGroupSizeMultiple: break;
      case SpvOpRetainEvent: break;
      case SpvOpReleaseEvent: break;
      case SpvOpCreateUserEvent: break;
      case SpvOpIsValidEvent: break;
      case SpvOpSetUserEventStatus: break;
      case SpvOpCaptureEventProfilingInfo: break;
      case SpvOpGetDefaultQueue: break;
      case SpvOpBuildNDRange: break;
      case SpvOpReadPipe: break;
      case SpvOpWritePipe: break;
      case SpvOpReservedReadPipe: break;
      case SpvOpReservedWritePipe: break;
      case SpvOpReserveReadPipePackets: break;
      case SpvOpReserveWritePipePackets: break;
      case SpvOpCommitReadPipe: break;
      case SpvOpCommitWritePipe: break;
      case SpvOpIsValidReserveId: break;
      case SpvOpGetNumPipePackets: break;
      case SpvOpGetMaxPipePackets: break;
      case SpvOpGroupReserveReadPipePackets: break;
      case SpvOpGroupReserveWritePipePackets: break;
      case SpvOpGroupCommitReadPipe: break;
      case SpvOpGroupCommitWritePipe: break;
    }
  }
  return SPV_SUCCESS;
}
// clang-format on
}
