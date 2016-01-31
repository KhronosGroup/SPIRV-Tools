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
#include <sstream>
#include <string>

#include "opcode.h"
#include "spirv_definition.h"
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

std::string ToString(spv_capability_mask_t mask,
                     const libspirv::AssemblyGrammar& grammar) {
  std::stringstream ss;
  for (int cap = SpvCapabilityMatrix;
       cap <= SpvCapabilityStorageImageWriteWithoutFormat; ++cap) {
    if (spvIsInBitfield(SPV_CAPABILITY_AS_MASK(cap), mask)) {
      spv_operand_desc desc;
      if (SPV_SUCCESS ==
          grammar.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
        ss << desc->name << " ";
      else
        ss << cap << " ";
    }
  }
  return ss.str();
}

}  // namespace anonymous

namespace libspirv {

spv_result_t CapCheck(ValidationState_t& _,
                      const spv_parsed_instruction_t* inst) {
  spv_opcode_desc opcode_desc;
  if (SPV_SUCCESS == _.grammar().lookupOpcode(inst->opcode, &opcode_desc) &&
      !_.HasAnyOf(opcode_desc->capabilities))
    return _.diag(SPV_ERROR_INVALID_CAPABILITY)
           << "Opcode " << spvOpcodeString(inst->opcode)
           << " requires one of these capabilities: "
           << ToString(opcode_desc->capabilities, _.grammar());
  for (int i = 0; i < inst->num_operands; ++i) {
    spv_operand_desc operand_desc;
    if (SPV_SUCCESS ==
            _.grammar().lookupOperand(inst->operands[i].type,
                                      inst->words[inst->operands[i].offset],
                                      &operand_desc) &&
        !_.HasAnyOf(operand_desc->capabilities))
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)
             << "Operand " << i + 1 << " of " << spvOpcodeString(inst->opcode)
             << " requires one of these capabilities: "
             << ToString(operand_desc->capabilities, _.grammar());
  }
  return SPV_SUCCESS;
}

// clang-format off
spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst) {
  if (_.is_enabled(SPV_VALIDATE_INSTRUCTION_BIT)) {
    if (inst->opcode == SpvOpCapability)
        _.registerCapability(
            static_cast<SpvCapability>(inst->words[inst->operands[0].offset]));
    if (inst->opcode == SpvOpVariable) {
        const auto storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[2].offset]);
        if (storage_class == SpvStorageClassGeneric)
          return _.diag(SPV_ERROR_INVALID_BINARY)
              << "OpVariable storage class cannot be Generic";
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
    }
    return CapCheck(_, inst);
  }
  return SPV_SUCCESS;
}
// clang-format on
}  // namespace libspirv
