// Copyright (c) 2021 The Khronos Group Inc.
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

#include "source/val/decoration.h"
#include <string>

namespace spvtools {
namespace val {

std::string LogStringForDecoration(uint32_t decoration) {
  switch (decoration) {
    case SpvDecorationRelaxedPrecision:
      return "RelaxedPrecision";
    case SpvDecorationSpecId:
      return "SpecId";
    case SpvDecorationBlock:
      return "Block";
    case SpvDecorationBufferBlock:
      return "BufferBlock";
    case SpvDecorationRowMajor:
      return "RowMajor";
    case SpvDecorationColMajor:
      return "ColMajor";
    case SpvDecorationArrayStride:
      return "ArrayStride";
    case SpvDecorationMatrixStride:
      return "MatrixStride";
    case SpvDecorationGLSLShared:
      return "GLSLShared";
    case SpvDecorationGLSLPacked:
      return "GLSLPacked";
    case SpvDecorationCPacked:
      return "CPacked";
    case SpvDecorationBuiltIn:
      return "BuiltIn";
    case SpvDecorationNoPerspective:
      return "NoPerspective";
    case SpvDecorationFlat:
      return "Flat";
    case SpvDecorationPatch:
      return "Patch";
    case SpvDecorationCentroid:
      return "Centroid";
    case SpvDecorationSample:
      return "Sample";
    case SpvDecorationInvariant:
      return "Invariant";
    case SpvDecorationRestrict:
      return "Restrict";
    case SpvDecorationAliased:
      return "Aliased";
    case SpvDecorationVolatile:
      return "Volatile";
    case SpvDecorationConstant:
      return "Constant";
    case SpvDecorationCoherent:
      return "Coherent";
    case SpvDecorationNonWritable:
      return "NonWritable";
    case SpvDecorationNonReadable:
      return "NonReadable";
    case SpvDecorationUniform:
      return "Uniform";
    case SpvDecorationSaturatedConversion:
      return "SaturatedConversion";
    case SpvDecorationStream:
      return "Stream";
    case SpvDecorationLocation:
      return "Location";
    case SpvDecorationComponent:
      return "Component";
    case SpvDecorationIndex:
      return "Index";
    case SpvDecorationBinding:
      return "Binding";
    case SpvDecorationDescriptorSet:
      return "DescriptorSet";
    case SpvDecorationOffset:
      return "Offset";
    case SpvDecorationXfbBuffer:
      return "XfbBuffer";
    case SpvDecorationXfbStride:
      return "XfbStride";
    case SpvDecorationFuncParamAttr:
      return "FuncParamAttr";
    case SpvDecorationFPRoundingMode:
      return "FPRoundingMode";
    case SpvDecorationFPFastMathMode:
      return "FPFastMathMode";
    case SpvDecorationLinkageAttributes:
      return "LinkageAttributes";
    case SpvDecorationNoContraction:
      return "NoContraction";
    case SpvDecorationInputAttachmentIndex:
      return "InputAttachmentIndex";
    case SpvDecorationAlignment:
      return "Alignment";
    case SpvDecorationMaxByteOffset:
      return "MaxByteOffset";
    case SpvDecorationAlignmentId:
      return "AlignmentId";
    case SpvDecorationMaxByteOffsetId:
      return "MaxByteOffsetId";
    case SpvDecorationNoSignedWrap:
      return "NoSignedWrap";
    case SpvDecorationNoUnsignedWrap:
      return "NoUnsignedWrap";
    case SpvDecorationExplicitInterpAMD:
      return "ExplicitInterpAMD";
    case SpvDecorationOverrideCoverageNV:
      return "OverrideCoverageNV";
    case SpvDecorationPassthroughNV:
      return "PassthroughNV";
    case SpvDecorationViewportRelativeNV:
      return "ViewportRelativeNV";
    case SpvDecorationSecondaryViewportRelativeNV:
      return "SecondaryViewportRelativeNV";
    case SpvDecorationPerPrimitiveNV:
      return "PerPrimitiveNV";
    case SpvDecorationPerViewNV:
      return "PerViewNV";
    case SpvDecorationPerTaskNV:
      return "PerTaskNV";
    case SpvDecorationPerVertexNV:
      return "PerVertexNV";
    case SpvDecorationNonUniformEXT:
      return "NonUniformEXT";
    case SpvDecorationRestrictPointerEXT:
      return "RestrictPointerEXT";
    case SpvDecorationAliasedPointerEXT:
      return "AliasedPointerEXT";
    case SpvDecorationHlslCounterBufferGOOGLE:
      return "HlslCounterBufferGOOGLE";
    case SpvDecorationHlslSemanticGOOGLE:
      return "HlslSemanticGOOGLE";
    default:
      break;
  }
  return "Unknown";
}

}  // namespace val
}  // namespace spvtools
