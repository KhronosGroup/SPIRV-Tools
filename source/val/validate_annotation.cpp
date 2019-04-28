// Copyright (c) 2018 Google LLC.
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

#include "source/val/validate.h"

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

#include "spirv-tools/instructions.hpp"

namespace spvtools {
namespace val {
namespace {

bool IsValidWebGPUDecoration(uint32_t decoration) {
  switch (decoration) {
    case SpvDecorationSpecId:
    case SpvDecorationBlock:
    case SpvDecorationRowMajor:
    case SpvDecorationColMajor:
    case SpvDecorationArrayStride:
    case SpvDecorationMatrixStride:
    case SpvDecorationBuiltIn:
    case SpvDecorationNoPerspective:
    case SpvDecorationFlat:
    case SpvDecorationCentroid:
    case SpvDecorationRestrict:
    case SpvDecorationAliased:
    case SpvDecorationNonWritable:
    case SpvDecorationNonReadable:
    case SpvDecorationUniform:
    case SpvDecorationLocation:
    case SpvDecorationComponent:
    case SpvDecorationIndex:
    case SpvDecorationBinding:
    case SpvDecorationDescriptorSet:
    case SpvDecorationOffset:
    case SpvDecorationNoContraction:
      return true;
    default:
      return false;
  }
}

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

// Returns true if the decoration takes ID parameters.
// TODO(dneto): This can be generated from the grammar.
bool DecorationTakesIdParameters(uint32_t type) {
  switch (static_cast<SpvDecoration>(type)) {
    case SpvDecorationUniformId:
    case SpvDecorationAlignmentId:
    case SpvDecorationMaxByteOffsetId:
    case SpvDecorationHlslCounterBufferGOOGLE:
      return true;
    default:
      break;
  }
  return false;
}

struct AnnotationPassHandler
    : Dispatch<ValidationState_t&, const Instruction*> {
  spv_result_t do_Decorate(const IDecorate& inst, ValidationState_t& _,
                           const Instruction* old) override {
    const auto decoration = inst.GetDecoration();
    if (decoration == SpvDecorationSpecId) {
      const auto target_id = inst.GetTarget();
      const auto target = _.FindDef(target_id);
      if (!target || !spvOpcodeIsScalarSpecConstant(target->opcode())) {
        return _.diag(SPV_ERROR_INVALID_ID, old)
               << "OpDecorate SpecId decoration target <id> '"
               << _.getIdName(target_id)
               << "' is not a scalar specialization constant.";
      }
    }

    if (spvIsWebGPUEnv(_.context()->target_env) &&
        !IsValidWebGPUDecoration(decoration)) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "OpDecorate decoration '" << LogStringForDecoration(decoration)
             << "' is not valid for the WebGPU execution environment.";
    }

    if (DecorationTakesIdParameters(decoration)) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "Decorations taking ID parameters may not be used with "
                "OpDecorateId";
    }
    // TODO: Add validations for all decorations.
    return SPV_SUCCESS;
  }

  spv_result_t do_DecorateId(const IDecorateId& inst, ValidationState_t& _,
                             const Instruction* old) override {
    if (!DecorationTakesIdParameters(inst.GetDecoration())) {
      return _.diag(SPV_ERROR_INVALID_ID, old) << "Decorations that don't take "
                                                  "ID parameters may not be "
                                                  "used with OpDecorateId";
    }
    // TODO: Add validations for these decorations.
    // UniformId is covered elsewhere.
    return SPV_SUCCESS;
  }

  spv_result_t do_MemberDecorate(const IMemberDecorate& inst,
                                 ValidationState_t& _,
                                 const Instruction* old) override {
    const auto struct_type_id = inst.GetStructureType();
    const auto struct_type = _.FindInst<ITypeStruct>(struct_type_id);
    if (!struct_type) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "OpMemberDecorate Structure type <id> '"
             << _.getIdName(struct_type_id) << "' is not a struct type.";
    }
    const auto member = inst.GetMember();
    const auto member_count = struct_type->GetMembers().size();
    if (member_count <= member) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "Index " << member
             << " provided in OpMemberDecorate for struct <id> "
             << _.getIdName(struct_type_id)
             << " is out of bounds. The structure has " << member_count
             << " members. Largest valid index is " << member_count - 1 << ".";
    }

    const auto decoration = inst.GetDecoration();
    if (spvIsWebGPUEnv(_.context()->target_env) &&
        !IsValidWebGPUDecoration(decoration)) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "OpMemberDecorate decoration  '" << _.getIdName(decoration)
             << "' is not valid for the WebGPU execution environment.";
    }

    return SPV_SUCCESS;
  }

  spv_result_t do_DecorationGroup(const IDecorationGroup& inst,
                                  ValidationState_t& _,
                                  const Instruction* old) override {
    if (spvIsWebGPUEnv(_.context()->target_env)) {
      return _.diag(SPV_ERROR_INVALID_BINARY, old)
             << "OpDecorationGroup is not allowed in the WebGPU execution "
             << "environment.";
    }

    const auto decoration_group = _.FindDef(inst.GetIdResult());
    for (auto pair : decoration_group->uses()) {
      auto use = pair.first;
      if (use->opcode() != SpvOpDecorate &&
          use->opcode() != SpvOpGroupDecorate &&
          use->opcode() != SpvOpGroupMemberDecorate &&
          use->opcode() != SpvOpName) {
        return _.diag(SPV_ERROR_INVALID_ID, old)
               << "Result id of OpDecorationGroup can only "
               << "be targeted by OpName, OpGroupDecorate, "
               << "OpDecorate, and OpGroupMemberDecorate";
      }
    }
    return SPV_SUCCESS;
  }

  spv_result_t do_GroupDecorate(const IGroupDecorate& inst,
                                ValidationState_t& _,
                                const Instruction* old) override {
    if (spvIsWebGPUEnv(_.context()->target_env)) {
      return _.diag(SPV_ERROR_INVALID_BINARY, old)
             << "OpGroupDecorate is not allowed in the WebGPU execution "
             << "environment.";
    }

    const auto decoration_group_id = inst.GetDecorationGroup();
    const auto decoration_group =
        _.FindInst<IDecorationGroup>(decoration_group_id);
    if (!decoration_group) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "OpGroupDecorate Decoration group <id> '"
             << _.getIdName(decoration_group_id)
             << "' is not a decoration group.";
    }
    for (auto target_id : inst.GetWords(1)) {
      auto target = _.FindDef(target_id);
      if (!target || target->opcode() == SpvOpDecorationGroup) {
        return _.diag(SPV_ERROR_INVALID_ID, old)
               << "OpGroupDecorate may not target OpDecorationGroup <id> '"
               << _.getIdName(target_id) << "'";
      }
    }
    return SPV_SUCCESS;
  }

  spv_result_t do_GroupMemberDecorate(const IGroupMemberDecorate& inst,
                                      ValidationState_t& _,
                                      const Instruction* old) override {
    if (spvIsWebGPUEnv(_.context()->target_env)) {
      return _.diag(SPV_ERROR_INVALID_BINARY, old)
             << "OpGroupMemberDecorate is not allowed in the WebGPU execution "
             << "environment.";
    }

    const auto decoration_group_id = inst.GetDecorationGroup();
    const auto decoration_group =
        _.FindInst<IDecorationGroup>(decoration_group_id);
    if (!decoration_group) {
      return _.diag(SPV_ERROR_INVALID_ID, old)
             << "OpGroupMemberDecorate Decoration group <id> '"
             << _.getIdName(decoration_group_id)
             << "' is not a decoration group.";
    }

    for (const auto& target : inst.GetTargets()) {
      const uint32_t struct_id = target.first;
      const uint32_t index = target.second;
      auto struct_instr = _.FindInst<ITypeStruct>(struct_id);
      if (!struct_instr) {
        return _.diag(SPV_ERROR_INVALID_ID, old)
               << "OpGroupMemberDecorate Structure type <id> '"
               << _.getIdName(struct_id) << "' is not a struct type.";
      }
      const auto num_struct_members = struct_instr->GetMembers().size();
      if (index >= num_struct_members) {
        return _.diag(SPV_ERROR_INVALID_ID, old)
               << "Index " << index
               << " provided in OpGroupMemberDecorate for struct <id> "
               << _.getIdName(struct_id)
               << " is out of bounds. The structure has " << num_struct_members
               << " members. Largest valid index is " << num_struct_members - 1
               << ".";
      }
    }
    return SPV_SUCCESS;
  }
};

// Registers necessary decoration(s) for the appropriate IDs based on the
// instruction.
struct RegisterDecorations : Dispatch<ValidationState_t&> {
  spv_result_t do_Decorate(const IDecorate& inst,
                           ValidationState_t& _) override {
    _.RegisterDecorationForId(
        inst.GetTarget(), Decoration(inst.GetDecoration(), inst.GetWords(2)));
    return SPV_SUCCESS;
  }
  spv_result_t do_DecorateId(const IDecorateId& inst,
                             ValidationState_t& _) override {
    _.RegisterDecorationForId(
        inst.GetTarget(), Decoration(inst.GetDecoration(), inst.GetWords(2)));
    return SPV_SUCCESS;
  }
  // TODO(dneto): SpvOpDecorateStringGOOGLE
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/2253
  spv_result_t do_MemberDecorate(const IMemberDecorate& inst,
                                 ValidationState_t& _) override {
    _.RegisterDecorationForId(
        inst.GetStructureType(),
        Decoration(inst.GetDecoration(), inst.GetWords(3), inst.GetMember()));
    return SPV_SUCCESS;
  }
  spv_result_t do_DecorationGroup(const IDecorationGroup&,
                                  ValidationState_t&) override {
    // We don't need to do anything right now. Assigning decorations to groups
    // will be taken care of via OpGroupDecorate.
    return SPV_SUCCESS;
  }
  spv_result_t do_GroupDecorate(const IGroupDecorate& inst,
                                ValidationState_t& _) override {
    std::vector<Decoration>& group_decorations =
        _.id_decorations(inst.GetDecorationGroup());
    for (auto target : inst.GetTargets())
      _.RegisterDecorationsForId(target, group_decorations.begin(),
                                 group_decorations.end());
    return SPV_SUCCESS;
  }
  spv_result_t do_GroupMemberDecorate(const IGroupMemberDecorate& inst,
                                      ValidationState_t& _) override {
    std::vector<Decoration>& group_decorations =
        _.id_decorations(inst.GetDecorationGroup());
    for (auto target : inst.GetTargets())
      // ID validation phase ensures this is in fact a struct instruction and
      // that the index is not out of bound.
      _.RegisterDecorationsForStructMember(target.first, target.second,
                                           group_decorations.begin(),
                                           group_decorations.end());
    return SPV_SUCCESS;
  }
};

}  // namespace

spv_result_t AnnotationPass(ValidationState_t& _, const Instruction* inst) {
  if (auto error = AnnotationPassHandler{}(inst->inst(), _, inst)) return error;

  // In order to validate decoration rules, we need to know all the decorations
  // that are applied to any given <id>.
  RegisterDecorations{}(inst->inst(), _);

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
