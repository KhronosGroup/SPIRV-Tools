// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_replace_id_with_synonym.h"

#include <algorithm>

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceIdWithSynonym::TransformationReplaceIdWithSynonym(
    const spvtools::fuzz::protobufs::TransformationReplaceIdWithSynonym&
        message)
    : message_(message) {}

TransformationReplaceIdWithSynonym::TransformationReplaceIdWithSynonym(
    protobufs::IdUseDescriptor id_use_descriptor, uint32_t synonymous_id) {
  *message_.mutable_id_use_descriptor() = std::move(id_use_descriptor);
  message_.set_synonymous_id(synonymous_id);
}

bool TransformationReplaceIdWithSynonym::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto id_of_interest = message_.id_use_descriptor().id_of_interest();

  // Does the fact manager know about the synonym?
  auto data_descriptor_for_synonymous_id =
      MakeDataDescriptor(message_.synonymous_id(), {});
  if (!transformation_context.GetFactManager()->IsSynonymous(
          MakeDataDescriptor(id_of_interest, {}),
          data_descriptor_for_synonymous_id)) {
    return false;
  }

  // Does the id use descriptor in the transformation identify an instruction?
  auto use_instruction =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  if (!use_instruction) {
    return false;
  }

  uint32_t type_id_of_interest =
      ir_context->get_def_use_mgr()->GetDef(id_of_interest)->type_id();
  uint32_t type_id_synonym = ir_context->get_def_use_mgr()
                                 ->GetDef(message_.synonymous_id())
                                 ->type_id();

  // If the id of interest and the synonym are scalar or vector integer
  // constants with different signedness, their use can only be swapped if the
  // instruction is agnostic to the signedness of the operand.
  if (!TypesAreCompatible(ir_context, use_instruction->opcode(),
                          message_.id_use_descriptor().in_operand_index(),
                          type_id_of_interest, type_id_synonym)) {
    return false;
  }

  // Is the use suitable for being replaced in principle?
  if (!fuzzerutil::IdUseCanBeReplaced(
          ir_context, use_instruction,
          message_.id_use_descriptor().in_operand_index())) {
    return false;
  }

  // The transformation is applicable if the synonymous id is available at the
  // use point.
  return fuzzerutil::IdIsAvailableAtUse(
      ir_context, use_instruction,
      message_.id_use_descriptor().in_operand_index(),
      message_.synonymous_id());
}

void TransformationReplaceIdWithSynonym::Apply(
    spvtools::opt::IRContext* ir_context,
    TransformationContext* /*unused*/) const {
  auto instruction_to_change =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  instruction_to_change->SetInOperand(
      message_.id_use_descriptor().in_operand_index(),
      {message_.synonymous_id()});
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationReplaceIdWithSynonym::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_replace_id_with_synonym() = message_;
  return result;
}

// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3582): Add all
//  opcodes that are agnostic to signedness of operands to function.
//  This is not exhaustive yet.
bool TransformationReplaceIdWithSynonym::IsAgnosticToSignednessOfOperand(
    SpvOp opcode, uint32_t use_in_operand_index) {
  switch (opcode) {
    case SpvOpSNegate:
    case SpvOpNot:
    case SpvOpIAdd:
    case SpvOpISub:
    case SpvOpIMul:
    case SpvOpSDiv:
    case SpvOpSRem:
    case SpvOpSMod:
    case SpvOpShiftRightLogical:
    case SpvOpShiftRightArithmetic:
    case SpvOpShiftLeftLogical:
    case SpvOpBitwiseOr:
    case SpvOpBitwiseXor:
    case SpvOpBitwiseAnd:
    case SpvOpIEqual:
    case SpvOpINotEqual:
    case SpvOpULessThan:
    case SpvOpSLessThan:
    case SpvOpUGreaterThan:
    case SpvOpSGreaterThan:
    case SpvOpULessThanEqual:
    case SpvOpSLessThanEqual:
    case SpvOpUGreaterThanEqual:
    case SpvOpSGreaterThanEqual:
      return true;
    case SpvOpAccessChain:
      // The signedness of indices does not matter.
      return use_in_operand_index > 0;
    default:
      // Conservatively assume that the id cannot be swapped in other
      // instructions.
      return false;
  }
}

bool TransformationReplaceIdWithSynonym::TypesAreCompatible(
    opt::IRContext* ir_context, SpvOp opcode, uint32_t use_in_operand_index,
    uint32_t type_id_1, uint32_t type_id_2) {
  assert(ir_context->get_type_mgr()->GetType(type_id_1) &&
         ir_context->get_type_mgr()->GetType(type_id_2) &&
         "Type ids are invalid");

  return type_id_1 == type_id_2 ||
         (IsAgnosticToSignednessOfOperand(opcode, use_in_operand_index) &&
          fuzzerutil::TypesAreEqualUpToSign(ir_context, type_id_1, type_id_2));
}

std::unordered_set<uint32_t> TransformationReplaceIdWithSynonym::GetFreshIds()
    const {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3851): Implement.
  assert(false && "Not implemented yet.");
  return {};
}

}  // namespace fuzz
}  // namespace spvtools
