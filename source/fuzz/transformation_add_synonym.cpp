// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_add_synonym.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/instruction_message.h"

namespace spvtools {
namespace fuzz {

TransformationAddSynonym::TransformationAddSynonym(
    protobufs::TransformationAddSynonym message)
    : message_(std::move(message)) {}

TransformationAddSynonym::TransformationAddSynonym(
    uint32_t result_id, const protobufs::InstructionDescriptor& insert_before,
    const protobufs::Instruction& synonymous_instruction) {
  message_.set_result_id(result_id);
  *message_.mutable_insert_before() = insert_before;
  *message_.mutable_synonymous_instruction() = synonymous_instruction;
}

bool TransformationAddSynonym::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that |message_.synonym_id| is valid.
  auto* synonym = ir_context->get_def_use_mgr()->GetDef(message_.result_id());
  if (!synonym) {
    return false;
  }

  // Check that |insert_before| is valid.
  auto* insert_before_inst =
      FindInstruction(message_.insert_before(), ir_context);
  if (!insert_before_inst) {
    return false;
  }

  // Check that we can insert |message._synonymous_instruction| before
  // |message_.insert_before| instruction.
  auto opcode = static_cast<SpvOp>(message_.synonymous_instruction().opcode());
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(opcode,
                                                    insert_before_inst)) {
    return false;
  }

  // Check that synonymous instruction has fresh id.
  if (!fuzzerutil::IsFreshId(ir_context,
                             message_.synonymous_instruction().result_id())) {
    return false;
  }

  // Domination rules must be satisfied.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, insert_before_inst, message_.result_id())) {
    return false;
  }

  // Check that new synonymous instruction is valid.
  switch (opcode) {
    case SpvOpIAdd:
    case SpvOpIMul:
    case SpvOpFAdd:
    case SpvOpFMul:
    case SpvOpLogicalOr:
    case SpvOpLogicalAnd: {
      if (message_.synonymous_instruction().input_operand_size() != 2) {
        return false;
      }

      const auto& lhs = message_.synonymous_instruction().input_operand(0);
      const auto& rhs = message_.synonymous_instruction().input_operand(1);
      if (lhs.operand_type() != SPV_OPERAND_TYPE_ID ||
          rhs.operand_type() != SPV_OPERAND_TYPE_ID) {
        return false;
      }

      auto lhs_type_id = fuzzerutil::GetTypeId(ir_context, lhs.operand_data(0));
      auto rhs_type_id = fuzzerutil::GetTypeId(ir_context, rhs.operand_data(0));
      if (lhs_type_id != rhs_type_id ||
          lhs_type_id != message_.synonymous_instruction().result_type_id()) {
        return false;
      }

      const auto* type = ir_context->get_type_mgr()->GetType(lhs_type_id);
      if (!type) {
        return false;
      }

      switch (opcode) {
        case SpvOpIAdd:
        case SpvOpIMul:
          return type->AsInteger() ||
                 (type->AsVector() &&
                  type->AsVector()->element_type()->AsInteger());
        case SpvOpFMul:
        case SpvOpFAdd:
          return type->AsFloat() ||
                 (type->AsVector() &&
                  type->AsVector()->element_type()->AsFloat());
        case SpvOpLogicalOr:
        case SpvOpLogicalAnd:
          return type->AsBool() || (type->AsVector() &&
                                    type->AsVector()->element_type()->AsBool());
        default:
          assert(false && "Unreachable");
          return false;
      }
    }
    default:
      assert(false && "Instruction is not supported");
      return false;
  }
}

void TransformationAddSynonym::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  FindInstruction(message_.insert_before(), ir_context)
      ->InsertBefore(InstructionFromMessage(ir_context,
                                            message_.synonymous_instruction()));

  fuzzerutil::UpdateModuleIdBound(
      ir_context, message_.synonymous_instruction().result_id());

  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);

  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.result_id(), {}),
      MakeDataDescriptor(message_.synonymous_instruction().result_id(), {}),
      ir_context);
}

protobufs::Transformation TransformationAddSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
