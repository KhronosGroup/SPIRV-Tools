// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_push_id_through_variable.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_load.h"
#include "source/fuzz/transformation_store.h"

namespace spvtools {
namespace fuzz {

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    const spvtools::fuzz::protobufs::TransformationPushIdThroughVariable&
        message)
    : message_(message) {}

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    uint32_t value_synonym_id, uint32_t value_id, uint32_t variable_id,
    const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_value_synonym_id(value_synonym_id);
  message_.set_value_id(value_id);
  message_.set_variable_id(variable_id);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationPushIdThroughVariable::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.value_synonym_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.value_synonym_id())) {
    return false;
  }

  // The variable instruction must exist and have a type.
  auto variable_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.variable_id());
  if (!variable_instruction || !variable_instruction->type_id()) {
    return false;
  }

  // The variable type instruction must be defined and be a OpTypePointer
  // instruction.
  auto variable_type_instruction =
      ir_context->get_def_use_mgr()->GetDef(variable_instruction->type_id());
  if (!variable_type_instruction ||
      variable_type_instruction->opcode() != SpvOpTypePointer) {
    return false;
  }

  // The variable must not be read-only.
  if (variable_instruction->IsReadOnlyPointer()) {
    return false;
  }

  // The instruction to insert before must be defined.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!instruction_to_insert_before) {
    return false;
  }

  // The instruction to insert before must belongs to a reachable block.
  if (!fuzzerutil::BlockIsReachableInItsFunction(
          ir_context,
          ir_context->get_instr_block(instruction_to_insert_before))) {
    return false;
  }

  // It must be valid to insert the OpStore and OpLoad instruction before it.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          SpvOpStore, instruction_to_insert_before) ||
      !fuzzerutil::CanInsertOpcodeBeforeInstruction(
          SpvOpLoad, instruction_to_insert_before)) {
    return false;
  }

  // The value instruction must be defined and have a type.
  auto value_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.value_id());
  if (!value_instruction || !value_instruction->type_id()) {
    return false;
  }

  // The variable pointee type must be equal to the value type.
  if (variable_type_instruction->GetSingleWordInOperand(1) !=
      value_instruction->type_id()) {
    return false;
  }

  // |message_.variable_id| must be available at the insertion point.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, instruction_to_insert_before, message_.variable_id())) {
    return false;
  }

  // |message_.value_id| must be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, instruction_to_insert_before, message_.value_id());
}

void TransformationPushIdThroughVariable::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  TransformationStore(message_.variable_id(), message_.value_id(),
                      message_.instruction_descriptor())
      .Apply(ir_context, transformation_context);

  TransformationLoad(message_.value_synonym_id(), message_.variable_id(),
                     message_.instruction_descriptor())
      .Apply(ir_context, transformation_context);

  transformation_context->GetFactManager()->AddFactDataSynonym(
      MakeDataDescriptor(message_.value_synonym_id(), {}),
      MakeDataDescriptor(message_.value_id(), {}), ir_context);
}

protobufs::Transformation TransformationPushIdThroughVariable::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_push_id_through_variable() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
