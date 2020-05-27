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
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_load.h"
#include "source/fuzz/transformation_store.h"

namespace spvtools {
namespace fuzz {

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    const spvtools::fuzz::protobufs::TransformationPushIdThroughVariable&
        message)
    : message_(message) {}

TransformationPushIdThroughVariable::TransformationPushIdThroughVariable(
    uint32_t value_id, uint32_t value_synonym_id, uint32_t variable_id,
    uint32_t pointer_type_id, uint32_t variable_storage_class,
    uint32_t function_id,
    const protobufs::InstructionDescriptor& instruction_descriptor) {
  message_.set_value_id(value_id);
  message_.set_value_synonym_id(value_synonym_id);
  message_.set_variable_id(variable_id);
  message_.set_pointer_type_id(pointer_type_id);
  message_.set_variable_storage_class(variable_storage_class);
  message_.set_function_id(function_id);
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationPushIdThroughVariable::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.value_synonym_id| and |message_.variable_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.value_synonym_id()) ||
      !fuzzerutil::IsFreshId(ir_context, message_.variable_id())) {
    return false;
  }

  // The instruction to insert before must be defined.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (!instruction_to_insert_before) {
    return false;
  }

  // The instruction to insert before must belong to a reachable block.
  auto basic_block = ir_context->get_instr_block(instruction_to_insert_before);
  if (!fuzzerutil::BlockIsReachableInItsFunction(ir_context, basic_block)) {
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

  // The pointer type instruction must be defined and be an OpTypePointer
  // instruction.
  auto pointer_type_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.pointer_type_id());
  if (!pointer_type_instruction ||
      pointer_type_instruction->opcode() != SpvOpTypePointer) {
    return false;
  }

  // The pointer type storage class must be equal to the variable storage class.
  if (pointer_type_instruction->GetSingleWordInOperand(0) !=
      message_.variable_storage_class()) {
    return false;
  }

  // The pointee type must be equal to the value type.
  if (pointer_type_instruction->GetSingleWordInOperand(1) !=
      value_instruction->type_id()) {
    return false;
  }

  // |message_.variable_storage_class| must be private or function.
  if (message_.variable_storage_class() != SpvStorageClassPrivate &&
      message_.variable_storage_class() != SpvStorageClassFunction) {
    return false;
  }

  // The function containing |message_.function_id| must be defined,
  // must be an OpFunction instruction and must be the function where
  // the |message_.instruction_descriptor| is defined.
  auto function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function || function->DefInst().opcode() != SpvOpFunction ||
      function != basic_block->GetParent()) {
    return false;
  }

  // |message_.value_id| must be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(
      ir_context, instruction_to_insert_before, message_.value_id());
}

void TransformationPushIdThroughVariable::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Adds whether a global or local variable.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.variable_id());
  if (message_.variable_storage_class() == SpvStorageClassPrivate) {
    ir_context->module()->AddGlobalValue(MakeUnique<opt::Instruction>(
        ir_context, SpvOpVariable, message_.pointer_type_id(),
        message_.variable_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassPrivate}}})));
  } else {
    fuzzerutil::FindFunction(ir_context, message_.function_id())
        ->begin()
        ->begin()
        ->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, SpvOpVariable, message_.pointer_type_id(),
            message_.variable_id(),
            opt::Instruction::OperandList({{SPV_OPERAND_TYPE_STORAGE_CLASS,
                                            {SpvStorageClassFunction}}})));
  }

  // Stores value id to variable id.
  FindInstruction(message_.instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpStore, 0, 0,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.value_id()}}})));

  // Loads variable id to value synonym id.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.value_synonym_id());
  FindInstruction(message_.instruction_descriptor(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpLoad,
          fuzzerutil::GetPointeeTypeIdFromPointerType(
              ir_context, message_.pointer_type_id()),
          message_.value_synonym_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.variable_id()}}})));

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // Adds the fact that |message_.value_synonym_id|
  // and |message_.value_id| are synonymous.
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
