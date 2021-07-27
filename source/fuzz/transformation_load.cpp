// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_load.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationLoad::TransformationLoad(protobufs::TransformationLoad message)
    : message_(std::move(message)) {}

TransformationLoad::TransformationLoad(
    uint32_t fresh_id, uint32_t pointer_id, bool is_atomic,
    uint32_t memory_scope, uint32_t memory_semantics,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  assert(!is_atomic && "Atomic load not fully developed yet.");
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  message_.set_is_atomic(is_atomic);
  message_.set_memory_scope_id(memory_scope);
  message_.set_memory_semantics_id(memory_semantics);

  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationLoad::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // The pointer must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }
  // The type must indeed be a pointer type.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(pointer->type_id());
  assert(pointer_type && "Type id must be defined.");
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return false;
  }
  // We do not want to allow loading from null or undefined pointers, as it is
  // not clear how punishing the consequences of doing so are from a semantics
  // point of view.
  switch (pointer->opcode()) {
    case SpvOpConstantNull:
    case SpvOpUndef:
      return false;
    default:
      break;
  }

  // Check the exists of memory scope and memory semantics ids.
  auto memory_scope_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_scope_id());
  auto memory_semantics_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_semantics_id());

  if (message_.is_atomic() && !memory_scope_instruction &&
      !memory_semantics_instruction) {
    return false;
  }
  // The memory scope and memory semantics instructions must have the
  // 'OpConstant' opcode.
  if (memory_scope_instruction->opcode() != SpvOpConstant &&
      memory_semantics_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  // The memory scope and memory semantics instructions must have an Integer
  // operand type with signedness matters.
  if (!ir_context->get_type_mgr()
           ->GetType(memory_scope_instruction->type_id())
           ->AsInteger() &&
      !ir_context->get_type_mgr()
           ->GetType(memory_semantics_instruction->type_id())
           ->AsInteger()) {
    return false;
  }

  // The memory scope constant value must be that of SpvScopeInvocation.
  auto memory_scope_const_value = memory_scope_instruction->GetInOperand(0);
  if (memory_scope_const_value.words[0] != SpvScopeInvocation) {
    return false;
  }

  // The memory semantics constant value must be either
  // SpvMemorySemanticsWorkgroupMemoryMask or
  // SpvMemorySemanticsUniformMemoryMask.
  auto memory_semantics_const_value =
      memory_semantics_instruction->GetInOperand(0);
  switch (memory_semantics_const_value.words[0]) {
    case SpvMemorySemanticsWorkgroupMemoryMask:
    case SpvMemorySemanticsUniformMemoryMask:
      return false;
    default:
      break;
  }

  // Determine which instruction we should be inserting before.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  // It must exist, ...
  if (!insert_before) {
    return false;
  }
  // ... and it must be legitimate to insert a store before it.
  if (!message_.is_atomic() &&
      !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, insert_before)) {
    return false;
  }

  if (message_.is_atomic() && !fuzzerutil::CanInsertOpcodeBeforeInstruction(
                                  SpvOpAtomicLoad, insert_before)) {
    return false;
  }

  // The pointer needs to be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    message_.pointer_id());
}

void TransformationLoad::Apply(opt::IRContext* ir_context,
                               TransformationContext* /*unused*/) const {
  if (message_.is_atomic()) {
    // OpAtomicLoad instruction.
    uint32_t result_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
        ir_context, fuzzerutil::GetTypeId(ir_context, message_.pointer_id()));
    fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
    auto insert_before =
        FindInstruction(message_.instruction_to_insert_before(), ir_context);
    auto new_instruction = MakeUnique<opt::Instruction>(
        ir_context, SpvOpAtomicLoad, result_type, message_.fresh_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID,
              {message_.pointer_id(), message_.memory_scope_id(),
               message_.memory_semantics_id()}}}));
    auto new_instruction_ptr = new_instruction.get();
    insert_before->InsertBefore(std::move(new_instruction));
    // Inform the def-use manager about the new instruction and record its basic
    // block.
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr,
                                ir_context->get_instr_block(insert_before));
  } else {
    // OpLoad instruction.
    uint32_t result_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
        ir_context, fuzzerutil::GetTypeId(ir_context, message_.pointer_id()));
    fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
    auto insert_before =
        FindInstruction(message_.instruction_to_insert_before(), ir_context);
    auto new_instruction = MakeUnique<opt::Instruction>(
        ir_context, SpvOpLoad, result_type, message_.fresh_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}}}));
    auto new_instruction_ptr = new_instruction.get();
    insert_before->InsertBefore(std::move(new_instruction));
    // Inform the def-use manager about the new instruction and record its basic
    // block.
    ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
    ir_context->set_instr_block(new_instruction_ptr,
                                ir_context->get_instr_block(insert_before));
  }
}

protobufs::Transformation TransformationLoad::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_load() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationLoad::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
