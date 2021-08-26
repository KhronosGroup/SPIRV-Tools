// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/transformation_add_memory_barrier.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddMemoryBarrier::TransformationAddMemoryBarrier(
    protobufs::TransformationAddMemoryBarrier message)
    : message_(std::move(message)) {}

TransformationAddMemoryBarrier::TransformationAddMemoryBarrier(
    uint32_t memory_scope_id, uint32_t memory_semantics_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_memory_scope_id(memory_scope_id);
  message_.set_memory_semantics_id(memory_semantics_id);
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationAddMemoryBarrier::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The Instruction would like to insert before must be existing.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);

  if (!insert_before) {
    return false;
  }

  // Must be legitimate to insert |SpvOpMemoryBarrier| before this instruction.
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpMemoryBarrier,
                                                    insert_before)) {
    return false;
  }

  // Memory scope id must exist, valid, and scope value is Invocation.
  if (!IsMemoryScopeIdValid(ir_context, insert_before)) {
    return false;
  }

  // Memory scope id must exist, be valid, and has suitable value for higher and
  // lower bits.
  if (!IsMemorySemancticsIdValid(ir_context, insert_before)) {
    return false;
  }
  return true;
}

void TransformationAddMemoryBarrier::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  auto new_instruction = MakeUnique<opt::Instruction>(
      ir_context, SpvOpMemoryBarrier, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_SCOPE_ID, {message_.memory_scope_id()}},
           {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
            {message_.memory_semantics_id()}}}));

  auto new_instruction_ptr = new_instruction.get();
  insert_before->InsertBefore(std::move(new_instruction));

  // Inform the def-use manager about the new instruction and record its basic
  // block.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr,
                              ir_context->get_instr_block(insert_before));
}

bool TransformationAddMemoryBarrier::IsMemoryScopeIdValid(
    opt::IRContext* ir_context,
    spvtools::opt::Instruction* insert_before) const {
  // The memory scope instruction must exist and must be OpConstant.
  auto memory_scope_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_scope_id());
  if (!memory_scope_instruction) {
    return false;
  }

  if (memory_scope_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  // The memory scope need to be available before |insert_before|.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.memory_scope_id())) {
    return false;
  }

  // The memory scope instruction must have an Integer operand with a 32
  // bits width.
  if (ir_context->get_def_use_mgr()
          ->GetDef(memory_scope_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto memory_scope_int_width =
      ir_context->get_def_use_mgr()
          ->GetDef(memory_scope_instruction->type_id())
          ->GetSingleWordInOperand(0);

  if (memory_scope_int_width != 32) {
    return false;
  }

  // The memory scope constant value must be SpvScopeInvocation.
  auto memory_scope_const_value =
      memory_scope_instruction->GetSingleWordInOperand(0);
  if (memory_scope_const_value != SpvScopeInvocation) {
    return false;
  }
  return true;
}

bool TransformationAddMemoryBarrier::IsMemorySemancticsIdValid(
    opt::IRContext* ir_context,
    spvtools::opt::Instruction* insert_before) const {
  // The memory semantics instruction must exist and must be OpConstant.
  auto memory_semantics_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_semantics_id());
  if (!memory_semantics_instruction) {
    return false;
  }

  if (memory_semantics_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  // The memory semantics need to be available before |insert_before|.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, insert_before, message_.memory_semantics_id())) {
    return false;
  }

  // The memory semantics instruction must have an Integer operand with a 32
  // bits width.
  if (ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto memory_semantics_int_width =
      ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction->type_id())
          ->GetSingleWordInOperand(0);

  if (memory_semantics_int_width != 32) {
    return false;
  }

  auto memory_semantics_value =
      memory_semantics_instruction->GetSingleWordInOperand(0);

  // Memory semantics higher bits must be one of the suitable memory masks.
  auto memory_semantics_higher_bits = static_cast<SpvMemorySemanticsMask>(
      memory_semantics_value & kMemorySemanticsHigherBitmask);
  switch (memory_semantics_higher_bits) {
    case SpvMemorySemanticsUniformMemoryMask:
    case SpvMemorySemanticsWorkgroupMemoryMask:
      break;

    default:
      return false;
  }

  // Memory semantics lower bits must be Relaxed(None).
  auto memory_semantics_lower_bits = static_cast<SpvMemorySemanticsMask>(
      memory_semantics_value & kMemorySemanticsLowerBitmask);

  if (memory_semantics_lower_bits != SpvMemorySemanticsMaskNone) {
    return false;
  }

  return true;
}

protobufs::Transformation TransformationAddMemoryBarrier::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_memory_barrier() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddMemoryBarrier::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
