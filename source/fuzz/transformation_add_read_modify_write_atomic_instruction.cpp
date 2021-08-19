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

#include "source/fuzz/transformation_add_read_modify_write_atomic_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddReadModifyWriteAtomicInstruction::
    TransformationAddReadModifyWriteAtomicInstruction(
        protobufs::TransformationAddReadModifyWriteAtomicInstruction message)
    : message_(std::move(message)) {}

TransformationAddReadModifyWriteAtomicInstruction::
    TransformationAddReadModifyWriteAtomicInstruction(
        uint32_t fresh_id, uint32_t pointer_id, uint32_t opcode,
        uint32_t memory_scope_id, uint32_t memory_semantics_id_1,
        uint32_t memory_semantics_id_2, uint32_t value_id,
        uint32_t comparator_id,
        const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  message_.set_opcode(opcode);
  message_.set_memory_scope_id(memory_scope_id);
  message_.set_memory_semantics_id_1(memory_semantics_id_1);
  message_.set_memory_semantics_id_2(memory_semantics_id_2);
  message_.set_value_id(value_id);
  message_.set_comparator_id(comparator_id);

  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
}

bool TransformationAddReadModifyWriteAtomicInstruction::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // The pointer must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }

  // Read-Modify-Write atomic instructions are valid only.
  if (message_.opcode() == SpvOpAtomicLoad ||
      message_.opcode() == SpvOpAtomicStore) {
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

  // Determine which instruction we should be inserting before.
  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  // It must exist, ...
  if (!insert_before) {
    return false;
  }

  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          static_cast<SpvOp>(message_.opcode()), insert_before)) {
    return false;
  }

  // The block we are inserting into needs to be dead, or else the pointee type
  // of the pointer needs to be irrelevant.
  if (!transformation_context.GetFactManager()->BlockIsDead(
          ir_context->get_instr_block(insert_before)->id()) &&
      !transformation_context.GetFactManager()->PointeeValueIsIrrelevant(
          message_.pointer_id())) {
    return false;
  }

  // Instruction must RMW be atomic instruction, Id must be suitable for
  // instruction, Check the validity of (Value, second memory semantics id,
  // comparator id) for instruction if found.
  switch (message_.opcode()) {
    case SpvOpAtomicExchange:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomicUMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomicUMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
    case SpvOpAtomicFAddEXT:

      if (message_.value_id() == 0 || message_.memory_semantics_id_2() != 0 ||
          message_.comparator_id() != 0) {
        return false;
      }
      if (!IsValueIdValid(ir_context, insert_before)) {
        return false;
      }
      break;

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:

      if (message_.value_id() == 0 || message_.memory_semantics_id_2() == 0 ||
          message_.comparator_id() == 0) {
        return false;
      }
      if (!IsValueIdValid(ir_context, insert_before)) {
        return false;
      }
      if (!IsMemorySemanticsId2Valid(ir_context, insert_before, pointer_type)) {
        return false;
      }
      if (!IsComparatorIdValid(ir_context, insert_before)) {
        return false;
      }
      break;

    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicFlagTestAndSet:
    case SpvOpAtomicFlagClear:

      if (message_.value_id() != 0 || message_.memory_semantics_id_2() != 0 ||
          message_.comparator_id() != 0) {
        return false;
      }
      break;

    default:
      break;
  }

  // Check the validity of memory scope.
  if (!IsMemoryScopeValid(ir_context, insert_before)) {
    return false;
  }

  // Check the validity of main(first) memory semantics operand.
  if (!IsMemorySemanticsId1Valid(ir_context, insert_before, pointer_type)) {
    return false;
  }

  // The pointer needs to be available at the insertion point.
  return fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                    message_.pointer_id());
}

void TransformationAddReadModifyWriteAtomicInstruction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto opcode = static_cast<SpvOp>(message_.opcode());

  auto new_instruction = GetInstruction(ir_context, opcode);

  auto insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  auto new_instruction_ptr = new_instruction.get();
  insert_before->InsertBefore(std::move(new_instruction));
  // Inform the def-use manager about the new instruction and record its basic
  // block.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(new_instruction_ptr);
  ir_context->set_instr_block(new_instruction_ptr,
                              ir_context->get_instr_block(insert_before));
}

bool TransformationAddReadModifyWriteAtomicInstruction::IsMemoryScopeValid(
    opt::IRContext* ir_context,
    spvtools::opt::Instruction* insert_before) const {
  // Check the exists of memory scope and memory semantics ids.
  auto memory_scope_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_scope_id());

  if (!memory_scope_instruction) {
    return false;
  }

  // The memory scope instruction must have the
  // 'OpConstant' opcode.
  if (memory_scope_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.memory_scope_id())) {
    return false;
  }

  // The memory scope instruction must be an integer
  // operand type with signedness does not matters.
  if (ir_context->get_def_use_mgr()
          ->GetDef(memory_scope_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  // The size of the integer for memory scope
  // instruction must be equal to 32 bits.
  auto memory_scope_int_width =
      ir_context->get_def_use_mgr()
          ->GetDef(memory_scope_instruction->type_id())
          ->GetSingleWordInOperand(0);

  if (memory_scope_int_width != 32) {
    return false;
  }

  // The memory scope constant value must be that of SpvScopeInvocation.
  auto memory_scope_const_value =
      memory_scope_instruction->GetSingleWordInOperand(0);
  if (memory_scope_const_value != SpvScopeInvocation) {
    return false;
  }
  return true;
}

bool TransformationAddReadModifyWriteAtomicInstruction::
    IsMemorySemanticsId1Valid(opt::IRContext* ir_context,
                              spvtools::opt::Instruction* insert_before,
                              spvtools::opt::Instruction* pointer_type) const {
  auto memory_semantics_instruction_1 =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_semantics_id_1());
  if (!memory_semantics_instruction_1) {
    return false;
  }
  if (memory_semantics_instruction_1->opcode() != SpvOpConstant) {
    return false;
  }
  // The memory semantics need to be available before |insert_before|.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, insert_before, message_.memory_semantics_id_1())) {
    return false;
  }

  if (ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction_1->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto memory_semantics_1_int_width =
      ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction_1->type_id())
          ->GetSingleWordInOperand(0);

  if (memory_semantics_1_int_width != 32) {
    return false;
  }

  // The memory semantics constant higher bits value must match the storage
  // class of the pointer being loaded from.
  auto memory_semantics_1_const_value = static_cast<SpvMemorySemanticsMask>(
      memory_semantics_instruction_1->GetSingleWordInOperand(0));

  auto memory_semantics_from_storage_class =
      fuzzerutil::GetMemorySemanticsForStorageClass(
          static_cast<SpvStorageClass>(
              pointer_type->GetSingleWordInOperand(0)));

  auto higher_bits_from_current_memory_semantics_1 =
      (memory_semantics_1_const_value & kMemorySemanticsHigherBitmask);
  if (higher_bits_from_current_memory_semantics_1 !=
      memory_semantics_from_storage_class) {
    return false;
  }

  // The memory semantics constant lower bits value equal to available masks for
  // the atomic(read-modify-write) instruction.
  auto lower_bits_from_current_memory_semantics_1 =
      static_cast<SpvMemorySemanticsMask>(memory_semantics_1_const_value &
                                          kMemorySemanticsLowerBitmask);

  if (lower_bits_from_current_memory_semantics_1 !=
          SpvMemorySemanticsMaskNone &&
      lower_bits_from_current_memory_semantics_1 !=
          SpvMemorySemanticsAcquireReleaseMask &&
      lower_bits_from_current_memory_semantics_1 !=
          SpvMemorySemanticsSequentiallyConsistentMask) {
    return false;
  }

  return true;
}

bool TransformationAddReadModifyWriteAtomicInstruction::
    IsMemorySemanticsId2Valid(opt::IRContext* ir_context,
                              spvtools::opt::Instruction* insert_before,
                              spvtools::opt::Instruction* pointer_type) const {
  auto memory_semantics_instruction_2 =
      ir_context->get_def_use_mgr()->GetDef(message_.memory_semantics_id_2());
  if (!memory_semantics_instruction_2) {
    return false;
  }
  if (memory_semantics_instruction_2->opcode() != SpvOpConstant) {
    return false;
  }
  // The memory semantics second operand need to be available before
  // |insert_before|.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, insert_before, message_.memory_semantics_id_2())) {
    return false;
  }

  if (ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction_2->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto memory_semantics_2_int_width =
      ir_context->get_def_use_mgr()
          ->GetDef(memory_semantics_instruction_2->type_id())
          ->GetSingleWordInOperand(0);

  if (memory_semantics_2_int_width != 32) {
    return false;
  }

  // The memory semantics second operand constant higher bits value must match
  // the storage class of the pointer being loaded from.
  auto memory_semantics_2_const_value = static_cast<SpvMemorySemanticsMask>(
      memory_semantics_instruction_2->GetSingleWordInOperand(0));

  auto memory_semantics_from_storage_class =
      fuzzerutil::GetMemorySemanticsForStorageClass(
          static_cast<SpvStorageClass>(
              pointer_type->GetSingleWordInOperand(0)));

  auto higher_bits_from_current_memory_semantics_2 =
      (memory_semantics_2_const_value & kMemorySemanticsHigherBitmask);
  if (higher_bits_from_current_memory_semantics_2 !=
      memory_semantics_from_storage_class) {
    return false;
  }

  // The memory semantics second operand constant lower bits value equal to
  // available masks for the atomic(read-modify-write) instruction.
  auto lower_bits_from_current_memory_semantics_2 =
      static_cast<SpvMemorySemanticsMask>(memory_semantics_2_const_value &
                                          kMemorySemanticsLowerBitmask);

  if (lower_bits_from_current_memory_semantics_2 !=
          SpvMemorySemanticsMaskNone &&
      lower_bits_from_current_memory_semantics_2 !=
          SpvMemorySemanticsAcquireReleaseMask &&
      lower_bits_from_current_memory_semantics_2 !=
          SpvMemorySemanticsSequentiallyConsistentMask) {
    return false;
  }

  return true;
}

bool TransformationAddReadModifyWriteAtomicInstruction::IsValueIdValid(
    opt::IRContext* ir_context,
    spvtools::opt::Instruction* insert_before) const {
  auto value_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.value_id());
  if (!value_instruction) {
    return false;
  }
  if (value_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.value_id())) {
    return false;
  }

  if (ir_context->get_def_use_mgr()
          ->GetDef(value_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto value_int_width = ir_context->get_def_use_mgr()
                             ->GetDef(value_instruction->type_id())
                             ->GetSingleWordInOperand(0);

  if (value_int_width != 32) {
    return false;
  }

  return true;
}

bool TransformationAddReadModifyWriteAtomicInstruction::IsComparatorIdValid(
    opt::IRContext* ir_context,
    spvtools::opt::Instruction* insert_before) const {
  auto comparator_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.comparator_id());
  if (!comparator_instruction) {
    return false;
  }
  if (comparator_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  if (!fuzzerutil::IdIsAvailableBeforeInstruction(ir_context, insert_before,
                                                  message_.comparator_id())) {
    return false;
  }

  if (ir_context->get_def_use_mgr()
          ->GetDef(comparator_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  auto comparator_int_width = ir_context->get_def_use_mgr()
                                  ->GetDef(comparator_instruction->type_id())
                                  ->GetSingleWordInOperand(0);

  if (comparator_int_width != 32) {
    return false;
  }

  return true;
}

std::unique_ptr<spvtools::opt::Instruction>
TransformationAddReadModifyWriteAtomicInstruction::GetInstruction(
    opt::IRContext* ir_context, SpvOp opcode) const {
  uint32_t result_type = fuzzerutil::GetPointeeTypeIdFromPointerType(
      ir_context, fuzzerutil::GetTypeId(ir_context, message_.pointer_id()));
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  std::unique_ptr<spvtools::opt::Instruction> new_instruction;
  switch (opcode) {
    case SpvOpAtomicExchange:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomicUMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomicUMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
    case SpvOpAtomicFAddEXT:

      return MakeUnique<opt::Instruction>(
          ir_context, opcode, result_type, message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
               {SPV_OPERAND_TYPE_SCOPE_ID, {message_.memory_scope_id()}},
               {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
                {message_.memory_semantics_id_1()}},
               {SPV_OPERAND_TYPE_ID, {message_.value_id()}}}));

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:

      return MakeUnique<opt::Instruction>(
          ir_context, opcode, result_type, message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
               {SPV_OPERAND_TYPE_SCOPE_ID, {message_.memory_scope_id()}},
               {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
                {message_.memory_semantics_id_1()}},
               {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
                {message_.memory_semantics_id_2()}},
               {SPV_OPERAND_TYPE_ID, {message_.value_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.comparator_id()}}}));

    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicFlagTestAndSet:
    case SpvOpAtomicFlagClear:

      return MakeUnique<opt::Instruction>(
          ir_context, opcode, result_type, message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.pointer_id()}},
               {SPV_OPERAND_TYPE_SCOPE_ID, {message_.memory_scope_id()}},
               {SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
                {message_.memory_semantics_id_1()}}}));

    default:
      assert(false);
      return new_instruction;
  }
}

protobufs::Transformation
TransformationAddReadModifyWriteAtomicInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_read_modify_write_atomic_instruction() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationAddReadModifyWriteAtomicInstruction::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
