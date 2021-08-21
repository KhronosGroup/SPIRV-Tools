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

#include "source/fuzz/transformation_changing_memory_scope.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationChangingMemoryScope::TransformationChangingMemoryScope(
    protobufs::TransformationChangingMemoryScope message)
    : message_(std::move(message)) {}

TransformationChangingMemoryScope::TransformationChangingMemoryScope(
    const protobufs::InstructionDescriptor& needed_instruction,
    uint32_t memory_scope_new_value_id) {
  *message_.mutable_needed_instruction() = needed_instruction;
  message_.set_memory_scope_new_value_id(memory_scope_new_value_id);
}

bool TransformationChangingMemoryScope::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The instruction must be exist.
  auto needed_instruction =
      FindInstruction(message_.needed_instruction(), ir_context);
  if (!needed_instruction) {
    return false;
  }

  // Instruction must be atomic instruction only.
  if (!IsAtomicInstruction(needed_instruction->opcode())) {
    return false;
  }

  // The new value instruction must exist and must be OpConstant.
  auto value_instruction = ir_context->get_def_use_mgr()->GetDef(
      message_.memory_scope_new_value_id());
  if (!value_instruction) {
    return false;
  }
  if (value_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  // The value instruction must be an Integer.
  if (ir_context->get_def_use_mgr()
          ->GetDef(value_instruction->type_id())
          ->opcode() != SpvOpTypeInt) {
    return false;
  }

  // The size of the integer for value instruction must be equal to 32 bits.
  auto value_instruction_int_width = ir_context->get_def_use_mgr()
                                         ->GetDef(value_instruction->type_id())
                                         ->GetSingleWordInOperand(0);

  if (value_instruction_int_width != 32) {
    return false;
  }

  // The new value of memory scope must available be wider than the older.
  auto memory_scope_in_operand =
      GetMemoryScopeInOperandIndex(needed_instruction->opcode());
  auto new_memory_scope_value =
      static_cast<SpvScope>(value_instruction->GetSingleWordInOperand(0));
  auto old_memory_scope_value = static_cast<SpvScope>(
      ir_context->get_def_use_mgr()
          ->GetDef(needed_instruction->GetSingleWordInOperand(
              memory_scope_in_operand))
          ->GetSingleWordInOperand(0));

  if (!IsValidScope(new_memory_scope_value, old_memory_scope_value)) {
    return false;
  }

  return true;
}

void TransformationChangingMemoryScope::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto needed_instruction =
      FindInstruction(message_.needed_instruction(), ir_context);

  uint32_t needed_index =
      GetMemoryScopeInOperandIndex(needed_instruction->opcode());

  needed_instruction->SetInOperand(needed_index,
                                   {message_.memory_scope_new_value_id()});
}

uint32_t TransformationChangingMemoryScope::GetMemoryScopeInOperandIndex(
    SpvOp opcode) {
  switch (opcode) {
    case SpvOpAtomicLoad:
    case SpvOpAtomicStore:
    case SpvOpAtomicExchange:
    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomicUMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomicUMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
    case SpvOpAtomicFlagTestAndSet:
    case SpvOpAtomicFlagClear:
    case SpvOpAtomicFAddEXT:
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
      return 1;

    default:
      assert(false);
      return -1;
  }
}

bool TransformationChangingMemoryScope::IsAtomicInstruction(SpvOp opcode) {
  switch (opcode) {
    // Atomic Instructions.
    case SpvOpAtomicLoad:
    case SpvOpAtomicStore:
    case SpvOpAtomicExchange:
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomicUMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomicUMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
    case SpvOpAtomicFlagTestAndSet:
    case SpvOpAtomicFlagClear:
    case SpvOpAtomicFAddEXT:
      return true;

    default:
      return false;
  }
}

bool TransformationChangingMemoryScope::IsValidScope(
    SpvScope new_memory_scope_value, SpvScope old_memory_scope_value) {
  switch (new_memory_scope_value) {
    case SpvScopeCrossDevice:
    case SpvScopeDevice:
    case SpvScopeWorkgroup:
    case SpvScopeSubgroup:
    case SpvScopeInvocation:
      if (new_memory_scope_value > old_memory_scope_value) {
        return false;
      }
      return true;

    default:
      return false;
  }
}

protobufs::Transformation TransformationChangingMemoryScope::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_changing_memory_scope() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationChangingMemoryScope::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
