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

#include "source/fuzz/transformation_changing_memory_semantics.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationChangingMemorySemantics::TransformationChangingMemorySemantics(
    protobufs::TransformationChangingMemorySemantics message)
    : message_(std::move(message)) {}

TransformationChangingMemorySemantics::TransformationChangingMemorySemantics(
    const protobufs::InstructionDescriptor& atomic_instruction,
    uint32_t memory_semantics_operand_index,
    uint32_t memory_semantics_new_value_id) {
  *message_.mutable_atomic_instruction() = atomic_instruction;
  message_.set_memory_semantics_operand_index(memory_semantics_operand_index);
  message_.set_memory_semantics_new_value_id(memory_semantics_new_value_id);
}

bool TransformationChangingMemorySemantics::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto needed_atomic_instruction =
      FindInstruction(message_.atomic_instruction(), ir_context);
  // The atomic instruction must exist.
  if (!needed_atomic_instruction) {
    return false;
  }

  // Memory semantics operand Index must be equal to 2 or 3 only.
  auto operand_index = message_.memory_semantics_operand_index();
  if (operand_index != 2 && operand_index != 3) {
    return false;
  }

  // The new instruction value must exist and must be OpConstant.
  auto new_instruction = ir_context->get_def_use_mgr()->GetDef(
      message_.memory_semantics_new_value_id());
  if (!new_instruction) {
    return false;
  }
  if (new_instruction->opcode() != SpvOpConstant) {
    return false;
  }
  auto new_memory_sematics_value = static_cast<SpvMemorySemanticsMask>(
      new_instruction->GetSingleWordInOperand(0));

  // Instruction must be atomic instruction only. Operand index must be equal to
  // 2 in the case of instruction that takes one memory semantics operand.
  // Instructions that takes two memory semantics, one of the value must be
  // stronger than the other.
  switch (needed_atomic_instruction->opcode()) {
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
      if (operand_index != 2) {
        return false;
      }
      break;

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
      if (new_memory_sematics_value ==
          static_cast<SpvMemorySemanticsMask>(
              ir_context->get_def_use_mgr()
                  ->GetDef(needed_atomic_instruction->GetSingleWordInOperand(
                      operand_index == 3 ? 2 : 3))
                  ->GetSingleWordInOperand(0))) {
        return false;
      }
      break;

    default:
      return false;
  }

  // The new memory semantics value must be larger than old.
  auto old_memory_sematics_value = static_cast<SpvMemorySemanticsMask>(
      ir_context->get_def_use_mgr()
          ->GetDef(
              needed_atomic_instruction->GetSingleWordInOperand(operand_index))
          ->GetSingleWordInOperand(0));
  if (new_memory_sematics_value <= old_memory_sematics_value) {
    return false;
  }

  if (!IsValidConverstion(needed_atomic_instruction->opcode(),
                          new_memory_sematics_value)) {
    return false;
  }

  return true;
}

void TransformationChangingMemorySemantics::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto needed_atomic_instruction =
      FindInstruction(message_.atomic_instruction(), ir_context);
  needed_atomic_instruction->SetInOperand(
      message_.memory_semantics_operand_index(),
      {message_.memory_semantics_new_value_id()});
}

bool TransformationChangingMemorySemantics::IsValidConverstion(
    SpvOp opcode, SpvMemorySemanticsMask new_memory_sematics_value) {
  switch (opcode) {
    case SpvOpAtomicLoad:
      return (new_memory_sematics_value == SpvMemorySemanticsMaskNone ||
              new_memory_sematics_value == SpvMemorySemanticsAcquireMask ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsSequentiallyConsistentMask);

    case SpvOpAtomicStore:
      return (new_memory_sematics_value == SpvMemorySemanticsMaskNone ||
              new_memory_sematics_value == SpvMemorySemanticsReleaseMask ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsSequentiallyConsistentMask);

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

      return (new_memory_sematics_value == SpvMemorySemanticsMaskNone ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsAcquireReleaseMask ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsSequentiallyConsistentMask);

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:

      return (new_memory_sematics_value == SpvMemorySemanticsMaskNone ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsAcquireReleaseMask ||
              new_memory_sematics_value ==
                  SpvMemorySemanticsSequentiallyConsistentMask);

    default:
      return false;
  }
}

protobufs::Transformation TransformationChangingMemorySemantics::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_changing_memory_semantics() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationChangingMemorySemantics::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools