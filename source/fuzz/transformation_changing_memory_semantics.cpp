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

  assert(IsNeededOpcodeWithAppropriateIndex(
             static_cast<SpvOp>(atomic_instruction.target_instruction_opcode()),
             memory_semantics_operand_index) &&
         "The instruction may not be an atomic instruction, or an operand "
         "index does not equal 0 or 1.");

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
  // Instruction must be atomic instruction only. Memory semantics operand Index
  // must be equal to 0 or 1 only. Operand index must be equal to 0 in the case
  // of instruction that takes one memory semantics operand and 0 or 1 for
  // instructions that takes two operands.
  if (!IsNeededOpcodeWithAppropriateIndex(
          needed_atomic_instruction->opcode(),
          message_.memory_semantics_operand_index())) {
    return false;
  }

  // The new value instruction must exist and must be OpConstant.
  auto value_instruction = ir_context->get_def_use_mgr()->GetDef(
      message_.memory_semantics_new_value_id());
  if (!value_instruction) {
    return false;
  }
  if (value_instruction->opcode() != SpvOpConstant) {
    return false;
  }

  // The first 5 bits of new memory semantics values must be suitable for needed
  // instruction. The first 5 bits of new value must be larger than the first 5
  // bits of old. Can't use Sequentially Consistent memory semantic if the
  // memory model is Vulkan.
  auto new_memory_sematics_value = value_instruction->GetSingleWordInOperand(0);
  uint32_t old_memory_sematics_value = 0;
  if (needed_atomic_instruction->opcode() == SpvOpMemoryBarrier) {
    // Memory semantics true index for the OpMemoryBarrier equal 1.
    old_memory_sematics_value =
        ir_context->get_def_use_mgr()
            ->GetDef(needed_atomic_instruction->GetSingleWordInOperand(1))
            ->GetSingleWordInOperand(0);
  } else {
    old_memory_sematics_value =
        ir_context->get_def_use_mgr()
            ->GetDef(needed_atomic_instruction->GetSingleWordInOperand(
                message_.memory_semantics_operand_index() == 0 ? 2 : 3))
            ->GetSingleWordInOperand(0);
  }
  auto first_5bits_new_memory_semantics =
      static_cast<SpvMemorySemanticsMask>(new_memory_sematics_value & 0x1F);
  auto first_5bits_old_memory_semantics =
      static_cast<SpvMemorySemanticsMask>(old_memory_sematics_value & 0x1F);
  auto memory_model = static_cast<SpvMemoryModel>(
      ir_context->module()->GetMemoryModel()->GetSingleWordInOperand(1));

  if (!IsValidConverstion(needed_atomic_instruction->opcode(),
                          first_5bits_old_memory_semantics,
                          first_5bits_new_memory_semantics, memory_model)) {
    return false;
  }

  // The higher bits value of old and new memory semantics id must be equal.
  auto higher_bits_new_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      new_memory_sematics_value & 0xFFFFFFE0);
  auto higher_bits_old_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      old_memory_sematics_value & 0xFFFFFFE0);
  if (higher_bits_new_memory_semantics != higher_bits_old_memory_semantics) {
    return false;
  }

  // Instructions that takes two memory semantics, one of the value must be
  // stronger than the other.
  if (needed_atomic_instruction->opcode() == SpvOpAtomicCompareExchange ||
      needed_atomic_instruction->opcode() == SpvOpAtomicCompareExchangeWeak) {
    auto other_memory_semantics_index = static_cast<SpvMemorySemanticsMask>(
        ir_context->get_def_use_mgr()
            ->GetDef(needed_atomic_instruction->GetSingleWordInOperand(
                message_.memory_semantics_operand_index() == 1 ? 2 : 3))
            ->GetSingleWordInOperand(0) &
        0x1F);

    if (first_5bits_new_memory_semantics == other_memory_semantics_index) {
      return false;
    }
  }

  return true;
}

void TransformationChangingMemorySemantics::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto needed_atomic_instruction =
      FindInstruction(message_.atomic_instruction(), ir_context);

  uint32_t needed_index = 0;
  if (needed_atomic_instruction->opcode() == SpvOpMemoryBarrier) {
    needed_index = 1;
  } else {
    needed_index = message_.memory_semantics_operand_index() == 0 ? 2 : 3;
  }

  uint32_t higher_bits_memory_semantics =
      needed_atomic_instruction->GetSingleWordInOperand(needed_index) &
      0xFFFFFFE0;

  auto new_memory_semantics_instruction = ir_context->get_def_use_mgr()->GetDef(
      message_.memory_semantics_new_value_id());
  // New memory semantics value is OR-ed with the second 27bits of old memory
  // semantics value.
  uint32_t total_value =
      new_memory_semantics_instruction->GetSingleWordInOperand(0) |
      higher_bits_memory_semantics;
  new_memory_semantics_instruction->SetInOperand(0, {total_value});

  needed_atomic_instruction->SetInOperand(
      needed_index, {message_.memory_semantics_new_value_id()});
}

bool TransformationChangingMemorySemantics::IsNeededOpcodeWithAppropriateIndex(
    SpvOp opcode, uint32_t operand_index) const {
  switch (opcode) {
    // Atomic Instructions
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
    // Barrier Instructions
    case SpvOpControlBarrier:
    case SpvOpMemoryBarrier:
    case SpvOpMemoryNamedBarrier:
      if (operand_index != 0) {
        return false;
      }
      break;

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
      if (operand_index != 0 && operand_index != 1) {
        return false;
      }
      break;

    default:
      return false;
  }
  return true;
}

bool TransformationChangingMemorySemantics::IsValidConverstion(
    SpvOp opcode, SpvMemorySemanticsMask first_5bits_old_memory_semantics,
    SpvMemorySemanticsMask first_5bits_new_memory_semantics,
    SpvMemoryModel memory_model) {
  std::set<SpvMemorySemanticsMask> atomic_load_set{
      SpvMemorySemanticsMaskNone, SpvMemorySemanticsAcquireMask,
      SpvMemorySemanticsSequentiallyConsistentMask};

  std::set<SpvMemorySemanticsMask> atomic_store_set{
      SpvMemorySemanticsMaskNone, SpvMemorySemanticsReleaseMask,
      SpvMemorySemanticsSequentiallyConsistentMask};

  std::set<SpvMemorySemanticsMask> atomic_rmw_set{
      SpvMemorySemanticsMaskNone, SpvMemorySemanticsAcquireReleaseMask,
      SpvMemorySemanticsSequentiallyConsistentMask};

  std::set<SpvMemorySemanticsMask> barrier_set{
      SpvMemorySemanticsMaskNone, SpvMemorySemanticsAcquireMask,
      SpvMemorySemanticsReleaseMask, SpvMemorySemanticsAcquireReleaseMask,
      SpvMemorySemanticsSequentiallyConsistentMask};

  if (first_5bits_new_memory_semantics ==
          SpvMemorySemanticsSequentiallyConsistentMask &&
      memory_model == SpvMemoryModelVulkan) {
    return false;
  }
  if (first_5bits_old_memory_semantics >= first_5bits_new_memory_semantics) {
    return false;
  }
  switch (opcode) {
    case SpvOpAtomicLoad:
      return (atomic_load_set.find(first_5bits_old_memory_semantics) !=
                  atomic_load_set.end() &&
              atomic_load_set.find(first_5bits_new_memory_semantics) !=
                  atomic_load_set.end());

    case SpvOpAtomicStore:
      return (atomic_store_set.find(first_5bits_old_memory_semantics) !=
                  atomic_store_set.end() &&
              atomic_store_set.find(first_5bits_new_memory_semantics) !=
                  atomic_store_set.end());

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
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:

      return (atomic_rmw_set.find(first_5bits_old_memory_semantics) !=
                  atomic_rmw_set.end() &&
              atomic_rmw_set.find(first_5bits_new_memory_semantics) !=
                  atomic_rmw_set.end());

    case SpvOpControlBarrier:
    case SpvOpMemoryBarrier:
    case SpvOpMemoryNamedBarrier:

      if (first_5bits_old_memory_semantics == SpvMemorySemanticsAcquireMask &&
          first_5bits_new_memory_semantics == SpvMemorySemanticsReleaseMask) {
        return false;
      }
      return (barrier_set.find(first_5bits_old_memory_semantics) !=
                  barrier_set.end() &&
              barrier_set.find(first_5bits_new_memory_semantics) !=
                  barrier_set.end());

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
