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

namespace {
const uint32_t kMemorySemanticsHigherBitmask = 0xFFFFFFE0;
const uint32_t kMemorySemanticsLowerBitmask = 0x1F;
}  // namespace

TransformationChangingMemorySemantics::TransformationChangingMemorySemantics(
    protobufs::TransformationChangingMemorySemantics message)
    : message_(std::move(message)) {
  assert(IsNeededOpcodeWithAppropriateIndex(
             static_cast<SpvOp>(
                 message_.atomic_instruction().target_instruction_opcode()),
             message_.memory_semantics_operand_index()) &&
         "The instruction may not be an atomic or barrier instruction. \
                The operands index may be not equal 0 or 1. \
                The index may be equal to one and the expected is zero.");
}

TransformationChangingMemorySemantics::TransformationChangingMemorySemantics(
    const protobufs::InstructionDescriptor& atomic_instruction,
    uint32_t memory_semantics_operand_index,
    uint32_t memory_semantics_new_value_id) {
  *message_.mutable_atomic_instruction() = atomic_instruction;

  assert(IsNeededOpcodeWithAppropriateIndex(
             static_cast<SpvOp>(atomic_instruction.target_instruction_opcode()),
             memory_semantics_operand_index) &&
         "The instruction may not be an atomic or barrier instruction. \
                The operands index may be not equal 0 or 1. \
                The index may be equal to one and the expected is zero.");

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

  // The new value instruction must exist and must be OpConstant.
  auto value_instruction = ir_context->get_def_use_mgr()->GetDef(
      message_.memory_semantics_new_value_id());
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
                GetMemorySemanticsOperandIndex(
                    needed_atomic_instruction->opcode(),
                        message_.memory_semantics_operand_index())))
            ->GetSingleWordInOperand(0);
  }
  auto first_5bits_new_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      new_memory_sematics_value & kMemorySemanticsLowerBitmask);
  auto first_5bits_old_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      old_memory_sematics_value & kMemorySemanticsLowerBitmask);
  auto memory_model = static_cast<SpvMemoryModel>(
      ir_context->module()->GetMemoryModel()->GetSingleWordInOperand(1));

  if (!IsValidConversion(needed_atomic_instruction->opcode(),
                         first_5bits_old_memory_semantics,
                         first_5bits_new_memory_semantics, memory_model)) {
    return false;
  }

  // The higher bits value of old and new memory semantics id must be equal.
  auto higher_bits_new_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      new_memory_sematics_value & kMemorySemanticsHigherBitmask);
  auto higher_bits_old_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      old_memory_sematics_value & kMemorySemanticsHigherBitmask);
  if (higher_bits_new_memory_semantics != higher_bits_old_memory_semantics) {
    return false;
  }

  // Instructions that take two memory semantics and id needed to change are
  // unequal, the equal id must be stronger than unequal id. Unequal id can't be
  // released or acquire/release memory semantics.
  if ((needed_atomic_instruction->opcode() == SpvOpAtomicCompareExchange ||
       needed_atomic_instruction->opcode() == SpvOpAtomicCompareExchangeWeak) &&
      message_.memory_semantics_operand_index() == 1) {
    auto equal_id_memory_semantics_first_5bits =
        static_cast<SpvMemorySemanticsMask>(
            ir_context->get_def_use_mgr()
                ->GetDef(needed_atomic_instruction->GetSingleWordInOperand(2))
                ->GetSingleWordInOperand(0) &
            kMemorySemanticsLowerBitmask);

    if (first_5bits_new_memory_semantics >
        equal_id_memory_semantics_first_5bits) {
      return false;
    }
    if (first_5bits_new_memory_semantics == SpvMemorySemanticsReleaseMask ||
        first_5bits_new_memory_semantics ==
            SpvMemorySemanticsAcquireReleaseMask) {
      return false;
    }
  }

  return true;
}

void TransformationChangingMemorySemantics::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto needed_atomic_instruction =
      FindInstruction(message_.atomic_instruction(), ir_context);

  uint32_t needed_index =
      GetMemorySemanticsOperandIndex(needed_atomic_instruction->opcode(),
                                     message_.memory_semantics_operand_index());

  needed_atomic_instruction->SetInOperand(
      needed_index, {message_.memory_semantics_new_value_id()});
}

uint32_t TransformationChangingMemorySemantics::GetMemorySemanticsOperandIndex(
    SpvOp opcode, uint32_t zero_or_one) {
  switch (opcode) {
    case SpvOpMemoryBarrier:
      assert(zero_or_one == 0 && "Zero_or_one not equal zero.");
      return 1;

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
    case SpvOpControlBarrier:
    case SpvOpMemoryNamedBarrier:
      return zero_or_one == 0 ? 2 : 3;

    default:
      assert(false);
      return -1;
  }
}

bool TransformationChangingMemorySemantics::IsNeededOpcodeWithAppropriateIndex(
    SpvOp opcode, uint32_t operand_index) {
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

bool TransformationChangingMemorySemantics::IsAtomicLoadMemorySemanticsValue(
    SpvMemorySemanticsMask memory_semantics_value) {
  switch (memory_semantics_value) {
    case SpvMemorySemanticsMaskNone:
    case SpvMemorySemanticsAcquireMask:
    case SpvMemorySemanticsSequentiallyConsistentMask:
      return true;

    default:
      return false;
  }
}

bool TransformationChangingMemorySemantics::IsAtomicStoreMemorySemanticsValue(
    SpvMemorySemanticsMask memory_semantics_value) {
  switch (memory_semantics_value) {
    case SpvMemorySemanticsMaskNone:
    case SpvMemorySemanticsReleaseMask:
    case SpvMemorySemanticsSequentiallyConsistentMask:
      return true;

    default:
      return false;
  }
}

bool TransformationChangingMemorySemantics::
    IsAtomicRMWInstructionsemorySemanticsValue(
        SpvMemorySemanticsMask memory_semantics_value) {
  switch (memory_semantics_value) {
    case SpvMemorySemanticsMaskNone:
    case SpvMemorySemanticsAcquireReleaseMask:
    case SpvMemorySemanticsSequentiallyConsistentMask:
      return true;

    default:
      return false;
  }
}

bool TransformationChangingMemorySemantics::
    IsBarrierInstructionsMemorySemanticsValue(
        SpvMemorySemanticsMask memory_semantics_value) {
  switch (memory_semantics_value) {
    case SpvMemorySemanticsMaskNone:
    case SpvMemorySemanticsAcquireMask:
    case SpvMemorySemanticsReleaseMask:
    case SpvMemorySemanticsAcquireReleaseMask:
    case SpvMemorySemanticsSequentiallyConsistentMask:
      return true;

    default:
      return false;
  }
}

bool TransformationChangingMemorySemantics::IsValidConversion(
    SpvOp opcode, SpvMemorySemanticsMask first_5bits_old_memory_semantics,
    SpvMemorySemanticsMask first_5bits_new_memory_semantics,
    SpvMemoryModel memory_model) {
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
      return IsAtomicLoadMemorySemanticsValue(
                 first_5bits_old_memory_semantics) &&
             IsAtomicLoadMemorySemanticsValue(first_5bits_new_memory_semantics);

    case SpvOpAtomicStore:
      return IsAtomicStoreMemorySemanticsValue(
                 first_5bits_old_memory_semantics) &&
             IsAtomicStoreMemorySemanticsValue(
                 first_5bits_new_memory_semantics);

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

      return IsAtomicRMWInstructionsemorySemanticsValue(
                 first_5bits_old_memory_semantics) &&
             IsAtomicRMWInstructionsemorySemanticsValue(
                 first_5bits_new_memory_semantics);

    case SpvOpControlBarrier:
    case SpvOpMemoryBarrier:
    case SpvOpMemoryNamedBarrier:

      // Forbidden because it will change the semantics.
      if (first_5bits_old_memory_semantics == SpvMemorySemanticsAcquireMask &&
          first_5bits_new_memory_semantics == SpvMemorySemanticsReleaseMask) {
        return false;
      }
      return IsBarrierInstructionsMemorySemanticsValue(
                 first_5bits_old_memory_semantics) &&
             IsBarrierInstructionsMemorySemanticsValue(
                 first_5bits_new_memory_semantics);

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
