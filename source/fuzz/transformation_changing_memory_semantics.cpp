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
    : message_(std::move(message)) {
  assert(GetNumberOfMemorySemanticsOperands(static_cast<SpvOp>(
             message_.instruction().target_instruction_opcode())) > 0 &&
         "The instruction does not have any memory semantics operands.");

  assert(message_.memory_semantics_operand_position() <
             GetNumberOfMemorySemanticsOperands(static_cast<SpvOp>(
                 message_.instruction().target_instruction_opcode())) &&
         "The operand position is out of bounds.");
}

TransformationChangingMemorySemantics::TransformationChangingMemorySemantics(
    const protobufs::InstructionDescriptor& instruction,
    uint32_t memory_semantics_operand_position,
    uint32_t memory_semantics_new_value_id) {
  *message_.mutable_instruction() = instruction;

  assert(GetNumberOfMemorySemanticsOperands(
             static_cast<SpvOp>(instruction.target_instruction_opcode())) > 0 &&
         "The instruction does not have any memory semantics operands.");
  assert(memory_semantics_operand_position <
             GetNumberOfMemorySemanticsOperands(
                 static_cast<SpvOp>(instruction.target_instruction_opcode())) &&
         "The operand position is out of bounds.");

  message_.set_memory_semantics_operand_position(
      memory_semantics_operand_position);
  message_.set_memory_semantics_new_value_id(memory_semantics_new_value_id);
}

bool TransformationChangingMemorySemantics::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto needed_instruction = FindInstruction(message_.instruction(), ir_context);
  // The atomic or barrier instruction must exist.
  if (!needed_instruction) {
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

  // The lower bits of the new memory semantics value must be suitable.
  auto new_memory_sematics_value = value_instruction->GetSingleWordInOperand(0);
  auto old_memory_sematics_value =
      ir_context->get_def_use_mgr()
          ->GetDef(needed_instruction->GetSingleWordInOperand(
              GetMemorySemanticsInOperandIndex(
                  needed_instruction->opcode(),
                  message_.memory_semantics_operand_position())))
          ->GetSingleWordInOperand(0);

  auto lower_bits_new_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      new_memory_sematics_value & kMemorySemanticsLowerBitmask);
  auto lower_bits_old_memory_semantics = static_cast<SpvMemorySemanticsMask>(
      old_memory_sematics_value & kMemorySemanticsLowerBitmask);
  auto memory_model = static_cast<SpvMemoryModel>(
      ir_context->module()->GetMemoryModel()->GetSingleWordInOperand(1));

  if (!IsSuitableStrengthening(
          ir_context, needed_instruction, lower_bits_old_memory_semantics,
          lower_bits_new_memory_semantics,
          message_.memory_semantics_operand_position(), memory_model)) {
    return false;
  }

  // The higher bits value of old and new memory semantics id must be equal.
  auto higher_bits_new_memory_semantics =
      new_memory_sematics_value & kMemorySemanticsHigherBitmask;
  auto higher_bits_old_memory_semantics =
      old_memory_sematics_value & kMemorySemanticsHigherBitmask;
  if (higher_bits_new_memory_semantics != higher_bits_old_memory_semantics) {
    return false;
  }

  return true;
}

void TransformationChangingMemorySemantics::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto needed_instruction = FindInstruction(message_.instruction(), ir_context);

  uint32_t needed_index = GetMemorySemanticsInOperandIndex(
      needed_instruction->opcode(),
      message_.memory_semantics_operand_position());

  needed_instruction->SetInOperand(needed_index,
                                   {message_.memory_semantics_new_value_id()});
}
uint32_t
TransformationChangingMemorySemantics::GetNumberOfMemorySemanticsOperands(
    SpvOp opcode) {
  switch (opcode) {
    // Atomic Instructions.
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
    // Barrier Instructions.
    case SpvOpControlBarrier:
    case SpvOpMemoryBarrier:
    case SpvOpMemoryNamedBarrier:
      return 1;

    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
      return 2;

    default:
      return 0;
  }
}
uint32_t
TransformationChangingMemorySemantics::GetMemorySemanticsInOperandIndex(
    SpvOp opcode, uint32_t memory_semantics_operand_position) {
  switch (opcode) {
    case SpvOpMemoryBarrier:
      assert(memory_semantics_operand_position == 0 &&
             "memory_semantics_operand_position not equal zero.");
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
      return memory_semantics_operand_position == 0 ? 2 : 3;

    default:
      assert(false);
      return -1;
  }
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

bool TransformationChangingMemorySemantics::IsSuitableStrengthening(
    opt::IRContext* ir_context, spvtools::opt::Instruction* needed_instruction,
    SpvMemorySemanticsMask lower_bits_old_memory_semantics,
    SpvMemorySemanticsMask lower_bits_new_memory_semantics,
    uint32_t memory_semantics_operand_position, SpvMemoryModel memory_model) {
  //  Our aim is to strengthen the memory order because this does not
  //  change the semantics of the SPIR-V module. However, some memory
  //  orders are invalid (according the SPIR-V specification) in certain
  //  cases, and others are technically allowed, but confusing. Thus, we
  //  only allow the following strengthenings depending on the type of
  //  instruction:
  //    - Load: None -> Acquire -> SequentiallyConsistent
  //    - Store: None -> Release -> SequentiallyConsistent
  //    - Read-modify-write: None -> AcquireRelease -> SequentiallyConsistent
  //    - Barrier: None -> Acquire / Release -> AcquireRelease ->
  //      SequentiallyConsistent
  // Note that in the barrier case, we can strengthen
  // from None to Acquire, or from None to Release, but we cannot go from
  // Acquire to Release. We can easily check all the above by checking that the
  // old and new memory order values are both one of the expected values for the
  // instruction type, and that: old value < new value, with a special case to
  // disallow going from Acquire to Release in a barrier instruction.

  // There are also other restrictions, explained below.
  // Can't use Sequentially Consistent with Vulkan memory model.
  if (lower_bits_new_memory_semantics ==
          SpvMemorySemanticsSequentiallyConsistentMask &&
      memory_model == SpvMemoryModelVulkan) {
    return false;
  }

  // Lower new bits can't be smaller than lower old bits because it will change
  // the semantics.
  if (lower_bits_old_memory_semantics >= lower_bits_new_memory_semantics) {
    return false;
  }

  // Compare and exchange instructions take two memory semantics values: "Equal"
  // and "Unequal". There are extra restrictions on the "Unequal" value.
  if ((needed_instruction->opcode() == SpvOpAtomicCompareExchange ||
       needed_instruction->opcode() == SpvOpAtomicCompareExchangeWeak) &&
      memory_semantics_operand_position ==
          kUnequalMemorySemanticsOperandPosition) {
    auto equal_memory_semantics_value_instruction =
        ir_context->get_def_use_mgr()->GetDef(
            needed_instruction->GetSingleWordInOperand(2));

    auto equal_memory_semantics_value_lower_bits =
        static_cast<SpvMemorySemanticsMask>(
            equal_memory_semantics_value_instruction->GetSingleWordInOperand(
                0) &
            kMemorySemanticsLowerBitmask);

    // The memory order of the "Unequal" value must not be stronger than the
    // "Equal" value.
    if (lower_bits_new_memory_semantics >
        equal_memory_semantics_value_lower_bits) {
      return false;
    }

    // The "Unequal" memory order (i.e. the lower bits) must not be Release or
    // AcquireRelease.
    if (lower_bits_new_memory_semantics == SpvMemorySemanticsReleaseMask ||
        lower_bits_new_memory_semantics ==
            SpvMemorySemanticsAcquireReleaseMask) {
      return false;
    }
  }

  // The old and new memory order values must both be expected values for the
  // given opcode.
  switch (needed_instruction->opcode()) {
    case SpvOpAtomicLoad:
      return IsAtomicLoadMemorySemanticsValue(
                 lower_bits_old_memory_semantics) &&
             IsAtomicLoadMemorySemanticsValue(lower_bits_new_memory_semantics);

    case SpvOpAtomicStore:
      return IsAtomicStoreMemorySemanticsValue(
                 lower_bits_old_memory_semantics) &&
             IsAtomicStoreMemorySemanticsValue(lower_bits_new_memory_semantics);

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
                 lower_bits_old_memory_semantics) &&
             IsAtomicRMWInstructionsemorySemanticsValue(
                 lower_bits_new_memory_semantics);

    case SpvOpControlBarrier:
    case SpvOpMemoryBarrier:
    case SpvOpMemoryNamedBarrier:

      // Forbidden because it will change the semantics.
      if (lower_bits_old_memory_semantics == SpvMemorySemanticsAcquireMask &&
          lower_bits_new_memory_semantics == SpvMemorySemanticsReleaseMask) {
        return false;
      }
      return IsBarrierInstructionsMemorySemanticsValue(
                 lower_bits_old_memory_semantics) &&
             IsBarrierInstructionsMemorySemanticsValue(
                 lower_bits_new_memory_semantics);

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
