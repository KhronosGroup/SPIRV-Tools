// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_set_memory_operands_mask.h"

#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    const spvtools::fuzz::protobufs::TransformationSetMemoryOperandsMask&
        message)
    : message_(message) {}

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    const protobufs::InstructionDescriptor& memory_access_instruction,
    uint32_t memory_operands_mask, uint32_t memory_operands_mask_index) {
  *message_.mutable_memory_access_instruction() = memory_access_instruction;
  message_.set_memory_operands_mask(memory_operands_mask);
  message_.set_memory_operands_mask_index(memory_operands_mask_index);
}

bool TransformationSetMemoryOperandsMask::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  auto instruction =
      FindInstruction(message_.memory_access_instruction(), context);
  if (!instruction) {
    return false;
  }
  if (!IsMemoryAccess(*instruction)) {
    return false;
  }

  if (message_.memory_operands_mask_index() != 0) {
    assert(message_.memory_operands_mask_index() == 1);
    assert(instruction->opcode() == SpvOpCopyMemory ||
           instruction->opcode() == SpvOpCopyMemorySized);
    assert(MultipleMemoryOperandMasksAreSupported(context));
  }

  return NewMaskIsValid(*instruction,
                        GetOriginalMaskInOperandIndex(*instruction));
}

void TransformationSetMemoryOperandsMask::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  auto instruction =
      FindInstruction(message_.memory_access_instruction(), context);
  auto original_mask_in_operand_index =
      GetOriginalMaskInOperandIndex(*instruction);
  if (original_mask_in_operand_index >= instruction->NumInOperands()) {
    instruction->AddOperand(
        {SPV_OPERAND_TYPE_MEMORY_ACCESS, {message_.memory_operands_mask()}});

  } else {
    instruction->SetInOperand(original_mask_in_operand_index,
                              {message_.memory_operands_mask()});
  }
}

protobufs::Transformation TransformationSetMemoryOperandsMask::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_set_memory_operands_mask() = message_;
  return result;
}

bool TransformationSetMemoryOperandsMask::IsMemoryAccess(
    const opt::Instruction& instruction) {
  switch (instruction.opcode()) {
    case SpvOpLoad:
    case SpvOpStore:
    case SpvOpCopyMemory:
    case SpvOpCopyMemorySized:
      return true;
    default:
      return false;
  }
}

bool TransformationSetMemoryOperandsMask::NewMaskIsValid(
    const opt::Instruction& instruction,
    uint32_t original_mask_in_operand_index) const {
  assert(original_mask_in_operand_index != 0 &&
         "The given mask index is not valid.");

  uint32_t original_mask =
      instruction.NumInOperands() > original_mask_in_operand_index
          ? instruction.GetSingleWordInOperand(original_mask_in_operand_index)
          : static_cast<uint32_t>(SpvMemoryAccessMaskNone);
  uint32_t new_mask = message_.memory_operands_mask();

  // Volatile must not be removed
  if ((original_mask & SpvMemoryAccessVolatileMask) &&
      !(new_mask & SpvMemoryAccessVolatileMask)) {
    return false;
  }

  // Nontemporal can be added or removed, and no other flag is allowed to
  // change.  We do this by checking that the masks are equal once we set
  // their Volatile and Nontemporal flags to the same value (this works
  // because valid manipulation of Volatile is checked above, and the manner
  // in which Nontemporal is manipulated does not matter).
  return (original_mask | SpvMemoryAccessVolatileMask |
          SpvMemoryAccessNontemporalMask) ==
         (new_mask | SpvMemoryAccessVolatileMask |
          SpvMemoryAccessNontemporalMask);
}

uint32_t TransformationSetMemoryOperandsMask::GetOriginalMaskInOperandIndex(
    const opt::Instruction& instruction) const {
  uint32_t first_mask_in_operand_index = 0;
  switch (instruction.opcode()) {
    case SpvOpLoad:
      first_mask_in_operand_index = 1;
      break;
    case SpvOpStore:
      first_mask_in_operand_index = 2;
      break;
    case SpvOpCopyMemory:
      first_mask_in_operand_index = 2;
      break;
    case SpvOpCopyMemorySized:
      first_mask_in_operand_index = 3;
      break;
    default:
      assert(false && "Unknown memory instruction.");
      break;
  }
  if (message_.memory_operands_mask_index() == 0) {
    return first_mask_in_operand_index;
  }
  assert(message_.memory_operands_mask_index() == 1 &&
         "Memory operands mask index must be 0 or 1.");

  uint32_t first_mask =
      instruction.GetSingleWordInOperand(first_mask_in_operand_index);
  uint32_t first_mask_extra_operand_count = 0;

  if (first_mask & SpvMemoryAccessAlignedMask) {
    first_mask_extra_operand_count++;
  }
  if (first_mask & (SpvMemoryAccessMakePointerAvailableMask |
                    SpvMemoryAccessMakePointerAvailableKHRMask)) {
    first_mask_extra_operand_count++;
  }
  if (first_mask & (SpvMemoryAccessMakePointerVisibleMask |
                    SpvMemoryAccessMakePointerVisibleKHRMask)) {
    first_mask_extra_operand_count++;
  }
  return first_mask_in_operand_index + first_mask_extra_operand_count + 1;
}

bool TransformationSetMemoryOperandsMask::
    MultipleMemoryOperandMasksAreSupported(opt::IRContext* context) {
  // TODO(afd): We capture the universal environments for which this loop
  //  control is definitely not supported.  The check should be refined on
  //  demand for other target environments.
  switch (context->grammar().target_env()) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
      return false;
    default:
      return true;
  }
}

}  // namespace fuzz
}  // namespace spvtools
