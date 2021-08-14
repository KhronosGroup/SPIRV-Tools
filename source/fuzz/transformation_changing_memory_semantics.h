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

#ifndef SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SEMANTICS_H_
#define SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SEMANTICS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// This transformation is responsible for changing the mask of memory
// semantics with a range of 5 bits of values.
class TransformationChangingMemorySemantics : public Transformation {
 public:
  explicit TransformationChangingMemorySemantics(
      protobufs::TransformationChangingMemorySemantics message);

  TransformationChangingMemorySemantics(
      const protobufs::InstructionDescriptor& atomic_instruction,
      uint32_t memory_semantics_operand_position,
      uint32_t memory_semantics_new_value_id);

  // - |message_.atomic_instruction| atomic instruction that would like to
  //   change its memory semantics value.
  // - |message_.memory_semantics_operand_position| position of atomic
  // instruction
  //   would like to change, must be equal to 2 or 3 only.
  // - |message_.memory_semantics_new_value_id| the new id of memory semantics
  //   that is will change with the old.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Change id of memory semantics for specific position with a new one.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Check if instruction is atomic instruction only. Check if Memory semantics
  // operand Position equal to 0 or 1. Check if operand position equal to 0 in
  // the case of instruction that takes one memory semantics operand and 0 or 1
  // for instructions that takes two operands.
  static bool IsNeededOpcodeWithAppropriatePosition(SpvOp opcode,
                                                    uint32_t position);

  // - |opcode| opcode must be atomic or barrier instruction.
  //   Valid memory semantics masks:
  //   OpAtomicLoad: None | Acquire | SequentiallyConsistent
  //   OpAtomicStore: None | Release | SequentiallyConsistent
  //   OpAtomicRMW instructions: None | AcquireRelease | SequentiallyConsistent
  //   Barrier instructions:  None | Acquire | Release | AcquireRelease |
  //   SequentiallyConsistent
  // - |lower_bits_old_memory_semantics| must be suitable for specific atomic
  //   or barrier instruction.
  // - |lower_bits_new_memory_semantics| should be larger than the old one,
  //   must be suitable for specific atomic or barrier instruction.
  // - |memory_model| return false if the memory model is Vulkan and memory
  //   semantics is Sequentially Consistent.
  static bool IsValidConversion(
      SpvOp opcode, SpvMemorySemanticsMask first_5bits_old_memory_semantics,
      SpvMemorySemanticsMask first_5bits_new_memory_semantics,
      SpvMemoryModel memory_model);

  // Check available memory semantics values for the atomic load instruction.
  static bool IsAtomicLoadMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Check available memory semantics values for the atomic store instruction.
  static bool IsAtomicStoreMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Check available memory semantics values for the atomic read modify write
  // instructions.
  static bool IsAtomicRMWInstructionsemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Check available memory semantics values for the barrier instructions.
  static bool IsBarrierInstructionsMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Return one if instruction is OpMemoryBarrier, else will return
  // suitable position value (two or three).
  static uint32_t GetMemorySemanticsOperandIndex(SpvOp opcode,
                                                 uint32_t zero_or_one);

  // Return number of memory semantic operands for the specific atomic or
  // barrier instruction.
  static uint32_t GetNumberOfMemorySemantics(SpvOp opcode);

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  static const uint32_t kMemorySemanticsHigherBitmask = 0xFFFFFFE0;
  static const uint32_t kMemorySemanticsLowerBitmask = 0x1F;

 private:
  protobufs::TransformationChangingMemorySemantics message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SEMANTICS_H_
