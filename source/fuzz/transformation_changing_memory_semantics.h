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

// Strengthens a memory semantics operand within an instruction. Only the memory
// order (the first 5 bits) is changed.
class TransformationChangingMemorySemantics : public Transformation {
 public:
  explicit TransformationChangingMemorySemantics(
      protobufs::TransformationChangingMemorySemantics message);

  TransformationChangingMemorySemantics(
      const protobufs::InstructionDescriptor& atomic_instruction,
      uint32_t memory_semantics_operand_position,
      uint32_t memory_semantics_new_value_id);

  // - |message_.atomic_instruction| atomic or barrier instruction that would
  //   like to change its memory semantics value.
  // - |message_.memory_semantics_operand_position| position of atomic
  //   or barrier instruction would like to change, must be equal to 0 or 1
  //   only.
  // - |message_.memory_semantics_new_value_id| the new id of memory semantics
  //   that is will change with the old.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Change id of memory semantics for specific position with a new one.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Returns true if the proposed memory order strengthening from
  // |first_5bits_old_memory_semantics| to |first_5bits_new_memory_semantics| is
  // both valid (according to the SPIR-V specification) and is actually a
  // strengthening that would not change the semantics of a SPIR-V module. For
  // full details, see the comments within the function body.
  static bool IsSuitableStrengthening(
      opt::IRContext* ir_context,
      spvtools::opt::Instruction* needed_atomic_instruction,
      SpvMemorySemanticsMask first_5bits_old_memory_semantics,
      SpvMemorySemanticsMask first_5bits_new_memory_semantics,
      uint32_t memory_semantics_operand_position, SpvMemoryModel memory_model);

  // Returns true if |memory_semantics_value| is suitable for the atomic load
  // instruction.
  static bool IsAtomicLoadMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Returns true if |memory_semantics_value| is suitable for the atomic store
  // instruction.
  static bool IsAtomicStoreMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Returns true if |memory_semantics_value| is suitable for the
  // atomic(read-modify-write) instructions.
  static bool IsAtomicRMWInstructionsemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Returns true if |memory_semantics_value| is suitable for the barrier
  // instructions.
  static bool IsBarrierInstructionsMemorySemanticsValue(
      SpvMemorySemanticsMask memory_semantics_value);

  // Returns the "in operand" index of the memory semantics operand for the
  // instruction.
  // |opcode|: the instruction opcode.
  // |memory_semantics_operand_position|: 0 for the first memory semantics
  // operand or 1 for the second memory semantics operand in the instruction.
  static uint32_t GetMemorySemanticsInOperandIndex(
      SpvOp opcode, uint32_t memory_semantics_operand_position);

  // Returns number of memory semantic operands for the specific atomic or
  // barrier instruction.
  static uint32_t GetNumberOfMemorySemantics(SpvOp opcode);

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  static const uint32_t kMemorySemanticsHigherBitmask = 0xFFFFFFE0;
  static const uint32_t kMemorySemanticsLowerBitmask = 0x1F;
  static const uint32_t kUnequalMemorySemanticsOperandPosition = 1;

 private:
  protobufs::TransformationChangingMemorySemantics message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SEMANTICS_H_
