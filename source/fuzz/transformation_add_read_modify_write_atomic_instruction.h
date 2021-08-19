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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_READ_MODIFY_WRITE_ATOMIC_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_READ_MODIFY_WRITE_ATOMIC_INSTRUCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// This transformation is responsible for adding atomic(Read-Modify-Write)
// instruction.
class TransformationAddReadModifyWriteAtomicInstruction
    : public Transformation {
 public:
  explicit TransformationAddReadModifyWriteAtomicInstruction(
      protobufs::TransformationAddReadModifyWriteAtomicInstruction message);

  TransformationAddReadModifyWriteAtomicInstruction(
      uint32_t fresh_id, uint32_t pointer_id, uint32_t opcode,
      uint32_t memory_scope_id, uint32_t memory_semantics_id_1,
      uint32_t memory_semantics_id_2, uint32_t value_id, uint32_t comparator_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // - |message_.fresh_id| must be fresh
  // - |message_.pointer_id| must be the id of a pointer
  // - |message_.opcode| atomic RMW  would like to apppend.
  // - |message_memory_scope_id| must be the id of
  //   an OpConstant 32 bit integer instruction with the value
  //   SpvScopeInvocation.
  // - |message_.memory_semantics_id_1| must be the id
  //   of an OpConstant 32 bit integer instruction with the values
  //   SpvMemorySemanticsWorkgroupMemoryMask or
  //   SpvMemorySemanticsUniformMemoryMask.
  // - |message_.memory_semantics_id_2| must be the id
  //   of an OpConstant 32 bit integer instruction with the values
  //   SpvMemorySemanticsWorkgroupMemoryMask or
  //   SpvMemorySemanticsUniformMemoryMask.
  // - |message_.value_id| The value to be used in specific instructions.
  // - |message_.comparator_id| The comparator will be used in specific
  //   instructions.
  // - The pointer must not be OpConstantNull or OpUndef
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which it is valid to insert an instruction.
  //   |message_.pointer_id| is available (according to dominance rules)
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction of the form:
  //   |message_.fresh_id| = OpAtomic.(RMW). %type |message_.pointer_id|
  // before the instruction identified by
  // |message_.instruction_to_insert_before|, where %type is the pointer's
  // pointee type.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Return atomic RMW instruction from the specific opcode.
  std::unique_ptr<spvtools::opt::Instruction> GetInstruction(
      opt::IRContext* ir_context, SpvOp opcode) const;

  // Memory scope id must exist and be valid.
  bool IsMemoryScopeValid(opt::IRContext* ir_context,
                          spvtools::opt::Instruction* insert_before) const;

  // Memory semantic main id must exist and be valid from a specific storage
  // class.
  bool IsMemorySemanticsId1Valid(
      opt::IRContext* ir_context, spvtools::opt::Instruction* insert_before,
      spvtools::opt::Instruction* pointer_type) const;

  // Memory semantics second id must exist and be valid for atomic instruction
  // that use it.
  bool IsMemorySemanticsId2Valid(
      opt::IRContext* ir_context, spvtools::opt::Instruction* insert_before,
      spvtools::opt::Instruction* pointer_type) const;

  // Value id must exist and be valid for atomic instruction that uses it.
  bool IsValueIdValid(opt::IRContext* ir_context,
                      spvtools::opt::Instruction* insert_before) const;

  // Comparator id must exist and be valid for atomic instruction that uses it.
  bool IsComparatorIdValid(opt::IRContext* ir_context,
                           spvtools::opt::Instruction* insert_before) const;

  static const uint32_t kMemorySemanticsHigherBitmask = 0xFFFFFFE0;
  static const uint32_t kMemorySemanticsLowerBitmask = 0x1F;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddReadModifyWriteAtomicInstruction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_READ_MODIFY_WRITE_ATOMIC_INSTRUCTION_H_
