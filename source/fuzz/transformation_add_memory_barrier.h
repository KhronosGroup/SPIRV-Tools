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

#ifndef SOURCE_FUZZ_TRANSFORMATION_Add_MEMORY_BARRIER_H_
#define SOURCE_FUZZ_TRANSFORMATION_Add_MEMORY_BARRIER_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Transformation is responsible for adding an OpMemoryBarrier.
class TransformationAddMemoryBarrier : public Transformation {
 public:
  explicit TransformationAddMemoryBarrier(
      protobufs::TransformationAddMemoryBarrier message);

  TransformationAddMemoryBarrier(
      uint32_t memory_scope_id, uint32_t memory_semantics_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // - |message_memory_scope_id| must be the id of an OpConstant 32 bit integer
  //   instruction with the value SpvScopeInvocation.
  // - |message_.memory_semantics_id| must be the id of an OpConstant 32 bit
  //   integer instruction with the values SpvMemorySemanticsWorkgroupMemoryMask
  //   or SpvMemorySemanticsUniformMemoryMask.
  // - |message_.instruction_to_insert_before| must identify an instruction
  //   before which it is valid to insert an OpMemoryBarrier.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds an instruction of the form:
  //   OpMemoryBarrier |memory_scope_id| |memory_semantics_id|
  // before the instruction identified by
  // |message_.instruction_to_insert_before|
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Returns false if Memory scope id not exists, not valid according to an
  // integer type with a width equal 32 bits, and scope value is not Invocation.
  bool IsMemoryScopeIdValid(opt::IRContext* ir_context,
                            spvtools::opt::Instruction* insert_before) const;

  // Returns false if Memory scope id not exists, not valid according to an
  // integer type with a width equal 32 bits, and semantics lower bits not equal
  // relaxed and higher bits not equal |SpvMemorySemanticsUniformMemoryMask| or
  // |SpvMemorySemanticsWorkgroupMemoryMask|.
  bool IsMemorySemancticsIdValid(
      opt::IRContext* ir_context,
      spvtools::opt::Instruction* insert_before) const;

  const uint32_t kMemorySemanticsHigherBitmask = 0xFFFFFFE0;
  const uint32_t kMemorySemanticsLowerBitmask = 0x1F;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddMemoryBarrier message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_Add_MEMORY_BARRIER_H_
