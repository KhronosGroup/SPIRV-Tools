// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_
#define SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_

#include <unordered_map>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPropagateInstructionUp : public Transformation {
 public:
  explicit TransformationPropagateInstructionUp(
      const protobufs::TransformationPropagateInstructionUp& message);

  TransformationPropagateInstructionUp(
      uint32_t block_id,
      const std::unordered_map<uint32_t, uint32_t>& predecessor_id_to_fresh_id);

  // TODO
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // TODO
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if this transformation can be applied to the block with id
  // |block_id|. Concretely, returns true iff:
  // - the block is not the entry point of its function
  // - the block contains an instruction that can be propagated
  static bool IsApplicableToTheBlock(opt::IRContext* ir_context,
                                     uint32_t block_id);

 private:
  // Returns the instruction that will be propagated into the predecessors of
  // the |block_id|. Returns nullptr if no such an instruction exists.
  static opt::Instruction* GetInstructionToPropagate(opt::IRContext* ir_context,
                                                     uint32_t block_id);

  // Returns true if |opcode| is supported by this transformation.
  static bool IsOpcodeSupported(SpvOp opcode);

  // Returns the last instruction in the |block| that allows us to insert
  // |opcode| above itself. Returns nullptr is no such instruction exists.
  static opt::Instruction* GetLastInsertBeforeInstruction(
      opt::BasicBlock* block, SpvOp opcode);

  protobufs::TransformationPropagateInstructionUp message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PROPAGATE_INSTRUCTION_UP_H_
