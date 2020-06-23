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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationMoveInstruction : public Transformation {
 public:
  explicit TransformationMoveInstruction(
      const protobufs::TransformationMoveInstruction& message);

  TransformationMoveInstruction(
      const protobufs::InstructionDescriptor& insert_before,
      const protobufs::InstructionDescriptor& target);

  // TODO
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // TODO
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true is the instruction, pointed to be |target_it|, can be moved
  // before |insert_before_it|.
  static bool CanMoveInstruction(opt::IRContext* ir_context,
                                 opt::BasicBlock::iterator insert_before_it,
                                 opt::BasicBlock::iterator target_it);

 private:
  // Returns true if there is no path from |source_it| to |dest_it| containing
  // a memory barrier as specified by the IsMemoryBarrier method.
  static bool PathsHaveNoMemoryBarriers(opt::IRContext* ir_context,
                                        opt::BasicBlock::iterator source_it,
                                        opt::BasicBlock::iterator dest_it);

  // Returns true if both |first| and |second| belong to the |block| and
  // |first| precedes or is equal to |second|.
  static bool IteratorsAreOrderedCorrectly(opt::BasicBlock* block,
                                           opt::BasicBlock::iterator first,
                                           opt::BasicBlock::iterator second);

  // Consider an attempt to move some instruction in the module from
  // position A to position B, s.t. there exists a path from A to B (or from B
  // to A depending on the relative location of those positions in the module)
  // that contains an |opcode|. This function returns true if such an attempt
  // changes module's semantics.
  static bool IsMemoryBarrier(SpvOp opcode);

  // Returns true if an instruction with |opcode| can't be moved.
  static bool CanMoveOpcode(SpvOp opcode);

  protobufs::TransformationMoveInstruction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MOVE_INSTRUCTION_H_
