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

#ifndef SOURCE_FUZZ_TRANSFORMATION_COPY_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_COPY_INSTRUCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationCopyInstruction : public Transformation {
 public:
  explicit TransformationCopyInstruction(
      const protobufs::TransformationCopyInstruction& message);

  TransformationCopyInstruction(
      const protobufs::InstructionDescriptor& instruction_descriptor,
      const std::vector<uint32_t>& fresh_ids);

  // TODO
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // TODO
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  static bool CanCopyInstruction(opt::IRContext* ir_context,
                                 opt::Instruction* inst);

 private:
  // Returns the last iterator in the |block| that can be used to insert
  // |opcode| before itself.
  static opt::BasicBlock::iterator GetLastInsertBeforeIterator(
      opt::BasicBlock* block, SpvOp opcode);

  protobufs::TransformationCopyInstruction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_COPY_INSTRUCTION_H_
