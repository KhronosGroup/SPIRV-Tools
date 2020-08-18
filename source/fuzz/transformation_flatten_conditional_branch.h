// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H
#define SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationFlattenConditionalBranch : public Transformation {
 public:
  explicit TransformationFlattenConditionalBranch(
      const protobufs::TransformationFlattenConditionalBranch& message);

  TransformationFlattenConditionalBranch(
      uint32_t header_block_id,
      std::vector<
          std::pair<protobufs::InstructionDescriptor, std::vector<uint32_t>>>
          instructions_to_fresh_ids = {},
      std::vector<uint32_t> overflow_ids = {});

  // - |message_.header_block_id| must be the label id of a selection header,
  //   which ends with an OpBranchConditional instruction.
  // - The header block and the merge block must describe a single-entry,
  //   single-exit region.
  // - The region must not contain barrier or OpSampledImage instructions.
  // - The region must not contain selection or loop constructs.
  // - For each instruction that requires additional fresh ids, then:
  //   - if the instruction is mapped to a list of fresh ids by
  //     |message_.instruction_to_fresh ids|, there must be enough fresh ids in
  //     this list;
  //   - if there is no such mapping, there must be enough fresh ids in
  //     |message_.overflow_id|
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Flattens the selection construct with header |message_.header_block_id|,
  // changing any OpPhi in the block where the flow converges to OpSelect and
  // enclosing any OpStore, OpLoad and OpFunctionCall in conditionals so that
  // they are only executed when they should.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if the conditional headed by |header| can be flattened,
  // according to the conditions of the IsApplicable method, assuming that
  // enough fresh ids are given. In this case, it fills the
  // |instructions_that_need_ids| set with all the instructions that would
  // require fresh ids.
  // Returns false otherwise.
  // Assumes that |header| is the header of a conditional, so its last two
  // instructions are OpSelectionMerge and OpBranchConditional.
  static bool ConditionalCanBeFlattened(
      opt::IRContext* ir_context, opt::BasicBlock* header,
      std::set<opt::Instruction*>* instructions_that_need_ids);

  // Returns the number of fresh ids needed to enclose the given instruction in
  // a conditional. That is:
  // - 2 if the instruction does not have a result id, needed for 2 new blocks
  // - 5 if the instruction has a result id: 3 for new blocks, 1 for a new
  //   OpUndef instruction, 1 for the instruction itself
  static uint32_t NumOfFreshIdsNeededByInstruction(
      opt::Instruction* instruction);

 private:
  protobufs::TransformationFlattenConditionalBranch message_;

  // Returns an unordered_map mapping instructions to lists of fresh ids. It
  // gets the information from |message_.instruction_to_fresh_ids|.
  std::unordered_map<opt::Instruction*, std::vector<uint32_t>>
  GetInstructionsToFreshIdsMapping(opt::IRContext* ir_context) const;

  // Splits the given block, adding a new selection construct so that the given
  // instruction is only executed if the boolean value of |condition_id| matches
  // the value of |exec_if_cond_true|.
  // The instruction must be one of OpStore, OpLoad and OpFunctionCall.
  // Assumes that all parameters are consistent.
  // 2 fresh ids are required if the instruction does not have a result id, 5
  // otherwise.
  // Returns the merge block created.
  opt::BasicBlock* EncloseInstructionInConditional(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      opt::BasicBlock* block, opt::Instruction* instruction,
      const std::vector<uint32_t>& fresh_ids, uint32_t condition_id,
      bool exec_if_cond_true) const;

  // Returns true if the given instruction either has no side effects or it can
  // be handled by being enclosed in a conditional.
  static bool InstructionCanBeHandled(opt::Instruction* instruction);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H
