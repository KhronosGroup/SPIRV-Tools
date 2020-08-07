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
          instruction_to_fresh_ids = {},
      std::vector<uint32_t> overflow_ids = {});

  // - |message_.header_block_id| must be the label id of a selection header,
  //   which ends with an OpBranchConditional instruction.
  // - The header block and the merge block must describe a single-entry,
  //   single-exit region.
  // - The region must not contain atomic or barrier instructions.
  // - The region must not contain selection or loop constructs.
  // - For each instruction that requires additional fresh ids, there must be
  //   enough fresh ids, which can either be found in the corresponding pair in
  //   |message_.instruction_to_fresh ids|, or in |message_.overflow_id| if
  //   there is no mapping or not enough ids are specified in the mapping.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationFlattenConditionalBranch message_;

  // Returns an unordered_map mapping instructions to lists of fresh ids. It
  // gets the information from |message_.instruction_to_fresh_ids|.
  std::unordered_map<opt::Instruction*, std::vector<uint32_t>>
  GetInstructionsToFreshIdsMapping(opt::IRContext* ir_context) const;

  // Returns the number of fresh ids needed to enclose the instruction with the
  // given opcode in a conditional. This can only be called on OpStore, OpLoad
  // and OpFunctionCall.
  uint32_t NumOfFreshIdsNeededByOpcode(SpvOp opcode) const;

  // Splits the given block, adding a new selection construct so that the given
  // instruction is only executed if the boolean value of |condition_id| matches
  // the value of |exec_if_cond_true|.
  // The instruction must be one of OpStore, OpLoad and OpFunctionCall.
  // Assumes that all parameters are valid and consistent.
  // 2 fresh ids are required if the instruction does not have a result id, 5
  // otherwise.
  // Returns the merge block created.
  opt::BasicBlock* EncloseInstructionInConditional(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      opt::BasicBlock* block, opt::Instruction* instruction,
      std::vector<uint32_t> fresh_ids, uint32_t condition_id,
      bool exec_if_cond_true) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_FLATTEN_CONDITIONAL_BRANCH_H
