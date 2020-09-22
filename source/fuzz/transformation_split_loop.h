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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SPLIT_LOOP_H_
#define SOURCE_FUZZ_TRANSFORMATION_SPLIT_LOOP_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSplitLoop : public Transformation {
 public:
  explicit TransformationSplitLoop(
      const protobufs::TransformationSplitLoop& message);

  explicit TransformationSplitLoop(
      uint32_t loop_header_id, uint32_t variable_counter_id,
      uint32_t variable_run_second_id, uint32_t constant_limit_id,
      uint32_t load_counter_fresh_id, uint32_t increment_counter_fresh_id,
      uint32_t condition_counter_fresh_id,
      uint32_t new_body_entry_block_fresh_id,
      uint32_t conditional_block_fresh_id, uint32_t load_run_second_fresh_id,
      uint32_t selection_merge_block_fresh_id,
      const std::vector<uint32_t>& logical_not_fresh_ids,
      const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
      const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id);

  // - |loop_header_id| must refer to an existing loop header.
  // - |variable_counter_id| must refer to a variable of type integer.
  // - |variable_run_second_id| must refer to a variable of type bool.
  // - |constant_limit_id| must correspond to an integer constant (the limit of
  //   iterations in the first loop).
  // - |load_counter_fresh_id|, |increment_counter_fresh_id|,
  //   |condition_counter_fresh_id|, |new_body_entry_block_fresh_id|,
  //   |conditional_block_fresh_id|, |load_run_second_fresh_id|,
  //   |selection_merge_block_fresh_id| must be distinct, fresh ids.
  // - |logical_not_fresh_ids| must contain at least as many distinct fresh ids
  //   as there are terminators of form "OpBranchConditional %cond %merge
  //   %other" in the loop.
  // - |original_label_to_duplicate_label| must contain at least a key for every
  //   block in the original loop.
  // - |original_id_to_duplicate_id| must contain at least a key for every
  //   result id in the original loop.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // A transformation that replaces a loop with two loops, which in total
  // iterate over the same loop body with the same condition of iteration.
  // The first loop is executed up to a constant number of times. If more
  // iterations are required, they are performed in the second loop.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the set of blocks dominated by |entry_block| and post-dominated
  // by |exit_block|.
  static std::set<opt::BasicBlock*> GetRegionBlocks(
      opt::IRContext* ir_context, opt::BasicBlock* entry_block,
      opt::BasicBlock* exit_block);

 private:
  protobufs::TransformationSplitLoop message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SPLIT_LOOP_H_
