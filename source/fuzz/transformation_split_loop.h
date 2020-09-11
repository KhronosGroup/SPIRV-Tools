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

#ifndef SPIRV_TOOLS_TRANSFORMATION_SPLIT_LOOP_H
#define SPIRV_TOOLS_TRANSFORMATION_SPLIT_LOOP_H

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
      uint32_t new_body_entry_block_fresh_id, uint32_t load_run_second_fresh_id,
      uint32_t selection_merge_block_fresh_id,
      const std::map<uint32_t, uint32_t>& original_label_to_duplicate_label,
      const std::map<uint32_t, uint32_t>& original_id_to_duplicate_id);

  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  static std::set<opt::BasicBlock*> GetRegionBlocks(
      opt::IRContext* ir_context, opt::BasicBlock* entry_block,
      opt::BasicBlock* exit_block);

 private:
  protobufs::TransformationSplitLoop message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_ADD_RELAXED_DECORATION_H
