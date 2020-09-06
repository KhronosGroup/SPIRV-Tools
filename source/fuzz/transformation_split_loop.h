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
      uint32_t variable_counter_fresh_id, uint32_t variable_exited_fresh_id,
      uint32_t load_counter_fresh_id, uint32_t load_exited_fresh_id,
      uint32_t iteration_limit_id, uint32_t condition_counter_fresh_id,
      uint32_t logical_not_fresh_id, uint32_t then_branch_fresh_id,
      uint32_t else_branch_fresh_id, uint32_t entry_block_id,
      uint32_t exit_block_id,
      std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
      std::map<uint32_t, uint32_t> original_label_to_duplicate_label);

  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationSplitLoop message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_ADD_RELAXED_DECORATION_H
