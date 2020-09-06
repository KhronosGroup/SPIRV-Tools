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

#include "source/fuzz/transformation_split_loop.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

/*
 * uint32_t variable_counter_fresh_id, uint32_t variable_exited_fresh_id,
      uint32_t load_counter_fresh_id, uint32_t load_exited_fresh_id,
      uint32_t iteration_limit_id, uint32_t condition_counter_fresh_id,
      uint32_t logical_not_fresh_id, uint32_t then_branch_fresh_id,
      uint32_t else_branch_fresh_id, uint32_t entry_block_id,
      uint32_t exit_block_id,
      std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
      std::map<uint32_t, uint32_t> original_label_to_duplicate_label
 */

TransformationSplitLoop::TransformationSplitLoop(
    const spvtools::fuzz::protobufs::TransformationSplitLoop& message)
    : message_(message) {}

TransformationSplitLoop::TransformationSplitLoop(
    uint32_t variable_counter_fresh_id, uint32_t variable_exited_fresh_id,
    uint32_t load_counter_fresh_id, uint32_t load_exited_fresh_id,
    uint32_t iteration_limit_id, uint32_t condition_counter_fresh_id,
    uint32_t logical_not_fresh_id, uint32_t then_branch_fresh_id,
    uint32_t else_branch_fresh_id, uint32_t entry_block_id,
    uint32_t exit_block_id,
    std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
    std::map<uint32_t, uint32_t> original_label_to_duplicate_label) {
  message_.set_variable_counter_fresh_id(variable_counter_fresh_id);
  message_.set_variable_exited_fresh_id(variable_exited_fresh_id);
  message_.set_load_counter_fresh_id(load_counter_fresh_id);
  message_.set_load_exited_fresh_id(load_exited_fresh_id);
  message_.set_iteration_limit_id(iteration_limit_id);
  message_.set_condition_counter_fresh_id(condition_counter_fresh_id);
  message_.set_logical_not_fresh_id(logical_not_fresh_id);
  message_.set_then_branch_fresh_id(then_branch_fresh_id);
  message_.set_else_branch_fresh_id(else_branch_fresh_id);
  message_.set_entry_block_id(entry_block_id);
  message_.set_exit_block_id(exit_block_id);
  *message_.mutable_original_id_to_duplicate_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_duplicate_id);
  *message_.mutable_original_label_to_duplicate_label() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_label_to_duplicate_label);
}

bool TransformationSplitLoop::IsApplicable(
    opt::IRContext* /*unused*/, const TransformationContext& /*unused*/) const {
  return false;
}

void TransformationSplitLoop::Apply(opt::IRContext* /*unused*/,
                                    TransformationContext* /*unused*/) const {}

protobufs::Transformation TransformationSplitLoop::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_split_loop() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
