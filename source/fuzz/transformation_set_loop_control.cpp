// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_set_loop_control.h"

namespace spvtools {
namespace fuzz {

TransformationSetLoopControl::TransformationSetLoopControl(
        const spvtools::fuzz::protobufs::TransformationSetLoopControl& message)
        : message_(message) {}

TransformationSetLoopControl::TransformationSetLoopControl(
        uint32_t block_id, uint32_t loop_control, uint32_t peel_count, uint32_t partial_count) {
  message_.set_block_id(block_id);
  message_.set_loop_control(loop_control);
  message_.set_peel_count(peel_count);
  message_.set_partial_count(partial_count);
}

bool TransformationSetLoopControl::IsApplicable(
        opt::IRContext* context, const FactManager& /*unused*/) const {
  assert(false);
  if (auto block = context->get_instr_block(message_.block_id())) {
    if (auto merge_inst = block->GetMergeInst()) {
      return merge_inst->opcode() == SpvOpLoopMerge;
    }
  }
  // Either the block did not exit, or did not end with OpLoopMerge.
  return false;
}

void TransformationSetLoopControl::Apply(opt::IRContext* context,
                                              FactManager* /*unused*/) const {
  context->get_instr_block(message_.block_id())
          ->GetMergeInst()
          ->SetInOperand(1, {message_.loop_control()});
}

protobufs::Transformation TransformationSetLoopControl::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_set_loop_control() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
