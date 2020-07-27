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

#include "transformation_add_loop_preheader.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace fuzz {
TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    const protobufs::TransformationAddLoopPreheader& message)
    : message_(message) {}

TransformationAddLoopPreheader::TransformationAddLoopPreheader(
    uint32_t loop_header_block, uint32_t fresh_id,
    std::vector<uint32_t> phi_id) {
  message_.set_loop_header_block(loop_header_block);
  message_.set_fresh_id(fresh_id);
  for (auto id : phi_id) {
    message_.add_phi_id(id);
  }
}

bool TransformationAddLoopPreheader::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // |message_.loop_header_block()| must be the id of a loop header block.
  opt::BasicBlock* loop_header_block =
      fuzzerutil::MaybeFindBlock(ir_context, message_.loop_header_block());
  if (!loop_header_block || !loop_header_block->IsLoopHeader()) {
    return false;
  }

  // The id for the preheader must actually be fresh.
  std::set<uint32_t> used_ids;
  if (!CheckIdIsFreshAndNotUsedByThisTransformation(message_.fresh_id(),
                                                    ir_context, &used_ids)) {
    return false;
  }

  // If the block only has one predecessor outside of the loop (and thus 2 in
  // total), then no additional fresh ids are necessary.
  if (ir_context->cfg()->preds(message_.fresh_id()).size() == 2) {
    return true;
  }

  // Count the number of OpPhi instructions.
  int32_t num_phi_insts = 0;
  loop_header_block->ForEachPhiInst(
      [&num_phi_insts](opt::Instruction* /* unused */) { num_phi_insts++; });

  // There must be enough fresh ids for the OpPhi instructions.
  if (num_phi_insts > message_.phi_id_size()) {
    return false;
  }

  // Check that the needed ids are fresh and distinct.
  for (int32_t i = 0; i < num_phi_insts; i++) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(message_.phi_id(i),
                                                      ir_context, &used_ids)) {
      return false;
    }
  }

  return true;
}

void TransformationAddLoopPreheader::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationAddLoopPreheader::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_loop_preheader() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
