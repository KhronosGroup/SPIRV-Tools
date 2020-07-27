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
    const TransformationContext& /* unused */) const {
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
  if (ir_context->cfg()->preds(message_.loop_header_block()).size() == 2) {
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
    opt::IRContext* ir_context,
    TransformationContext* /* transformation_context */) const {
  // Find the loop header.
  opt::BasicBlock* loop_header =
      fuzzerutil::MaybeFindBlock(ir_context, message_.loop_header_block());

  auto dominator_analysis =
      ir_context->GetDominatorAnalysis(loop_header->GetParent());

  uint32_t back_edge_block_id = 0;

  // Update the branching instructions of the out-of-loop predecessors of the
  // header.
  ir_context->get_def_use_mgr()->ForEachUse(
      loop_header->id(),
      [&dominator_analysis, &loop_header, &ir_context, &back_edge_block_id,
       this](opt::Instruction* use_inst, uint32_t use_index) {

        if (dominator_analysis->Dominates(loop_header->GetLabelInst(),
                                          use_inst)) {
          // If |use_inst| is a branch instruction dominated by the header, the
          // block containing it is the back-edge block.
          if (use_inst->IsBranch()) {
            back_edge_block_id = ir_context->get_instr_block(use_inst)->id();
          }
          // References to the header inside the loop should not be updated
          return;
        }

        // If |use_inst| is not a branch or merge instruction, it should not be
        // changed.
        if (!use_inst->IsBranch() &&
            use_inst->opcode() != SpvOpSelectionMerge &&
            use_inst->opcode() != SpvOpLoopMerge) {
          return;
        }

        // Update the reference.
        use_inst->SetOperand(use_index, {message_.fresh_id()});
      });

  // Make a new block for the preheader.
  std::unique_ptr<opt::BasicBlock> preheader = MakeUnique<opt::BasicBlock>(
      std::unique_ptr<opt::Instruction>(new opt::Instruction(
          ir_context, SpvOpLabel, 0, message_.fresh_id(), {})));

  // Update id bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());

  preheader->AddInstruction(std::unique_ptr<opt::Instruction>(
      new opt::Instruction(ir_context, SpvOpBranch, 0, 0,
                           std::initializer_list<opt::Operand>{opt::Operand(
                               spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
                               {loop_header->id()})})));
  loop_header->GetParent()->InsertBasicBlockBefore(std::move(preheader),
                                                   loop_header);

  // Invalidate analyses because the structure of the program changed.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationAddLoopPreheader::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_loop_preheader() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
