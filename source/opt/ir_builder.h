// Copyright (c) 2018 Google Inc.
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

#ifndef LIBSPIRV_OPT_IR_BUILDER_H_
#define LIBSPIRV_OPT_IR_BUILDER_H_

#include "opt/basic_block.h"
#include "opt/instruction.h"
#include "opt/ir_context.h"

namespace spvtools {
namespace opt {

// In SPIR-V, ids are encoded as uint16_t, this id is guarantied to be always
// invalid.
constexpr uint32_t kInvalidId = (uint32_t)-1;

// Helper class to abstract instruction construction and insertion.
// |AnalysisToPreserve| ask the InstructionBuilder to preserve requested
// analysis.
// Supported analysis:
//   - Def-use analysis
template <ir::IRContext::Analysis AnalysisToPreserve =
              ir::IRContext::kAnalysisNone>
class InstructionBuilder {
  static_assert(!(AnalysisToPreserve & ~ir::IRContext::kAnalysisDefUse),
                "Only Def-use analysis update is supported");

 public:
  using InsertionPointTy = spvtools::ir::BasicBlock::iterator;

  InstructionBuilder(ir::IRContext* context, InsertionPointTy insert_before)
      : context_(context), insert_before_(insert_before) {}

  // Creates a new selection merge instruction.
  // The id |merge_id| is the merge basic block id.
  ir::Instruction* AddSelectionMerge(uint32_t merge_id) {
    std::unique_ptr<ir::Instruction> new_branch_merge(new ir::Instruction(
        GetContext(), SpvOpSelectionMerge, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {merge_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
    return AddInstruction(std::move(new_branch_merge));
  }

  // Creates a new branch instruction to |label_id|.
  // Note that the user is responsible of making sure the final basic block is
  // well formed.
  ir::Instruction* AddBranch(uint32_t label_id) {
    std::unique_ptr<ir::Instruction> new_branch(new ir::Instruction(
        GetContext(), SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
    return AddInstruction(std::move(new_branch));
  }

  // Creates a new conditional instruction.
  // The id |cond_id| is the condition, must be of type bool.
  // The id |true_id| is the basic block id to branch to is the condition is
  // true.
  // The id |false_id| is the basic block id to branch to is the condition is
  // false.
  // The id |merge_id| is the merge basic block id (for structured CFG), if
  // |merge_id| equals kInvalidId then no
  // Note that the user is responsible of making sure the final basic block is
  // well formed.
  ir::Instruction* AddBranchCond(uint32_t cond_id, uint32_t true_id,
                                 uint32_t false_id,
                                 uint32_t merge_id = kInvalidId) {
    if (merge_id != kInvalidId) {
      AddSelectionMerge(merge_id);
    }
    std::unique_ptr<ir::Instruction> new_branch(new ir::Instruction(
        GetContext(), SpvOpBranchConditional, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
    return AddInstruction(std::move(new_branch));
  }

  // Creates a phi instruction.
  // The id |type| must be the id of the phi instruction's type.
  // The vector |incomings| must be a sequence of pairs of <def id, parent
  // id>.
  ir::Instruction* AddPhi(uint32_t type,
                          const std::vector<uint32_t>& incomings) {
    assert(incomings.size() % 2 == 0 && "A sequence of pairs is expected");
    std::vector<ir::Operand> phi_ops;
    for (size_t i = 0; i < incomings.size(); i += 2) {
      phi_ops.push_back({SPV_OPERAND_TYPE_ID, {incomings[i]}});
      phi_ops.push_back({SPV_OPERAND_TYPE_ID, {incomings[i + 1]}});
    }
    std::unique_ptr<ir::Instruction> phi_inst(new ir::Instruction(
        GetContext(), SpvOpPhi, type, GetContext()->TakeNextId(), phi_ops));
    return AddInstruction(std::move(phi_inst));
  }

  // Inserts the new instruction before the insertion point.
  ir::Instruction* AddInstruction(std::unique_ptr<ir::Instruction>&& insn) {
    ir::Instruction* insn_ptr = &*insert_before_.InsertBefore(std::move(insn));
    UpdateDefUseMgr(insn_ptr);
    return insn_ptr;
  }

  // Returns the insertion point iterator.
  InsertionPointTy GetInsertPoint() { return insert_before_; }

  // Returns the context which instructions are constructed for.
  ir::IRContext* GetContext() const { return context_; }

 private:
  // Returns true if the users requested to update an analysis.
  inline static constexpr bool IsAnalysisUpdateRequested(
      ir::IRContext::Analysis analysis) {
    return AnalysisToPreserve & analysis;
  }

  // Updates the def/use manager if the user requested it. If he did not request
  // an update, this function does nothing.
  inline void UpdateDefUseMgr(ir::Instruction* insn) {
    if (IsAnalysisUpdateRequested(ir::IRContext::kAnalysisDefUse))
      GetContext()->get_def_use_mgr()->AnalyzeInstDefUse(insn);
  }

  ir::IRContext* context_;
  InsertionPointTy insert_before_;
};

}  // namespace spvtools
}  // namespace opt

#endif  // LIBSPIRV_OPT_IR_BUILDER_H_
