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

// In SPIR-V, ids are encoded as uint16_t, this id is guaranteed to be always
// invalid.
const uint32_t kInvalidId = std::numeric_limits<uint32_t>::max();

// Helper class to abstract instruction construction and insertion.
// The instruction builder can preserve the following analyses (specified via
// the constructors):
//   - Def-use analysis
//   - Instruction to block analysis
class InstructionBuilder {
 public:
  using InsertionPointTy = spvtools::ir::BasicBlock::iterator;

  // Creates an InstructionBuilder, all new instructions will be inserted before
  // the instruction |insert_before|.
  InstructionBuilder(
      ir::IRContext* context, ir::Instruction* insert_before,
      ir::IRContext::Analysis preserved_analyses = ir::IRContext::kAnalysisNone)
      : InstructionBuilder(context, context->get_instr_block(insert_before),
                           InsertionPointTy(insert_before),
                           preserved_analyses) {}

  // Creates an InstructionBuilder, all new instructions will be inserted at the
  // end of the basic block |parent_block|.
  InstructionBuilder(
      ir::IRContext* context, ir::BasicBlock* parent_block,
      ir::IRContext::Analysis preserved_analyses = ir::IRContext::kAnalysisNone)
      : InstructionBuilder(context, parent_block, parent_block->end(),
                           preserved_analyses) {}

  // Creates a new selection merge instruction.
  // The id |merge_id| is the merge basic block id.
  ir::Instruction* AddSelectionMerge(
      uint32_t merge_id,
      uint32_t selection_control = SpvSelectionControlMaskNone) {
    std::unique_ptr<ir::Instruction> new_branch_merge(new ir::Instruction(
        GetContext(), SpvOpSelectionMerge, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {merge_id}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_SELECTION_CONTROL,
          {selection_control}}}));
    return AddInstruction(std::move(new_branch_merge));
  }

  // Creates a new branch instruction to |label_id|.
  // Note that the user must make sure the final basic block is
  // well formed.
  ir::Instruction* AddBranch(uint32_t label_id) {
    std::unique_ptr<ir::Instruction> new_branch(new ir::Instruction(
        GetContext(), SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
    return AddInstruction(std::move(new_branch));
  }

  // Creates a new conditional instruction and the associated selection merge
  // instruction if requested.
  // The id |cond_id| is the id of the condition instruction, must be of
  // type bool.
  // The id |true_id| is the id of the basic block to branch to if the condition
  // is true.
  // The id |false_id| is the id of the basic block to branch to if the
  // condition is false.
  // The id |merge_id| is the id of the merge basic block for the selection
  // merge instruction. If |merge_id| equals kInvalidId then no selection merge
  // instruction will be created.
  // The value |selection_control| is the selection control flag for the
  // selection merge instruction.
  // Note that the user must make sure the final basic block is
  // well formed.
  ir::Instruction* AddConditionalBranch(
      uint32_t cond_id, uint32_t true_id, uint32_t false_id,
      uint32_t merge_id = kInvalidId,
      uint32_t selection_control = SpvSelectionControlMaskNone) {
    if (merge_id != kInvalidId) {
      AddSelectionMerge(merge_id, selection_control);
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
  // The vector |incomings| must be a sequence of pairs of <definition id,
  // parent id>.
  ir::Instruction* AddPhi(uint32_t type,
                          const std::vector<uint32_t>& incomings) {
    assert(incomings.size() % 2 == 0 && "A sequence of pairs is expected");
    std::vector<ir::Operand> phi_ops;
    for (size_t i = 0; i < incomings.size(); i++) {
      phi_ops.push_back({SPV_OPERAND_TYPE_ID, {incomings[i]}});
    }
    std::unique_ptr<ir::Instruction> phi_inst(new ir::Instruction(
        GetContext(), SpvOpPhi, type, GetContext()->TakeNextId(), phi_ops));
    return AddInstruction(std::move(phi_inst));
  }

  // Creates a select instruction.
  // |type| must match the types of |true_value| and |false_value|. It is up to
  // the caller to ensure that |cond| is a correct type (bool or vector of
  // bool) for |type|.
  ir::Instruction* AddSelect(uint32_t type, uint32_t cond, uint32_t true_value,
                             uint32_t false_value) {
    std::unique_ptr<ir::Instruction> select(new ir::Instruction(
        GetContext(), SpvOpSelect, type, GetContext()->TakeNextId(),
        std::initializer_list<ir::Operand>{
            {SPV_OPERAND_TYPE_ID, {cond}},
            {SPV_OPERAND_TYPE_ID, {true_value}},
            {SPV_OPERAND_TYPE_ID, {false_value}}}));
    return AddInstruction(std::move(select));
  }

  // Create a composite construct.
  // |type| should be a composite type and the number of elements it has should
  // match the size od |ids|.
  ir::Instruction* AddCompositeConstruct(uint32_t type,
                                         const std::vector<uint32_t>& ids) {
    std::vector<ir::Operand> ops;
    for (auto id : ids) {
      ops.emplace_back(SPV_OPERAND_TYPE_ID,
                       std::initializer_list<uint32_t>{id});
    }
    std::unique_ptr<ir::Instruction> construct(
        new ir::Instruction(GetContext(), SpvOpCompositeConstruct, type,
                            GetContext()->TakeNextId(), ops));
    return AddInstruction(std::move(construct));
  }

  // Inserts the new instruction before the insertion point.
  ir::Instruction* AddInstruction(std::unique_ptr<ir::Instruction>&& insn) {
    ir::Instruction* insn_ptr = &*insert_before_.InsertBefore(std::move(insn));
    UpdateInstrToBlockMapping(insn_ptr);
    UpdateDefUseMgr(insn_ptr);
    return insn_ptr;
  }

  // Returns the insertion point iterator.
  InsertionPointTy GetInsertPoint() { return insert_before_; }

  // Change the insertion point to insert before the instruction
  // |insert_before|.
  void SetInsertPoint(ir::Instruction* insert_before) {
    parent_ = context_->get_instr_block(insert_before);
    insert_before_ = InsertionPointTy(insert_before);
  }

  // Change the insertion point to insert at the end of the basic block
  // |parent_block|.
  void SetInsertPoint(ir::BasicBlock* parent_block) {
    parent_ = parent_block;
    insert_before_ = parent_block->end();
  }

  // Returns the context which instructions are constructed for.
  ir::IRContext* GetContext() const { return context_; }

  // Returns the set of preserved analyses.
  inline ir::IRContext::Analysis GetPreservedAnalysis() const {
    return preserved_analyses_;
  }

 private:
  InstructionBuilder(ir::IRContext* context, ir::BasicBlock* parent,
                     InsertionPointTy insert_before,
                     ir::IRContext::Analysis preserved_analyses)
      : context_(context),
        parent_(parent),
        insert_before_(insert_before),
        preserved_analyses_(preserved_analyses) {
    assert(!(preserved_analyses_ &
             ~(ir::IRContext::kAnalysisDefUse |
               ir::IRContext::kAnalysisInstrToBlockMapping)));
  }

  // Returns true if the users requested to update |analysis|.
  inline bool IsAnalysisUpdateRequested(
      ir::IRContext::Analysis analysis) const {
    return preserved_analyses_ & analysis;
  }

  // Updates the def/use manager if the user requested it. If he did not request
  // an update, this function does nothing.
  inline void UpdateDefUseMgr(ir::Instruction* insn) {
    if (IsAnalysisUpdateRequested(ir::IRContext::kAnalysisDefUse))
      GetContext()->get_def_use_mgr()->AnalyzeInstDefUse(insn);
  }

  // Updates the instruction to block analysis if the user requested it. If he
  // did not request an update, this function does nothing.
  inline void UpdateInstrToBlockMapping(ir::Instruction* insn) {
    if (IsAnalysisUpdateRequested(
            ir::IRContext::kAnalysisInstrToBlockMapping) &&
        parent_)
      GetContext()->set_instr_block(insn, parent_);
  }

  ir::IRContext* context_;
  ir::BasicBlock* parent_;
  InsertionPointTy insert_before_;
  const ir::IRContext::Analysis preserved_analyses_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_IR_BUILDER_H_
