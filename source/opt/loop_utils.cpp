// Copyright (c) 2018 Google LLC.
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

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "opt/cfg.h"
#include "opt/ir_builder.h"
#include "opt/ir_context.h"
#include "opt/loop_descriptor.h"
#include "opt/loop_utils.h"

namespace spvtools {
namespace opt {

namespace {
// Return true if |bb| is dominated by at least one block in |exits|
static inline bool DominatesAnExit(
    ir::BasicBlock* bb, const std::unordered_set<ir::BasicBlock*>& exits,
    const opt::DominatorTree& dom_tree) {
  for (ir::BasicBlock* e_bb : exits)
    if (dom_tree.Dominates(bb, e_bb)) return true;
  return false;
}

// Utility class to rewrite out-of-loop uses of an in-loop definition in terms
// of phi instructions to achieve a LCSSA form.
// For a given definition, the class user registers phi instructions using that
// definition in all loop exit blocks by which the definition escapes.
// Then, when rewriting a use of the definition, the rewriter walks the
// paths from the use the loop exits. At each step, it will insert a phi
// instruction to merge the incoming value according to exit blocks definition.
class LCSSARewriter {
 public:
  LCSSARewriter(ir::IRContext* context, const opt::DominatorTree& dom_tree,
                const std::unordered_set<ir::BasicBlock*>& exit_bb,
                const ir::Instruction& def_insn)
      : context_(context),
        cfg_(context_->cfg()),
        dom_tree_(dom_tree),
        insn_type_(def_insn.type_id()),
        exit_bb_(exit_bb) {}

  // Rewrites the use of |def_insn_| by the instruction |user| at the index
  // |operand_index| in terms of phi instruction. This recursively builds new
  // phi instructions from |user| to the loop exit blocks' phis. The use of
  // |def_insn_| in |user| is replaced by the relevant phi instruction at the
  // end of the operation.
  // It is assumed that |user| does not dominates any of the loop exit basic
  // block. This operation does not update the def/use manager, instead it
  // records what needs to be updated. The actual update is performed by
  // UpdateManagers.
  void RewriteUse(ir::BasicBlock* bb, ir::Instruction* user,
                  uint32_t operand_index) {
    assert((user->opcode() != SpvOpPhi || bb != GetParent(user)) &&
           "The root basic block must be the incoming edge if |user| is a phi "
           "instruction");
    assert(
        (user->opcode() == SpvOpPhi || bb == GetParent(user)) &&
        "The root basic block must be the instruction parent if |user| is not "
        "phi instruction");

    const ir::Instruction& new_def = GetOrBuildIncoming(bb->id());

    user->SetOperand(operand_index, {new_def.result_id()});
    rewritten.insert(user);
  }

  // Notifies the addition of a phi node built to close the loop.
  inline void RegisterExitPhi(ir::BasicBlock* bb, ir::Instruction* phi) {
    bb_to_phi[bb->id()] = phi;
    rewritten.insert(phi);
  }

  // In-place update of some managers (avoid full invalidation).
  inline void UpdateManagers() {
    opt::analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();
    // Register all new definitions.
    for (ir::Instruction* insn : rewritten) {
      def_use_mgr->AnalyzeInstDef(insn);
    }
    // Register all new uses.
    for (ir::Instruction* insn : rewritten) {
      def_use_mgr->AnalyzeInstUse(insn);
    }
  }

 private:
  // Return the basic block that |instr| belongs to.
  ir::BasicBlock* GetParent(ir::Instruction* instr) {
    return context_->get_instr_block(instr);
  }

  // Return the new def to use for the basic block |bb_id|.
  // If |bb_id| does not have a suitable def to use then we:
  //   - return the common def used by all predecessors;
  //   - if there is no common def, then we build a new phi instr at the
  //     beginning of |bb_id| and return this new instruction.
  const ir::Instruction& GetOrBuildIncoming(uint32_t bb_id) {
    assert(cfg_->block(bb_id) != nullptr && "Unknown basic block");

    ir::Instruction*& incoming_phi = bb_to_phi[bb_id];
    if (incoming_phi) {
      return *incoming_phi;
    }

    // Check if one of the loop exit basic block dominates |bb_id|.
    for (const ir::BasicBlock* e_bb : exit_bb_) {
      if (dom_tree_.Dominates(e_bb->id(), bb_id)) {
        incoming_phi = bb_to_phi[e_bb->id()];
        assert(incoming_phi && "No closing phi node ?");
        return *incoming_phi;
      }
    }

    // Process parents, they will returns their suitable phi.
    // If they are all the same, this means this basic block is dominated by a
    // common block, so we won't need to build a phi instruction.
    std::vector<uint32_t> incomings;
    for (uint32_t pred_id : cfg_->preds(bb_id)) {
      incomings.push_back(GetOrBuildIncoming(pred_id).result_id());
      incomings.push_back(pred_id);
    }
    uint32_t first_id = incomings.front();
    size_t idx = 0;
    for (; idx < incomings.size(); idx += 2)
      if (first_id != incomings[idx]) break;

    if (idx >= incomings.size()) {
      incoming_phi = bb_to_phi[incomings[1]];
      return *incoming_phi;
    }

    // We have at least 2 definitions to merge, so we need a phi instruction.
    ir::BasicBlock* block = cfg_->block(bb_id);

    opt::InstructionBuilder<> builder(context_, &*block->begin());
    incoming_phi = builder.AddPhi(insn_type_, incomings);

    rewritten.insert(incoming_phi);

    return *incoming_phi;
  }

  ir::IRContext* context_;
  ir::CFG* cfg_;
  const opt::DominatorTree& dom_tree_;
  uint32_t insn_type_;
  std::unordered_map<uint32_t, ir::Instruction*> bb_to_phi;
  std::unordered_set<ir::Instruction*> rewritten;
  const std::unordered_set<ir::BasicBlock*>& exit_bb_;
};
}  // namespace

void LoopUtils::CreateLoopDedicateExits() {
  ir::Function* function = loop_->GetHeaderBlock()->GetParent();
  ir::LoopDescriptor& loop_desc = *context_->GetLoopDescriptor(function);
  ir::CFG& cfg = *context_->cfg();
  opt::analysis::DefUseManager* def_use_mgr = context_->get_def_use_mgr();

  constexpr ir::IRContext::Analysis PreservedAnalyses =
      ir::IRContext::kAnalysisDefUse |
      ir::IRContext::kAnalysisInstrToBlockMapping;

  // Gathers the set of basic block that are not in this loop and have at least
  // one predecessor in the loop and one not in the loop.
  std::unordered_set<uint32_t> exit_bb_set;
  loop_->GetExitBlocks(context_, &exit_bb_set);

  std::unordered_set<ir::BasicBlock*> new_loop_exits;
  bool made_change = false;
  // For each block, we create a new one that gathers all branches from
  // the loop and fall into the block.
  for (uint32_t non_dedicate_id : exit_bb_set) {
    ir::BasicBlock* non_dedicate = cfg.block(non_dedicate_id);
    const std::vector<uint32_t>& bb_pred = cfg.preds(non_dedicate_id);
    // Ignore the block if:
    //   - all the predecessors are in the loop;
    //   - and has an unconditional branch;
    //   - and any other instructions are phi.
    if (non_dedicate->tail()->opcode() == SpvOpBranch) {
      if (std::all_of(bb_pred.begin(), bb_pred.end(), [this](uint32_t id) {
            return loop_->IsInsideLoop(id);
          })) {
        ir::BasicBlock::iterator it = non_dedicate->tail();
        if (it == non_dedicate->begin() || (--it)->opcode() == SpvOpPhi) {
          new_loop_exits.insert(non_dedicate);
          continue;
        }
      }
    }

    made_change = true;
    ir::Function::iterator insert_pt = function->begin();
    for (; insert_pt != function->end() && &*insert_pt != non_dedicate;
         ++insert_pt) {
    }
    assert(insert_pt != function->end() && "Basic Block not found");

    // Create the dedicate exit basic block.
    ir::BasicBlock& exit = *insert_pt.InsertBefore(
        std::unique_ptr<ir::BasicBlock>(new ir::BasicBlock(
            std::unique_ptr<ir::Instruction>(new ir::Instruction(
                context_, SpvOpLabel, 0, context_->TakeNextId(), {})))));
    exit.SetParent(function);

    // Redirect in loop predecessors to |exit| block.
    for (uint32_t exit_pred_id : bb_pred) {
      if (loop_->IsInsideLoop(exit_pred_id)) {
        ir::BasicBlock* pred_block = cfg.block(exit_pred_id);
        pred_block->ForEachSuccessorLabel([non_dedicate, &exit](uint32_t* id) {
          if (*id == non_dedicate->id()) *id = exit.id();
        });
        // Update the CFG.
        // |non_dedicate|'s predecessor list will be updated at the end of the
        // loop.
        cfg.RegisterBlock(pred_block);
      }
    }

    // Register the label to the def/use manager, requires for the phi patching.
    def_use_mgr->AnalyzeInstDefUse(exit.GetLabelInst());
    context_->set_instr_block(exit.GetLabelInst(), &exit);

    // Patch the phi nodes.
    opt::InstructionBuilder<PreservedAnalyses> builder(context_,
                                                       &*exit.begin());
    non_dedicate->ForEachPhiInst([&builder, &exit, def_use_mgr,
                                  this](ir::Instruction* phi) {
      // New phi operands for this instruction.
      std::vector<uint32_t> new_phi_op;
      // Phi operands for the dedicated exit block.
      std::vector<uint32_t> exit_phi_op;
      for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
        uint32_t def_id = phi->GetSingleWordInOperand(i);
        uint32_t incoming_id = phi->GetSingleWordInOperand(i + 1);
        if (loop_->IsInsideLoop(incoming_id)) {
          exit_phi_op.push_back(def_id);
          exit_phi_op.push_back(incoming_id);
        } else {
          new_phi_op.push_back(def_id);
          new_phi_op.push_back(incoming_id);
        }
      }

      // Build the new phi instruction dedicated exit block.
      ir::Instruction* exit_phi = builder.AddPhi(phi->type_id(), exit_phi_op);
      // Build the new incoming branch.
      new_phi_op.push_back(exit_phi->result_id());
      new_phi_op.push_back(exit.id());
      // Rewrite operands.
      uint32_t idx = 0;
      for (; idx < new_phi_op.size(); idx++)
        phi->SetInOperand(idx, {new_phi_op[idx]});
      // Remove extra operands, from last to first (more efficient).
      for (uint32_t j = phi->NumInOperands() - 1; j >= idx; j--)
        phi->RemoveInOperand(j);
      // Update the def/use manager for this |phi|.
      def_use_mgr->AnalyzeInstUse(phi);
    });
    // now jump from our dedicate basic block to the old exit.
    builder.AddBranch(non_dedicate->id());
    // Update the CFG.
    cfg.RegisterBlock(&exit);
    cfg.RemoveNonExistingEdges(non_dedicate->id());
    new_loop_exits.insert(&exit);
    // If non_dedicate is in a loop, add the new dedicated exit in that loop.
    if (ir::Loop* parent_loop = loop_desc[non_dedicate])
      parent_loop->AddBasicBlock(&exit);
  }

  if (new_loop_exits.size() == 1) {
    loop_->SetMergeBlock(*new_loop_exits.begin());
  }

  if (made_change) {
    context_->InvalidateAnalysesExceptFor(
        PreservedAnalyses | ir::IRContext::kAnalysisCFG |
        ir::IRContext::Analysis::kAnalysisLoopAnalysis);
  }
}

void LoopUtils::MakeLoopClosedSSA() {
  CreateLoopDedicateExits();

  ir::Function* function = loop_->GetHeaderBlock()->GetParent();
  ir::CFG& cfg = *context_->cfg();
  opt::DominatorTree& dom_tree =
      context_->GetDominatorAnalysis(function, cfg)->GetDomTree();

  opt::analysis::DefUseManager* def_use_manager = context_->get_def_use_mgr();
  std::unordered_set<ir::BasicBlock*> exit_bb;
  for (uint32_t bb_id : loop_->GetBlocks()) {
    ir::BasicBlock* bb = cfg.block(bb_id);
    bb->ForEachSuccessorLabel([&exit_bb, &cfg, this](uint32_t succ) {
      if (!loop_->IsInsideLoop(succ)) exit_bb.insert(cfg.block(succ));
    });
  }

  for (uint32_t bb_id : loop_->GetBlocks()) {
    ir::BasicBlock* bb = cfg.block(bb_id);
    // If bb does not dominate an exit block, then it cannot have escaping defs.
    if (!DominatesAnExit(bb, exit_bb, dom_tree)) continue;
    for (ir::Instruction& inst : *bb) {
      std::unordered_set<ir::BasicBlock*> processed_exit;
      LCSSARewriter rewriter(context_, dom_tree, exit_bb, inst);
      def_use_manager->ForEachUse(
          &inst, [&rewriter, &exit_bb, &processed_exit, &inst, &dom_tree, &cfg,
                  this](ir::Instruction* use, uint32_t operand_index) {
            if (loop_->IsInsideLoop(use)) return;

            ir::BasicBlock* use_parent = context_->get_instr_block(use);
            assert(use_parent);
            if (use->opcode() == SpvOpPhi) {
              // If the use is a Phi instruction and the incoming block is
              // coming from the loop, then that's consistent with LCSSA form.
              if (exit_bb.count(use_parent)) {
                rewriter.RegisterExitPhi(use_parent, use);
                return;
              } else {
                // That's not an exit block, but the user is a phi instruction.
                // Consider the incoming branch only: |use_parent| must be
                // dominated by one of the exit block.
                use_parent = context_->get_instr_block(
                    use->GetSingleWordOperand(operand_index + 1));
              }
            }

            for (ir::BasicBlock* e_bb : exit_bb) {
              if (processed_exit.count(e_bb)) continue;
              processed_exit.insert(e_bb);

              // If the current exit basic block does not dominate |use| then
              // |inst| does not escape through |e_bb|.
              if (!dom_tree.Dominates(e_bb, use_parent)) continue;

              opt::InstructionBuilder<> builder(context_, &*e_bb->begin());
              const std::vector<uint32_t>& preds = cfg.preds(e_bb->id());
              std::vector<uint32_t> incoming;
              incoming.reserve(preds.size() * 2);
              for (uint32_t pred_id : preds) {
                incoming.push_back(inst.result_id());
                incoming.push_back(pred_id);
              }
              rewriter.RegisterExitPhi(
                  e_bb, builder.AddPhi(inst.type_id(), incoming));
            }

            // Rewrite the use. Note that this call does not invalidate the
            // def/use manager. So this operation is safe.
            rewriter.RewriteUse(use_parent, use, operand_index);
          });
      rewriter.UpdateManagers();
    }
  }

  context_->InvalidateAnalysesExceptFor(
      ir::IRContext::Analysis::kAnalysisDefUse |
      ir::IRContext::Analysis::kAnalysisCFG |
      ir::IRContext::Analysis::kAnalysisDominatorAnalysis |
      ir::IRContext::Analysis::kAnalysisLoopAnalysis);
}

}  // namespace opt
}  // namespace spvtools
