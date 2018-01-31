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
                ir::BasicBlock* merge_block)
      : context_(context),
        cfg_(context_->cfg()),
        dom_tree_(dom_tree),
        exit_bb_(exit_bb),
        merge_block_id_(merge_block ? merge_block->id() : 0) {}

  struct UseRewriter {
    explicit UseRewriter(LCSSARewriter* base, const ir::Instruction& def_insn)
        : base_(base), insn_type_(def_insn.type_id()) {}
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
      assert(
          (user->opcode() != SpvOpPhi || bb != GetParent(user)) &&
          "The root basic block must be the incoming edge if |user| is a phi "
          "instruction");
      assert((user->opcode() == SpvOpPhi || bb == GetParent(user)) &&
             "The root basic block must be the instruction parent if |user| is "
             "not "
             "phi instruction");

      ir::Instruction* new_def = GetOrBuildIncoming(bb->id());

      user->SetOperand(operand_index, {new_def->result_id()});
      rewritten_.insert(user);
    }

    // Notifies the addition of a phi node built to close the loop.
    inline void RegisterExitPhi(ir::BasicBlock* bb, ir::Instruction* phi) {
      bb_to_phi_[bb->id()] = phi;
      rewritten_.insert(phi);
    }

    // In-place update of some managers (avoid full invalidation).
    inline void UpdateManagers() {
      opt::analysis::DefUseManager* def_use_mgr =
          base_->context_->get_def_use_mgr();
      // Register all new definitions.
      for (ir::Instruction* insn : rewritten_) {
        def_use_mgr->AnalyzeInstDef(insn);
      }
      // Register all new uses.
      for (ir::Instruction* insn : rewritten_) {
        def_use_mgr->AnalyzeInstUse(insn);
      }
    }

   private:
    // Return the basic block that |instr| belongs to.
    ir::BasicBlock* GetParent(ir::Instruction* instr) {
      return base_->context_->get_instr_block(instr);
    }

    // Return the new def to use for the basic block |bb_id|.
    // If |bb_id| does not have a suitable def to use then we:
    //   - return the common def used by all predecessors;
    //   - if there is no common def, then we build a new phi instr at the
    //     beginning of |bb_id| and return this new instruction.
    ir::Instruction* GetOrBuildIncoming(uint32_t bb_id) {
      assert(base_->cfg_->block(bb_id) != nullptr && "Unknown basic block");

      ir::Instruction*& incoming_phi = bb_to_phi_[bb_id];
      if (incoming_phi) {
        return incoming_phi;
      }

      // Get the block that defines the value to use for each predecessor.
      // If the vector has 1 value, then it means that this block does not need
      // to build a phi instruction unless |bb_id| is the loop merge block.
      const std::vector<uint32_t>& defining_blocks =
          base_->GetDefiningBlocks(bb_id);

      // Special case for structured loops: merge block might be different from
      // the exit block set. To maintain structured properties it will ease
      // transformations if the merge block also holds a phi instruction like
      // the exit ones.
      if (defining_blocks.size() > 1 || bb_id == base_->merge_block_id_) {
        assert(bb_id != base_->merge_block_id_ &&
               defining_blocks.size() == base_->cfg_->preds(bb_id).size());
        assert(bb_id == base_->merge_block_id_ &&
               (defining_blocks.size() == base_->cfg_->preds(bb_id).size() ||
                defining_blocks.size() == 1));
        std::vector<uint32_t> incomings;
        const std::vector<uint32_t>& bb_preds = base_->cfg_->preds(bb_id);
        for (size_t i = 0; i < bb_preds.size(); i++) {
          uint32_t def_bb = i < incomings.size() ? incomings[i] : incomings[0];
          incomings.push_back(GetOrBuildIncoming(def_bb)->result_id());
          incomings.push_back(bb_preds[i]);
        }
        opt::InstructionBuilder<> builder(base_->context_,
                                          &*base_->cfg_->block(bb_id)->begin());
        incoming_phi = builder.AddPhi(insn_type_, incomings);

        rewritten_.insert(incoming_phi);
      } else {
        incoming_phi = GetOrBuildIncoming(defining_blocks[0]);
      }

      return incoming_phi;
    }

    LCSSARewriter* base_;
    uint32_t insn_type_;
    std::unordered_map<uint32_t, ir::Instruction*> bb_to_phi_;
    std::unordered_set<ir::Instruction*> rewritten_;
  };

 private:
  // Return the new def to use for the basic block |bb_id|.
  // If |bb_id| does not have a suitable def to use then we:
  //   - return the common def used by all predecessors;
  //   - if there is no common def, then we build a new phi instr at the
  //     beginning of |bb_id| and return this new instruction.
  const std::vector<uint32_t>& GetDefiningBlocks(uint32_t bb_id) {
    assert(cfg_->block(bb_id) != nullptr && "Unknown basic block");
    std::vector<uint32_t>& defining_blocks = bb_to_defining_blocks_[bb_id];

    if (defining_blocks.size()) return defining_blocks;

    // Check if one of the loop exit basic block dominates |bb_id|.
    for (const ir::BasicBlock* e_bb : exit_bb_) {
      if (dom_tree_.Dominates(e_bb->id(), bb_id)) {
        defining_blocks.push_back(e_bb->id());
        return defining_blocks;
      }
    }

    // Process parents, they will returns their suitable blocks.
    // If they are all the same, this means this basic block is dominated by a
    // common block, so we won't need to build a phi instruction.
    for (uint32_t pred_id : cfg_->preds(bb_id)) {
      const std::vector<uint32_t>& pred_blocks = GetDefiningBlocks(pred_id);
      if (pred_blocks.size() == 1)
        defining_blocks.push_back(pred_blocks[0]);
      else
        defining_blocks.push_back(pred_id);
    }
    assert(defining_blocks.size());
    if (std::all_of(defining_blocks.begin(), defining_blocks.end(),
                    [&defining_blocks](uint32_t id) {
                      return id == defining_blocks[0];
                    })) {
      // No need for a phi.
      defining_blocks.resize(1);
    }

    return defining_blocks;
  }

  ir::IRContext* context_;
  ir::CFG* cfg_;
  const opt::DominatorTree& dom_tree_;
  const std::unordered_set<ir::BasicBlock*>& exit_bb_;
  uint32_t merge_block_id_;
  // This map represent the set of known paths. For each key, the vector
  // represent the set of blocks holding the definition to be used to build the
  // phi instruction.
  // If the vector has 0 value, then the path is unknown yet, and must be built.
  // If the vector has 1 value, then the value defined by that basic block
  //   should be used.
  // If the vector has more than 1 value, then a phi node must be created, the
  //   basic block ordering is the same as the predecessor ordering.
  std::unordered_map<uint32_t, std::vector<uint32_t>> bb_to_defining_blocks_;
};

// Make the set |blocks| closed SSA. The set is closed SSA if all the uses
// outside the set are phi instructions in exiting basic block set (hold by
// |lcssa_rewriter|).
inline void MakeSetClosedSSA(ir::IRContext* context, ir::Function* function,
                             const std::unordered_set<uint32_t>& blocks,
                             const std::unordered_set<ir::BasicBlock*>& exit_bb,
                             LCSSARewriter* lcssa_rewriter) {
  ir::CFG& cfg = *context->cfg();
  opt::DominatorTree& dom_tree =
      context->GetDominatorAnalysis(function, cfg)->GetDomTree();
  opt::analysis::DefUseManager* def_use_manager = context->get_def_use_mgr();

  for (uint32_t bb_id : blocks) {
    ir::BasicBlock* bb = cfg.block(bb_id);
    // If bb does not dominate an exit block, then it cannot have escaping defs.
    if (!DominatesAnExit(bb, exit_bb, dom_tree)) continue;
    for (ir::Instruction& inst : *bb) {
      std::unordered_set<ir::BasicBlock*> processed_exit;
      LCSSARewriter::UseRewriter rewriter(lcssa_rewriter, inst);
      def_use_manager->ForEachUse(
          &inst, [&blocks, &rewriter, &exit_bb, &processed_exit, &inst, &cfg,
                  context](ir::Instruction* use, uint32_t operand_index) {
            ir::BasicBlock* use_parent = context->get_instr_block(use);
            assert(use_parent);
            if (blocks.count(use_parent->id())) return;

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
                use_parent = context->get_instr_block(
                    use->GetSingleWordOperand(operand_index + 1));
              }
            }

            for (ir::BasicBlock* e_bb : exit_bb) {
              if (processed_exit.count(e_bb)) continue;
              processed_exit.insert(e_bb);

              opt::InstructionBuilder<> builder(context, &*e_bb->begin());
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
}

}  // namespace

void LoopUtils::CreateLoopDedicatedExits() {
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
    // Ignore the block if all the predecessors are in the loop.
    if (std::all_of(bb_pred.begin(), bb_pred.end(),
                    [this](uint32_t id) { return loop_->IsInsideLoop(id); })) {
      new_loop_exits.insert(non_dedicate);
      continue;
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
  CreateLoopDedicatedExits();

  ir::Function* function = loop_->GetHeaderBlock()->GetParent();
  ir::CFG& cfg = *context_->cfg();
  opt::DominatorTree& dom_tree =
      context_->GetDominatorAnalysis(function, cfg)->GetDomTree();

  std::unordered_set<ir::BasicBlock*> exit_bb;
  {
    std::unordered_set<uint32_t> exit_bb_id;
    loop_->GetExitBlocks(context_, &exit_bb_id);
    for (uint32_t bb_id : exit_bb_id) {
      exit_bb.insert(cfg.block(bb_id));
    }
  }

  LCSSARewriter lcssa_rewriter(context_, dom_tree, exit_bb,
                               loop_->GetMergeBlock());
  MakeSetClosedSSA(context_, function, loop_->GetBlocks(), exit_bb,
                   &lcssa_rewriter);

  // Make sure all defs post-dominated by the merge block have their last use no
  // further than the merge block.
  if (loop_->GetMergeBlock()) {
    std::unordered_set<uint32_t> merging_bb_id;
    loop_->GetMergingBlocks(context_, &merging_bb_id);
    merging_bb_id.erase(loop_->GetMergeBlock()->id());
    // Reset the exit set, now only the merge block is the exit.
    exit_bb.clear();
    exit_bb.insert(loop_->GetMergeBlock());
    // LCSSARewriter is reusable here only because it forces the creation of a
    // phi instruction in the merge block.
    MakeSetClosedSSA(context_, function, merging_bb_id, exit_bb,
                     &lcssa_rewriter);
  }

  context_->InvalidateAnalysesExceptFor(
      ir::IRContext::Analysis::kAnalysisDefUse |
      ir::IRContext::Analysis::kAnalysisCFG |
      ir::IRContext::Analysis::kAnalysisDominatorAnalysis |
      ir::IRContext::Analysis::kAnalysisLoopAnalysis);
}

}  // namespace opt
}  // namespace spvtools
