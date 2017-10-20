// Copyright (c) 2017 Google Inc.
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

// This file implements a pass to cleanup the CFG to remove superfluous
// constructs (e.g., unreachable basic blocks, empty control flow structures,
// etc)

#include <queue>
#include <unordered_set>

#include "cfg_cleanup_pass.h"

#include "function.h"
#include "module.h"

namespace spvtools {
namespace opt {

uint32_t CFGCleanupPass::TypeToUndef(uint32_t type_id) {
  const auto uitr = type2undefs_.find(type_id);
  if (uitr != type2undefs_.end()) {
    return uitr->second;
  }

  const uint32_t undefId = TakeNextId();
  std::unique_ptr<ir::Instruction> undef_inst(
      new ir::Instruction(SpvOpUndef, type_id, undefId, {}));
  def_use_mgr_->AnalyzeInstDefUse(&*undef_inst);
  module_->AddGlobalValue(std::move(undef_inst));
  type2undefs_[type_id] = undefId;

  return undefId;
}

// Remove all |phi| operands coming from unreachable blocks (i.e., blocks not in
// |reachable_blocks|).  There are two types of removal that this function can
// perform:
//
// 1- Any operand that comes directly from an unreachable block is completely
//    removed.  Since the block is unreachable, the edge between the unreachable
//    block and the block holding |phi| has been removed.
//
// 2- Any operand that comes via a live block and was defined at an unreachable
//    block gets its value replaced with an OpUndef value. Since the argument
//    was generated in an unreachable block, it no longer exists, so it cannot
//    be referenced.  However, since the value does not reach |phi| directly
//    from the unreachable block, the operand cannot be removed from |phi|.
//    Therefore, we replace the argument value with OpUndef.
//
// For example, in the switch() below, assume that we want to remove the
// argument with value %11 coming from block %41.
//
//          [ ... ]
//          %41 = OpLabel                    <--- Unreachable block
//          %11 = OpLoad %int %y
//          [ ... ]
//                OpSelectionMerge %16 None
//                OpSwitch %12 %16 10 %13 13 %14 18 %15
//          %13 = OpLabel
//                OpBranch %16
//          %14 = OpLabel
//                OpStore %outparm %int_14
//                OpBranch %16
//          %15 = OpLabel
//                OpStore %outparm %int_15
//                OpBranch %16
//          %16 = OpLabel
//          %30 = OpPhi %int %11 %41 %int_42 %13 %11 %14 %11 %15
//
// Since %41 is now an unreachable block, the first operand of |phi| needs to
// be removed completely.  But the operands (%11 %14) and (%11 %15) cannot be
// removed because %14 and %15 are reachable blocks.  Since %11 no longer exist,
// in those arguments, we replace all references to %11 with an OpUndef value.
// This results in |phi| looking like:
//
//           %50 = OpUndef %int
//           [ ... ]
//           %30 = OpPhi %int %int_42 %13 %50 %14 %50 %15
void CFGCleanupPass::RemovePhiOperands(
    ir::Instruction* phi,
    std::unordered_set<ir::BasicBlock*> reachable_blocks) {
  std::vector<ir::Operand> keep_operands;
  uint32_t type_id = 0;
  // The id of an undefined value we've generated.
  uint32_t undef_id = 0;

  // Traverse all the operands in |phi|. Build the new operand vector by adding
  // all the original operands from |phi| except the unwanted ones.
  for (uint32_t i = 0; i < phi->NumOperands();) {
    if (i < 2) {
      // The first two arguments are always preserved.
      keep_operands.push_back(phi->GetOperand(i));
      ++i;
      continue;
    }

    // The remaining Phi arguments come in pairs. Index 'i' contains the
    // variable id, index 'i + 1' is the originating block id.
    assert(i % 2 == 0 && i < phi->NumOperands() - 1 &&
           "malformed Phi arguments");

    ir::BasicBlock *in_block = label2block_[phi->GetSingleWordOperand(i + 1)];
    if (reachable_blocks.find(in_block) == reachable_blocks.end()) {
      // If the incoming block is unreachable, remove both operands as this
      // means that the |phi| has lost an incoming edge.
      i += 2;
      continue;
    }

    // In all other cases, the operand must be kept but may need to be changed.
    uint32_t arg_id = phi->GetSingleWordOperand(i);
    ir::BasicBlock *def_block = def_block_[arg_id];
    if (def_block &&
        reachable_blocks.find(def_block_[arg_id]) == reachable_blocks.end()) {
      // If the current |phi| argument was defined in an unreachable block, it
      // means that this |phi| argument is no longer defined. Replace it with
      // |undef_id|.
      if (!undef_id) {
        type_id = def_use_mgr_->GetDef(arg_id)->type_id();
        undef_id = TypeToUndef(type_id);
      }
      keep_operands.push_back(
          ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {undef_id}));
    } else {
      // Otherwise, the argument comes from a reachable block or from no block
      // at all (meaning that it was defined in the global section of the
      // program).  In both cases, keep the argument intact.
      keep_operands.push_back(phi->GetOperand(i));
    }

    keep_operands.push_back(phi->GetOperand(i + 1));

    i += 2;
  }

  phi->ReplaceOperands(keep_operands);
}

void CFGCleanupPass::RemoveBlock(ir::Function::iterator* bi) {
  auto& rm_block = **bi;

  // Remove instructions from the block.
  rm_block.ForEachInst([&rm_block, this](ir::Instruction* inst) {
    // Note that we do not kill the block label instruction here. The label
    // instruction is needed to identify the block, which is needed by the
    // removal of phi operands.
    if (inst != rm_block.GetLabelInst()) {
      KillNamesAndDecorates(inst);
      def_use_mgr_->KillInst(inst);
    }
  });

  // Remove the label instruction last.
  auto label = rm_block.GetLabelInst();
  KillNamesAndDecorates(label);
  def_use_mgr_->KillInst(label);

  *bi = bi->Erase();
}

bool CFGCleanupPass::RemoveUnreachableBlocks(ir::Function* func) {
  bool modified = false;

  // Mark reachable all blocks reachable from the function's entry block.
  std::unordered_set<ir::BasicBlock*> reachable_blocks;
  std::unordered_set<ir::BasicBlock*> visited_blocks;
  std::queue<ir::BasicBlock*> worklist;
  reachable_blocks.insert(func->entry().get());

  // Initially mark the function entry point as reachable.
  worklist.push(func->entry().get());

  auto mark_reachable = [&reachable_blocks, &visited_blocks, &worklist,
                         this](uint32_t label_id) {
    auto successor = label2block_[label_id];
    if (visited_blocks.count(successor) == 0) {
      reachable_blocks.insert(successor);
      worklist.push(successor);
      visited_blocks.insert(successor);
    }
  };

  // Transitively mark all blocks reachable from the entry as reachable.
  while (!worklist.empty()) {
    ir::BasicBlock* block = worklist.front();
    worklist.pop();

    // All the successors of a live block are also live.
    block->ForEachSuccessorLabel(mark_reachable);

    // All the Merge and ContinueTarget blocks of a live block are also live.
    block->ForMergeAndContinueLabel(mark_reachable);
  }

  // Update operands of Phi nodes that reference unreachable blocks.
  for (auto& block : *func) {
    // If the block is about to be removed, don't bother updating its
    // Phi instructions.
    if (reachable_blocks.count(&block) == 0) {
      continue;
    }

    // If the block is reachable and has Phi instructions, remove all
    // operands from its Phi instructions that reference unreachable blocks.
    // If the block has no Phi instructions, this is a no-op.
    block.ForEachPhiInst(
        [&block, &reachable_blocks, this](ir::Instruction* phi) {
          RemovePhiOperands(phi, reachable_blocks);
        });
  }

  // Erase unreachable blocks.
  for (auto ebi = func->begin(); ebi != func->end();) {
    if (reachable_blocks.count(&*ebi) == 0) {
      RemoveBlock(&ebi);
      modified = true;
    } else {
      ++ebi;
    }
  }

  return modified;
}

bool CFGCleanupPass::CFGCleanup(ir::Function* func) {
  bool modified = false;
  modified |= RemoveUnreachableBlocks(func);
  return modified;
}

void CFGCleanupPass::Initialize(ir::Module* module) {
  // Initialize the DefUse manager. TODO(dnovillo): Re-factor all this into the
  // module or some other context class for the optimizer.
  module_ = module;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  FindNamedOrDecoratedIds();

  // Initialize next unused Id. TODO(dnovillo): Re-factor into the module or
  // some other context class for the optimizer.
  next_id_ = module_->id_bound();

  // Initialize block lookup map.
  label2block_.clear();
  for (auto& fn : *module) {
    for (auto& block : fn) {
      label2block_[block.id()] = &block;

      // Build a map between SSA names to the block they are defined in.
      // TODO(dnovillo): This is expensive and unnecessary if ir::Instruction
      // instances could figure out what basic block they belong to. Remove this
      // once this is possible.
      block.ForEachInst([this, &block](ir::Instruction* inst) {
        uint32_t result_id = inst->result_id();
        if (result_id > 0) {
          def_block_[result_id] = &block;
        }
      });
    }
  }
}

Pass::Status CFGCleanupPass::Process(ir::Module* module) {
  Initialize(module);

  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) { return CFGCleanup(fp); };
  bool modified = ProcessReachableCallTree(pfn, module);
  FinalizeNextId(module_);
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
