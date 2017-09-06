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

void CFGCleanupPass::RemoveFromReachedPhiOperands(const ir::BasicBlock& block,
                                                  ir::Instruction* inst) {
  uint32_t inst_id = inst->result_id();
  if (inst_id == 0) {
    return;
  }

  analysis::UseList* uses = def_use_mgr_->GetUses(inst_id);
  if (uses == nullptr) {
    return;
  }

  for (auto u : *uses) {
    if (u.inst->opcode() != SpvOpPhi) {
      continue;
    }

    ir::Instruction* phi_inst = u.inst;
    std::vector<ir::Operand> keep_operands;
    for (uint32_t i = 0; i < phi_inst->NumOperands(); i++) {
      const ir::Operand& var_op = phi_inst->GetOperand(i);
      if (i >= 2 && i < phi_inst->NumOperands() - 1) {
        // PHI arguments start at index 2. Each argument consists of two
        // operands: the variable id and the originating block id.
        const ir::Operand& block_op = phi_inst->GetOperand(i + 1);
        assert(var_op.words.size() == 1 && block_op.words.size() == 1 &&
               "Phi operands should have exactly one word in them.");
        uint32_t var_id = var_op.words.front();
        uint32_t block_id = block_op.words.front();
        if (var_id == inst_id && block_id == block.id()) {
          i++;
          continue;
        }
      }

      keep_operands.push_back(var_op);
    }

    phi_inst->ReplaceOperands(keep_operands);
  }
}

void CFGCleanupPass::RemoveBlock(ir::Function::iterator* bi) {
  auto& block = **bi;
  block.ForEachInst([&block, this](ir::Instruction* inst) {
    // Note that we do not kill the block label instruction here. The label
    // instruction is needed to identify the block, which is needed by the
    // removal of PHI operands.
    if (inst != block.GetLabelInst()) {
      RemoveFromReachedPhiOperands(block, inst);
      KillNamesAndDecorates(inst);
      def_use_mgr_->KillInst(inst);
    }
  });

  // Remove the label instruction last.
  def_use_mgr_->KillInst(block.GetLabelInst());

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
  // Initialize the DefUse manager.
  module_ = module;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  FindNamedOrDecoratedIds();

  // Initialize block lookup map.
  label2block_.clear();
  for (auto& fn : *module) {
    for (auto& block : fn) {
      label2block_[block.id()] = &block;
    }
  }
}

Pass::Status CFGCleanupPass::Process(ir::Module* module) {
  Initialize(module);

  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) { return CFGCleanup(fp); };
  bool modified = ProcessReachableCallTree(pfn, module);
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
