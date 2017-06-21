// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "dead_branch_elim_pass.h"

#include "cfa.h"
#include "iterator.h"

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvBranchCondConditionalId = 0;
static const int kSpvBranchCondTrueLabId = 1;
static const int kSpvBranchCondFalseLabId = 2;
static const int kSpvSelectionMergeMergeBlockId = 0;
static const int kSpvPhiVal0Id = 0;
static const int kSpvPhiLab0Id = 1;
static const int kSpvPhiVal1Id = 2;
static const int kSpvLoopMergeMergeBlockId = 0;

namespace spvtools {
namespace opt {

uint32_t DeadBranchElimPass::MergeBlockIdIfAny(
    const ir::BasicBlock& blk) const {
  auto merge_ii = blk.cend();
  --merge_ii;
  uint32_t mbid = 0;
  if (merge_ii != blk.cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge)
      mbid = merge_ii->GetSingleWordOperand(kSpvLoopMergeMergeBlockId);
    else if (merge_ii->opcode() == SpvOpSelectionMerge)
      mbid = merge_ii->GetSingleWordOperand(kSpvSelectionMergeMergeBlockId);
  }
  return mbid;
}

void DeadBranchElimPass::ComputeStructuredSuccessors(ir::Function* func) {
  // If header, make merge block first successor.
  for (auto& blk : *func) {
    uint32_t mbid = MergeBlockIdIfAny(blk);
    if (mbid != 0)
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
    // add true successors
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

DeadBranchElimPass::GetBlocksFunction
DeadBranchElimPass::StructuredSuccessorsFunction() {
  return [this](const ir::BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };
}

void DeadBranchElimPass::ComputeStructuredOrder(
    ir::Function* func, std::list<ir::BasicBlock*>* order) {
  // Compute structured successors and do DFS
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
    &*func->begin(), StructuredSuccessorsFunction(), ignore_block,
    [&](cbb_ptr b) { order->push_front(const_cast<ir::BasicBlock*>(b)); }, ignore_edge);
}

void DeadBranchElimPass::GetConstCondition(
    uint32_t condId, bool* condVal, bool* condIsConst) {
  ir::Instruction* cInst = def_use_mgr_->GetDef(condId);
  switch (cInst->opcode()) {
    case SpvOpConstantFalse: {
      *condVal = false;
      *condIsConst = true;
    } break;
    case SpvOpConstantTrue: {
      *condVal = true;
      *condIsConst = true;
    } break;
    case SpvOpLogicalNot: {
      bool negVal;
      (void)GetConstCondition(cInst->GetSingleWordInOperand(0),
          &negVal, condIsConst);
      if (*condIsConst)
        *condVal = !negVal;
    } break;
    default: {
      *condIsConst = false;
    } break;
  }
}

void DeadBranchElimPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void DeadBranchElimPass::KillAllInsts(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    def_use_mgr_->KillInst(ip);
  });
}

bool DeadBranchElimPass::GetConstConditionalBranch(ir::BasicBlock* bp,
    ir::Instruction** branchInst, ir::Instruction** mergeInst,
    bool *condVal) {
  auto ii = bp->end();
  --ii;
  *branchInst = &*ii;
  if ((*branchInst)->opcode() != SpvOpBranchConditional)
    return false;
  --ii;
  *mergeInst = &*ii;
  if ((*mergeInst)->opcode() != SpvOpSelectionMerge)
    return false;
  bool condIsConst;
  (void) GetConstCondition(
      (*branchInst)->GetSingleWordInOperand(kSpvBranchCondConditionalId),
      condVal, &condIsConst);
  return condIsConst;
}

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  // Traverse blocks in structured order
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  std::unordered_set<ir::BasicBlock*> elimBlocks;
  bool modified = false;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Skip blocks that are already in the elimination set
    if (elimBlocks.find(*bi) != elimBlocks.end())
      continue;
    // Skip blocks that don't have constant conditional branch
    ir::Instruction* br;
    ir::Instruction* mergeInst;
    bool condVal;
    if (!GetConstConditionalBranch(*bi, &br, &mergeInst, &condVal))
      continue;

    // Replace conditional branch with unconditional branch
    uint32_t trueLabId = br->GetSingleWordInOperand(kSpvBranchCondTrueLabId);
    uint32_t falseLabId = br->GetSingleWordInOperand(kSpvBranchCondFalseLabId);
    uint32_t mergeLabId =
        mergeInst->GetSingleWordInOperand(kSpvSelectionMergeMergeBlockId);
    uint32_t liveLabId = condVal == true ? trueLabId : falseLabId;
    uint32_t deadLabId = condVal == true ? falseLabId : trueLabId;
    AddBranch(liveLabId, *bi);
    def_use_mgr_->KillInst(br);
    def_use_mgr_->KillInst(mergeInst);

    // Iterate to merge block deleting dead blocks
    std::unordered_set<uint32_t> deadLabIds;
    deadLabIds.insert(deadLabId);
    auto dbi = bi;
    ++dbi;
    uint32_t dLabId = (*dbi)->id();
    while (dLabId != mergeLabId) {
      if (deadLabIds.find(dLabId) != deadLabIds.end()) {
        // Add successor blocks to dead block set
        (*dbi)->ForEachSuccessorLabel([&deadLabIds](uint32_t succ) {
          deadLabIds.insert(succ);
        });
        // Add merge block to dead block set in case it has
        // no predecessors.
        uint32_t dMergeLabId = MergeBlockIdIfAny(**dbi);
        if (dMergeLabId != 0)
          deadLabIds.insert(dMergeLabId);
        // Kill use/def for all instructions and delete block
        KillAllInsts(*dbi);
        elimBlocks.insert(*dbi);
      }
      ++dbi;
      dLabId = (*dbi)->id();
    }

    // Process phi instructions in merge block.
    // deadLabIds are now blocks which cannot precede merge block.
    // If eliminated branch is to merge label, add current block to dead blocks.
    if (deadLabId == mergeLabId)
      deadLabIds.insert((*bi)->id());
    (*dbi)->ForEachPhiInst([&deadLabIds, this](ir::Instruction* phiInst) {
      uint32_t phiLabId0 = phiInst->GetSingleWordInOperand(kSpvPhiLab0Id);
      bool useFirst = deadLabIds.find(phiLabId0) == deadLabIds.end();
      uint32_t phiValIdx = useFirst ? kSpvPhiVal0Id : kSpvPhiVal1Id;
      uint32_t replId = phiInst->GetSingleWordInOperand(phiValIdx);
      uint32_t phiId = phiInst->result_id();
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(phiInst);
    });
    modified = true;
  }

  // Erase dead blocks
  for (auto ebi = func->begin(); ebi != func->end(); )
    if (elimBlocks.find(&*ebi) != elimBlocks.end())
      ebi = ebi.Erase();
    else
      ++ebi;
  return modified;
}

void DeadBranchElimPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  id2block_.clear();
  block2structured_succs_.clear();
  for (auto& fn : *module_) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
  }

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
};

Pass::Status DeadBranchElimPass::ProcessImpl() {
  // Current functionality assumes structured control flow. 
  // TODO(greg-lunarg): Handle non-structured control-flow.
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  bool modified = false;
  for (const auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = EliminateDeadBranches(fn) || modified;
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

DeadBranchElimPass::DeadBranchElimPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status DeadBranchElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

