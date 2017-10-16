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

namespace spvtools {
namespace opt {

namespace {

const uint32_t kBranchTargetLabIdInIdx = 0;
const uint32_t kBranchCondTrueLabIdInIdx = 1;
const uint32_t kBranchCondFalseLabIdInIdx = 2;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeContinueBlockIdInIdx = 1;

} // anonymous namespace

uint32_t DeadBranchElimPass::MergeBlockIdIfAny(
    const ir::BasicBlock& blk, uint32_t* cbid) const {
  auto merge_ii = blk.cend();
  --merge_ii;
  uint32_t mbid = 0;
  *cbid = 0;
  if (merge_ii != blk.cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx);
      *cbid = merge_ii->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
    }
    else if (merge_ii->opcode() == SpvOpSelectionMerge) {
      mbid = merge_ii->GetSingleWordInOperand(
          kSelectionMergeMergeBlockIdInIdx);
    }
  }
  return mbid;
}

void DeadBranchElimPass::ComputeStructuredSuccessors(ir::Function* func) {
  // If header, make merge block first successor. If a loop header, make
  // the second successor the continue target.
  for (auto& blk : *func) {
    uint32_t cbid;
    uint32_t mbid = MergeBlockIdIfAny(blk, &cbid);
    if (mbid != 0) {
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
      if (cbid != 0)
        block2structured_succs_[&blk].push_back(id2block_[cbid]);
    }
    // add true successors
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}

void DeadBranchElimPass::ComputeStructuredOrder(
    ir::Function* func, std::list<ir::BasicBlock*>* order) {
  // Compute structured successors and do DFS
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  auto get_structured_successors = [this](const ir::BasicBlock* block) {
      return &(block2structured_succs_[block]); };
  // TODO(greg-lunarg): Get rid of const_cast by making moving const
  // out of the cfa.h prototypes and into the invoking code.
  auto post_order = [&](cbb_ptr b) {
      order->push_front(const_cast<ir::BasicBlock*>(b)); };
  
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
      &*func->begin(), get_structured_successors, ignore_block, post_order,
      ignore_edge);
}

bool DeadBranchElimPass::GetConstCondition(uint32_t condId, bool* condVal) {
  bool condIsConst;
  ir::Instruction* cInst = def_use_mgr_->GetDef(condId);
  switch (cInst->opcode()) {
    case SpvOpConstantFalse: {
      *condVal = false;
      condIsConst = true;
    } break;
    case SpvOpConstantTrue: {
      *condVal = true;
      condIsConst = true;
    } break;
    case SpvOpLogicalNot: {
      bool negVal;
      condIsConst = GetConstCondition(cInst->GetSingleWordInOperand(0),
          &negVal);
      if (condIsConst)
        *condVal = !negVal;
    } break;
    default: {
      condIsConst = false;
    } break;
  }
  return condIsConst;
}

bool DeadBranchElimPass::GetConstInteger(uint32_t selId, uint32_t* selVal) {
  ir::Instruction* sInst = def_use_mgr_->GetDef(selId);
  uint32_t typeId = sInst->type_id();
  ir::Instruction* typeInst = def_use_mgr_->GetDef(typeId);
  if (!typeInst || (typeInst->opcode() != SpvOpTypeInt)) return false;
  // TODO(greg-lunarg): Support non-32 bit ints
  if (typeInst->GetSingleWordInOperand(0) != 32)
    return false;
  if (sInst->opcode() == SpvOpConstant) {
    *selVal = sInst->GetSingleWordInOperand(0);
    return true;
  }
  else if (sInst->opcode() == SpvOpConstantNull) {
    *selVal = 0;
    return true;
  }
  return false;
}

void DeadBranchElimPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void DeadBranchElimPass::AddSelectionMerge(uint32_t labelId,
    ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newMerge(
    new ir::Instruction(SpvOpSelectionMerge, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newMerge);
  bp->AddInstruction(std::move(newMerge));
}

void DeadBranchElimPass::AddBranchConditional(uint32_t condId,
    uint32_t trueLabId, uint32_t falseLabId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranchCond(
    new ir::Instruction(SpvOpBranchConditional, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {condId}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {trueLabId}},
         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {falseLabId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranchCond);
  bp->AddInstruction(std::move(newBranchCond));
}

void DeadBranchElimPass::KillAllInsts(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    KillNamesAndDecorates(ip);
    def_use_mgr_->KillInst(ip);
  });
}

bool DeadBranchElimPass::GetSelectionBranch(ir::BasicBlock* bp,
    ir::Instruction** branchInst, ir::Instruction** mergeInst,
    uint32_t *condId) {
  auto ii = bp->end();
  --ii;
  *branchInst = &*ii;
  if (ii == bp->begin())
    return false;
  --ii;
  *mergeInst = &*ii;
  if ((*mergeInst)->opcode() != SpvOpSelectionMerge)
    return false;
  // SPIR-V says the terminator for an OpSelectionMerge must be
  // either a conditional branch or a switch.
  assert((*branchInst)->opcode() == SpvOpBranchConditional ||
         (*branchInst)->opcode() == SpvOpSwitch);
  // Both BranchConidtional and Switch have their conditional value at 0.
  *condId = (*branchInst)->GetSingleWordInOperand(0);
  return true;
}

bool DeadBranchElimPass::HasNonPhiNonBackedgeRef(uint32_t labelId) {
  analysis::UseList* uses = def_use_mgr_->GetUses(labelId);
  if (uses == nullptr)
    return false;
  for (auto u : *uses) {
    if (u.inst->opcode() != SpvOpPhi &&
        backedges_.find(u.inst) == backedges_.end())
      return true;
  }
  return false;
}

void DeadBranchElimPass::ComputeBackEdges(
    std::list<ir::BasicBlock*>& structuredOrder) {
  backedges_.clear();
  std::unordered_set<uint32_t> visited;
  // In structured order, edges to visited blocks are back edges
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    visited.insert((*bi)->id());
    auto ii = (*bi)->end();
    --ii;
    switch(ii->opcode()) {
      case SpvOpBranch: {
        const uint32_t labId = ii->GetSingleWordInOperand(
            kBranchTargetLabIdInIdx);
        if (visited.find(labId) != visited.end())
          backedges_.insert(&*ii);
      } break;
      case SpvOpBranchConditional: {
        const uint32_t tLabId = ii->GetSingleWordInOperand(
            kBranchCondTrueLabIdInIdx);
        if (visited.find(tLabId) != visited.end()) {
          backedges_.insert(&*ii);
          break;
        }
        const uint32_t fLabId = ii->GetSingleWordInOperand(
            kBranchCondFalseLabIdInIdx);
        if (visited.find(fLabId) != visited.end())
          backedges_.insert(&*ii);
      } break;
      default:
        break;
    }
  }
}

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  // Traverse blocks in structured order
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  ComputeBackEdges(structuredOrder);
  std::unordered_set<ir::BasicBlock*> elimBlocks;
  bool modified = false;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Skip blocks that are already in the elimination set
    if (elimBlocks.find(*bi) != elimBlocks.end())
      continue;
    // Skip blocks that don't have conditional branch preceded
    // by OpSelectionMerge
    ir::Instruction* br;
    ir::Instruction* mergeInst;
    uint32_t condId;
    if (!GetSelectionBranch(*bi, &br, &mergeInst, &condId))
      continue;

    // If constant condition/selector, replace conditional branch/switch
    // with unconditional branch and delete merge
    uint32_t liveLabId;
    if (br->opcode() == SpvOpBranchConditional) {
      bool condVal;
      if (!GetConstCondition(condId, &condVal))
        continue;
      liveLabId = (condVal == true) ? 
          br->GetSingleWordInOperand(kBranchCondTrueLabIdInIdx) :
          br->GetSingleWordInOperand(kBranchCondFalseLabIdInIdx);
    }
    else {
      assert(br->opcode() == SpvOpSwitch);
      // Search switch operands for selector value, set liveLabId to
      // corresponding label, use default if not found
      uint32_t selVal;
      if (!GetConstInteger(condId, &selVal))
        continue;
      uint32_t icnt = 0;
      uint32_t caseVal;
      br->ForEachInOperand(
            [&icnt,&caseVal,&selVal,&liveLabId](const uint32_t* idp) {
        if (icnt == 1) {
          // Start with default label
          liveLabId = *idp;
        }
        else if (icnt > 1) {
          if (icnt % 2 == 0) {
            caseVal = *idp;
          }
          else {
            if (caseVal == selVal)
              liveLabId = *idp;
          }
        }
        ++icnt;
      });
    }

    const uint32_t mergeLabId =
        mergeInst->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
    AddBranch(liveLabId, *bi);
    def_use_mgr_->KillInst(br);
    def_use_mgr_->KillInst(mergeInst);

    modified = true;

    // Iterate to merge block adding dead blocks to elimination set
    auto dbi = bi;
    ++dbi;
    uint32_t dLabId = (*dbi)->id();
    while (dLabId != mergeLabId) {
      if (!HasNonPhiNonBackedgeRef(dLabId)) {
        // Kill use/def for all instructions and mark block for elimination
        KillAllInsts(*dbi);
        elimBlocks.insert(*dbi);
      }
      ++dbi;
      dLabId = (*dbi)->id();
    }

    // If merge block is unreachable, continue eliminating blocks until
    // a live block or last block is reached.
    while (!HasNonPhiNonBackedgeRef(dLabId)) {
      KillAllInsts(*dbi);
      elimBlocks.insert(*dbi);
      ++dbi;
      if (dbi == structuredOrder.end())
        break;
      dLabId = (*dbi)->id();
    }

    // If last block reached, look for next dead branch
    if (dbi == structuredOrder.end())
      continue;

    // Create set of dead predecessors in preparation for phi update.
    // Add the header block if the live branch is not the merge block.
    std::unordered_set<ir::BasicBlock*> deadPreds(elimBlocks);
    if (liveLabId != dLabId)
      deadPreds.insert(*bi);

    // Update phi instructions in terminating block.
    for (auto pii = (*dbi)->begin(); ; ++pii) {
      // Skip NoOps, break at end of phis
      SpvOp op = pii->opcode();
      if (op == SpvOpNop)
        continue;
      if (op != SpvOpPhi)
        break;
      // Count phi's live predecessors with lcnt and remember last one
      // with lidx.
      uint32_t lcnt = 0;
      uint32_t lidx = 0;
      uint32_t icnt = 0;
      pii->ForEachInId(
          [&deadPreds,&icnt,&lcnt,&lidx,this](uint32_t* idp) {
        if (icnt % 2 == 1) {
          if (deadPreds.find(id2block_[*idp]) == deadPreds.end()) {
            ++lcnt;
            lidx = icnt - 1;
          }
        }
        ++icnt;
      });
      // If just one live predecessor, replace resultid with live value id.
      uint32_t replId;
      if (lcnt == 1) {
        replId = pii->GetSingleWordInOperand(lidx);
      }
      else {
        // Otherwise create new phi eliminating dead predecessor entries
        assert(lcnt > 1);
        replId = TakeNextId();
        std::vector<ir::Operand> phi_in_opnds;
        icnt = 0;
        uint32_t lastId;
        pii->ForEachInId(
            [&deadPreds,&icnt,&phi_in_opnds,&lastId,this](uint32_t* idp) {
          if (icnt % 2 == 1) {
            if (deadPreds.find(id2block_[*idp]) == deadPreds.end()) {
              phi_in_opnds.push_back(
                  {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {lastId}});
              phi_in_opnds.push_back(
                  {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {*idp}});
            }
          }
          else {
            lastId = *idp;
          }
          ++icnt;
        });
        std::unique_ptr<ir::Instruction> newPhi(new ir::Instruction(
            SpvOpPhi, pii->type_id(), replId, phi_in_opnds));
        def_use_mgr_->AnalyzeInstDefUse(&*newPhi);
        pii = pii.InsertBefore(std::move(newPhi));
        ++pii;
      }
      const uint32_t phiId = pii->result_id();
      KillNamesAndDecorates(phiId);
      (void)def_use_mgr_->ReplaceAllUsesWith(phiId, replId);
      def_use_mgr_->KillInst(&*pii);
    }
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
  id2block_.clear();
  block2structured_succs_.clear();

  // Initialize block map
  for (auto& fn : *module_)
    for (auto& blk : fn)
      id2block_[blk.id()] = &blk;

  // TODO(greg-lunarg): Reuse def/use from previous passes
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize next unused Id.
  InitNextId();

  // Initialize extension whitelist
  InitExtensions();
};

bool DeadBranchElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : module_->extensions()) {
    const char* extName = reinterpret_cast<const char*>(
        &ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status DeadBranchElimPass::ProcessImpl() {
  // Current functionality assumes structured control flow. 
  // TODO(greg-lunarg): Handle non-structured control-flow.
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;
  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : module_->annotations())
    if (ai.opcode() == SpvOpGroupDecorate)
      return Status::SuccessWithoutChange;
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported())
    return Status::SuccessWithoutChange;
  // Collect all named and decorated ids
  FindNamedOrDecoratedIds();
  // Process all entry point functions
  ProcessFunction pfn = [this](ir::Function* fp) {
    return EliminateDeadBranches(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, module_);
  FinalizeNextId();
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

DeadBranchElimPass::DeadBranchElimPass() {}

Pass::Status DeadBranchElimPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

void DeadBranchElimPass::InitExtensions() {
  extensions_whitelist_.clear();
  extensions_whitelist_.insert({
    "SPV_AMD_shader_explicit_vertex_parameter",
    "SPV_AMD_shader_trinary_minmax",
    "SPV_AMD_gcn_shader",
    "SPV_KHR_shader_ballot",
    "SPV_AMD_shader_ballot",
    "SPV_AMD_gpu_shader_half_float",
    "SPV_KHR_shader_draw_parameters",
    "SPV_KHR_subgroup_vote",
    "SPV_KHR_16bit_storage",
    "SPV_KHR_device_group",
    "SPV_KHR_multiview",
    "SPV_NVX_multiview_per_view_attributes",
    "SPV_NV_viewport_array2",
    "SPV_NV_stereo_view_rendering",
    "SPV_NV_sample_mask_override_coverage",
    "SPV_NV_geometry_shader_passthrough",
    "SPV_AMD_texture_gather_bias_lod",
    "SPV_KHR_storage_buffer_storage_class",
    "SPV_KHR_variable_pointers",
    "SPV_AMD_gpu_shader_int16",
    "SPV_KHR_post_depth_coverage",
    "SPV_KHR_shader_atomic_counter_ops",
  });
}

}  // namespace opt
}  // namespace spvtools

