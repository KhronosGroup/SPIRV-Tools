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
#include "ir_context.h"
#include "iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kBranchTargetLabIdInIdx = 0;
const uint32_t kBranchCondTrueLabIdInIdx = 1;
const uint32_t kBranchCondFalseLabIdInIdx = 2;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;

}  // anonymous namespace

bool DeadBranchElimPass::GetConstCondition(uint32_t condId, bool* condVal) {
  bool condIsConst;
  ir::Instruction* cInst = get_def_use_mgr()->GetDef(condId);
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
      condIsConst =
          GetConstCondition(cInst->GetSingleWordInOperand(0), &negVal);
      if (condIsConst) *condVal = !negVal;
    } break;
    default: { condIsConst = false; } break;
  }
  return condIsConst;
}

bool DeadBranchElimPass::GetConstInteger(uint32_t selId, uint32_t* selVal) {
  ir::Instruction* sInst = get_def_use_mgr()->GetDef(selId);
  uint32_t typeId = sInst->type_id();
  ir::Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  if (!typeInst || (typeInst->opcode() != SpvOpTypeInt)) return false;
  // TODO(greg-lunarg): Support non-32 bit ints
  if (typeInst->GetSingleWordInOperand(0) != 32) return false;
  if (sInst->opcode() == SpvOpConstant) {
    *selVal = sInst->GetSingleWordInOperand(0);
    return true;
  } else if (sInst->opcode() == SpvOpConstantNull) {
    *selVal = 0;
    return true;
  }
  return false;
}

void DeadBranchElimPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
      context(), SpvOpBranch, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

void DeadBranchElimPass::AddSelectionMerge(uint32_t labelId,
                                           ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newMerge(new ir::Instruction(
      context(), SpvOpSelectionMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newMerge);
  bp->AddInstruction(std::move(newMerge));
}

void DeadBranchElimPass::AddBranchConditional(uint32_t condId,
                                              uint32_t trueLabId,
                                              uint32_t falseLabId,
                                              ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranchCond(new ir::Instruction(
      context(), SpvOpBranchConditional, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {condId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {trueLabId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {falseLabId}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranchCond);
  bp->AddInstruction(std::move(newBranchCond));
}

bool DeadBranchElimPass::GetSelectionBranch(ir::BasicBlock* bp,
                                            ir::Instruction** branchInst,
                                            ir::Instruction** mergeInst,
                                            uint32_t* condId) {
  auto ii = bp->end();
  --ii;
  *branchInst = &*ii;
  if (ii == bp->begin()) return false;
  --ii;
  *mergeInst = &*ii;
  if ((*mergeInst)->opcode() != SpvOpSelectionMerge) return false;
  // SPIR-V says the terminator for an OpSelectionMerge must be
  // either a conditional branch or a switch.
  assert((*branchInst)->opcode() == SpvOpBranchConditional ||
         (*branchInst)->opcode() == SpvOpSwitch);
  // Both BranchConidtional and Switch have their conditional value at 0.
  *condId = (*branchInst)->GetSingleWordInOperand(0);
  return true;
}

bool DeadBranchElimPass::HasNonPhiNonBackedgeRef(uint32_t labelId) {
  bool nonPhiNonBackedgeRef = false;
  get_def_use_mgr()->ForEachUser(
      labelId, [this, &nonPhiNonBackedgeRef](ir::Instruction* user) {
        if (user->opcode() != SpvOpPhi &&
            backedges_.find(user) == backedges_.end()) {
          nonPhiNonBackedgeRef = true;
        }
      });
  return nonPhiNonBackedgeRef;
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
    switch (ii->opcode()) {
      case SpvOpBranch: {
        const uint32_t labId =
            ii->GetSingleWordInOperand(kBranchTargetLabIdInIdx);
        if (visited.find(labId) != visited.end()) backedges_.insert(&*ii);
      } break;
      case SpvOpBranchConditional: {
        const uint32_t tLabId =
            ii->GetSingleWordInOperand(kBranchCondTrueLabIdInIdx);
        if (visited.find(tLabId) != visited.end()) {
          backedges_.insert(&*ii);
          break;
        }
        const uint32_t fLabId =
            ii->GetSingleWordInOperand(kBranchCondFalseLabIdInIdx);
        if (visited.find(fLabId) != visited.end()) backedges_.insert(&*ii);
      } break;
      default:
        break;
    }
  }
}

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  // Traverse blocks in structured order
  std::list<ir::BasicBlock*> structuredOrder;
  cfg()->ComputeStructuredOrder(func, &*func->begin(), &structuredOrder);
  ComputeBackEdges(structuredOrder);
  std::unordered_set<ir::BasicBlock*> elimBlocks;
  bool modified = false;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Skip blocks that are already in the elimination set
    if (elimBlocks.find(*bi) != elimBlocks.end()) continue;
    // Skip blocks that don't have conditional branch preceded
    // by OpSelectionMerge
    ir::Instruction* br;
    ir::Instruction* mergeInst;
    uint32_t condId;
    if (!GetSelectionBranch(*bi, &br, &mergeInst, &condId)) continue;

    // If constant condition/selector, replace conditional branch/switch
    // with unconditional branch and delete merge
    uint32_t liveLabId;
    if (br->opcode() == SpvOpBranchConditional) {
      bool condVal;
      if (!GetConstCondition(condId, &condVal)) continue;
      liveLabId = (condVal == true)
                      ? br->GetSingleWordInOperand(kBranchCondTrueLabIdInIdx)
                      : br->GetSingleWordInOperand(kBranchCondFalseLabIdInIdx);
    } else {
      assert(br->opcode() == SpvOpSwitch);
      // Search switch operands for selector value, set liveLabId to
      // corresponding label, use default if not found
      uint32_t selVal;
      if (!GetConstInteger(condId, &selVal)) continue;
      uint32_t icnt = 0;
      uint32_t caseVal;
      br->ForEachInOperand(
          [&icnt, &caseVal, &selVal, &liveLabId](const uint32_t* idp) {
            if (icnt == 1) {
              // Start with default label
              liveLabId = *idp;
            } else if (icnt > 1) {
              if (icnt % 2 == 0) {
                caseVal = *idp;
              } else {
                if (caseVal == selVal) liveLabId = *idp;
              }
            }
            ++icnt;
          });
    }

    const uint32_t mergeLabId =
        mergeInst->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
    AddBranch(liveLabId, *bi);
    context()->KillInst(br);
    context()->KillInst(mergeInst);

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
      if (dbi == structuredOrder.end()) break;
      dLabId = (*dbi)->id();
    }

    // If last block reached, look for next dead branch
    if (dbi == structuredOrder.end()) continue;

    // Create set of dead predecessors in preparation for phi update.
    // Add the header block if the live branch is not the merge block.
    std::unordered_set<ir::BasicBlock*> deadPreds(elimBlocks);
    if (liveLabId != dLabId) deadPreds.insert(*bi);

    // Update phi instructions in terminating block.
    for (auto pii = (*dbi)->begin();; ++pii) {
      // Skip NoOps, break at end of phis
      SpvOp op = pii->opcode();
      if (op == SpvOpNop) continue;
      if (op != SpvOpPhi) break;
      // Count phi's live predecessors with lcnt and remember last one
      // with lidx.
      uint32_t lcnt = 0;
      uint32_t lidx = 0;
      uint32_t icnt = 0;
      pii->ForEachInId([&deadPreds, &icnt, &lcnt, &lidx, this](uint32_t* idp) {
        if (icnt % 2 == 1) {
          if (deadPreds.find(cfg()->block(*idp)) == deadPreds.end()) {
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
      } else {
        // Otherwise create new phi eliminating dead predecessor entries
        assert(lcnt > 1);
        replId = TakeNextId();
        std::vector<ir::Operand> phi_in_opnds;
        icnt = 0;
        uint32_t lastId;
        pii->ForEachInId(
            [&deadPreds, &icnt, &phi_in_opnds, &lastId, this](uint32_t* idp) {
              if (icnt % 2 == 1) {
                if (deadPreds.find(cfg()->block(*idp)) == deadPreds.end()) {
                  phi_in_opnds.push_back(
                      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {lastId}});
                  phi_in_opnds.push_back(
                      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {*idp}});
                }
              } else {
                lastId = *idp;
              }
              ++icnt;
            });
        std::unique_ptr<ir::Instruction> newPhi(new ir::Instruction(
            context(), SpvOpPhi, pii->type_id(), replId, phi_in_opnds));
        get_def_use_mgr()->AnalyzeInstDefUse(&*newPhi);
        pii = pii.InsertBefore(std::move(newPhi));
        ++pii;
      }
      const uint32_t phiId = pii->result_id();
      context()->KillNamesAndDecorates(phiId);
      (void)context()->ReplaceAllUsesWith(phiId, replId);
      context()->KillInst(&*pii);
    }
  }

  // Erase dead blocks
  for (auto ebi = func->begin(); ebi != func->end();)
    if (elimBlocks.find(&*ebi) != elimBlocks.end())
      ebi = ebi.Erase();
    else
      ++ebi;
  return modified;
}

void DeadBranchElimPass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize extension whitelist
  InitExtensions();
};

bool DeadBranchElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status DeadBranchElimPass::ProcessImpl() {
  // Current functionality assumes structured control flow.
  // TODO(greg-lunarg): Handle non-structured control-flow.
  if (!get_module()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;
  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == SpvOpGroupDecorate) return Status::SuccessWithoutChange;
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process all entry point functions
  ProcessFunction pfn = [this](ir::Function* fp) {
    return EliminateDeadBranches(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

DeadBranchElimPass::DeadBranchElimPass() {}

Pass::Status DeadBranchElimPass::Process(ir::IRContext* module) {
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
