// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "dead_branch_elim_pass.h"

#include "cfa.h"
#include "ir_context.h"
#include "iterator.h"
#include "make_unique.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kBranchCondTrueLabIdInIdx = 1;
const uint32_t kBranchCondFalseLabIdInIdx = 2;

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
  assert(get_def_use_mgr()->GetDef(labelId) != nullptr);
  std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
      context(), SpvOpBranch, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  // This pass only requires correct instruction block mappings for the input.
  // This pass does not preserve the block mapping, so it is not kept
  // up-to-date during processing.
  auto get_parent_block = [this](uint32_t id) {
    return context()->get_instr_block(get_def_use_mgr()->GetDef(id));
  };

  // Mark live blocks reachable from the entry. Simplify constant branches and
  // switches as you proceed, to limit the number of live blocks. Be careful not
  // to eliminate backedges even if they are dead, but the header is live.
  // Likewise, unreachable merge blocks named in live merge instruction must be
  // retained (though they may be clobbered).
  //
  // |continues| maps the continue target to its corresponding header.
  std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> continues;
  bool modified = false;
  std::unordered_set<ir::BasicBlock*> live_blocks;
  std::vector<ir::BasicBlock*> stack;
  stack.push_back(&*func->begin());
  while (!stack.empty()) {
    ir::BasicBlock* block = stack.back();
    stack.pop_back();

    // Live blocks doubles as visited set.
    if (!live_blocks.insert(block).second) continue;

    uint32_t cont_id = block->ContinueBlockIdIfAny();
    if (cont_id != 0) continues[get_parent_block(cont_id)] = block;

    ir::Instruction* terminator = block->terminator();
    uint32_t live_lab_id = 0;
    // Check if the terminator has a single valid successor.
    if (terminator->opcode() == SpvOpBranchConditional) {
      bool condVal;
      if (GetConstCondition(terminator->GetSingleWordInOperand(0u), &condVal)) {
        live_lab_id = terminator->GetSingleWordInOperand(
            condVal ? kBranchCondTrueLabIdInIdx : kBranchCondFalseLabIdInIdx);
      }
    } else if (terminator->opcode() == SpvOpSwitch) {
      uint32_t sel_val;
      if (GetConstInteger(terminator->GetSingleWordInOperand(0u), &sel_val)) {
        // Search switch operands for selector value, set live_lab_id to
        // corresponding label, use default if not found.
        uint32_t icnt = 0;
        uint32_t case_val;
        terminator->ForEachInOperand(
            [&icnt, &case_val, &sel_val, &live_lab_id](const uint32_t* idp) {
              if (icnt == 1) {
                // Start with default label.
                live_lab_id = *idp;
              } else if (icnt > 1) {
                if (icnt % 2 == 0) {
                  case_val = *idp;
                } else {
                  if (case_val == sel_val) live_lab_id = *idp;
                }
              }
              ++icnt;
            });
      }
    }

    // Don't simplify branches of continue blocks. A path from the continue to
    // the header is required.
    // TODO(alan-baker): They can be simplified iff there remains a path to the
    // backedge. Structured control flow should guarantee one path hits the
    // backedge, but I've removed the requirement for structured control flow
    // from this pass.
    bool simplify = live_lab_id != 0 && !continues.count(block);

    if (simplify) {
      modified = true;
      // Replace with unconditional branch.
      // Remove the merge instruction if it is a selection merge.
      AddBranch(live_lab_id, block);
      context()->KillInst(terminator);
      ir::Instruction* mergeInst = block->GetMergeInst();
      if (mergeInst->opcode() == SpvOpSelectionMerge) {
        context()->KillInst(mergeInst);
      }
      stack.push_back(get_parent_block(live_lab_id));
    } else {
      // All successors are live.
      block->ForEachSuccessorLabel(
          [&stack, get_parent_block](const uint32_t label) {
            stack.push_back(get_parent_block(label));
          });
    }
  }

  // Check for unreachable merge and continue blocks with live headers, those
  // blocks must remain. Continues are tracked separately so that when updating
  // live phi nodes with an edge from a continue I can replace it with an
  // undef (because we clobber the instructions inside continue block).
  //
  // |unreachable_continues| maps continue targets that cannot be reached to
  // merge instruction that declares them.
  std::unordered_set<ir::BasicBlock*> unreachable_merges;
  std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> unreachable_continues;
  for (auto block : live_blocks) {
    uint32_t merge_id = block->MergeBlockIdIfAny();
    if (merge_id != 0) {
      ir::BasicBlock* merge_block = get_parent_block(merge_id);
      if (!live_blocks.count(merge_block)) {
        unreachable_merges.insert(merge_block);
      }
      uint32_t cont_id = block->ContinueBlockIdIfAny();
      if (cont_id != 0) {
        ir::BasicBlock* cont_block = get_parent_block(cont_id);
        if (!live_blocks.count(cont_block)) {
          unreachable_continues[cont_block] = block;
        }
      }
    }
  }

  // Fix phis in reachable blocks so that only live (or unremovable) incoming
  // edges are present. If the block now only has a single live incoming edge,
  // remove the phi and replace its uses with its data input.
  for (auto& block : *func) {
    if (live_blocks.count(&block)) {
      for (auto iter = block.begin(); iter != block.end();) {
        if (iter->opcode() != SpvOpPhi) {
          break;
        }

        bool changed = false;
        bool backedge_added = false;
        ir::Instruction* inst = &*iter;
        std::vector<ir::Operand> operands;
        // Build a complete set of operands (not just input operands). Start
        // with type and result id operands.
        operands.push_back(inst->GetOperand(0u));
        operands.push_back(inst->GetOperand(1u));
        // Iterate through the incoming labels and determine which to keep
        // and/or modify.
        for (uint32_t i = 1; i < inst->NumInOperands(); i += 2) {
          ir::BasicBlock* inc =
              get_parent_block(inst->GetSingleWordInOperand(i));
          if (unreachable_continues.count(inc) &&
              unreachable_continues[inc] == &block) {
            // Replace incoming value with undef if this phi exists in the loop
            // header. Otherwise, this edge is not live since the unreachable
            // continue block will be replaced with an unconditional branch to
            // the header only.
            operands.emplace_back(
                SPV_OPERAND_TYPE_ID,
                std::initializer_list<uint32_t>{Type2Undef(inst->type_id())});
            operands.push_back(inst->GetInOperand(i));
            changed = true;
            backedge_added = true;
          } else if (live_blocks.count(inc) && inc->IsSuccessor(&block)) {
            // Keep live incoming edge.
            operands.push_back(inst->GetInOperand(i - 1));
            operands.push_back(inst->GetInOperand(i));
          } else {
            // Remove incoming edge.
            changed = true;
          }
        }

        if (changed) {
          uint32_t continue_id = block.ContinueBlockIdIfAny();
          if (!backedge_added && continue_id != 0 &&
              unreachable_continues.count(get_parent_block(continue_id))) {
            // Changed the backedge to branch from the continue block instead
            // of a successor of the continue block. Add an entry to the phi to
            // provide an undef for the continue block. Since the successor of
            // the continue must also be unreachable (dominated by the continue
            // block), any entry for the original backedge has been removed
            // from the phi operands.
            operands.emplace_back(
                SPV_OPERAND_TYPE_ID,
                std::initializer_list<uint32_t>{Type2Undef(inst->type_id())});
            operands.emplace_back(SPV_OPERAND_TYPE_ID,
                                  std::initializer_list<uint32_t>{continue_id});
          }

          // Replace the phi with either a single value or a rebuilt phi.
          // uint32_t replId;
          //
          // We always have type and result id operands. So this phi has a
          // single source if are two more operands beyond those.
          if (operands.size() == 4) {
            // First input data operands is at index 2.
            uint32_t replId = operands[2u].words[0];
            context()->ReplaceAllUsesWith(inst->result_id(), replId);
            iter = context()->KillInst(&*inst);
          } else {
            // We've rewritten the operands, so first instruct the def/use
            // manager to forget uses that in the phi before we replace them.
            // After replacing operands update the def/use manager by
            // re-analyzing the used ids in this phi.
            get_def_use_mgr()->EraseUseRecordsOfOperandIds(inst);
            inst->ReplaceOperands(operands);
            get_def_use_mgr()->AnalyzeInstUse(inst);
            ++iter;
          }
        } else {
          ++iter;
        }
      }
    }
  }

  // Erase dead blocks. Any block captured in |unreachable_merges| or
  // |unreachable_continues| is a dead block that is required to remain due to
  // a live merge instruction in the corresponding header. These blocks will
  // have their instructions clobbered and will become a label and terminator.
  // Unreachable merge blocks are terminated by OpReachable, while unreachable
  // continue blocks are terminated by an unconditional branch to the header.
  // Otherwise, blocks are dead if not explicitly captured in |live_blocks| and
  // are totally removed.
  for (auto ebi = func->begin(); ebi != func->end();) {
    if (unreachable_merges.count(&*ebi)) {
      // Make unreachable, but leave the label.
      KillAllInsts(&*ebi, false);
      // Add unreachable terminator.
      ebi->AddInstruction(
          MakeUnique<ir::Instruction>(context(), SpvOpUnreachable, 0, 0,
                                      std::initializer_list<ir::Operand>{}));
      ++ebi;
      modified = true;
    } else if (unreachable_continues.count(&*ebi)) {
      // Make unreachable, but leave the label.
      KillAllInsts(&*ebi, false);
      // Add unconditional branch to header.
      uint32_t cont_id = unreachable_continues[&*ebi]->id();
      ebi->AddInstruction(
          MakeUnique<ir::Instruction>(context(), SpvOpBranch, 0, 0,
                                      std::initializer_list<ir::Operand>{
                                          {SPV_OPERAND_TYPE_ID, {cont_id}}}));
      get_def_use_mgr()->AnalyzeInstUse(&*ebi->tail());
      ++ebi;
      modified = true;
    } else if (!live_blocks.count(&*ebi)) {
      // Kill this block.
      KillAllInsts(&*ebi);
      ebi = ebi.Erase();
      modified = true;
    } else {
      ++ebi;
    }
  }
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
  bool modified = ProcessReachableCallTree(pfn, context());
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
