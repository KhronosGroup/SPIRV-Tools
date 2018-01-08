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
  context()->set_instr_block(&*newBranch, bp);
  bp->AddInstruction(std::move(newBranch));
}

bool DeadBranchElimPass::EliminateDeadBranches(ir::Function* func) {
  auto get_parent_block = [this](uint32_t id) {
    return context()->get_instr_block(get_def_use_mgr()->GetDef(id));
  };

  // Mark live blocks reachable from the entry. Simplify constant branches and
  // switches as you proceed to limit the number of live blocks. Be careful not
  // to eliminate backedges even if they are dead, but the header is live.
  std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> continues;
  bool modified = false;
  std::unordered_set<ir::BasicBlock*> liveBlocks;
  std::vector<ir::BasicBlock*> stack;
  stack.push_back(&*func->begin());
  while (!stack.empty()) {
    ir::BasicBlock* block = stack.back();
    stack.pop_back();

    // Live blocks doubles as visited set.
    if (!liveBlocks.insert(block).second) continue;

    uint32_t contId = block->ContinueBlockIdIfAny();
    if (contId != 0) continues[get_parent_block(contId)] = block;

    ir::Instruction* terminator = &*block->tail();
    uint32_t liveLabId = 0;
    // Check if the terminator has a single valid successor.
    if (terminator->opcode() == SpvOpBranchConditional) {
      bool condVal;
      if (GetConstCondition(terminator->GetSingleWordInOperand(0u), &condVal)) {
        liveLabId = terminator->GetSingleWordInOperand(
            condVal ? kBranchCondTrueLabIdInIdx : kBranchCondFalseLabIdInIdx);
      }
    } else if (terminator->opcode() == SpvOpSwitch) {
      uint32_t selVal;
      if (GetConstInteger(terminator->GetSingleWordInOperand(0u), &selVal)) {
        // Search switch operands for selector value, set liveLabId to
        // corresponding label, use default if not found.
        uint32_t icnt = 0;
        uint32_t caseVal;
        terminator->ForEachInOperand(
            [&icnt, &caseVal, &selVal, &liveLabId](const uint32_t* idp) {
              if (icnt == 1) {
                // Start with default label.
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
    }

    // Don't simplify branches of continue blocks. A path from the continue to
    // the merge is required.
    // TODO(alan-baker): They can be simplified iff there remains a path to the
    // backedge. Structured control flow should guarantee one path hits the
    // backedge, but I've removed the requirement for structured control flow
    // from this pass.
    bool simplify = liveLabId != 0 && !continues.count(block);

    if (simplify) {
      modified = true;
      // Replace with unconditional branch.
      // Remove the merge instruction if it is a selection merge.
      AddBranch(liveLabId, block);
      context()->KillInst(terminator);
      ir::Instruction* mergeInst = block->GetMergeInst();
      if (mergeInst->opcode() == SpvOpSelectionMerge) {
        context()->KillInst(mergeInst);
      }
      stack.push_back(get_parent_block(liveLabId));
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
  // undef (because I still clobber the continue block).
  std::unordered_set<ir::BasicBlock*> unreachableMerges;
  std::unordered_map<ir::BasicBlock*, ir::BasicBlock*> unreachableContinues;
  for (auto block : liveBlocks) {
    uint32_t mergeId = block->MergeBlockIdIfAny();
    if (mergeId != 0) {
      ir::BasicBlock* mergeBlock = get_parent_block(mergeId);
      if (!liveBlocks.count(mergeBlock)) {
        unreachableMerges.insert(mergeBlock);
      }
      uint32_t contId = block->ContinueBlockIdIfAny();
      if (contId != 0) {
        ir::BasicBlock* contBlock = get_parent_block(contId);
        if (!liveBlocks.count(contBlock)) {
          unreachableContinues[contBlock] = block;
        }
      }
    }
  }

  for (auto& block : *func) {
    if (liveBlocks.count(&block)) {
      // Fix phis so that only live (or unremovable) incoming edges are
      // present. If the block now only has a single live incoming edge, remove
      // the phi and replace its uses with its data input.
      for (auto iter = block.begin(); iter != block.end();) {
        if (iter->opcode() != SpvOpPhi) {
          break;
        }

        bool changed = false;
        bool backedge_added = false;
        ir::Instruction* inst = &*iter;
        std::vector<ir::Operand> operands;
        // Iterate through the incoming labels and determine which to keep
        // and/or modify.
        for (uint32_t i = 1; i < inst->NumInOperands(); i += 2) {
          ir::BasicBlock* inc =
              get_parent_block(inst->GetSingleWordInOperand(i));
          if (unreachableContinues.count(inc)) {
            // Replace incoming value with undef.
            operands.emplace_back(
                SPV_OPERAND_TYPE_ID,
                std::initializer_list<uint32_t>{Type2Undef(inst->type_id())});
            operands.push_back(inst->GetInOperand(i));
            changed = true;
            backedge_added = true;
          } else if (liveBlocks.count(inc) && inc->IsSuccessor(&block)) {
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
          if (!backedge_added && continue_id != 0) {
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
          uint32_t replId;
          if (operands.size() == 2) {
            replId = operands[0].words[0];
          } else {
            replId = TakeNextId();
            std::unique_ptr<ir::Instruction> newPhi(
                new ir::Instruction(context(), SpvOpPhi, inst->type_id(),
                                    replId, std::move(operands)));
            get_def_use_mgr()->AnalyzeInstDefUse(&*newPhi);
            context()->set_instr_block(&*newPhi, &block);
            inst->InsertBefore(std::move(newPhi));
          }
          context()->ReplaceAllUsesWith(inst->result_id(), replId);
          iter = context()->KillInst(&*inst);
        } else {
          ++iter;
        }
      }
    }
  }

  // Erase dead blocks.
  for (auto ebi = func->begin(); ebi != func->end();) {
    if (unreachableMerges.count(&*ebi)) {
      // Make unreachable, but leave the label.
      KillAllInsts(&*ebi, false);
      // Add unreachable terminator.
      ebi->AddInstruction(
          MakeUnique<ir::Instruction>(context(), SpvOpUnreachable, 0, 0,
                                      std::initializer_list<ir::Operand>{}));
      ++ebi;
      modified = true;
    } else if (unreachableContinues.count(&*ebi)) {
      // Make unreachable, but leave the label.
      KillAllInsts(&*ebi, false);
      // Add unconditional branch to header.
      uint32_t contId = unreachableContinues[&*ebi]->id();
      ebi->AddInstruction(MakeUnique<ir::Instruction>(
          context(), SpvOpBranch, 0, 0,
          std::initializer_list<ir::Operand>{{SPV_OPERAND_TYPE_ID, {contId}}}));
      get_def_use_mgr()->AnalyzeInstUse(&*ebi->tail());
      ++ebi;
      modified = true;
    } else if (!liveBlocks.count(&*ebi)) {
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
  // bool modified = ProcessEntryPointCallTree(pfn, get_module());
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
