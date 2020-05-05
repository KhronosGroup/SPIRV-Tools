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

#include "source/opt/inline_pass.h"

#include <unordered_set>
#include <utility>

#include "source/cfa.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

// Indices of operands in SPIR-V instructions

static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;

namespace spvtools {
namespace opt {

uint32_t InlinePass::AddPointerToType(uint32_t type_id,
                                      SpvStorageClass storage_class) {
  uint32_t resultId = context()->TakeNextId();
  if (resultId == 0) {
    return resultId;
  }

  std::unique_ptr<Instruction> type_inst(
      new Instruction(context(), SpvOpTypePointer, 0, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {uint32_t(storage_class)}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_id}}}));
  context()->AddType(std::move(type_inst));
  analysis::Type* pointeeTy;
  std::unique_ptr<analysis::Pointer> pointerTy;
  std::tie(pointeeTy, pointerTy) =
      context()->get_type_mgr()->GetTypeAndPointerType(type_id,
                                                       SpvStorageClassFunction);
  context()->get_type_mgr()->RegisterType(resultId, *pointerTy);
  return resultId;
}

void InlinePass::AddBranch(uint32_t label_id,
                           std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddBranchCond(uint32_t cond_id, uint32_t true_id,
                               uint32_t false_id,
                               std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranchConditional, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddLoopMerge(uint32_t merge_id, uint32_t continue_id,
                              std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoopMerge(new Instruction(
      context(), SpvOpLoopMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {merge_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {continue_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LOOP_CONTROL, {0}}}));
  (*block_ptr)->AddInstruction(std::move(newLoopMerge));
}

void InlinePass::AddStore(uint32_t ptr_id, uint32_t val_id,
                          std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newStore(
      new Instruction(context(), SpvOpStore, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {val_id}}}));
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newLoad(
      new Instruction(context(), SpvOpLoad, type_id, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}}}));
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

std::unique_ptr<Instruction> InlinePass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  return newLabel;
}

uint32_t InlinePass::GetFalseId() {
  if (false_id_ != 0) return false_id_;
  false_id_ = get_module()->GetGlobalValue(SpvOpConstantFalse);
  if (false_id_ != 0) return false_id_;
  uint32_t boolId = get_module()->GetGlobalValue(SpvOpTypeBool);
  if (boolId == 0) {
    boolId = context()->TakeNextId();
    if (boolId == 0) {
      return 0;
    }
    get_module()->AddGlobalValue(SpvOpTypeBool, boolId, 0);
  }
  false_id_ = context()->TakeNextId();
  if (false_id_ == 0) {
    return 0;
  }
  get_module()->AddGlobalValue(SpvOpConstantFalse, false_id_, boolId);
  return false_id_;
}

void InlinePass::MapParams(
    Function* calleeFn, BasicBlock::iterator call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  int param_idx = 0;
  calleeFn->ForEachParam(
      [&call_inst_itr, &param_idx, &callee2caller](const Instruction* cpi) {
        const uint32_t pid = cpi->result_id();
        (*callee2caller)[pid] = call_inst_itr->GetSingleWordOperand(
            kSpvFunctionCallArgumentId + param_idx);
        ++param_idx;
      });
}

bool InlinePass::CloneAndMapLocals(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  auto callee_block_itr = calleeFn->begin();
  auto callee_var_itr = callee_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<Instruction> var_inst(callee_var_itr->Clone(context()));
    uint32_t newId = context()->TakeNextId();
    if (newId == 0) {
      return false;
    }
    get_decoration_mgr()->CloneDecorations(callee_var_itr->result_id(), newId);
    var_inst->SetResultId(newId);
    (*callee2caller)[callee_var_itr->result_id()] = newId;
    new_vars->push_back(std::move(var_inst));
    ++callee_var_itr;
  }
  return true;
}

uint32_t InlinePass::CreateReturnVar(
    Function* calleeFn, std::vector<std::unique_ptr<Instruction>>* new_vars) {
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  assert(type_mgr->GetType(calleeTypeId)->AsVoid() == nullptr &&
         "Cannot create a return variable of type void.");
  // Find or create ptr to callee return type.
  uint32_t returnVarTypeId =
      type_mgr->FindPointerToType(calleeTypeId, SpvStorageClassFunction);

  if (returnVarTypeId == 0) {
    returnVarTypeId = AddPointerToType(calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0) {
      return 0;
    }
  }

  // Add return var to new function scope variables.
  returnVarId = context()->TakeNextId();
  if (returnVarId == 0) {
    return 0;
  }

  std::unique_ptr<Instruction> var_inst(
      new Instruction(context(), SpvOpVariable, returnVarTypeId, returnVarId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {SpvStorageClassFunction}}}));
  new_vars->push_back(std::move(var_inst));
  get_decoration_mgr()->CloneDecorations(calleeFn->result_id(), returnVarId);
  return returnVarId;
}

bool InlinePass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

bool InlinePass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    std::unique_ptr<BasicBlock>* block_ptr) {
  return (*inst)->WhileEachInId([&postCallSB, &preCallSB, &block_ptr,
                                 this](uint32_t* iid) {
    const auto mapItr = (*postCallSB).find(*iid);
    if (mapItr == (*postCallSB).end()) {
      const auto mapItr2 = (*preCallSB).find(*iid);
      if (mapItr2 != (*preCallSB).end()) {
        // Clone pre-call same-block ops, map result id.
        const Instruction* inInst = mapItr2->second;
        std::unique_ptr<Instruction> sb_inst(inInst->Clone(context()));
        if (!CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr)) {
          return false;
        }

        const uint32_t rid = sb_inst->result_id();
        const uint32_t nid = context()->TakeNextId();
        if (nid == 0) {
          return false;
        }
        get_decoration_mgr()->CloneDecorations(rid, nid);
        sb_inst->SetResultId(nid);
        (*postCallSB)[rid] = nid;
        *iid = nid;
        (*block_ptr)->AddInstruction(std::move(sb_inst));
      }
    } else {
      // Reset same-block op operand.
      *iid = mapItr->second;
    }
    return true;
  });
}

void InlinePass::MoveInstsBeforeEntryBlock(
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    BasicBlock* new_blk_ptr, BasicBlock::iterator call_inst_itr,
    UptrVectorIterator<BasicBlock> call_block_itr) {
  for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
       cii = call_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> cp_inst(inst);
    // Remember same-block ops for possible regeneration.
    if (IsSameBlockOp(&*cp_inst)) {
      auto* sb_inst_ptr = cp_inst.get();
      (*preCallSB)[cp_inst->result_id()] = sb_inst_ptr;
    }
    new_blk_ptr->AddInstruction(std::move(cp_inst));
  }
}

std::unique_ptr<BasicBlock> InlinePass::AddGuardBlock(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock> new_blk_ptr, uint32_t entry_blk_label_id) {
  const auto guard_block_id = context()->TakeNextId();
  if (guard_block_id == 0) {
    return nullptr;
  }
  AddBranch(guard_block_id, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Start the next block.
  new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(guard_block_id));
  // Reset the mapping of the callee's entry block to point to
  // the guard block.  Do this so we can fix up phis later on to
  // satisfy dominance.
  (*callee2caller)[entry_blk_label_id] = guard_block_id;
  return new_blk_ptr;
}

std::unique_ptr<BasicBlock> InlinePass::InsertHeaderBlockForSingleTripLoop(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock>* single_trip_loop_cont_blk,
    uint32_t* returnLabelId, std::unique_ptr<BasicBlock> new_blk_ptr,
    uint32_t entry_blk_label_id) {
  uint32_t singleTripLoopHeaderId = context()->TakeNextId();
  if (singleTripLoopHeaderId == 0) {
    return nullptr;
  }
  AddBranch(singleTripLoopHeaderId, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
  new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(singleTripLoopHeaderId));
  *returnLabelId = context()->TakeNextId();
  uint32_t singleTripLoopContinueId = context()->TakeNextId();
  if (*returnLabelId == 0 || singleTripLoopContinueId == 0) {
    return nullptr;
  }
  AddLoopMerge(*returnLabelId, singleTripLoopContinueId, &new_blk_ptr);
  uint32_t postHeaderId = context()->TakeNextId();
  if (postHeaderId == 0) {
    return nullptr;
  }
  AddBranch(postHeaderId, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
  new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(postHeaderId));
  // Reset the mapping of the callee's entry block to point to
  // the post-header block.  Do this so we can fix up phis later
  // on to satisfy dominance.
  (*callee2caller)[entry_blk_label_id] = postHeaderId;

  *single_trip_loop_cont_blk =
      MakeUnique<BasicBlock>(NewLabel(singleTripLoopContinueId));
  uint32_t false_id = GetFalseId();
  if (false_id == 0) {
    return nullptr;
  }
  AddBranchCond(false_id, singleTripLoopHeaderId, *returnLabelId,
                single_trip_loop_cont_blk);
  return new_blk_ptr;
}

InstructionList::iterator InlinePass::AddStoresForVariableInitializers(
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock>* new_blk_ptr,
    UptrVectorIterator<BasicBlock> callee_first_block_itr) {
  auto callee_var_itr = callee_first_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    if (callee_var_itr->NumInOperands() == 2) {
      assert(callee2caller->count(callee_var_itr->result_id()) &&
             "Expected the variable to have already been mapped.");
      uint32_t new_var_id = callee2caller->at(callee_var_itr->result_id());

      // The initializer must be a constant or global value.  No mapped
      // should be used.
      uint32_t val_id = callee_var_itr->GetSingleWordInOperand(1);
      AddStore(new_var_id, val_id, new_blk_ptr);
    }
    ++callee_var_itr;
  }
  return callee_var_itr;
}

bool InlinePass::InlineInstructionInBB(
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    BasicBlock* new_blk_ptr, const Instruction* inst) {
  // Copy callee instruction and remap all input Ids.
  std::unique_ptr<Instruction> cp_inst(inst->Clone(context()));
  bool succeeded =
      cp_inst->WhileEachInId([&callee2caller](uint32_t* iid) {
        const auto mapItr = callee2caller->find(*iid);
        if (mapItr != callee2caller->end()) {
          *iid = mapItr->second;
        }
        return true;
      });
  if (!succeeded) {
    return false;
  }
  // If result id is non-zero, remap it. If already mapped, use mapped
  // value, else use next id.
  const uint32_t rid = cp_inst->result_id();
  if (rid != 0) {
    const auto mapItr = callee2caller->find(rid);
    uint32_t nid;
    if (mapItr != callee2caller->end()) {
      nid = mapItr->second;
    } else {
      nid = context()->TakeNextId();
      if (nid == 0) {
        return false;
      }
      (*callee2caller)[rid] = nid;
    }
    cp_inst->SetResultId(nid);
    get_decoration_mgr()->CloneDecorations(rid, nid);
  }
  new_blk_ptr->AddInstruction(std::move(cp_inst));
  return true;
}

bool InlinePass::InlineTerminationInstructionInBB(
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock>* new_blk_ptr, uint32_t* returnLabelId,
    bool* prevInstWasReturn, const Instruction* inst, uint32_t returnVarId) {
  assert(IsTerminatorInst(inst->opcode()));
  switch (inst->opcode()) {
    case SpvOpUnreachable:
    case SpvOpKill: {
      // Generate a return label so that we split the block with the
      // function call. Copy the terminator into the new block.
      if (*returnLabelId == 0) {
        *returnLabelId = context()->TakeNextId();
        if (*returnLabelId == 0) {
          return false;
        }
      }
      std::unique_ptr<Instruction> terminator(
          new Instruction(context(), inst->opcode(), 0, 0, {}));
      new_blk_ptr->get()->AddInstruction(std::move(terminator));
      break;
    }
    case SpvOpReturnValue: {
      // Store return value to return variable.
      assert(returnVarId != 0);
      uint32_t valId = inst->GetInOperand(kSpvReturnValueId).words[0];
      const auto mapItr = callee2caller->find(valId);
      if (mapItr != callee2caller->end()) {
        valId = mapItr->second;
      }
      AddStore(returnVarId, valId, new_blk_ptr);

      // Remember we saw a return; if followed by a label, will need to
      // insert branch.
      *prevInstWasReturn = true;
    } break;
    case SpvOpReturn: {
      // Remember we saw a return; if followed by a label, will need to
      // insert branch.
      *prevInstWasReturn = true;
    } break;
    default:
      return InlineInstructionInBB(callee2caller, new_blk_ptr->get(), inst);
      break;
  }
  return true;
}

bool InlinePass::InlineEntryBlock(
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock>* new_blk_ptr, uint32_t* returnLabelId,
    bool* prevInstWasReturn, UptrVectorIterator<BasicBlock> callee_first_block,
    uint32_t returnVarId) {
  auto callee_inst_itr = AddStoresForVariableInitializers(
      callee2caller, new_blk_ptr, callee_first_block);

  auto callee_first_block_tail = callee_first_block->tail();
  while (callee_inst_itr != callee_first_block_tail) {
    if (!InlineInstructionInBB(callee2caller, new_blk_ptr->get(),
                               &*callee_inst_itr)) {
      return false;
    }
    ++callee_inst_itr;
  }
  InlineTerminationInstructionInBB(callee2caller, new_blk_ptr, returnLabelId,
                                   prevInstWasReturn, &*callee_inst_itr,
                                   returnVarId);
  return true;
}

uint32_t InlinePass::InlineLabel(
    const Instruction* inst, std::unique_ptr<BasicBlock>* new_blk_ptr,
    uint32_t* returnLabelId, bool* prevInstWasReturn,
    const std::unordered_map<uint32_t, uint32_t>& callee2caller) {
  // If previous instruction was early return, insert branch
  // instruction to return block.
  if (*prevInstWasReturn) {
    if (*returnLabelId == 0) {
      *returnLabelId = context()->TakeNextId();
      if (*returnLabelId == 0) {
        return 0;
      }
    }
    AddBranch(*returnLabelId, new_blk_ptr);
    *prevInstWasReturn = false;
  }
  // If result id is already mapped, use it, otherwise get a new
  // one.
  const uint32_t rid = inst->result_id();
  const auto mapItr = callee2caller.find(rid);
  return (mapItr != callee2caller.end()) ? mapItr->second
                                         : context()->TakeNextId();
}

std::unique_ptr<BasicBlock> InlinePass::InlineBasicBlocks(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::unique_ptr<BasicBlock> new_blk_ptr, uint32_t* returnLabelId,
    bool* prevInstWasReturn, Function* calleeFn, uint32_t returnVarId) {
  auto callee_block_itr = calleeFn->begin();
  ++callee_block_itr;

  while (callee_block_itr != calleeFn->end()) {
    // Inline OpLabel instruction and create next block.
    uint32_t labelId =
        InlineLabel(callee_block_itr->GetLabelInst(), &new_blk_ptr,
                    returnLabelId, prevInstWasReturn, *callee2caller);
    if (labelId == 0) {
      return nullptr;
    }
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(labelId));

    auto tail_inst_itr = callee_block_itr->tail();
    for (auto inst_itr = callee_block_itr->begin(); inst_itr != tail_inst_itr;
         ++inst_itr) {
      if (!InlineInstructionInBB(callee2caller, new_blk_ptr.get(),
                                 &*inst_itr)) {
        return nullptr;
      }
    }
    InlineTerminationInstructionInBB(callee2caller, &new_blk_ptr, returnLabelId,
                                     prevInstWasReturn, &*tail_inst_itr,
                                     returnVarId);
    ++callee_block_itr;
  }
  return new_blk_ptr;
}

bool InlinePass::MoveCallerInstsAfterFunctionCall(
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unique_ptr<BasicBlock>* new_blk_ptr,
    BasicBlock::iterator call_inst_itr, bool multiBlocks) {
  // Copy remaining instructions from caller block.
  for (Instruction* inst = call_inst_itr->NextNode(); inst;
       inst = call_inst_itr->NextNode()) {
    inst->RemoveFromList();
    std::unique_ptr<Instruction> cp_inst(inst);
    // If multiple blocks generated, regenerate any same-block
    // instruction that has not been seen in this last block.
    if (multiBlocks) {
      if (!CloneSameBlockOps(&cp_inst, postCallSB, preCallSB, new_blk_ptr)) {
        return false;
      }

      // Remember same-block ops in this block.
      if (IsSameBlockOp(&*cp_inst)) {
        const uint32_t rid = cp_inst->result_id();
        (*postCallSB)[rid] = rid;
      }
    }
    new_blk_ptr->get()->AddInstruction(std::move(cp_inst));
  }

  return true;
}

void InlinePass::MoveLoopMergeInstToFirstBlock(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Move the OpLoopMerge from the last block back to the first, where
  // it belongs.
  auto& first = new_blocks->front();
  auto& last = new_blocks->back();
  assert(first != last);

  // Insert a modified copy of the loop merge into the first block.
  auto loop_merge_itr = last->tail();
  --loop_merge_itr;
  assert(loop_merge_itr->opcode() == SpvOpLoopMerge);
  std::unique_ptr<Instruction> cp_inst(loop_merge_itr->Clone(context()));
  first->tail().InsertBefore(std::move(cp_inst));

  // Remove the loop merge from the last block.
  loop_merge_itr->RemoveFromList();
  delete &*loop_merge_itr;
}

bool InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator call_inst_itr,
    UptrVectorIterator<BasicBlock> call_block_itr) {
  // Map from all ids in the callee to their equivalent id in the caller
  // as callee instructions are copied into caller.
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call same-block insts
  std::unordered_map<uint32_t, Instruction*> preCallSB;
  // Post-call same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB;

  // Invalidate the def-use chains.  They are not kept up to date while
  // inlining.  However, certain calls try to keep them up-to-date if they are
  // valid.  These operations can fail.
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);

  // If the caller is a loop header and the callee has multiple blocks, then the
  // normal inlining logic will place the OpLoopMerge in the last of several
  // blocks in the loop.  Instead, it should be placed at the end of the first
  // block.  We'll wait to move the OpLoopMerge until the end of the regular
  // inlining logic, and only if necessary.
  bool caller_is_loop_header = call_block_itr->GetLoopMergeInst() != nullptr;

  // Single-trip loop continue block
  std::unique_ptr<BasicBlock> single_trip_loop_cont_blk;

  Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller);

  // Define caller local variables for all callee variables and create map to
  // them.
  if (!CloneAndMapLocals(calleeFn, new_vars, &callee2caller)) {
    return false;
  }

  // First block needs to use label of original block
  // but map callee label in case of phi reference.
  uint32_t entry_blk_label_id = calleeFn->begin()->GetLabelInst()->result_id();
  callee2caller[entry_blk_label_id] = call_block_itr->id();
  std::unique_ptr<BasicBlock> new_blk_ptr =
      MakeUnique<BasicBlock>(NewLabel(call_block_itr->id()));

  // Move instructions of original caller block up to call instruction.
  MoveInstsBeforeEntryBlock(&preCallSB, new_blk_ptr.get(), call_inst_itr,
                            call_block_itr);

  if (caller_is_loop_header &&
      (*(calleeFn->begin())).GetMergeInst() != nullptr) {
    // We can't place both the caller's merge instruction and
    // another merge instruction in the same block.  So split the
    // calling block. Insert an unconditional branch to a new guard
    // block.  Later, once we know the ID of the last block,  we
    // will move the caller's OpLoopMerge from the last generated
    // block into the first block. We also wait to avoid
    // invalidating various iterators.
    new_blk_ptr = AddGuardBlock(new_blocks, &callee2caller,
                                std::move(new_blk_ptr), entry_blk_label_id);
    if (new_blk_ptr == nullptr) return false;
  }

  uint32_t returnLabelId = 0;
  bool earlyReturn = false;
  if (early_return_funcs_.find(calleeFn->result_id()) !=
      early_return_funcs_.end()) {
    // If callee has early return, insert a header block for
    // single-trip loop that will encompass callee code.  Start
    // postheader block.
    //
    // Note: Consider the following combination:
    //  - the caller is a single block loop
    //  - the callee does not begin with a structure header
    //  - the callee has multiple returns.
    // We still need to split the caller block and insert a guard
    // block. But we only need to do it once. We haven't done it yet,
    // but the single-trip loop header will serve the same purpose.
    new_blk_ptr = InsertHeaderBlockForSingleTripLoop(
        new_blocks, &callee2caller, &single_trip_loop_cont_blk, &returnLabelId,
        std::move(new_blk_ptr), entry_blk_label_id);
    if (new_blk_ptr == nullptr) return false;
    earlyReturn = true;
  }

  // Create return var if needed.
  const uint32_t calleeTypeId = calleeFn->type_id();
  uint32_t returnVarId = 0;
  analysis::Type* calleeType = context()->get_type_mgr()->GetType(calleeTypeId);
  if (calleeType->AsVoid() == nullptr) {
    returnVarId = CreateReturnVar(calleeFn, new_vars);
    if (returnVarId == 0) {
      return false;
    }
  }

  // Create set of callee result ids. Used to detect forward references
  calleeFn->WhileEachInst([&callee2caller, this](const Instruction* cpi) {
    const uint32_t rid = cpi->result_id();
    if (rid != 0 && callee2caller.find(rid) == callee2caller.end()) {
      const uint32_t nid = context()->TakeNextId();
      if (nid == 0) return false;
      callee2caller[rid] = nid;
      return true;
    }
    return true;
  });

  // Inline the entry block of the callee function.
  bool prevInstWasReturn = false;
  if (!InlineEntryBlock(&callee2caller, &new_blk_ptr, &returnLabelId,
                        &prevInstWasReturn, calleeFn->begin(), returnVarId)) {
    return false;
  }

  // Inline blocks of the callee function other than the entry block.
  new_blk_ptr = InlineBasicBlocks(new_blocks, &callee2caller,
                                  std::move(new_blk_ptr), &returnLabelId,
                                  &prevInstWasReturn, calleeFn, returnVarId);
  if (new_blk_ptr == nullptr) return false;

  // If there was an early return, we generated a return label id
  // for it.  Now we have to generate the return block with that Id.
  if (returnLabelId != 0) {
    // If previous instruction was return, insert branch instruction
    // to return block.
    if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);

    new_blocks->push_back(std::move(new_blk_ptr));

    // If we generated a loop header for the single-trip loop
    // to accommodate early returns, insert the continue
    // target block now, with a false branch back to the loop
    // header.
    if (earlyReturn)
      new_blocks->push_back(std::move(single_trip_loop_cont_blk));

    // Generate the return block.
    new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(returnLabelId));
  }

  // Load return value into result id of call, if it exists.
  if (returnVarId != 0) {
    const uint32_t resId = call_inst_itr->result_id();
    assert(resId != 0);
    AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr);
  }

  // Move instructions of original caller block after call instruction.
  if (!MoveCallerInstsAfterFunctionCall(&preCallSB, &postCallSB, &new_blk_ptr,
                                        call_inst_itr,
                                        calleeFn->begin() != calleeFn->end()))
    return false;

  // Finalize inline code.
  new_blocks->push_back(std::move(new_blk_ptr));

  if (caller_is_loop_header && (new_blocks->size() > 1))
    MoveLoopMergeInstToFirstBlock(new_blocks);

  // Update block map given replacement blocks.
  for (auto& blk : *new_blocks) {
    id2block_[blk->id()] = &*blk;
  }
  return true;
}

bool InlinePass::IsInlinableFunctionCall(const Instruction* inst) {
  if (inst->opcode() != SpvOp::SpvOpFunctionCall) return false;
  const uint32_t calleeFnId =
      inst->GetSingleWordOperand(kSpvFunctionCallFunctionId);
  const auto ci = inlinable_.find(calleeFnId);
  return ci != inlinable_.cend();
}

void InlinePass::UpdateSucceedingPhis(
    std::vector<std::unique_ptr<BasicBlock>>& new_blocks) {
  const auto firstBlk = new_blocks.begin();
  const auto lastBlk = new_blocks.end() - 1;
  const uint32_t firstId = (*firstBlk)->id();
  const uint32_t lastId = (*lastBlk)->id();
  const BasicBlock& const_last_block = *lastBlk->get();
  const_last_block.ForEachSuccessorLabel(
      [&firstId, &lastId, this](const uint32_t succ) {
        BasicBlock* sbp = this->id2block_[succ];
        sbp->ForEachPhiInst([&firstId, &lastId](Instruction* phi) {
          phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
            if (*id == firstId) *id = lastId;
          });
        });
      });
}

bool InlinePass::HasNoReturnInStructuredConstruct(Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return false;
  const auto structured_analysis = context()->GetStructuredCFGAnalysis();
  // Search for returns in structured construct.
  bool return_in_construct = false;
  for (auto& blk : *func) {
    auto terminal_ii = blk.cend();
    --terminal_ii;
    if (spvOpcodeIsReturn(terminal_ii->opcode()) &&
        structured_analysis->ContainingConstruct(blk.id()) != 0) {
      return_in_construct = true;
      break;
    }
  }
  return !return_in_construct;
}

bool InlinePass::HasNoReturnInLoop(Function* func) {
  // If control not structured, do not do loop/return analysis
  // TODO: Analyze returns in non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return false;
  const auto structured_analysis = context()->GetStructuredCFGAnalysis();
  // Search for returns in structured construct.
  bool return_in_loop = false;
  for (auto& blk : *func) {
    auto terminal_ii = blk.cend();
    --terminal_ii;
    if (spvOpcodeIsReturn(terminal_ii->opcode()) &&
        structured_analysis->ContainingLoop(blk.id()) != 0) {
      return_in_loop = true;
      break;
    }
  }
  return !return_in_loop;
}

void InlinePass::AnalyzeReturns(Function* func) {
  if (HasNoReturnInLoop(func)) {
    no_return_in_loop_.insert(func->result_id());
    if (!HasNoReturnInStructuredConstruct(func))
      early_return_funcs_.insert(func->result_id());
  }
}

bool InlinePass::IsInlinableFunction(Function* func) {
  // We can only inline a function if it has blocks.
  if (func->cbegin() == func->cend()) return false;
  // Do not inline functions with returns in loops. Currently early return
  // functions are inlined by wrapping them in a one trip loop and implementing
  // the returns as a branch to the loop's merge block. However, this can only
  // done validly if the return was not in a loop in the original function.
  // Also remember functions with multiple (early) returns.
  AnalyzeReturns(func);
  if (no_return_in_loop_.find(func->result_id()) == no_return_in_loop_.cend()) {
    return false;
  }

  if (func->IsRecursive()) {
    return false;
  }

  // Do not inline functions with an OpKill if they are called from a continue
  // construct. If it is inlined into a continue construct it will generate
  // invalid code.
  bool func_is_called_from_continue =
      funcs_called_from_continue_.count(func->result_id()) != 0;

  if (func_is_called_from_continue && ContainsKill(func)) {
    return false;
  }

  return true;
}

bool InlinePass::ContainsKill(Function* func) const {
  return !func->WhileEachInst(
      [](Instruction* inst) { return inst->opcode() != SpvOpKill; });
}

void InlinePass::InitializeInline() {
  false_id_ = 0;

  // clear collections
  id2function_.clear();
  id2block_.clear();
  inlinable_.clear();
  no_return_in_loop_.clear();
  early_return_funcs_.clear();
  funcs_called_from_continue_ =
      context()->GetStructuredCFGAnalysis()->FindFuncsCalledFromContinue();

  for (auto& fn : *get_module()) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
    // Compute inlinability
    if (IsInlinableFunction(&fn)) inlinable_.insert(fn.result_id());
  }
}

InlinePass::InlinePass() {}

}  // namespace opt
}  // namespace spvtools
