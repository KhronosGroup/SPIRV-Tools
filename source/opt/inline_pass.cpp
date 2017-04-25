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

#include "inline_pass.h"

// Indices of operands in SPIR-V instructions

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;
static const int kSpvTypePointerStorageClass = 1;
static const int kSpvTypePointerTypeId = 2;

namespace spvtools {
namespace opt {

uint32_t InlinePass::FindPointerToType(uint32_t type_id,
                                       SpvStorageClass storage_class) {
  ir::Module::inst_iterator type_itr = module_->types_values_begin();
  for (; type_itr != module_->types_values_end(); ++type_itr) {
    const ir::Instruction* type_inst = &*type_itr;
    if (type_inst->opcode() == SpvOpTypePointer &&
        type_inst->GetSingleWordOperand(kSpvTypePointerTypeId) == type_id &&
        type_inst->GetSingleWordOperand(kSpvTypePointerStorageClass) ==
            storage_class)
      return type_inst->result_id();
  }
  return 0;
}

uint32_t InlinePass::AddPointerToType(uint32_t type_id,
                                      SpvStorageClass storage_class) {
  uint32_t resultId = TakeNextId();
  std::unique_ptr<ir::Instruction> type_inst(new ir::Instruction(
      SpvOpTypePointer, 0, resultId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
        {uint32_t(storage_class)}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_id}}}));
  module_->AddType(std::move(type_inst));
  return resultId;
}

void InlinePass::AddBranch(uint32_t label_id,
                           std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newBranch(new ir::Instruction(
      SpvOpBranch, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddStore(uint32_t ptr_id, uint32_t val_id,
                          std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newStore(new ir::Instruction(
      SpvOpStore, 0, 0, {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}},
                         {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {val_id}}}));
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<ir::BasicBlock>* block_ptr) {
  std::unique_ptr<ir::Instruction> newLoad(new ir::Instruction(
      SpvOpLoad, type_id, resultId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}}}));
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

std::unique_ptr<ir::Instruction> InlinePass::NewLabel(uint32_t label_id) {
  std::unique_ptr<ir::Instruction> newLabel(
      new ir::Instruction(SpvOpLabel, 0, label_id, {}));
  return newLabel;
}

void InlinePass::MapParams(
    ir::Function* calleeFn,
    ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  int param_idx = 0;
  calleeFn->ForEachParam(
      [&call_inst_itr, &param_idx, &callee2caller](const ir::Instruction* cpi) {
        const uint32_t pid = cpi->result_id();
        (*callee2caller)[pid] = call_inst_itr->GetSingleWordOperand(
            kSpvFunctionCallArgumentId + param_idx);
        param_idx++;
      });
}

void InlinePass::CloneAndMapLocals(
    ir::Function* calleeFn,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
    std::unordered_map<uint32_t, uint32_t>* callee2caller) {
  auto callee_block_itr = calleeFn->begin();
  auto callee_var_itr = callee_block_itr->begin();
  while (callee_var_itr->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<ir::Instruction> var_inst(
        new ir::Instruction(*callee_var_itr));
    uint32_t newId = TakeNextId();
    var_inst->SetResultId(newId);
    (*callee2caller)[callee_var_itr->result_id()] = newId;
    new_vars->push_back(std::move(var_inst));
    callee_var_itr++;
  }
}

uint32_t InlinePass::CreateReturnVar(
    ir::Function* calleeFn,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars) {
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  const ir::Instruction* calleeType =
      def_use_mgr_->id_to_defs().find(calleeTypeId)->second;
  if (calleeType->opcode() != SpvOpTypeVoid) {
    // Find or create ptr to callee return type.
    uint32_t returnVarTypeId =
        FindPointerToType(calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0)
      returnVarTypeId = AddPointerToType(calleeTypeId, SpvStorageClassFunction);
    // Add return var to new function scope variables.
    returnVarId = TakeNextId();
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(
        SpvOpVariable, returnVarTypeId, returnVarId,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
          {SpvStorageClassFunction}}}));
    new_vars->push_back(std::move(var_inst));
  }
  return returnVarId;
}

bool InlinePass::IsSameBlockOp(const ir::Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

void InlinePass::CloneSameBlockOps(
    std::unique_ptr<ir::Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, ir::Instruction*>* preCallSB,
    std::unique_ptr<ir::BasicBlock>* block_ptr) {
  (*inst)
      ->ForEachInId([&postCallSB, &preCallSB, &block_ptr, this](uint32_t* iid) {
        const auto mapItr = (*postCallSB).find(*iid);
        if (mapItr == (*postCallSB).end()) {
          const auto mapItr2 = (*preCallSB).find(*iid);
          if (mapItr2 != (*preCallSB).end()) {
            // Clone pre-call same-block ops, map result id.
            const ir::Instruction* inInst = mapItr2->second;
            std::unique_ptr<ir::Instruction> sb_inst(
                new ir::Instruction(*inInst));
            CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr);
            const uint32_t rid = sb_inst->result_id();
            const uint32_t nid = this->TakeNextId();
            sb_inst->SetResultId(nid);
            (*postCallSB)[rid] = nid;
            *iid = nid;
            (*block_ptr)->AddInstruction(std::move(sb_inst));
          }
        } else {
          // Reset same-block op operand.
          *iid = mapItr->second;
        }
      });
}

void InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<ir::BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<ir::Instruction>>* new_vars,
    ir::UptrVectorIterator<ir::Instruction> call_inst_itr,
    ir::UptrVectorIterator<ir::BasicBlock> call_block_itr) {
  // Map from all ids in the callee to their equivalent id in the caller
  // as callee instructions are copied into caller.
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call same-block insts
  std::unordered_map<uint32_t, ir::Instruction*> preCallSB;
  // Post-call same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB;

  ir::Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller);

  // Define caller local variables for all callee variables and create map to
  // them.
  CloneAndMapLocals(calleeFn, new_vars, &callee2caller);

  // Create return var if needed.
  uint32_t returnVarId = CreateReturnVar(calleeFn, new_vars);

  // Clone and map callee code. Copy caller block code to beginning of
  // first block and end of last block.
  bool prevInstWasReturn = false;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  const uint32_t calleeTypeId = calleeFn->type_id();
  std::unique_ptr<ir::BasicBlock> new_blk_ptr;
  calleeFn->ForEachInst([&new_blocks, &callee2caller, &call_block_itr,
                         &call_inst_itr, &new_blk_ptr, &prevInstWasReturn,
                         &returnLabelId, &returnVarId, &calleeTypeId,
                         &multiBlocks, &postCallSB, &preCallSB, this](
      const ir::Instruction* cpi) {
    switch (cpi->opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
      case SpvOpVariable:
        // Already processed
        break;
      case SpvOpLabel: {
        // If previous instruction was early return, insert branch
        // instruction to return block.
        if (prevInstWasReturn) {
          if (returnLabelId == 0) returnLabelId = this->TakeNextId();
          AddBranch(returnLabelId, &new_blk_ptr);
          prevInstWasReturn = false;
        }
        // Finish current block (if it exists) and get label for next block.
        uint32_t labelId;
        bool firstBlock = false;
        if (new_blk_ptr != nullptr) {
          new_blocks->push_back(std::move(new_blk_ptr));
          // If result id is already mapped, use it, otherwise get a new
          // one.
          const uint32_t rid = cpi->result_id();
          const auto mapItr = callee2caller.find(rid);
          labelId = (mapItr != callee2caller.end()) ? mapItr->second
                                                    : this->TakeNextId();
        } else {
          // First block needs to use label of original block
          // but map callee label in case of phi reference.
          labelId = call_block_itr->label_id();
          callee2caller[cpi->result_id()] = labelId;
          firstBlock = true;
        }
        // Create first/next block.
        new_blk_ptr.reset(new ir::BasicBlock(NewLabel(labelId)));
        if (firstBlock) {
          // Copy contents of original caller block up to call instruction.
          for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
               cii++) {
            std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cii));
            // Remember same-block ops for possible regeneration.
            if (IsSameBlockOp(&*cp_inst)) {
              auto* sb_inst_ptr = cp_inst.get();
              preCallSB[cp_inst->result_id()] = sb_inst_ptr;
            }
            new_blk_ptr->AddInstruction(std::move(cp_inst));
          }
        } else {
          multiBlocks = true;
        }
      } break;
      case SpvOpReturnValue: {
        // Store return value to return variable.
        assert(returnVarId != 0);
        uint32_t valId = cpi->GetInOperand(kSpvReturnValueId).words[0];
        const auto mapItr = callee2caller.find(valId);
        if (mapItr != callee2caller.end()) {
          valId = mapItr->second;
        }
        AddStore(returnVarId, valId, &new_blk_ptr);

        // Remember we saw a return; if followed by a label, will need to
        // insert branch.
        prevInstWasReturn = true;
      } break;
      case SpvOpReturn: {
        // Remember we saw a return; if followed by a label, will need to
        // insert branch.
        prevInstWasReturn = true;
      } break;
      case SpvOpFunctionEnd: {
        // If there was an early return, create return label/block.
        // If previous instruction was return, insert branch instruction
        // to return block.
        if (returnLabelId != 0) {
          if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);
          new_blocks->push_back(std::move(new_blk_ptr));
          new_blk_ptr.reset(new ir::BasicBlock(NewLabel(returnLabelId)));
          multiBlocks = true;
        }
        // Load return value into result id of call, if it exists.
        if (returnVarId != 0) {
          const uint32_t resId = call_inst_itr->result_id();
          assert(resId != 0);
          AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr);
        }
        // Copy remaining instructions from caller block.
        auto cii = call_inst_itr;
        for (cii++; cii != call_block_itr->end(); cii++) {
          std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cii));
          // If multiple blocks generated, regenerate any same-block
          // instruction that has not been seen in this last block.
          if (multiBlocks) {
            CloneSameBlockOps(&cp_inst, &postCallSB, &preCallSB, &new_blk_ptr);
            // Remember same-block ops in this block.
            if (IsSameBlockOp(&*cp_inst)) {
              const uint32_t rid = cp_inst->result_id();
              postCallSB[rid] = rid;
            }
          }
          new_blk_ptr->AddInstruction(std::move(cp_inst));
        }
        // Finalize inline code.
        new_blocks->push_back(std::move(new_blk_ptr));
      } break;
      default: {
        // Copy callee instruction and remap all input Ids.
        std::unique_ptr<ir::Instruction> cp_inst(new ir::Instruction(*cpi));
        cp_inst->ForEachInId([&callee2caller, &cpi, this](uint32_t* iid) {
          const auto mapItr = callee2caller.find(*iid);
          if (mapItr != callee2caller.end()) {
            *iid = mapItr->second;
          } else if (cpi->has_labels()) {
            const ir::Instruction* inst =
                def_use_mgr_->id_to_defs().find(*iid)->second;
            if (inst->opcode() == SpvOpLabel) {
              // Forward label reference. Allocate a new label id, map it,
              // use it and check for it at each label.
              const uint32_t nid = this->TakeNextId();
              callee2caller[*iid] = nid;
              *iid = nid;
            }
          }
        });
        // Map and reset result id.
        const uint32_t rid = cp_inst->result_id();
        if (rid != 0) {
          const uint32_t nid = this->TakeNextId();
          callee2caller[rid] = nid;
          cp_inst->SetResultId(nid);
        }
        new_blk_ptr->AddInstruction(std::move(cp_inst));
      } break;
    }
  });
  // Update block map given replacement blocks.
  for (auto& blk : *new_blocks) {
    id2block_[blk->label_id()] = &*blk;
  }
}

bool InlinePass::IsInlinableFunctionCall(const ir::Instruction* inst) {
  if (inst->opcode() != SpvOp::SpvOpFunctionCall) return false;
  const uint32_t calleeFnId =
      inst->GetSingleWordOperand(kSpvFunctionCallFunctionId);
  const auto ci = inlinable_.find(calleeFnId);
  return ci != inlinable_.cend();
}

bool InlinePass::Inline(ir::Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii)) {
        // Inline call.
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(&newBlocks, &newVars, ii, bi);
        // Update phi functions in successor blocks if call block
        // is replaced with more than one block.
        if (newBlocks.size() > 1) {
          const auto firstBlk = newBlocks.begin();
          const auto lastBlk = newBlocks.end() - 1;
          const uint32_t firstId = (*firstBlk)->label_id();
          const uint32_t lastId = (*lastBlk)->label_id();
          (*lastBlk)->ForEachSuccessorLabel(
              [&firstId, &lastId, this](uint32_t succ) {
                ir::BasicBlock* sbp = this->id2block_[succ];
                sbp->ForEachPhiInst([&firstId, &lastId](ir::Instruction* phi) {
                  phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
                    if (*id == firstId) *id = lastId;
                  });
                });
              });
        }
        // Replace old calling block with new block(s).
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0) func->begin()->begin().InsertBefore(&newVars);
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ii++;
      }
    }
  }
  return modified;
}

bool InlinePass::IsInlinableFunction(const ir::Function* func) {
  // We can only inline a function if it has blocks.
  if (func->cbegin() == func->cend())
    return false;
  // Do not inline functions with multiple returns
  // TODO(greg-lunarg): Enable inlining if no return is in loop
  int returnCnt = 0;
  for (auto bi = func->cbegin(); bi != func->cend(); bi++) {
    auto li = bi->cend();
    li--;
    if (li->opcode() == SpvOpReturn || li->opcode() == SpvOpReturnValue) {
      if (returnCnt > 0)
        return false;
      returnCnt++;
    }
  }
  return true;
}

void InlinePass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id.
  next_id_ = module->id_bound();

  // Save module.
  module_ = module;

  // Initialize function and block maps.
  id2function_.clear();
  id2block_.clear();
  inlinable_.clear();
  for (auto& fn : *module_) {
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.label_id()] = &blk;
    }
    if (IsInlinableFunction(&fn))
      inlinable_.insert(fn.result_id());
  }
};

Pass::Status InlinePass::ProcessImpl() {
  // Do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = modified || Inline(fn);
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlinePass::InlinePass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status InlinePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
