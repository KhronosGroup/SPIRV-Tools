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

#include "aggressive_dead_code_elim_pass.h"

#include "iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kEntryPointFunctionIdInIdx = 1;
const uint32_t kStorePtrIdInIdx = 0;
const uint32_t kLoadPtrIdInIdx = 0;
const uint32_t kAccessChainPtrIdInIdx = 0;
const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kCopyObjectOperandInIdx = 0;
const uint32_t kNameTargetIdInIdx = 0;

}  // namespace anonymous

bool AggressiveDCEPass::IsNonPtrAccessChain(const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

ir::Instruction* AggressiveDCEPass::GetPtr(
      ir::Instruction* ip, uint32_t* varId) {
  const SpvOp op = ip->opcode();
  assert(op == SpvOpStore || op == SpvOpLoad);
  *varId = ip->GetSingleWordInOperand(
      op == SpvOpStore ? kStorePtrIdInIdx : kLoadPtrIdInIdx);
  ir::Instruction* ptrInst = def_use_mgr_->GetDef(*varId);
  ir::Instruction* varInst = ptrInst;
  while (varInst->opcode() != SpvOpVariable) {
    if (IsNonPtrAccessChain(varInst->opcode())) {
      *varId = varInst->GetSingleWordInOperand(kAccessChainPtrIdInIdx);
    }
    else {
      assert(varInst->opcode() == SpvOpCopyObject);
      *varId = varInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    }
    varInst = def_use_mgr_->GetDef(*varId);
  }
  return ptrInst;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId) {
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  return varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) ==
      SpvStorageClassFunction;
}

void AggressiveDCEPass::AddStores(uint32_t ptrId) {
  const analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
  if (uses == nullptr)
    return;
  for (const auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    if (op == SpvOpStore)
      worklist_.push(u.inst);
    else if (op != SpvOpLoad)
      AddStores(u.inst->result_id());
  }
}

bool AggressiveDCEPass::AggressiveDCE(ir::Function* func) {
  bool modified = false;
  // Add non-local stores, block terminating and merge instructions
  // to worklist. If function call encountered, return false, unmodified.
  // TODO(greg-lunarg): Improve in presence of function calls
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      switch (inst.opcode()) {
      case SpvOpStore: {
        uint32_t varId;
        (void) GetPtr(&inst, &varId);
        if (!IsLocalVar(varId)) {
          worklist_.push(&inst);
        }
      } break;
      case SpvOpLoopMerge:
      case SpvOpSelectionMerge:
      case SpvOpBranch:
      case SpvOpBranchConditional:
      case SpvOpSwitch: 
      case SpvOpKill: 
      case SpvOpUnreachable: 
      case SpvOpReturn:
      case SpvOpReturnValue: {
        worklist_.push(&inst);
      } break;
      case SpvOpFunctionCall: {
        return false;
      } break;
      default:
        break;
      }
    }
  }
  // Perform closure on live instruction set. 
  while (!worklist_.empty()) {
    ir::Instruction* liveInst = worklist_.front();
    live_insts_.insert(liveInst);
    // Add all operand instructions if not already live
    liveInst->ForEachInId([this](const uint32_t* iid) {
      ir::Instruction* inInst = def_use_mgr_->GetDef(*iid);
      if (live_insts_.find(inInst) == live_insts_.end())
        worklist_.push(inInst);
    });
    // If local load, add all variable's stores if variable not already live
    if (liveInst->opcode() == SpvOpLoad) {
      uint32_t varId;
      (void) GetPtr(liveInst, &varId);
      if (IsLocalVar(varId)) {
        if (live_local_vars_.find(varId) == live_local_vars_.end()) {
          AddStores(varId);
          live_local_vars_.insert(varId);
        }
      }
    }
    worklist_.pop();
  }
  // Mark all non-live instructions dead
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      if (live_insts_.find(&inst) != live_insts_.end())
        continue;
      dead_insts_.insert(&inst);
    }
  }
  // Remove debug statements referencing dead instructions. This must
  // be done before killing the instructions, otherwise there are dead
  // objects in the def/use database.
  for (auto& di : module_->debugs()) {
    if (di.opcode() != SpvOpName)
      continue;
    const uint32_t tId = di.GetSingleWordInOperand(kNameTargetIdInIdx);
    const ir::Instruction* tInst = def_use_mgr_->GetDef(tId);
    if (dead_insts_.find(tInst) == dead_insts_.end())
      continue;
    def_use_mgr_->KillInst(&di);
    modified = true;
  }
  // Kill dead instructions
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      if (dead_insts_.find(&inst) == dead_insts_.end())
        continue;
      def_use_mgr_->KillInst(&inst);
      modified = true;
    }
  }
  return modified;
}

void AggressiveDCEPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize id-to-function map
  id2function_.clear();
  for (auto& fn : *module_)
    id2function_[fn.result_id()] = &fn;

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
}

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes structured control flow. 
  // TODO(greg-lunarg): Handle non-structured control-flow.
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  // Current functionality assumes logical addressing only
  // TODO(greg-lunarg): Handle non-logical addressing
  if (module_->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;

  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx)];
    modified = modified || AggressiveDCE(fn);
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

AggressiveDCEPass::AggressiveDCEPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status AggressiveDCEPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

