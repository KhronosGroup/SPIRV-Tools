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

#include "mem_pass.h"

#include "cfa.h"
#include "iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kAccessChainPtrIdInIdx = 0;
const uint32_t kCopyObjectOperandInIdx = 0;
const uint32_t kLoadPtrIdInIdx = 0;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kStorePtrIdInIdx = 0;
const uint32_t kStoreValIdInIdx = 1;
const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kTypePointerTypeIdInIdx = 1;

}  // namespace

bool MemPass::IsBaseTargetType(const ir::Instruction* typeInst) const {
  switch (typeInst->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeBool:
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeImage:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
      return true;
    default:
      break;
  }
  return false;
}

bool MemPass::IsTargetType(const ir::Instruction* typeInst) const {
  if (IsBaseTargetType(typeInst)) return true;
  if (typeInst->opcode() == SpvOpTypeArray)
    return IsBaseTargetType(
        get_def_use_mgr()->GetDef(typeInst->GetSingleWordOperand(1)));
  if (typeInst->opcode() != SpvOpTypeStruct) return false;
  // All struct members must be math type
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp, this](const uint32_t* tid) {
    ir::Instruction* compTypeInst = get_def_use_mgr()->GetDef(*tid);
    if (!IsBaseTargetType(compTypeInst)) ++nonMathComp;
  });
  return nonMathComp == 0;
}

bool MemPass::IsNonPtrAccessChain(const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

bool MemPass::IsPtr(uint32_t ptrId) {
  uint32_t varId = ptrId;
  ir::Instruction* ptrInst = get_def_use_mgr()->GetDef(varId);
  while (ptrInst->opcode() == SpvOpCopyObject) {
    varId = ptrInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    ptrInst = get_def_use_mgr()->GetDef(varId);
  }
  const SpvOp op = ptrInst->opcode();
  if (op == SpvOpVariable || IsNonPtrAccessChain(op)) return true;
  if (op != SpvOpFunctionParameter) return false;
  const uint32_t varTypeId = ptrInst->type_id();
  const ir::Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  return varTypeInst->opcode() == SpvOpTypePointer;
}

ir::Instruction* MemPass::GetPtr(uint32_t ptrId, uint32_t* varId) {
  *varId = ptrId;
  ir::Instruction* ptrInst = get_def_use_mgr()->GetDef(*varId);
  while (ptrInst->opcode() == SpvOpCopyObject) {
    *varId = ptrInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    ptrInst = get_def_use_mgr()->GetDef(*varId);
  }
  ir::Instruction* varInst = ptrInst;
  while (varInst->opcode() != SpvOpVariable &&
         varInst->opcode() != SpvOpFunctionParameter) {
    if (IsNonPtrAccessChain(varInst->opcode())) {
      *varId = varInst->GetSingleWordInOperand(kAccessChainPtrIdInIdx);
    } else {
      assert(varInst->opcode() == SpvOpCopyObject);
      *varId = varInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    }
    varInst = get_def_use_mgr()->GetDef(*varId);
  }
  return ptrInst;
}

ir::Instruction* MemPass::GetPtr(ir::Instruction* ip, uint32_t* varId) {
  const SpvOp op = ip->opcode();
  assert(op == SpvOpStore || op == SpvOpLoad);
  const uint32_t ptrId = ip->GetSingleWordInOperand(
      op == SpvOpStore ? kStorePtrIdInIdx : kLoadPtrIdInIdx);
  return GetPtr(ptrId, varId);
}

void MemPass::FindNamedOrDecoratedIds() {
  named_or_decorated_ids_.clear();
  for (auto& di : get_module()->debugs2())
    if (di.opcode() == SpvOpName)
      named_or_decorated_ids_.insert(di.GetSingleWordInOperand(0));
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == SpvOpDecorate || ai.opcode() == SpvOpDecorateId)
      named_or_decorated_ids_.insert(ai.GetSingleWordInOperand(0));
}

bool MemPass::HasOnlyNamesAndDecorates(uint32_t id) const {
  analysis::UseList* uses = get_def_use_mgr()->GetUses(id);
  if (uses == nullptr) return true;
  if (named_or_decorated_ids_.find(id) == named_or_decorated_ids_.end())
    return false;
  for (auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    if (op != SpvOpName && !IsNonTypeDecorate(op)) return false;
  }
  return true;
}

void MemPass::KillNamesAndDecorates(uint32_t id) {
  // TODO(greg-lunarg): Remove id from any OpGroupDecorate and
  // kill if no other operands.
  if (named_or_decorated_ids_.find(id) == named_or_decorated_ids_.end()) return;
  analysis::UseList* uses = get_def_use_mgr()->GetUses(id);
  if (uses == nullptr) return;
  std::list<ir::Instruction*> killList;
  for (auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    if (op == SpvOpName || IsNonTypeDecorate(op)) killList.push_back(u.inst);
  }
  for (auto kip : killList) get_def_use_mgr()->KillInst(kip);
}

void MemPass::KillNamesAndDecorates(ir::Instruction* inst) {
  const uint32_t rId = inst->result_id();
  if (rId == 0) return;
  KillNamesAndDecorates(rId);
}

void MemPass::KillAllInsts(ir::BasicBlock* bp) {
  bp->ForEachInst([this](ir::Instruction* ip) {
    KillNamesAndDecorates(ip);
    def_use_mgr_->KillInst(ip);
  });
}

bool MemPass::HasLoads(uint32_t varId) const {
  analysis::UseList* uses = get_def_use_mgr()->GetUses(varId);
  if (uses == nullptr) return false;
  for (auto u : *uses) {
    SpvOp op = u.inst->opcode();
    // TODO(): The following is slightly conservative. Could be
    // better handling of non-store/name.
    if (IsNonPtrAccessChain(op) || op == SpvOpCopyObject) {
      if (HasLoads(u.inst->result_id())) return true;
    } else if (op != SpvOpStore && op != SpvOpName)
      return true;
  }
  return false;
}

bool MemPass::IsLiveVar(uint32_t varId) const {
  const ir::Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  // assume live if not a variable eg. function parameter
  if (varInst->opcode() != SpvOpVariable) return true;
  // non-function scope vars are live
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) !=
      SpvStorageClassFunction)
    return true;
  // test if variable is loaded from
  return HasLoads(varId);
}

bool MemPass::IsLiveStore(ir::Instruction* storeInst) {
  // get store's variable
  uint32_t varId;
  (void)GetPtr(storeInst, &varId);
  return IsLiveVar(varId);
}

void MemPass::AddStores(uint32_t ptr_id, std::queue<ir::Instruction*>* insts) {
  analysis::UseList* uses = get_def_use_mgr()->GetUses(ptr_id);
  if (uses != nullptr) {
    for (auto u : *uses) {
      if (IsNonPtrAccessChain(u.inst->opcode()))
        AddStores(u.inst->result_id(), insts);
      else if (u.inst->opcode() == SpvOpStore)
        insts->push(u.inst);
    }
  }
}

void MemPass::DCEInst(ir::Instruction* inst) {
  std::queue<ir::Instruction*> deadInsts;
  deadInsts.push(inst);
  while (!deadInsts.empty()) {
    ir::Instruction* di = deadInsts.front();
    // Don't delete labels
    if (di->opcode() == SpvOpLabel) {
      deadInsts.pop();
      continue;
    }
    // Remember operands
    std::vector<uint32_t> ids;
    di->ForEachInId([&ids](uint32_t* iid) { ids.push_back(*iid); });
    uint32_t varId = 0;
    // Remember variable if dead load
    if (di->opcode() == SpvOpLoad) (void)GetPtr(di, &varId);
    KillNamesAndDecorates(di);
    get_def_use_mgr()->KillInst(di);
    // For all operands with no remaining uses, add their instruction
    // to the dead instruction queue.
    for (auto id : ids)
      if (HasOnlyNamesAndDecorates(id))
        deadInsts.push(get_def_use_mgr()->GetDef(id));
    // if a load was deleted and it was the variable's
    // last load, add all its stores to dead queue
    if (varId != 0 && !IsLiveVar(varId)) AddStores(varId, &deadInsts);
    deadInsts.pop();
  }
}

void MemPass::ReplaceAndDeleteLoad(ir::Instruction* loadInst, uint32_t replId) {
  const uint32_t loadId = loadInst->result_id();
  KillNamesAndDecorates(loadId);
  (void)get_def_use_mgr()->ReplaceAllUsesWith(loadId, replId);
  DCEInst(loadInst);
}

MemPass::MemPass() {}

bool MemPass::HasOnlySupportedRefs(uint32_t varId) {
  if (supported_ref_vars_.find(varId) != supported_ref_vars_.end()) return true;
  analysis::UseList* uses = get_def_use_mgr()->GetUses(varId);
  if (uses == nullptr) return true;
  for (auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    if (op != SpvOpStore && op != SpvOpLoad && op != SpvOpName &&
        !IsNonTypeDecorate(op))
      return false;
  }
  supported_ref_vars_.insert(varId);
  return true;
}

void MemPass::InitSSARewrite(ir::Function* func) {
  // Initialize function and block maps.
  id2block_.clear();
  block2structured_succs_.clear();
  for (auto& fn : *get_module())
    for (auto& blk : fn) id2block_[blk.id()] = &blk;

  // Clear collections.
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();
  visitedBlocks_.clear();
  type2undefs_.clear();
  supported_ref_vars_.clear();
  block2structured_succs_.clear();
  label2preds_.clear();
  label2ssa_map_.clear();
  phis_to_patch_.clear();

  // Init predecessors
  label2preds_.clear();
  for (auto& blk : *func) {
    uint32_t blkId = blk.id();
    blk.ForEachSuccessorLabel(
        [&blkId, this](uint32_t sbid) { label2preds_[sbid].push_back(blkId); });
  }

  // Collect target (and non-) variable sets. Remove variables with
  // non-load/store refs from target variable set
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      switch (inst.opcode()) {
        case SpvOpStore:
        case SpvOpLoad: {
          uint32_t varId;
          (void)GetPtr(&inst, &varId);
          if (!IsTargetVar(varId)) break;
          if (HasOnlySupportedRefs(varId)) break;
          seen_non_target_vars_.insert(varId);
          seen_target_vars_.erase(varId);
        } break;
        default:
          break;
      }
    }
  }
}

bool MemPass::IsLiveAfter(uint32_t var_id, uint32_t label) const {
  // For now, return very conservative result: true. This will result in
  // correct, but possibly usused, phi code to be generated. A subsequent
  // DCE pass should eliminate this code.
  // TODO(greg-lunarg): Return more accurate information
  (void)var_id;
  (void)label;
  return true;
}

void MemPass::SSABlockInitSinglePred(ir::BasicBlock* block_ptr) {
  // Copy map entry from single predecessor
  const uint32_t label = block_ptr->id();
  const uint32_t predLabel = label2preds_[label].front();
  assert(visitedBlocks_.find(predLabel) != visitedBlocks_.end());
  label2ssa_map_[label] = label2ssa_map_[predLabel];
}

uint32_t MemPass::Type2Undef(uint32_t type_id) {
  const auto uitr = type2undefs_.find(type_id);
  if (uitr != type2undefs_.end()) return uitr->second;
  const uint32_t undefId = TakeNextId();
  std::unique_ptr<ir::Instruction> undef_inst(
      new ir::Instruction(SpvOpUndef, type_id, undefId, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*undef_inst);
  get_module()->AddGlobalValue(std::move(undef_inst));
  type2undefs_[type_id] = undefId;
  return undefId;
}

void MemPass::SSABlockInitLoopHeader(
    std::list<ir::BasicBlock*>::iterator block_itr) {
  const uint32_t label = (*block_itr)->id();

  // Determine the back-edge label.
  uint32_t backLabel = 0;
  for (uint32_t predLabel : label2preds_[label])
    if (visitedBlocks_.find(predLabel) == visitedBlocks_.end()) {
      assert(backLabel == 0);
      backLabel = predLabel;
      break;
    }
  assert(backLabel != 0);

  // Determine merge block.
  auto mergeInst = (*block_itr)->end();
  --mergeInst;
  --mergeInst;
  uint32_t mergeLabel =
      mergeInst->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx);

  // Collect all live variables and a default value for each across all
  // non-backedge predecesors. Must be ordered map because phis are
  // generated based on order and test results will otherwise vary across
  // platforms.
  std::map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    for (auto var_val : label2ssa_map_[predLabel]) {
      uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // Add all stored variables in loop. Set their default value id to zero.
  for (auto bi = block_itr; (*bi)->id() != mergeLabel; ++bi) {
    ir::BasicBlock* bp = *bi;
    for (auto ii = bp->begin(); ii != bp->end(); ++ii) {
      if (ii->opcode() != SpvOpStore) {
        continue;
      }
      uint32_t varId;
      (void)GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId)) {
        continue;
      }
      liveVars[varId] = 0;
    }
  }
  // Insert phi for all live variables that require them. All variables
  // defined in loop require a phi. Otherwise all variables
  // with differing predecessor values require a phi.
  auto insertItr = (*block_itr)->begin();
  for (auto var_val : liveVars) {
    const uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label)) {
      continue;
    }
    const uint32_t val0Id = var_val.second;
    bool needsPhi = false;
    if (val0Id != 0) {
      for (uint32_t predLabel : label2preds_[label]) {
        // Skip back edge predecessor.
        if (predLabel == backLabel) continue;
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        // Missing (undef) values always cause difference with (defined) value
        if (var_val_itr == label2ssa_map_[predLabel].end()) {
          needsPhi = true;
          break;
        }
        if (var_val_itr->second != val0Id) {
          needsPhi = true;
          break;
        }
      }
    } else {
      needsPhi = true;
    }

    // If val is the same for all predecessors, enter it in map
    if (!needsPhi) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }

    // Val differs across predecessors. Add phi op to block and
    // add its result id to the map. For back edge predecessor,
    // use the variable id. We will patch this after visiting back
    // edge predecessor. For predecessors that do not define a value,
    // use undef.
    std::vector<ir::Operand> phi_in_operands;
    uint32_t typeId = GetPointeeTypeId(get_def_use_mgr()->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      uint32_t valId;
      if (predLabel == backLabel) {
        valId = varId;
      } else {
        const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
        if (var_val_itr == label2ssa_map_[predLabel].end())
          valId = Type2Undef(typeId);
        else
          valId = var_val_itr->second;
      }
      phi_in_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {valId}});
      phi_in_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {predLabel}});
    }
    const uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
        new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    // The only phis requiring patching are the ones we create.
    phis_to_patch_.insert(phiId);
    // Only analyze the phi define now; analyze the phi uses after the
    // phi backedge predecessor value is patched.
    get_def_use_mgr()->AnalyzeInstDef(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({varId, phiId});
  }
}

void MemPass::SSABlockInitMultiPred(ir::BasicBlock* block_ptr) {
  const uint32_t label = block_ptr->id();
  // Collect all live variables and a default value for each across all
  // predecesors. Must be ordered map because phis are generated based on
  // order and test results will otherwise vary across platforms.
  std::map<uint32_t, uint32_t> liveVars;
  for (uint32_t predLabel : label2preds_[label]) {
    assert(visitedBlocks_.find(predLabel) != visitedBlocks_.end());
    for (auto var_val : label2ssa_map_[predLabel]) {
      const uint32_t varId = var_val.first;
      liveVars[varId] = var_val.second;
    }
  }
  // For each live variable, look for a difference in values across
  // predecessors that would require a phi and insert one.
  auto insertItr = block_ptr->begin();
  for (auto var_val : liveVars) {
    const uint32_t varId = var_val.first;
    if (!IsLiveAfter(varId, label)) continue;
    const uint32_t val0Id = var_val.second;
    bool differs = false;
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // Missing values cause a difference because we'll need to create an
      // undef for that predecessor.
      if (var_val_itr == label2ssa_map_[predLabel].end()) {
        differs = true;
        break;
      }
      if (var_val_itr->second != val0Id) {
        differs = true;
        break;
      }
    }
    // If val is the same for all predecessors, enter it in map
    if (!differs) {
      label2ssa_map_[label].insert(var_val);
      continue;
    }
    // Val differs across predecessors. Add phi op to block and add its result
    // id to the map.
    std::vector<ir::Operand> phi_in_operands;
    const uint32_t typeId = GetPointeeTypeId(get_def_use_mgr()->GetDef(varId));
    for (uint32_t predLabel : label2preds_[label]) {
      const auto var_val_itr = label2ssa_map_[predLabel].find(varId);
      // If variable not defined on this path, use undef
      const uint32_t valId = (var_val_itr != label2ssa_map_[predLabel].end())
                                 ? var_val_itr->second
                                 : Type2Undef(typeId);
      phi_in_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {valId}});
      phi_in_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {predLabel}});
    }
    const uint32_t phiId = TakeNextId();
    std::unique_ptr<ir::Instruction> newPhi(
        new ir::Instruction(SpvOpPhi, typeId, phiId, phi_in_operands));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newPhi);
    insertItr = insertItr.InsertBefore(std::move(newPhi));
    ++insertItr;
    label2ssa_map_[label].insert({varId, phiId});
  }
}

void MemPass::SSABlockInit(std::list<ir::BasicBlock*>::iterator block_itr) {
  const size_t numPreds = label2preds_[(*block_itr)->id()].size();
  if (numPreds == 0) return;
  if (numPreds == 1)
    SSABlockInitSinglePred(*block_itr);
  else if (IsLoopHeader(*block_itr))
    SSABlockInitLoopHeader(block_itr);
  else
    SSABlockInitMultiPred(*block_itr);
}

bool MemPass::IsTargetVar(uint32_t varId) {
  if (seen_non_target_vars_.find(varId) != seen_non_target_vars_.end())
    return false;
  if (seen_target_vars_.find(varId) != seen_target_vars_.end()) return true;
  const ir::Instruction* varInst = get_def_use_mgr()->GetDef(varId);
  if (varInst->opcode() != SpvOpVariable) return false;
  ;
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  if (varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) !=
      SpvStorageClassFunction) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
      varTypeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx);
  ir::Instruction* varPteTypeInst = get_def_use_mgr()->GetDef(varPteTypeId);
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

void MemPass::PatchPhis(uint32_t header_id, uint32_t back_id) {
  ir::BasicBlock* header = id2block_[header_id];
  auto phiItr = header->begin();
  for (; phiItr->opcode() == SpvOpPhi; ++phiItr) {
    // Only patch phis that we created in a loop header.
    // There might be other phis unrelated to our optimizations.
    if (0 == phis_to_patch_.count(phiItr->result_id())) continue;

    // Find phi operand index for back edge
    uint32_t cnt = 0;
    uint32_t idx = phiItr->NumInOperands();
    phiItr->ForEachInId([&cnt, &back_id, &idx](uint32_t* iid) {
      if (cnt % 2 == 1 && *iid == back_id) idx = cnt - 1;
      ++cnt;
    });
    assert(idx != phiItr->NumInOperands());
    // Replace temporary phi operand with variable's value in backedge block
    // map. Use undef if variable not in map.
    const uint32_t varId = phiItr->GetSingleWordInOperand(idx);
    const auto valItr = label2ssa_map_[back_id].find(varId);
    uint32_t valId =
        (valItr != label2ssa_map_[back_id].end())
            ? valItr->second
            : Type2Undef(GetPointeeTypeId(get_def_use_mgr()->GetDef(varId)));
    phiItr->SetInOperand(idx, {valId});
    // Analyze uses now that they are complete
    get_def_use_mgr()->AnalyzeInstUse(&*phiItr);
  }
}

Pass::Status MemPass::InsertPhiInstructions(ir::Function* func) {
  // TODO(dnovillo) the current Phi placement mechanism assumes structured
  // control-flow. This should be generalized
  // (https://github.com/KhronosGroup/SPIRV-Tools/issues/893).
  if (!get_module()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  // Collect all named and decorated ids.
  FindNamedOrDecoratedIds();

  // Initialize the data structures used to insert Phi instructions.
  InitSSARewrite(func);

  // Process all blocks in structured order. This is just one way (the
  // simplest?) to make sure all predecessors blocks are processed before
  // a block itself.
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    // Skip pseudo entry block
    if (*bi == &pseudo_entry_block_) continue;
    // Initialize this block's label2ssa_map_ entry using predecessor maps.
    // Then process all stores and loads of targeted variables.
    SSABlockInit(bi);
    ir::BasicBlock* bp = *bi;
    const uint32_t label = bp->id();
    for (auto ii = bp->begin(); ii != bp->end(); ++ii) {
      switch (ii->opcode()) {
        case SpvOpStore: {
          uint32_t varId;
          (void)GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) break;
          // Register new stored value for the variable
          label2ssa_map_[label][varId] =
              ii->GetSingleWordInOperand(kStoreValIdInIdx);
        } break;
        case SpvOpLoad: {
          uint32_t varId;
          (void)GetPtr(&*ii, &varId);
          if (!IsTargetVar(varId)) break;
          uint32_t replId = 0;
          const auto ssaItr = label2ssa_map_.find(label);
          if (ssaItr != label2ssa_map_.end()) {
            const auto valItr = ssaItr->second.find(varId);
            if (valItr != ssaItr->second.end()) replId = valItr->second;
          }
          // If variable is not defined, use undef
          if (replId == 0) {
            replId = Type2Undef(GetPointeeTypeId(get_def_use_mgr()->GetDef(varId)));
          }
          // Replace load's id with the last stored value id for variable
          // and delete load. Kill any names or decorates using id before
          // replacing to prevent incorrect replacement in those instructions.
          const uint32_t loadId = ii->result_id();
          KillNamesAndDecorates(loadId);
          (void)get_def_use_mgr()->ReplaceAllUsesWith(loadId, replId);
          get_def_use_mgr()->KillInst(&*ii);
        } break;
        default: { } break; }
    }
    visitedBlocks_.insert(label);
    // Look for successor backedge and patch phis in loop header
    // if found.
    uint32_t header = 0;
    bp->ForEachSuccessorLabel([&header, this](uint32_t succ) {
      if (visitedBlocks_.find(succ) == visitedBlocks_.end()) return;
      assert(header == 0);
      header = succ;
    });
    if (header != 0) PatchPhis(header, label);
  }

  return Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools

