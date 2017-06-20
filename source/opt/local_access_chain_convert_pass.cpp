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

#include "iterator.h"
#include "local_access_chain_convert_pass.h"

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvStorePtrId = 0;
static const int kSpvStoreValId = 1;
static const int kSpvLoadPtrId = 0;
static const int kSpvAccessChainPtrId = 0;
static const int kSpvTypePointerStorageClass = 0;
static const int kSpvTypePointerTypeId = 1;
static const int kSpvConstantValue = 0;
static const int kSpvTypeIntWidth = 0;

namespace spvtools {
namespace opt {

bool LocalAccessChainConvertPass::IsNonPtrAccessChain(
    const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

bool LocalAccessChainConvertPass::IsMathType(
    const ir::Instruction* typeInst) const {
  switch (typeInst->opcode()) {
  case SpvOpTypeInt:
  case SpvOpTypeFloat:
  case SpvOpTypeBool:
  case SpvOpTypeVector:
  case SpvOpTypeMatrix:
    return true;
  default:
    break;
  }
  return false;
}

bool LocalAccessChainConvertPass::IsTargetType(
    const ir::Instruction* typeInst) const {
  if (IsMathType(typeInst))
    return true;
  if (typeInst->opcode() == SpvOpTypeArray)
    return IsMathType(def_use_mgr_->GetDef(typeInst->GetSingleWordOperand(1)));
  if (typeInst->opcode() != SpvOpTypeStruct)
    return false;
  // All struct members must be math type
  int nonMathComp = 0;
  typeInst->ForEachInId([&nonMathComp,this](const uint32_t* tid) {
    ir::Instruction* compTypeInst = def_use_mgr_->GetDef(*tid);
    if (!IsMathType(compTypeInst)) ++nonMathComp;
  });
  return nonMathComp == 0;
}

ir::Instruction* LocalAccessChainConvertPass::GetPtr(
    ir::Instruction* ip,
    uint32_t* varId) {
  const uint32_t ptrId = ip->GetSingleWordInOperand(
    ip->opcode() == SpvOpStore ? kSpvStorePtrId : kSpvLoadPtrId);
  ir::Instruction* ptrInst = def_use_mgr_->GetDef(ptrId);
  *varId = IsNonPtrAccessChain(ptrInst->opcode()) ?
    ptrInst->GetSingleWordInOperand(kSpvAccessChainPtrId) :
    ptrId;
  return ptrInst;
}

bool LocalAccessChainConvertPass::IsTargetVar(uint32_t varId) {
  if (seen_non_target_vars_.find(varId) != seen_non_target_vars_.end())
    return false;
  if (seen_target_vars_.find(varId) != seen_target_vars_.end())
    return true;
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  if (varInst->opcode() != SpvOpVariable)
    return false;;
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  if (varTypeInst->GetSingleWordInOperand(kSpvTypePointerStorageClass) !=
    SpvStorageClassFunction) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  const uint32_t varPteTypeId =
    varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeId);
  ir::Instruction* varPteTypeInst = def_use_mgr_->GetDef(varPteTypeId);
  if (!IsTargetType(varPteTypeInst)) {
    seen_non_target_vars_.insert(varId);
    return false;
  }
  seen_target_vars_.insert(varId);
  return true;
}

void LocalAccessChainConvertPass::DeleteIfUseless(ir::Instruction* inst) {
  const uint32_t resId = inst->result_id();
  assert(resId != 0);
  analysis::UseList* uses = def_use_mgr_->GetUses(resId);
  if (uses == nullptr)
    def_use_mgr_->KillInst(inst);
}

void LocalAccessChainConvertPass::ReplaceAndDeleteLoad(
    ir::Instruction* loadInst,
    uint32_t replId,
    ir::Instruction* ptrInst) {
  const uint32_t loadId = loadInst->result_id();
  (void) def_use_mgr_->ReplaceAllUsesWith(loadId, replId);
  // remove load instruction
  def_use_mgr_->KillInst(loadInst);
  // if access chain, see if it can be removed as well
  if (IsNonPtrAccessChain(ptrInst->opcode())) {
    DeleteIfUseless(ptrInst);
  }
}

uint32_t LocalAccessChainConvertPass::GetPointeeTypeId(
    const ir::Instruction* ptrInst) const {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const ir::Instruction* ptrTypeInst = def_use_mgr_->GetDef(ptrTypeId);
  return ptrTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeId);
}

void LocalAccessChainConvertPass::BuildAndAppendInst(
    SpvOp opcode,
    uint32_t typeId,
    uint32_t resultId,
    const std::vector<ir::Operand>& in_opnds,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts) {
  std::unique_ptr<ir::Instruction> newInst(new ir::Instruction(
      opcode, typeId, resultId, in_opnds));
  def_use_mgr_->AnalyzeInstDefUse(&*newInst);
  newInsts->emplace_back(std::move(newInst));
}

uint32_t LocalAccessChainConvertPass::BuildAndAppendVarLoad(
    const ir::Instruction* ptrInst,
    uint32_t* varId,
    uint32_t* varPteTypeId,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts) {
  const uint32_t ldResultId = TakeNextId();
  *varId = ptrInst->GetSingleWordInOperand(kSpvAccessChainPtrId);
  const ir::Instruction* varInst = def_use_mgr_->GetDef(*varId);
  assert(varInst->opcode() == SpvOpVariable);
  *varPteTypeId = GetPointeeTypeId(varInst);
  BuildAndAppendInst(SpvOpLoad, *varPteTypeId, ldResultId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {*varId}}}, newInsts);
  return ldResultId;
}

void LocalAccessChainConvertPass::AppendConstantOperands(
    const ir::Instruction* ptrInst,
    std::vector<ir::Operand>* in_opnds) {
  uint32_t iidIdx = 0;
  ptrInst->ForEachInId([&iidIdx, &in_opnds, this](const uint32_t *iid) {
    if (iidIdx > 0) {
      const ir::Instruction* cInst = def_use_mgr_->GetDef(*iid);
      uint32_t val = cInst->GetSingleWordInOperand(kSpvConstantValue);
      in_opnds->push_back(
        {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {val}});
    }
    ++iidIdx;
  });
}

uint32_t LocalAccessChainConvertPass::GenAccessChainLoadReplacement(
    const ir::Instruction* ptrInst,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts) {

  // Build and append load of variable in ptrInst
  uint32_t varId;
  uint32_t varPteTypeId;
  const uint32_t ldResultId = BuildAndAppendVarLoad(ptrInst, &varId,
                                                    &varPteTypeId, newInsts);

  // Build and append Extract
  const uint32_t extResultId = TakeNextId();
  const uint32_t ptrPteTypeId = GetPointeeTypeId(ptrInst);
  std::vector<ir::Operand> ext_in_opnds = 
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ldResultId}}};
  AppendConstantOperands(ptrInst, &ext_in_opnds);
  BuildAndAppendInst(SpvOpCompositeExtract, ptrPteTypeId, extResultId, 
                     ext_in_opnds, newInsts);
  return extResultId;
}

void LocalAccessChainConvertPass::GenAccessChainStoreReplacement(
    const ir::Instruction* ptrInst,
    uint32_t valId,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts) {

  // Build and append load of variable in ptrInst
  uint32_t varId;
  uint32_t varPteTypeId;
  const uint32_t ldResultId = BuildAndAppendVarLoad(ptrInst, &varId,
                                                    &varPteTypeId, newInsts);

  // Build and append Insert
  const uint32_t insResultId = TakeNextId();
  std::vector<ir::Operand> ins_in_opnds = 
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {valId}}, 
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ldResultId}}};
  AppendConstantOperands(ptrInst, &ins_in_opnds);
  BuildAndAppendInst(
      SpvOpCompositeInsert, varPteTypeId, insResultId, ins_in_opnds, newInsts);

  // Build and append Store
  BuildAndAppendInst(SpvOpStore, 0, 0, 
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {varId}}, 
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {insResultId}}},
      newInsts);
}

bool LocalAccessChainConvertPass::IsConstantIndexAccessChain(
    const ir::Instruction* acp) const {
  uint32_t inIdx = 0;
  uint32_t nonConstCnt = 0;
  acp->ForEachInId([&inIdx, &nonConstCnt, this](const uint32_t* tid) {
    if (inIdx > 0) {
      ir::Instruction* opInst = def_use_mgr_->GetDef(*tid);
      if (opInst->opcode() != SpvOpConstant) ++nonConstCnt;
    }
    ++inIdx;
  });
  return nonConstCnt == 0;
}

void LocalAccessChainConvertPass::FindTargetVars(ir::Function* func) {
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpStore:
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        // For now, only convert non-ptr access chains
        if (!IsNonPtrAccessChain(ptrInst->opcode()))
          break;
        // For now, only convert non-nested access chains
        // TODO(): Convert nested access chains
        if (!IsTargetVar(varId))
          break;
        // Rule out variables accessed with non-constant indices
        if (!IsConstantIndexAccessChain(ptrInst)) {
          seen_non_target_vars_.insert(varId);
          seen_target_vars_.erase(varId);
          break;
        }
      } break;
      default:
        break;
      }
    }
  }
}

bool LocalAccessChainConvertPass::ConvertLocalAccessChains(ir::Function* func) {
  FindTargetVars(func);
  // Replace access chains of all targeted variables with equivalent
  // extract and insert sequences
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
      case SpvOpLoad: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsNonPtrAccessChain(ptrInst->opcode()))
          break;
        if (!IsTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        uint32_t replId =
            GenAccessChainLoadReplacement(ptrInst, &newInsts);
        ReplaceAndDeleteLoad(&*ii, replId, ptrInst);
        ++ii;
        ii = ii.InsertBefore(&newInsts);
        ++ii;
        modified = true;
      } break;
      case SpvOpStore: {
        uint32_t varId;
        ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
        if (!IsNonPtrAccessChain(ptrInst->opcode()))
          break;
        if (!IsTargetVar(varId))
          break;
        std::vector<std::unique_ptr<ir::Instruction>> newInsts;
        uint32_t valId = ii->GetSingleWordInOperand(kSpvStoreValId);
        GenAccessChainStoreReplacement(ptrInst, valId, &newInsts);
        def_use_mgr_->KillInst(&*ii);
        DeleteIfUseless(ptrInst);
        ++ii;
        ii = ii.InsertBefore(&newInsts);
        ++ii;
        ++ii;
        modified = true;
      } break;
      default:
        break;
      }
    }
  }
  return modified;
}

void LocalAccessChainConvertPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize function and block maps
  id2function_.clear();
  for (auto& fn : *module_) 
    id2function_[fn.result_id()] = &fn;

  // Initialize Target Variable Caches
  seen_target_vars_.clear();
  seen_non_target_vars_.clear();

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize next unused Id.
  next_id_ = module->id_bound();
};

Pass::Status LocalAccessChainConvertPass::ProcessImpl() {
  // If non-32-bit integer type in module, terminate processing
  // TODO(): Handle non-32-bit integer constants in access chains
  for (const ir::Instruction& inst : module_->types_values())
    if (inst.opcode() == SpvOpTypeInt &&
        inst.GetSingleWordInOperand(kSpvTypeIntWidth) != 32)
      return Status::SuccessWithoutChange;
  // Process all entry point functions.
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordOperand(kSpvEntryPointFunctionId)];
    modified = ConvertLocalAccessChains(fn) || modified;
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalAccessChainConvertPass::LocalAccessChainConvertPass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status LocalAccessChainConvertPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

