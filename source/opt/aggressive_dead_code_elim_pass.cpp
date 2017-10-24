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

#include "cfa.h"
#include "iterator.h"
#include "spirv/1.0/GLSL.std.450.h"

#include <stack>

namespace spvtools {
namespace opt {

namespace {

const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kExtInstSetIdInIndx = 0;
const uint32_t kExtInstInstructionInIndx = 1;
const uint32_t kEntryPointFunctionIdInIdx = 1;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kLoopMergeContinueBlockIdInIdx = 1;

}  // namespace anonymous

bool AggressiveDCEPass::IsVarOfStorage(uint32_t varId, 
      uint32_t storageClass) {
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  const SpvOp op = varInst->opcode();
  if (op != SpvOpVariable) 
    return false;
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  if (varTypeInst->opcode() != SpvOpTypePointer)
    return false;
  return varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) ==
      storageClass;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId) {
 return IsVarOfStorage(varId, SpvStorageClassFunction) ||
        (IsVarOfStorage(varId, SpvStorageClassPrivate) && private_like_local_);
}

void AggressiveDCEPass::AddStores(uint32_t ptrId) {
  const analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
  if (uses == nullptr)
    return;
  for (const auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    switch (op) {
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpCopyObject: {
        AddStores(u.inst->result_id());
      } break;
      case SpvOpLoad:
        break;
      // If default, assume it stores eg frexp, modf, function call
      case SpvOpStore:
      default: {
        if (!IsLive(u.inst))
          AddToWorklist(u.inst);
      } break;
    }
  }
}

bool AggressiveDCEPass::IsCombinator(uint32_t op) const {
  return combinator_ops_shader_.find(op) != combinator_ops_shader_.end();
}

bool AggressiveDCEPass::IsCombinatorExt(ir::Instruction* inst) const {
  assert(inst->opcode() == SpvOpExtInst);
  if (inst->GetSingleWordInOperand(kExtInstSetIdInIndx) == glsl_std_450_id_) {
    uint32_t op = inst->GetSingleWordInOperand(kExtInstInstructionInIndx);
    return combinator_ops_glsl_std_450_.find(op) !=
        combinator_ops_glsl_std_450_.end();
  }
  else
    return false;
}

bool AggressiveDCEPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : module_->extensions()) {
    const char* extName = reinterpret_cast<const char*>(
        &ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

bool AggressiveDCEPass::KillInstIfTargetDead(ir::Instruction* inst) {
  const uint32_t tId = inst->GetSingleWordInOperand(0);
  const ir::Instruction* tInst = def_use_mgr_->GetDef(tId);
  if (dead_insts_.find(tInst) != dead_insts_.end()) {
    def_use_mgr_->KillInst(inst);
    return true;
  }
  return false;
}

void AggressiveDCEPass::ProcessLoad(uint32_t varId) {
  // Only process locals
  if (!IsLocalVar(varId))
    return;
  // Return if already processed
  if (live_local_vars_.find(varId) != live_local_vars_.end()) 
    return;
  // Mark all stores to varId as live
  AddStores(varId);
  // Cache varId as processed
  live_local_vars_.insert(varId);
}

uint32_t AggressiveDCEPass::MergeBlockIdIfAny(
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

void AggressiveDCEPass::ComputeStructuredSuccessors(ir::Function* func) {
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

void AggressiveDCEPass::ComputeStructuredOrder(
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

bool AggressiveDCEPass::IsStructuredIfHeader(ir::BasicBlock* bp,
      ir::Instruction** mergeInst, ir::Instruction** branchInst,
      uint32_t* mergeBlockId) {
  auto ii = bp->end();
  --ii;
  if (ii->opcode() != SpvOpBranchConditional)
    return false;
  if (ii == bp->begin())
    return false;
  *branchInst = &*ii;
  --ii;
  if (ii->opcode() != SpvOpSelectionMerge)
    return false;
  *mergeInst = &*ii;
  *mergeBlockId = ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
  return true;
}

void AggressiveDCEPass::ComputeBlock2BranchMaps(
      std::list<ir::BasicBlock*>& structuredOrder) {
  block2merge_.clear();
  block2branch_.clear();
  std::stack<ir::Instruction*> currentMergeInst;
  std::stack<ir::Instruction*> currentBranchInst;
  std::stack<uint32_t> currentMergeBlockId;
  currentMergeInst.push(nullptr);
  currentBranchInst.push(nullptr);
  currentMergeBlockId.push(0);
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    if ((*bi)->id() == currentMergeBlockId.top()) {
      currentMergeBlockId.pop();
      currentMergeInst.pop();
      currentBranchInst.pop();
    }
    block2merge_[*bi] = currentMergeInst.top();
    block2branch_[*bi] = currentBranchInst.top();
    ir::Instruction* mergeInst;
    ir::Instruction* branchInst;
    uint32_t mergeBlockId;
    if (IsStructuredIfHeader(*bi, &mergeInst, &branchInst, &mergeBlockId)) {
      currentMergeBlockId.push(mergeBlockId);
      currentMergeInst.push(mergeInst);
      currentBranchInst.push(branchInst);
    }
  }
}

void AggressiveDCEPass::ComputeInst2BlockMap(ir::Function* func) {
  for (auto& blk : *func) {
    blk.ForEachInst([&blk,this](ir::Instruction* ip) {
      inst2block_[ip] = &blk;
    });
  }
}

void AggressiveDCEPass::AddBranch(uint32_t labelId, ir::BasicBlock* bp) {
  std::unique_ptr<ir::Instruction> newBranch(
    new ir::Instruction(SpvOpBranch, 0, 0,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {labelId}}}));
  def_use_mgr_->AnalyzeInstDefUse(&*newBranch);
  bp->AddInstruction(std::move(newBranch));
}

bool AggressiveDCEPass::AggressiveDCE(ir::Function* func) {
  // Compute map from instruction to block
  ComputeInst2BlockMap(func);
  // Compute map from block to controlling conditional branch
  std::list<ir::BasicBlock*> structuredOrder;
  ComputeStructuredOrder(func, &structuredOrder);
  ComputeBlock2BranchMaps(structuredOrder);
  bool modified = false;
  // Add all control flow and instructions with external side effects 
  // to worklist
  // TODO(greg-lunarg): Handle Frexp, Modf more optimally
  call_in_func_ = false;
  func_is_entry_point_ = false;
  private_stores_.clear();
  std::stack<bool> skip_control_flow;
  std::stack<uint32_t> currentMergeBlockId;
  skip_control_flow.push(false);
  currentMergeBlockId.push(false);
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    if ((*bi)->id() == currentMergeBlockId.top()) {
      skip_control_flow.pop();
      currentMergeBlockId.pop();
    }
    for (auto ii = (*bi)->begin(); ii != (*bi)->end(); ++ii) {
      SpvOp op = ii->opcode();
      switch (op) {
        case SpvOpStore: {
          uint32_t varId;
          (void) GetPtr(&*ii, &varId);
          // Mark stores as live if their variable is not function scope
          // and is not private scope. Remember private stores for possible
          // later inclusion
          if (IsVarOfStorage(varId, SpvStorageClassPrivate))
            private_stores_.push_back(&*ii);
          else if (!IsVarOfStorage(varId, SpvStorageClassFunction))
            AddToWorklist(&*ii);
        } break;
        case SpvOpExtInst: {
          // eg. GLSL frexp, modf
          if (!IsCombinatorExt(&*ii))
            AddToWorklist(&*ii);
        } break;
        case SpvOpLoopMerge: {
          skip_control_flow.push(false);
          currentMergeBlockId.push(
              ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx));
          AddToWorklist(&*ii);
        } break;
        case SpvOpSelectionMerge: {
          auto bri = ii;
          ++bri;
          bool is_structured_if = bri->opcode() == SpvOpBranchConditional;
          skip_control_flow.push(is_structured_if);
          currentMergeBlockId.push(
              ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx));
          if (!is_structured_if)
            AddToWorklist(&*ii);
        } break;
        case SpvOpBranch:
        case SpvOpBranchConditional: {
          if (!skip_control_flow.top())
            AddToWorklist(&*ii);
        } break;
        default: {
          // Function calls, atomics, function params, function returns, etc.
          // TODO(greg-lunarg): function calls live only if write to non-local
          if (!IsCombinator(op))
            AddToWorklist(&*ii);
          // Remember function calls
          if (op == SpvOpFunctionCall)
            call_in_func_ = true;
        } break;
      }
    }
  }
  // See if current function is an entry point
  for (auto& ei : module_->entry_points()) {
    if (ei.GetSingleWordInOperand(kEntryPointFunctionIdInIdx) ==
        func->result_id()) {
      func_is_entry_point_ = true;
      break;
    }
  }
  // If the current function is an entry point and has no function calls,
  // we can optimize private variables as locals
  private_like_local_ = func_is_entry_point_ && !call_in_func_;
  // If privates are not like local, add their stores to worklist
  if (!private_like_local_)
    for (auto& ps : private_stores_)
      AddToWorklist(ps);
  // Add OpGroupDecorates to worklist because they are a pain to remove
  // ids from.
  // TODO(greg-lunarg): Handle dead ids in OpGroupDecorate
  for (auto& ai : module_->annotations()) {
    if (ai.opcode() == SpvOpGroupDecorate)
      AddToWorklist(&ai);
  }
  // Perform closure on live instruction set. 
  while (!worklist_.empty()) {
    ir::Instruction* liveInst = worklist_.front();
    // Add all operand instructions if not already live
    liveInst->ForEachInId([this](const uint32_t* iid) {
      ir::Instruction* inInst = def_use_mgr_->GetDef(*iid);
      if (!IsLive(inInst))
        AddToWorklist(inInst);
    });
    // If in a structured if construct, add the controlling conditional branch
    // and its merge
    ir::BasicBlock* blk = inst2block_[liveInst];
    ir::Instruction* branchInst = block2branch_[blk];
    if (branchInst != nullptr && !IsLive(branchInst)) {
      AddToWorklist(branchInst);
      AddToWorklist(block2merge_[blk]);
    }
    // If local load, add all variable's stores if variable not already live
    if (liveInst->opcode() == SpvOpLoad) {
      uint32_t varId;
      (void) GetPtr(liveInst, &varId);
      ProcessLoad(varId);
    }
    // If function call, treat as if it loads from all pointer arguments
    else if (liveInst->opcode() == SpvOpFunctionCall) {
      liveInst->ForEachInId([this](const uint32_t* iid) {
        // Skip non-ptr args
        if (!IsPtr(*iid)) return;
        uint32_t varId;
        (void) GetPtr(*iid, &varId);
        ProcessLoad(varId);
      });
    }
    // If function parameter, treat as if it's result id is loaded from
    else if (liveInst->opcode() == SpvOpFunctionParameter) {
      ProcessLoad(liveInst->result_id());
    }
    worklist_.pop();
  }
  // Mark all non-live instructions dead, except non-if branches
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end(); ++bi) {
    for (auto ii = (*bi)->begin(); ii != (*bi)->end(); ++ii) {
      if (IsLive(&*ii))
        continue;
      if (IsBranch(ii->opcode())) {
        if (ii == (*bi)->begin())
          continue;
        auto mi = ii;
        --mi;
        if (mi->opcode() != SpvOpSelectionMerge)
          continue;
      }
      dead_insts_.insert(&*ii);
    }
  }
  // Remove debug and annotation statements referencing dead instructions.
  // This must be done before killing the instructions, otherwise there are
  // dead objects in the def/use database.
  for (auto& di : module_->debugs2()) {
    if (di.opcode() != SpvOpName)
      continue;
    if (KillInstIfTargetDead(&di))
      modified = true;
  }
  for (auto& ai : module_->annotations()) {
    if (ai.opcode() != SpvOpDecorate && ai.opcode() != SpvOpDecorateId)
      continue;
    if (KillInstIfTargetDead(&ai))
      modified = true;
  }
  // Kill dead instructions and remember dead blocks
  std::unordered_set<ir::BasicBlock*> dead_blocks;
  for (auto bi = structuredOrder.begin(); bi != structuredOrder.end();) {
    uint32_t mergeBlockId = 0;
    for (auto ii = (*bi)->begin(); ii != (*bi)->end(); ++ii) {
      if (dead_insts_.find(&*ii) == dead_insts_.end())
        continue;
      // If dead instruction is selection merge, remember merge block
      // for new branch at end of block
      if (ii->opcode() == SpvOpSelectionMerge)
        mergeBlockId = 
            ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
      def_use_mgr_->KillInst(&*ii);
      modified = true;
    }
    // If a structured if was deleted, add a branch to its merge block,
    // and traverse to the merge block killing all instructions on the way
    // and remembering dead blocks for later deletion from function.
    if (mergeBlockId != 0) {
      AddBranch(mergeBlockId, *bi);
      for (++bi; (*bi)->id() != mergeBlockId; ++bi) {
        KillAllInsts(*bi);
        dead_blocks.insert(*bi);
      }
    }
    else {
      ++bi;
    }
  }
  // Erase dead blocks from function
  for (auto ebi = func->begin(); ebi != func->end(); )
  if (dead_blocks.find(&*ebi) != dead_blocks.end())
    ebi = ebi.Erase();
  else
    ++ebi;
  return modified;
}

void AggressiveDCEPass::Initialize(ir::Module* module) {
  module_ = module;

  // Clear collections
  worklist_ = std::queue<ir::Instruction*>{};
  live_insts_.clear();
  live_local_vars_.clear();
  dead_insts_.clear();
  combinator_ops_shader_.clear();
  combinator_ops_glsl_std_450_.clear();
  block2structured_succs_.clear();

  // Initialize block map
  id2block_.clear();
  for (auto& fn : *module_)
    for (auto& blk : fn)
      id2block_[blk.id()] = &blk;

  // TODO(greg-lunarg): Reuse def/use from previous passes
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));

  // Initialize extensions whitelist
  InitExtensions();
}

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes shader capability 
  // TODO(greg-lunarg): Handle additional capabilities
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;
  // Current functionality assumes logical addressing only
  // TODO(greg-lunarg): Handle non-logical addressing
  if (module_->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;
  // If any extensions in the module are not explicitly supported,
  // return unmodified. 
  if (!AllExtensionsSupported())
    return Status::SuccessWithoutChange;
  // Initialize combinator whitelists
  InitCombinatorSets();
  // Process all entry point functions
  ProcessFunction pfn = [this](ir::Function* fp) {
    return AggressiveDCE(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, module_);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

AggressiveDCEPass::AggressiveDCEPass() {}

Pass::Status AggressiveDCEPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

void AggressiveDCEPass::InitCombinatorSets() {
  combinator_ops_shader_ = {
    SpvOpNop,
    SpvOpUndef,
    SpvOpVariable,
    SpvOpImageTexelPointer,
    SpvOpLoad,
    SpvOpAccessChain,
    SpvOpInBoundsAccessChain,
    SpvOpArrayLength,
    SpvOpVectorExtractDynamic,
    SpvOpVectorInsertDynamic,
    SpvOpVectorShuffle,
    SpvOpCompositeConstruct,
    SpvOpCompositeExtract,
    SpvOpCompositeInsert,
    SpvOpCopyObject,
    SpvOpTranspose,
    SpvOpSampledImage,
    SpvOpImageSampleImplicitLod,
    SpvOpImageSampleExplicitLod,
    SpvOpImageSampleDrefImplicitLod,
    SpvOpImageSampleDrefExplicitLod,
    SpvOpImageSampleProjImplicitLod,
    SpvOpImageSampleProjExplicitLod,
    SpvOpImageSampleProjDrefImplicitLod,
    SpvOpImageSampleProjDrefExplicitLod,
    SpvOpImageFetch,
    SpvOpImageGather,
    SpvOpImageDrefGather,
    SpvOpImageRead,
    SpvOpImage,
    SpvOpConvertFToU,
    SpvOpConvertFToS,
    SpvOpConvertSToF,
    SpvOpConvertUToF,
    SpvOpUConvert,
    SpvOpSConvert,
    SpvOpFConvert,
    SpvOpQuantizeToF16,
    SpvOpBitcast,
    SpvOpSNegate,
    SpvOpFNegate,
    SpvOpIAdd,
    SpvOpFAdd,
    SpvOpISub,
    SpvOpFSub,
    SpvOpIMul,
    SpvOpFMul,
    SpvOpUDiv,
    SpvOpSDiv,
    SpvOpFDiv,
    SpvOpUMod,
    SpvOpSRem,
    SpvOpSMod,
    SpvOpFRem,
    SpvOpFMod,
    SpvOpVectorTimesScalar,
    SpvOpMatrixTimesScalar,
    SpvOpVectorTimesMatrix,
    SpvOpMatrixTimesVector,
    SpvOpMatrixTimesMatrix,
    SpvOpOuterProduct,
    SpvOpDot,
    SpvOpIAddCarry,
    SpvOpISubBorrow,
    SpvOpUMulExtended,
    SpvOpSMulExtended,
    SpvOpAny,
    SpvOpAll,
    SpvOpIsNan,
    SpvOpIsInf,
    SpvOpLogicalEqual,
    SpvOpLogicalNotEqual,
    SpvOpLogicalOr,
    SpvOpLogicalAnd,
    SpvOpLogicalNot,
    SpvOpSelect,
    SpvOpIEqual,
    SpvOpINotEqual,
    SpvOpUGreaterThan,
    SpvOpSGreaterThan,
    SpvOpUGreaterThanEqual,
    SpvOpSGreaterThanEqual,
    SpvOpULessThan,
    SpvOpSLessThan,
    SpvOpULessThanEqual,
    SpvOpSLessThanEqual,
    SpvOpFOrdEqual,
    SpvOpFUnordEqual,
    SpvOpFOrdNotEqual,
    SpvOpFUnordNotEqual,
    SpvOpFOrdLessThan,
    SpvOpFUnordLessThan,
    SpvOpFOrdGreaterThan,
    SpvOpFUnordGreaterThan,
    SpvOpFOrdLessThanEqual,
    SpvOpFUnordLessThanEqual,
    SpvOpFOrdGreaterThanEqual,
    SpvOpFUnordGreaterThanEqual,
    SpvOpShiftRightLogical,
    SpvOpShiftRightArithmetic,
    SpvOpShiftLeftLogical,
    SpvOpBitwiseOr,
    SpvOpBitwiseXor,
    SpvOpBitwiseAnd,
    SpvOpNot,
    SpvOpBitFieldInsert,
    SpvOpBitFieldSExtract,
    SpvOpBitFieldUExtract,
    SpvOpBitReverse,
    SpvOpBitCount,
    SpvOpDPdx,
    SpvOpDPdy,
    SpvOpFwidth,
    SpvOpDPdxFine,
    SpvOpDPdyFine,
    SpvOpFwidthFine,
    SpvOpDPdxCoarse,
    SpvOpDPdyCoarse,
    SpvOpFwidthCoarse,
    SpvOpPhi,
    SpvOpImageSparseSampleImplicitLod,
    SpvOpImageSparseSampleExplicitLod,
    SpvOpImageSparseSampleDrefImplicitLod,
    SpvOpImageSparseSampleDrefExplicitLod,
    SpvOpImageSparseSampleProjImplicitLod,
    SpvOpImageSparseSampleProjExplicitLod,
    SpvOpImageSparseSampleProjDrefImplicitLod,
    SpvOpImageSparseSampleProjDrefExplicitLod,
    SpvOpImageSparseFetch,
    SpvOpImageSparseGather,
    SpvOpImageSparseDrefGather,
    SpvOpImageSparseTexelsResident,
    SpvOpImageSparseRead,
    SpvOpSizeOf
    // TODO(dneto): Add instructions enabled by ImageQuery
  };

  // Find supported extension instruction set ids
  glsl_std_450_id_ = module_->GetExtInstImportId("GLSL.std.450");

  combinator_ops_glsl_std_450_ = {
    GLSLstd450Round,
    GLSLstd450RoundEven,
    GLSLstd450Trunc,
    GLSLstd450FAbs,
    GLSLstd450SAbs,
    GLSLstd450FSign,
    GLSLstd450SSign,
    GLSLstd450Floor,
    GLSLstd450Ceil,
    GLSLstd450Fract,
    GLSLstd450Radians,
    GLSLstd450Degrees,
    GLSLstd450Sin,
    GLSLstd450Cos,
    GLSLstd450Tan,
    GLSLstd450Asin,
    GLSLstd450Acos,
    GLSLstd450Atan,
    GLSLstd450Sinh,
    GLSLstd450Cosh,
    GLSLstd450Tanh,
    GLSLstd450Asinh,
    GLSLstd450Acosh,
    GLSLstd450Atanh,
    GLSLstd450Atan2,
    GLSLstd450Pow,
    GLSLstd450Exp,
    GLSLstd450Log,
    GLSLstd450Exp2,
    GLSLstd450Log2,
    GLSLstd450Sqrt,
    GLSLstd450InverseSqrt,
    GLSLstd450Determinant,
    GLSLstd450MatrixInverse,
    GLSLstd450ModfStruct,
    GLSLstd450FMin,
    GLSLstd450UMin,
    GLSLstd450SMin,
    GLSLstd450FMax,
    GLSLstd450UMax,
    GLSLstd450SMax,
    GLSLstd450FClamp,
    GLSLstd450UClamp,
    GLSLstd450SClamp,
    GLSLstd450FMix,
    GLSLstd450IMix,
    GLSLstd450Step,
    GLSLstd450SmoothStep,
    GLSLstd450Fma,
    GLSLstd450FrexpStruct,
    GLSLstd450Ldexp,
    GLSLstd450PackSnorm4x8,
    GLSLstd450PackUnorm4x8,
    GLSLstd450PackSnorm2x16,
    GLSLstd450PackUnorm2x16,
    GLSLstd450PackHalf2x16,
    GLSLstd450PackDouble2x32,
    GLSLstd450UnpackSnorm2x16,
    GLSLstd450UnpackUnorm2x16,
    GLSLstd450UnpackHalf2x16,
    GLSLstd450UnpackSnorm4x8,
    GLSLstd450UnpackUnorm4x8,
    GLSLstd450UnpackDouble2x32,
    GLSLstd450Length,
    GLSLstd450Distance,
    GLSLstd450Cross,
    GLSLstd450Normalize,
    GLSLstd450FaceForward,
    GLSLstd450Reflect,
    GLSLstd450Refract,
    GLSLstd450FindILsb,
    GLSLstd450FindSMsb,
    GLSLstd450FindUMsb,
    GLSLstd450InterpolateAtCentroid,
    GLSLstd450InterpolateAtSample,
    GLSLstd450InterpolateAtOffset,
    GLSLstd450NMin,
    GLSLstd450NMax,
    GLSLstd450NClamp
  };
}

void AggressiveDCEPass::InitExtensions() {
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
    // SPV_KHR_variable_pointers
    //   Currently do not support extended pointer expressions
    "SPV_AMD_gpu_shader_int16",
    "SPV_KHR_post_depth_coverage",
    "SPV_KHR_shader_atomic_counter_ops",
  });
}

}  // namespace opt
}  // namespace spvtools

