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

#include "OpenCLDebugInfo100.h"
#include "source/cfa.h"
#include "source/util/make_unique.h"

// Indices of operands in SPIR-V instructions

static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;

// Constants for OpenCL.DebugInfo.100 extension instructions.

static const uint32_t kDebugExtensionOperand = 2;
static const uint32_t kOpLineOperandLineIndex = 1;
static const uint32_t kLineOperandIndexDebugFunction = 7;
static const uint32_t kLineOperandIndexDebugLexicalBlock = 5;
static const uint32_t kDebugInlinedAtOperandInlinedIndex = 6;
static const uint32_t kDebugFunctionOperandFunctionIndex = 13;
static const uint32_t kDebugDeclareOperandVariableIndex = 5;

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
                          std::unique_ptr<BasicBlock>* block_ptr,
                          const std::vector<Instruction>& line_insts,
                          uint32_t dbg_scope) {
  std::unique_ptr<Instruction> newStore(
      new Instruction(context(), SpvOpStore, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {val_id}}}));
  if (!line_insts.empty()) {
    auto& new_lines = newStore->dbg_line_insts();
    new_lines.insert(new_lines.end(), line_insts.begin(), line_insts.end());
  }
  if (dbg_scope) newStore->SetDebugScope(dbg_scope);
  (*block_ptr)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<BasicBlock>* block_ptr,
                         const std::vector<Instruction>& line_insts,
                         uint32_t dbg_scope) {
  std::unique_ptr<Instruction> newLoad(
      new Instruction(context(), SpvOpLoad, type_id, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {ptr_id}}}));
  if (!line_insts.empty()) {
    auto& new_lines = newLoad->dbg_line_insts();
    new_lines.insert(new_lines.end(), line_insts.begin(), line_insts.end());
  }
  if (dbg_scope) newLoad->SetDebugScope(dbg_scope);
  (*block_ptr)->AddInstruction(std::move(newLoad));
}

std::unique_ptr<Instruction> InlinePass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  return newLabel;
}

uint32_t InlinePass::CreateDebugInlinedAt(const std::vector<Instruction>& lines,
                                          const DebugScope& fn_call_scope) {
  // TODO: Return the existing one using a map structure. It reduces the number
  // of DebugDeclare, but increases the code complexity.

  if (!get_module()->ContainsOpenCL100DebugInstrunctions()) return 0;

  uint32_t line_number = 0;
  if (lines.empty()) {
    auto it = id2lexical_scope_.find(fn_call_scope.GetLexicalScope());
    if (it == id2lexical_scope_.end()) return 0;
    OpenCLDebugInfo100Instructions debug_opcode =
        it->second->GetOpenCL100DebugOpcode();
    switch (debug_opcode) {
      case OpenCLDebugInfo100DebugFunction:
      case OpenCLDebugInfo100DebugTypeComposite:
        line_number =
            it->second->GetSingleWordOperand(kLineOperandIndexDebugFunction);
        break;
      case OpenCLDebugInfo100DebugLexicalBlock:
        line_number = it->second->GetSingleWordOperand(
            kLineOperandIndexDebugLexicalBlock);
        break;
      case OpenCLDebugInfo100DebugCompilationUnit:
        break;
      default:
        // Unreachable!
        break;
    }
  } else {
    line_number = lines[0].GetSingleWordOperand(kOpLineOperandLineIndex);
  }

  Instruction& first_dbg_inst = *(get_module()->ext_inst_debuginfo_begin());
  uint32_t ret_id = context()->TakeNextId();
  std::unique_ptr<Instruction> inlined_at(new Instruction(
      context(), SpvOpExtInst, first_dbg_inst.type_id(), ret_id,
      {
          first_dbg_inst.GetOperand(kDebugExtensionOperand),
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(OpenCLDebugInfo100DebugInlinedAt)}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {line_number}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
           {fn_call_scope.GetLexicalScope()}},
      }));
  // The function call instruction itself already has DebugInlinedAt
  // information. DebugInlinedAt of the closest caller of the instruction
  // must come first. We put the DebugInlinedAt of the function call at the
  // tail in the recurive DebugInlinedAt chain.
  if (fn_call_scope.GetInlinedAt()) {
    inlined_at->AddOperand({spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
                            {fn_call_scope.GetInlinedAt()}});
  }
  id2inlined_at_[ret_id] = inlined_at.get();
  get_module()->AddExtInstDebugInfo(std::move(inlined_at));
  return ret_id;
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

Instruction* InlinePass::CloneFunctionParamDebugDeclare(uint32_t from,
                                                        uint32_t to) {
  auto it = param_id2debugdecl_.find(from);
  if (it == param_id2debugdecl_.end()) return nullptr;

  auto* clone = it->second->Clone(context());
  clone->SetResultId(context()->TakeNextId());
  clone->GetOperand(kDebugDeclareOperandVariableIndex).words[0] = to;

  if (param_id2debugdecl_.find(to) == param_id2debugdecl_.end())
    param_id2debugdecl_[to] = clone;
  return clone;
}

void InlinePass::MapParams(
    Function* calleeFn, BasicBlock::iterator call_inst_itr,
    std::unordered_map<uint32_t, uint32_t>* callee2caller,
    std::vector<Instruction*>* dbg_insts_in_callee_header) {
  int param_idx = 0;
  calleeFn->ForEachParam([&call_inst_itr, &param_idx, &callee2caller,
                          &dbg_insts_in_callee_header,
                          this](const Instruction* cpi) {
    const uint32_t pid = cpi->result_id();
    const uint32_t operandId = call_inst_itr->GetSingleWordOperand(
        kSpvFunctionCallArgumentId + param_idx);
    (*callee2caller)[pid] = operandId;
    // Clone DebugDeclare for OpFunctionParameter and update its
    // Variable operand to the operand of the function call. We will
    // put it in caller function's header.
    auto* dbgDecl = CloneFunctionParamDebugDeclare(pid, operandId);
    if (dbgDecl) {
      dbg_insts_in_callee_header->push_back(dbgDecl);
    }
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
    // Note that we do not have to clone DebugDeclare/DebugValue for
    // local variables of callee because InlinePass::GenInlineCode()
    // will iterate all instructions of callee to copy them and update
    // all operands properly using callee2caller.
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
  // Note that we do not have to handle DebugDeclare/DebugValue similar
  // to DebugDeclare/DebugValue for local variables of callee.
  // CloneSameBlockOps() will handle DebugDeclare/DebugValue for return
  // value properly.
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
  // DebugDeclare or DebugValue instructions in the header of callee
  // that links DebugLocalVariable instructions to OpFunctionParameter.
  std::vector<Instruction*> dbg_insts_in_callee_header;

  // Invalidate the def-use chains.  They are not kept up to date while
  // inlining.  However, certain calls try to keep them up-to-date if they are
  // valid.  These operations can fail.
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);

  Function* calleeFn = id2function_[call_inst_itr->GetSingleWordOperand(
      kSpvFunctionCallFunctionId)];

  // Check for multiple returns in the callee.
  auto fi = early_return_funcs_.find(calleeFn->result_id());
  const bool earlyReturn = fi != early_return_funcs_.end();

  // Map parameters to actual arguments.
  MapParams(calleeFn, call_inst_itr, &callee2caller,
            &dbg_insts_in_callee_header);

  // Define caller local variables for all callee variables and create map to
  // them.
  if (!CloneAndMapLocals(calleeFn, new_vars, &callee2caller)) {
    return false;
  }

  // Get the DebugScope of callee function. We want to set the DebugScope
  // of newly added variable/store/load for return statements as the
  // callee's scope.
  auto it_scope = func_id2dbg_func_id_.find(calleeFn->result_id());
  uint32_t debug_scope = 0;
  if (it_scope != func_id2dbg_func_id_.end()) debug_scope = it_scope->second;

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
  std::unordered_set<uint32_t> callee_result_ids;
  calleeFn->ForEachInst([&callee_result_ids](const Instruction* cpi) {
    const uint32_t rid = cpi->result_id();
    if (rid != 0) callee_result_ids.insert(rid);
  });

  // If the caller is a loop header and the callee has multiple blocks, then the
  // normal inlining logic will place the OpLoopMerge in the last of several
  // blocks in the loop.  Instead, it should be placed at the end of the first
  // block.  We'll wait to move the OpLoopMerge until the end of the regular
  // inlining logic, and only if necessary.
  bool caller_is_loop_header = false;
  if (call_block_itr->GetLoopMergeInst()) {
    caller_is_loop_header = true;
  }

  bool callee_begins_with_structured_header =
      (*(calleeFn->begin())).GetMergeInst() != nullptr;

  // Clone and map callee code. Copy caller block code to beginning of
  // first block and end of last block.
  bool prevInstWasReturn = false;
  uint32_t singleTripLoopHeaderId = 0;
  uint32_t singleTripLoopContinueId = 0;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  // new_blk_ptr is a new basic block in the caller.  New instructions are
  // written to it.  It is created when we encounter the OpLabel
  // of the first callee block.  It is appended to new_blocks only when
  // it is complete.
  std::unique_ptr<BasicBlock> new_blk_ptr;
  // Instructions that were not parts of callee function.
  std::vector<Instruction*> not_inlined_insts;
  bool successful = calleeFn->WhileEachInst(
      [&new_blocks, &callee2caller, &call_block_itr, &call_inst_itr,
       &new_blk_ptr, &prevInstWasReturn, &returnLabelId, &returnVarId,
       caller_is_loop_header, callee_begins_with_structured_header,
       &calleeTypeId, &multiBlocks, &postCallSB, &preCallSB, earlyReturn,
       &singleTripLoopHeaderId, &singleTripLoopContinueId, &callee_result_ids,
       &not_inlined_insts, &debug_scope, &dbg_insts_in_callee_header,
       this](const Instruction* cpi) {
        // DebugDeclare for function parameters are already processed.
        if (get_module()->ContainsOpenCL100DebugInstrunctions() &&
            (cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugDeclare ||
             cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue)) {
          auto id =
              cpi->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
          if (param_id2debugdecl_.find(id) != param_id2debugdecl_.end())
            return true;
        }
        switch (cpi->opcode()) {
          case SpvOpFunction:
          case SpvOpFunctionParameter:
            // Already processed
            break;
          case SpvOpVariable:
            if (cpi->NumInOperands() == 2) {
              assert(callee2caller.count(cpi->result_id()) &&
                     "Expected the variable to have already been mapped.");
              uint32_t new_var_id = callee2caller.at(cpi->result_id());

              // The initializer must be a constant or global value.  No mapped
              // should be used.
              uint32_t val_id = cpi->GetSingleWordInOperand(1);
              AddStore(new_var_id, val_id, &new_blk_ptr,
                       call_inst_itr->dbg_line_insts(), debug_scope);
            }
            break;
          case SpvOpUnreachable:
          case SpvOpKill: {
            // Generate a return label so that we split the block with the
            // function call. Copy the terminator into the new block.
            if (returnLabelId == 0) {
              returnLabelId = context()->TakeNextId();
              if (returnLabelId == 0) {
                return false;
              }
            }
            std::unique_ptr<Instruction> terminator(
                new Instruction(context(), cpi->opcode(), 0, 0, {}));
            new_blk_ptr->AddInstruction(std::move(terminator));
            break;
          }
          case SpvOpLabel: {
            // If previous instruction was early return, insert branch
            // instruction to return block.
            if (prevInstWasReturn) {
              if (returnLabelId == 0) {
                returnLabelId = context()->TakeNextId();
                if (returnLabelId == 0) {
                  return false;
                }
              }
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
              labelId = (mapItr != callee2caller.end())
                            ? mapItr->second
                            : context()->TakeNextId();
              if (labelId == 0) {
                return false;
              }
            } else {
              // First block needs to use label of original block
              // but map callee label in case of phi reference.
              labelId = call_block_itr->id();
              callee2caller[cpi->result_id()] = labelId;
              firstBlock = true;
            }
            // Create first/next block.
            new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(labelId));
            if (firstBlock) {
              // Copy contents of original caller block up to call instruction.
              for (auto cii = call_block_itr->begin(); cii != call_inst_itr;
                   cii = call_block_itr->begin()) {
                Instruction* inst = &*cii;
                inst->RemoveFromList();
                std::unique_ptr<Instruction> cp_inst(inst);
                if (get_module()->ContainsOpenCL100DebugInstrunctions())
                  not_inlined_insts.push_back(cp_inst.get());
                // Remember same-block ops for possible regeneration.
                if (IsSameBlockOp(&*cp_inst)) {
                  auto* sb_inst_ptr = cp_inst.get();
                  preCallSB[cp_inst->result_id()] = sb_inst_ptr;
                }
                new_blk_ptr->AddInstruction(std::move(cp_inst));
              }

              // Add DebugDeclare instructions for callee's function
              // parameters to caller's body.
              for (auto* i : dbg_insts_in_callee_header) {
                new_blk_ptr->AddInstruction(std::unique_ptr<Instruction>(i));
              }

              if (caller_is_loop_header &&
                  callee_begins_with_structured_header) {
                // We can't place both the caller's merge instruction and
                // another merge instruction in the same block.  So split the
                // calling block. Insert an unconditional branch to a new guard
                // block.  Later, once we know the ID of the last block,  we
                // will move the caller's OpLoopMerge from the last generated
                // block into the first block. We also wait to avoid
                // invalidating various iterators.
                const auto guard_block_id = context()->TakeNextId();
                if (guard_block_id == 0) {
                  return false;
                }
                AddBranch(guard_block_id, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                // Start the next block.
                new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(guard_block_id));
                // Reset the mapping of the callee's entry block to point to
                // the guard block.  Do this so we can fix up phis later on to
                // satisfy dominance.
                callee2caller[cpi->result_id()] = guard_block_id;
              }
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
              if (earlyReturn) {
                singleTripLoopHeaderId = context()->TakeNextId();
                if (singleTripLoopHeaderId == 0) {
                  return false;
                }
                AddBranch(singleTripLoopHeaderId, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr =
                    MakeUnique<BasicBlock>(NewLabel(singleTripLoopHeaderId));
                returnLabelId = context()->TakeNextId();
                singleTripLoopContinueId = context()->TakeNextId();
                if (returnLabelId == 0 || singleTripLoopContinueId == 0) {
                  return false;
                }
                AddLoopMerge(returnLabelId, singleTripLoopContinueId,
                             &new_blk_ptr);
                uint32_t postHeaderId = context()->TakeNextId();
                if (postHeaderId == 0) {
                  return false;
                }
                AddBranch(postHeaderId, &new_blk_ptr);
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(postHeaderId));
                multiBlocks = true;
                // Reset the mapping of the callee's entry block to point to
                // the post-header block.  Do this so we can fix up phis later
                // on to satisfy dominance.
                callee2caller[cpi->result_id()] = postHeaderId;
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
            AddStore(returnVarId, valId, &new_blk_ptr,
                     call_inst_itr->dbg_line_insts(), debug_scope);

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
            // If there was an early return, we generated a return label id
            // for it.  Now we have to generate the return block with that Id.
            if (returnLabelId != 0) {
              // If previous instruction was return, insert branch instruction
              // to return block.
              if (prevInstWasReturn) AddBranch(returnLabelId, &new_blk_ptr);
              if (earlyReturn) {
                // If we generated a loop header for the single-trip loop
                // to accommodate early returns, insert the continue
                // target block now, with a false branch back to the loop
                // header.
                new_blocks->push_back(std::move(new_blk_ptr));
                new_blk_ptr =
                    MakeUnique<BasicBlock>(NewLabel(singleTripLoopContinueId));
                uint32_t false_id = GetFalseId();
                if (false_id == 0) {
                  return false;
                }
                AddBranchCond(false_id, singleTripLoopHeaderId, returnLabelId,
                              &new_blk_ptr);
              }
              // Generate the return block.
              new_blocks->push_back(std::move(new_blk_ptr));
              new_blk_ptr = MakeUnique<BasicBlock>(NewLabel(returnLabelId));
              multiBlocks = true;
            }
            // Load return value into result id of call, if it exists.
            if (returnVarId != 0) {
              const uint32_t resId = call_inst_itr->result_id();
              assert(resId != 0);
              AddLoad(calleeTypeId, resId, returnVarId, &new_blk_ptr,
                      call_inst_itr->dbg_line_insts(), debug_scope);
            }
            // Copy remaining instructions from caller block.
            for (Instruction* inst = call_inst_itr->NextNode(); inst;
                 inst = call_inst_itr->NextNode()) {
              inst->RemoveFromList();
              std::unique_ptr<Instruction> cp_inst(inst);
              if (get_module()->ContainsOpenCL100DebugInstrunctions())
                not_inlined_insts.push_back(cp_inst.get());
              // If multiple blocks generated, regenerate any same-block
              // instruction that has not been seen in this last block.
              if (multiBlocks) {
                if (!CloneSameBlockOps(&cp_inst, &postCallSB, &preCallSB,
                                       &new_blk_ptr)) {
                  return false;
                }

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
            std::unique_ptr<Instruction> cp_inst(cpi->Clone(context()));
            bool succeeded = cp_inst->WhileEachInId(
                [&callee2caller, &callee_result_ids, this](uint32_t* iid) {
                  const auto mapItr = callee2caller.find(*iid);
                  if (mapItr != callee2caller.end()) {
                    *iid = mapItr->second;
                  } else if (callee_result_ids.find(*iid) !=
                             callee_result_ids.end()) {
                    // Forward reference. Allocate a new id, map it,
                    // use it and check for it when remapping result ids
                    const uint32_t nid = context()->TakeNextId();
                    if (nid == 0) {
                      return false;
                    }
                    callee2caller[*iid] = nid;
                    *iid = nid;
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
              const auto mapItr = callee2caller.find(rid);
              uint32_t nid;
              if (mapItr != callee2caller.end()) {
                nid = mapItr->second;
              } else {
                nid = context()->TakeNextId();
                if (nid == 0) {
                  return false;
                }
                callee2caller[rid] = nid;
              }
              cp_inst->SetResultId(nid);
              get_decoration_mgr()->CloneDecorations(rid, nid);
            }
            new_blk_ptr->AddInstruction(std::move(cp_inst));
          } break;
        }
        return true;
      });

  if (!successful) {
    return false;
  }

  if (caller_is_loop_header && (new_blocks->size() > 1)) {
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

  // Add DebugInlineAt instructions for the function inlining information.
  // |new_blocks| has three types of instruction at this point:
  // Type1. instructions that were not parts of callee function
  // Type2. inlined instructions from callee function
  // Type3. inlined instructions from another function to callee
  //        function which are now recursively inlined to the caller function
  //        once more.
  // We can simply add DebugInlineAt instructions for Type2 and will skip it
  // for Type1. Type3 is complicated. We must create recurive DebugInlineAt
  // instructions.
  // For example, assuming that `%color = OpLoad %v4float %in_var_COLOR` was
  // initially generated for an expression in a function `foo` but it was
  // already inlined to `bar` with `DebugInlinedAt %bar` information, we must
  // create `%inlined_to_main = DebugInlinedAt %main` and use recursive inlining
  // information like `DebugInlinedAt %bar %inlined_to_main` when we inline
  // `bar` to `main`.
  if (get_module()->ContainsOpenCL100DebugInstrunctions()) {
    const auto& call_inst_dbg_scope = call_inst_itr->GetDebugScope();
    if (call_inst_dbg_scope.GetLexicalScope()) {
      const uint32_t inlined_at = CreateDebugInlinedAt(
          call_inst_itr->dbg_line_insts(), call_inst_dbg_scope);
      if (!inlined_at) return false;

      uint32_t prev_inlined_at = 0;
      uint32_t last_inlined_at_chain = inlined_at;
      uint32_t not_inlined_insts_idx = 0;

      auto fn_update_inlined_at = [&not_inlined_insts, &not_inlined_insts_idx,
                                   &inlined_at, &prev_inlined_at,
                                   &last_inlined_at_chain,
                                   this](Instruction* cpi) {
        // Type1. instructions that were not parts of callee function.
        if (not_inlined_insts_idx < not_inlined_insts.size() &&
            cpi == not_inlined_insts[not_inlined_insts_idx]) {
          ++not_inlined_insts_idx;
          return true;
        }

        if (!cpi->GetDebugScope().GetLexicalScope()) return true;

        // Type2. inlined instructions from callee function.
        DebugScope new_scope(cpi->GetDebugScope().GetLexicalScope(), 0);
        if (!cpi->GetDebugScope().GetInlinedAt()) {
          new_scope.SetInlinedAt(inlined_at);
          cpi->SetDebugScope(new_scope);
          return true;
        }

        // We want to reuse the DebugInlinedAt previously created
        // if the current DebugInlinedAt initially had the same
        // DebugInlinedAt with the previous one.
        if (cpi->GetDebugScope().GetInlinedAt() == prev_inlined_at) {
          new_scope.SetInlinedAt(last_inlined_at_chain);
          cpi->SetDebugScope(new_scope);
          return true;
        }
        prev_inlined_at = cpi->GetDebugScope().GetInlinedAt();

        // Type3. inlined instructions from another function to callee and
        // now recursively inlined to the caller.
        //
        // For example, initially, we had three functions: foo() { bar(); },
        // bar() { /* some code */ zoo(); }, zoo() { /* some code */ }.
        //
        // Now zoo() is inlined to bar():
        // bar() {
        //   /* some code of bar() */
        //   /* some code of zoo(), which is DebugInlinedAt bar() */
        // }
        //
        // After zoo() is inlined to foo():
        // foo() {
        //   /* some code of bar() */
        //   /* some code of zoo(), which is DebugInlinedAt bar() %bar_to_zoo
        //                          where %bar_to_zoo = DebugInlinedAt zoo() */
        // }
        //
        // When we change `DebugInlinedAt bar()` to
        // `DebugInlinedAt bar() %bar_to_zoo` and create `%bar_to_zoo`, we need
        // two new DebugInlinedAt. If the instruction already has a chain of
        // DebugInlinedAt, we must clone the chain and put DebugInlinedAt for
        // this function call at the end of the new chain.
        uint32_t chain_iter_id = cpi->GetDebugScope().GetInlinedAt();
        uint32_t head_inlined_at_id_of_chain = 0;
        std::vector<Instruction*> new_inlined_at_insts;
        uint32_t tail_id = inlined_at;
        do {
          // If a path from |chain_iter_id| to |tail_id| exists, we
          // can reuse it.
          auto pair_key = std::pair<uint32_t, uint32_t>(chain_iter_id, tail_id);
          auto path_it = inlined_at_chain_.find(pair_key);
          if (path_it != inlined_at_chain_.end()) {
            // Update tail and head of DebugInlinedAt chain.
            tail_id = path_it->second;
            if (!head_inlined_at_id_of_chain)
              head_inlined_at_id_of_chain = tail_id;
            break;
          }
          auto it_inlined_at = id2inlined_at_.find(chain_iter_id);
          if (it_inlined_at == id2inlined_at_.end()) return false;
          Instruction* new_inlined_at = it_inlined_at->second->Clone(context());
          new_inlined_at->SetResultId(context()->TakeNextId());
          id2inlined_at_[new_inlined_at->result_id()] = new_inlined_at;
          inlined_at_chain_[pair_key] = new_inlined_at->result_id();
          // Previous DebugInlinedAt must have the current new DebugInlinedAt
          // as its Inlined operand, which makes a recursive DebugInlinedAt
          // chain.
          if (!new_inlined_at_insts.empty()) {
            new_inlined_at_insts.back()
                ->GetOperand(kDebugInlinedAtOperandInlinedIndex)
                .words[0] = new_inlined_at->result_id();
          }
          new_inlined_at_insts.push_back(new_inlined_at);
          if (!head_inlined_at_id_of_chain)
            head_inlined_at_id_of_chain = new_inlined_at->result_id();
          if (new_inlined_at->NumOperands() <=
              kDebugInlinedAtOperandInlinedIndex) {
            break;
          }
          chain_iter_id = new_inlined_at->GetSingleWordOperand(
              kDebugInlinedAtOperandInlinedIndex);
        } while (chain_iter_id);

        // Set |tail_id| as the tail of the DebugInlinedAt chain.
        if (!new_inlined_at_insts.empty()) {
          if (new_inlined_at_insts.back()->NumOperands() <=
              kDebugInlinedAtOperandInlinedIndex) {
            new_inlined_at_insts.back()->AddOperand(
                {SPV_OPERAND_TYPE_RESULT_ID, {tail_id}});
          } else {
            new_inlined_at_insts.back()
                ->GetOperand(kDebugInlinedAtOperandInlinedIndex)
                .words[0] = tail_id;
          }
        }

        // Add all new DebugInlinedAt to module.
        for (int i = static_cast<int>(new_inlined_at_insts.size()) - 1; i >= 0;
             --i) {
          id2inlined_at_[new_inlined_at_insts[i]->result_id()] =
              new_inlined_at_insts[i];
          get_module()->AddExtInstDebugInfo(
              std::unique_ptr<Instruction>(new_inlined_at_insts[i]));
        }

        new_scope.SetInlinedAt(head_inlined_at_id_of_chain);
        cpi->SetDebugScope(new_scope);
        last_inlined_at_chain = head_inlined_at_id_of_chain;
        return true;
      };
      for (auto& var : *new_vars) {
        successful = var->WhileEachInst(fn_update_inlined_at);
        if (!successful) return false;
      }
      for (auto& blk : *new_blocks) {
        successful = blk->WhileEachInst(fn_update_inlined_at);
        if (!successful) return false;
      }
    }
  }

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

  if (get_module()->ContainsOpenCL100DebugInstrunctions()) {
    param_id2debugdecl_.clear();
    for (auto& fn : *get_module()) {
      fn.ForEachHeaderDebugInstructions([this](Instruction* cpi) {
        OpenCLDebugInfo100Instructions opcode = cpi->GetOpenCL100DebugOpcode();
        if (opcode == OpenCLDebugInfo100DebugDeclare ||
            opcode == OpenCLDebugInfo100DebugValue) {
          auto id =
              cpi->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
          if (param_id2debugdecl_.find(id) == param_id2debugdecl_.end()) {
            param_id2debugdecl_[id] = cpi;
          }
        }
      });
    }

    id2inlined_at_.clear();
    id2lexical_scope_.clear();
    func_id2dbg_func_id_.clear();

    // Initialize debug lexical scope and debug function maps.
    for (auto& i : get_module()->ext_inst_debuginfo()) {
      OpenCLDebugInfo100Instructions opcode = i.GetOpenCL100DebugOpcode();
      switch (opcode) {
        case OpenCLDebugInfo100DebugInlinedAt: {
          id2inlined_at_[i.result_id()] = &i;
          break;
        }
        case OpenCLDebugInfo100DebugFunction: {
          // TODO: Report a validation error if multiple DebugFunction
          //       have the same OpFunction operand.
          func_id2dbg_func_id_[i.GetSingleWordOperand(
              kDebugFunctionOperandFunctionIndex)] = i.result_id();
          id2lexical_scope_[i.result_id()] = &i;
          break;
        }
        case OpenCLDebugInfo100DebugTypeComposite:
        case OpenCLDebugInfo100DebugLexicalBlock:
        case OpenCLDebugInfo100DebugCompilationUnit:
          id2lexical_scope_[i.result_id()] = &i;
          break;
        default:
          break;
      }
    }

    // Keep all paths from each DebugInlinedAt to its recursive tail
    inlined_at_chain_.clear();
    for (auto& i : get_module()->ext_inst_debuginfo()) {
      if (i.GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugInlinedAt) {
        // Find the tail DebugInlinedAt of the current DebugInlinedAt path.
        auto chain_iter = &i;
        uint32_t tail_id = chain_iter->result_id();
        while (chain_iter) {
          if (chain_iter->NumOperands() <= kDebugInlinedAtOperandInlinedIndex)
            break;
          uint32_t next_inlined_at_id = chain_iter->GetSingleWordOperand(
              kDebugInlinedAtOperandInlinedIndex);
          if (!next_inlined_at_id) break;
          tail_id = next_inlined_at_id;
          chain_iter = id2inlined_at_[next_inlined_at_id];
        }
        // Add a mapping between the path from a DebugInlinedAt to
        // its tail DebugInlinedAt and the DebugInlinedAt.
        chain_iter = &i;
        while (chain_iter) {
          inlined_at_chain_[std::pair<uint32_t, uint32_t>(
              chain_iter->result_id(), tail_id)] = chain_iter->result_id();
          if (chain_iter->NumOperands() <= kDebugInlinedAtOperandInlinedIndex)
            break;
          uint32_t next_inlined_at_id = chain_iter->GetSingleWordOperand(
              kDebugInlinedAtOperandInlinedIndex);
          if (!next_inlined_at_id) break;
          chain_iter = id2inlined_at_[next_inlined_at_id];
        }
      }
    }
  }
}

InlinePass::InlinePass() {}

}  // namespace opt
}  // namespace spvtools
