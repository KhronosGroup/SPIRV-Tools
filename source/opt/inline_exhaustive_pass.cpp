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

#include "source/opt/inline_exhaustive_pass.h"

#include <utility>

namespace spvtools {
namespace opt {
namespace {
// Indices of operands in SPIR-V instructions
constexpr int kSpvFunctionCallArgumentId = 3;
}  // namespace

Pass::Status InlineExhaustivePass::PassAccessChainByVariable(Function* func, BasicBlock::iterator call_inst_itr) {
  Pass::Status status = Pass::Status::SuccessWithoutChange;

  // Iterate over the function arguments.
  for(uint32_t arg_idx = kSpvFunctionCallArgumentId; arg_idx < call_inst_itr->NumOperands(); ++arg_idx) {
    uint32_t arg_id = call_inst_itr->GetSingleWordOperand(arg_idx);
    // Look for function arguments that are access chains.
    auto arg_inst = get_def_use_mgr()->GetDef(arg_id);
    if(arg_inst->opcode() == spv::Op::OpAccessChain) {
      // Create a new variable.
      auto var_result_id = TakeNextId();
      std::unique_ptr<Instruction> var_inst(new Instruction(
        context(), spv::Op::OpVariable, arg_inst->type_id(), var_result_id,
        {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
          {(uint32_t)spv::StorageClass::Function}}}));

      // Update the def/use of the instruction and set its basic block.
      context()->AnalyzeDefUse(&*var_inst);
      auto basic_block = &*func->begin();
      context()->set_instr_block(&*var_inst, basic_block);

      // Insert the variable at the head of the first block.
      func->begin()->begin().InsertBefore(std::move(var_inst));

      // Insert instructions to copy the access chain pointee into the variable before the function call.
      auto type_mgr = context()->get_type_mgr();
      auto pointee_type_id = type_mgr->GetId(type_mgr->GetType(arg_inst->type_id())->AsPointer()->pointee_type());
      auto load_result_id = TakeNextId();
      auto debug_line_inst = call_inst_itr->dbg_line_inst();
      auto debug_scope = call_inst_itr->GetDebugScope();
      auto load_ac_inst = MakeLoad(pointee_type_id, load_result_id, arg_id,
        debug_line_inst, debug_scope, basic_block);
      call_inst_itr->InsertBefore(std::move(load_ac_inst));

      auto store_var_inst = MakeStore(var_result_id, load_result_id, debug_line_inst,
        debug_scope, basic_block);
      call_inst_itr->InsertBefore(std::move(store_var_inst));

      // Substitute the variable into the function call argument.
      call_inst_itr->SetOperand(arg_idx, {var_result_id});

      // Insert instructions to copy the variable back into the access chain pointee after the function call.
      load_result_id = TakeNextId();
      auto insert_iter = call_inst_itr->NextNode();
      auto load_var_inst =  MakeLoad(arg_inst->type_id(), load_result_id, var_result_id,
        debug_line_inst, debug_scope, basic_block);
      insert_iter->InsertBefore(std::move(load_var_inst));

      auto store_ac_inst = MakeStore(arg_id, load_result_id, debug_line_inst,
        debug_scope, basic_block);
      insert_iter->InsertBefore(std::move(store_ac_inst));

      status = Pass::Status::SuccessWithChange;
    }
  }

  return status;
}

std::unique_ptr<Instruction> InlineExhaustivePass::MakeLoad(uint32_t result_type_id,
  uint32_t result_id, uint32_t pointer_id, Instruction const* debug_line_inst,
  DebugScope const& debug_scope, BasicBlock* basic_block) {
  std::unique_ptr<Instruction> load_inst(
    new Instruction(context(), spv::Op::OpLoad, result_type_id, result_id,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {pointer_id}}}));
  load_inst->AddDebugLine(debug_line_inst);
  load_inst->SetDebugScope(debug_scope);
  context()->AnalyzeDefUse(&*load_inst);
  context()->set_instr_block(&*load_inst, basic_block);

  return load_inst;
}

std::unique_ptr<Instruction> InlineExhaustivePass::MakeStore(uint32_t pointer_id, uint32_t object_id,
  Instruction const* debug_line_inst, DebugScope const& debug_scope, BasicBlock* basic_block) {
  std::unique_ptr<Instruction> store_inst(
    new Instruction(context(), spv::Op::OpStore, 0, 0,
    {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {pointer_id}},
    {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {object_id}}}));
  store_inst->AddDebugLine(debug_line_inst);
  store_inst->SetDebugScope(debug_scope);
  context()->AnalyzeDefUse(&*store_inst);
  context()->set_instr_block(&*store_inst, basic_block);

  return store_inst;
}

Pass::Status InlineExhaustivePass::InlineExhaustive(Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii)) {
        // Inline call.
        std::vector<std::unique_ptr<BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<Instruction>> newVars;
        if (!GenInlineCode(&newBlocks, &newVars, ii, bi)) {
          return Status::Failure;
        }
        // If call block is replaced with more than one block, point
        // succeeding phis at new last block.
        if (newBlocks.size() > 1) UpdateSucceedingPhis(newBlocks);
        // Replace old calling block with new block(s).

        bi = bi.Erase();

        for (auto& bb : newBlocks) {
          bb->SetParent(func);
        }
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0)
          func->begin()->begin().InsertBefore(std::move(newVars));
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ++ii;
      }
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

Pass::Status InlineExhaustivePass::FindAndReplaceAccessChains(Function* func) {
  Pass::Status status = Pass::Status::SuccessWithoutChange;

  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (IsInlinableFunctionCall(&*ii)) {
        status = CombineStatus(status, PassAccessChainByVariable(func, ii));
      }
    }
  }

  return status;
}

Pass::Status InlineExhaustivePass::ProcessImpl() {
  Status status = Status::SuccessWithoutChange;

  // Substitute variables for access chain function agruments.
  if (get_feature_mgr()->HasExtension(kSPV_KHR_non_semantic_info) &&
    get_module()->GetExtInstImportId("NonSemantic.Shader.DebugInfo.100") != 0) {
    ProcessFunction pfn = [&status, this](Function* fp) {
      status = CombineStatus(status, FindAndReplaceAccessChains(fp));
      return false;
    };
    context()->ProcessReachableCallTree(pfn);
  }

  // Attempt exhaustive inlining on each entry point function in module
  ProcessFunction pfn = [&status, this](Function* fp) {
    status = CombineStatus(status, InlineExhaustive(fp));
    return false;
  };
  context()->ProcessReachableCallTree(pfn);
  return status;
}

InlineExhaustivePass::InlineExhaustivePass() = default;

Pass::Status InlineExhaustivePass::Process() {
  InitializeInline();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
