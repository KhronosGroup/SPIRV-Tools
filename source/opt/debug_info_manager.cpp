// Copyright (c) 2020 Google LLC
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

#include "source/opt/debug_info_manager.h"

#include <cassert>

#include "source/opt/ir_context.h"

// Constants for OpenCL.DebugInfo.100 extension instructions.

static const uint32_t kOpLineOperandLineIndex = 1;
static const uint32_t kLineOperandIndexDebugFunction = 7;
static const uint32_t kLineOperandIndexDebugLexicalBlock = 5;
static const uint32_t kDebugFunctionOperandFunctionIndex = 13;
static const uint32_t kDebugDeclareOperandVariableIndex = 5;

namespace spvtools {
namespace opt {
namespace analysis {

DebugInfoManager::DebugInfoManager(IRContext* c) : context_(c) {
  AnalyzeDebugInsts(*c->module());
}

bool DebugInfoManager::WhileEachDebugInsts(
    OpenCLDebugInfo100Instructions dbg_opcode,
    const std::function<bool(Instruction*)>& f) {
  for (auto it = context()->module()->ext_inst_debuginfo_begin();
       it != context()->module()->ext_inst_debuginfo_end(); ++it) {
    if (it->GetOpenCL100DebugOpcode() != dbg_opcode) continue;
    if (!f(&*it)) return false;
  }
  return true;
}

void DebugInfoManager::ForEachDebugInsts(
    OpenCLDebugInfo100Instructions dbg_opcode,
    const std::function<void(Instruction*)>& f) {
  WhileEachDebugInsts(dbg_opcode, [&f](Instruction* dbg_inst) {
    f(dbg_inst);
    return true;
  });
}

uint32_t DebugInfoManager::CreateDebugInlinedAt(
    const std::vector<Instruction>& lines, const DebugScope& scope) {
  if (context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo() ==
      0)
    return kNoInlinedAt;

  uint32_t line_number = 0;
  if (lines.empty()) {
    auto lexical_scope_it = id_to_dbg_inst_.find(scope.GetLexicalScope());
    if (lexical_scope_it == id_to_dbg_inst_.end()) return kNoInlinedAt;
    OpenCLDebugInfo100Instructions debug_opcode =
        lexical_scope_it->second->GetOpenCL100DebugOpcode();
    switch (debug_opcode) {
      case OpenCLDebugInfo100DebugFunction:
      case OpenCLDebugInfo100DebugTypeComposite:
        line_number = lexical_scope_it->second->GetSingleWordOperand(
            kLineOperandIndexDebugFunction);
        break;
      case OpenCLDebugInfo100DebugLexicalBlock:
        line_number = lexical_scope_it->second->GetSingleWordOperand(
            kLineOperandIndexDebugLexicalBlock);
        break;
      case OpenCLDebugInfo100DebugCompilationUnit:
        break;
      default:
        assert(false &&
               "Unreachable. a debug extension instruction for a "
               "lexical scope must be DebugFunction, DebugTypeComposite, "
               "DebugLexicalBlock, or DebugCompilationUnit.");
        break;
    }
  } else {
    line_number = lines[0].GetSingleWordOperand(kOpLineOperandLineIndex);
  }

  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> inlined_at(new Instruction(
      context(), SpvOpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      result_id,
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
           {context()
                ->get_feature_mgr()
                ->GetExtInstImportId_OpenCL100DebugInfo()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(OpenCLDebugInfo100DebugInlinedAt)}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {line_number}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {scope.GetLexicalScope()}},
      }));
  // |scope| already has DebugInlinedAt. We put the existing DebugInlinedAt
  // into the Inlined operand of this new DebugInlinedAt.
  if (scope.GetInlinedAt() != kNoInlinedAt) {
    inlined_at->AddOperand({spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
                            {scope.GetInlinedAt()}});
  }
  id_to_dbg_inst_[result_id] = inlined_at.get();
  context()->module()->AddExtInstDebugInfo(std::move(inlined_at));
  return result_id;
}

Instruction* DebugInfoManager::CreateDebugInfoNone() {
  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> dbg_info_none_inst(new Instruction(
      context(), SpvOpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      result_id,
      {
          {SPV_OPERAND_TYPE_RESULT_ID,
           {context()
                ->get_feature_mgr()
                ->GetExtInstImportId_OpenCL100DebugInfo()}},
          {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(OpenCLDebugInfo100DebugInfoNone)}},
      }));

  id_to_dbg_inst_[result_id] = dbg_info_none_inst.get();

  // Add to the front of |ext_inst_debuginfo_|.
  return context()->module()->ext_inst_debuginfo_begin()->InsertBefore(
      std::move(dbg_info_none_inst));
}

Instruction* DebugInfoManager::CloneDebugDeclare(uint32_t from, uint32_t to) {
  auto debugdecl_it = local_var_id_to_dbgdecl_.find(from);
  if (debugdecl_it == local_var_id_to_dbgdecl_.end()) return nullptr;

  auto* clone = debugdecl_it->second->Clone(context());
  clone->SetResultId(context()->TakeNextId());
  clone->GetOperand(kDebugDeclareOperandVariableIndex).words[0] = to;

  if (local_var_id_to_dbgdecl_.find(to) == local_var_id_to_dbgdecl_.end())
    local_var_id_to_dbgdecl_[to] = clone;
  return clone;
}

Instruction* DebugInfoManager::GetDebugInlinedAt(uint32_t dbg_inlined_at_id) {
  auto it_inlined_at = id_to_dbg_inst_.find(dbg_inlined_at_id);
  if (it_inlined_at == id_to_dbg_inst_.end()) return nullptr;
  if (it_inlined_at->second->GetOpenCL100DebugOpcode() !=
      OpenCLDebugInfo100DebugInlinedAt) {
    return nullptr;
  }
  return it_inlined_at->second;
}

Instruction* DebugInfoManager::CloneDebugInlinedAt(uint32_t dbg_inlined_at_id) {
  auto* inlined_at = GetDebugInlinedAt(dbg_inlined_at_id);
  if (inlined_at == nullptr) return nullptr;
  auto* new_inlined_at = inlined_at->Clone(context());
  new_inlined_at->SetResultId(context()->TakeNextId());
  id_to_dbg_inst_[new_inlined_at->result_id()] = new_inlined_at;
  return new_inlined_at;
}

void DebugInfoManager::AnalyzeDebugInsts(Module& module) {
  for (auto& inst : module.ext_inst_debuginfo()) {
    id_to_dbg_inst_[inst.result_id()] = &inst;
    if (inst.GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction) {
      auto fn_id =
          inst.GetSingleWordOperand(kDebugFunctionOperandFunctionIndex);
      assert(fn_id_to_dbg_fn_.find(fn_id) == fn_id_to_dbg_fn_.end() &&
             "Two DebugFunction instruction exists for a single OpFunction.");
      fn_id_to_dbg_fn_[fn_id] = &inst;
    }
  }

  module.ForEachInst([this](Instruction* cpi) {
    if (cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugDeclare ||
        cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue) {
      auto var_id =
          cpi->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
      if (local_var_id_to_dbgdecl_.find(var_id) !=
          local_var_id_to_dbgdecl_.end()) {
        return;
      }
      local_var_id_to_dbgdecl_[var_id] = cpi;
    }
  });
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
