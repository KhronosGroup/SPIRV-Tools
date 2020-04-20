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
  assert(c->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo() &&
         "Module does not include debug info extension instruction.");
  AnalyzeDebugInsts(*c->module());
}

Instruction* DebugInfoManager::GetDbgDeclareForVar(uint32_t var_id) {
  auto debugdecl_it = local_var_id_to_dbgdecl_.find(var_id);
  return debugdecl_it == local_var_id_to_dbgdecl_.end() ? nullptr
                                                        : debugdecl_it->second;
}

void DebugInfoManager::RegisterDbgDeclareForVar(Instruction* inst) {
  assert((inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugDeclare ||
          inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue) &&
         "inst is not a DebugDeclare or DebugValue");
  auto var_id = inst->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
  if (local_var_id_to_dbgdecl_.find(var_id) != local_var_id_to_dbgdecl_.end()) {
    return;
  }
  local_var_id_to_dbgdecl_[var_id] = inst;
}

Instruction* DebugInfoManager::GetDbgInst(uint32_t id) {
  auto dbg_inst_it = id_to_dbg_inst_.find(id);
  return dbg_inst_it == id_to_dbg_inst_.end() ? nullptr : dbg_inst_it->second;
}

void DebugInfoManager::RegisterDbgInst(Instruction* inst) {
  assert(
      inst->NumInOperands() != 0 &&
      context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo() ==
          inst->GetInOperand(0).words[0] &&
      "Given instruction is not a debug instruction");
  id_to_dbg_inst_[inst->result_id()] = inst;
}

void DebugInfoManager::RegisterDbgFunction(Instruction* inst) {
  assert(inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction &&
         "inst is not a DebugFunction");
  auto fn_id = inst->GetSingleWordOperand(kDebugFunctionOperandFunctionIndex);
  fn_id_to_dbg_fn_[fn_id] = inst;
}

uint32_t DebugInfoManager::CreateDebugInlinedAt(const Instruction* line,
                                                const DebugScope& scope) {
  if (context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo() ==
      0)
    return kNoInlinedAt;

  uint32_t line_number = 0;
  if (line == nullptr) {
    auto* lexical_scope_inst = GetDbgInst(scope.GetLexicalScope());
    if (lexical_scope_inst == nullptr) return kNoInlinedAt;
    OpenCLDebugInfo100Instructions debug_opcode =
        lexical_scope_inst->GetOpenCL100DebugOpcode();
    switch (debug_opcode) {
      case OpenCLDebugInfo100DebugFunction:
        line_number = lexical_scope_inst->GetSingleWordOperand(
            kLineOperandIndexDebugFunction);
        break;
      case OpenCLDebugInfo100DebugLexicalBlock:
        line_number = lexical_scope_inst->GetSingleWordOperand(
            kLineOperandIndexDebugLexicalBlock);
        break;
      case OpenCLDebugInfo100DebugTypeComposite:
      case OpenCLDebugInfo100DebugCompilationUnit:
        assert(false &&
               "DebugTypeComposite and DebugCompilationUnit are lexical "
               "scopes, but we inline functions into a function or a block "
               "of a function, not into a struct/class or a global scope.");
        break;
      default:
        assert(false &&
               "Unreachable. a debug extension instruction for a "
               "lexical scope must be DebugFunction, DebugTypeComposite, "
               "DebugLexicalBlock, or DebugCompilationUnit.");
        break;
    }
  } else {
    line_number = line->GetSingleWordOperand(kOpLineOperandLineIndex);
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
  RegisterDbgInst(inlined_at.get());
  context()->module()->AddExtInstDebugInfo(std::move(inlined_at));
  return result_id;
}

Instruction* DebugInfoManager::GetDebugInfoNone() {
  if (debug_info_none_inst_ != nullptr) return debug_info_none_inst_;

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

  // Add to the front of |ext_inst_debuginfo_|.
  debug_info_none_inst_ =
      context()->module()->ext_inst_debuginfo_begin()->InsertBefore(
          std::move(dbg_info_none_inst));

  RegisterDbgInst(debug_info_none_inst_);
  return debug_info_none_inst_;
}

Instruction* DebugInfoManager::CloneDebugDeclare(uint32_t orig_var_id,
                                                 uint32_t new_var_id) {
  Instruction* dbgdecl = GetDbgDeclareForVar(orig_var_id);
  if (dbgdecl == nullptr) return nullptr;

  auto* clone = dbgdecl->Clone(context());
  clone->SetResultId(context()->TakeNextId());
  clone->GetOperand(kDebugDeclareOperandVariableIndex).words[0] = new_var_id;

  RegisterDbgDeclareForVar(clone);
  return clone;
}

Instruction* DebugInfoManager::GetDebugInlinedAt(uint32_t dbg_inlined_at_id) {
  auto* inlined_at = GetDbgInst(dbg_inlined_at_id);
  if (inlined_at == nullptr) return nullptr;
  if (inlined_at->GetOpenCL100DebugOpcode() !=
      OpenCLDebugInfo100DebugInlinedAt) {
    return nullptr;
  }
  return inlined_at;
}

Instruction* DebugInfoManager::CloneDebugInlinedAt(uint32_t clone_inlined_at_id,
                                                   Instruction* insert_before) {
  auto* inlined_at = GetDebugInlinedAt(clone_inlined_at_id);
  if (inlined_at == nullptr) return nullptr;
  std::unique_ptr<Instruction> new_inlined_at(inlined_at->Clone(context()));
  new_inlined_at->SetResultId(context()->TakeNextId());
  RegisterDbgInst(new_inlined_at.get());
  if (insert_before != nullptr)
    return insert_before->InsertBefore(std::move(new_inlined_at));
  return context()->module()->ext_inst_debuginfo_end()->InsertBefore(
      std::move(new_inlined_at));
}

void DebugInfoManager::AnalyzeDebugInsts(Module& module) {
  debug_info_none_inst_ = nullptr;
  for (auto& inst : module.ext_inst_debuginfo()) {
    RegisterDbgInst(&inst);
    if (inst.GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction) {
      assert(GetDebugFunction(inst.GetSingleWordOperand(
                 kDebugFunctionOperandFunctionIndex)) == nullptr &&
             "Two DebugFunction instruction exists for a single OpFunction.");
      RegisterDbgFunction(&inst);
    }
    if (debug_info_none_inst_ == nullptr &&
        inst.GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugInfoNone) {
      debug_info_none_inst_ = &inst;
    }
  }

  module.ForEachInst([this](Instruction* cpi) {
    if (cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugDeclare ||
        cpi->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugValue) {
      RegisterDbgDeclareForVar(cpi);
    }
  });
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
