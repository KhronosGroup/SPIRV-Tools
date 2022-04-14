// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#include "source/opt/eliminate_dead_output_stores_pass.h"

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace {

const uint32_t kDecorationLocationInIdx = 2;
const uint32_t kOpDecorateMemberMemberInIdx = 1;
const uint32_t kOpDecorateMemberLocationInIdx = 3;

}  // namespace

namespace spvtools {
namespace opt {

Pass::Status EliminateDeadOutputStoresPass::Process() {
  // Current functionality assumes shader capability
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;
  Pass::Status status;
  if (analyze_)
    status = DoDeadOutputStoreAnalysis();
  else
    status = DoDeadOutputStoreElimination();
  return status;
}

Pass::Status EliminateDeadOutputStoresPass::DoDeadOutputStoreAnalysis() {
  // Current functionality only supports frag, tesc, tese or geom shaders.
  // Report failure for any other stage.
  auto stage = get_stage();
  if (stage != SpvExecutionModelFragment &&
      stage != SpvExecutionModelTessellationControl &&
      stage != SpvExecutionModelTessellationEvaluation &&
      stage != SpvExecutionModelGeometry)
    return Status::Failure;
  InitializeAnalysis();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Process all input variables
  for (auto& var : context()->types_values()) {
    if (var.opcode() != SpvOpVariable) {
      continue;
    }
    analysis::Type* var_type = type_mgr->GetType(var.type_id());
    analysis::Pointer* ptr_type = var_type->AsPointer();
    if (ptr_type->storage_class() != SpvStorageClassInput) {
      continue;
    }
    // If builtin var, continue to next variable
    auto var_id = var.result_id();
    if (deco_mgr->HasDecoration(var_id, SpvDecorationBuiltIn)) continue;
    // If interface block with builtin members, continue to next variable.
    // Input interface blocks will only appear in tesc, tese and geom shaders.
    // Will need to strip off one level of arrayness to get to block type.
    auto pte_type = ptr_type->pointee_type();
    auto arr_type = pte_type->AsArray();
    if (arr_type) {
      auto elt_type = arr_type->element_type();
      auto str_type = elt_type->AsStruct();
      if (str_type) {
        auto str_type_id = type_mgr->GetId(str_type);
        if (deco_mgr->HasDecoration(str_type_id, SpvDecorationBuiltIn))
          continue;
      }
    }
    // Mark all used locations of var live
    def_use_mgr->ForEachUser(var_id, [this, &var](Instruction* user) {
      auto op = user->opcode();
      if (op == SpvOpEntryPoint || op == SpvOpName || op == SpvOpDecorate)
        return;
      MarkRefLive(user, &var);
    });
  }
  // This function does not change the module
  return Status::SuccessWithoutChange;
}

void EliminateDeadOutputStoresPass::InitializeAnalysis() {
  live_inputs_->clear();
}

void EliminateDeadOutputStoresPass::InitializeElimination() {
  kill_list_.clear();
}

void EliminateDeadOutputStoresPass::MarkLocsLive(uint32_t start, uint32_t num) {
  auto finish = start + num;
  for (uint32_t u = start; u < finish; ++u) {
    live_inputs_->insert(u);
  }
}

bool EliminateDeadOutputStoresPass::AnyLocsAreLive(uint32_t start,
                                                   uint32_t num) {
  auto finish = start + num;
  for (uint32_t u = start; u < finish; ++u) {
    if (live_inputs_->find(u) != live_inputs_->end()) return true;
  }
  return false;
}

uint32_t EliminateDeadOutputStoresPass::GetLocSize(
    const analysis::Type* type) const {
  auto arr_type = type->AsArray();
  if (arr_type) {
    auto comp_type = arr_type->element_type();
    auto len_info = arr_type->length_info();
    assert(len_info.words[0] == analysis::Array::LengthInfo::kConstant &&
           "unexpected array length");
    auto comp_len = len_info.words[1];
    return comp_len * GetLocSize(comp_type);
  }
  auto struct_type = type->AsStruct();
  if (struct_type) {
    uint32_t size = 0u;
    for (auto& el_type : struct_type->element_types())
      size += GetLocSize(el_type);
    return size;
  }
  auto mat_type = type->AsMatrix();
  if (mat_type) {
    auto cnt = mat_type->element_count();
    auto comp_type = mat_type->element_type();
    return cnt * GetLocSize(comp_type);
  }
  auto vec_type = type->AsVector();
  if (vec_type) {
    auto comp_type = vec_type->element_type();
    if (comp_type->AsInteger()) return 1;
    auto float_type = comp_type->AsFloat();
    assert(float_type && "unexpected vector component type");
    auto width = float_type->width();
    if (width == 32) return 1;
    assert(width == 64 && "unexpected float type width");
    auto comp_cnt = vec_type->element_count();
    return (comp_cnt > 2) ? 2 : 1;
  }
  assert((type->AsInteger() || type->AsFloat()) && "unexpected input type");
  return 1;
}

const analysis::Type* EliminateDeadOutputStoresPass::GetComponentType(
    uint32_t index, const analysis::Type* agg_type) const {
  auto arr_type = agg_type->AsArray();
  if (arr_type) return arr_type->element_type();
  auto struct_type = agg_type->AsStruct();
  if (struct_type) return struct_type->element_types()[index];
  auto mat_type = agg_type->AsMatrix();
  if (mat_type) return mat_type->element_type();
  auto vec_type = agg_type->AsVector();
  assert(vec_type && "unexpected non-aggregate type");
  return vec_type->element_type();
}

uint32_t EliminateDeadOutputStoresPass::GetLocOffset(
    uint32_t index, const analysis::Type* agg_type) const {
  auto arr_type = agg_type->AsArray();
  if (arr_type) return index * GetLocSize(arr_type->element_type());
  auto struct_type = agg_type->AsStruct();
  if (struct_type) {
    uint32_t offset = 0u;
    uint32_t cnt = 0u;
    for (auto& el_type : struct_type->element_types()) {
      if (cnt == index) break;
      offset += GetLocSize(el_type);
      ++cnt;
    }
    return offset;
  }
  auto mat_type = agg_type->AsMatrix();
  if (mat_type) return index * GetLocSize(mat_type->element_type());
  auto vec_type = agg_type->AsVector();
  assert(vec_type && "unexpected non-aggregate type");
  auto comp_type = vec_type->element_type();
  auto flt_type = comp_type->AsFloat();
  if (flt_type && flt_type->width() == 64u && index >= 2u) return 1;
  return 0;
}

void EliminateDeadOutputStoresPass::AnalyzeAccessChain(
    const Instruction* ac, const analysis::Type** curr_type, uint32_t* offset,
    bool* no_loc, bool input) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // For tesc, tese and geom input shaders, and tesc output shaders,
  // first array index does not contribute to offset.
  auto stage = get_stage();
  bool skip_first_index = false;
  if ((input && (stage == SpvExecutionModelTessellationControl ||
                 stage == SpvExecutionModelTessellationEvaluation ||
                 stage == SpvExecutionModelGeometry)) ||
      (!input && stage == SpvExecutionModelTessellationControl))
    skip_first_index = true;
  uint32_t ocnt = 0;
  ac->WhileEachInOperand([this, &ocnt, def_use_mgr, type_mgr, deco_mgr,
                          &curr_type, &offset, no_loc,
                          skip_first_index](const uint32_t* opnd) {
    if (ocnt >= 1) {
      // Skip first index's contribution to offset if indicated
      if (ocnt == 1 && skip_first_index) {
        auto arr_type = (*curr_type)->AsArray();
        assert(arr_type && "unexpected wrapper type");
        *curr_type = arr_type->element_type();
        ocnt++;
        return true;
      }
      // If any non-constant index, mark the entire current object and return.
      auto idx_inst = def_use_mgr->GetDef(*opnd);
      if (idx_inst->opcode() != SpvOpConstant) return false;
      // If current type is struct, look for location decoration on member and
      // reset offset if found.
      auto index = idx_inst->GetSingleWordInOperand(0);
      auto str_type = (*curr_type)->AsStruct();
      if (str_type) {
        uint32_t loc = 0;
        auto str_type_id = type_mgr->GetId(str_type);
        bool no_mem_loc = deco_mgr->WhileEachDecoration(
            str_type_id, SpvDecorationLocation,
            [&loc, index, no_loc](const Instruction& deco) {
              assert(deco.opcode() == SpvOpMemberDecorate &&
                     "unexpected decoration");
              if (deco.GetSingleWordInOperand(kOpDecorateMemberMemberInIdx) ==
                  index) {
                loc =
                    deco.GetSingleWordInOperand(kOpDecorateMemberLocationInIdx);
                *no_loc = false;
                return false;
              }
              return true;
            });
        if (!no_mem_loc) {
          *offset = loc;
          *curr_type = GetComponentType(index, *curr_type);
          ocnt++;
          return true;
        }
      }

      // Update offset and current type based on constant index.
      *offset += GetLocOffset(index, *curr_type);
      *curr_type = GetComponentType(index, *curr_type);
    }
    ocnt++;
    return true;
  });
}

void EliminateDeadOutputStoresPass::MarkRefLive(const Instruction* ref,
                                                Instruction* var) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Find variable location if present.
  uint32_t loc = 0;
  auto var_id = var->result_id();
  bool no_loc = deco_mgr->WhileEachDecoration(
      var_id, SpvDecorationLocation, [&loc](const Instruction& deco) {
        assert(deco.opcode() == SpvOpDecorate && "unexpected decoration");
        loc = deco.GetSingleWordInOperand(kDecorationLocationInIdx);
        return false;
      });
  // If use is a load, mark all locations of var
  auto ptr_type = type_mgr->GetType(var->type_id())->AsPointer();
  assert(ptr_type && "unexpected var type");
  auto var_type = ptr_type->pointee_type();
  if (ref->opcode() == SpvOpLoad) {
    assert(!no_loc && "missing input variable location");
    MarkLocsLive(loc, GetLocSize(var_type));
    return;
  }
  // Mark just those locations indicated by access chain
  assert((ref->opcode() == SpvOpAccessChain ||
          ref->opcode() == SpvOpInBoundsAccessChain) &&
         "unexpected use of input variable");
  // Traverse access chain, compute location offset and type of reference
  // through constant indices and mark those locs live. Assert if no location
  // found.
  uint32_t offset = loc;
  auto curr_type = var_type;
  AnalyzeAccessChain(ref, &curr_type, &offset, &no_loc);
  assert(!no_loc && "missing input variable location");
  MarkLocsLive(offset, GetLocSize(curr_type));
}

void EliminateDeadOutputStoresPass::KillAllDeadStoresOfRef(Instruction* ref,
                                                           Instruction* var) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Find variable location if present.
  uint32_t loc = 0;
  auto var_id = var->result_id();
  bool no_loc = deco_mgr->WhileEachDecoration(
      var_id, SpvDecorationLocation, [&loc](const Instruction& deco) {
        assert(deco.opcode() == SpvOpDecorate && "unexpected decoration");
        loc = deco.GetSingleWordInOperand(kDecorationLocationInIdx);
        return false;
      });
  // Compute offset and final type of reference. If no location found
  // or any stored locations are live, return without removing stores.
  auto ptr_type = type_mgr->GetType(var->type_id())->AsPointer();
  assert(ptr_type && "unexpected var type");
  auto var_type = ptr_type->pointee_type();
  uint32_t offset = loc;
  auto curr_type = var_type;
  if (ref->opcode() == SpvOpAccessChain ||
      ref->opcode() == SpvOpInBoundsAccessChain)
    AnalyzeAccessChain(ref, &curr_type, &offset, &no_loc, /* input */ false);
  if (no_loc || AnyLocsAreLive(offset, GetLocSize(curr_type))) return;
  // Kill all stores based on this reference
  if (ref->opcode() == SpvOpStore) {
    kill_list_.push_back(ref);
  } else {
    assert((ref->opcode() == SpvOpAccessChain ||
            ref->opcode() == SpvOpInBoundsAccessChain) &&
           "unexpected use of output variable");
    def_use_mgr->ForEachUser(ref, [this](Instruction* user) {
      if (user->opcode() == SpvOpStore) kill_list_.push_back(user);
    });
  }
}

Pass::Status EliminateDeadOutputStoresPass::DoDeadOutputStoreElimination() {
  // Current implementation only supports vert, tesc, tese, geom shaders
  auto stage = get_stage();
  if (stage != SpvExecutionModelVertex &&
      stage != SpvExecutionModelTessellationControl &&
      stage != SpvExecutionModelTessellationEvaluation &&
      stage != SpvExecutionModelGeometry)
    return Status::Failure;
  InitializeElimination();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Process all output variables
  for (auto& var : context()->types_values()) {
    if (var.opcode() != SpvOpVariable) {
      continue;
    }
    analysis::Type* var_type = type_mgr->GetType(var.type_id());
    analysis::Pointer* ptr_type = var_type->AsPointer();
    if (ptr_type->storage_class() != SpvStorageClassOutput) {
      continue;
    }
    // If builtin decoration on variable, continue to next variable
    auto var_id = var.result_id();
    if (deco_mgr->HasDecoration(var_id, SpvDecorationBuiltIn)) continue;
    // If interface block with builtin members, continue to next variable.
    // Strip off outer array type if present.
    auto curr_type = ptr_type->pointee_type();
    auto arr_type = curr_type->AsArray();
    if (arr_type) curr_type = arr_type->element_type();
    auto str_type = curr_type->AsStruct();
    if (str_type) {
      auto str_type_id = type_mgr->GetId(str_type);
      if (deco_mgr->HasDecoration(str_type_id, SpvDecorationBuiltIn)) continue;
    }
    // For each store or access chain using var, if all its locations are dead,
    // kill store or all access chain's stores
    def_use_mgr->ForEachUser(var_id, [this, &var](Instruction* user) {
      auto op = user->opcode();
      if (op == SpvOpEntryPoint || op == SpvOpName || op == SpvOpDecorate)
        return;
      KillAllDeadStoresOfRef(user, &var);
    });
  }
  for (auto& kinst : kill_list_) context()->KillInst(kinst);

  return kill_list_.empty() ? Status::SuccessWithoutChange
                            : Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools
