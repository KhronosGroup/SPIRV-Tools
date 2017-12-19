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

#include "local_ssa_elim_pass.h"

#include "cfa.h"
#include "iterator.h"

namespace spvtools {
namespace opt {

bool LocalMultiStoreElimPass::EliminateMultiStoreLocal(ir::Function* func) {
  // Add Phi instructions to the function.
  if (InsertPhiInstructions(func) == Status::SuccessWithoutChange) return false;

  // Remove all target variable stores.
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    std::vector<ir::Instruction*> dead_instructions;
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOpStore) continue;
      uint32_t varId;
      (void)GetPtr(&*ii, &varId);
      if (!IsTargetVar(varId)) continue;
      assert(!HasLoads(varId));
      dead_instructions.push_back(&*ii);
      modified = true;
    }

    while (!dead_instructions.empty()) {
      ir::Instruction* inst = dead_instructions.back();
      dead_instructions.pop_back();
      DCEInst(inst, [&dead_instructions](ir::Instruction* other_inst) {
        auto i = std::find(dead_instructions.begin(), dead_instructions.end(),
                           other_inst);
        if (i != dead_instructions.end()) {
          dead_instructions.erase(i);
        }
      });
    }
  }

  return modified;
}

void LocalMultiStoreElimPass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize extension whitelist
  InitExtensions();
};

bool LocalMultiStoreElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status LocalMultiStoreElimPass::ProcessImpl() {
  // Assumes all control flow structured.
  // TODO(greg-lunarg): Do SSA rewrite for non-structured control flow
  if (!context()->get_feature_mgr()->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;
  // Assumes relaxed logical addressing only (see instruction.h)
  // TODO(greg-lunarg): Add support for physical addressing
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;
  // Do not process if module contains OpGroupDecorate. Additional
  // support required in KillNamesAndDecorates().
  // TODO(greg-lunarg): Add support for OpGroupDecorate
  for (auto& ai : get_module()->annotations())
    if (ai.opcode() == SpvOpGroupDecorate) return Status::SuccessWithoutChange;
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process functions
  ProcessFunction pfn = [this](ir::Function* fp) {
    return EliminateMultiStoreLocal(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LocalMultiStoreElimPass::LocalMultiStoreElimPass() {}

Pass::Status LocalMultiStoreElimPass::Process(ir::IRContext* c) {
  Initialize(c);
  return ProcessImpl();
}

void LocalMultiStoreElimPass::InitExtensions() {
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
