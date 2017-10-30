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

#include "insert_extract_elim.h"

#include "iterator.h"
#include "ir_context.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;

} // anonymous namespace

bool InsertExtractElimPass::ExtInsMatch(const ir::Instruction* extInst,
    const ir::Instruction* insInst) const {
  if (extInst->NumInOperands() != insInst->NumInOperands() - 1)
    return false;
  uint32_t numIdx = extInst->NumInOperands() - 1;
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetSingleWordInOperand(i + 1) !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::ExtInsConflict(const ir::Instruction* extInst,
    const ir::Instruction* insInst) const {
  if (extInst->NumInOperands() == insInst->NumInOperands() - 1)
    return false;
  uint32_t extNumIdx = extInst->NumInOperands() - 1;
  uint32_t insNumIdx = insInst->NumInOperands() - 2;
  uint32_t numIdx = std::min(extNumIdx, insNumIdx);
  for (uint32_t i = 0; i < numIdx; ++i)
    if (extInst->GetSingleWordInOperand(i + 1) !=
        insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool InsertExtractElimPass::IsVectorType(uint32_t typeId) {
  ir::Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  return typeInst->opcode() == SpvOpTypeVector;
}

bool InsertExtractElimPass::EliminateInsertExtract(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      switch (ii->opcode()) {
        case SpvOpCompositeExtract: {
          uint32_t cid = ii->GetSingleWordInOperand(kExtractCompositeIdInIdx);
          ir::Instruction* cinst = get_def_use_mgr()->GetDef(cid);
          uint32_t replId = 0;
          while (cinst->opcode() == SpvOpCompositeInsert) {
            if (ExtInsConflict(&*ii, cinst))
              break;
            if (ExtInsMatch(&*ii, cinst)) {
              replId = cinst->GetSingleWordInOperand(kInsertObjectIdInIdx);
              break;
            }
            cid = cinst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
            cinst = get_def_use_mgr()->GetDef(cid);
          }
          // If search ended with CompositeConstruct or ConstantComposite
          // and the extract has one index, return the appropriate component.
          // If a vector CompositeConstruct we make sure all preceding
          // components are of component type (not vector composition).
          // TODO(greg-lunarg): Handle multiple-indices, ConstantNull, special
          // vector composition, and additional CompositeInsert.
          if ((cinst->opcode() == SpvOpCompositeConstruct ||
               cinst->opcode() == SpvOpConstantComposite) &&
              (*ii).NumInOperands() == 2) {
            uint32_t compIdx = (*ii).GetSingleWordInOperand(1);
            if (IsVectorType(cinst->type_id())) {
              if (compIdx < cinst->NumInOperands()) {
                uint32_t i = 0;
                for (; i <= compIdx; i++) {
                  uint32_t compId = cinst->GetSingleWordInOperand(i);
                  ir::Instruction* compInst = get_def_use_mgr()->GetDef(compId);
                  if (compInst->type_id() != (*ii).type_id())
                    break;
                }
                if (i > compIdx)
                  replId = cinst->GetSingleWordInOperand(compIdx);
              }
            }
            else {
              replId = cinst->GetSingleWordInOperand(compIdx);
            }
          }
          if (replId != 0) {
            const uint32_t extId = ii->result_id();
            (void)get_def_use_mgr()->ReplaceAllUsesWith(extId, replId);
            get_def_use_mgr()->KillInst(&*ii);
            modified = true;
          }
        } break;
        default:
          break;
      }
    }
  }
  return modified;
}

void InsertExtractElimPass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize extension whitelist
  InitExtensions();
};

bool InsertExtractElimPass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName = reinterpret_cast<const char*>(
        &ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status InsertExtractElimPass::ProcessImpl() {
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported())
    return Status::SuccessWithoutChange;
  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) {
    return EliminateInsertExtract(fp);
  };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InsertExtractElimPass::InsertExtractElimPass() {}

Pass::Status InsertExtractElimPass::Process(ir::IRContext* c) {
  Initialize(c);
  return ProcessImpl();
}

void InsertExtractElimPass::InitExtensions() {
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
    "SPV_KHR_variable_pointers",
    "SPV_AMD_gpu_shader_int16",
    "SPV_KHR_post_depth_coverage",
    "SPV_KHR_shader_atomic_counter_ops",
  });
}

}  // namespace opt
}  // namespace spvtools

