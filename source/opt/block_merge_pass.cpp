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

#include "block_merge_pass.h"

#include "ir_context.h"
#include "iterator.h"

namespace spvtools {
namespace opt {

void BlockMergePass::KillInstAndName(ir::Instruction* inst) {
  std::vector<ir::Instruction*> to_kill;
  get_def_use_mgr()->ForEachUser(inst, [&to_kill](ir::Instruction* user) {
    if (user->opcode() == SpvOpName) {
      to_kill.push_back(user);
    }
  });
  for (auto i : to_kill) {
    context()->KillInst(i);
  }
  context()->KillInst(inst);
}

bool BlockMergePass::MergeBlocks(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end();) {
    // Find block with single successor which has no other predecessors.
    auto ii = bi->end();
    --ii;
    ir::Instruction* br = &*ii;
    if (br->opcode() != SpvOpBranch) {
      ++bi;
      continue;
    }

    const uint32_t lab_id = br->GetSingleWordInOperand(0);
    if (cfg()->preds(lab_id).size() != 1) {
      ++bi;
      continue;
    }

    bool pred_is_header = IsHeader(&*bi);
    bool succ_is_header = IsHeader(lab_id);
    if (pred_is_header && succ_is_header) {
      // Cannot merge two headers together.
      ++bi;
      continue;
    }

    bool pred_is_merge = IsMerge(&*bi);
    bool succ_is_merge = IsMerge(lab_id);
    if (pred_is_merge && succ_is_merge) {
      // Cannot merge two merges together.
      ++bi;
      continue;
    }

    // Merge blocks.
    ir::Instruction* merge_inst = bi->GetMergeInst();
    context()->KillInst(br);
    auto sbi = bi;
    for (; sbi != func->end(); ++sbi)
      if (sbi->id() == lab_id) break;
    // If bi is sbi's only predecessor, it dominates sbi and thus
    // sbi must follow bi in func's ordering.
    assert(sbi != func->end());
    bi->AddInstructions(&*sbi);
    if (merge_inst) {
      if (pred_is_header && lab_id == merge_inst->GetSingleWordInOperand(0u)) {
        // Merging the header and merge blocks, so remove the structured control
        // flow declaration.
        context()->KillInst(merge_inst);
      } else {
        // Move the merge instruction to just before the terminator.
        merge_inst->InsertBefore(bi->terminator());
      }
    }
    context()->ReplaceAllUsesWith(lab_id, bi->id());
    KillInstAndName(sbi->GetLabelInst());
    (void)sbi.Erase();
    // Reprocess block.
    modified = true;
  }
  return modified;
}

bool BlockMergePass::IsHeader(ir::BasicBlock* block) {
  return block->GetMergeInst() != nullptr;
}

bool BlockMergePass::IsHeader(uint32_t id) {
  return IsHeader(context()->get_instr_block(get_def_use_mgr()->GetDef(id)));
}

bool BlockMergePass::IsMerge(uint32_t id) {
  return !get_def_use_mgr()->WhileEachUse(id, [](ir::Instruction* user,
                                                 uint32_t index) {
    SpvOp op = user->opcode();
    if ((op == SpvOpLoopMerge || op == SpvOpSelectionMerge) && index == 0u) {
      return false;
    }
    return true;
  });
}

bool BlockMergePass::IsMerge(ir::BasicBlock* block) {
  return IsMerge(block->id());
}

void BlockMergePass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize extension whitelist
  InitExtensions();
};

bool BlockMergePass::AllExtensionsSupported() const {
  // If any extension not in whitelist, return false
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
        reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (extensions_whitelist_.find(extName) == extensions_whitelist_.end())
      return false;
  }
  return true;
}

Pass::Status BlockMergePass::ProcessImpl() {
  // Do not process if any disallowed extensions are enabled
  if (!AllExtensionsSupported()) return Status::SuccessWithoutChange;
  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) { return MergeBlocks(fp); };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

BlockMergePass::BlockMergePass() {}

Pass::Status BlockMergePass::Process(ir::IRContext* c) {
  Initialize(c);
  return ProcessImpl();
}

void BlockMergePass::InitExtensions() {
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
      "SPV_EXT_shader_stencil_export",
      "SPV_EXT_shader_viewport_index_layer",
      "SPV_AMD_shader_image_load_store_lod",
      "SPV_AMD_shader_fragment_mask",
      "SPV_EXT_fragment_fully_covered",
      "SPV_AMD_gpu_shader_half_float_fetch",
  });
}

}  // namespace opt
}  // namespace spvtools
