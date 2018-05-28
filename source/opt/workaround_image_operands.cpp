// Copyright (c) 2018 Google Inc.
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

#include "workaround_image_operands.h"

#include <list>
#include <stack>

namespace spvtools {
namespace opt {

Pass::Status DeanonymizeImages::Process(ir::IRContext* c) {
  InitializeProcessing(c);
  bool modified = false;
  modified = FixupOpTypeImage();
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool DeanonymizeImages::FixupOpTypeImage() {
  auto* def_use_mgr = context()->get_def_use_mgr();
  auto* type_mgr = context()->get_type_mgr();

  bool modified = false;
  auto* module = get_module();

  std::vector<analysis::Image*> unknown_depth_images;

  for (auto type : *type_mgr) {
    // 2 means no indication as to whether this is a depth or non-depth image
    if (auto* t = type.second->AsImage()) {
      if (t->depth() == 2) {
        unknown_depth_images.push_back(t);
        modified = true;
      }
    }
  }

  std::vector<ir::Instruction*> fixed_instructions;

  for (auto img_type : unknown_depth_images) {
    def_use_mgr->ForEachUse(
        type_mgr->GetId(img_type),
        [this, &fixed_instructions, &def_use_mgr, img_type, type_mgr,
         &modified](ir::Instruction* inst, uint32_t) {
          if (inst->opcode() == SpvOpTypeSampledImage) {
            analysis::SampledImage* sampled_image_type =
                type_mgr->GetType(inst->result_id())->AsSampledImage();

            def_use_mgr->ForEachUse(
                inst, [this, &fixed_instructions, def_use_mgr, img_type,
                       type_mgr, sampled_image_type,
                       &modified](ir::Instruction* op_type_sampled_image_usage,
                                  uint32_t idx) {
                  if (op_type_sampled_image_usage->opcode() ==
                      SpvOpSampledImage) {
                    auto op_sampled_image = op_type_sampled_image_usage;
                    def_use_mgr->ForEachUse(
                        op_sampled_image,
                        [this, &fixed_instructions, op_sampled_image,
                         sampled_image_type, def_use_mgr, img_type, type_mgr,
                         &modified](ir::Instruction* sampled_image_usage,
                                    uint32_t idx) {
                          bool has_depth = false;
                          if (sampled_image_usage->opcode() ==
                                  SpvOpImageDrefGather ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSampleDrefImplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSampleDrefExplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSampleProjDrefImplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSampleProjDrefExplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSparseSampleDrefImplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSparseSampleDrefExplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSparseSampleProjDrefImplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSparseSampleProjDrefExplicitLod ||
                              sampled_image_usage->opcode() ==
                                  SpvOpImageSparseDrefGather) {
                            has_depth = true;
                          }
                          if (!has_depth) {
                            return;
                          }
                          analysis::Type* t = type_mgr->GetType(
                              type_mgr->GetId(img_type->sampled_type()));
                          analysis::Image img(
                              t, img_type->dim(), has_depth ? 1 : 0,
                              img_type->is_arrayed(),
                              img_type->is_multisampled(), img_type->sampled(),
                              img_type->format(), img_type->access_qualifier());
                          for (auto& dec : img_type->decorations()) {
                            img.AddDecoration(std::vector<uint32_t>(dec));
                          }
                          uint32_t new_image =
                              type_mgr->GetTypeInstruction(&img);

                          analysis::SampledImage sampled_img(
                              type_mgr->GetType(new_image));
                          for (auto& dec : sampled_image_type->decorations()) {
                            sampled_img.AddDecoration(
                                std::vector<uint32_t>(dec));
                          }

                          uint32_t sampled_image_type =
                              type_mgr->GetTypeInstruction(&sampled_img);

                          auto new_sampled_image =
                              op_sampled_image->Clone(context());
                          new_sampled_image->SetResultType(sampled_image_type);

                          new_sampled_image->InsertBefore(op_sampled_image);
                          new_sampled_image->SetResultId(
                              context()->TakeNextId());

                          sampled_image_usage->SetOperand(
                              idx, {new_sampled_image->result_id()});
                        });
                  }
                });
          }
        });
  }

  for (auto img_type : unknown_depth_images) {
    analysis::Type* t =
        type_mgr->GetType(type_mgr->GetId(img_type->sampled_type()));
    ir::Instruction* inst =
        get_def_use_mgr()->GetDef(type_mgr->GetId(img_type));
    inst->SetOperand(3, {0});
  }
  return modified;
}  // namespace opt

}  // namespace opt
}  // namespace spvtools
