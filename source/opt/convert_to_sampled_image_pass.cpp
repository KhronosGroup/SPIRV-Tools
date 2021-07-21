// Copyright (c) 2021 Google LLC
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

#include "source/opt/convert_to_sampled_image_pass.h"

#include <cctype>
#include <cstring>
#include <tuple>

#include "source/util/make_unique.h"
#include "source/util/parse_number.h"

namespace spvtools {
namespace opt {

using VectorOfDescriptorSetAndBindingPairs =
    std::vector<DescriptorSetAndBinding>;
using DescriptorSetBindingToInstruction =
    ConvertToSampledImagePass::DescriptorSetBindingToInstruction;

namespace {

using utils::ParseNumber;

// Inserts a pair of |descriptor_set_binding| and |inst| to
// |descriptor_set_binding_to_inst| if the map does not contain
// |descriptor_set_binding_to| key. Returns true if the insertion took place.
bool InsertDescriptorSetBindingAndInstructionPair(
    DescriptorSetBindingToInstruction* descriptor_set_binding_to_inst,
    const DescriptorSetAndBinding& descriptor_set_binding, Instruction* inst) {
  if (descriptor_set_binding_to_inst->find(descriptor_set_binding) !=
      descriptor_set_binding_to_inst->end()) {
    return false;
  }
  return descriptor_set_binding_to_inst->insert({descriptor_set_binding, inst})
      .second;
}

// Returns true if the given char is ':', '\0' or considered as blank space
// (i.e.: '\n', '\r', '\v', '\t', '\f' and ' ').
bool IsSeparator(char ch) {
  return std::strchr(":\0", ch) || std::isspace(ch) != 0;
}

// Reads characters starting from |str| until it meets a separator. Parses a
// number from the characters and stores it into |number|. Returns the pointer
// to the separator if it succeeds. Otherwise, returns nullptr.
const char* ParseNumberUntilSeparator(const char* str, uint32_t* number) {
  const char* number_begin = str;
  while (!IsSeparator(*str)) str++;
  const char* number_end = str;
  std::string number_in_str(number_begin, number_end - number_begin);
  if (!utils::ParseNumber(number_in_str.c_str(), number)) {
    // The descriptor set is not a valid uint32 number.
    return nullptr;
  }
  return str;
}

}  // namespace

bool ConvertToSampledImagePass::GetDescriptorSetBinding(
    const Instruction& inst,
    DescriptorSetAndBinding* descriptor_set_binding) const {
  auto* decoration_manager = context()->get_decoration_mgr();
  bool found_descriptor_set_to_convert = false;
  bool found_binding_to_convert = false;
  for (auto decorate :
       decoration_manager->GetDecorationsFor(inst.result_id(), false)) {
    uint32_t decoration = decorate->GetSingleWordInOperand(1u);
    if (decoration == SpvDecorationDescriptorSet) {
      if (found_descriptor_set_to_convert) {
        assert(false && "A resource has two OpDecorate for the descriptor set");
        return false;
      }
      descriptor_set_binding->descriptor_set =
          decorate->GetSingleWordInOperand(2u);
      found_descriptor_set_to_convert = true;
    } else if (decoration == SpvDecorationBinding) {
      if (found_binding_to_convert) {
        assert(false && "A resource has two OpDecorate for the binding");
        return false;
      }
      descriptor_set_binding->binding = decorate->GetSingleWordInOperand(2u);
      found_binding_to_convert = true;
    }
  }
  return found_descriptor_set_to_convert && found_binding_to_convert;
}

bool ConvertToSampledImagePass::
    IsDescriptorSetBindingPairForSampledImageConversion(
        const DescriptorSetAndBinding& descriptor_set_binding) const {
  return descriptor_set_binding_pairs_.find(descriptor_set_binding) !=
         descriptor_set_binding_pairs_.end();
}

const analysis::Type* ConvertToSampledImagePass::GetVariableType(
    const Instruction& variable) const {
  if (variable.opcode() != SpvOpVariable) return nullptr;
  auto* type = context()->get_type_mgr()->GetType(variable.type_id());
  auto* pointer_type = type->AsPointer();
  if (!pointer_type) return nullptr;

  return pointer_type->pointee_type();
}

SpvStorageClass ConvertToSampledImagePass::GetStorageClass(
    const Instruction& variable) const {
  assert(variable.opcode() == SpvOpVariable);
  auto* type = context()->get_type_mgr()->GetType(variable.type_id());
  auto* pointer_type = type->AsPointer();
  if (!pointer_type) return SpvStorageClassMax;

  return pointer_type->storage_class();
}

bool ConvertToSampledImagePass::CollectResourcesToConvert(
    DescriptorSetBindingToInstruction* descriptor_set_binding_pair_to_sampler,
    DescriptorSetBindingToInstruction* descriptor_set_binding_pair_to_image)
    const {
  for (auto& inst : context()->types_values()) {
    const auto* variable_type = GetVariableType(inst);
    if (variable_type == nullptr) continue;

    DescriptorSetAndBinding descriptor_set_binding;
    if (!GetDescriptorSetBinding(inst, &descriptor_set_binding)) continue;

    if (!IsDescriptorSetBindingPairForSampledImageConversion(
            descriptor_set_binding)) {
      continue;
    }

    if (variable_type->AsImage()) {
      if (!InsertDescriptorSetBindingAndInstructionPair(
              descriptor_set_binding_pair_to_image, descriptor_set_binding,
              &inst)) {
        return false;
      }
    } else if (variable_type->AsSampler()) {
      if (!InsertDescriptorSetBindingAndInstructionPair(
              descriptor_set_binding_pair_to_sampler, descriptor_set_binding,
              &inst)) {
        return false;
      }
    }
  }
  return true;
}

Pass::Status ConvertToSampledImagePass::Process() {
  Status status = Status::SuccessWithoutChange;

  DescriptorSetBindingToInstruction descriptor_set_binding_pair_to_sampler,
      descriptor_set_binding_pair_to_image;
  if (!CollectResourcesToConvert(&descriptor_set_binding_pair_to_sampler,
                                 &descriptor_set_binding_pair_to_image)) {
    return Status::Failure;
  }

  for (auto& image : descriptor_set_binding_pair_to_image) {
    status = CombineStatus(
        status, UpdateImageVariableToSampledImage(image.second, image.first));
    if (status == Status::Failure) {
      return status;
    }
  }

  for (const auto& sampler : descriptor_set_binding_pair_to_sampler) {
    // Converting only a Sampler to Sampled Image is not allowed. It must have a
    // corresponding image to combine the sampler with.
    auto image_itr = descriptor_set_binding_pair_to_image.find(sampler.first);
    if (image_itr == descriptor_set_binding_pair_to_image.end() ||
        image_itr->second == nullptr) {
      return Status::Failure;
    }

    status = CombineStatus(
        status, CheckUsesOfSamplerVariable(sampler.second, image_itr->second));
    if (status == Status::Failure) {
      return status;
    }
  }

  return status;
}

void ConvertToSampledImagePass::FindUses(const Instruction* inst,
                                         std::vector<Instruction*>* uses,
                                         uint32_t user_opcode) const {
  auto* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->ForEachUser(inst, [uses, user_opcode, this](Instruction* user) {
    if (user->opcode() == user_opcode) {
      uses->push_back(user);
    } else if (user->opcode() == SpvOpCopyObject) {
      FindUses(user, uses, user_opcode);
    }
  });
}

void ConvertToSampledImagePass::FindUsesOfImage(
    const Instruction* image, std::vector<Instruction*>* uses) const {
  auto* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->ForEachUser(image, [uses, this](Instruction* user) {
    switch (user->opcode()) {
      case SpvOpImageFetch:
      case SpvOpImageRead:
      case SpvOpImageWrite:
      case SpvOpImageQueryFormat:
      case SpvOpImageQueryOrder:
      case SpvOpImageQuerySizeLod:
      case SpvOpImageQuerySize:
      case SpvOpImageQueryLevels:
      case SpvOpImageQuerySamples:
      case SpvOpImageSparseFetch:
        uses->push_back(user);
      default:
        break;
    }
    if (user->opcode() == SpvOpCopyObject) {
      FindUsesOfImage(user, uses);
    }
  });
}

void ConvertToSampledImagePass::ReplaceUsesWith(
    const Instruction* inst, uint32_t replaced_inst_id) const {
  auto* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->ForEachUser(
      inst, [inst, replaced_inst_id, this](Instruction* user) {
        user->ForEachInOperand([inst, replaced_inst_id](uint32_t* operand) {
          if (*operand == inst->result_id()) *operand = replaced_inst_id;
        });
      });
}

Instruction* ConvertToSampledImagePass::CreateImageExtraction(
    Instruction* sampled_image) {
  auto* type_mgr = context()->get_type_mgr();
  auto* sampled_image_type =
      type_mgr->GetType(sampled_image->type_id())->AsSampledImage();
  std::unique_ptr<Instruction> image_extraction(new Instruction(
      context(), SpvOpImage,
      type_mgr->GetTypeInstruction(sampled_image_type->image_type()),
      context()->TakeNextId(),
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
           {sampled_image->result_id()}},
      }));
  context()->get_def_use_mgr()->AnalyzeInstDefUse(image_extraction.get());
  sampled_image->NextNode()->InsertBefore(std::move(image_extraction));
  return sampled_image->NextNode();
}

uint32_t ConvertToSampledImagePass::GetSampledImageTypeForImage(
    Instruction* image_variable) {
  const auto* variable_type = GetVariableType(*image_variable);
  if (variable_type == nullptr) return 0;
  const auto* image_type = variable_type->AsImage();
  if (image_type == nullptr) return 0;

  analysis::Image image_type_for_sampled_image(*image_type);
  analysis::SampledImage sampled_image_type(&image_type_for_sampled_image);
  return context()->get_type_mgr()->GetTypeInstruction(&sampled_image_type);
}

Instruction* ConvertToSampledImagePass::UpdateImageUses(
    Instruction* sampled_image_load) {
  std::vector<Instruction*> uses_of_load;
  FindUsesOfImage(sampled_image_load, &uses_of_load);
  if (uses_of_load.empty()) return nullptr;

  auto* extracted_image = CreateImageExtraction(sampled_image_load);
  for (auto* user : uses_of_load) {
    user->SetInOperand(0, {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                           {extracted_image->result_id()}});
    context()->get_def_use_mgr()->AnalyzeInstUse(user);
  }
  return extracted_image;
}

bool ConvertToSampledImagePass::
    IsSamplerOfSampledImageDecoratedByDescriptorSetBinding(
        Instruction* sampled_image_inst,
        const DescriptorSetAndBinding& descriptor_set_binding) {
  auto* def_use_mgr = context()->get_def_use_mgr();
  uint32_t sampler_id = sampled_image_inst->GetSingleWordInOperand(1u);
  auto* sampler_load = def_use_mgr->GetDef(sampler_id);
  if (sampler_load->opcode() != SpvOpLoad) return false;
  auto* sampler = def_use_mgr->GetDef(sampler_load->GetSingleWordInOperand(0u));
  DescriptorSetAndBinding sampler_descriptor_set_binding;
  return GetDescriptorSetBinding(*sampler, &sampler_descriptor_set_binding) &&
         sampler_descriptor_set_binding == descriptor_set_binding;
}

void ConvertToSampledImagePass::UpdateSampledImageUses(
    Instruction* image_load, Instruction* image_extraction,
    const DescriptorSetAndBinding& image_descriptor_set_binding) {
  std::vector<Instruction*> sampled_image_users;
  FindUses(image_load, &sampled_image_users, SpvOpSampledImage);

  auto* def_use_mgr = context()->get_def_use_mgr();
  for (auto* sampled_image_inst : sampled_image_users) {
    if (IsSamplerOfSampledImageDecoratedByDescriptorSetBinding(
            sampled_image_inst, image_descriptor_set_binding)) {
      ReplaceUsesWith(sampled_image_inst, image_load->result_id());
      def_use_mgr->AnalyzeInstUse(image_load);
      context()->KillInst(sampled_image_inst);
    } else {
      if (!image_extraction)
        image_extraction = CreateImageExtraction(image_load);
      sampled_image_inst->SetInOperand(0, {image_extraction->result_id()});
      def_use_mgr->AnalyzeInstUse(sampled_image_inst);
    }
  }
}

void ConvertToSampledImagePass::MoveInstructionNextToType(Instruction* inst,
                                                          uint32_t type_id) {
  auto* type_inst = context()->get_def_use_mgr()->GetDef(type_id);
  inst->SetResultType(type_id);
  inst->RemoveFromList();
  inst->InsertAfter(type_inst);
}

bool ConvertToSampledImagePass::ConvertImageVariableToSampledImage(
    Instruction* image_variable, uint32_t sampled_image_type_id) {
  auto* sampled_image_type =
      context()->get_type_mgr()->GetType(sampled_image_type_id);
  if (sampled_image_type == nullptr) return false;
  auto storage_class = GetStorageClass(*image_variable);
  if (storage_class == SpvStorageClassMax) return false;
  analysis::Pointer sampled_image_pointer(sampled_image_type, storage_class);

  // Make sure |image_variable| is behind its type i.e., avoid the forward
  // reference.
  uint32_t type_id =
      context()->get_type_mgr()->GetTypeInstruction(&sampled_image_pointer);
  MoveInstructionNextToType(image_variable, type_id);
  return true;
}

Pass::Status ConvertToSampledImagePass::UpdateImageVariableToSampledImage(
    Instruction* image_variable,
    const DescriptorSetAndBinding& descriptor_set_binding) {
  std::vector<Instruction*> image_variable_loads;
  FindUses(image_variable, &image_variable_loads, SpvOpLoad);
  if (image_variable_loads.empty()) return Status::SuccessWithoutChange;

  const uint32_t sampled_image_type_id =
      GetSampledImageTypeForImage(image_variable);
  if (!sampled_image_type_id) return Status::Failure;

  for (auto* load : image_variable_loads) {
    load->SetResultType(sampled_image_type_id);
    auto* image_extraction = UpdateImageUses(load);
    UpdateSampledImageUses(load, image_extraction, descriptor_set_binding);
  }

  return ConvertImageVariableToSampledImage(image_variable,
                                            sampled_image_type_id)
             ? Status::SuccessWithChange
             : Status::Failure;
}

bool ConvertToSampledImagePass::DoesSampledImageReferenceImage(
    Instruction* sampled_image_inst, Instruction* image_variable) {
  if (sampled_image_inst->opcode() != SpvOpSampledImage) return false;
  auto* def_use_mgr = context()->get_def_use_mgr();
  auto* image_load =
      def_use_mgr->GetDef(sampled_image_inst->GetSingleWordInOperand(0u));
  if (image_load->opcode() != SpvOpLoad) return false;
  auto* image = def_use_mgr->GetDef(image_load->GetSingleWordInOperand(0u));
  return image->opcode() == SpvOpVariable &&
         image->result_id() == image_variable->result_id();
}

Pass::Status ConvertToSampledImagePass::CheckUsesOfSamplerVariable(
    const Instruction* sampler_variable,
    Instruction* image_to_be_combined_with) {
  if (image_to_be_combined_with == nullptr) return Status::Failure;

  std::vector<Instruction*> sampler_variable_loads;
  FindUses(sampler_variable, &sampler_variable_loads, SpvOpLoad);
  for (auto* load : sampler_variable_loads) {
    std::vector<Instruction*> sampled_image_users;
    FindUses(load, &sampled_image_users, SpvOpSampledImage);
    for (auto* sampled_image_inst : sampled_image_users) {
      if (!DoesSampledImageReferenceImage(sampled_image_inst,
                                          image_to_be_combined_with)) {
        return Status::Failure;
      }
    }
  }
  return Status::SuccessWithoutChange;
}

std::unique_ptr<VectorOfDescriptorSetAndBindingPairs>
ConvertToSampledImagePass::ParseDescriptorSetBindingPairsString(
    const char* str) {
  if (!str) return nullptr;

  auto descriptor_set_binding_pairs =
      MakeUnique<VectorOfDescriptorSetAndBindingPairs>();

  // The parsing loop, break when points to the end.
  while (*str) {
    while (std::isspace(*str)) str++;  // skip leading spaces.

    // Parse the descriptor set.
    uint32_t descriptor_set = 0;
    str = ParseNumberUntilSeparator(str, &descriptor_set);
    if (str == nullptr) return nullptr;

    // Find the ':', spaces between the descriptor set and the ':' are not
    // allowed.
    if (*str++ != ':') {
      // ':' not found
      return nullptr;
    }

    // Parse the binding.
    uint32_t binding = 0;
    str = ParseNumberUntilSeparator(str, &binding);
    if (str == nullptr) return nullptr;

    descriptor_set_binding_pairs->push_back({descriptor_set, binding});

    // Skip trailing spaces.
    while (std::isspace(*str)) str++;
  }

  return descriptor_set_binding_pairs;
}

}  // namespace opt
}  // namespace spvtools
