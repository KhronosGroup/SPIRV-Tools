// Copyright (c) 2018 Google LLC
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

#include "upgrade_memory_model.h"

#include "make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status UpgradeMemoryModel::Process(ir::IRContext* context) {
  InitializeProcessing(context);

  ir::Instruction* memory_model = get_module()->GetMemoryModel();
  if (memory_model->GetSingleWordInOperand(0u) != SpvAddressingModelLogical ||
      memory_model->GetSingleWordInOperand(1u) != SpvMemoryModelGLSL450) {
    return Pass::Status::SuccessWithoutChange;
  }

  // Overall changes necessary:
  // 1. Add the OpExtension.
  // 2. Add the OpCapability.
  // 3. Modify the memory model.
  get_module()->AddCapability(MakeUnique<ir::Instruction>(
      context, SpvOpCapability, 0, 0,
      std::initializer_list<ir::Operand>{
          {SPV_OPERAND_TYPE_CAPABILITY, {SpvCapabilityVulkanMemoryModelKHR}}}));
  const std::string extension = "SPV_KHR_vulkan_memory_model";
  std::vector<uint32_t> words(extension.size() / 4 + 1, 0);
  char* dst = reinterpret_cast<char*>(words.data());
  strncpy(dst, extension.c_str(), extension.size());
  get_module()->AddExtension(MakeUnique<ir::Instruction>(
      context, SpvOpExtension, 0, 0,
      std::initializer_list<ir::Operand>{
          {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
  memory_model->SetInOperand(1u, {SpvMemoryModelVulkanKHR});

  return Pass::Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools
