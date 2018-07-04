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

  // Only update Logical GLSL450 to Logical VulkanKHR.
  ir::Instruction* memory_model = get_module()->GetMemoryModel();
  if (memory_model->GetSingleWordInOperand(0u) != SpvAddressingModelLogical ||
      memory_model->GetSingleWordInOperand(1u) != SpvMemoryModelGLSL450) {
    return Pass::Status::SuccessWithoutChange;
  }

  UpgradeMemoryModelInstruction();
  UpgradeInstructions();
  CleanupDecorations();

  return Pass::Status::SuccessWithChange;
}

void UpgradeMemoryModel::UpgradeMemoryModelInstruction() {
  // Overall changes necessary:
  // 1. Add the OpExtension.
  // 2. Add the OpCapability.
  // 3. Modify the memory model.
  ir::Instruction* memory_model = get_module()->GetMemoryModel();
  get_module()->AddCapability(MakeUnique<ir::Instruction>(
      context(), SpvOpCapability, 0, 0,
      std::initializer_list<ir::Operand>{
          {SPV_OPERAND_TYPE_CAPABILITY, {SpvCapabilityVulkanMemoryModelKHR}}}));
  const std::string extension = "SPV_KHR_vulkan_memory_model";
  std::vector<uint32_t> words(extension.size() / 4 + 1, 0);
  char* dst = reinterpret_cast<char*>(words.data());
  strncpy(dst, extension.c_str(), extension.size());
  get_module()->AddExtension(MakeUnique<ir::Instruction>(
      context(), SpvOpExtension, 0, 0,
      std::initializer_list<ir::Operand>{
          {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
  memory_model->SetInOperand(1u, {SpvMemoryModelVulkanKHR});
}

void UpgradeMemoryModel::UpgradeInstructions() {
  // Coherent and Volatile decorations are deprecated. Remove them and replace
  // with flags on the memory/image operations. The decorations can occur on
  // OpVariable, OpFunctionParameter (of pointer type) and OpStructType (member
  // decoration). Trace from the decoration target(s) to the final memory/image
  // instructions. Additionally, Workgroup storage class variables and function
  // parameters are implicitly coherent in GLSL450.

  for (auto global : get_module()->types_values()) {
    std::vector<ir::Instruction*> decorations =
        get_decoration_mgr()->GetDecorationsFor(global.result_id(), false);
    if (global.opcode() == SpvOpTypeStruct) {
    } else if (global.opcode() == SpvOpVariable) {
      bool is_coherent = false;
      bool is_volatile = false;
      for (auto dec : decorations) {
        is_coherent |= dec->GetSingleWordInOperand(1u) == SpvDecorationCoherent;
        is_volatile |= dec->GetSingleWordInOperand(1u) == SpvDecorationVolatile;
      }

      SpvScope scope = SpvScopeDevice;
      SpvStorageClass storage_class =
          static_cast<SpvStorageClass>(global.GetSingleWordInOperand(0u));
      if (storage_class == SpvStorageClassWorkgroup) {
        is_coherent = true;
        scope = SpvScopeWorkgroup;
      }

      // Nothing to upgrade.
      if (!is_coherent && !is_volatile) continue;

      Tracker tracker(&global);
      tracker.is_volatile = is_volatile;
      tracker.is_coherent = is_coherent;
      tracker.in_operand = 0u;
      tracker.nesting = 0u;
      tracker.member_index = -1;
      tracker.scope = scope;
      UpgradeInstruction(tracker);
    }
  }
}

void UpgradeMemoryModel::UpgradeInstruction(const Tracker& tracker) {
  std::vector<Tracker> stack;
  stack.push_back(tracker);

  std::unordered_set<ir::Instruction*> visited;
  while (!stack.empty()) {
    Tracker current = stack.back();
    stack.pop_back();

    if (!visited.insert(current.inst).second) continue;

    UpgradeFlags(current);
    if (current.is_coherent) {
      switch (current.inst->opcode()) {
        case SpvOpLoad:
        case SpvOpStore:
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
        case SpvOpImageWrite:
          current.inst->AddOperand(
              {SPV_OPERAND_TYPE_ID, {GetScopeConstant(current.scope)}});
          break;
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized:
        default:
          break;
      }
    }
  }
}

void UpgradeMemoryModel::UpgradeFlags(const Tracker& tracker) {
  uint32_t flags = 0;
  ir::Instruction* inst = tracker.inst;
  if (tracker.is_volatile) {
    switch (inst->opcode()) {
      case SpvOpLoad:
      case SpvOpStore:
      case SpvOpCopyMemory:
      case SpvOpCopyMemorySized:
        flags |= SpvMemoryAccessVolatileMask;
        break;
      case SpvOpImageRead:
      case SpvOpImageSparseRead:
      case SpvOpImageWrite:
        flags |= SpvImageOperandsVolatileTexelKHRMask;
        break;
      default:
        break;
    }
  }

  if (tracker.is_coherent) {
    switch (inst->opcode()) {
      case SpvOpLoad:
        flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
        flags |= SpvMemoryAccessMakePointerAvailableKHRMask;
        break;
      case SpvOpStore:
        flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
        flags |= SpvMemoryAccessMakePointerVisibleKHRMask;
        break;
      case SpvOpCopyMemory:
      case SpvOpCopyMemorySized:
        flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
        if (tracker.in_operand == 0u)
          flags |= SpvMemoryAccessMakePointerAvailableKHRMask;
        if (tracker.in_operand == 1u)
          flags |= SpvMemoryAccessMakePointerVisibleKHRMask;
        break;
      case SpvOpImageRead:
      case SpvOpImageSparseRead:
        flags |= SpvImageOperandsNonPrivateTexelKHRMask;
        flags |= SpvImageOperandsMakeTexelAvailableKHRMask;
        break;
      case SpvOpImageWrite:
        flags |= SpvImageOperandsNonPrivateTexelKHRMask;
        flags |= SpvImageOperandsMakeTexelVisibleKHRMask;
        break;
      default:
        break;
    }
  }

  switch (inst->opcode()) {
    case SpvOpLoad:
      flags |= inst->GetSingleWordInOperand(1u);
      inst->SetInOperand(1u, {flags});
      break;
    case SpvOpStore:
    case SpvOpCopyMemory:
    case SpvOpImageRead:
    case SpvOpImageSparseRead:
      flags |= inst->GetSingleWordInOperand(2u);
      inst->SetInOperand(2u, {flags});
      break;
    case SpvOpCopyMemorySized:
    case SpvOpImageWrite:
      flags |= inst->GetSingleWordInOperand(3u);
      inst->SetInOperand(3u, {flags});
      break;
    default:
      break;
  }
}

uint32_t UpgradeMemoryModel::GetScopeConstant(SpvScope scope) {
  analysis::Integer int_ty(32, false);
  uint32_t int_id = context()->get_type_mgr()->GetTypeInstruction(&int_ty);
  const analysis::Constant* constant =
      context()->get_constant_mgr()->GetConstant(
          context()->get_type_mgr()->GetType(int_id), {scope});
  return context()
      ->get_constant_mgr()
      ->GetDefiningInstruction(constant)
      ->result_id();
}

void UpgradeMemoryModel::CleanupDecorations() {
  // All of the volatile and coherent decorations have been dealt with, so now
  // we can just remove them.
  get_module()->ForEachInst([this](ir::Instruction* inst) {
    if (inst->result_id() != 0) {
      context()->get_decoration_mgr()->RemoveDecorationsFrom(
          inst->result_id(), [](const ir::Instruction& dec) {
            switch (dec.opcode()) {
              case SpvOpDecorate:
              case SpvOpDecorateId:
                if (dec.GetSingleWordInOperand(1u) == SpvDecorationCoherent ||
                    dec.GetSingleWordInOperand(1u) == SpvDecorationVolatile)
                  return true;
                break;
              case SpvOpMemberDecorate:
                if (dec.GetSingleWordInOperand(2u) == SpvDecorationCoherent ||
                    dec.GetSingleWordInOperand(2u) == SpvDecorationVolatile)
                  return true;
                break;
              default:
                break;
            }
            return false;
          });
    }
  });
}

}  // namespace opt
}  // namespace spvtools
