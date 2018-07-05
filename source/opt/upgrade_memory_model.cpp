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

#include <utility>

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

  for (auto& func : *get_module()) {
    func.ForEachInst([this](ir::Instruction* inst) {
      bool is_coherent = false;
      bool is_volatile = false;
      SpvScope scope = SpvScopeDevice;
      switch (inst->opcode()) {
        case SpvOpLoad:
        case SpvOpStore:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
        case SpvOpImageWrite:
          break;
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized:
          break;
        default:
          break;
      }

      if (!is_coherent && !is_volatile) return;

      switch (inst->opcode()) {
        case SpvOpLoad:
          UpgradeFlags(inst, 1u, is_coherent, is_volatile, false, true);
          inst->AddOperand(
              {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(scope)}});
          break;
        case SpvOpStore:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, true, true);
          inst->AddOperand(
              {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(scope)}});
          break;
        default:
          break;
      }
    });
  }
}

std::tuple<bool, bool, SpvScope> UpgradeMemoryModel::GetInstructionAttributes(
    uint32_t id) {
  // |id| is a pointer used in a memory/image instruction. Need to determine if
  // that pointer points to volatile or coherent memory. Workgroup storage
  // class is implicitly coherent and cannot be decorated with volatile, so
  // short circuit that case.
  ir::Instruction* inst = context()->get_def_use_mgr()->GetDef(id);
  analysis::Type* type = context()->get_type_mgr()->GetType(inst->type_id());
  assert(type->AsPointer());
  if (type->AsPointer()->storage_class() == SpvStorageClassWorkgroup) {
    return std::make_tuple(true, false, SpvScopeWorkgroup);
  }

  bool is_coherent = false;
  bool is_volatile = false;
  std::vector<std::pair<ir::Instruction*, std::vector<uint32_t>>> stack;
  stack.push_back(std::make_pair(context()->get_def_use_mgr()->GetDef(id),
                                 std::vector<uint32_t>()));
  while (!stack.empty()) {
    ir::Instruction* def = stack.back().first;
    std::vector<uint32_t> indices = stack.back().second;
    stack.pop_back();

    switch (inst->opcode()) {
      case SpvOpVariable:
      case SpvOpFunctionParameter:
        is_coherent |= HasDecoration(def, indices, SpvDecorationCoherent);
        is_volatile |= HasDecoration(def, indices, SpvDecorationVolatile);
        break;
      case SpvOpTypePointer:
        break;
      default:
        break;
    }
  }

  return std::make_tuple(is_coherent, is_volatile, SpvScopeDevice);
}

bool UpgradeMemoryModel::HasDecoration(const ir::Instruction* inst,
                                       const std::vector<uint32_t>& indices,
                                       SpvDecoration decoration) {
  (void)indices;
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      inst->result_id(), decoration, [](const ir::Instruction& i) {
        if (i.opcode() == SpvOpDecorate || i.opcode() == SpvOpDecorateId) {
          return false;
        } else if (i.opcode() == SpvOpMemberDecorate) {
        }

        return true;
      });
}

void UpgradeMemoryModel::UpgradeFlags(ir::Instruction* inst,
                                      uint32_t in_operand, bool is_coherent,
                                      bool is_volatile, bool visible,
                                      bool is_memory) {
  uint32_t flags = 0;
  if (inst->NumInOperands() > in_operand) {
    inst->GetSingleWordInOperand(in_operand);
  }
  if (is_coherent) {
    if (is_memory) {
      flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
      if (visible) {
        flags |= SpvMemoryAccessMakePointerVisibleKHRMask;
      } else {
        flags |= SpvMemoryAccessMakePointerAvailableKHRMask;
      }
    } else {
      flags |= SpvImageOperandsNonPrivateTexelKHRMask;
      if (visible) {
        flags |= SpvImageOperandsMakeTexelVisibleKHRMask;
      } else {
        flags |= SpvImageOperandsMakeTexelAvailableKHRMask;
      }
    }
  }

  if (is_volatile) {
    if (is_memory) {
      flags |= SpvMemoryAccessVolatileMask;
    } else {
      flags |= SpvImageOperandsVolatileTexelKHRMask;
    }
  }

  if (inst->NumInOperands() > in_operand) {
    inst->SetInOperand(in_operand, {flags});
  } else if (is_memory) {
    inst->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS, {flags}});
  } else {
    inst->AddOperand({SPV_OPERAND_TYPE_OPTIONAL_IMAGE, {flags}});
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
