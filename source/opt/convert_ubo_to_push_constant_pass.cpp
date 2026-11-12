// Copyright (c) 2026 Diligent Graphics LLC
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

#include "source/opt/convert_ubo_to_push_constant_pass.h"

#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include "source/opt/decoration_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/type_manager.h"
#include "source/spirv_target_env.h"

namespace spvtools {
namespace opt {

bool ConvertUBOToPushConstantPass::Fail(const std::string& message) {
  if (consumer()) {
    const std::string diagnostic = std::string(name()) + ": " + message;
    consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, diagnostic.c_str());
  }
  return false;
}

bool ConvertUBOToPushConstantPass::HasDecorationValue(
    uint32_t id, spv::Decoration decoration, uint32_t value) {
  bool found = false;
  get_decoration_mgr()->ForEachDecoration(
      id, static_cast<uint32_t>(decoration), [&](const Instruction& inst) {
        found |= inst.GetSingleWordInOperand(2) == value;
      });
  return found;
}

Instruction* ConvertUBOToPushConstantPass::FindTargetUniformBuffer() {
  std::unordered_set<uint32_t> named_ids;
  if (!select_by_binding_) {
    if (name_.empty()) {
      Fail("the variable or block name must not be empty");
      return nullptr;
    }
    for (auto& inst : context()->module()->debugs2()) {
      if (inst.opcode() == spv::Op::OpName &&
          inst.GetInOperand(1).AsString() == name_) {
        named_ids.insert(inst.GetSingleWordInOperand(0));
      }
    }
  }

  Instruction* target = nullptr;
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() != spv::Op::OpVariable ||
        inst.GetSingleWordInOperand(0) !=
            static_cast<uint32_t>(spv::StorageClass::Uniform)) {
      continue;
    }
    auto* pointer = get_def_use_mgr()->GetDef(inst.type_id());
    assert(pointer->opcode() == spv::Op::OpTypePointer &&
           "OpVariable must have a pointer type");
    const uint32_t type_id = pointer->GetSingleWordInOperand(1);
    if (get_def_use_mgr()->GetDef(type_id)->opcode() != spv::Op::OpTypeStruct ||
        !get_decoration_mgr()->HasDecoration(type_id, spv::Decoration::Block)) {
      continue;
    }
    const bool matches =
        select_by_binding_
            ? HasDecorationValue(inst.result_id(),
                                 spv::Decoration::DescriptorSet,
                                 descriptor_set_) &&
                  HasDecorationValue(inst.result_id(), spv::Decoration::Binding,
                                     binding_)
            : named_ids.count(inst.result_id()) || named_ids.count(type_id);
    if (!matches) {
      continue;
    }
    if (target) {
      Fail("selection matches more than one uniform buffer");
      return nullptr;
    }
    target = &inst;
  }
  if (!target) {
    Fail("no matching uniform buffer with a Block-decorated struct");
  }
  return target;
}

bool ConvertUBOToPushConstantPass::CheckPushConstantType(uint32_t type_id) {
  std::vector<uint32_t> pending{type_id};
  std::unordered_set<uint32_t> visited;
  while (!pending.empty()) {
    const uint32_t id = pending.back();
    pending.pop_back();
    if (!visited.insert(id).second) {
      continue;
    }
    const auto* type = get_def_use_mgr()->GetDef(id);
    switch (type->opcode()) {
      case spv::Op::OpTypePointer:
        // Do not follow physical pointers stored in the block: the pointed-to
        // memory remains in its original storage class.
        break;
      case spv::Op::OpTypeRuntimeArray:
        return Fail("push constants cannot contain a runtime array");
      case spv::Op::OpTypeStruct:
        for (uint32_t i = 0; i < type->NumInOperands(); ++i) {
          pending.push_back(type->GetSingleWordInOperand(i));
        }
        break;
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeVectorIdEXT:
      case spv::Op::OpTypeMatrix:
        pending.push_back(type->GetSingleWordInOperand(0));
        break;
      case spv::Op::OpTypeInt:
      case spv::Op::OpTypeFloat: {
        const uint32_t width = type->GetSingleWordInOperand(0);
        auto* features = context()->get_feature_mgr();
        // The 8-bit storage capabilities apply to integers, not Float8EXT.
        if (type->opcode() == spv::Op::OpTypeInt && width == 8 &&
            !features->HasCapability(spv::Capability::Int8) &&
            !features->HasCapability(spv::Capability::StoragePushConstant8)) {
          return Fail("8-bit members require StoragePushConstant8 or Int8");
        }
        if (width == 16 &&
            !features->HasCapability(spv::Capability::StoragePushConstant16) &&
            !features->HasCapability(type->opcode() == spv::Op::OpTypeInt
                                         ? spv::Capability::Int16
                                         : spv::Capability::Float16)) {
          return Fail(
              "16-bit members require StoragePushConstant16 or the "
              "corresponding arithmetic capability");
        }
        break;
      }
      default:
        break;
    }
  }
  return true;
}

bool ConvertUBOToPushConstantPass::CheckMemoryAccess(const Instruction& inst) {
  // Operand tags distinguish masks from their alignment/scope operands,
  // including copies with separate source/target masks.
  for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
    const auto& operand = inst.GetInOperand(i);
    if ((operand.type == SPV_OPERAND_TYPE_MEMORY_ACCESS ||
         operand.type == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS) &&
        (operand.words[0] &
         static_cast<uint32_t>(spv::MemoryAccessMask::NonPrivatePointer))) {
      // Current validation checks both copy-pointer storage classes for each
      // mask, so even a target-only flag is unsupported.
      return Fail(
          "NonPrivatePointer memory access is not supported "
          "for push constants");
    }
  }
  return true;
}

bool ConvertUBOToPushConstantPass::CollectAndValidateBufferPointers(
    Instruction* variable, std::vector<Instruction*>* pointers) {
  pointers->push_back(variable);
  std::unordered_set<uint32_t> visited{variable->result_id()};
  for (size_t i = 0; i < pointers->size(); ++i) {
    const bool supported = get_def_use_mgr()->WhileEachUse(
        (*pointers)[i], [&](Instruction* user, uint32_t operand_index) {
          switch (user->opcode()) {
            case spv::Op::OpName:
            case spv::Op::OpDecorate:
            case spv::Op::OpDecorateId:
            case spv::Op::OpDecorateStringGOOGLE:
            case spv::Op::OpGroupDecorate:
            case spv::Op::OpEntryPoint:
              return true;
            case spv::Op::OpAccessChain:
            case spv::Op::OpInBoundsAccessChain:
            case spv::Op::OpCopyObject:
              // These instructions have a type, a result ID, and then the base
              // pointer. Following only the base preserves unrelated pointers.
              if (operand_index != 2) {
                break;
              }
              if (visited.insert(user->result_id()).second) {
                pointers->push_back(user);
              }
              return true;
            case spv::Op::OpLoad:
              // Loading a physical pointer reads a value from the block; it
              // does not make the loaded pointer point into push constants.
              if (operand_index != 2) {
                break;
              }
              return CheckMemoryAccess(*user);
            case spv::Op::OpCopyMemory:
              if (operand_index != 1) {
                break;
              }
              return CheckMemoryAccess(*user);
            case spv::Op::OpFunctionCall:
              return Fail(
                  "pointer arguments must be inlined before conversion");
            default:
              // Debug records reference the variable without deriving a
              // pointer.
              if (user->IsCommonDebugInstr() ||
                  user->IsNonSemanticInstruction()) {
                return true;
              }
              break;
          }
          return Fail(std::string("unsupported pointer use: Op") +
                      spvOpcodeString(static_cast<uint32_t>(user->opcode())));
        });
    if (!supported) {
      return false;
    }
  }
  return true;
}

bool ConvertUBOToPushConstantPass::CheckTargetVariable(
    const Instruction& target) {
  // Universal SPIR-V allows Uniform initializers, but PushConstant does not.
  if (target.NumInOperands() > 1) {
    Fail("uniform buffers with initializers cannot be converted");
    return false;
  }
  if (get_decoration_mgr()->HasDecoration(target.result_id(),
                                          spv::Decoration::NonReadable)) {
    Fail("NonReadable uniform buffers cannot be converted");
    return false;
  }

  // The pass does not merge blocks or change offsets. Reject an existing push
  // constant block, rather than introducing conflicting interfaces.
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() == spv::Op::OpVariable &&
        inst.GetSingleWordInOperand(0) ==
            static_cast<uint32_t>(spv::StorageClass::PushConstant)) {
      Fail("the module already contains a push constant variable");
      return false;
    }
  }
  return true;
}

bool ConvertUBOToPushConstantPass::RewriteBufferPointers(
    Instruction* target, const std::vector<Instruction*>& pointers) {
  // Create separate pointer types so other UBOs sharing the same block or
  // member types keep their Uniform storage class and exact type identity.
  std::unordered_map<uint32_t, uint32_t> pointer_types;
  for (auto* inst : pointers) {
    const uint32_t old_type = inst->type_id();
    if (pointer_types.count(old_type)) {
      continue;
    }
    const uint32_t pointee =
        get_def_use_mgr()->GetDef(old_type)->GetSingleWordInOperand(1);
    const uint32_t new_type = context()->get_type_mgr()->FindPointerToType(
        pointee, spv::StorageClass::PushConstant);
    if (!new_type) {
      return false;
    }
    pointer_types.emplace(old_type, new_type);
  }

  // FindPointerToType appends new types. Move the root pointer type before the
  // global variable only if necessary; existing earlier types must stay put.
  auto* root_type =
      get_def_use_mgr()->GetDef(pointer_types.at(target->type_id()));
  for (auto& inst : context()->types_values()) {
    if (&inst == root_type) {
      break;
    }
    if (&inst == target) {
      root_type->RemoveFromList();
      root_type->InsertBefore(target);
      break;
    }
  }

  target->SetInOperand(
      0, {static_cast<uint32_t>(spv::StorageClass::PushConstant)});
  for (auto* inst : pointers) {
    inst->SetResultType(pointer_types.at(inst->type_id()));
    context()->UpdateDefUse(inst);
  }
  return true;
}

void ConvertUBOToPushConstantPass::RemoveBufferDecorations(
    uint32_t variable_id) {
  get_decoration_mgr()->RemoveDecorationsFrom(
      variable_id, [](const Instruction& inst) {
        if (inst.opcode() != spv::Op::OpDecorate) {
          return false;
        }
        const auto decoration =
            static_cast<spv::Decoration>(inst.GetSingleWordInOperand(1));
        // Push constants are inherently read-only; NonWritable is redundant
        // and is not valid on a PushConstant variable.
        return decoration == spv::Decoration::Binding ||
               decoration == spv::Decoration::DescriptorSet ||
               decoration == spv::Decoration::NonWritable;
      });
}

Pass::Status ConvertUBOToPushConstantPass::Process() {
  if (spvIsOpenGLEnv(context()->grammar().target_env())) {
    Fail("push constants are not supported in OpenGL");
    return Status::Failure;
  }

  Instruction* target = FindTargetUniformBuffer();
  if (!target || !CheckTargetVariable(*target)) {
    return Status::Failure;
  }

  const uint32_t block_type =
      get_def_use_mgr()->GetDef(target->type_id())->GetSingleWordInOperand(1);
  std::vector<Instruction*> pointers;
  if (!CheckPushConstantType(block_type) ||
      !CollectAndValidateBufferPointers(target, &pointers)) {
    return Status::Failure;
  }

  if (!RewriteBufferPointers(target, pointers)) {
    return Status::Failure;
  }
  RemoveBufferDecorations(target->result_id());
  return Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools
