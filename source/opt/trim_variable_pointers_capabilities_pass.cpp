// Copyright (c) 2026 Google Inc.
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

#include "source/opt/trim_variable_pointers_capabilities_pass.h"

#include <optional>
#include <stack>
#include <unordered_set>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

namespace {

constexpr uint32_t kOpTypePointerStorageClassIndex = 0;
constexpr uint32_t kTypePointerTypeIdInIndex = 1;

struct RequiredVariablePointerCapabilities {
  bool variable_pointers = false;
  bool variable_pointers_storage_buffer = false;

  void Add(spv::Capability capability) {
    switch (capability) {
      case spv::Capability::VariablePointers:
        variable_pointers = true;
        break;
      case spv::Capability::VariablePointersStorageBuffer:
        variable_pointers_storage_buffer = true;
        break;
      default:
        break;
    }
  }
};

template <class UnaryPredicate>
void DFSWhile(const Instruction* instruction, UnaryPredicate condition) {
  std::stack<uint32_t> instructions_to_visit;
  std::unordered_set<uint32_t> visited_instructions;
  instructions_to_visit.push(instruction->result_id());
  const auto* def_use_mgr = instruction->context()->get_def_use_mgr();

  while (!instructions_to_visit.empty()) {
    const Instruction* item = def_use_mgr->GetDef(instructions_to_visit.top());
    instructions_to_visit.pop();

    if (item == nullptr) {
      continue;
    }

    if (visited_instructions.count(item->result_id()) != 0) {
      continue;
    }
    visited_instructions.insert(item->result_id());

    if (!condition(item)) {
      continue;
    }

    if (item->opcode() == spv::Op::OpTypePointer) {
      instructions_to_visit.push(
          item->GetSingleWordInOperand(kTypePointerTypeIdInIndex));
      continue;
    }

    if (item->opcode() == spv::Op::OpTypeMatrix ||
        item->opcode() == spv::Op::OpTypeVector ||
        item->opcode() == spv::Op::OpTypeArray ||
        item->opcode() == spv::Op::OpTypeRuntimeArray) {
      instructions_to_visit.push(item->GetSingleWordInOperand(0));
      continue;
    }

    if (item->opcode() == spv::Op::OpTypeStruct) {
      item->ForEachInOperand([&instructions_to_visit](const uint32_t* op_id) {
        instructions_to_visit.push(*op_id);
      });
    }
  }
}

template <class UnaryPredicate>
bool AnyTypeOf(const Instruction* instruction, UnaryPredicate predicate) {
  if (instruction == nullptr || !IsTypeInst(instruction->opcode())) {
    return false;
  }

  bool found_one = false;
  DFSWhile(instruction, [&found_one, predicate](const Instruction* node) {
    if (found_one || predicate(node)) {
      found_one = true;
      return false;
    }

    return true;
  });
  return found_one;
}

std::optional<spv::Capability> GetVariablePointerCapability(
    spv::StorageClass storage_class) {
  switch (storage_class) {
    case spv::StorageClass::StorageBuffer:
      return spv::Capability::VariablePointersStorageBuffer;
    case spv::StorageClass::Workgroup:
      return spv::Capability::VariablePointers;
    default:
      return std::nullopt;
  }
}

std::optional<spv::StorageClass> GetLogicalPointerStorageClass(
    const Instruction* type_instruction) {
  if (type_instruction == nullptr) {
    return std::nullopt;
  }

  if (type_instruction->opcode() != spv::Op::OpTypePointer &&
      type_instruction->opcode() != spv::Op::OpTypeUntypedPointerKHR) {
    return std::nullopt;
  }

  const auto storage_class = static_cast<spv::StorageClass>(
      type_instruction->GetSingleWordInOperand(kOpTypePointerStorageClassIndex));
  if (storage_class == spv::StorageClass::PhysicalStorageBuffer) {
    return std::nullopt;
  }

  return storage_class;
}

std::optional<spv::StorageClass> GetLogicalPointerResultStorageClass(
    const Instruction* instruction) {
  if (instruction == nullptr || instruction->type_id() == 0) {
    return std::nullopt;
  }

  return GetLogicalPointerStorageClass(
      instruction->context()->get_def_use_mgr()->GetDef(instruction->type_id()));
}

void AddCapabilityForStorageClass(
    std::optional<spv::StorageClass> storage_class,
    RequiredVariablePointerCapabilities* required_capabilities) {
  if (!storage_class) {
    return;
  }

  if (const auto capability = GetVariablePointerCapability(*storage_class)) {
    required_capabilities->Add(*capability);
  }
}

void AddVariablePointerCapabilityForResult(
    const Instruction* instruction,
    RequiredVariablePointerCapabilities* required_capabilities) {
  AddCapabilityForStorageClass(
      GetLogicalPointerResultStorageClass(instruction), required_capabilities);
}

void AddVariablePointerCapabilitiesFromPointerOperands(
    const Instruction* instruction,
    RequiredVariablePointerCapabilities* required_capabilities) {
  instruction->ForEachInId([instruction, required_capabilities](const uint32_t* id) {
    const auto* operand_instruction =
        instruction->context()->get_def_use_mgr()->GetDef(*id);
    AddCapabilityForStorageClass(
        GetLogicalPointerResultStorageClass(operand_instruction),
        required_capabilities);
  });
}

void AddVariablePointerCapabilitiesFromAllocatedType(
    const Instruction* instruction,
    RequiredVariablePointerCapabilities* required_capabilities) {
  if (instruction->type_id() == 0) {
    return;
  }

  const auto* variable_type =
      instruction->context()->get_def_use_mgr()->GetDef(instruction->type_id());
  if (variable_type == nullptr ||
      (variable_type->opcode() != spv::Op::OpTypePointer &&
       variable_type->opcode() != spv::Op::OpTypeUntypedPointerKHR)) {
    return;
  }

  if (variable_type->opcode() == spv::Op::OpTypeUntypedPointerKHR) {
    AddCapabilityForStorageClass(GetLogicalPointerStorageClass(variable_type),
                                 required_capabilities);
    return;
  }

  const auto* pointee_type =
      instruction->context()->get_def_use_mgr()->GetDef(
          variable_type->GetSingleWordInOperand(kTypePointerTypeIdInIndex));
  if (pointee_type == nullptr) {
    return;
  }

  if (AnyTypeOf(pointee_type, [](const Instruction* type_instruction) {
        return GetLogicalPointerStorageClass(type_instruction) ==
               spv::StorageClass::StorageBuffer;
      })) {
    required_capabilities->Add(
        spv::Capability::VariablePointersStorageBuffer);
  }

  if (AnyTypeOf(pointee_type, [](const Instruction* type_instruction) {
        return GetLogicalPointerStorageClass(type_instruction) ==
               spv::StorageClass::Workgroup;
      })) {
    required_capabilities->Add(spv::Capability::VariablePointers);
  }
}

void AddVariablePointerCapabilityRequirements(
    const Instruction* instruction,
    RequiredVariablePointerCapabilities* required_capabilities) {
  switch (instruction->opcode()) {
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpLoad:
    case spv::Op::OpConstantNull:
    case spv::Op::OpFunction:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpUntypedPtrAccessChainKHR:
    case spv::Op::OpUntypedInBoundsPtrAccessChainKHR:
      AddVariablePointerCapabilityForResult(instruction, required_capabilities);
      break;
    default:
      break;
  }

  switch (instruction->opcode()) {
    case spv::Op::OpReturnValue:
    case spv::Op::OpStore:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
    case spv::Op::OpPtrDiff:
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpVariable:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpUntypedPtrAccessChainKHR:
    case spv::Op::OpUntypedInBoundsPtrAccessChainKHR:
      AddVariablePointerCapabilitiesFromPointerOperands(
          instruction, required_capabilities);
      break;
    default:
      break;
  }

  switch (instruction->opcode()) {
    case spv::Op::OpVariable:
    case spv::Op::OpUntypedVariableKHR:
      AddVariablePointerCapabilitiesFromAllocatedType(
          instruction, required_capabilities);
      break;
    default:
      break;
  }
}

bool CanRemoveVariablePointers(
    const RequiredVariablePointerCapabilities& required_capabilities,
    bool has_explicit_variable_pointers,
    bool has_explicit_variable_pointers_storage_buffer) {
  return has_explicit_variable_pointers &&
         !required_capabilities.variable_pointers &&
         (!required_capabilities.variable_pointers_storage_buffer ||
          has_explicit_variable_pointers_storage_buffer);
}

bool CanRemoveVariablePointersStorageBuffer(
    const RequiredVariablePointerCapabilities& required_capabilities,
    bool has_explicit_variable_pointers,
    bool has_explicit_variable_pointers_storage_buffer) {
  return has_explicit_variable_pointers_storage_buffer &&
         !required_capabilities.variable_pointers_storage_buffer &&
         (!required_capabilities.variable_pointers ||
          has_explicit_variable_pointers);
}

}  // namespace

Pass::Status TrimVariablePointersCapabilitiesPass::Process() {
  const bool has_explicit_variable_pointers = get_module()->HasExplicitCapability(
      static_cast<uint32_t>(spv::Capability::VariablePointers));
  const bool has_explicit_variable_pointers_storage_buffer =
      get_module()->HasExplicitCapability(
          static_cast<uint32_t>(
              spv::Capability::VariablePointersStorageBuffer));

  if (!has_explicit_variable_pointers &&
      !has_explicit_variable_pointers_storage_buffer) {
    return Status::SuccessWithoutChange;
  }

  RequiredVariablePointerCapabilities required_capabilities;
  get_module()->ForEachInst(
      [&required_capabilities](Instruction* instruction) {
        AddVariablePointerCapabilityRequirements(instruction,
                                                 &required_capabilities);
      },
      true);

  bool modified = false;
  if (CanRemoveVariablePointers(required_capabilities,
                                has_explicit_variable_pointers,
                                has_explicit_variable_pointers_storage_buffer)) {
    context()->RemoveCapability(spv::Capability::VariablePointers);
    modified = true;
  }

  if (CanRemoveVariablePointersStorageBuffer(
          required_capabilities, has_explicit_variable_pointers,
          has_explicit_variable_pointers_storage_buffer)) {
    context()->RemoveCapability(spv::Capability::VariablePointersStorageBuffer);
    modified = true;
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
