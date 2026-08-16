// Copyright (c) 2026 The Khronos Group Inc.
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

#include "source/opt/nonwritable_propagation_pass.h"

#include <vector>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kPointerTypePointeeIndex = 1;
constexpr uint32_t kMemberDecorateMemberIndex = 1;
}  // namespace

Pass::Status NonWritablePropagationPass::Process() {
  bool changed = false;

  for (Instruction& var : get_module()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable) {
      continue;
    }

    // Already decorated, so there is nothing to propagate.  Note this asks
    // about the variable's id, not the struct type's, so the per-member
    // decorations cannot make this true by themselves.
    if (context()->get_decoration_mgr()->HasDecoration(
            var.result_id(), spv::Decoration::NonWritable)) {
      continue;
    }

    Instruction* struct_type = GetBufferStructType(var);
    if (struct_type == nullptr) {
      continue;
    }

    if (!AllMembersNonWritable(*struct_type)) {
      continue;
    }

    context()->get_decoration_mgr()->AddDecoration(
        var.result_id(), uint32_t(spv::Decoration::NonWritable));
    changed = true;
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Instruction* NonWritablePropagationPass::GetBufferStructType(
    const Instruction& var) const {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  Instruction* ptr_type = def_use_mgr->GetDef(var.type_id());
  if (ptr_type == nullptr) {
    return nullptr;
  }

  // The validator only accepts NonWritable on a variable pointing at a
  // uniform block, storage buffer, storage image or tensor -- see
  // CheckNonReadableWritableDecorations in validate_decorations.cpp.  Of
  // those, only the buffers have members to propagate from.  Together these
  // two cover Uniform+Block, Uniform+BufferBlock and StorageBuffer+Block.
  if (!ptr_type->IsVulkanStorageBuffer() &&
      !ptr_type->IsVulkanUniformBuffer()) {
    return nullptr;
  }

  Instruction* base_type = def_use_mgr->GetDef(
      ptr_type->GetSingleWordInOperand(kPointerTypePointeeIndex));
  if (base_type == nullptr) {
    return nullptr;
  }

  // Unpack the optional layer of arraying, matching what the two helpers
  // above already did to decide this was a buffer at all.
  if (base_type->opcode() == spv::Op::OpTypeArray ||
      base_type->opcode() == spv::Op::OpTypeRuntimeArray) {
    base_type = def_use_mgr->GetDef(base_type->GetSingleWordInOperand(0));
    if (base_type == nullptr) {
      return nullptr;
    }
  }

  if (base_type->opcode() != spv::Op::OpTypeStruct) {
    return nullptr;
  }

  return base_type;
}

bool NonWritablePropagationPass::AllMembersNonWritable(
    const Instruction& struct_type) const {
  // For OpTypeStruct the in-operands are exactly the member types.
  const uint32_t member_count = struct_type.NumInOperands();

  // A struct with no members is vacuously "all members NonWritable".
  // Decorating on that basis would be asserting read-only with no evidence
  // for it, so leave these alone.
  if (member_count == 0) {
    return false;
  }

  std::vector<bool> is_non_writable(member_count, false);
  context()->get_decoration_mgr()->ForEachDecoration(
      struct_type.result_id(), uint32_t(spv::Decoration::NonWritable),
      [&is_non_writable, member_count](const Instruction& deco) {
        // ForEachDecoration also reports OpDecorate on the struct type
        // itself, which says nothing about the individual members.
        if (deco.opcode() != spv::Op::OpMemberDecorate) {
          return;
        }
        const uint32_t member =
            deco.GetSingleWordInOperand(kMemberDecorateMemberIndex);
        if (member < member_count) {
          is_non_writable[member] = true;
        }
      });

  for (bool member_is_non_writable : is_non_writable) {
    if (!member_is_non_writable) {
      return false;
    }
  }
  return true;
}

}  // namespace opt
}  // namespace spvtools
