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

#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kPointerTypePointeeIndex = 1;
constexpr uint32_t kUntypedVariableStorageClassIndex = 0;
constexpr uint32_t kUntypedVariableDataTypeIndex = 1;
constexpr uint32_t kMemberDecorateMemberIndex = 1;
constexpr uint32_t kMemberDecorateDecorationIndex = 2;
}  // namespace

Pass::Status NonWritablePropagationPass::Process() {
  // Collect first, mutate after: several variables can share one struct
  // type, and stripping that type's member decorations while other
  // variables are still being examined would make the result depend on
  // iteration order.
  std::vector<std::pair<Instruction*, Instruction*>> vars_to_decorate;

  for (Instruction& var : get_module()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable &&
        var.opcode() != spv::Op::OpUntypedVariableKHR) {
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

    vars_to_decorate.push_back({&var, struct_type});
  }

  std::unordered_set<uint32_t> structs_to_strip;
  for (const auto& [var, struct_type] : vars_to_decorate) {
    context()->get_decoration_mgr()->AddDecoration(
        var->result_id(), uint32_t(spv::Decoration::NonWritable));
    structs_to_strip.insert(struct_type->result_id());
  }

  // The propagated decoration now carries the information, so the
  // per-member spelling is redundant on these structs and is removed.
  for (uint32_t struct_id : structs_to_strip) {
    context()->get_decoration_mgr()->RemoveDecorationsFrom(
        struct_id, [](const Instruction& inst) {
          return inst.opcode() == spv::Op::OpMemberDecorate &&
                 inst.GetSingleWordInOperand(kMemberDecorateDecorationIndex) ==
                     uint32_t(spv::Decoration::NonWritable);
        });
  }

  return vars_to_decorate.empty() ? Status::SuccessWithoutChange
                                  : Status::SuccessWithChange;
}

Instruction* NonWritablePropagationPass::GetBufferStructType(
    const Instruction& var) const {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  Instruction* base_type = nullptr;
  bool need_block_check = false;
  spv::StorageClass storage_class = spv::StorageClass::Max;

  if (var.opcode() == spv::Op::OpVariable) {
    Instruction* ptr_type = def_use_mgr->GetDef(var.type_id());
    if (ptr_type == nullptr) {
      return nullptr;
    }

    // The validator only accepts NonWritable on a variable pointing at a
    // uniform block, storage buffer, storage image or tensor -- see
    // CheckNonReadableWritableDecorations in validate_decorations.cpp.  Of
    // those, only the buffers have members to propagate from.  Together
    // these two cover Uniform+Block, Uniform+BufferBlock and
    // StorageBuffer+Block.
    if (!ptr_type->IsVulkanStorageBuffer() &&
        !ptr_type->IsVulkanUniformBuffer()) {
      return nullptr;
    }

    base_type = def_use_mgr->GetDef(
        ptr_type->GetSingleWordInOperand(kPointerTypePointeeIndex));
  } else {
    // OpUntypedVariableKHR: the pointer type carries no pointee, so the
    // struct comes from the optional data-type operand.  Without that
    // operand there is nothing to inspect.
    if (var.NumInOperands() <= kUntypedVariableDataTypeIndex) {
      return nullptr;
    }

    storage_class = spv::StorageClass(
        var.GetSingleWordInOperand(kUntypedVariableStorageClassIndex));
    if (storage_class != spv::StorageClass::Uniform &&
        storage_class != spv::StorageClass::StorageBuffer) {
      return nullptr;
    }

    base_type = def_use_mgr->GetDef(
        var.GetSingleWordInOperand(kUntypedVariableDataTypeIndex));
    need_block_check = true;
  }

  if (base_type == nullptr) {
    return nullptr;
  }

  // Unpack the optional layer of arraying (a descriptor array of buffers).
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

  // For the typed path the two helpers above already proved block-ness.
  // The untyped path checks it here, accepting the same set:
  // StorageBuffer+Block, Uniform+Block and Uniform+BufferBlock.
  if (need_block_check) {
    analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
    const bool has_block =
        deco_mgr->HasDecoration(base_type->result_id(), spv::Decoration::Block);
    const bool has_buffer_block = deco_mgr->HasDecoration(
        base_type->result_id(), spv::Decoration::BufferBlock);
    if (storage_class == spv::StorageClass::StorageBuffer && !has_block) {
      return nullptr;
    }
    if (storage_class == spv::StorageClass::Uniform && !has_block &&
        !has_buffer_block) {
      return nullptr;
    }
  }

  return base_type;
}

bool NonWritablePropagationPass::AllMembersNonWritable(
    const Instruction& struct_type) const {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  // For OpTypeStruct the in-operands are exactly the member types.
  const uint32_t member_count = struct_type.NumInOperands();

  // A struct with no members is vacuously "all members NonWritable".
  // Decorating on that basis would be asserting read-only with no evidence
  // for it, so leave these alone.
  if (member_count == 0) {
    return false;
  }

  // Members that are themselves structs (directly or through an array) are
  // exempt from the check: DXC does not decorate such members even in a
  // read-only buffer, so requiring them would make the pass a no-op on
  // exactly the modules it exists to fix up.
  std::vector<bool> considered(member_count, true);
  uint32_t considered_count = 0;
  for (uint32_t i = 0; i < member_count; ++i) {
    Instruction* member_type =
        def_use_mgr->GetDef(struct_type.GetSingleWordInOperand(i));
    while (member_type != nullptr &&
           (member_type->opcode() == spv::Op::OpTypeArray ||
            member_type->opcode() == spv::Op::OpTypeRuntimeArray)) {
      member_type = def_use_mgr->GetDef(member_type->GetSingleWordInOperand(0));
    }
    if (member_type != nullptr &&
        member_type->opcode() == spv::Op::OpTypeStruct) {
      considered[i] = false;
    } else {
      ++considered_count;
    }
  }

  // If every member is exempt there is no evidence in either direction --
  // a writable buffer of structs looks exactly the same.  Leave it alone.
  if (considered_count == 0) {
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

  for (uint32_t i = 0; i < member_count; ++i) {
    if (considered[i] && !is_non_writable[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace opt
}  // namespace spvtools
