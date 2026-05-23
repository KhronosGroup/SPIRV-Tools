// Copyright (c) 2026 Google LLC
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

#include "convert_to_untyped.h"

#include <deque>
#include <iostream>
#include <limits>
#include <unordered_set>

#include "source/operand.h"

namespace spvtools {
namespace opt {

Pass::Status ConvertToUntyped::Process() {
  if (HasUnsupportedFeatures()) {
    return Pass::Status::SuccessWithoutChange;
  }

  AddUntypedEnable();
  ConvertPointers();

  uint32_t max_id = std::max(context()->max_id_bound(), max_id_);
  context()->set_max_id_bound(max_id);
  get_module()->SetIdBound(max_id);

  return Pass::Status::SuccessWithChange;
}

bool ConvertToUntyped::HasUnsupportedFeatures() {
  // Only support converting shaders.
  const auto& caps = context()->get_feature_mgr()->GetCapabilities();
  if (!caps.contains(spv::Capability::Shader)) {
    return true;
  }

  // TODO: support the following capabilities
  if (caps.contains(spv::Capability::VariablePointersStorageBuffer)) {
    return true;
  }
  if (caps.contains(spv::Capability::WorkgroupMemoryExplicitLayoutKHR)) {
    return true;
  }

  // Unsupported
  CapabilitySet unsupported{
      spv::Capability::RawAccessChainsNV,
      spv::Capability::PredicatedIOINTEL,
      spv::Capability::GraphARM,
      spv::Capability::CooperativeMatrixReductionsNV,
      spv::Capability::CooperativeMatrixConversionsNV,
      spv::Capability::CooperativeMatrixPerElementOperationsNV,
      spv::Capability::CooperativeMatrixTensorAddressingNV,
      spv::Capability::CooperativeMatrixBlockLoadsNV,
      spv::Capability::CooperativeVectorNV,
      spv::Capability::CooperativeVectorTrainingNV};
  return caps.HasAnyOf(unsupported);
}

void ConvertToUntyped::AddUntypedEnable() {
  bool needs_ext = true;
  for (auto& ei : get_module()->extensions()) {
    const std::string name = ei.GetInOperand(0).AsString();
    if (name == "SPV_KHR_untyped_pointers") {
      needs_ext = false;
      break;
    }
  }
  if (needs_ext) {
    std::vector<uint32_t> words =
        spvtools::utils::MakeVector("SPV_KHR_untyped_pointers");
    context()->AddExtension(
        MakeUnique<Instruction>(context(), spv::Op::OpExtension, 0, 0,
                                std::initializer_list<Operand>{
                                    {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
  }

  if (!context()->get_feature_mgr()->HasCapability(
          spv::Capability::UntypedPointersKHR)) {
    context()->AddCapability(MakeUnique<Instruction>(
        context(), spv::Op::OpCapability, 0, 0,
        std::initializer_list<Operand>{
            {SPV_OPERAND_TYPE_CAPABILITY,
             {uint32_t(spv::Capability::UntypedPointersKHR)}}}));
  } else {
    // Untyped pointers capability is already declared, pre-populate base_ptrs_.
    // All possibly supported storage classes are listed for extensibility
    // purposes.
    std::vector<spv::StorageClass> storage_classes{
        spv::StorageClass::StorageBuffer,
        spv::StorageClass::Uniform,
        spv::StorageClass::Workgroup,
        spv::StorageClass::PushConstant,
        spv::StorageClass::PhysicalStorageBuffer,
        spv::StorageClass::UniformConstant};
    for (auto sc : storage_classes) {
      if (!SupportedStorageClass(sc)) {
        continue;
      }
      const analysis::Pointer ptr_type{nullptr, sc};
      auto registered = context()->get_type_mgr()->GetRegisteredType(&ptr_type);
      if (registered) {
        auto id = context()->get_type_mgr()->GetId(registered);
        if (id != 0) {
          base_ptrs_[uint32_t(sc)] = context()->get_def_use_mgr()->GetDef(id);
        }
      }
    }
  }
}

uint32_t ConvertToUntyped::NextId() {
  uint32_t id = context()->TakeNextUniqueId();
  max_id_ = std::max(id, max_id_);
  return id;
}

bool ConvertToUntyped::SupportedStorageClass(spv::StorageClass sc) {
  switch (sc) {
    case spv::StorageClass::StorageBuffer:
    case spv::StorageClass::Uniform:
    case spv::StorageClass::PushConstant:
      return true;
    default:
      return false;
  }
}

bool ConvertToUntyped::SupportedStorageClass(uint32_t sc) {
  return SupportedStorageClass(static_cast<spv::StorageClass>(sc));
}

bool ConvertToUntyped::ShouldConvert(const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpTypePointer:
      return SupportedStorageClass(inst->GetSingleWordInOperand(0));
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpVariable:
      return SupportedStorageClass(context()
                                       ->get_type_mgr()
                                       ->GetType(inst->type_id())
                                       ->AsPointer()
                                       ->storage_class());
    case spv::Op::OpCopyMemory: {
      const auto* dst =
          context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      const auto* dst_ptr =
          context()->get_type_mgr()->GetType(dst->type_id())->AsPointer();
      const auto* src =
          context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
      const auto* src_ptr =
          context()->get_type_mgr()->GetType(src->type_id())->AsPointer();
      return SupportedStorageClass(dst_ptr->storage_class()) ||
             SupportedStorageClass(src_ptr->storage_class());
    }
    case spv::Op::OpArrayLength: {
      const auto* obj =
          context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      const auto* ptr =
          context()->get_type_mgr()->GetType(obj->type_id())->AsPointer();
      return SupportedStorageClass(ptr->storage_class());
    }
    case spv::Op::OpCooperativeMatrixLoadKHR:
    case spv::Op::OpCooperativeMatrixStoreKHR: {
      const auto* ptr =
          context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      const auto* ptr_ty =
          context()->get_type_mgr()->GetType(ptr->type_id())->AsPointer();
      return SupportedStorageClass(ptr_ty->storage_class());
    }
    default:
      return false;
  }
}

std::pair<bool, uint32_t> ConvertToUntyped::MatrixStride(Instruction* inst) {
  std::deque<uint32_t> indices;
  bool keep_going = true;
  while (keep_going) {
    keep_going = false;
    switch (inst->opcode()) {
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
        for (uint32_t i = inst->NumInOperands() - 1; i > 0; i--) {
          if (auto* constant =
                  context()->get_constant_mgr()->FindDeclaredConstant(
                      inst->GetSingleWordInOperand(i))) {
            uint64_t ext = constant->GetZeroExtendedValue();
            if (ext > std::numeric_limits<uint32_t>::max()) {
              return std::make_pair(false, 0);
            }
            indices.push_front(static_cast<uint32_t>(ext));
          } else {
            // Assume we aren't at the start to accumulate a stride.
            indices.push_front(1);
          }
        }
        inst = context()->get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(0));
        keep_going = true;
        break;
      case spv::Op::OpCopyObject:
        inst = context()->get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(0));
        break;
      case spv::Op::OpVariable:
        break;
      default:
        // A variable pointer cannot point to a matrix or inside it so we can
        // stop searching at this point.
        return std::make_pair(false, 0);
    }
  }

  bool row_major = false;
  uint32_t mat_stride = 0;
  const auto* type = context()
                         ->get_type_mgr()
                         ->GetType(inst->type_id())
                         ->AsPointer()
                         ->pointee_type();
  for (uint32_t i = 0; i < indices.size(); i++) {
    switch (type->kind()) {
      case analysis::Type::kStruct: {
        auto where = type->AsStruct()->element_decorations().find(indices[i]);
        if (where != type->AsStruct()->element_decorations().end()) {
          const auto& decs = where->second;
          for (auto dec : decs) {
            if (dec[0] == uint32_t(spv::Decoration::MatrixStride)) {
              mat_stride = dec[1];
            }
            if (dec[0] == uint32_t(spv::Decoration::RowMajor)) {
              row_major = true;
            }
          }
        }
        type = type->AsStruct()->element_types()[indices[i]];
        break;
      }
      case analysis::Type::kArray:
        type = type->AsArray()->element_type();
        break;
      case analysis::Type::kRuntimeArray:
        type = type->AsRuntimeArray();
      case analysis::Type::kMatrix:
        return std::make_pair(row_major, mat_stride);
      // No other types could hold a matrix
      default:
        return std::make_pair(false, 0);
    }
  }

  if (type->AsMatrix()) {
    return std::make_pair(row_major, mat_stride);
  }
  return std::make_pair(false, 0);
}

Instruction* ConvertToUntyped::ConvertPointer(Instruction* inst) {
  Instruction* replacement = nullptr;
  auto sc = inst->GetSingleWordInOperand(0);
  auto decs = context()->get_decoration_mgr()->GetDecorationsFor(
      inst->result_id(), false);
  // De-duplicate undecorated pointers.
  if (!decs.empty() || !base_ptrs_.count(sc)) {
    uint32_t id = NextId();
    auto unique = MakeUnique<Instruction>(
        context(), spv::Op::OpTypeUntypedPointerKHR, 0, id,
        std::initializer_list<Operand>{{SPV_OPERAND_TYPE_STORAGE_CLASS, {sc}}});
    replacement = &*unique;
    context()->AddType(std::move(unique));
    if (!decs.empty()) {
      context()->get_decoration_mgr()->CloneDecorations(inst->result_id(), id);
    } else {
      base_ptrs_[sc] = replacement;
    }
  } else {
    replacement = base_ptrs_[sc];
  }
  return replacement;
}

Instruction* ConvertToUntyped::ConvertVariable(Instruction* inst) {
  auto ptr_ty =
      context()->get_type_mgr()->GetType(inst->type_id())->AsPointer();
  auto data =
      context()->get_type_mgr()->GetTypeInstruction(ptr_ty->pointee_type());
  uint32_t id = NextId();
  // No initializers are permitted on any supported storage class.
  auto unique = MakeUnique<Instruction>(
      context(), spv::Op::OpUntypedVariableKHR, remapped_ids_[inst->type_id()],
      id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS,
           {static_cast<uint32_t>(ptr_ty->storage_class())}},
          {SPV_OPERAND_TYPE_ID, {data}}});
  auto replacement = &*unique;
  // All supported storage classes are module scope variables.
  context()->AddGlobalValue(std::move(unique));
  context()->get_decoration_mgr()->CloneDecorations(inst->result_id(), id);
  return replacement;
}

Instruction* ConvertToUntyped::ConvertAccessChain(Instruction* inst) {
  const bool inbounds = inst->opcode() == spv::Op::OpInBoundsAccessChain;
  auto base =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  auto ptr_ty =
      context()->get_type_mgr()->GetType(base->type_id())->AsPointer();
  auto data =
      context()->get_type_mgr()->GetTypeInstruction(ptr_ty->pointee_type());
  // Add the base type, but then reuse all the other operands.
  std::vector<Operand> operands;
  operands.reserve(inst->NumInOperands() + 1);
  operands.push_back({SPV_OPERAND_TYPE_ID, {data}});
  for (uint32_t i = 0; i < inst->NumInOperands(); i++) {
    operands.push_back(inst->GetInOperand(i));
  }
  uint32_t id = NextId();
  auto unique = MakeUnique<Instruction>(
      context(),
      (inbounds ? spv::Op::OpUntypedInBoundsAccessChainKHR
                : spv::Op::OpUntypedAccessChainKHR),
      remapped_ids_[inst->type_id()], id, operands);
  auto replacement = inst->InsertBefore(std::move(unique));
  context()->get_decoration_mgr()->CloneDecorations(inst->result_id(), id);
  return replacement;
}

Instruction* ConvertToUntyped::ConvertArrayLength(Instruction* inst) {
  auto structure =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  auto ptr_ty =
      context()->get_type_mgr()->GetType(structure->type_id())->AsPointer();
  auto data =
      context()->get_type_mgr()->GetTypeInstruction(ptr_ty->pointee_type());
  // Add the structure type, but then reuse all the other operands.
  std::vector<Operand> operands;
  operands.reserve(inst->NumInOperands() + 1);
  operands.push_back({SPV_OPERAND_TYPE_ID, {data}});
  for (uint32_t i = 0; i < inst->NumInOperands(); i++) {
    operands.push_back(inst->GetInOperand(i));
  }
  uint32_t id = NextId();
  auto unique =
      MakeUnique<Instruction>(context(), spv::Op::OpUntypedArrayLengthKHR,
                              inst->type_id(), id, operands);
  auto replacement = inst->InsertBefore(std::move(unique));
  return replacement;
}

std::vector<Operand> ConvertToUntyped::StoreOperands(
    const std::vector<Operand>& operands, uint32_t align) {
  // Add placeholders for the pointer and object.
  std::vector<Operand> st_operands{{SPV_OPERAND_TYPE_ID, {0}},
                                   {SPV_OPERAND_TYPE_ID, {0}}};
  if (operands.empty() && align != 0) {
    // Add alignment
    st_operands.push_back({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                           {uint32_t(spv::MemoryAccessMask::Aligned)}});
    st_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {align}});
  } else if (!operands.empty()) {
    // Add/Remove alignment
    // Copy MakePointerAvailable scope
    // Drop MakePointerVisible (and ignore scope)
    uint32_t idx = 0;
    st_operands.push_back(operands[idx++]);
    if ((operands[0].words[0] & uint32_t(spv::MemoryAccessMask::Aligned)) !=
        0) {
      idx++;
      if (align == 0) {
        st_operands[2].words[0] ^= uint32_t(spv::MemoryAccessMask::Aligned);
      }
    }
    if (align != 0) {
      st_operands[2].words[0] |= uint32_t(spv::MemoryAccessMask::Aligned);
      st_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {align}});
    }

    if ((operands[0].words[0] &
         uint32_t(spv::MemoryAccessMask::MakePointerAvailable)) != 0) {
      st_operands.push_back(operands[idx]);
    }

    st_operands[2].words[0] &=
        ~uint32_t(spv::MemoryAccessMask::MakePointerVisible);
  }

  return st_operands;
}

std::vector<Operand> ConvertToUntyped::LoadOperands(
    const std::vector<Operand>& operands, uint32_t align) {
  // Add placeholder operand for pointer
  std::vector<Operand> ld_operands{{SPV_OPERAND_TYPE_ID, {0}}};
  if (operands.empty() && align != 0) {
    // Add alignment
    ld_operands.push_back({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                           {uint32_t(spv::MemoryAccessMask::Aligned)}});
    ld_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {align}});
  } else if (!operands.empty()) {
    // Add/Remove alignment
    // Drop MakePointerAvailable (and ignore scope)
    // Copy MakePointerVisible and scope
    uint32_t idx = 0;
    ld_operands.push_back(operands[idx++]);
    if ((operands[0].words[0] & uint32_t(spv::MemoryAccessMask::Aligned)) !=
        0) {
      idx++;
      if (align == 0) {
        ld_operands[1].words[0] ^= uint32_t(spv::MemoryAccessMask::Aligned);
      }
    }
    if (align != 0) {
      ld_operands[1].words[0] |= uint32_t(spv::MemoryAccessMask::Aligned);
      ld_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {align}});
    }

    if ((operands[0].words[0] &
         uint32_t(spv::MemoryAccessMask::MakePointerAvailable)) != 0) {
      idx++;
      ld_operands[1].words[0] ^=
          uint32_t(spv::MemoryAccessMask::MakePointerAvailable);
    }

    if ((operands[0].words[0] &
         uint32_t(spv::MemoryAccessMask::MakePointerVisible)) != 0) {
      ld_operands.push_back(operands[idx]);
    }
  }

  return ld_operands;
}

void ConvertToUntyped::ConvertCopyMemory(Instruction* inst) {
  auto dst =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  auto dst_type =
      context()->get_type_mgr()->GetType(dst->type_id())->AsPointer();
  auto dst_sc = dst_type->storage_class();
  auto dst_untyped = SupportedStorageClass(dst_sc);
  auto src =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
  auto src_type =
      context()->get_type_mgr()->GetType(src->type_id())->AsPointer();
  auto src_sc = src_type->storage_class();
  auto src_untyped = SupportedStorageClass(src_sc);

  // Get the memory access operands for src and dst.
  std::vector<Operand> dst_operands;
  std::vector<Operand> src_operands;
  if (inst->NumInOperands() > 2) {
    uint32_t i = 2;
    for (; i < inst->NumInOperands(); i++) {
      auto& op = inst->GetInOperand(i);
      if (i > 2 && op.type == SPV_OPERAND_TYPE_MEMORY_ACCESS) {
        break;
      }
      dst_operands.push_back(op);
    }
    if (inst->NumInOperands() > i) {
      // The are separate src and dst memory access operands.
      for (; i < inst->NumInOperands(); i++) {
        src_operands.push_back(inst->GetInOperand(i));
      }
    } else {
      // Shared memory access operands.
      src_operands = dst_operands;
    }
  }

  auto [dst_row_major, dst_mat_stride] = MatrixStride(dst);
  auto [src_row_major, src_mat_stride] = MatrixStride(src);

  // src and dst pointee types must match.
  auto pointee_type = dst_type->pointee_type();

  // Matrices (and their columns) are a special case.
  // The layout information is not on the matrix itself so we could be copying
  // matrices with completely different layouts. Just break them down into
  // component loads and stores.
  if ((dst_mat_stride != 0 || src_mat_stride != 0) &&
      !pointee_type->AsFloat()) {
    uint32_t num_cols = 1;
    uint32_t num_rows = 1;
    const analysis::Type* ele_type = nullptr;
    if (auto mat_type = pointee_type->AsMatrix()) {
      num_cols = mat_type->element_count();
      auto vec_type = mat_type->element_type()->AsVector();
      num_rows = vec_type->element_count();
      ele_type = vec_type->element_type();
    } else if (auto vec_type = pointee_type->AsVector()) {
      num_rows = vec_type->element_count();
      ele_type = vec_type->element_type();
    }

    uint32_t align = ele_type->AsFloat()->width() / 8;

    // Setup operands for the loads and stores.
    auto ld_operands = LoadOperands(src_operands, align);
    auto st_operands = StoreOperands(dst_operands, align);

    // Setup the access operands.
    spv::Op src_access_op =
        src_untyped ? spv::Op::OpUntypedAccessChainKHR : spv::Op::OpAccessChain;
    spv::Op dst_access_op =
        dst_untyped ? spv::Op::OpUntypedAccessChainKHR : spv::Op::OpAccessChain;
    uint32_t base_type_id = context()->get_type_mgr()->GetId(pointee_type);
    uint32_t ele_type_id = context()->get_type_mgr()->GetId(ele_type);

    std::vector<Operand> src_access_operands;
    if (src_untyped) {
      src_access_operands.push_back({SPV_OPERAND_TYPE_ID, {base_type_id}});
    }
    src_access_operands.push_back({SPV_OPERAND_TYPE_ID, {src->result_id()}});
    src_access_operands.push_back({SPV_OPERAND_TYPE_ID, {0}});

    std::vector<Operand> dst_access_operands;
    if (dst_untyped) {
      dst_access_operands.push_back({SPV_OPERAND_TYPE_ID, {base_type_id}});
    }
    dst_access_operands.push_back({SPV_OPERAND_TYPE_ID, {dst->result_id()}});
    dst_access_operands.push_back({SPV_OPERAND_TYPE_ID, {0}});

    if (num_cols > 1) {
      src_access_operands.push_back({SPV_OPERAND_TYPE_ID, {0}});
      dst_access_operands.push_back({SPV_OPERAND_TYPE_ID, {0}});
    }

    // Finally, unroll the copy as individual loads and stores.
    const uint32_t src_access_idx = src_untyped ? 2 : 1;
    const uint32_t dst_access_idx = dst_untyped ? 2 : 1;
    for (uint32_t c = 0; c < num_cols; c++) {
      for (uint32_t r = 0; r < num_rows; r++) {
        // Update the access operands
        if (num_cols == 1) {
          uint32_t row_id = context()->get_constant_mgr()->GetUIntConstId(r);
          src_access_operands[src_access_idx].words[0] = row_id;
          dst_access_operands[dst_access_idx].words[0] = row_id;
        } else {
          uint32_t col_id = context()->get_constant_mgr()->GetUIntConstId(c);
          uint32_t row_id = context()->get_constant_mgr()->GetUIntConstId(r);
          src_access_operands[src_access_idx].words[0] = col_id;
          src_access_operands[src_access_idx + 1].words[0] = row_id;
          dst_access_operands[dst_access_idx].words[0] = col_id;
          dst_access_operands[dst_access_idx + 1].words[0] = row_id;
        }

        uint32_t src_access_id = NextId();
        inst->InsertBefore(MakeUnique<Instruction>(
            context(), src_access_op, base_ptrs_[uint32_t(src_sc)]->result_id(),
            src_access_id, src_access_operands));

        uint32_t src_ld = NextId();
        // Update load operand ids
        ld_operands[0].words[0] = src_access_id;
        inst->InsertBefore(MakeUnique<Instruction>(
            context(), spv::Op::OpLoad, ele_type_id, src_ld, ld_operands));

        uint32_t dst_access_id = NextId();
        inst->InsertBefore(MakeUnique<Instruction>(
            context(), dst_access_op, base_ptrs_[uint32_t(dst_sc)]->result_id(),
            dst_access_id, dst_access_operands));

        // Update store operand ids
        st_operands[0].words[0] = dst_access_id;
        st_operands[1].words[0] = src_ld;
        inst->InsertBefore(MakeUnique<Instruction>(context(), spv::Op::OpStore,
                                                   0, 0, st_operands));
      }
    }

    return;
  }

  // For other cases just load from src and store to dst.
  // Don't bother with OpCopyMemorySized, too many shenanigans are necessary for
  // that.
  //
  // We don't need to set an align.
  auto ld_operands = LoadOperands(src_operands, 0);
  auto st_operands = StoreOperands(dst_operands, 0);

  ld_operands[0].words[0] = src->result_id();
  auto ld = inst->InsertBefore(MakeUnique<Instruction>(
      context(), spv::Op::OpLoad,
      context()->get_type_mgr()->GetId(pointee_type), NextId(), ld_operands));

  st_operands[0].words[0] = dst->result_id();
  st_operands[1].words[0] = ld->result_id();
  inst->InsertBefore(
      MakeUnique<Instruction>(context(), spv::Op::OpStore, 0, 0, st_operands));
}

void ConvertToUntyped::UpdateCooperativeMatrixLoadStore(Instruction* inst) {
  uint32_t stride_idx =
      inst->opcode() == spv::Op::OpCooperativeMatrixLoadKHR ? 2 : 3;
  // With typed pointers the stride is counted in elements, but with untyped
  // pointers the stride is interpreted in bytes. So if there is no stride
  // specified, nothing needs done.
  if (inst->NumInOperands() <= stride_idx) {
    return;
  }

  auto ptr =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  auto ptr_ty = context()->get_type_mgr()->GetType(ptr->type_id())->AsPointer();

  // The pointer must point to a scalar or vector of numeric type so there is a
  // very limited number of permutations to consider.
  //
  // Also the memory is in contiguous locations, so just use the size (e.g.
  // don't worry about 3-element vectors).
  auto type = ptr_ty->pointee_type();
  uint32_t size = 1;
  if (auto vec_ty = type->AsVector()) {
    size *= vec_ty->element_count();
    type = vec_ty->element_type();
  } else if (auto long_vec_ty = type->AsCooperativeVectorNV()) {
    size *= long_vec_ty->components();
    type = long_vec_ty->component_type();
  }

  if (auto int_type = type->AsInteger()) {
    size *= int_type->width() / 8;
  } else if (auto float_type = type->AsFloat()) {
    size *= float_type->width() / 8;
  }

  auto mul = inst->InsertBefore(MakeUnique<Instruction>(
      context(), spv::Op::OpIMul, context()->get_type_mgr()->GetUIntTypeId(),
      NextId(),
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(stride_idx)}},
          {SPV_OPERAND_TYPE_ID,
           {context()->get_constant_mgr()->GetUIntConstId(size)}}}));

  inst->SetInOperand(stride_idx, {mul->result_id()});
}

void ConvertToUntyped::Convert(Instruction* inst) {
  Instruction* replacement = nullptr;
  switch (inst->opcode()) {
    case spv::Op::OpTypePointer:
      replacement = ConvertPointer(inst);
      to_delete_.push_back(inst);
      break;
    case spv::Op::OpVariable:
      replacement = ConvertVariable(inst);
      to_delete_.push_back(inst);
      break;
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
      replacement = ConvertAccessChain(inst);
      to_delete_.push_back(inst);
      break;
    case spv::Op::OpArrayLength:
      replacement = ConvertArrayLength(inst);
      to_delete_.push_back(inst);
      break;
    case spv::Op::OpCopyMemory:
      ConvertCopyMemory(inst);
      to_delete_.push_back(inst);
      return;
    case spv::Op::OpCooperativeMatrixLoadKHR:
    case spv::Op::OpCooperativeMatrixStoreKHR:
      // Note: no deletion here.
      UpdateCooperativeMatrixLoadStore(inst);
      return;
    default:
      assert(false && "unhandled opcode");
      break;
  }

  remapped_ids_[inst->result_id()] = replacement->result_id();
}

void ConvertToUntyped::ConvertPointers() {
  // Convert instructions that change opcode with untyped pointers.
  std::vector<Instruction*> to_convert;
  get_module()->ForEachInst([this, &to_convert](Instruction* inst) {
    if (ShouldConvert(inst)) {
      to_convert.push_back(inst);
    }
  });

  for (auto* inst : to_convert) {
    Convert(inst);
  }

  // Update operands of instructions
  get_module()->ForEachInst([this](Instruction* inst) {
    for (uint32_t i = 0; i < inst->NumOperands(); i++) {
      auto& op = inst->GetOperand(i);
      if (spvIsIdType(op.type) && op.type != SPV_OPERAND_TYPE_RESULT_ID) {
        auto where = remapped_ids_.find(op.words[0]);
        if (where != remapped_ids_.end()) {
          op.words[0] = where->second;
        }
      }
    }
  });

  // Delete dead instructions
  for (auto* inst : to_delete_) {
    context()->KillInst(inst);
  }
}

}  // namespace opt
}  // namespace spvtools
