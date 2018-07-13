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
  UpgradeBarriers();

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
      bool src_coherent = false;
      bool src_volatile = false;
      bool dst_coherent = false;
      bool dst_volatile = false;
      SpvScope scope = SpvScopeDevice;
      SpvScope src_scope = SpvScopeDevice;
      SpvScope dst_scope = SpvScopeDevice;
      switch (inst->opcode()) {
        case SpvOpLoad:
        case SpvOpStore:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
        case SpvOpImageWrite:
          std::tie(is_coherent, is_volatile, scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          break;
        case SpvOpCopyMemory:
        case SpvOpCopyMemorySized:
          std::tie(dst_coherent, dst_volatile, dst_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(0u));
          std::tie(src_coherent, src_volatile, src_scope) =
              GetInstructionAttributes(inst->GetSingleWordInOperand(1u));
          break;
        default:
          break;
      }

      switch (inst->opcode()) {
        case SpvOpLoad:
          UpgradeFlags(inst, 1u, is_coherent, is_volatile, kAvailability,
                       kMemory);
          break;
        case SpvOpStore:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpCopyMemory:
          UpgradeFlags(inst, 2u, dst_coherent, dst_volatile, kAvailability,
                       kMemory);
          UpgradeFlags(inst, 2u, src_coherent, src_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpCopyMemorySized:
          UpgradeFlags(inst, 3u, dst_coherent, dst_volatile, kAvailability,
                       kMemory);
          UpgradeFlags(inst, 3u, src_coherent, src_volatile, kVisibility,
                       kMemory);
          break;
        case SpvOpImageRead:
        case SpvOpImageSparseRead:
          UpgradeFlags(inst, 2u, is_coherent, is_volatile, kAvailability,
                       kImage);
          break;
        case SpvOpImageWrite:
          UpgradeFlags(inst, 3u, is_coherent, is_volatile, kVisibility, kImage);
          break;
        default:
          break;
      }

      // |is_coherent| is never used for the same instructions as
      // |src_coherent| and |dst_coherent|.
      if (is_coherent) {
        inst->AddOperand(
            {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(scope)}});
      }
      // According to SPV_KHR_vulkan_memory_model, if both available and
      // visible flags are used the first scope operand is for availability
      // (reads) and the second is for visibiity (writes).
      if (src_coherent) {
        inst->AddOperand(
            {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(src_scope)}});
      }
      if (dst_coherent) {
        inst->AddOperand(
            {SPV_OPERAND_TYPE_SCOPE_ID, {GetScopeConstant(dst_scope)}});
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
  if (type->AsPointer() &&
      type->AsPointer()->storage_class() == SpvStorageClassWorkgroup) {
    return std::make_tuple(true, false, SpvScopeWorkgroup);
  }

  bool is_coherent = false;
  bool is_volatile = false;
  std::unordered_set<uint32_t> visited;
  std::tie(is_coherent, is_volatile) =
      TraceInstruction(context()->get_def_use_mgr()->GetDef(id),
                       std::vector<uint32_t>(), &visited);

  return std::make_tuple(is_coherent, is_volatile, SpvScopeDevice);
}

std::pair<bool, bool> UpgradeMemoryModel::TraceInstruction(
    ir::Instruction* inst, std::vector<uint32_t> indices,
    std::unordered_set<uint32_t>* visited) {
  auto iter = cache_.find(std::make_pair(inst->result_id(), indices));
  if (iter != cache_.end()) {
    return iter->second;
  }

  if (!visited->insert(inst->result_id()).second) {
    return std::make_pair(false, false);
  }

  // Initialize the cache before |indices| is (potentially) modified.
  auto& cached_result = cache_[std::make_pair(inst->result_id(), indices)];
  cached_result.first = false;
  cached_result.second = false;

  bool is_coherent = false;
  bool is_volatile = false;
  switch (inst->opcode()) {
    case SpvOpVariable:
    case SpvOpFunctionParameter:
      is_coherent |= HasDecoration(inst, 0, SpvDecorationCoherent);
      is_volatile |= HasDecoration(inst, 0, SpvDecorationVolatile);
      if (!is_coherent || !is_volatile) {
        bool type_coherent = false;
        bool type_volatile = false;
        std::tie(type_coherent, type_volatile) =
            CheckType(inst->type_id(), indices);
        is_coherent |= type_coherent;
        is_volatile |= type_volatile;
      }
      break;
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
      // Store indices in reverse order.
      for (uint32_t i = inst->NumInOperands() - 1; i > 0; --i) {
        indices.push_back(inst->GetSingleWordInOperand(i));
      }
      break;
    case SpvOpPtrAccessChain:
      // Store indices in reverse order. Skip the |Element| operand.
      for (uint32_t i = inst->NumInOperands() - 1; i > 1; --i) {
        indices.push_back(inst->GetSingleWordInOperand(i));
      }
      break;
    default:
      break;
  }

  // No point searching further.
  if (is_coherent && is_volatile) {
    cached_result.first = true;
    cached_result.second = true;
    return std::make_pair(true, true);
  }

  // Variables and function parameters are sources. Continue searching until we
  // reach them.
  if (inst->opcode() != SpvOpVariable &&
      inst->opcode() != SpvOpFunctionParameter) {
    inst->ForEachInId([this, &is_coherent, &is_volatile, &indices,
                       &visited](const uint32_t* id_ptr) {
      ir::Instruction* op_inst = context()->get_def_use_mgr()->GetDef(*id_ptr);
      const analysis::Type* type =
          context()->get_type_mgr()->GetType(op_inst->type_id());
      if (type &&
          (type->AsPointer() || type->AsImage() || type->AsSampledImage())) {
        bool operand_coherent = false;
        bool operand_volatile = false;
        std::tie(operand_coherent, operand_volatile) =
            TraceInstruction(op_inst, indices, visited);
        is_coherent |= operand_coherent;
        is_volatile |= operand_volatile;
      }
    });
  }

  cached_result.first = is_coherent;
  cached_result.second = is_volatile;
  return std::make_pair(is_coherent, is_volatile);
}

std::pair<bool, bool> UpgradeMemoryModel::CheckType(
    uint32_t type_id, const std::vector<uint32_t>& indices) {
  bool is_coherent = false;
  bool is_volatile = false;
  ir::Instruction* type_inst = context()->get_def_use_mgr()->GetDef(type_id);
  assert(type_inst->opcode() == SpvOpTypePointer);
  ir::Instruction* element_inst = context()->get_def_use_mgr()->GetDef(
      type_inst->GetSingleWordInOperand(1u));
  for (int i = (int)indices.size() - 1; i >= 0; --i) {
    if (is_coherent && is_volatile) break;

    if (element_inst->opcode() == SpvOpTypePointer) {
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(1u));
    } else if (element_inst->opcode() == SpvOpTypeStruct) {
      uint32_t index = indices.at(i);
      ir::Instruction* index_inst = context()->get_def_use_mgr()->GetDef(index);
      assert(index_inst->opcode() == SpvOpConstant);
      uint64_t value = GetIndexValue(index_inst);
      is_coherent |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   SpvDecorationCoherent);
      is_volatile |= HasDecoration(element_inst, static_cast<uint32_t>(value),
                                   SpvDecorationVolatile);
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(static_cast<uint32_t>(value)));
    } else {
      assert(spvOpcodeIsComposite(element_inst->opcode()));
      element_inst = context()->get_def_use_mgr()->GetDef(
          element_inst->GetSingleWordInOperand(1u));
    }
  }

  if (!is_coherent || !is_volatile) {
    bool remaining_coherent = false;
    bool remaining_volatile = false;
    std::tie(remaining_coherent, remaining_volatile) =
        CheckAllTypes(element_inst);
    is_coherent |= remaining_coherent;
    is_volatile |= remaining_volatile;
  }

  return std::make_pair(is_coherent, is_volatile);
}

std::pair<bool, bool> UpgradeMemoryModel::CheckAllTypes(
    const ir::Instruction* inst) {
  std::unordered_set<const ir::Instruction*> visited;
  std::vector<const ir::Instruction*> stack;
  stack.push_back(inst);

  bool is_coherent = false;
  bool is_volatile = false;
  while (!stack.empty()) {
    const ir::Instruction* def = stack.back();
    stack.pop_back();

    if (!visited.insert(def).second) continue;

    if (def->opcode() == SpvOpTypeStruct) {
      // Any member decorated with coherent and/or volatile is enough to have
      // the related operation be flagged as coherent and/or volatile.
      is_coherent |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   SpvDecorationCoherent);
      is_volatile |= HasDecoration(def, std::numeric_limits<uint32_t>::max(),
                                   SpvDecorationVolatile);
      if (is_coherent && is_volatile)
        return std::make_pair(is_coherent, is_volatile);

      // Check the subtypes.
      for (uint32_t i = 0; i < def->NumInOperands(); ++i) {
        stack.push_back(context()->get_def_use_mgr()->GetDef(
            def->GetSingleWordInOperand(i)));
      }
    } else if (spvOpcodeIsComposite(def->opcode())) {
      stack.push_back(context()->get_def_use_mgr()->GetDef(
          def->GetSingleWordInOperand(0u)));
    } else if (def->opcode() == SpvOpTypePointer) {
      stack.push_back(context()->get_def_use_mgr()->GetDef(
          def->GetSingleWordInOperand(1u)));
    }
  }

  return std::make_pair(is_coherent, is_volatile);
}

uint64_t UpgradeMemoryModel::GetIndexValue(ir::Instruction* index_inst) {
  const analysis::Constant* index_constant =
      context()->get_constant_mgr()->GetConstantFromInst(index_inst);
  assert(index_constant->AsIntConstant());
  if (index_constant->type()->AsInteger()->IsSigned()) {
    if (index_constant->type()->AsInteger()->width() == 32) {
      return index_constant->GetS32();
    } else {
      return index_constant->GetS64();
    }
  } else {
    if (index_constant->type()->AsInteger()->width() == 32) {
      return index_constant->GetU32();
    } else {
      return index_constant->GetU64();
    }
  }
}

bool UpgradeMemoryModel::HasDecoration(const ir::Instruction* inst,
                                       uint32_t value,
                                       SpvDecoration decoration) {
  // If the iteration was terminated early then an appropriate decoration was
  // found.
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      inst->result_id(), decoration, [value](const ir::Instruction& i) {
        if (i.opcode() == SpvOpDecorate || i.opcode() == SpvOpDecorateId) {
          return false;
        } else if (i.opcode() == SpvOpMemberDecorate) {
          if (value == i.GetSingleWordInOperand(1u) ||
              value == std::numeric_limits<uint32_t>::max())
            return false;
        }

        return true;
      });
}

void UpgradeMemoryModel::UpgradeFlags(ir::Instruction* inst,
                                      uint32_t in_operand, bool is_coherent,
                                      bool is_volatile,
                                      OperationType operation_type,
                                      InstructionType inst_type) {
  if (!is_coherent && !is_volatile) return;

  uint32_t flags = 0;
  if (inst->NumInOperands() > in_operand) {
    flags |= inst->GetSingleWordInOperand(in_operand);
  }
  if (is_coherent) {
    if (inst_type == kMemory) {
      flags |= SpvMemoryAccessNonPrivatePointerKHRMask;
      if (operation_type == kVisibility) {
        flags |= SpvMemoryAccessMakePointerVisibleKHRMask;
      } else {
        flags |= SpvMemoryAccessMakePointerAvailableKHRMask;
      }
    } else {
      flags |= SpvImageOperandsNonPrivateTexelKHRMask;
      if (operation_type == kVisibility) {
        flags |= SpvImageOperandsMakeTexelVisibleKHRMask;
      } else {
        flags |= SpvImageOperandsMakeTexelAvailableKHRMask;
      }
    }
  }

  if (is_volatile) {
    if (inst_type == kMemory) {
      flags |= SpvMemoryAccessVolatileMask;
    } else {
      flags |= SpvImageOperandsVolatileTexelKHRMask;
    }
  }

  if (inst->NumInOperands() > in_operand) {
    inst->SetInOperand(in_operand, {flags});
  } else if (inst_type == kMemory) {
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

void UpgradeMemoryModel::UpgradeBarriers() {
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;
  for (auto& fn : *get_module()) id2function[fn.result_id()] = &fn;

  std::vector<ir::Instruction*> barriers;
  // Collects all the control barriers in |function|. Returns true if the
  // function operates on the Output storage class.
  ProcessFunction CollectBarriers = [this, &barriers](ir::Function* function) {
    bool operates_on_output = false;
    for (auto& block : *function) {
      block.ForEachInst([this, &barriers,
                         &operates_on_output](ir::Instruction* inst) {
        if (inst->opcode() == SpvOpControlBarrier) {
          barriers.push_back(inst);
        } else if (!operates_on_output) {
          // This instruction operates on output storage class if it is a
          // pointer to output type or any input operand is a pointer to output
          // type.
          analysis::Type* type =
              context()->get_type_mgr()->GetType(inst->type_id());
          if (type && type->AsPointer() &&
              type->AsPointer()->storage_class() == SpvStorageClassOutput) {
            operates_on_output = true;
            return;
          }
          inst->ForEachInId([this, &operates_on_output](uint32_t* id_ptr) {
            ir::Instruction* op_inst =
                context()->get_def_use_mgr()->GetDef(*id_ptr);
            analysis::Type* op_type =
                context()->get_type_mgr()->GetType(op_inst->type_id());
            if (op_type && op_type->AsPointer() &&
                op_type->AsPointer()->storage_class() == SpvStorageClassOutput)
              operates_on_output = true;
          });
        }
      });
    }
    return operates_on_output;
  };

  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points())
    if (e.GetSingleWordInOperand(0u) == SpvExecutionModelTessellationControl) {
      roots.push(e.GetSingleWordInOperand(1u));
      if (ProcessCallTreeFromRoots(CollectBarriers, id2function, &roots)) {
        for (auto barrier : barriers) {
          // Add OutputMemoryKHR to the semantics of the barriers.
          uint32_t semantics_id = barrier->GetSingleWordInOperand(2u);
          ir::Instruction* semantics_inst =
              context()->get_def_use_mgr()->GetDef(semantics_id);
          analysis::Type* semantics_type =
              context()->get_type_mgr()->GetType(semantics_inst->type_id());
          uint64_t semantics_value = GetIndexValue(semantics_inst);
          const analysis::Constant* constant =
              context()->get_constant_mgr()->GetConstant(
                  semantics_type, {static_cast<uint32_t>(semantics_value) |
                                   SpvMemorySemanticsOutputMemoryKHRMask});
          barrier->SetInOperand(2u, {context()
                                         ->get_constant_mgr()
                                         ->GetDefiningInstruction(constant)
                                         ->result_id()});
        }
      }
      barriers.clear();
    }
}

}  // namespace opt
}  // namespace spvtools
