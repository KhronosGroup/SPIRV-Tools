// Copyright (c) 2022 Google LLC
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

#include "source/opt/interface_var_sroa.h"

#include <iostream>

#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/log.h"
#include "source/opt/type_manager.h"
#include "source/util/make_unique.h"

const static uint32_t kOpDecorateDecorationInOperandIndex = 1;
const static uint32_t kOpDecorateLiteralInOperandIndex = 2;
const static uint32_t kOpVariableStorageClassInOperandIndex = 0;
const static uint32_t kOpTypeArrayElemTypeInOperandIndex = 0;
const static uint32_t kOpTypeArrayLengthInOperandIndex = 1;
const static uint32_t kOpTypeMatrixColCountInOperandIndex = 1;
const static uint32_t kOpTypeMatrixColTypeInOperandIndex = 0;
const static uint32_t kOpTypePtrTypeInOperandIndex = 1;
const static uint32_t kOpConstantValueInOperandIndex = 0;

namespace spvtools {
namespace opt {
namespace {

// Get the length of the OpTypeArray |array_type|.
uint32_t GetArrayLength(analysis::DefUseManager* def_use_mgr,
                        Instruction* array_type) {
  assert(array_type->opcode() == SpvOpTypeArray);
  uint32_t const_int_id =
      array_type->GetSingleWordInOperand(kOpTypeArrayLengthInOperandIndex);
  Instruction* array_length_inst = def_use_mgr->GetDef(const_int_id);
  assert(array_length_inst->opcode() == SpvOpConstant);
  return array_length_inst->GetSingleWordInOperand(
      kOpConstantValueInOperandIndex);
}

// Get the element type instruction of the OpTypeArray |array_type|.
Instruction* GetArrayElementType(analysis::DefUseManager* def_use_mgr,
                                 Instruction* array_type) {
  assert(array_type->opcode() == SpvOpTypeArray);
  uint32_t elem_type_id =
      array_type->GetSingleWordInOperand(kOpTypeArrayElemTypeInOperandIndex);
  return def_use_mgr->GetDef(elem_type_id);
}

// Get the column type instruction of the OpTypeMatrix |matrix_type|.
Instruction* GetMatrixColumnType(analysis::DefUseManager* def_use_mgr,
                                 Instruction* matrix_type) {
  assert(matrix_type->opcode() == SpvOpTypeMatrix);
  uint32_t column_type_id =
      matrix_type->GetSingleWordInOperand(kOpTypeMatrixColTypeInOperandIndex);
  return def_use_mgr->GetDef(column_type_id);
}

// Traverses the component type of OpTypeArray or OpTypeMatrix. Repeats it
// |depth_to_component| times recursively and returns the component type.
// |type_id| is the result id of the OpTypeArray or OpTypeMatrix instruction.
uint32_t GetComponentTypeOfArrayMatrix(analysis::DefUseManager* def_use_mgr,
                                       uint32_t type_id,
                                       uint32_t depth_to_component) {
  if (depth_to_component == 0) return type_id;

  Instruction* type_inst = def_use_mgr->GetDef(type_id);
  if (type_inst->opcode() == SpvOpTypeArray) {
    uint32_t elem_type_id =
        type_inst->GetSingleWordInOperand(kOpTypeArrayElemTypeInOperandIndex);
    return GetComponentTypeOfArrayMatrix(def_use_mgr, elem_type_id,
                                         depth_to_component - 1);
  }

  assert(type_inst->opcode() == SpvOpTypeMatrix);
  uint32_t column_type_id =
      type_inst->GetSingleWordInOperand(kOpTypeMatrixColTypeInOperandIndex);
  return GetComponentTypeOfArrayMatrix(def_use_mgr, column_type_id,
                                       depth_to_component - 1);
}

// Creates an OpDecorate instruction whose Target is |var_id| and Decoration is
// |decoration|. Adds |literal| as an extra operand of the instruction.
void CreateDecoration(analysis::DecorationManager* decoration_mgr,
                      uint32_t var_id, SpvDecoration decoration,
                      uint32_t literal) {
  std::vector<Operand> operands({
      {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {var_id}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_DECORATION,
       {static_cast<uint32_t>(decoration)}},
      {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {literal}},
  });
  decoration_mgr->AddDecoration(SpvOpDecorate, std::move(operands));
}

// Replaces load instructions with composite construct instructions in all the
// users of the loads. |loads_to_composites| is the mapping from each load to
// its corresponding OpCompositeConstruct.
void ReplaceLoadWithCompositeConstruct(
    IRContext* context,
    const std::unordered_map<Instruction*, Instruction*>& loads_to_composites) {
  for (const auto& load_and_composite : loads_to_composites) {
    Instruction* load = load_and_composite.first;
    Instruction* composite_construct = load_and_composite.second;

    std::vector<Instruction*> users;
    context->get_def_use_mgr()->ForEachUse(
        load, [&users, composite_construct](Instruction* user, uint32_t index) {
          user->GetOperand(index).words[0] = composite_construct->result_id();
          users.push_back(user);
        });

    for (Instruction* user : users)
      context->get_def_use_mgr()->AnalyzeInstUse(user);
  }
}

// Returns the storage class of the instruction |var|.
SpvStorageClass GetStorageClass(Instruction* var) {
  return static_cast<SpvStorageClass>(
      var->GetSingleWordInOperand(kOpVariableStorageClassInOperandIndex));
}

}  // namespace

bool InterfaceVariableScalarReplacement::HasExtraArrayness(Instruction* var) {
  return !context()->get_decoration_mgr()->HasDecoration(var->result_id(),
                                                         SpvDecorationPatch);
}

bool InterfaceVariableScalarReplacement::GetVariableLocation(
    Instruction* var, uint32_t* location) {
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      var->result_id(), SpvDecorationLocation, [location](const Instruction& inst) {
        *location =
            inst.GetSingleWordInOperand(kOpDecorateLiteralInOperandIndex);
        return false;
      });
}

bool InterfaceVariableScalarReplacement::GetVariableComponent(
    Instruction* var, uint32_t* component) {
  return !context()->get_decoration_mgr()->WhileEachDecoration(
      var->result_id(), SpvDecorationComponent, [component](const Instruction& inst) {
        *component =
            inst.GetSingleWordInOperand(kOpDecorateLiteralInOperandIndex);
        return false;
      });
}

std::vector<Instruction*>
InterfaceVariableScalarReplacement::CollectInterfaceVariables() {
  std::vector<Instruction*> interface_vars;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() != SpvOpVariable) continue;

    SpvStorageClass storage_class = GetStorageClass(&inst);
    if (storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      continue;
    }

    interface_vars.push_back(&inst);
  }
  return interface_vars;
}

void InterfaceVariableScalarReplacement::KillInstructionAndUsers(
    Instruction* inst) {
  if (inst->opcode() == SpvOpEntryPoint) {
    return;
  }
  if (inst->opcode() != SpvOpAccessChain) {
    context()->KillInst(inst);
    return;
  }
  context()->get_def_use_mgr()->ForEachUser(
      inst, [this](Instruction* user) { KillInstructionAndUsers(user); });
  context()->KillInst(inst);
}

void InterfaceVariableScalarReplacement::KillInstructionsAndUsers(
    const std::vector<Instruction*>& insts) {
  for (Instruction* inst : insts) {
    KillInstructionAndUsers(inst);
  }
}

void InterfaceVariableScalarReplacement::KillLocationAndComponentDecorations(
    uint32_t var_id) {
  context()->get_decoration_mgr()->RemoveDecorationsFrom(
      var_id, [](const Instruction& inst) {
        uint32_t decoration =
            inst.GetSingleWordInOperand(kOpDecorateDecorationInOperandIndex);
        return decoration == SpvDecorationLocation ||
               decoration == SpvDecorationComponent;
      });
}

bool InterfaceVariableScalarReplacement::FlattenInterfaceVariable(
    Instruction* interface_var, Instruction* interface_var_type,
    uint32_t location, uint32_t component, uint32_t extra_array_length) {
  NestedCompositeComponents flattened_interface_vars =
      CreateFlattenedInterfaceVarsForReplacement(
          interface_var_type, GetStorageClass(interface_var),
          extra_array_length);

  AddLocationAndComponentDecorations(flattened_interface_vars, &location,
                                     component);
  KillLocationAndComponentDecorations(interface_var->result_id());

  if (!ReplaceInterfaceVarWithFlattenedVars(interface_var,
                                            extra_array_length,
                                            flattened_interface_vars)) {
    return false;
  }

  context()->KillInst(interface_var);
  return true;
}

bool InterfaceVariableScalarReplacement::ReplaceInterfaceVarWithFlattenedVars(
    Instruction* interface_var, uint32_t extra_arrayness,
    const NestedCompositeComponents& flattened_interface_vars) {
  std::vector<Instruction*> users;
  context()->get_def_use_mgr()->ForEachUser(
      interface_var, [&users](Instruction* user) { users.push_back(user); });

  std::vector<uint32_t> interface_var_component_indices;
  std::unordered_map<Instruction*, Instruction*> loads_to_composites;
  std::unordered_map<Instruction*, Instruction*>
      loads_for_access_chain_to_composites;
  if (extra_arrayness != 0) {
    // Note that the extra arrayness is the first dimenion of the array
    // interface variable.
    for (uint32_t index = 0; index < extra_arrayness; ++index) {
      std::unordered_map<Instruction*, Instruction*> loads_to_component_values;
      if (!ReplaceInterfaceVarComponentsWithFlattenedVars(
              interface_var, users, flattened_interface_vars,
              interface_var_component_indices, &index,
              &loads_to_component_values,
              &loads_for_access_chain_to_composites)) {
        return false;
      }
      AddComponentsToCompositesForLoads(loads_to_component_values,
                                        &loads_to_composites, 0);
    }
  } else if (!ReplaceInterfaceVarComponentsWithFlattenedVars(
                 interface_var, users, flattened_interface_vars,
                 interface_var_component_indices, nullptr, &loads_to_composites,
                 &loads_for_access_chain_to_composites)) {
    return false;
  }

  ReplaceLoadWithCompositeConstruct(context(), loads_to_composites);
  ReplaceLoadWithCompositeConstruct(context(),
                                    loads_for_access_chain_to_composites);

  KillInstructionsAndUsers(users);
  return true;
}

void InterfaceVariableScalarReplacement::AddLocationAndComponentDecorations(
    const NestedCompositeComponents& flattened_vars, uint32_t* location,
    uint32_t component) {
  if (!flattened_vars.HasMultipleComponents()) {
    uint32_t var_id = flattened_vars.GetComponentVariable()->result_id();
    CreateDecoration(context()->get_decoration_mgr(), var_id,
                     SpvDecorationLocation, *location);
    CreateDecoration(context()->get_decoration_mgr(), var_id,
                     SpvDecorationComponent, component);
    ++(*location);
    return;
  }
  for (const auto& flattened_var : flattened_vars.GetComponents()) {
    AddLocationAndComponentDecorations(flattened_var, location, component);
  }
}

bool InterfaceVariableScalarReplacement::
    ReplaceInterfaceVarComponentsWithFlattenedVars(
        Instruction* interface_var,
        const std::vector<Instruction*>& interface_var_users,
        const NestedCompositeComponents& flattened_interface_vars,
        std::vector<uint32_t>& interface_var_component_indices,
        const uint32_t* extra_array_index,
        std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
        std::unordered_map<Instruction*, Instruction*>*
            loads_for_access_chain_to_composites) {
  if (!flattened_interface_vars.HasMultipleComponents()) {
    for (Instruction* interface_var_user : interface_var_users) {
      if (!ReplaceInterfaceVarComponentWithFlattenedVar(
              interface_var, interface_var_user,
              flattened_interface_vars.GetComponentVariable(),
              interface_var_component_indices, extra_array_index,
              loads_to_composites, loads_for_access_chain_to_composites)) {
        return false;
      }
    }
    return true;
  }
  return ReplaceMultipleComponentsOfInterfaceVarWithFlattenedVars(
      interface_var, interface_var_users,
      flattened_interface_vars.GetComponents(), interface_var_component_indices,
      extra_array_index, loads_to_composites,
      loads_for_access_chain_to_composites);
}

bool InterfaceVariableScalarReplacement::
    ReplaceMultipleComponentsOfInterfaceVarWithFlattenedVars(
        Instruction* interface_var,
        const std::vector<Instruction*>& interface_var_users,
        const std::vector<NestedCompositeComponents>& components,
        std::vector<uint32_t>& interface_var_component_indices,
        const uint32_t* extra_array_index,
        std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
        std::unordered_map<Instruction*, Instruction*>*
            loads_for_access_chain_to_composites) {
  for (uint32_t i = 0; i < components.size(); ++i) {
    interface_var_component_indices.push_back(i);
    std::unordered_map<Instruction*, Instruction*> loads_to_component_values;
    std::unordered_map<Instruction*, Instruction*>
        loads_for_access_chain_to_component_values;
    if (!ReplaceInterfaceVarComponentsWithFlattenedVars(
            interface_var, interface_var_users, components[i],
            interface_var_component_indices, extra_array_index,
            &loads_to_component_values,
            &loads_for_access_chain_to_component_values)) {
      return false;
    }
    interface_var_component_indices.pop_back();

    uint32_t depth_to_component =
        static_cast<uint32_t>(interface_var_component_indices.size());
    AddComponentsToCompositesForLoads(
        loads_for_access_chain_to_component_values,
        loads_for_access_chain_to_composites, depth_to_component);
    if (extra_array_index) ++depth_to_component;
    AddComponentsToCompositesForLoads(loads_to_component_values,
                                      loads_to_composites, depth_to_component);
  }
  return true;
}

bool InterfaceVariableScalarReplacement::
    ReplaceInterfaceVarComponentWithFlattenedVar(
        Instruction* interface_var, Instruction* interface_var_user,
        Instruction* flattened_var,
        const std::vector<uint32_t>& interface_var_component_indices,
        const uint32_t* extra_array_index,
        std::unordered_map<Instruction*, Instruction*>*
            loads_to_component_values,
        std::unordered_map<Instruction*, Instruction*>*
            loads_for_access_chain_to_component_values) {
  SpvOp opcode = interface_var_user->opcode();
  if (opcode == SpvOpStore || opcode == SpvOpLoad) {
    CreateLoadOrStoreToFlattenedInterfaceVar(
        interface_var_user, interface_var_component_indices, flattened_var,
        extra_array_index, loads_to_component_values);
    return true;
  }

  // Copy OpName and annotation instructions only once. Therefore, we create
  // them only for the first element of the extra array.
  if (extra_array_index && *extra_array_index != 0) return true;

  if (opcode == SpvOpDecorateId || opcode == SpvOpDecorateString ||
      opcode == SpvOpDecorate) {
    CloneAnnotationForVariable(interface_var_user, flattened_var->result_id());
    return true;
  }

  if (opcode == SpvOpName) {
    std::unique_ptr<Instruction> new_inst(interface_var_user->Clone(context()));
    new_inst->SetInOperand(0, {flattened_var->result_id()});
    context()->AddDebug2Inst(std::move(new_inst));
    return true;
  }

  if (opcode == SpvOpEntryPoint) {
    return ReplaceInterfaceVarInEntryPoint(interface_var, interface_var_user,
                                           flattened_var->result_id());
  }

  if (opcode == SpvOpAccessChain) {
    ReplaceAccessChainWithFlattenedVar(
        interface_var_user, interface_var_component_indices, flattened_var,
        loads_for_access_chain_to_component_values);
    return true;
  }

  std::string message("Unhandled instruction");
  message += "\n  " + interface_var_user->PrettyPrint(
                          SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  message +=
      "\nfor flattening interface variable\n  " +
      interface_var->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
  return false;
}

void InterfaceVariableScalarReplacement::UseBaseAccessChainForAccessChain(
    Instruction* access_chain, Instruction* base_access_chain) {
  assert(base_access_chain->opcode() == SpvOpAccessChain &&
         access_chain->opcode() == SpvOpAccessChain &&
         access_chain->GetSingleWordInOperand(0) ==
             base_access_chain->result_id());
  Instruction::OperandList new_operands;
  for (uint32_t i = 0; i < base_access_chain->NumInOperands(); ++i) {
    new_operands.emplace_back(base_access_chain->GetInOperand(i));
  }
  for (uint32_t i = 1; i < access_chain->NumInOperands(); ++i) {
    new_operands.emplace_back(access_chain->GetInOperand(i));
  }
  access_chain->SetInOperands(std::move(new_operands));
}

Instruction* InterfaceVariableScalarReplacement::CreateAccessChainToVar(
    uint32_t var_type_id, Instruction* var, Instruction* access_chain,
    uint32_t* component_type_id) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  *component_type_id = GetComponentTypeOfArrayMatrix(
      def_use_mgr, var_type_id, access_chain->NumInOperands() - 1);

  uint32_t ptr_type_id =
      GetPointerType(*component_type_id, GetStorageClass(var));

  std::unique_ptr<Instruction> new_access_chain(
      new Instruction(context(), SpvOpAccessChain, ptr_type_id, TakeNextId(),
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_ID, {var->result_id()}}}));
  for (uint32_t i = 1; i < access_chain->NumInOperands(); ++i) {
    auto operand = access_chain->GetInOperand(i);
    new_access_chain->AddOperand(std::move(operand));
  }

  Instruction* inst = new_access_chain.get();
  def_use_mgr->AnalyzeInstDefUse(inst);
  access_chain->InsertBefore(std::move(new_access_chain));
  return inst;
}

Instruction* InterfaceVariableScalarReplacement::CreateAccessChainWithIndex(
    uint32_t component_type_id, Instruction* var, uint32_t index,
    Instruction* insert_before) {
  uint32_t ptr_type_id =
      GetPointerType(component_type_id, GetStorageClass(var));
  uint32_t index_id = context()->get_constant_mgr()->GetUIntConst(index);
  std::unique_ptr<Instruction> new_access_chain(
      new Instruction(context(), SpvOpAccessChain, ptr_type_id, TakeNextId(),
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_ID, {var->result_id()}},
                          {SPV_OPERAND_TYPE_ID, {index_id}},
                      }));
  Instruction* inst = new_access_chain.get();
  context()->get_def_use_mgr()->AnalyzeInstDefUse(inst);
  insert_before->InsertBefore(std::move(new_access_chain));
  return inst;
}

void InterfaceVariableScalarReplacement::ReplaceAccessChainWithFlattenedVar(
    Instruction* access_chain,
    const std::vector<uint32_t>& interface_var_component_indices,
    Instruction* flattened_var,
    std::unordered_map<Instruction*, Instruction*>* loads_to_component_values) {
  // Note that we have a strong assumption that |access_chain| has only a single
  // index that is for the extra arrayness.
  context()->get_def_use_mgr()->ForEachUser(
      access_chain,
      [this, access_chain, &interface_var_component_indices, flattened_var,
       loads_to_component_values](Instruction* user) {
        switch (user->opcode()) {
          case SpvOpAccessChain:
            UseBaseAccessChainForAccessChain(user, access_chain);
            ReplaceAccessChainWithFlattenedVar(
                user, interface_var_component_indices, flattened_var,
                loads_to_component_values);
            return;
          case SpvOpLoad:
          case SpvOpStore:
            CreateLoadOrStoreToFlattenedInterfaceVarAccessChain(
                access_chain, user, interface_var_component_indices,
                flattened_var, loads_to_component_values);
            return;
          default:
            break;
        }
      });
}

void InterfaceVariableScalarReplacement::CloneAnnotationForVariable(
    Instruction* annotation_inst, uint32_t var_id) {
  assert(annotation_inst->opcode() == SpvOpDecorate ||
         annotation_inst->opcode() == SpvOpDecorateId ||
         annotation_inst->opcode() == SpvOpDecorateString);
  std::unique_ptr<Instruction> new_inst(annotation_inst->Clone(context()));
  new_inst->SetInOperand(0, {var_id});
  context()->AddAnnotationInst(std::move(new_inst));
}

bool InterfaceVariableScalarReplacement::ReplaceInterfaceVarInEntryPoint(
    Instruction* interface_var, Instruction* entry_point,
    uint32_t flattened_var_id) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t interface_var_id = interface_var->result_id();
  if (interface_vars_removed_from_entry_point_operands_.find(
          interface_var_id) !=
      interface_vars_removed_from_entry_point_operands_.end()) {
    entry_point->AddOperand({SPV_OPERAND_TYPE_ID, {flattened_var_id}});
    def_use_mgr->AnalyzeInstUse(entry_point);
    return true;
  }

  bool success = !entry_point->WhileEachInId(
      [&interface_var_id, &flattened_var_id](uint32_t* id) {
        if (*id == interface_var_id) {
          *id = flattened_var_id;
          return false;
        }
        return true;
      });
  if (!success) {
    std::string message(
        "interface variable is not an operand of the entry point");
    message += "\n  " + interface_var->PrettyPrint(
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
    message += "\n  " + entry_point->PrettyPrint(
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
    context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
    return false;
  }

  def_use_mgr->AnalyzeInstUse(entry_point);
  interface_vars_removed_from_entry_point_operands_.insert(interface_var_id);
  return true;
}

uint32_t InterfaceVariableScalarReplacement::GetPointeeTypeIdOfVar(
    Instruction* var) {
  assert(var->opcode() == SpvOpVariable);

  uint32_t ptr_type_id = var->type_id();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  Instruction* ptr_type_inst = def_use_mgr->GetDef(ptr_type_id);

  assert(ptr_type_inst->opcode() == SpvOpTypePointer &&
         "Variable must have a pointer type.");
  return ptr_type_inst->GetSingleWordInOperand(kOpTypePtrTypeInOperandIndex);
}

void InterfaceVariableScalarReplacement::
    CreateLoadOrStoreToFlattenedInterfaceVar(
        Instruction* load_or_store,
        const std::vector<uint32_t>& interface_var_component_indices,
        Instruction* flattened_var, const uint32_t* extra_array_index,
        std::unordered_map<Instruction*, Instruction*>*
            loads_to_component_values) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(flattened_var);
  Instruction* ptr = flattened_var;
  if (extra_array_index) {
    auto* ty_mgr = context()->get_type_mgr();
    analysis::Array* array_type = ty_mgr->GetType(component_type_id)->AsArray();
    assert(array_type != nullptr);
    component_type_id = ty_mgr->GetTypeInstruction(array_type->element_type());
    ptr = CreateAccessChainWithIndex(component_type_id, flattened_var,
                                     *extra_array_index, load_or_store);
  }

  if (load_or_store->opcode() == SpvOpStore) {
    CreateStoreToFlattenedInterfaceVar(component_type_id, load_or_store,
                                       interface_var_component_indices, ptr,
                                       extra_array_index);
    return;
  }

  assert(load_or_store->opcode() == SpvOpLoad);
  Instruction* component_value =
      CreateLoad(component_type_id, ptr, load_or_store);
  loads_to_component_values->insert({load_or_store, component_value});
}

Instruction* InterfaceVariableScalarReplacement::CreateLoad(
    uint32_t type_id, Instruction* ptr, Instruction* insert_before) {
  std::unique_ptr<Instruction> load_in_unique_ptr(
      new Instruction(context(), SpvOpLoad, type_id, TakeNextId(),
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_ID, {ptr->result_id()}}}));
  Instruction* load = load_in_unique_ptr.get();
  context()->get_def_use_mgr()->AnalyzeInstDefUse(load);
  insert_before->InsertBefore(std::move(load_in_unique_ptr));
  return load;
}

void InterfaceVariableScalarReplacement::CreateStoreToFlattenedInterfaceVar(
    uint32_t component_type_id, Instruction* store_to_interface_var,
    const std::vector<uint32_t>& interface_var_component_indices,
    Instruction* ptr_to_flattened_var, const uint32_t* extra_array_index) {
  uint32_t value_id = store_to_interface_var->GetSingleWordInOperand(1);
  std::unique_ptr<Instruction> composite_extract(CreateCompositeExtract(
      component_type_id, value_id, interface_var_component_indices,
      extra_array_index));

  std::unique_ptr<Instruction> new_store(
      store_to_interface_var->Clone(context()));
  new_store->SetInOperand(0, {ptr_to_flattened_var->result_id()});
  new_store->SetInOperand(1, {composite_extract->result_id()});

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->AnalyzeInstDefUse(composite_extract.get());
  def_use_mgr->AnalyzeInstDefUse(new_store.get());

  store_to_interface_var->InsertBefore(std::move(composite_extract));
  store_to_interface_var->InsertBefore(std::move(new_store));
}

Instruction* InterfaceVariableScalarReplacement::CreateCompositeExtract(
    uint32_t type_id, uint32_t composite_id,
    const std::vector<uint32_t>& indexes, const uint32_t* extra_first_index) {
  uint32_t component_id = TakeNextId();
  Instruction* composite_extract = new Instruction(
      context(), SpvOpCompositeExtract, type_id, component_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {composite_id}}});
  if (extra_first_index) {
    composite_extract->AddOperand(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {*extra_first_index}});
  }
  for (uint32_t index : indexes) {
    composite_extract->AddOperand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}});
  }
  return composite_extract;
}

void InterfaceVariableScalarReplacement::
    CreateLoadOrStoreToFlattenedInterfaceVarAccessChain(
        Instruction* access_chain, Instruction* load_or_store,
        const std::vector<uint32_t>& interface_var_component_indices,
        Instruction* flattened_var,
        std::unordered_map<Instruction*, Instruction*>*
            loads_to_component_values) {
  uint32_t component_type_id = GetPointeeTypeIdOfVar(flattened_var);
  Instruction* ptr = flattened_var;
  if (access_chain != nullptr) {
    ptr = CreateAccessChainToVar(component_type_id, flattened_var, access_chain,
                                 &component_type_id);
  }

  if (load_or_store->opcode() == SpvOpStore) {
    CreateStoreToFlattenedInterfaceVar(component_type_id, load_or_store,
                                       interface_var_component_indices, ptr,
                                       nullptr);
    return;
  }

  assert(load_or_store->opcode() == SpvOpLoad);
  Instruction* component_value =
      CreateLoad(component_type_id, ptr, load_or_store);
  loads_to_component_values->insert({load_or_store, component_value});
}

Instruction*
InterfaceVariableScalarReplacement::CreateCompositeConstructForComponentOfLoad(
    Instruction* load, uint32_t depth_to_component) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t type_id = load->type_id();
  if (depth_to_component != 0) {
    type_id = GetComponentTypeOfArrayMatrix(def_use_mgr, load->type_id(),
                                            depth_to_component);
  }
  uint32_t new_id = context()->TakeNextId();
  std::unique_ptr<Instruction> new_composite_construct(
      new Instruction(context(), SpvOpCompositeConstruct, type_id, new_id, {}));
  Instruction* composite_construct = new_composite_construct.get();
  def_use_mgr->AnalyzeInstDefUse(composite_construct);

  // Insert |new_composite_construct| after |load|. When there are multiple
  // recursive composite construct instructions for a load, we have to place the
  // composite construct with a lower depth later because it constructs the
  // composite that contains other composites with lower depths.
  auto* insert_before = load->NextNode();
  while (true) {
    auto itr =
        composite_ids_to_component_depths.find(insert_before->result_id());
    if (itr == composite_ids_to_component_depths.end()) break;
    if (itr->second <= depth_to_component) break;
    insert_before = insert_before->NextNode();
  }
  insert_before->InsertBefore(std::move(new_composite_construct));
  composite_ids_to_component_depths.insert({new_id, depth_to_component});
  return composite_construct;
}

void InterfaceVariableScalarReplacement::AddComponentsToCompositesForLoads(
    const std::unordered_map<Instruction*, Instruction*>&
        loads_to_component_values,
    std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
    uint32_t depth_to_component) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  for (auto& load_and_component_vale : loads_to_component_values) {
    Instruction* load = load_and_component_vale.first;
    Instruction* component_value = load_and_component_vale.second;
    Instruction* composite_construct = nullptr;
    auto itr = loads_to_composites->find(load);
    if (itr == loads_to_composites->end()) {
      composite_construct =
          CreateCompositeConstructForComponentOfLoad(load, depth_to_component);
      loads_to_composites->insert({load, composite_construct});
    } else {
      composite_construct = itr->second;
    }
    composite_construct->AddOperand(
        {SPV_OPERAND_TYPE_ID, {component_value->result_id()}});
    def_use_mgr->AnalyzeInstDefUse(composite_construct);
  }
}

uint32_t InterfaceVariableScalarReplacement::GetArrayType(
    uint32_t elem_type_id, uint32_t array_length) {
  analysis::Type* elem_type = context()->get_type_mgr()->GetType(elem_type_id);
  uint32_t array_length_id =
      context()->get_constant_mgr()->GetUIntConst(array_length);
  analysis::Array array_type(
      elem_type,
      analysis::Array::LengthInfo{array_length_id, {0, array_length}});
  return context()->get_type_mgr()->GetTypeInstruction(&array_type);
}

uint32_t InterfaceVariableScalarReplacement::GetPointerType(
    uint32_t type_id, SpvStorageClass storage_class) {
  analysis::Type* type = context()->get_type_mgr()->GetType(type_id);
  analysis::Pointer ptr_type(type, storage_class);
  return context()->get_type_mgr()->GetTypeInstruction(&ptr_type);
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateFlattenedInterfaceVarsForArray(
    Instruction* interface_var_type, SpvStorageClass storage_class,
    uint32_t extra_array_length) {
  assert(interface_var_type->opcode() == SpvOpTypeArray);

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t array_length = GetArrayLength(def_use_mgr, interface_var_type);
  Instruction* elem_type = GetArrayElementType(def_use_mgr, interface_var_type);

  NestedCompositeComponents flattened_vars;
  while (array_length > 0) {
    NestedCompositeComponents flattened_vars_for_element =
        CreateFlattenedInterfaceVarsForReplacement(elem_type, storage_class,
                                                   extra_array_length);
    flattened_vars.AddComponent(flattened_vars_for_element);
    --array_length;
  }
  return flattened_vars;
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateFlattenedInterfaceVarsForMatrix(
    Instruction* interface_var_type, SpvStorageClass storage_class,
    uint32_t extra_array_length) {
  assert(interface_var_type->opcode() == SpvOpTypeMatrix);

  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t column_count = interface_var_type->GetSingleWordInOperand(
      kOpTypeMatrixColCountInOperandIndex);
  Instruction* column_type =
      GetMatrixColumnType(def_use_mgr, interface_var_type);

  NestedCompositeComponents flattened_vars;
  while (column_count > 0) {
    NestedCompositeComponents flattened_vars_for_column =
        CreateFlattenedInterfaceVarsForReplacement(column_type, storage_class,
                                                   extra_array_length);
    flattened_vars.AddComponent(flattened_vars_for_column);
    --column_count;
  }
  return flattened_vars;
}

InterfaceVariableScalarReplacement::NestedCompositeComponents
InterfaceVariableScalarReplacement::CreateFlattenedInterfaceVarsForReplacement(
    Instruction* interface_var_type, SpvStorageClass storage_class,
    uint32_t extra_array_length) {
  // Handle array case.
  if (interface_var_type->opcode() == SpvOpTypeArray) {
    return CreateFlattenedInterfaceVarsForArray(
        interface_var_type, storage_class, extra_array_length);
  }

  // Handle matrix case.
  if (interface_var_type->opcode() == SpvOpTypeMatrix) {
    return CreateFlattenedInterfaceVarsForMatrix(
        interface_var_type, storage_class, extra_array_length);
  }

  // Handle scalar or vector case.
  NestedCompositeComponents flattened_var;
  uint32_t type_id = interface_var_type->result_id();
  if (extra_array_length != 0) {
    type_id = GetArrayType(type_id, extra_array_length);
  }
  uint32_t ptr_type_id =
      context()->get_type_mgr()->FindPointerToType(type_id, storage_class);
  uint32_t id = TakeNextId();
  std::unique_ptr<Instruction> variable(
      new Instruction(context(), SpvOpVariable, ptr_type_id, id,
                      std::initializer_list<Operand>{
                          {SPV_OPERAND_TYPE_STORAGE_CLASS,
                           {static_cast<uint32_t>(storage_class)}}}));
  flattened_var.SetSingleComponentVariable(variable.get());
  context()->AddGlobalValue(std::move(variable));
  return flattened_var;
}

Instruction* InterfaceVariableScalarReplacement::GetTypeOfVariable(
    Instruction* var) {
  uint32_t pointee_type_id = GetPointeeTypeIdOfVar(var);
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  return def_use_mgr->GetDef(pointee_type_id);
}

Pass::Status InterfaceVariableScalarReplacement::Process() {
  std::vector<Instruction*> interface_vars = CollectInterfaceVariables();

  Pass::Status status = Status::SuccessWithoutChange;
  for (Instruction* interface_var : interface_vars) {
    uint32_t location, component;
    if (!GetVariableLocation(interface_var, &location)) continue;
    if (!GetVariableComponent(interface_var, &component)) component = 0;

    Instruction* interface_var_type = GetTypeOfVariable(interface_var);
    uint32_t extra_array_length = 0;
    if (HasExtraArrayness(interface_var)) {
      extra_array_length = GetArrayLength(context()->get_def_use_mgr(), interface_var_type);
      interface_var_type =
          GetArrayElementType(context()->get_def_use_mgr(), interface_var_type);
    }
    if (interface_var_type->opcode() != SpvOpTypeArray &&
        interface_var_type->opcode() != SpvOpTypeMatrix) {
      continue;
    }

    if (!FlattenInterfaceVariable(interface_var, interface_var_type,
                                  location, component, extra_array_length)) {
      return Pass::Status::Failure;
    }
    status = Pass::Status::SuccessWithChange;
  }

  return status;
}

}  // namespace opt
}  // namespace spvtools
