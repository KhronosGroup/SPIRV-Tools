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

#ifndef SOURCE_OPT_INTERFACE_VAR_SROA_H_
#define SOURCE_OPT_INTERFACE_VAR_SROA_H_

#include <unordered_set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// A struct for the information of an interface variable's location, component,
// extra arrayness, and whether it is an input or output interface variable.
// Note that interface variables of the tessellation shaders have the extra
// arrayness as the first dimension of the array.
//
// For example, when we generate the following HLSL code to SPIR-V
//
//  struct HS_OUTPUT {
//    float3 pos[5] : POSITION;
//    ...
//  };
//  ...
//  [patchconstantfunc("main_hs_patch")]
//  [outputcontrolpoints(3)]
//  HS_OUTPUT main_hs(...) { ... }
//  ...
//
//  HS_PATCH_OUTPUT main_hs_patch(const OutputPatch<HS_OUTPUT, 3> patch) {
//  ...
//  }
//
// because of `OutputPatch<HS_OUTPUT, 3> patch`, we add the extra arrayness 3
// for `float3 pos[5]`. In SPIR-V, its type will be
// `%_ptr_Output__arr__arr_v3float_uint_5_uint_3`.
struct InterfaceVariableInfo {
  uint32_t location;
  uint32_t component;
  uint32_t extra_arrayness;
  bool is_input_var;

  bool operator==(const InterfaceVariableInfo& another) const {
    return another.location == location && another.component == component &&
           another.is_input_var == is_input_var;
  }
};

// See optimizer.hpp for documentation.
//
// Note that the current implementation of this pass covers only store, load,
// access chain instructions for the interface variables. Supporting other types
// of instructions is a future work.
class InterfaceVariableScalarReplacement : public Pass {
 public:
  // Hashing functor for InterfaceVariableInfo.
  struct InterfaceVariableInfoHash {
    size_t operator()(const InterfaceVariableInfo& info) const {
      return std::hash<uint32_t>()(info.location) ^
             std::hash<uint32_t>()(info.component) ^
             std::hash<uint32_t>()(static_cast<uint32_t>(info.is_input_var));
    }
  };

  using SetOfInterfaceVariableLocationInfo =
      std::unordered_set<InterfaceVariableInfo, InterfaceVariableInfoHash>;

  explicit InterfaceVariableScalarReplacement(
      const std::vector<InterfaceVariableInfo>& interface_variable_info)
      : interface_variable_info_(interface_variable_info.begin(),
                                 interface_variable_info.end()) {}

  const char* name() const override {
    return "interface-variable-scalar-replacement";
  }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDecorations | IRContext::kAnalysisDefUse |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // A struct containing components of a composite variable. If the composite
  // consists of multiple or recursive components, |component_variable| is
  // nullptr and |nested_composite_components| keeps the components. If it has a
  // single component, |nested_composite_components| is empty and
  // |component_variable| is the component. Note that each element of
  // |nested_composite_components| has the NestedCompositeComponents struct as
  // its type that can recursively keep the components.
  struct NestedCompositeComponents {
    NestedCompositeComponents() : component_variable(nullptr) {}

    bool HasMultipleComponents() const {
      return !nested_composite_components.empty();
    }

    const std::vector<NestedCompositeComponents>& GetComponents() const {
      return nested_composite_components;
    }

    void AddComponent(const NestedCompositeComponents& component) {
      nested_composite_components.push_back(component);
    }

    Instruction* GetComponentVariable() const { return component_variable; }

    void SetSingleComponentVariable(Instruction* var) {
      component_variable = var;
    }

   private:
    std::vector<NestedCompositeComponents> nested_composite_components;
    Instruction* component_variable;
  };

  // Checks all interface variables that have Location and Component decorations
  // that are a part of |interface_variable_info_| and collects the mapping from
  // interface variable ids to all InterfaceVariableInfo.
  void CollectInterfaceVariablesToFlatten(
      std::unordered_map<uint32_t, InterfaceVariableInfo>*
          interface_var_ids_to_interface_var_info);

  // Returns true if InterfaceVariableInfo of the variable whose id |var_id|
  // exists in |interface_variable_info_|. Returns the InterfaceVariableInfo
  // via |interface_var_info|. If |is_input_var| is true, the variable is an
  // input variable.
  bool FindTargetInterfaceVariableInfo(
      uint32_t var_id, bool is_input_var,
      InterfaceVariableInfo* interface_var_info);

  // Finds a Location BuiltIn decoration of |var_id| and returns it via
  // |location|. Returns true whether the location exists or not.
  bool GetVariableLocation(uint32_t var_id, uint32_t* location);

  // Finds a Component BuiltIn decoration of |var_id| and returns it via
  // |component|. Returns true whether the component exists or not.
  bool GetVariableComponent(uint32_t var_id, uint32_t* component);

  // Returns the interface variable instruction whose result id is
  // |interface_var_id|.
  Instruction* GetInterfaceVariable(uint32_t interface_var_id);

  // Returns the type of |var| as an instruction.
  Instruction* GetTypeOfVariable(Instruction* var);

  // Flattens an interface variable |interface_var| whose type is
  // |interface_var_type| and returns whether it succeeds or not.
  // |interface_var_info| keeps the information of location, component, extra
  // arrayness, and whether it is an input or output interface variable.
  bool FlattenInterfaceVariable(
      Instruction* interface_var, Instruction* interface_var_type,
      const InterfaceVariableInfo& interface_var_info);

  // Creates flattened variables with the storage classe |storage_class| for the
  // interface variable whose type is |interface_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to all the
  // flattened variables.
  NestedCompositeComponents CreateFlattenedInterfaceVarsForReplacement(
      Instruction* interface_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Creates flattened variables with the storage classe |storage_class| for the
  // interface variable whose type is OpTypeArray |interface_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to all the
  // flattened variables.
  NestedCompositeComponents CreateFlattenedInterfaceVarsForArray(
      Instruction* interface_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Creates flattened variables with the storage classe |storage_class| for the
  // interface variable whose type is OpTypeMatrix |interface_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to all the
  // flattened variables.
  NestedCompositeComponents CreateFlattenedInterfaceVarsForMatrix(
      Instruction* interface_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Recursively adds Location and Component decorations to variables in
  // |flattened_vars| with |location| and |component|. Increases |location| by
  // one after it actually adds Location and Component decorations for a
  // variable.
  void AddLocationAndComponentDecorations(
      const NestedCompositeComponents& flattened_vars, uint32_t* location,
      uint32_t component);

  // Replaces the interface variable |interface_var| with
  // |flattened_interface_vars| and returns whether it succeeds or not.
  // |extra_arrayness| is the extra arrayness of the interface variable.
  // |flattened_interface_vars| contains the nested flattened variables to
  // replace the interface variable with.
  bool ReplaceInterfaceVarWithFlattenedVars(
      Instruction* interface_var, uint32_t extra_arrayness,
      const NestedCompositeComponents& flattened_interface_vars);

  // Replaces |interface_var| in the operands of instructions
  // |interface_var_users| with |flattened_interface_vars|. This is a recursive
  // method and |interface_var_component_indices| is used to specify which
  // recursive component of |interface_var| is replaced. Returns composite
  // construct instructions to be replaced with load instructions of
  // |interface_var_users| via |loads_to_composites|. Returns composite
  // construct instructions to be replaced with load instructions of access
  // chain instructions in |interface_var_users| via
  // |loads_for_access_chain_to_composites|.
  bool ReplaceInterfaceVarComponentsWithFlattenedVars(
      Instruction* interface_var,
      const std::vector<Instruction*>& interface_var_users,
      const NestedCompositeComponents& flattened_interface_vars,
      std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces |interface_var| in the operands of instructions
  // |interface_var_users| with |components| that is a vector of components for
  // the interface variable |interface_var|. This is a recursive method and
  // |interface_var_component_indices| is used to specify which recursive
  // component of |interface_var| is replaced. Returns composite construct
  // instructions to be replaced with load instructions of |interface_var_users|
  // via |loads_to_composites|. Returns composite construct instructions to be
  // replaced with load instructions of access chain instructions in
  // |interface_var_users| via |loads_for_access_chain_to_composites|.
  bool ReplaceMultipleComponentsOfInterfaceVarWithFlattenedVars(
      Instruction* interface_var,
      const std::vector<Instruction*>& interface_var_users,
      const std::vector<NestedCompositeComponents>& components,
      std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces a component of |interface_var| that is used as an operand of
  // instruction |interface_var_user| with |flattened_var|.
  // |interface_var_component_indices| is a vector of recursive indices for
  // which recursive component of |interface_var| is replaced. If
  // |interface_var_user| is a load, returns the component value via
  // |loads_to_component_values|. If |interface_var_user| is an access chain,
  // returns the component value for loads of |interface_var_user| via
  // |loads_for_access_chain_to_component_values|.
  bool ReplaceInterfaceVarComponentWithFlattenedVar(
      Instruction* interface_var, Instruction* interface_var_user,
      Instruction* flattened_var,
      const std::vector<uint32_t>& interface_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_component_values,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_component_values);

  // When |load_or_store| is a load or store instruction for an interface
  // variable, creates a load or store for one of its components. Since
  // |flattened_var| is a component of the interface variable, the new load or
  // store has |flattened_var| or the access chain to |extra_array_index| th
  // component of |flattened_var| (when |extra_array_index| is not nullptr) as
  // Pointer operand.
  void CreateLoadOrStoreToFlattenedInterfaceVar(
      Instruction* load_or_store,
      const std::vector<uint32_t>& interface_var_component_indices,
      Instruction* flattened_var, const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // When |access_chain| is an access chain to an interface variable and
  // |load_or_store| is a load or store instruction for |access_chain|, creates
  // a load or store for one of its components. Since |flattened_var| is
  // a component of the interface variable, the new load or store has the access
  // chain to a component of |flattened_var| as Pointer operand. For example,
  //
  // Before:
  //  %ptr = OpAccessChain %ptr_type %var %idx
  //  %value = OpLoad %type %ptr
  //
  // After:
  //  %flattened_ptr = OpAccessChain %flattened_ptr_type %flattened_var %idx
  //  %flattened_value = OpLoad %flattened_type %flattened_ptr
  //  ..
  //  %value = OpCompositeConstruct %type %flattened_value ..
  void CreateLoadOrStoreToFlattenedInterfaceVarAccessChain(
      Instruction* access_chain, Instruction* load_or_store,
      const std::vector<uint32_t>& interface_var_component_indices,
      Instruction* flattened_var,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // When |store_to_interface_var| stores a value to an interface variable,
  // creates instructions to store one of the value's components to
  // |ptr_to_flattened_var|. |interface_var_component_indices| contains
  // recursive indices to the component of the value. When |extra_array_index|
  // is not nullptr, the first index of the component must be
  // |extra_array_index|.
  //
  // Store to interface var:
  //   OpStore %interface_var %value
  //
  // Store to flattened var:
  //   %composite_extract = OpCompositeExtract %type %value
  //                                           <interface_var_component_indices>
  //   OpStore %ptr_to_flattened_var %composite_extract
  void CreateStoreToFlattenedInterfaceVar(
      uint32_t component_type_id, Instruction* store_to_interface_var,
      const std::vector<uint32_t>& interface_var_component_indices,
      Instruction* ptr_to_flattened_var, const uint32_t* extra_array_index);

  // Creates new OpCompositeExtract with |type_id| for Result Type,
  // |composite_id| for Composite operand, and |indexes| for Indexes operands.
  // If |extra_first_index| is not nullptr, uses it as the first Indexes
  // operand.
  Instruction* CreateCompositeExtract(uint32_t type_id, uint32_t composite_id,
                                      const std::vector<uint32_t>& indexes,
                                      const uint32_t* extra_first_index);

  // Creates a new OpLoad whose Result Type is |type_id| and Pointer operand is
  // |ptr|. Inserts the new instruction before |insert_before|.
  Instruction* CreateLoad(uint32_t type_id, Instruction* ptr,
                          Instruction* insert_before);

  // Clones an annotation instruction |annotation_inst| and sets the target
  // operand of the new annotation instruction as |var_id|.
  void CloneAnnotationForVariable(Instruction* annotation_inst,
                                  uint32_t var_id);

  // Replaces the interface variable |interface_var| in the operands of the
  // entry point |entry_point| with |flattened_var_id|. If it cannot find
  // |interface_var| from the operands of the entry point |entry_point|, adds
  // |flattened_var_id| as an operand of the entry point |entry_point|.
  bool ReplaceInterfaceVarInEntryPoint(Instruction* interface_var,
                                       Instruction* entry_point,
                                       uint32_t flattened_var_id);

  // Creates an access chain instruction whose Base operand is |var| and Indexes
  // operand is |index|. |component_type_id| is the id of the type instruction
  // that is the type of component. Inserts the new access chain before
  // |insert_before|.
  Instruction* CreateAccessChainWithIndex(uint32_t component_type_id,
                                          Instruction* var, uint32_t index,
                                          Instruction* insert_before);

  // Returns the pointee type of the type of variable |var|.
  uint32_t GetPointeeTypeIdOfVar(Instruction* var);

  // Replaces the access chain |access_chain| and its users with a new access
  // chain that points |flattened_var| as the Base operand having
  // |interface_var_component_indices| as Indexes operands and users of the new
  // access chain. When some of the users are load instructions, returns the
  // original load instruction to the new instruction that loads a component of
  // the original load value via |loads_to_component_values|.
  void ReplaceAccessChainWithFlattenedVar(
      Instruction* access_chain,
      const std::vector<uint32_t>& interface_var_component_indices,
      Instruction* flattened_var,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // Assuming that |access_chain| is an access chain instruction whose Base
  // operand is |base_access_chain|, replaces the operands of |access_chain|
  // with operands of |base_access_chain| and Indexes operands of
  // |access_chain|.
  void UseBaseAccessChainForAccessChain(Instruction* access_chain,
                                        Instruction* base_access_chain);

  // Creates composite construct instructions for load instructions that are the
  // keys of |loads_to_component_values| if no such composite construct
  // instructions exist. Adds a component of the composite as an operand of the
  // created composite construct instruction. Each value of
  // |loads_to_component_values| is the component. Returns the created composite
  // construct instructions using |loads_to_composites|. |depth_to_component| is
  // the number of recursive access steps to get the component from the
  // composite.
  void AddComponentsToCompositesForLoads(
      const std::unordered_map<Instruction*, Instruction*>&
          loads_to_component_values,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      uint32_t depth_to_component);

  // Creates a composite construct instruction for a component of the value of
  // instruction |load| in |depth_to_component| th recursive depth and inserts
  // it after |load|.
  Instruction* CreateCompositeConstructForComponentOfLoad(
      Instruction* load, uint32_t depth_to_component);

  // Creates a new access chain instruction that points to instruction |var|
  // whose type is the instruction with result id |var_type_id|. The new access
  // chain will have the same Indexes operands as |access_chain|. Returns the
  // type id of the component that is pointed by the new access chain via
  // |component_type_id|.
  Instruction* CreateAccessChainToVar(uint32_t var_type_id, Instruction* var,
                                      Instruction* access_chain,
                                      uint32_t* component_type_id);

  // Returns the result id of OpTypeArray instrunction whose Element Type
  // operand is |elem_type_id| and Length operand is |array_length|.
  uint32_t GetArrayType(uint32_t elem_type_id, uint32_t array_length);

  // Returns the result id of OpTypePointer instrunction whose Type
  // operand is |type_id| and Storage Class operand is |storage_class|.
  uint32_t GetPointerType(uint32_t type_id, SpvStorageClass storage_class);

  // Kills an instrunction |inst| and its users.
  void KillInstructionAndUsers(Instruction* inst);

  // Kills a vector of instrunctions |insts| and their users.
  void KillInstructionsAndUsers(const std::vector<Instruction*>& insts);

  // Kills all OpDecorate instructions for Location and Component of the
  // variable whose id is |var_id|.
  void KillLocationAndComponentDecorations(uint32_t var_id);

  // A set of InterfaceVariableInfo that includes all locations and
  // components of interface variables to be flattened in this pass.
  SetOfInterfaceVariableLocationInfo interface_variable_info_;

  // A set of interface variable ids that were already removed from operands of
  // the entry point.
  std::unordered_set<uint32_t>
      interface_vars_removed_from_entry_point_operands_;

  // A mapping from ids of new composite construct instructions that load
  // instructions are replaced with to the recursive depth of the component of
  // load that the new component construct instruction is used for.
  std::unordered_map<uint32_t, uint32_t> composite_ids_to_component_depths;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INTERFACE_VAR_SROA_H_
