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

#ifndef SOURCE_OPT_FLATTEN_ARRAY_MATRIX_STAGE_VAR_H_
#define SOURCE_OPT_FLATTEN_ARRAY_MATRIX_STAGE_VAR_H_

#include <unordered_set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// A struct for the information of a stage variable's location, component,
// extra arrayness, and whether it is an input or output stage variable. Note
// that stage variables of the tessellation shaders have the extra arrayness.
struct StageVariableInfo {
  uint32_t location;
  uint32_t component;
  uint32_t extra_arrayness;
  bool is_input_var;

  bool operator==(const StageVariableInfo& another) const {
    return another.location == location && another.component == component &&
           another.is_input_var == is_input_var;
  }
};

class FlattenArrayMatrixStageVariable : public Pass {
 public:
  // Hashing functor for StageVariableInfo.
  struct StageVariableInfoHash {
    size_t operator()(const StageVariableInfo& info) const {
      return std::hash<uint32_t>()(info.location) ^
             std::hash<uint32_t>()(info.component) ^
             std::hash<uint32_t>()(static_cast<uint32_t>(info.is_input_var));
    }
  };

  using SetOfStageVariableLocationInfo =
      std::unordered_set<StageVariableInfo, StageVariableInfoHash>;

  explicit FlattenArrayMatrixStageVariable(
      const std::vector<StageVariableInfo>& stage_variable_info)
      : stage_variable_info_(stage_variable_info.begin(),
                             stage_variable_info.end()) {}

  const char* name() const override {
    return "flatten-array-matrix-stage-variable";
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

  // Checks all stage variables that have Location and Component decorations
  // that are a part of |stage_variable_info_| and collects the mapping from
  // stage variable ids to all StageVariableInfo.
  void CollectStageVariablesToFlatten(
      std::unordered_map<uint32_t, StageVariableInfo>*
          stage_var_ids_to_stage_var_info);

  // Returns true if variable whose id is |var_id| that has Location decoration
  // with |location| is decorated by a Component decoration and the tuple of
  // |location|, the component, and |is_input_var| is one of
  // |stage_variable_info_|. Also returns true if the variable with id |var_id|
  // has Location decoration |location| but does not have a Component decoration
  // and the tuple of |location|, component 0, and |is_input_var| is one of
  // |stage_variable_info_| (no Component decoration means its component is 0).
  // Otherwise, returns false. If the variable is a target, returns the stage
  // variable information using |stage_var_info|.
  bool IsTargetStageVariable(uint32_t var_id, uint32_t location,
                             bool is_input_var,
                             StageVariableInfo* stage_var_info);

  // Returns the stage variable instruction whose result id is |stage_var_id|.
  Instruction* GetStageVariable(uint32_t stage_var_id);

  // Returns the type of |stage_var| as an instruction. If |has_extra_arrayness|
  // is true, it means the stage variable has the extra arrayness and it has to
  // ignore the first dimension of the array type.
  Instruction* GetTypeOfStageVariable(Instruction* stage_var,
                                      bool has_extra_arrayness);

  // Flattens a stage variable |stage_var| whose type is |stage_var_type| and
  // returns whether it succeeds or not. |stage_var_info| keeps the information
  // of location, component, extra arrayness, and whether it is an input or
  // output stage variable.
  bool FlattenStageVariable(Instruction* stage_var, Instruction* stage_var_type,
                            const StageVariableInfo& stage_var_info);

  // Creates flattened variables with the storage classe |storage_class| for the
  // stage variable whose type is |stage_var_type|. If |extra_array_length| is
  // not zero, adds the extra arrayness to all the flattened variables.
  NestedCompositeComponents CreateFlattenedStageVarsForReplacement(
      Instruction* stage_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Creates flattened variables with the storage classe |storage_class| for the
  // stage variable whose type is OpTypeArray |stage_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to all the
  // flattened variables.
  NestedCompositeComponents CreateFlattenedStageVarsForArray(
      Instruction* stage_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Creates flattened variables with the storage classe |storage_class| for the
  // stage variable whose type is OpTypeMatrix |stage_var_type|. If
  // |extra_array_length| is not zero, adds the extra arrayness to all the
  // flattened variables.
  NestedCompositeComponents CreateFlattenedStageVarsForMatrix(
      Instruction* stage_var_type, SpvStorageClass storage_class,
      uint32_t extra_array_length);

  // Recursively adds Location and Component decorations to variables in
  // |flattened_vars| with |location| and |component|. Increases |location| by
  // one after it actually adds Location and Component decorations for a
  // variable.
  void AddLocationAndComponentDecorations(
      const NestedCompositeComponents& flattened_vars, uint32_t* location,
      uint32_t component);

  // Replaces the stage variable |stage_var| with |flattened_stage_vars| and
  // returns whether it succeeds or not. |extra_arrayness| is the extra
  // arrayness of the stage variable. |flattened_stage_vars| contains the nested
  // flattened variables to replace the stage variable with.
  bool ReplaceStageVarWithFlattenedVars(
      Instruction* stage_var, uint32_t extra_arrayness,
      const NestedCompositeComponents& flattened_stage_vars);

  // Replaces |stage_var| in the operands of instructions |stage_var_users|
  // with |flattened_stage_vars|. This is a recursive method and
  // |stage_var_component_indices| is used to specify which recursive component
  // of |stage_var| is replaced. Returns composite construct instructions to be
  // replaced with load instructions of |stage_var_users| via
  // |loads_to_composites|. Returns composite construct instructions to be
  // replaced with load instructions of access chain instructions in
  // |stage_var_users| via |loads_for_access_chain_to_composites|.
  bool ReplaceStageVarComponentsWithFlattenedVars(
      Instruction* stage_var, const std::vector<Instruction*>& stage_var_users,
      const NestedCompositeComponents& flattened_stage_vars,
      std::vector<uint32_t>& stage_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces |stage_var| in the operands of instructions |stage_var_users|
  // with |components| that is a vector of components for the stage variable
  // |stage_var|. This is a recursive method and |stage_var_component_indices|
  // is used to specify which recursive component of |stage_var| is replaced.
  // Returns composite construct instructions to be replaced with load
  // instructions of |stage_var_users| via |loads_to_composites|. Returns
  // composite construct instructions to be replaced with load instructions of
  // access chain instructions in |stage_var_users| via
  // |loads_for_access_chain_to_composites|.
  bool ReplaceMultipleComponentsOfStageVarWithFlattenedVars(
      Instruction* stage_var, const std::vector<Instruction*>& stage_var_users,
      const std::vector<NestedCompositeComponents>& components,
      std::vector<uint32_t>& stage_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_composites,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_composites);

  // Replaces a component of |stage_var| that is used as an operand of
  // instruction |stage_var_user| with |flattened_var|.
  // |stage_var_component_indices| is a vector of recursive indices for which
  // recursive component of |stage_var| is replaced. If |stage_var_user| is a
  // load, returns the component value via |loads_to_component_values|. If
  // |stage_var_user| is an access chain, returns the component value for loads
  // of |stage_var_user| via |loads_for_access_chain_to_component_values|
  bool ReplaceStageVarComponentWithFlattenedVar(
      Instruction* stage_var, Instruction* stage_var_user,
      Instruction* flattened_var,
      const std::vector<uint32_t>& stage_var_component_indices,
      const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>* loads_to_component_values,
      std::unordered_map<Instruction*, Instruction*>*
          loads_for_access_chain_to_component_values);

  // When |load_or_store| is a load or store instruction for a stage variable,
  // creates a load or store for one of its components. Since |flattened_var| is
  // a component of the stage variable, the new load or store has
  // |flattened_var| or the access chain to |extra_array_index| th component of
  // |flattened_var| (when |extra_array_index| is not nullptr) as Pointer
  // operand.
  void CreateLoadOrStoreToFlattenedStageVar(
      Instruction* load_or_store,
      const std::vector<uint32_t>& stage_var_component_indices,
      Instruction* flattened_var, const uint32_t* extra_array_index,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // When |access_chain| is an access chain to a stage variable and
  // |load_or_store| is a load or store instruction for |access_chain|, creates
  // a load or store for one of its components. Since |flattened_var| is
  // a component of the stage variable, the new load or store has the access
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
  void CreateLoadOrStoreToFlattenedStageVarAccessChain(
      Instruction* access_chain, Instruction* load_or_store,
      const std::vector<uint32_t>& stage_var_component_indices,
      Instruction* flattened_var,
      std::unordered_map<Instruction*, Instruction*>*
          loads_to_component_values);

  // When |store_to_stage_var| stores a value to a stage variable, creates
  // instructions to store one of the value's components to
  // |ptr_to_flattened_var|. |stage_var_component_indices| contains recursive
  // indices to the component of the value. When |extra_array_index| is not
  // nullptr, the first index of the component must be |extra_array_index|.
  //
  // Store to stage var:
  //   OpStore %stage_var %value
  //
  // Store to flattened var:
  //   %composite_extract = OpCompositeExtract %type %value
  //                                           <stage_var_component_indices>
  //   OpStore %ptr_to_flattened_var %composite_extract
  void CreateStoreToFlattenedStageVar(
      uint32_t component_type_id, Instruction* store_to_stage_var,
      const std::vector<uint32_t>& stage_var_component_indices,
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

  // Replaces the stage variable |stage_var| in the operands of the entry point
  // |entry_point| with |flattened_var_id|. If it cannot find |stage_var| from
  // the operands of the entry point |entry_point|, adds |flattened_var_id| as
  // an operand of the entry point |entry_point|.
  bool ReplaceStageVarInEntryPoint(Instruction* stage_var,
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
  // |stage_var_component_indices| as Indexes operands and users of the new
  // access chain. When some of the users are load instructions, returns the
  // original load instruction to the new instruction that loads a component of
  // the original load value via |loads_to_component_values|.
  void ReplaceAccessChainWithFlattenedVar(
      Instruction* access_chain,
      const std::vector<uint32_t>& stage_var_component_indices,
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

  // A set of StageVariableInfo that includes all locations and
  // components of stage variables to be flattened in this pass.
  SetOfStageVariableLocationInfo stage_variable_info_;

  // A set of stage variable ids that were already removed from operands of the
  // entry point.
  std::unordered_set<uint32_t> stage_vars_removed_from_entry_point_operands_;

  // A mapping from ids of new composite construct instructions that load
  // instructions are replaced with to the recursive depth of the component of
  // load that the new component construct instruction is used for.
  std::unordered_map<uint32_t, uint32_t> composite_ids_to_component_depths;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FLATTEN_ARRAY_MATRIX_STAGE_VAR_H_
