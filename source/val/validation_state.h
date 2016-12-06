// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef LIBSPIRV_VAL_VALIDATIONSTATE_H_
#define LIBSPIRV_VAL_VALIDATIONSTATE_H_

#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "assembly_grammar.h"
#include "diagnostic.h"
#include "enum_set.h"
#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"
#include "spirv_definition.h"
#include "val/function.h"
#include "val/instruction.h"

namespace libspirv {

/// This enum represents the sections of a SPIRV module. See section 2.4
/// of the SPIRV spec for additional details of the order. The enumerant values
/// are in the same order as the vector returned by GetModuleOrder
enum ModuleLayoutSection {
  kLayoutCapabilities,          /// < Section 2.4 #1
  kLayoutExtensions,            /// < Section 2.4 #2
  kLayoutExtInstImport,         /// < Section 2.4 #3
  kLayoutMemoryModel,           /// < Section 2.4 #4
  kLayoutEntryPoint,            /// < Section 2.4 #5
  kLayoutExecutionMode,         /// < Section 2.4 #6
  kLayoutDebug1,                /// < Section 2.4 #7 > 1
  kLayoutDebug2,                /// < Section 2.4 #7 > 2
  kLayoutAnnotations,           /// < Section 2.4 #8
  kLayoutTypes,                 /// < Section 2.4 #9
  kLayoutFunctionDeclarations,  /// < Section 2.4 #10
  kLayoutFunctionDefinitions    /// < Section 2.4 #11
};

/// This class manages the state of the SPIR-V validation as it is being parsed.
class ValidationState_t {
 public:
  ValidationState_t(const spv_const_context context);

  /// Returns the context
  spv_const_context context() const { return context_; }

  /// Forward declares the id in the module
  spv_result_t ForwardDeclareId(uint32_t id);

  /// Removes a forward declared ID if it has been defined
  spv_result_t RemoveIfForwardDeclared(uint32_t id);

  /// Registers an ID as a forward pointer
  spv_result_t RegisterForwardPointer(uint32_t id);

  /// Returns whether or not an ID is a forward pointer
  bool IsForwardPointer(uint32_t id) const;

  /// Assigns a name to an ID
  void AssignNameToId(uint32_t id, std::string name);

  /// Returns a string representation of the ID in the format <id>[Name] where
  /// the <id> is the numeric valid of the id and the Name is a name assigned by
  /// the OpName instruction
  std::string getIdName(uint32_t id) const;

  /// Accessor function for ID bound.
  uint32_t getIdBound() const;

  /// Mutator function for ID bound.
  void setIdBound(uint32_t bound);

  /// Like getIdName but does not display the id if the \p id has a name
  std::string getIdOrName(uint32_t id) const;

  /// Returns the number of ID which have been forward referenced but not
  /// defined
  size_t unresolved_forward_id_count() const;

  /// Returns a vector of unresolved forward ids.
  std::vector<uint32_t> UnresolvedForwardIds() const;

  /// Returns true if the id has been defined
  bool IsDefinedId(uint32_t id) const;

  /// Increments the instruction count. Used for diagnostic
  int increment_instruction_count();

  /// Returns the current layout section which is being processed
  ModuleLayoutSection current_layout_section() const;

  /// Increments the module_layout_order_section_
  void ProgressToNextLayoutSectionOrder();

  /// Determines if the op instruction is part of the current section
  bool IsOpcodeInCurrentLayoutSection(SpvOp op);

  libspirv::DiagnosticStream diag(spv_result_t error_code) const;

  /// Returns the function states
  std::deque<Function>& functions();

  /// Returns the function states
  Function& current_function();

  /// Returns true if the called after a function instruction but before the
  /// function end instruction
  bool in_function_body() const;

  /// Returns true if called after a label instruction but before a branch
  /// instruction
  bool in_block() const;

  /// Returns a list of entry point function ids
  std::vector<uint32_t>& entry_points() { return entry_points_; }
  const std::vector<uint32_t>& entry_points() const { return entry_points_; }

  /// Registers the capability and its dependent capabilities
  void RegisterCapability(SpvCapability cap);

  /// Registers the function in the module. Subsequent instructions will be
  /// called against this function
  spv_result_t RegisterFunction(uint32_t id, uint32_t ret_type_id,
                                SpvFunctionControlMask function_control,
                                uint32_t function_type_id);

  /// Register a function end instruction
  spv_result_t RegisterFunctionEnd();

  /// Returns true if the capability is enabled in the module.
  bool HasCapability(SpvCapability cap) const {
    return module_capabilities_.Contains(cap);
  }

  /// Returns true if any of the capabilities are enabled, or if the given
  /// capabilities is the empty set.
  bool HasAnyOf(const libspirv::CapabilitySet& capabilities) const;

  /// Sets the addressing model of this module (logical/physical).
  void set_addressing_model(SpvAddressingModel am);

  /// Returns the addressing model of this module, or Logical if uninitialized.
  SpvAddressingModel addressing_model() const;

  /// Sets the memory model of this module.
  void set_memory_model(SpvMemoryModel mm);

  /// Returns the memory model of this module, or Simple if uninitialized.
  SpvMemoryModel memory_model() const;

  AssemblyGrammar& grammar() { return grammar_; }

  /// Registers the instruction
  void RegisterInstruction(const spv_parsed_instruction_t& inst);

  /// Finds id's def, if it exists.  If found, returns the definition otherwise
  /// nullptr
  const Instruction* FindDef(uint32_t id) const;

  /// Finds id's def, if it exists.  If found, returns the definition otherwise
  /// nullptr
  Instruction* FindDef(uint32_t id);

  /// Returns a deque of instructions in the order they appear in the binary
  const std::deque<Instruction>& ordered_instructions() {
    return ordered_instructions_;
  }

  /// Returns a map of instructions mapped by their result id
  const std::unordered_map<uint32_t, Instruction*>& all_definitions() const {
    return all_definitions_;
  }

  /// Returns a vector containing the Ids of instructions that consume the given
  /// SampledImage id.
  std::vector<uint32_t> getSampledImageConsumers(uint32_t id) const;

  /// Records cons_id as a consumer of sampled_image_id.
  void RegisterSampledImageConsumer(uint32_t sampled_image_id,
                                    uint32_t cons_id);

  /// Returns the number of Global Variables
  uint32_t num_global_vars() { return num_global_vars_; }

  /// Returns the number of Local Variables
  uint32_t num_local_vars() { return num_local_vars_; }

  /// Increments the number of Global Variables
  void incrementNumGlobalVars() { ++num_global_vars_; }

  /// Increments the number of Local Variables
  void incrementNumLocalVars() { ++num_local_vars_; }

  /// Sets the struct nesting depth for a given struct ID
  void set_struct_nesting_depth(uint32_t id, uint32_t depth) {
    struct_nesting_depth_[id] = depth;
  }

  /// Returns the nesting depth of a given structure ID
  uint32_t struct_nesting_depth(uint32_t id) {
    return struct_nesting_depth_[id];
  }

 private:
  ValidationState_t(const ValidationState_t&);

  const spv_const_context context_;

  /// Tracks the number of instructions evaluated by the validator
  int instruction_counter_;

  /// IDs which have been forward declared but have not been defined
  std::unordered_set<uint32_t> unresolved_forward_ids_;

  /// IDs that have been declared as forward pointers.
  std::unordered_set<uint32_t> forward_pointer_ids_;

  /// Stores a vector of instructions that use the result of a given
  /// OpSampledImage instruction.
  std::unordered_map<uint32_t, std::vector<uint32_t>> sampled_image_consumers_;

  /// A map of operand IDs and their names defined by the OpName instruction
  std::unordered_map<uint32_t, std::string> operand_names_;

  /// The section of the code being processed
  ModuleLayoutSection current_layout_section_;

  /// A list of functions in the module
  std::deque<Function> module_functions_;

  /// The capabilities available in the module
  libspirv::CapabilitySet
      module_capabilities_;  /// Module's declared capabilities.

  /// List of all instructions in the order they appear in the binary
  std::deque<Instruction> ordered_instructions_;

  /// Instructions that can be referenced by Ids
  std::unordered_map<uint32_t, Instruction*> all_definitions_;

  /// IDs that are entry points, ie, arguments to OpEntryPoint.
  std::vector<uint32_t> entry_points_;

  /// ID Bound from the Header
  uint32_t id_bound_;

  /// Number of Global Variables (Storage Class other than 'Function')
  uint32_t num_global_vars_;

  /// Number of Local Variables ('Function' Storage Class)
  uint32_t num_local_vars_;

  /// Structure Nesting Depth
  std::unordered_map<uint32_t, uint32_t> struct_nesting_depth_;

  AssemblyGrammar grammar_;

  SpvAddressingModel addressing_model_;
  SpvMemoryModel memory_model_;

  /// NOTE: See correspoding getter functions
  bool in_function_;
};

}  /// namespace libspirv

#endif  /// LIBSPIRV_VAL_VALIDATIONSTATE_H_
