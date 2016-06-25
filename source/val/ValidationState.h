// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#ifndef LIBSPIRV_VAL_VALIDATIONSTATE_H_
#define LIBSPIRV_VAL_VALIDATIONSTATE_H_

#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "assembly_grammar.h"
#include "diagnostic.h"
#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"
#include "spirv_definition.h"

namespace libspirv {

// Universal Limit of ResultID + 1
static const uint32_t kInvalidId = 0x400000;

// Info about a result ID.
typedef struct spv_id_info_t {
  /// Id value.
  uint32_t id;
  /// Type id, or 0 if no type.
  uint32_t type_id;
  /// Opcode of the instruction defining the id.
  SpvOp opcode;
  /// Binary words of the instruction defining the id.
  std::vector<uint32_t> words;
} spv_id_info_t;

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

class Function;

/// This class manages the state of the SPIR-V validation as it is being parsed.
class ValidationState_t {
 public:
  ValidationState_t(spv_diagnostic* diagnostic,
                    const spv_const_context context);

  /// Forward declares the id in the module
  spv_result_t ForwardDeclareId(uint32_t id);

  /// Removes a forward declared ID if it has been defined
  spv_result_t RemoveIfForwardDeclared(uint32_t id);

  /// Assigns a name to an ID
  void AssignNameToId(uint32_t id, std::string name);

  /// Returns a string representation of the ID in the format <id>[Name] where
  /// the <id> is the numeric valid of the id and the Name is a name assigned by
  /// the OpName instruction
  std::string getIdName(uint32_t id) const;

  /// Like getIdName but does not display the id if the \p id has a name
  std::string getIdOrName(uint32_t id) const;

  /// Returns the number of ID which have been forward referenced but not
  /// defined
  size_t unresolved_forward_id_count() const;

  /// Returns a list of unresolved forward ids.
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
  std::list<Function>& functions();

  /// Returns the function states
  Function& current_function();

  /// Returns true if the called after a function instruction but before the
  /// function end instruction
  bool in_function_body() const;

  /// Returns true if called after a label instruction but before a branch
  /// instruction
  bool in_block() const;

  /// Keeps track of ID definitions and uses.
  class UseDefTracker {
   public:
    void AddDef(const spv_id_info_t& def) { defs_[def.id] = def; }

    void AddUse(uint32_t id) { uses_.insert(id); }

    /// Finds id's def, if it exists.  If found, returns <true, def>. Otherwise,
    /// returns <false, something>.
    std::pair<bool, spv_id_info_t> FindDef(uint32_t id) const {
      if (defs_.count(id) == 0) {
        return std::make_pair(false, spv_id_info_t{});
      } else {
        /// We are in a const function, so we cannot use defs.operator[]().
        /// Luckily we know the key exists, so defs_.at() won't throw an
        /// exception.
        return std::make_pair(true, defs_.at(id));
      }
    }

    /// Returns uses of IDs lacking defs.
    std::unordered_set<uint32_t> FindUsesWithoutDefs() const {
      auto diff = uses_;
      for (const auto d : defs_) diff.erase(d.first);
      return diff;
    }

   private:
    std::unordered_set<uint32_t> uses_;
    std::unordered_map<uint32_t, spv_id_info_t> defs_;
  };

  UseDefTracker& usedefs() { return usedefs_; }
  const UseDefTracker& usedefs() const { return usedefs_; }

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
  bool has_capability(SpvCapability cap) const;

  /// Returns true if any of the capabilities are enabled.  Always true for
  /// capabilities==0.
  bool HasAnyOf(spv_capability_mask_t capabilities) const;

  /// Sets the addressing model of this module (logical/physical).
  void set_addressing_model(SpvAddressingModel am);

  /// Returns the addressing model of this module, or Logical if uninitialized.
  SpvAddressingModel addressing_model() const;

  /// Sets the memory model of this module.
  void set_memory_model(SpvMemoryModel mm);

  /// Returns the memory model of this module, or Simple if uninitialized.
  SpvMemoryModel memory_model() const;

  AssemblyGrammar& grammar() { return grammar_; }

 private:
  spv_diagnostic* diagnostic_;
  /// Tracks the number of instructions evaluated by the validator
  int instruction_counter_;

  /// IDs which have been forward declared but have not been defined
  std::unordered_set<uint32_t> unresolved_forward_ids_;

  /// A map of operand IDs and their names defined by the OpName instruction
  std::map<uint32_t, std::string> operand_names_;

  /// The section of the code being processed
  ModuleLayoutSection current_layout_section_;

  /// A list of functions in the module
  std::list<Function> module_functions_;

  /// Mask of the capabilities available in the module
  spv_capability_mask_t
      module_capabilities_;  /// Module's declared capabilities.

  /// Definitions and uses of all the IDs in the module.
  UseDefTracker usedefs_;

  /// IDs that are entry points, ie, arguments to OpEntryPoint.
  std::vector<uint32_t> entry_points_;

  AssemblyGrammar grammar_;

  SpvAddressingModel addressing_model_;
  SpvMemoryModel memory_model_;

  /// NOTE: See correspoding getter functions
  bool in_function_;
};

}  /// namespace libspirv

#endif  /// LIBSPIRV_VAL_VALIDATIONSTATE_H_
