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

#ifndef LIBSPIRV_VALIDATE_TYPES_H_
#define LIBSPIRV_VALIDATE_TYPES_H_

#include "binary.h"
#include "diagnostic.h"
#include "libspirv/libspirv.h"

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace libspirv {

// This enum represents the sections of a SPIRV module. See section 2.4
// of the SPIRV spec for additional details of the order. The enumerant values
// are in the same order as the vector returned by GetModuleOrder
enum ModuleLayoutSection {
  kLayoutCapabilities,          // < Section 2.4 #1
  kLayoutExtensions,            // < Section 2.4 #2
  kLayoutExtInstImport,         // < Section 2.4 #3
  kLayoutMemoryModel,           // < Section 2.4 #4
  kLayoutEntryPoint,            // < Section 2.4 #5
  kLayoutExecutionMode,         // < Section 2.4 #6
  kLayoutDebug1,                // < Section 2.4 #7 > 1
  kLayoutDebug2,                // < Section 2.4 #7 > 2
  kLayoutAnnotations,           // < Section 2.4 #8
  kLayoutTypes,                 // < Section 2.4 #9
  kLayoutFunctionDeclarations,  // < Section 2.4 #10
  kLayoutFunctionDefinitions    // < Section 2.4 #11
};

enum class FunctionDecl {
  kFunctionDeclUnknown,      // < Unknown function declaration
  kFunctionDeclDeclaration,  // < Function declaration
  kFunctionDeclDefinition    // < Function definition
};

class ValidationState_t;

class Functions {
 public:
  Functions(ValidationState_t& module);

  // Registers the function in the module. Subsequent instructions will be
  // called against this function
  spv_result_t RegisterFunction(uint32_t id, uint32_t ret_type_id,
                                uint32_t function_control,
                                uint32_t function_type_id);

  // Registers a function parameter in the current function
  spv_result_t RegisterFunctionParameter(uint32_t id, uint32_t type_id);

  // Register a function end instruction
  spv_result_t RegisterFunctionEnd();

  // Sets the declaration type of the current function
  spv_result_t RegisterSetFunctionDeclType(FunctionDecl type);

  // Registers a block in the current function. Subsequent block instructions
  // will target this block
  // @param id The ID of the label of the block
  spv_result_t RegisterBlock(uint32_t id);

  // Registers a variable in the current block
  spv_result_t RegisterBlockVariable(uint32_t type_id, uint32_t id,
                                     SpvStorageClass storage, uint32_t init_id);

  spv_result_t RegisterBlockLoopMerge(uint32_t merge_id, uint32_t continue_id,
                                      SpvLoopControlMask control);

  spv_result_t RegisterBlockSelectionMerge(uint32_t merge_id,
                                           SpvSelectionControlMask control);

  // Registers the end of the block
  spv_result_t RegisterBlockEnd();

  // Returns the number of blocks in the current function being parsed
  size_t get_block_count();

  // Retuns true if the called after a function instruction but before the
  // function end instruction
  bool in_function_body() const;

  // Returns true if called after a label instruction but before a branch
  // instruction
  bool in_block() const;

  libspirv::DiagnosticStream diag(spv_result_t error_code) const;

 private:
  // Parent module
  ValidationState_t& module_;

  // Funciton IDs in a module
  std::vector<uint32_t> id_;

  // OpTypeFunction IDs of each of the id_ functions
  std::vector<uint32_t> type_id_;

  // The type of declaration of each function
  std::vector<FunctionDecl> declaration_type_;

  // TODO(umar): Probably needs better abstractions
  // The beginning of the block of functions
  std::vector<std::vector<uint32_t>> block_ids_;

  // The variable IDs of the functions
  std::vector<std::vector<uint32_t>> variable_ids_;

  // The function parameter ids of the functions
  std::vector<std::vector<uint32_t>> parameter_ids_;

  bool in_function_;
  bool in_block_;
  FunctionDecl function_stage;
};

class ValidationState_t {
 public:
  ValidationState_t(spv_diagnostic* diag, uint32_t options);

  // Defines the \p id for the module
  spv_result_t defineId(uint32_t id);

  // Forward declares the id in the module
  spv_result_t forwardDeclareId(uint32_t id);

  // Removes a forward declared ID if it has been defined
  spv_result_t removeIfForwardDeclared(uint32_t id);

  // Assigns a name to an ID
  void assignNameToId(uint32_t id, std::string name);

  // Returns a string representation of the ID in the format <id>[Name] where
  // the <id> is the numeric valid of the id and the Name is a name assigned by
  // the OpName instruction
  std::string getIdName(uint32_t id) const;

  // Returns the number of ID which have been forward referenced but not defined
  size_t unresolvedForwardIdCount() const;

  // Returns a list of unresolved forward ids.
  std::vector<uint32_t> unresolvedForwardIds() const;

  // Returns true if the id has been defined
  bool isDefinedId(uint32_t id) const;

  // Returns true if an spv_validate_options_t option is enabled in the
  // validation instruction
  bool is_enabled(spv_validate_options_t flag) const;

  // Increments the instruction count. Used for diagnostic
  int incrementInstructionCount();

  // Returns the current layout section which is being processed
  ModuleLayoutSection getLayoutStage() const;

  // Increments the module_layout_order_stage_
  void progressToNextLayoutStageOrder();

  // Determines if the op instruction is part of the current stage
  bool isOpcodeInCurrentLayoutStage(SpvOp op);

  libspirv::DiagnosticStream diag(spv_result_t error_code) const;

  // Returns the function states
  Functions& get_functions();

  // Retuns true if the called after a function instruction but before the
  // function end instruction
  bool in_function_body() const;

  // Returns true if called after a label instruction but before a branch
  // instruction
  bool in_block() const;

 private:
  spv_diagnostic* diagnostic_;
  // Tracks the number of instructions evaluated by the validator
  int instruction_counter_;

  // All IDs which have been defined
  std::unordered_set<uint32_t> defined_ids_;

  // IDs which have been forward declared but have not been defined
  std::unordered_set<uint32_t> unresolved_forward_ids_;

  // Validation options to determine the passes to execute
  uint32_t validation_flags_;

  std::map<uint32_t, std::string> operand_names_;

  // The section of the code being processed
  ModuleLayoutSection current_layout_stage_;

  Functions module_functions_;

  std::vector<SpvCapability> module_capabilities_;
};
}

#endif
