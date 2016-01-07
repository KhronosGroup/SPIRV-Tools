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

// This enum represents the sections of a SPIRV module. The MODULE section
// contains instructions who's scope spans the entire module. The FUNCTION
// section includes SPIRV function and function definitions
enum class ModuleLayoutSection {
  kModule,    // < Module scope instructions are executed
  kFunction,  // < Function scope instructions are executed
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

  // The stage which is being processed by the validation. Partially based on
  // Section 2.4. Logical Layout of a Module
  uint32_t module_layout_order_stage_;

  // The section of the code being processed
  ModuleLayoutSection current_layout_stage_;
};
}

#endif
