// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_MODULE_H_
#define LIBSPIRV_OPT_MODULE_H_

#include <functional>
#include <utility>
#include <vector>

#include "function.h"
#include "instruction.h"

namespace spvtools {
namespace ir {

// A struct for containing the module header information.
struct ModuleHeader {
  uint32_t magic_number;
  uint32_t version;
  uint32_t generator;
  uint32_t bound;
  uint32_t reserved;
};

// A SPIR-V module. It contains all the information for a SPIR-V module and
// serves as the backbone of optimization transformations.
class Module {
 public:
  // Creates an empty module with zero'd header.
  Module() : header_({}) {}

  // Sets the header to the given |header|.
  void SetHeader(const ModuleHeader& header) { header_ = header; }
  // Appends a capability instruction to this module.
  void AddCapability(Instruction&& c) { capabilities_.push_back(std::move(c)); }
  // Appends an extension instruction to this module.
  void AddExtension(Instruction&& e) { extensions_.push_back(std::move(e)); }
  // Appends an extended instruction set instruction to this module.
  void AddExtInstImport(Instruction&& e) {
    ext_inst_imports_.push_back(std::move(e));
  }
  // Appends a memory model instruction to this module.
  void SetMemoryModel(Instruction&& m) { memory_model_ = std::move(m); }
  // Appends an entry point instruction to this module.
  void AddEntryPoint(Instruction&& e) { entry_points_.push_back(std::move(e)); }
  // Appends an execution mode instruction to this module.
  void AddExecutionMode(Instruction&& e) {
    execution_modes_.push_back(std::move(e));
  }
  // Appends a debug instruction (excluding OpLine & OpNoLine) to this module.
  void AddDebugInst(Instruction&& d) { debugs_.push_back(std::move(d)); }
  // Appends an annotation instruction to this module.
  void AddAnnotationInst(Instruction&& a) {
    annotations_.push_back(std::move(a));
  }
  // Appends a type-declaration instruction to this module.
  void AddType(Instruction&& t) { types_values_.push_back(std::move(t)); }
  // Appends a constant-creation instruction to this module.
  void AddConstant(Instruction&& c) { types_values_.push_back(std::move(c)); }
  // Appends a global variable-declaration instruction to this module.
  void AddGlobalVariable(Instruction&& v) {
    types_values_.push_back(std::move(v));
  }
  // Appends a function to this module.
  void AddFunction(Function&& f) { functions_.push_back(std::move(f)); }

  // Returns a vector of pointers to type-declaration instructions in this
  // module.
  std::vector<Instruction*> types();
  const std::vector<Instruction>& debugs() const { return debugs_; }
  std::vector<Instruction>& debugs() { return debugs_; }
  const std::vector<Instruction>& annotations() const { return annotations_; }
  std::vector<Instruction>& annotations() { return annotations_; }
  const std::vector<Function>& functions() const { return functions_; }
  std::vector<Function>& functions() { return functions_; }

  // Invokes function |f| on all instructions in this module.
  void ForEachInst(const std::function<void(Instruction*)>& f);

  // Pushes the binary segments for this instruction into the back of *|binary|.
  // If |skip_nop| is true and this is a OpNop, do nothing.
  void ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const;

 private:
  ModuleHeader header_;  // Module header

  // The following fields respect the "Logical Layout of a Module" in
  // Section 2.4 of the SPIR-V specification.
  std::vector<Instruction> capabilities_;
  std::vector<Instruction> extensions_;
  std::vector<Instruction> ext_inst_imports_;
  Instruction memory_model_;  // A module only has one memory model instruction.
  std::vector<Instruction> entry_points_;
  std::vector<Instruction> execution_modes_;
  std::vector<Instruction> debugs_;
  std::vector<Instruction> annotations_;
  // Type declarations, constants, and global variable declarations.
  std::vector<Instruction> types_values_;
  std::vector<Function> functions_;
};

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_MODULE_H_
