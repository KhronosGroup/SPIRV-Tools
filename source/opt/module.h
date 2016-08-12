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
#include <memory>
#include <utility>
#include <vector>

#include "function.h"
#include "instruction.h"
#include "iterator.h"

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
  using iterator = UptrVectorIterator<Function>;
  using const_iterator = UptrVectorIterator<Function, true>;
  using inst_iterator = UptrVectorIterator<Instruction>;
  using const_inst_iterator = UptrVectorIterator<Instruction, true>;

  // Creates an empty module with zero'd header.
  Module() : header_({}) {}

  // Sets the header to the given |header|.
  void SetHeader(const ModuleHeader& header) { header_ = header; }
  // Appends a capability instruction to this module.
  inline void AddCapability(std::unique_ptr<Instruction> c);
  // Appends an extension instruction to this module.
  inline void AddExtension(std::unique_ptr<Instruction> e);
  // Appends an extended instruction set instruction to this module.
  inline void AddExtInstImport(std::unique_ptr<Instruction> e);
  // Set the memory model for this module.
  inline void SetMemoryModel(std::unique_ptr<Instruction> m);
  // Appends an entry point instruction to this module.
  inline void AddEntryPoint(std::unique_ptr<Instruction> e);
  // Appends an execution mode instruction to this module.
  inline void AddExecutionMode(std::unique_ptr<Instruction> e);
  // Appends a debug instruction (excluding OpLine & OpNoLine) to this module.
  inline void AddDebugInst(std::unique_ptr<Instruction> d);
  // Appends an annotation instruction to this module.
  inline void AddAnnotationInst(std::unique_ptr<Instruction> a);
  // Appends a type-declaration instruction to this module.
  inline void AddType(std::unique_ptr<Instruction> t);
  // Appends a constant-creation instruction to this module.
  inline void AddConstant(std::unique_ptr<Instruction> c);
  // Appends a global variable-declaration instruction to this module.
  inline void AddGlobalVariable(std::unique_ptr<Instruction> v);
  // Appends a function to this module.
  inline void AddFunction(std::unique_ptr<Function> f);

  // Returns a vector of pointers to type-declaration instructions in this
  // module.
  std::vector<Instruction*> GetTypes();
  std::vector<const Instruction*> GetTypes() const;
  // Returns a vector of pointers to constant-creation instructions in this
  // module.
  std::vector<Instruction*> GetConstants();
  std::vector<const Instruction*> GetConstants() const;

  // Iterators for debug instructions (excluding OpLine & OpNoLine) contained in
  // this module.
  inline inst_iterator debug_begin();
  inline inst_iterator debug_end();
  inline IteratorRange<inst_iterator> debugs();
  inline IteratorRange<const_inst_iterator> debugs() const;

  // Clears all debug instructions (excluding OpLine & OpNoLine).
  void debug_clear() { debugs_.clear(); }

  // Iterators for annotation instructions contained in this module.
  IteratorRange<inst_iterator> annotations();
  IteratorRange<const_inst_iterator> annotations() const;

  // Iterators for functions contained in this module.
  iterator begin() { return iterator(&functions_, functions_.begin()); }
  iterator end() { return iterator(&functions_, functions_.end()); }
  inline const_iterator cbegin() const;
  inline const_iterator cend() const;

  // Invokes function |f| on all instructions in this module.
  void ForEachInst(const std::function<void(Instruction*)>& f);

  // Pushes the binary segments for this instruction into the back of *|binary|.
  // If |skip_nop| is true and this is a OpNop, do nothing.
  void ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const;

 private:
  ModuleHeader header_;  // Module header

  // The following fields respect the "Logical Layout of a Module" in
  // Section 2.4 of the SPIR-V specification.
  std::vector<std::unique_ptr<Instruction>> capabilities_;
  std::vector<std::unique_ptr<Instruction>> extensions_;
  std::vector<std::unique_ptr<Instruction>> ext_inst_imports_;
  // A module only has one memory model instruction.
  std::unique_ptr<Instruction> memory_model_;
  std::vector<std::unique_ptr<Instruction>> entry_points_;
  std::vector<std::unique_ptr<Instruction>> execution_modes_;
  std::vector<std::unique_ptr<Instruction>> debugs_;
  std::vector<std::unique_ptr<Instruction>> annotations_;
  // Type declarations, constants, and global variable declarations.
  std::vector<std::unique_ptr<Instruction>> types_values_;
  std::vector<std::unique_ptr<Function>> functions_;
};

inline void Module::AddCapability(std::unique_ptr<Instruction> c) {
  capabilities_.emplace_back(std::move(c));
}

inline void Module::AddExtension(std::unique_ptr<Instruction> e) {
  extensions_.emplace_back(std::move(e));
}

inline void Module::AddExtInstImport(std::unique_ptr<Instruction> e) {
  ext_inst_imports_.emplace_back(std::move(e));
}

inline void Module::SetMemoryModel(std::unique_ptr<Instruction> m) {
  memory_model_ = std::move(m);
}

inline void Module::AddEntryPoint(std::unique_ptr<Instruction> e) {
  entry_points_.emplace_back(std::move(e));
}

inline void Module::AddExecutionMode(std::unique_ptr<Instruction> e) {
  execution_modes_.emplace_back(std::move(e));
}

inline void Module::AddDebugInst(std::unique_ptr<Instruction> d) {
  debugs_.emplace_back(std::move(d));
}

inline void Module::AddAnnotationInst(std::unique_ptr<Instruction> a) {
  annotations_.emplace_back(std::move(a));
}

inline void Module::AddType(std::unique_ptr<Instruction> t) {
  types_values_.emplace_back(std::move(t));
}

inline void Module::AddConstant(std::unique_ptr<Instruction> c) {
  types_values_.emplace_back(std::move(c));
}

inline void Module::AddGlobalVariable(std::unique_ptr<Instruction> v) {
  types_values_.emplace_back(std::move(v));
}

inline void Module::AddFunction(std::unique_ptr<Function> f) {
  functions_.emplace_back(std::move(f));
}

inline Module::inst_iterator Module::debug_begin() {
  return inst_iterator(&debugs_, debugs_.begin());
}
inline Module::inst_iterator Module::debug_end() {
  return inst_iterator(&debugs_, debugs_.end());
}

inline IteratorRange<Module::inst_iterator> Module::debugs() {
  return IteratorRange<inst_iterator>(inst_iterator(&debugs_, debugs_.begin()),
                                      inst_iterator(&debugs_, debugs_.end()));
}

inline IteratorRange<Module::const_inst_iterator> Module::debugs() const {
  return IteratorRange<const_inst_iterator>(
      const_inst_iterator(&debugs_, debugs_.cbegin()),
      const_inst_iterator(&debugs_, debugs_.cend()));
}

inline IteratorRange<Module::inst_iterator> Module::annotations() {
  return IteratorRange<inst_iterator>(
      inst_iterator(&annotations_, annotations_.begin()),
      inst_iterator(&annotations_, annotations_.end()));
}

inline IteratorRange<Module::const_inst_iterator> Module::annotations() const {
  return IteratorRange<const_inst_iterator>(
      const_inst_iterator(&annotations_, annotations_.cbegin()),
      const_inst_iterator(&annotations_, annotations_.cend()));
}

inline Module::const_iterator Module::cbegin() const {
  return const_iterator(&functions_, functions_.cbegin());
}

inline Module::const_iterator Module::cend() const {
  return const_iterator(&functions_, functions_.cend());
}

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_MODULE_H_
