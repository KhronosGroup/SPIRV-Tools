// Copyright (c) 2017 Google Inc.
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

#ifndef SPIRV_TOOLS_IR_CONTEXT_H
#define SPIRV_TOOLS_IR_CONTEXT_H

#include "module.h"

#include <iostream>

namespace spvtools {
namespace ir {

class IRContext {
 public:
  IRContext(std::unique_ptr<Module>&& m) : module_(std::move(m)) {}
  Module* module() const { return module_.get(); }

  inline void SetIdBound(uint32_t i);
  inline uint32_t IdBound() const;

  // Returns a vector of pointers to constant-creation instructions in this
  // context.
  inline std::vector<Instruction*> GetConstants();
  inline std::vector<const Instruction*> GetConstants() const;

  // Iterators for annotation instructions contained in this context.
  inline Module::inst_iterator annotation_begin();
  inline Module::inst_iterator annotation_end();
  inline IteratorRange<Module::inst_iterator> annotations();
  inline IteratorRange<Module::const_inst_iterator> annotations() const;

  // Iterators for capabilities instructions contained in this module.
  inline Module::inst_iterator capability_begin();
  inline Module::inst_iterator capability_end();
  inline IteratorRange<Module::inst_iterator> capabilities();
  inline IteratorRange<Module::const_inst_iterator> capabilities() const;

  // Iterators for types, constants and global variables instructions.
  inline ir::Module::inst_iterator types_values_begin();
  inline ir::Module::inst_iterator types_values_end();
  inline IteratorRange<Module::inst_iterator> types_values();
  inline IteratorRange<Module::const_inst_iterator> types_values() const;

  // Iterators for extension instructions contained in this module.
  inline Module::inst_iterator ext_inst_import_begin();
  inline Module::inst_iterator ext_inst_import_end();

  // There are several kinds of debug instructions, according to where they can
  // appear in the logical layout of a module:
  //  - Section 7a:  OpString, OpSourceExtension, OpSource, OpSourceContinued
  //  - Section 7b:  OpName, OpMemberName
  //  - Section 7c:  OpModuleProcessed
  //  - Mostly anywhere: OpLine and OpNoLine
  //

  // Iterators for debug 1 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7a.
  inline Module::inst_iterator debug1_begin();
  inline Module::inst_iterator debug1_end();
  inline IteratorRange<Module::inst_iterator> debugs1();
  inline IteratorRange<Module::const_inst_iterator> debugs1() const;

  // Iterators for debug 2 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7b.
  inline Module::inst_iterator debug2_begin();
  inline Module::inst_iterator debug2_end();
  inline IteratorRange<Module::inst_iterator> debugs2();
  inline IteratorRange<Module::const_inst_iterator> debugs2() const;

  // Iterators for debug 3 instructions (excluding OpLine & OpNoLine) contained
  // in this module.  These are for layout section 7c.
  inline Module::inst_iterator debug3_begin();
  inline Module::inst_iterator debug3_end();
  inline IteratorRange<Module::inst_iterator> debugs3();
  inline IteratorRange<Module::const_inst_iterator> debugs3() const;

  // Clears all debug instructions (excluding OpLine & OpNoLine).
  inline void debug_clear();

  // Appends a capability instruction to this module.
  inline void AddCapability(std::unique_ptr<Instruction>&& c);
  // Appends an extension instruction to this module.
  inline void AddExtension(std::unique_ptr<Instruction>&& e);
  // Appends an extended instruction set instruction to this module.
  inline void AddExtInstImport(std::unique_ptr<Instruction>&& e);
  // Set the memory model for this module.
  inline void SetMemoryModel(std::unique_ptr<Instruction>&& m);
  // Appends an entry point instruction to this module.
  inline void AddEntryPoint(std::unique_ptr<Instruction>&& e);
  // Appends an execution mode instruction to this module.
  inline void AddExecutionMode(std::unique_ptr<Instruction>&& e);
  // Appends a debug 1 instruction (excluding OpLine & OpNoLine) to this module.
  // "debug 1" instructions are the ones in layout section 7.a), see section
  // 2.4 Logical Layout of a Module from the SPIR-V specification.
  inline void AddDebug1Inst(std::unique_ptr<Instruction>&& d);
  // Appends a debug 2 instruction (excluding OpLine & OpNoLine) to this module.
  // "debug 2" instructions are the ones in layout section 7.b), see section
  // 2.4 Logical Layout of a Module from the SPIR-V specification.
  inline void AddDebug2Inst(std::unique_ptr<Instruction>&& d);
  // Appends a debug 3 instruction (OpModuleProcessed) to this module.
  // This is due to decision by the SPIR Working Group, pending publication.
  inline void AddDebug3Inst(std::unique_ptr<Instruction>&& d);
  // Appends an annotation instruction to this module.
  inline void AddAnnotationInst(std::unique_ptr<Instruction>&& a);
  // Appends a type-declaration instruction to this module.
  inline void AddType(std::unique_ptr<Instruction>&& t);
  // Appends a constant, global variable, or OpUndef instruction to this module.
  inline void AddGlobalValue(std::unique_ptr<Instruction>&& v);
  // Appends a function to this module.
  inline void AddFunction(std::unique_ptr<Function>&& f);

 private:
  std::unique_ptr<Module> module_;
};

void IRContext::SetIdBound(uint32_t i) { module_->SetIdBound(i); }

uint32_t IRContext::IdBound() const { return module()->IdBound(); }

std::vector<Instruction*> spvtools::ir::IRContext::GetConstants() {
  return module()->GetConstants();
}

std::vector<const Instruction*> IRContext::GetConstants() const {
  return ((const Module*)module())->GetConstants();
}

Module::inst_iterator IRContext::annotation_begin() {
  return module()->annotation_begin();
}

Module::inst_iterator IRContext::annotation_end() {
  return module()->annotation_end();
}

IteratorRange<Module::inst_iterator> IRContext::annotations() {
  return module_->annotations();
}

IteratorRange<Module::const_inst_iterator> IRContext::annotations() const {
  return ((const Module*)module_.get())->annotations();
}

Module::inst_iterator IRContext::capability_begin() {
  return module()->capability_begin();
}

Module::inst_iterator IRContext::capability_end() {
  return module()->capability_end();
}

IteratorRange<Module::inst_iterator> IRContext::capabilities() {
  return module()->capabilities();
}

IteratorRange<Module::const_inst_iterator> IRContext::capabilities() const {
  return ((const Module*)module())->capabilities();
}

ir::Module::inst_iterator IRContext::types_values_begin() {
  return module()->types_values_begin();
}

ir::Module::inst_iterator IRContext::types_values_end() {
  return module()->types_values_end();
}

IteratorRange<Module::inst_iterator> IRContext::types_values() {
  return module()->types_values();
}

IteratorRange<Module::const_inst_iterator> IRContext::types_values() const {
  return ((const Module*)module_.get())->types_values();
}

Module::inst_iterator IRContext::ext_inst_import_begin() {
  return module()->ext_inst_import_begin();
}

Module::inst_iterator IRContext::ext_inst_import_end() {
  return module()->ext_inst_import_end();
}

Module::inst_iterator IRContext::debug1_begin() {
  return module()->debug1_begin();
}

Module::inst_iterator IRContext::debug1_end() { return module()->debug1_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs1() {
  return module()->debugs1();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs1() const {
  return ((const Module*)module_.get())->debugs1();
}

Module::inst_iterator IRContext::debug2_begin() {
  return module()->debug2_begin();
}
Module::inst_iterator IRContext::debug2_end() { return module()->debug2_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs2() {
  return module()->debugs2();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs2() const {
  return ((const Module*)module_.get())->debugs2();
}

Module::inst_iterator IRContext::debug3_begin() {
  return module()->debug3_begin();
}

Module::inst_iterator IRContext::debug3_end() { return module()->debug3_end(); }

IteratorRange<Module::inst_iterator> IRContext::debugs3() {
  return module()->debugs3();
}

IteratorRange<Module::const_inst_iterator> IRContext::debugs3() const {
  return ((const Module*)module_.get())->debugs3();
}

void IRContext::debug_clear() { module_->debug_clear(); }

void IRContext::AddCapability(std::unique_ptr<Instruction>&& c) {
  module()->AddCapability(std::move(c));
}

void IRContext::AddExtension(std::unique_ptr<Instruction>&& e) {
  module()->AddExtension(std::move(e));
}

void IRContext::AddExtInstImport(std::unique_ptr<Instruction>&& e) {
  module()->AddExtInstImport(std::move(e));
}

void IRContext::SetMemoryModel(std::unique_ptr<Instruction>&& m) {
  module()->SetMemoryModel(std::move(m));
}

void IRContext::AddEntryPoint(std::unique_ptr<Instruction>&& e) {
  module()->AddEntryPoint(std::move(e));
}

void IRContext::AddExecutionMode(std::unique_ptr<Instruction>&& e) {
  module()->AddExecutionMode(std::move(e));
}

void IRContext::AddDebug1Inst(std::unique_ptr<Instruction>&& d) {
  module()->AddDebug1Inst(std::move(d));
}

void IRContext::AddDebug2Inst(std::unique_ptr<Instruction>&& d) {
  module()->AddDebug2Inst(std::move(d));
}

void IRContext::AddDebug3Inst(std::unique_ptr<Instruction>&& d) {
  module()->AddDebug3Inst(std::move(d));
}

void IRContext::AddAnnotationInst(std::unique_ptr<Instruction>&& a) {
  module()->AddAnnotationInst(std::move(a));
}

void IRContext::AddType(std::unique_ptr<Instruction>&& t) {
  module()->AddType(std::move(t));
}

void IRContext::AddGlobalValue(std::unique_ptr<Instruction>&& v) {
  module()->AddGlobalValue(std::move(v));
}

void IRContext::AddFunction(std::unique_ptr<Function>&& f) {
  module()->AddFunction(std::move(f));
}
}  // namespace ir
}  // namespace spvtools
#endif  // SPIRV_TOOLS_IR_CONTEXT_H
