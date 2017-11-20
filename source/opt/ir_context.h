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

#include "decoration_manager.h"
#include "def_use_manager.h"
#include "module.h"

#include <algorithm>
#include <iostream>
#include <limits>

namespace spvtools {
namespace ir {

class IRContext {
 public:
  // Available analyses.
  //
  // When adding a new analysis:
  //
  // 1. Enum values should be powers of 2. These are cast into uint32_t
  //    bitmasks, so we can have at most 31 analyses represented.
  //
  // 2. Make sure it gets invalidated or preserved by IRContext methods that add
  //    or remove IR elements (e.g., KillDef, KillInst, ReplaceAllUsesWith).
  //
  // 3. Add handling code in BuildInvalidAnalyses and
  //    InvalidateAnalysesExceptFor.
  enum Analysis {
    kAnalysisNone = 0 << 0,
    kAnalysisBegin = 1 << 0,
    kAnalysisDefUse = kAnalysisBegin,
    kAnalysisInstrToBlockMapping = 1 << 1,
    kAnalysisDecorations = 1 << 2,
    kAnalysisEnd = 1 << 3
  };

  friend inline Analysis operator|(Analysis lhs, Analysis rhs);
  friend inline Analysis& operator|=(Analysis& lhs, Analysis rhs);
  friend inline Analysis operator<<(Analysis a, int shift);
  friend inline Analysis& operator<<=(Analysis& a, int shift);

  // Create an |IRContext| that contains an owned |Module|
  IRContext(spvtools::MessageConsumer c)
      : unique_id_(0),
        module_(new Module()),
        consumer_(std::move(c)),
        def_use_mgr_(nullptr),
        valid_analyses_(kAnalysisNone)
  {
    module_->SetContext(this);
  }

  IRContext(std::unique_ptr<Module>&& m, spvtools::MessageConsumer c)
      : unique_id_(0),
        module_(std::move(m)),
        consumer_(std::move(c)),
        def_use_mgr_(nullptr),
        valid_analyses_(kAnalysisNone)
  {
    module_->SetContext(this);
  }
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

  // Returns a pointer to a def-use manager.  If the def-use manager is
  // invalid, it is rebuilt first.
  opt::analysis::DefUseManager* get_def_use_mgr() {
    if (!AreAnalysesValid(kAnalysisDefUse)) {
      BuildDefUseManager();
    }
    return def_use_mgr_.get();
  }

  // Returns the basic block for instruction |instr|. Re-builds the instruction
  // block map, if needed.
  ir::BasicBlock* get_instr_block(ir::Instruction* instr) {
    if (!AreAnalysesValid(kAnalysisInstrToBlockMapping)) {
      BuildInstrToBlockMapping();
    }
    auto entry = instr_to_block_.find(instr);
    return (entry != instr_to_block_.end()) ? entry->second : nullptr;
  }

  // Returns a pointer the decoration manager.  If the decoration manger is
  // invalid, it is rebuilt first.
  opt::analysis::DecorationManager* get_decoration_mgr() {
    if (!AreAnalysesValid(kAnalysisDecorations)) {
      BuildDecorationManager();
    }
    return decoration_mgr_.get();
  };

  // Sets the message consumer to the given |consumer|. |consumer| which will be
  // invoked every time there is a message to be communicated to the outside.
  void SetMessageConsumer(spvtools::MessageConsumer c) {
    consumer_ = std::move(c);
  }

  // Returns the reference to the message consumer for this pass.
  const spvtools::MessageConsumer& consumer() const { return consumer_; }

  // Rebuilds the analyses in |set| that are invalid.
  void BuildInvalidAnalyses(Analysis set);

  // Invalidates all of the analyses except for those in |preserved_analyses|.
  void InvalidateAnalysesExceptFor(Analysis preserved_analyses);

  // Invalidates the analyses marked in |analyses_to_invalidate|.
  void InvalidateAnalyses(Analysis analyses_to_invalidate);

  // Turns the instruction defining the given |id| into a Nop. Returns true on
  // success, false if the given |id| is not defined at all. This method also
  // erases both the uses of |id| and the information of this |id|-generating
  // instruction's uses of its operands.
  bool KillDef(uint32_t id);

  // Turns the given instruction |inst| to a Nop. This method erases the
  // information of the given instruction's uses of its operands. If |inst|
  // defines an result id, the uses of the result id will also be erased.
  void KillInst(ir::Instruction* inst);

  // Returns true if all of the given analyses are valid.
  bool AreAnalysesValid(Analysis set) { return (set & valid_analyses_) == set; }

  // Replaces all uses of |before| id with |after| id. Returns true if any
  // replacement happens. This method does not kill the definition of the
  // |before| id. If |after| is the same as |before|, does nothing and returns
  // false.
  bool ReplaceAllUsesWith(uint32_t before, uint32_t after);

  // Returns true if all of the analyses that are suppose to be valid are
  // actually valid.
  bool IsConsistent();

  // Informs the IRContext that the uses of |inst| are going to change, and that
  // is should forget everything it know about the current uses.  Any valid
  // analyses will be updated accordingly.
  void ForgetUses(Instruction* inst);

  // The IRContext will look at the uses of |inst| and update any valid analyses
  // will be updated accordingly.
  void AnalyzeUses(Instruction* inst);

  // Kill all name and decorate ops targeting |id|.
  void KillNamesAndDecorates(uint32_t id);

  // Kill all name and decorate ops targeting the result id of |inst|.
  void KillNamesAndDecorates(ir::Instruction* inst);

  // Returns the next unique id for use by an instruction.
  inline uint32_t TakeNextUniqueId() {
    assert(unique_id_ != std::numeric_limits<uint32_t>::max());

    // Skip zero.
    return ++unique_id_;
  }

 private:
  // Builds the def-use manager from scratch, even if it was already valid.
  void BuildDefUseManager() {
    def_use_mgr_.reset(new opt::analysis::DefUseManager(module()));
    valid_analyses_ = valid_analyses_ | kAnalysisDefUse;
  }

  // Builds the instruction-block map for the whole module.
  void BuildInstrToBlockMapping() {
    instr_to_block_.clear();
    for (auto& fn : *module_) {
      for (auto& block : fn) {
        block.ForEachInst([this, &block](ir::Instruction* inst) {
          instr_to_block_[inst] = &block;
        });
      }
    }
    valid_analyses_ = valid_analyses_ | kAnalysisInstrToBlockMapping;
  }

  void BuildDecorationManager() {
    decoration_mgr_.reset(new opt::analysis::DecorationManager(module()));
    valid_analyses_ = valid_analyses_ | kAnalysisDecorations;
  }

  // An unique identifier for this instruction. Can be used to order
  // instructions in a container.
  //
  // This member is initialized to 0, but always issues this value plus one.
  // Therefore, 0 is not a valid unique id for an instruction.
  uint32_t unique_id_;

  std::unique_ptr<Module> module_;
  spvtools::MessageConsumer consumer_;
  std::unique_ptr<opt::analysis::DefUseManager> def_use_mgr_;
  std::unique_ptr<opt::analysis::DecorationManager> decoration_mgr_;

  // A map from instructions the the basic block they belong to. This mapping is
  // built on-demand when get_instr_block() is called.
  //
  // NOTE: Do not traverse this map. Ever. Use the function and basic block
  // iterators to traverse instructions.
  std::unordered_map<ir::Instruction*, ir::BasicBlock*> instr_to_block_;

  // A bitset indicating which analyes are currently valid.
  Analysis valid_analyses_;
};

inline ir::IRContext::Analysis operator|(ir::IRContext::Analysis lhs,
                                         ir::IRContext::Analysis rhs) {
  return static_cast<ir::IRContext::Analysis>(static_cast<int>(lhs) |
                                              static_cast<int>(rhs));
}

inline ir::IRContext::Analysis& operator|=(ir::IRContext::Analysis& lhs,
                                           ir::IRContext::Analysis rhs) {
  lhs = static_cast<ir::IRContext::Analysis>(static_cast<int>(lhs) |
                                             static_cast<int>(rhs));
  return lhs;
}

inline ir::IRContext::Analysis operator<<(ir::IRContext::Analysis a,
                                          int shift) {
  return static_cast<ir::IRContext::Analysis>(static_cast<int>(a) << shift);
}

inline ir::IRContext::Analysis& operator<<=(ir::IRContext::Analysis& a,
                                            int shift) {
  a = static_cast<ir::IRContext::Analysis>(static_cast<int>(a) << shift);
  return a;
}

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
  if (AreAnalysesValid(kAnalysisDecorations)) {
    get_decoration_mgr()->AddDecoration(a.get());
  }
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
