// Copyright (c) 2016 Google Inc.
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

// This file defines the language constructs for representing a SPIR-V
// module in memory.

#ifndef LIBSPIRV_OPT_BASIC_BLOCK_H_
#define LIBSPIRV_OPT_BASIC_BLOCK_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "instruction.h"
#include "instruction_list.h"
#include "iterator.h"

namespace spvtools {
namespace ir {

class Function;
class IRContext;

// A SPIR-V basic block.
class BasicBlock {
 public:
  using iterator = InstructionList::iterator;
  using const_iterator = InstructionList::const_iterator;

  // Creates a basic block with the given starting |label|.
  inline explicit BasicBlock(std::unique_ptr<Instruction> label);

  explicit BasicBlock(const BasicBlock& bb) = delete;

  // Creates a clone of the basic block in the given |context|
  //
  // The parent function will default to null and needs to be explicitly set by
  // the user.
  BasicBlock* Clone(IRContext*) const;

  // Sets the enclosing function for this basic block.
  void SetParent(Function* function) { function_ = function; }

  // Return the enclosing function
  inline Function* GetParent() const { return function_; }

  // Appends an instruction to this basic block.
  inline void AddInstruction(std::unique_ptr<Instruction> i);

  // Appends all of block's instructions (except label) to this block
  inline void AddInstructions(BasicBlock* bp);

  // The label starting this basic block.
  Instruction* GetLabelInst() { return label_.get(); }
  const Instruction* GetLabelInst() const { return label_.get(); }

  // Returns the merge instruction in this basic block, if it exists.
  // Otherwise return null.  May be used whenever tail() can be used.
  const Instruction* GetMergeInst() const;
  Instruction* GetMergeInst();

  // Returns the OpLoopMerge instruciton in this basic block, if it exists.
  // Otherwise return null.  May be used whenever tail() can be used.
  const Instruction* GetLoopMergeInst() const;
  Instruction* GetLoopMergeInst();

  // Returns the id of the label at the top of this block
  inline uint32_t id() const { return label_->result_id(); }

  iterator begin() { return insts_.begin(); }
  iterator end() { return insts_.end(); }
  const_iterator begin() const { return insts_.cbegin(); }
  const_iterator end() const { return insts_.cend(); }
  const_iterator cbegin() const { return insts_.cbegin(); }
  const_iterator cend() const { return insts_.cend(); }

  // Returns an iterator pointing to the last instruction.  This may only
  // be used if this block has an instruction other than the OpLabel
  // that defines it.
  iterator tail() {
    assert(!insts_.empty());
    return --end();
  }

  // Returns a const iterator, but othewrise similar to tail().
  const_iterator ctail() const {
    assert(!insts_.empty());
    return --insts_.cend();
  }

  // Returns true if the basic block has at least one successor.
  inline bool hasSuccessor() const { return ctail()->IsBranch(); }

  // Runs the given function |f| on each instruction in this basic block, and
  // optionally on the debug line instructions that might precede them.
  inline void ForEachInst(const std::function<void(Instruction*)>& f,
                          bool run_on_debug_line_insts = false);
  inline void ForEachInst(const std::function<void(const Instruction*)>& f,
                          bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on each Phi instruction in this basic block,
  // and optionally on the debug line instructions that might precede them.
  inline void ForEachPhiInst(const std::function<void(Instruction*)>& f,
                             bool run_on_debug_line_insts = false);

  // Runs the given function |f| on each label id of each successor block
  void ForEachSuccessorLabel(
      const std::function<void(const uint32_t)>& f) const;

  // Returns true if |block| is a direct successor of |this|.
  bool IsSuccessor(const ir::BasicBlock* block) const;

  // Runs the given function |f| on the merge and continue label, if any
  void ForMergeAndContinueLabel(const std::function<void(const uint32_t)>& f);

  // Returns true if this basic block has any Phi instructions.
  bool HasPhiInstructions() {
    int count = 0;
    ForEachPhiInst([&count](ir::Instruction*) {
      ++count;
      return;
    });
    return count > 0;
  }

  // Return true if this block is a loop header block.
  bool IsLoopHeader() const { return GetLoopMergeInst() != nullptr; }

  // Returns the ID of the merge block declared by a merge instruction in this
  // block, if any.  If none, returns zero.
  uint32_t MergeBlockIdIfAny() const;

  // Returns the ID of the continue block declared by a merge instruction in
  // this block, if any.  If none, returns zero.
  uint32_t ContinueBlockIdIfAny() const;

  // Returns the terminator instruction.  Assumes the terminator exists.
  Instruction* terminator() { return &*tail(); }
  const Instruction* terminator() const { return &*ctail(); }

  // Returns true if this basic block exits this function and returns to its
  // caller.
  bool IsReturn() const { return ctail()->IsReturn(); }

  // Returns true if this basic block exits this function or aborts execution.
  bool IsReturnOrAbort() const { return ctail()->IsReturnOrAbort(); }

 private:
  // The enclosing function.
  Function* function_;
  // The label starting this basic block.
  std::unique_ptr<Instruction> label_;
  // Instructions inside this basic block, but not the OpLabel.
  InstructionList insts_;
};

// Pretty-prints |block| to |str|. Returns |str|.
std::ostream& operator<<(std::ostream& str, const BasicBlock& block);

inline BasicBlock::BasicBlock(std::unique_ptr<Instruction> label)
    : function_(nullptr), label_(std::move(label)) {}

inline void BasicBlock::AddInstruction(std::unique_ptr<Instruction> i) {
  insts_.push_back(std::move(i));
}

inline void BasicBlock::AddInstructions(BasicBlock* bp) {
  auto bEnd = end();
  (void)bEnd.MoveBefore(&bp->insts_);
}

inline void BasicBlock::ForEachInst(const std::function<void(Instruction*)>& f,
                                    bool run_on_debug_line_insts) {
  if (label_) label_->ForEachInst(f, run_on_debug_line_insts);
  if (insts_.empty()) {
    return;
  }

  Instruction* inst = &insts_.front();
  while (inst != nullptr) {
    Instruction* next_instruction = inst->NextNode();
    inst->ForEachInst(f, run_on_debug_line_insts);
    inst = next_instruction;
  }
}

inline void BasicBlock::ForEachInst(
    const std::function<void(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  if (label_)
    static_cast<const Instruction*>(label_.get())
        ->ForEachInst(f, run_on_debug_line_insts);
  for (const auto& inst : insts_)
    static_cast<const Instruction*>(&inst)->ForEachInst(
        f, run_on_debug_line_insts);
}

inline void BasicBlock::ForEachPhiInst(
    const std::function<void(Instruction*)>& f, bool run_on_debug_line_insts) {
  for (auto& inst : insts_) {
    if (inst.opcode() != SpvOpPhi) break;
    inst.ForEachInst(f, run_on_debug_line_insts);
  }
}

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_BASIC_BLOCK_H_
