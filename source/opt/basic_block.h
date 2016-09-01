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
#include "iterator.h"

namespace spvtools {
namespace ir {

class Function;

// A SPIR-V basic block.
class BasicBlock {
 public:
  using iterator = UptrVectorIterator<Instruction>;
  using const_iterator = UptrVectorIterator<Instruction, true>;

  // Creates a basic block with the given starting |label|.
  inline explicit BasicBlock(std::unique_ptr<Instruction> label);

  // Sets the enclosing function for this basic block.
  void SetParent(Function* function) { function_ = function; }
  // Appends an instruction to this basic block.
  inline void AddInstruction(std::unique_ptr<Instruction> i);

  iterator begin() { return iterator(&insts_, insts_.begin()); }
  iterator end() { return iterator(&insts_, insts_.end()); }
  const_iterator cbegin() { return const_iterator(&insts_, insts_.cbegin()); }
  const_iterator cend() { return const_iterator(&insts_, insts_.cend()); }

  // Runs the given function |f| on each instruction in this basic block, and
  // optionally on the debug line instructions that might precede them.
  inline void ForEachInst(const std::function<void(Instruction*)>& f,
                          bool run_on_debug_line_insts = false);
  inline void ForEachInst(const std::function<void(const Instruction*)>& f,
                          bool run_on_debug_line_insts = false) const;

 private:
  // The enclosing function.
  Function* function_;
  // The label starting this basic block.
  std::unique_ptr<Instruction> label_;
  // Instructions inside this basic block, but not the OpLabel.
  std::vector<std::unique_ptr<Instruction>> insts_;
};

inline BasicBlock::BasicBlock(std::unique_ptr<Instruction> label)
    : function_(nullptr), label_(std::move(label)) {}

inline void BasicBlock::AddInstruction(std::unique_ptr<Instruction> i) {
  insts_.emplace_back(std::move(i));
}

inline void BasicBlock::ForEachInst(const std::function<void(Instruction*)>& f,
                                    bool run_on_debug_line_insts) {
  if (label_) label_->ForEachInst(f, run_on_debug_line_insts);
  for (auto& inst : insts_) inst->ForEachInst(f, run_on_debug_line_insts);
}

inline void BasicBlock::ForEachInst(
    const std::function<void(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  if (label_)
    static_cast<const Instruction*>(label_.get())
        ->ForEachInst(f, run_on_debug_line_insts);
  for (const auto& inst : insts_)
    static_cast<const Instruction*>(inst.get())
        ->ForEachInst(f, run_on_debug_line_insts);
}

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_BASIC_BLOCK_H_
