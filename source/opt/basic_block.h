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

// This file defines the language constructs for representing a SPIR-V
// module in memory.

#ifndef LIBSPIRV_OPT_BASIC_BLOCK_H_
#define LIBSPIRV_OPT_BASIC_BLOCK_H_

#include <functional>
#include <vector>
#include <utility>

#include "instruction.h"

namespace spvtools {
namespace ir {

class Function;

// A SPIR-V basic block.
class BasicBlock {
 public:
  // Creates a basic block with the given enclosing |function| and starting
  // |label|.
  BasicBlock(Instruction&& label)
      : function_(nullptr), label_(std::move(label)) {}

  // Sets the enclosing function for this basic block.
  void SetParent(Function* function) { function_ = function; }
  // Appends an instruction to this basic block.
  void AddInstruction(Instruction&& i) { insts_.push_back(std::move(i)); }

  // Runs the given function |f| on each instruction in this basic block.
  inline void ForEachInst(const std::function<void(Instruction*)>& f);

  // Pushes the binary segments for this instruction into the back of *|binary|.
  // If |skip_nop| is true and this is a OpNop, do nothing.
  inline void ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const;

 private:
  Function* function_;              // The enclosing function.
  Instruction label_;               // The label starting this basic block.
  std::vector<Instruction> insts_;  // Instructions inside this basic block.
};

inline void BasicBlock::ForEachInst(
    const std::function<void(Instruction*)>& f) {
  label_.ForEachInst(f);
  for (auto& inst : insts_) f(&inst);
}

inline void BasicBlock::ToBinary(std::vector<uint32_t>* binary,
                                 bool skip_nop) const {
  label_.ToBinary(binary, skip_nop);
  for (const auto& inst : insts_) inst.ToBinary(binary, skip_nop);
}

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_BASIC_BLOCK_H_
