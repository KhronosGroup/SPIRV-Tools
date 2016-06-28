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

#ifndef LIBSPIRV_OPT_CONSTRUCTS_H_
#define LIBSPIRV_OPT_CONSTRUCTS_H_

#include <functional>
#include <vector>
#include <utility>

#include "basic_block.h"
#include "instruction.h"

namespace spvtools {
namespace ir {

class Module;

// A SPIR-V function.
class Function {
 public:
  Function(Instruction&& def_inst)
      : module_(nullptr),
        def_inst_(std::move(def_inst)),
        end_inst_(SpvOpFunctionEnd) {}

  // Sets the enclosing module for this function.
  void SetParent(Module* module) { module_ = module; }
  // Appends a parameter to this function.
  void AddParameter(Instruction&& p) { params_.push_back(std::move(p)); }
  // Appends a basic block to this function.
  void AddBasicBlock(BasicBlock&& b) { blocks_.push_back(std::move(b)); }

  const std::vector<BasicBlock>& basic_blocks() const { return blocks_; }
  std::vector<BasicBlock>& basic_blocks() { return blocks_; }

  // Runs the given function |f| on each instruction in this basic block.
  void ForEachInst(const std::function<void(Instruction*)>& f);

  // Pushes the binary segments for this instruction into the back of *|binary|.
  // If |skip_nop| is true and this is a OpNop, do nothing.
  void ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const;

 private:
  Module* module_;        // The enclosing module.
  Instruction def_inst_;  // The instruction definining this function.
  std::vector<Instruction> params_;  // All parameters to this function.
  std::vector<BasicBlock> blocks_;   // All basic blocks inside this function.
  Instruction end_inst_;             // The OpFunctionEnd instruction.
};

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_CONSTRUCTS_H_
