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

#include "BasicBlock.h"

#include <vector>

using std::vector;

namespace libspirv {

BasicBlock::BasicBlock(uint32_t id)
    : id_(id),
      immediate_dominator_(nullptr),
      predecessors_(),
      successors_(),
      reachable_(false) {}

void BasicBlock::SetImmediateDominator(BasicBlock* dom_block) {
  immediate_dominator_ = dom_block;
}

const BasicBlock* BasicBlock::GetImmediateDominator() const {
  return immediate_dominator_;
}

BasicBlock* BasicBlock::GetImmediateDominator() { return immediate_dominator_; }

void BasicBlock::RegisterSuccessors(vector<BasicBlock*> next_blocks) {
  for (auto& block : next_blocks) {
    block->predecessors_.push_back(this);
    successors_.push_back(block);
    if (block->reachable_ == false) block->set_reachability(reachable_);
  }
}

void BasicBlock::RegisterBranchInstruction(SpvOp branch_instruction) {
  if (branch_instruction == SpvOpUnreachable) reachable_ = false;
  return;
}

BasicBlock::DominatorIterator::DominatorIterator() : current_(nullptr) {}
BasicBlock::DominatorIterator::DominatorIterator(const BasicBlock* block)
    : current_(block) {}

BasicBlock::DominatorIterator& BasicBlock::DominatorIterator::operator++() {
  if (current_ == current_->GetImmediateDominator()) {
    current_ = nullptr;
  } else {
    current_ = current_->GetImmediateDominator();
  }
  return *this;
}

const BasicBlock::DominatorIterator BasicBlock::dom_begin() const {
  return DominatorIterator(this);
}

BasicBlock::DominatorIterator BasicBlock::dom_begin() {
  return DominatorIterator(this);
}

const BasicBlock::DominatorIterator BasicBlock::dom_end() const {
  return DominatorIterator();
}

BasicBlock::DominatorIterator BasicBlock::dom_end() {
  return DominatorIterator();
}

bool operator==(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return lhs.current_ == rhs.current_;
}

bool operator!=(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return !(lhs == rhs);
}

const BasicBlock*& BasicBlock::DominatorIterator::operator*() {
  return current_;
}
}  // namespace libspirv
