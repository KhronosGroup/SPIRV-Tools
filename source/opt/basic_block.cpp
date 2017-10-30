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

#include "basic_block.h"

#include "make_unique.h"

namespace spvtools {
namespace ir {

namespace {

const uint32_t kLoopMergeContinueBlockIdInIdx = 1;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;

}  // namespace

BasicBlock::BasicBlock(const BasicBlock& bb)
    : function_(nullptr),
      label_(MakeUnique<Instruction>(bb.GetLabelInst())),
      insts_() {
  for (auto& inst : bb.insts_)
    AddInstruction(std::unique_ptr<Instruction>(inst.Clone()));
}

const Instruction* BasicBlock::GetMergeInst() const {
  const Instruction* result = nullptr;
  // If it exists, the merge instruction immediately precedes the
  // terminator.
  auto iter = ctail();
  if (iter != cbegin()) {
    --iter;
    const auto opcode = iter->opcode();
    if (opcode == SpvOpLoopMerge || opcode == SpvOpSelectionMerge) {
      result = &*iter;
    }
  }
  return result;
}

Instruction* BasicBlock::GetMergeInst() {
  Instruction* result = nullptr;
  // If it exists, the merge instruction immediately precedes the
  // terminator.
  auto iter = tail();
  if (iter != begin()) {
    --iter;
    const auto opcode = iter->opcode();
    if (opcode == SpvOpLoopMerge || opcode == SpvOpSelectionMerge) {
      result = &*iter;
    }
  }
  return result;
}

const Instruction* BasicBlock::GetLoopMergeInst() const {
  if (auto* merge = GetMergeInst()) {
    if (merge->opcode() == SpvOpLoopMerge) {
      return merge;
    }
  }
  return nullptr;
}

Instruction* BasicBlock::GetLoopMergeInst() {
  if (auto* merge = GetMergeInst()) {
    if (merge->opcode() == SpvOpLoopMerge) {
      return merge;
    }
  }
  return nullptr;
}

void BasicBlock::ForEachSuccessorLabel(
    const std::function<void(const uint32_t)>& f) {
  const auto br = &insts_.back();
  switch (br->opcode()) {
    case SpvOpBranch: {
      f(br->GetOperand(0).words[0]);
    } break;
    case SpvOpBranchConditional:
    case SpvOpSwitch: {
      bool is_first = true;
      br->ForEachInId([&is_first, &f](const uint32_t* idp) {
        if (!is_first) f(*idp);
        is_first = false;
      });
    } break;
    default:
      break;
  }
}

void BasicBlock::ForMergeAndContinueLabel(
    const std::function<void(const uint32_t)>& f) {
  auto ii = insts_.end();
  --ii;
  if (ii == insts_.begin()) return;
  --ii;
  if (ii->opcode() == SpvOpSelectionMerge || ii->opcode() == SpvOpLoopMerge)
    ii->ForEachInId([&f](const uint32_t* idp) { f(*idp); });
}

uint32_t BasicBlock::MergeBlockIdIfAny(uint32_t* cbid) {
  auto merge_ii = cend();
  --merge_ii;
  if (cbid != nullptr) {
    *cbid = 0;
  }
  uint32_t mbid = 0;
  if (merge_ii != cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx);
      if (cbid != nullptr) {
        *cbid =
            merge_ii->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
      }
    } else if (merge_ii->opcode() == SpvOpSelectionMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
    }
  }
  return mbid;
}

}  // namespace ir
}  // namespace spvtools
