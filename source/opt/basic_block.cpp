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

#include "source/opt/basic_block.h"

#include <ostream>

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/make_unique.h"
#include "source/opt/module.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {
namespace {

const uint32_t kLoopMergeContinueBlockIdInIdx = 1;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;

}  // namespace

BasicBlock* BasicBlock::Clone(IRContext* context) const {
  BasicBlock* clone = new BasicBlock(
      std::unique_ptr<Instruction>(GetLabelInst()->Clone(context)));
  for (const auto& inst : insts_)
    // Use the incoming context
    clone->AddInstruction(std::unique_ptr<Instruction>(inst.Clone(context)));
  return clone;
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

void BasicBlock::KillAllInsts(bool killLabel) {
  ForEachInst([killLabel](Instruction* ip) {
    if (killLabel || ip->opcode() != SpvOpLabel) {
      ip->context()->KillInst(ip);
    }
  });
}

void BasicBlock::ForEachSuccessorLabel(
    const std::function<void(const uint32_t)>& f) const {
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

void BasicBlock::ForEachSuccessorLabel(
    const std::function<void(uint32_t*)>& f) {
  auto br = &insts_.back();
  switch (br->opcode()) {
    case SpvOpBranch: {
      uint32_t tmp_id = br->GetOperand(0).words[0];
      f(&tmp_id);
      if (tmp_id != br->GetOperand(0).words[0]) br->SetOperand(0, {tmp_id});
    } break;
    case SpvOpBranchConditional:
    case SpvOpSwitch: {
      bool is_first = true;
      br->ForEachInId([&is_first, &f](uint32_t* idp) {
        if (!is_first) f(idp);
        is_first = false;
      });
    } break;
    default:
      break;
  }
}

bool BasicBlock::IsSuccessor(const BasicBlock* block) const {
  uint32_t succId = block->id();
  bool isSuccessor = false;
  ForEachSuccessorLabel([&isSuccessor, succId](const uint32_t label) {
    if (label == succId) isSuccessor = true;
  });
  return isSuccessor;
}

void BasicBlock::ForMergeAndContinueLabel(
    const std::function<void(const uint32_t)>& f) {
  auto ii = insts_.end();
  --ii;
  if (ii == insts_.begin()) return;
  --ii;
  if (ii->opcode() == SpvOpSelectionMerge || ii->opcode() == SpvOpLoopMerge) {
    ii->ForEachInId([&f](const uint32_t* idp) { f(*idp); });
  }
}

uint32_t BasicBlock::MergeBlockIdIfAny() const {
  auto merge_ii = cend();
  --merge_ii;
  uint32_t mbid = 0;
  if (merge_ii != cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx);
    } else if (merge_ii->opcode() == SpvOpSelectionMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
    }
  }

  return mbid;
}

uint32_t BasicBlock::ContinueBlockIdIfAny() const {
  auto merge_ii = cend();
  --merge_ii;
  uint32_t cbid = 0;
  if (merge_ii != cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == SpvOpLoopMerge) {
      cbid = merge_ii->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
    }
  }
  return cbid;
}

std::ostream& operator<<(std::ostream& str, const BasicBlock& block) {
  str << block.PrettyPrint();
  return str;
}

std::string BasicBlock::PrettyPrint(uint32_t options) const {
  std::ostringstream str;
  ForEachInst([&str, options](const Instruction* inst) {
    str << inst->PrettyPrint(options);
    if (!IsTerminatorInst(inst->opcode())) {
      str << std::endl;
    }
  });
  return str.str();
}

BasicBlock* BasicBlock::SplitBasicBlock(IRContext* context, uint32_t label_id,
                                        iterator iter) {
  assert(!insts_.empty());

  BasicBlock* new_block = new BasicBlock(MakeUnique<Instruction>(
      context, SpvOpLabel, 0, label_id, std::initializer_list<Operand>{}));

  new_block->insts_.Splice(new_block->end(), &insts_, iter, end());
  new_block->SetParent(GetParent());

  context->AnalyzeDefUse(new_block->GetLabelInst());

  // Update the phi nodes in the successor blocks to reference the new block id.
  const_cast<const BasicBlock*>(new_block)->ForEachSuccessorLabel(
      [new_block, this, context](const uint32_t label) {
        BasicBlock* target_bb = context->get_instr_block(label);
        target_bb->ForEachPhiInst(
            [this, new_block, context](Instruction* phi_inst) {
              bool changed = false;
              for (uint32_t i = 1; i < phi_inst->NumInOperands(); i += 2) {
                if (phi_inst->GetSingleWordInOperand(i) == this->id()) {
                  changed = true;
                  phi_inst->SetInOperand(i, {new_block->id()});
                }
              }

              if (changed) {
                context->UpdateDefUse(phi_inst);
              }
            });
      });

  if (context->AreAnalysesValid(IRContext::kAnalysisInstrToBlockMapping)) {
    context->set_instr_block(new_block->GetLabelInst(), new_block);
    new_block->ForEachInst([new_block, context](Instruction* inst) {
      context->set_instr_block(inst, new_block);
    });
  }

  return new_block;
}

}  // namespace opt
}  // namespace spvtools
