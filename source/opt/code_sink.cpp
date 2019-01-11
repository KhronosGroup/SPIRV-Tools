// Copyright (c) 2018 Google LLC
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

#include "code_sink.h"

#include <set>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

Pass::Status CodeSinkingPass::Process() {
  bool modified = false;

  for (Function& function : *get_module()) {
    bool has_memory_sync = HasMemorySync(&function);
    cfg()->ForEachBlockInPostOrder(
        function.entry().get(), [&modified, has_memory_sync, this](BasicBlock* bb) {
          if (SinkInstInBB(bb, has_memory_sync)) {
            modified = true;
          }
        });
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool CodeSinkingPass::SinkInstInBB(BasicBlock* bb, bool function_has_memory_sync) {
  bool modified = false;
  for( auto inst = bb->rbegin(); inst != bb->rend(); ++inst) {
    if (SinkInst(&*inst, function_has_memory_sync)) {
      inst = bb->rbegin();
      modified = true;
    }
  }
  return modified;
}

bool CodeSinkingPass::SinkInst(Instruction* inst, bool function_has_memory_sync) {
/*
  if (inst->result_id() == 0) {
    return false;
  }

  if (!inst->IsOpcodeCodeMotionSafe()) {
    return false;
  }
*/

  if (inst->opcode() != SpvOpLoad && inst->opcode() != SpvOpAccessChain) {
    return false;
  }

  if (ReferencesMutableMemory(inst, function_has_memory_sync)) {
    return false;
  }

  if (BasicBlock* target_bb = FindNewBasicBlockFor(inst)) {
    Instruction* pos = &*target_bb->begin();
    while(pos->opcode() == SpvOpPhi) {
      pos = pos->NextNode();
    }

    inst->InsertBefore(pos);
    context()->set_instr_block(inst, target_bb);
    return true;
  }
  return false;
}

BasicBlock* CodeSinkingPass::FindNewBasicBlockFor(Instruction* inst) {
  assert(inst->result_id() != 0 && "Instruction should not have a result.");
  BasicBlock* original_bb = context()->get_instr_block(inst);
  BasicBlock* bb = original_bb;

  std::unordered_set<uint32_t> bbs_with_uses;
  get_def_use_mgr()->ForEachUse(inst, [&bbs_with_uses, this](Instruction* use, uint32_t idx) {
    if (use->opcode() != SpvOpPhi) {
      bbs_with_uses.insert(context()->get_instr_block(use)->id());
    } else {
      bbs_with_uses.insert(use->GetSingleWordOperand(idx+1));
    }
  });

  while(true) {
    if (bbs_with_uses.count(bb->id())) {
      break;
    }

    if (bb->terminator()->opcode() == SpvOpBranch) {
      uint32_t succ_bb_id = bb->terminator()->GetSingleWordInOperand(0);
      if (cfg()->preds(succ_bb_id).size() == 1) {
        bb = context()->get_instr_block(succ_bb_id);
        continue;
      } else {
        break;
      }
    }

    Instruction* merge_inst = bb->GetMergeInst();
    if (merge_inst == nullptr) {
      break;
    }

    if (merge_inst->opcode() == SpvOpSelectionMerge) {
      bool used_in_multiple_blocks = false;
      uint32_t bb_used_in = 0;

      bb->ForEachSuccessorLabel([this, bb, &bb_used_in, &used_in_multiple_blocks,&bbs_with_uses](uint32_t *succ_bb_id) {
        if (IntersectsPath(*succ_bb_id, bb->MergeBlockIdIfAny(), bbs_with_uses)) {
          if (bb_used_in == 0) {
            bb_used_in = *succ_bb_id;
          } else {
            used_in_multiple_blocks = true;
          }
        }
      });

      if (used_in_multiple_blocks) {
        break;
      }

      if (bb_used_in == 0) {
        bb = context()->get_instr_block(bb->MergeBlockIdIfAny());
      } else {
        if (IntersectsPath(bb->MergeBlockIdIfAny(), original_bb->id(), bbs_with_uses)) {
          break;
        }
        bb = context()->get_instr_block(bb_used_in);
      }
      continue;
    }
    break;
  }
  return (bb != original_bb ? bb : nullptr);
}

bool CodeSinkingPass::ReferencesMutableMemory(Instruction* inst, bool function_has_memory_sync) {
  if (!inst->IsLoad()) {
    return false;
  }

  Instruction* base_ptr = inst->GetBaseAddress();
  if (base_ptr->opcode() != SpvOpVariable) {
    return true;
  }

  if (base_ptr->IsReadOnlyVariable()) {
    return false;
  }

  if (function_has_memory_sync) {
    return true;
  }

  if (base_ptr->GetSingleWordInOperand(0) != SpvStorageClassUniform) {
    return true;
  }

  return HasPossibleStore(base_ptr);
}

bool CodeSinkingPass::HasMemorySync(Function* function) {
  return function->WhileEachInst([](Instruction* inst) {
    switch(inst->opcode()) {
      case SpvOpControlBarrier:
        return inst->GetSingleWordInOperand(2) != SpvMemorySemanticsMaskNone;
      case SpvOpMemoryBarrier:
        return true;
      case SpvOpAtomicLoad:
      case SpvOpAtomicStore:
      case SpvOpAtomicExchange:
      case SpvOpAtomicIIncrement:
      case SpvOpAtomicIDecrement:
      case SpvOpAtomicIAdd:
      case SpvOpAtomicISub:
      case SpvOpAtomicSMin:
      case SpvOpAtomicUMin:
      case SpvOpAtomicSMax:
      case SpvOpAtomicUMax:
      case SpvOpAtomicAnd:
      case SpvOpAtomicOr:
      case SpvOpAtomicXor:
      case SpvOpAtomicFlagTestAndSet:
      case SpvOpAtomicFlagClear:
        return inst->GetSingleWordInOperand(2) != SpvMemorySemanticsMaskNone;
      case SpvOpAtomicCompareExchange:
      case SpvOpAtomicCompareExchangeWeak:
        return inst->GetSingleWordInOperand(2) != SpvMemorySemanticsMaskNone ||
            inst->GetSingleWordInOperand(3) != SpvMemorySemanticsMaskNone;
      default:
        return false;
    }
  });
}
bool CodeSinkingPass::HasPossibleStore(Instruction* var_inst) {
  assert(var_inst->opcode() == SpvOpVariable || var_inst->opcode() == SpvOpAccessChain || var_inst->opcode() == SpvOpPtrAccessChain);

  return get_def_use_mgr()->WhileEachUser(var_inst, [this](Instruction* use) {
    switch (use->opcode()) {
      case SpvOpStore:
        return true;
      case SpvOpAccessChain:
      case SpvOpPtrAccessChain:
        return HasPossibleStore(use);
      default:
        return false;
    }
  });
}

bool CodeSinkingPass::IntersectsPath(uint32_t start,
                                     uint32_t end,
                                     const std::unordered_set<uint32_t>& set) {
  std::vector<uint32_t> worklist;
  worklist.push_back(start);
  std::unordered_set<uint32_t> already_done;
  already_done.insert(start);

  while(!worklist.empty()) {
    BasicBlock* bb = context()->get_instr_block(worklist.back());
    worklist.pop_back();

    if (bb->id() == end) {
      continue;
    }

    if (set.count(bb->id())) {
      return true;
    }

    bb->ForEachSuccessorLabel([&already_done, &worklist](uint32_t* succ_bb_id) {
      if (already_done.insert(*succ_bb_id).second) {
        worklist.push_back(*succ_bb_id);
      }
    });
  }
  return false;
}

// namespace opt

}  // namespace opt
}  // namespace spvtools
