// Copyright (c) 2023 Google Inc.
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

#include "source/opt/invocation_interlock_placement_pass.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <optional>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/enum_set.h"
#include "source/enum_string_mapping.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/spirv_target_env.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {

namespace {
constexpr uint32_t kEntryPointExecutionModelInIdx = 0;
constexpr uint32_t kEntryPointFunctionIdInIdx = 1;
constexpr uint32_t kFunctionCallFunctionIdInIdx = 0;

template <typename T, typename U>
constexpr bool has(T container, U value) {
  return container.find(value) != container.end();
}
}  // namespace

bool InvocationInterlockPlacementPass::hasSingleNextBlock(uint32_t block_id,
                                                          bool forward_flow) {
  if (forward_flow) {
    // We are traversing forward, so check whether there is a single successor.
    BasicBlock* block = cfg()->block(block_id);

    return block->tail()->opcode() != spv::Op::OpBranchConditional &&
           block->tail()->opcode() != spv::Op::OpSwitch;
  } else {
    // We are traversing backward, so check whether there is a single
    // predecessor.
    return cfg()->preds(block_id).size() == 1;
  }
}

void InvocationInterlockPlacementPass::forEachNext(
    uint32_t block_id, bool forward_flow, std::function<void(uint32_t)> f) {
  if (forward_flow) {
    BasicBlock* block = cfg()->block(block_id);

    block->ForEachSuccessorLabel([f](uint32_t succ_id) { f(succ_id); });
  } else {
    for (uint32_t pred_id : cfg()->preds(block_id)) {
      f(pred_id);
    }
  }
}

void InvocationInterlockPlacementPass::addInstructionAtEdge(BasicBlock* block,
                                                            bool forward_flow) {
  if (forward_flow) {
    // Insert a begin instruction at the end of the block.
    Instruction* begin_inst =
        new Instruction(context(), spv::Op::OpBeginInvocationInterlockEXT);
    begin_inst->InsertAfter(&*--block->tail());
  } else {
    // Insert an end instruction at the end of the block.
    Instruction* end_inst =
        new Instruction(context(), spv::Op::OpEndInvocationInterlockEXT);
    end_inst->InsertBefore(&*block->begin());
  }
}

bool InvocationInterlockPlacementPass::killDuplicateBegin(BasicBlock* block) {
  bool found = false;

  return context()->KillInstructionIf(
      block->begin(), block->end(), [&found](Instruction* inst) {
        if (inst->opcode() == spv::Op::OpBeginInvocationInterlockEXT) {
          if (found) {
            return true;
          }
          found = true;
        }
        return false;
      });
}

bool InvocationInterlockPlacementPass::killDuplicateEnd(BasicBlock* block) {
  std::vector<Instruction*> to_kill;
  block->ForEachInst([&to_kill](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpEndInvocationInterlockEXT) {
      to_kill.push_back(inst);
    }
  });

  if (to_kill.size() <= 1) {
    return false;
  }

  to_kill.pop_back();

  for (Instruction* inst : to_kill) {
    context()->KillInst(inst);
  }

  return true;
}

InvocationInterlockPlacementPass::ExtractionResult
InvocationInterlockPlacementPass::removeInstructionsFromFunction(
    Function* func) {
  if (has(extracted_functions_, func)) {
    return extracted_functions_[func];
  }

  bool had_begin = false;
  bool had_end = false;

  func->ForEachInst([this, func, &had_begin, &had_end](Instruction* inst) {
    switch (inst->opcode()) {
      case spv::Op::OpBeginInvocationInterlockEXT:
        context()->KillInst(inst);
        had_begin = true;
        break;
      case spv::Op::OpEndInvocationInterlockEXT:
        context()->KillInst(inst);
        had_end = true;
        break;
      case spv::Op::OpFunctionCall: {
        uint32_t function_id =
            inst->GetSingleWordInOperand(kFunctionCallFunctionIdInIdx);
        Function* inner_func = context()->GetFunction(function_id);
        if (inner_func == func) {
          // ignore recursive calls
          return;
        }
        ExtractionResult result = removeInstructionsFromFunction(inner_func);
        had_begin = had_begin || result.had_begin;
        had_end = had_end || result.had_end;
        break;
      }
      default:
        break;
    }
  });

  ExtractionResult result = {had_begin, had_end};
  extracted_functions_[func] = result;
  return result;
}

bool InvocationInterlockPlacementPass::extractInstructionsFromCalls(
    std::vector<BasicBlock*> blocks) {
  bool modified = false;

  for (BasicBlock* block : blocks) {
    block->ForEachInst([this, &modified](Instruction* inst) {
      if (inst->opcode() == spv::Op::OpFunctionCall) {
        uint32_t function_id =
            inst->GetSingleWordInOperand(kFunctionCallFunctionIdInIdx);
        Function* func = context()->GetFunction(function_id);
        ExtractionResult result = removeInstructionsFromFunction(func);

        if (result.had_begin) {
          Instruction* new_inst = new Instruction(
              context(), spv::Op::OpBeginInvocationInterlockEXT);
          new_inst->InsertBefore(inst);
          modified = true;
        }
        if (result.had_end) {
          Instruction* new_inst =
              new Instruction(context(), spv::Op::OpEndInvocationInterlockEXT);
          new_inst->InsertAfter(inst);
          modified = true;
        }
      }
    });
  }
  return modified;
}

void InvocationInterlockPlacementPass::computeBeginAndEnd(
    std::vector<BasicBlock*> blocks) {
  for (BasicBlock* block : blocks) {
    block->ForEachInst([this, block](Instruction* inst) {
      switch (inst->opcode()) {
        case spv::Op::OpBeginInvocationInterlockEXT:
          begin_.insert(block->id());
          break;
        case spv::Op::OpEndInvocationInterlockEXT:
          end_.insert(block->id());
          break;
        default:
          break;
      }
    });
  }
}

void InvocationInterlockPlacementPass::computeReachableBlocks(
    BlockSet& inside, BlockSet& previous_inside, bool forward_flow) {
  BlockSet& starting_nodes = forward_flow ? begin_ : end_;
  inside = starting_nodes;

  std::deque<uint32_t> worklist;
  worklist.insert(worklist.begin(), starting_nodes.begin(),
                  starting_nodes.end());

  while (!worklist.empty()) {
    uint32_t block_id = worklist.front();
    worklist.pop_front();

    forEachNext(block_id, forward_flow,
                [&inside, &previous_inside, &worklist](uint32_t next_id) {
                  previous_inside.insert(next_id);
                  if (!has(inside, next_id)) {
                    worklist.push_back(next_id);
                  }
                  inside.insert(next_id);
                });
  }
}

bool InvocationInterlockPlacementPass::removeUnneededInstructions(
    BasicBlock* block) {
  bool modified = false;
  if (!has(predecessors_after_begin_, block->id()) &&
      has(after_begin_, block->id())) {
    // None of the previous blocks are in the critical section, but this block
    // is. This can only happen if this block already has at least one begin
    // instruction. Leave the first begin instruction, and remove any others.
    modified |= killDuplicateBegin(block);
  } else if (has(predecessors_after_begin_, block->id())) {
    // At least one previous block is in the critical section; remove all
    // begin instructions in this block.
    modified |= context()->KillInstructionIf(
        block->begin(), block->end(), [](Instruction* inst) {
          return inst->opcode() == spv::Op::OpBeginInvocationInterlockEXT;
        });
  }

  if (!has(successors_before_end_, block->id()) &&
      has(before_end_, block->id())) {
    // Same as above
    modified |= killDuplicateEnd(block);
  } else if (has(successors_before_end_, block->id())) {
    modified |= context()->KillInstructionIf(
        block->begin(), block->end(), [](Instruction* inst) {
          return inst->opcode() == spv::Op::OpEndInvocationInterlockEXT;
        });
  }
  return modified;
}

BasicBlock* InvocationInterlockPlacementPass::splitEdge(BasicBlock* block,
                                                        uint32_t succ_id) {
  // Create a new block to replace the critical edge.
  auto new_succ_temp = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, TakeNextId(),
                              std::initializer_list<Operand>{}));
  auto* new_succ = new_succ_temp.get();

  // Insert the new block into the function.
  block->GetParent()->InsertBasicBlockAfter(std::move(new_succ_temp), block);

  new_succ->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpBranch, 0, 0,
      std::initializer_list<Operand>{
          Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {succ_id})}));

  assert(block->tail()->opcode() == spv::Op::OpBranchConditional ||
         block->tail()->opcode() == spv::Op::OpSwitch);

  // Update the first branch to successor to instead branch to
  // new_successor
  block->tail()->WhileEachInId([new_succ, succ_id](uint32_t* branch_id) {
    if (*branch_id == succ_id) {
      *branch_id = new_succ->id();
      return false;
    }
    return true;
  });

  return new_succ;
}

bool InvocationInterlockPlacementPass::placeInstructionsForEdge(
    BasicBlock* block, uint32_t next_id, bool forward_flow) {
  bool modified = false;

  BlockSet& inside = forward_flow ? after_begin_ : before_end_;
  BlockSet& previous_inside =
      forward_flow ? predecessors_after_begin_ : successors_before_end_;

  if (has(previous_inside, next_id) && !has(inside, block->id())) {
    // This block is not in the critical section but the next block is, so
    // we need to add begin or end instructions to the edge.

    modified = true;

    if (hasSingleNextBlock(block->id(), forward_flow)) {
      // This is the only next block.
      addInstructionAtEdge(block, forward_flow);
    } else {
      // This block has multiple next blocks. Split the edge and insert the
      // instruction in the new next block.
      BasicBlock* new_branch;
      if (forward_flow) {
        new_branch = splitEdge(block, next_id);
      } else {
        new_branch = splitEdge(cfg()->block(next_id), block->id());
      }

      auto inst = new Instruction(
          context(), forward_flow ? spv::Op::OpBeginInvocationInterlockEXT
                                  : spv::Op::OpEndInvocationInterlockEXT);
      inst->InsertBefore(&*new_branch->tail());
    }
  }

  return modified;
}

bool InvocationInterlockPlacementPass::placeInstructions(BasicBlock* block) {
  bool modified = false;

  block->ForEachSuccessorLabel([this, block, &modified](uint32_t succ_id) {
    modified |=
        placeInstructionsForEdge(block, succ_id, /* flowForward= */ true);
    modified |= placeInstructionsForEdge(cfg()->block(succ_id), block->id(),
                                         /* flowForward= */ false);
  });

  return modified;
}

bool InvocationInterlockPlacementPass::processFragmentShaderEntry(
    Function* entry_func) {
  bool modified = false;

  // Save the original order of blocks in the function, so we don't iterate over
  // newly-added blocks.
  std::vector<BasicBlock*> original_blocks;
  for (auto bi = entry_func->begin(); bi != entry_func->end(); ++bi) {
    original_blocks.push_back(&*bi);
  }

  modified |= extractInstructionsFromCalls(original_blocks);
  computeBeginAndEnd(original_blocks);

  computeReachableBlocks(after_begin_, predecessors_after_begin_,
                         /* forward_flow= */ true);
  computeReachableBlocks(before_end_, successors_before_end_,
                         /* forward_flow= */ false);

  for (BasicBlock* block : original_blocks) {
    modified |= removeUnneededInstructions(block);
    modified |= placeInstructions(block);
  }
  return modified;
}

Pass::Status InvocationInterlockPlacementPass::Process() {
  // Skip this pass if the necessary extension or capability is missing
  if (!context()->get_feature_mgr()->HasExtension(
          kSPV_EXT_fragment_shader_interlock) ||
      !(context()->get_feature_mgr()->HasCapability(
            spv::Capability::FragmentShaderSampleInterlockEXT) ||
        context()->get_feature_mgr()->HasCapability(
            spv::Capability::FragmentShaderPixelInterlockEXT) ||
        context()->get_feature_mgr()->HasCapability(
            spv::Capability::FragmentShaderShadingRateInterlockEXT)))
    return Status::SuccessWithoutChange;

  bool modified = false;

  for (Instruction& entry_inst : context()->module()->entry_points()) {
    uint32_t entry_id =
        entry_inst.GetSingleWordInOperand(kEntryPointFunctionIdInIdx);
    Function* entry_func = context()->GetFunction(entry_id);

    auto execution_model = spv::ExecutionModel(
        entry_inst.GetSingleWordInOperand(kEntryPointExecutionModelInIdx));

    if (execution_model != spv::ExecutionModel::Fragment) {
      continue;
    }

    modified |= processFragmentShaderEntry(entry_func);
  }

  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
