// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "pass.h"

#include "cfa.h"
#include "iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kEntryPointFunctionIdInIdx = 1;
const uint32_t kLoopMergeContinueBlockIdInIdx = 1;
const uint32_t kLoopMergeMergeBlockIdInIdx = 0;
const uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
const uint32_t kTypePointerTypeIdInIdx = 1;

// Universal Limit of ResultID + 1
const int kInvalidId = 0x400000;

}  // namespace

Pass::Pass()
    : pseudo_entry_block_(std::unique_ptr<ir::Instruction>(
          new ir::Instruction(SpvOpLabel, 0, 0, {}))),
      pseudo_exit_block_(std::unique_ptr<ir::Instruction>(
          new ir::Instruction(SpvOpLabel, 0, kInvalidId, {}))),
      consumer_(nullptr),
      def_use_mgr_(nullptr),
      next_id_(0),
      context_(nullptr) {}

void Pass::AddCalls(ir::Function* func, std::queue<uint32_t>* todo) {
  for (auto bi = func->begin(); bi != func->end(); ++bi)
    for (auto ii = bi->begin(); ii != bi->end(); ++ii)
      if (ii->opcode() == SpvOpFunctionCall)
        todo->push(ii->GetSingleWordInOperand(0));
}

bool Pass::ProcessEntryPointCallTree(ProcessFunction& pfn, ir::Module* module) {
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;
  for (auto& fn : *module) id2function[fn.result_id()] = &fn;

  // Collect all of the entry points as the roots.
  std::queue<uint32_t> roots;
  for (auto& e : module->entry_points())
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  return ProcessCallTreeFromRoots(pfn, id2function, &roots);
}

bool Pass::ProcessReachableCallTree(ProcessFunction& pfn,
                                    ir::IRContext* irContext) {
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;
  for (auto& fn : *irContext->module()) id2function[fn.result_id()] = &fn;

  std::queue<uint32_t> roots;

  // Add all entry points since they can be reached from outside the module.
  for (auto& e : irContext->module()->entry_points())
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));

  // Add all exported functions since they can be reached from outside the
  // module.
  for (auto& a : irContext->annotations()) {
    // TODO: Handle group decorations as well.  Currently not generate by any
    // front-end, but could be coming.
    if (a.opcode() == SpvOp::SpvOpDecorate) {
      if (a.GetSingleWordOperand(1) ==
          SpvDecoration::SpvDecorationLinkageAttributes) {
        uint32_t lastOperand = a.NumOperands() - 1;
        if (a.GetSingleWordOperand(lastOperand) ==
            SpvLinkageType::SpvLinkageTypeExport) {
          uint32_t id = a.GetSingleWordOperand(0);
          if (id2function.count(id) != 0) roots.push(id);
        }
      }
    }
  }

  return ProcessCallTreeFromRoots(pfn, id2function, &roots);
}

bool Pass::ProcessCallTreeFromRoots(
    ProcessFunction& pfn,
    const std::unordered_map<uint32_t, ir::Function*>& id2function,
    std::queue<uint32_t>* roots) {
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> done;

  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      ir::Function* fn = id2function.at(fi);
      modified = pfn(fn) || modified;
      AddCalls(fn, roots);
    }
  }
  return modified;
}

bool Pass::IsLoopHeader(ir::BasicBlock* block_ptr) const {
  auto iItr = block_ptr->end();
  --iItr;
  if (iItr == block_ptr->begin())
    return false;
  --iItr;
  return iItr->opcode() == SpvOpLoopMerge;
}

uint32_t Pass::GetPointeeTypeId(const ir::Instruction* ptrInst) const {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const ir::Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
  return ptrTypeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx);
}

void Pass::ComputeStructuredOrder(ir::Function* func, ir::BasicBlock* root,
                                  std::list<ir::BasicBlock*>* order) {
  // Compute structured successors and do DFS
  ComputeStructuredSuccessors(func);
  auto ignore_block = [](cbb_ptr) {};
  auto ignore_edge = [](cbb_ptr, cbb_ptr) {};
  auto get_structured_successors = [this](const ir::BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };

  // TODO(greg-lunarg): Get rid of const_cast by making moving const
  // out of the cfa.h prototypes and into the invoking code.
  auto post_order = [&](cbb_ptr b) {
    order->push_front(const_cast<ir::BasicBlock*>(b));
  };
  spvtools::CFA<ir::BasicBlock>::DepthFirstTraversal(
      root, get_structured_successors, ignore_block, post_order,
      ignore_edge);
}

void Pass::ComputeStructuredSuccessors(ir::Function *func) {
  block2structured_succs_.clear();
  for (auto& blk : *func) {
    // If no predecessors in function, make successor to pseudo entry
    if (label2preds_[blk.id()].size() == 0)
      block2structured_succs_[&pseudo_entry_block_].push_back(&blk);
    // If header, make merge block first successor and continue block second
    // successor if there is one.
    uint32_t cbid;
    const uint32_t mbid = MergeBlockIdIfAny(blk, &cbid);
    if (mbid != 0) {
      block2structured_succs_[&blk].push_back(id2block_[mbid]);
      if (cbid != 0)
        block2structured_succs_[&blk].push_back(id2block_[cbid]);
    }
    // add true successors
    blk.ForEachSuccessorLabel([&blk, this](uint32_t sbid) {
      block2structured_succs_[&blk].push_back(id2block_[sbid]);
    });
  }
}


uint32_t Pass::MergeBlockIdIfAny(const ir::BasicBlock& blk, uint32_t* cbid) {
  auto merge_ii = blk.cend();
  --merge_ii;
  if (cbid != nullptr) {
    *cbid = 0;
  }
  uint32_t mbid = 0;
  if (merge_ii != blk.cbegin()) {
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
}  // namespace opt
}  // namespace spvtools

