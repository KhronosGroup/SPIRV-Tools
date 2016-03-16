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

#include "validate.h"
#include "validate_passes.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>

using std::get;
using std::vector;
using std::transform;
using std::tuple;
using std::make_tuple;
using std::unordered_map;
using libspirv::BasicBlock;

namespace libspirv {

namespace {

using bb_ptr = BasicBlock *;
using bb_iter = vector<BasicBlock *>::iterator;
enum                    {kBlock, kIter  , kEnd };
using stack_info = tuple<bb_ptr, bb_iter, bb_iter>;

stack_info CreateStack(BasicBlock &block) {
  return make_tuple(&block, begin(block.get_out_blocks()),
                    end(block.get_out_blocks()));
}

template <typename T>
bool Contains(vector<T> &vec, T val) {
  return find(begin(vec), end(vec), val) == end(vec);
}
}

vector<BasicBlock*>
PostOrderSort(BasicBlock &entry, size_t size) {
  vector<bb_ptr> out;
  vector<stack_info> stack;
  vector<uint32_t> processed;

  stack.reserve(size);
  stack.emplace_back(CreateStack(entry));
  processed.push_back(entry.get_id());

  while (!stack.empty()) {
    stack_info &top = stack.back();
    if (get<kIter>(top) == get<kEnd>(top)) {
      // No children left to process
      out.push_back(get<kBlock>(top));
      stack.pop_back();
    } else {
      bb_iter &child_iter = get<kIter>(top);

      if (Contains(processed, (*child_iter)->get_id())) {
        // Process next child
        stack.emplace_back(CreateStack(**child_iter));
        processed.push_back((*child_iter)->get_id());
      }
      child_iter++;
    }
  }
  return out;
}

  //  *for all nodes, b /* initialize the dominators array */
  //  *  doms[b] ← Undefined
  //  *  doms[start node] ← start node
  //  *  Changed ← true
  //  *  while (Changed)
  //  *    Changed ← false
  //  *      for all nodes, b, in reverse postorder (except start node)
  //          new idom ← first (processed) predecessor of b /* (pick one) */
  //          for all other predecessors, p, of b
  //            if doms[p] = Undefined /* i.e., if doms[p] already calculated */
  //              new idom ← intersect(p, new idom)
  //            if doms[b] = new idom
  //              doms[b] ← new idom
  //              Changed ← true
  //    function intersect(b1, b2) returns node
  //      finger1 ← b1
  //      finger2 ← b2
  //      while (finger1 != finger2)
  //        while (finger1 < finger2)
  //          finger1 = doms[finger1]
  //        while (finger2 < finger1)
  //          finger2 = doms[finger2]
  //      return finger1

BasicBlock*
intersect(BasicBlock* b1, BasicBlock* b2) {
  BasicBlock* finger1 = b1;
  BasicBlock* finger2 = b2;
  while(finger1 != finger2) {
    while (finger1 < finger2) {
      finger1 = finger1->get_dominators().back();
    }
    while (finger2 < finger1) {
      finger2 = finger2->get_dominators().back();
    }
  }
  return finger1;
}

void
CalculateDominators(vector<BasicBlock*> &blocks) {
  vector<BasicBlock*> post_order = PostOrderSort(*blocks[0], 10);
  for(auto &block : post_order) {
    block->get_dominators().clear();
  }
  post_order[0]->get_dominators().push_back(post_order[0]);
  bool changed = true;

  vector<BasicBlock*> pred;
  pred.push_back(post_order[0]);

  while (changed) {
    changed = false;

    for(auto b = post_order.rbegin() + 1; b != post_order.rend(); b++) {
      if(post_order[0] == *b) {
        continue;
      }
      BasicBlock* new_idom = *b;
      for(auto p = pred.rbegin(); p != pred.rend(); p++) {
        if((*p)->get_dominators().empty()){
          new_idom = intersect(*p, new_idom);
        }
      }
    }
  }
}

// TODO(umar): Support for merge instructions
// TODO(umar): Structured control flow checks
spv_result_t CfgPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst) {
  SpvOp opcode = inst->opcode;
  switch (opcode) {
    case SpvOpLabel:
      spvCheckReturn(_.get_functions().RegisterBlock(inst->result_id));
      break;
    case SpvOpLoopMerge: {
      // TODO(umar): mark current block as a loop header
      uint32_t merge_block = inst->words[inst->operands[0].offset];
      uint32_t continue_block = inst->words[inst->operands[1].offset];

      if(_.get_functions().IsMergeBlock(merge_block)) {
        return _.diag(SPV_ERROR_INVALID_CFG)
          << "Block " << _.getIdName(merge_block) << " is already a merge block for another header";
      }

      spvCheckReturn(_.get_functions().RegisterLoopMerge(merge_block, continue_block));
    } break;
    case SpvOpSelectionMerge: {
      uint32_t merge_block = inst->words[inst->operands[0].offset];

      if(_.get_functions().IsMergeBlock(merge_block)) {
        return _.diag(SPV_ERROR_INVALID_CFG)
          << "Block " << _.getIdName(merge_block) << " is already a merge block for another header";
      }

      spvCheckReturn(_.get_functions().RegisterSelectionMerge(merge_block));
    } break;
    case SpvOpBranch:
      spvCheckReturn(_.get_functions().RegisterBlockEnd(inst->words[inst->operands[0].offset]));
      break;
    case SpvOpBranchConditional: {
      uint32_t tlabel = inst->words[inst->operands[1].offset];
      uint32_t flabel = inst->words[inst->operands[2].offset];

      spvCheckReturn(_.get_functions().RegisterBlockEnd({tlabel, flabel}));
    } break;

    case SpvOpSwitch: {
      vector<uint32_t> cases((inst->num_operands - 2) / 2);
        for(int i = 3; i < inst->num_operands; i += 2) {
          cases.push_back(inst->words[inst->operands[i].offset]);
        }
        spvCheckReturn(_.get_functions().RegisterBlockEnd(cases));
    } break;
    case SpvOpKill:
    case SpvOpReturn:
    case SpvOpReturnValue:
    case SpvOpUnreachable:
      spvCheckReturn(_.get_functions().RegisterBlockEnd());
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}
}
