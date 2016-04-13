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
#include <utility>

using std::get;
using std::make_tuple;
using std::make_pair;
using std::numeric_limits;
using std::pair;
using std::transform;
using std::tuple;
using std::unordered_map;
using std::vector;

using libspirv::BasicBlock;

namespace libspirv {

namespace {

using bb_ptr = BasicBlock *;
using cbb_ptr = const BasicBlock *;
using bb_iter = vector<BasicBlock *>::const_iterator;
enum                    {kBlock, kIter  , kEnd };
using stack_info = tuple<cbb_ptr, bb_iter, bb_iter>;

stack_info CreateStack(const BasicBlock &block) {
  return make_tuple(&block, begin(block.get_successors()),
                    end(block.get_successors()));
}

template <typename T>
bool Contains(const vector<T> &vec, T val) {
  return find(begin(vec), end(vec), val) == end(vec);
}

}

vector<const BasicBlock*>
PostOrderSort(const BasicBlock &entry, size_t size) {
  vector<cbb_ptr> out;
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

vector< pair<BasicBlock*, BasicBlock*> >
CalculateDominators(const BasicBlock &first_block) {
  vector<cbb_ptr> postorder = PostOrderSort(first_block, 10);
  const size_t undefined_dom = static_cast<uint32_t>(postorder.size());

  enum block_info           { DOM , INDEX };
  using block_detail = pair<size_t, size_t>;

  // pair(Block, postorder index of dominator)
  unordered_map<cbb_ptr, block_detail> idoms;
  for(size_t i = 0; i < postorder.size(); i++) {
    idoms[postorder[i]] = make_pair(undefined_dom, i);
  }

  get<DOM>(idoms[postorder.back()]) = get<INDEX>(idoms[postorder.back()]);

  bool changed = true;
  while (changed) {
    changed = false;
    for(auto b = postorder.rbegin() + 1; b != postorder.rend(); b++) {
      //printf("processing: %d\n", (*b)->get_id());
      size_t &b_dom = get<DOM>(idoms[*b]);
      const vector<BasicBlock*>& predecessors = (*b)->get_predecessors();

      // first processed predecessor
      BasicBlock* idom = *find_if(begin(predecessors),
                                      end(predecessors),
                                      [&idoms, undefined_dom] (BasicBlock* pred) {
                                        return get<DOM>(idoms[pred]) != undefined_dom;
                                      });
      size_t idom_idx = get<INDEX>(idoms[idom]);

      // all other predecessors
      for(auto p : predecessors) {
        if(idom == p) { continue; }
        if(get<DOM>(idoms[p]) != undefined_dom) {
          size_t finger1 = get<INDEX>(idoms[p]);
          size_t finger2 = idom_idx;
          while(finger1 != finger2) {
            while (finger1 < finger2) {finger1 = get<DOM>(idoms[postorder[finger1]]);}
            while (finger2 < finger1) {finger2 = get<DOM>(idoms[postorder[finger2]]);}
          }
          idom_idx = finger1;
        }
      }
      if(b_dom != idom_idx) {
        b_dom = idom_idx;
        changed = true;
      }
    }
  }

  vector<pair<bb_ptr, bb_ptr> > out;
  for(auto idom : idoms) {
    // NOTE: performing a const cast for convenience usage with UpdateImmediateDominators
    out.push_back({const_cast<BasicBlock*>(get<0>(idom)),
                   const_cast<BasicBlock*>(postorder[get<DOM>(get<1>(idom))])});
  }
  return out;
}

void
UpdateImmediateDominators(vector<pair<bb_ptr, bb_ptr> >& dom_edges) {
  for(auto &edge : dom_edges) {
    get<0>(edge)->SetImmediateDominator(get<1>(edge));
  }
}

void printDominatorList(BasicBlock &b) {
  std::cout << b.get_id() << " is dominated by: ";
  const BasicBlock* bb = &b;
  while (bb->GetImmediateDominator() != bb) {
    bb = bb->GetImmediateDominator();
    std::cout << bb->get_id() << " ";
  }
}

// TODO(umar): Support for merge instructions
// TODO(umar): Structured control flow checks
spv_result_t CfgPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst) {
  SpvOp opcode = static_cast<SpvOp>(inst->opcode);
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
