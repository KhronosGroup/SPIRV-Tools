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
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::find;
using std::get;
using std::make_pair;
using std::numeric_limits;
using std::pair;
using std::transform;
using std::unordered_map;
using std::unordered_set;
using std::vector;

using libspirv::BasicBlock;

namespace libspirv {

namespace {

using bb_ptr = BasicBlock*;
using cbb_ptr = const BasicBlock*;
using bb_iter = vector<BasicBlock*>::const_iterator;

/// @brief Sorts the blocks in a CFG given the entry node
///
/// Returns a vector of basic block pointers in a Control Flow Graph(CFG) which
/// are sorted in the order they were accessed in a post order traversal.
///
/// @param[in] entry the first block of a CFG
/// @param[in] depth_hint a hint about the depth of the CFG
///
/// @return A vector of pointers in the order they were access in a post order
/// traversal
vector<const BasicBlock*> PostOrderSort(const BasicBlock& entry, size_t size) {
  struct block_info {
    cbb_ptr block;
    bb_iter iter;
  };

  vector<cbb_ptr> out;
  vector<block_info> staged;
  unordered_set<uint32_t> processed;

  staged.reserve(size);
  staged.emplace_back(block_info{&entry, begin(entry.get_successors())});
  processed.insert(entry.get_id());

  while (!staged.empty()) {
    block_info& top = staged.back();
    if (top.iter == end(top.block->get_successors())) {
      out.push_back(top.block);
      staged.pop_back();
    } else {
      BasicBlock* child = *top.iter;
      top.iter++;
      if (processed.find(child->get_id()) == end(processed)) {
        staged.emplace_back(block_info{child, begin(child->get_successors())});
        processed.insert(child->get_id());
      }
    }
  }
  return out;
}
}  // namespace

vector<pair<BasicBlock*, BasicBlock*>> CalculateDominators(
    const BasicBlock& first_block) {
  struct block_detail {
    size_t dominator;  ///< The index of blocks's dominator in post order array
    size_t postorder_index;  ///< The index of the block in the post order array
  };

  vector<cbb_ptr> postorder = PostOrderSort(first_block, 10);
  const size_t undefined_dom = static_cast<size_t>(postorder.size());

  unordered_map<cbb_ptr, block_detail> idoms;
  for (size_t i = 0; i < postorder.size(); i++) {
    idoms[postorder[i]] = {undefined_dom, i};
  }

  idoms[postorder.back()].dominator = idoms[postorder.back()].postorder_index;

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto b = postorder.rbegin() + 1; b != postorder.rend(); b++) {
      size_t& b_dom = idoms[*b].dominator;
      const vector<BasicBlock*>& predecessors = (*b)->get_predecessors();

      // first processed predecessor
      auto res = find_if(begin(predecessors), end(predecessors),
                         [&idoms, undefined_dom](BasicBlock* pred) {
                           return idoms[pred].dominator != undefined_dom;
                         });
      assert(res != end(predecessors));
      BasicBlock* idom = *res;
      size_t idom_idx = idoms[idom].postorder_index;

      // all other predecessors
      for (auto p : predecessors) {
        if (idom == p || p->is_reachable() == false) {
          continue;
        }
        if (idoms[p].dominator != undefined_dom) {
          size_t finger1 = idoms[p].postorder_index;
          size_t finger2 = idom_idx;
          while (finger1 != finger2) {
            while (finger1 < finger2) {
              finger1 = idoms[postorder[finger1]].dominator;
            }
            while (finger2 < finger1) {
              finger2 = idoms[postorder[finger2]].dominator;
            }
          }
          idom_idx = finger1;
        }
      }
      if (b_dom != idom_idx) {
        b_dom = idom_idx;
        changed = true;
      }
    }
  }

  vector<pair<bb_ptr, bb_ptr>> out;
  for (auto idom : idoms) {
    // NOTE: performing a const cast for convenient usage with
    // UpdateImmediateDominators
    out.push_back({const_cast<BasicBlock*>(get<0>(idom)),
                   const_cast<BasicBlock*>(postorder[get<1>(idom).dominator])});
  }
  return out;
}

void UpdateImmediateDominators(vector<pair<bb_ptr, bb_ptr>>& dom_edges) {
  for (auto& edge : dom_edges) {
    get<0>(edge)->SetImmediateDominator(get<1>(edge));
  }
}

void printDominatorList(BasicBlock& b) {
  std::cout << b.get_id() << " is dominated by: ";
  const BasicBlock* bb = &b;
  while (bb->GetImmediateDominator() != bb) {
    bb = bb->GetImmediateDominator();
    std::cout << bb->get_id() << " ";
  }
}

#define CFG_ASSERT(ASSERT_FUNC, TARGET) \
  if (spv_result_t rcode = ASSERT_FUNC(_, TARGET)) return rcode

spv_result_t FirstBlockAssert(ValidationState_t& _, uint32_t target) {
  if (_.get_current_function().IsFirstBlock(target)) {
    return _.diag(SPV_ERROR_INVALID_CFG)
           << "First block " << _.getIdName(target) << " of funciton "
           << _.getIdName(_.get_current_function().get_id())
           << " is targeted by block "
           << _.getIdName(
                  _.get_current_function().get_current_block()->get_id());
  }
  return SPV_SUCCESS;
}

spv_result_t MergeBlockAssert(ValidationState_t& _, uint32_t merge_block) {
  if (_.get_current_function().IsMergeBlock(merge_block)) {
    return _.diag(SPV_ERROR_INVALID_CFG)
           << "Block " << _.getIdName(merge_block)
           << " is already a merge block for another header";
  }
  return SPV_SUCCESS;
}

spv_result_t PerformCfgChecks(ValidationState_t& _) {
  for (auto& function : _.get_functions()) {
    // Updates each blocks immediate dominators
    if (auto* first_block = function.get_first_block()) {
      auto edges = libspirv::CalculateDominators(*first_block);
      libspirv::UpdateImmediateDominators(edges);
    }

    // Check if the order of blocks in the binary appear before the blocks they
    // dominate
    auto& blocks = function.get_blocks();
    if (blocks.empty() == false) {
      for (auto block = begin(blocks) + 1; block != end(blocks); block++) {
        if (auto idom = (*block)->GetImmediateDominator()) {
          if (block == std::find(begin(blocks), block, idom)) {
            return _.diag(SPV_ERROR_INVALID_CFG)
                   << "Block " << _.getIdName((*block)->get_id())
                   << " appears in the binary before its dominator "
                   << _.getIdName(idom->get_id());
          }
        }
      }
    }

    // Check all referenced blocks are defined within a function
    if (function.get_undefined_block_count() != 0) {
      std::stringstream ss;
      ss << "{";
      for (auto undefined_block : function.get_undefined_blocks()) {
        ss << _.getIdName(undefined_block) << " ";
      }
      return _.diag(SPV_ERROR_INVALID_CFG)
             << "Block(s) " << ss.str() << "\b}"
             << " are referenced but not defined in function "
             << _.getIdName(function.get_id());
    }

    // Check all headers dominate their merge blocks
    for (CFConstruct& construct : function.get_constructs()) {
      auto header = construct.get_header();
      auto merge = construct.get_merge();
      // auto cont = construct.get_continue();

      if (merge->is_reachable() &&
          find(merge->dom_begin(), merge->dom_end(), header) ==
              merge->dom_end()) {
        return _.diag(SPV_ERROR_INVALID_CFG)
               << "Header block " << _.getIdName(header->get_id())
               << " doesn't dominate its merge block "
               << _.getIdName(merge->get_id());
      }
    }

    // TODO(umar): All CFG back edges must branch to a loop header, with each
    // loop header having exactly one back edge branching to it

    // TODO(umar): For a given loop, its back-edge block must post dominate the
    // OpLoopMerge's Continue Target, and that Continue Target must dominate the
    // back-edge block
  }
  return SPV_SUCCESS;
}

spv_result_t CfgPass(ValidationState_t& _,
                     const spv_parsed_instruction_t* inst) {
  SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  switch (opcode) {
    case SpvOpLabel:
      spvCheckReturn(_.get_current_function().RegisterBlock(inst->result_id));
      break;
    case SpvOpLoopMerge: {
      // TODO(umar): mark current block as a loop header
      uint32_t merge_block = inst->words[inst->operands[0].offset];
      uint32_t continue_block = inst->words[inst->operands[1].offset];
      CFG_ASSERT(MergeBlockAssert, merge_block);

      spvCheckReturn(_.get_current_function().RegisterLoopMerge(
          merge_block, continue_block));
    } break;
    case SpvOpSelectionMerge: {
      uint32_t merge_block = inst->words[inst->operands[0].offset];
      CFG_ASSERT(MergeBlockAssert, merge_block);

      spvCheckReturn(
          _.get_current_function().RegisterSelectionMerge(merge_block));
    } break;
    case SpvOpBranch: {
      uint32_t target = inst->words[inst->operands[0].offset];
      CFG_ASSERT(FirstBlockAssert, target);

      _.get_current_function().RegisterBlockEnd({target}, opcode);
    } break;
    case SpvOpBranchConditional: {
      uint32_t tlabel = inst->words[inst->operands[1].offset];
      uint32_t flabel = inst->words[inst->operands[2].offset];
      CFG_ASSERT(FirstBlockAssert, tlabel);
      CFG_ASSERT(FirstBlockAssert, flabel);

      _.get_current_function().RegisterBlockEnd({tlabel, flabel}, opcode);
    } break;

    case SpvOpSwitch: {
      vector<uint32_t> cases;
      for (int i = 1; i < inst->num_operands; i += 2) {
        uint32_t target = inst->words[inst->operands[i].offset];
        CFG_ASSERT(FirstBlockAssert, target);
        cases.push_back(target);
      }
      _.get_current_function().RegisterBlockEnd({cases}, opcode);
    } break;
    case SpvOpKill:
    case SpvOpReturn:
    case SpvOpReturnValue:
    case SpvOpUnreachable:
      _.get_current_function().RegisterBlockEnd({}, opcode);
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}
}  // namespace libspirv
