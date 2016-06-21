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

#include <cassert>

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "val/BasicBlock.h"
#include "val/Construct.h"
#include "val/Function.h"
#include "val/ValidationState.h"

using std::find;
using std::function;
using std::get;
using std::ignore;
using std::make_pair;
using std::numeric_limits;
using std::pair;
using std::set;
using std::string;
using std::tie;
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

struct block_info {
  cbb_ptr block;  ///< pointer to the block
  bb_iter iter;   ///< Iterator to the current child node being processed
};

/// Returns true if a block with @p id is found in the @p work_list vector
///
/// @param[in] work_list Set of blocks visited in the the depth first traversal
///                   of the CFG
/// @param[in] id The ID of the block being checked
/// @return true if the edge work_list.back().block->get_id() => id is a
/// back-edge
bool FindInWorkList(const vector<block_info>& work_list, uint32_t id) {
  for (const auto b : work_list) {
    if (b.block->get_id() == id) return true;
  }
  return false;
}

/// @brief Depth first traversal starting from the \p entry BasicBlock
///
/// This function performs a depth first traversal from the \p entry
/// BasicBlock and calls the pre/postorder functions when it needs to process
/// the node in pre order, post order. It also calls the backedge function
/// when a back edge is encountered
///
/// @param[in] entry The root BasicBlock of a CFG tree
/// @param[in] successor_func  A function which will return a pointer to the
///                            successor nodes
/// @param[in] preorder   A function that will be called for every block in a
///                       CFG following preorder traversal semantics
/// @param[in] postorder  A function that will be called for every block in a
///                       CFG following postorder traversal semantics
/// @param[in] backedge   A function that will be called when a backedge is
///                       encountered during a traversal
/// NOTE: The @p successor_func return a pointer to a collection such that
/// iterators to that collection remain valid for the lifetime of the algorithm
void DepthFirstTraversal(const BasicBlock& entry,
                         get_blocks_func successor_func,
                         function<void(cbb_ptr)> preorder,
                         function<void(cbb_ptr)> postorder,
                         function<void(cbb_ptr, cbb_ptr)> backedge) {
  vector<cbb_ptr> out;
  unordered_set<uint32_t> processed;
  /// NOTE: work_list is the sequence of nodes from the entry node to the node
  /// being processed in the traversal
  vector<block_info> work_list;

  work_list.reserve(10);
  work_list.push_back({&entry, begin(*successor_func(&entry))});
  preorder(&entry);

  while (!work_list.empty()) {
    block_info& top = work_list.back();
    if (top.iter == end(*successor_func(top.block))) {
      postorder(top.block);
      work_list.pop_back();
    } else {
      BasicBlock* child = *top.iter;
      top.iter++;
      if (FindInWorkList(work_list, child->get_id())) {
        backedge(top.block, child);
      }
      if (processed.count(child->get_id()) == 0) {
        preorder(child);
        work_list.emplace_back(
            block_info{child, begin(*successor_func(child))});
        processed.insert(child->get_id());
      }
    }
  }
}

/// Returns the successor of a basic block.
/// NOTE: This will be passed as a function pointer to when calculating
/// the dominator and post dominator
const vector<BasicBlock*>* successor(const BasicBlock* b) {
  return b->get_successors();
}

const vector<BasicBlock*>* predecessor(const BasicBlock* b) {
  return b->get_predecessors();
}

}  // namespace

vector<pair<BasicBlock*, BasicBlock*>> CalculateDominators(
    const vector<cbb_ptr>& postorder, get_blocks_func predecessor_func) {
  struct block_detail {
    size_t dominator;  ///< The index of blocks's dominator in post order array
    size_t postorder_index;  ///< The index of the block in the post order array
  };

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
      const vector<BasicBlock*>* predecessors = predecessor_func(*b);

      // first processed predecessor
      auto res = find_if(begin(*predecessors), end(*predecessors),
                         [&idoms, undefined_dom](BasicBlock* pred) {
                           return idoms[pred].dominator != undefined_dom;
                         });
      assert(res != end(*predecessors));
      BasicBlock* idom = *res;
      size_t idom_idx = idoms[idom].postorder_index;

      // all other predecessors
      for (auto p : *predecessors) {
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

void UpdateImmediateDominators(
    const vector<pair<bb_ptr, bb_ptr>>& dom_edges,
    function<void(BasicBlock*, BasicBlock*)> set_func) {
  for (auto& edge : dom_edges) {
    set_func(get<0>(edge), get<1>(edge));
  }
}

void printDominatorList(const BasicBlock& b) {
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
  if (_.get_current_function().IsBlockType(merge_block, kBlockTypeMerge)) {
    return _.diag(SPV_ERROR_INVALID_CFG)
           << "Block " << _.getIdName(merge_block)
           << " is already a merge block for another header";
  }
  return SPV_SUCCESS;
}

/// Update the continue construct's exit blocks once the backedge blocks are
/// identified in the CFG.
void UpdateContinueConstructExitBlocks(
    Function& function, const vector<pair<uint32_t, uint32_t>>& back_edges) {
  auto& constructs = function.get_constructs();
  // TODO(umar): Think of a faster way to do this
  for (auto& edge : back_edges) {
    uint32_t back_edge_block_id;
    uint32_t loop_header_block_id;
    tie(back_edge_block_id, loop_header_block_id) = edge;

    auto is_this_header = [=](Construct& c) {
      return c.get_type() == ConstructType::kLoop &&
             c.get_entry()->get_id() == loop_header_block_id;
    };

    for (auto construct : constructs) {
      if (is_this_header(construct)) {
        Construct* continue_construct =
            construct.get_corresponding_constructs().back();
        assert(continue_construct->get_type() == ConstructType::kContinue);

        BasicBlock* back_edge_block;
        tie(back_edge_block, ignore) = function.GetBlock(back_edge_block_id);
        continue_construct->set_exit(back_edge_block);
      }
    }
  }
}

/// Constructs an error message for construct validation errors
string ConstructErrorString(const Construct& construct,
                            const string& header_string,
                            const string& exit_string,
                            bool post_dominate = false) {
  string construct_name;
  string header_name;
  string exit_name;
  string dominate_text;
  if (post_dominate) {
    dominate_text = "is not post dominated by";
  } else {
    dominate_text = "does not dominate";
  }

  switch (construct.get_type()) {
    case ConstructType::kSelection:
      construct_name = "selection";
      header_name = "selection header";
      exit_name = "merge block";
      break;
    case ConstructType::kLoop:
      construct_name = "loop";
      header_name = "loop header";
      exit_name = "merge block";
      break;
    case ConstructType::kContinue:
      construct_name = "continue";
      header_name = "continue target";
      exit_name = "back-edge block";
      break;
    case ConstructType::kCase:
      construct_name = "case";
      header_name = "case block";
      exit_name = "exit block";  // TODO(umar): there has to be a better name
      break;
    default:
      assert(1 == 0 && "Not defined type");
  }
  // TODO(umar): Add header block for continue constructs to error message
  return "The " + construct_name + " construct with the " + header_name + " " +
         header_string + " " + dominate_text + " the " + exit_name + " " +
         exit_string;
}

spv_result_t StructuredControlFlowChecks(
    const ValidationState_t& _, const Function& function,
    const vector<pair<uint32_t, uint32_t>>& back_edges) {
  /// Check all backedges target only loop headers and have exactly one
  /// back-edge branching to it
  set<uint32_t> loop_headers;
  for (auto back_edge : back_edges) {
    uint32_t back_edge_block;
    uint32_t header_block;
    tie(back_edge_block, header_block) = back_edge;
    if (!function.IsBlockType(header_block, kBlockTypeLoop)) {
      return _.diag(SPV_ERROR_INVALID_CFG)
             << "Back-edges (" << _.getIdName(back_edge_block) << " -> "
             << _.getIdName(header_block)
             << ") can only be formed between a block and a loop header.";
    }
    bool success;
    tie(ignore, success) = loop_headers.insert(header_block);
    if (!success) {
      // TODO(umar): List the back-edge blocks that are branching to loop
      // header
      return _.diag(SPV_ERROR_INVALID_CFG)
             << "Loop header " << _.getIdName(header_block)
             << " targeted by multiple back-edges";
    }
  }

  // Check construct rules
  for (const Construct& construct : function.get_constructs()) {
    auto header = construct.get_entry();
    auto merge = construct.get_exit();

    // if the merge block is reachable then it's dominated by the header
    if (merge->is_reachable() &&
        find(merge->dom_begin(), merge->dom_end(), header) ==
            merge->dom_end()) {
      return _.diag(SPV_ERROR_INVALID_CFG)
             << ConstructErrorString(construct, _.getIdName(header->get_id()),
                                     _.getIdName(merge->get_id()));
    }
    if (construct.get_type() == ConstructType::kContinue) {
      if (find(header->pdom_begin(), header->pdom_end(), merge) ==
          merge->pdom_end()) {
        return _.diag(SPV_ERROR_INVALID_CFG)
               << ConstructErrorString(construct, _.getIdName(header->get_id()),
                                       _.getIdName(merge->get_id()), true);
      }
    }
    // TODO(umar):  an OpSwitch block dominates all its defined case
    // constructs
    // TODO(umar):  each case construct has at most one branch to another
    // case construct
    // TODO(umar):  each case construct is branched to by at most one other
    // case construct
    // TODO(umar):  if Target T1 branches to Target T2, or if Target T1
    // branches to the Default and the Default branches to Target T2, then
    // T1 must immediately precede T2 in the list of the OpSwitch Target
    // operands
  }
  return SPV_SUCCESS;
}

spv_result_t PerformCfgChecks(ValidationState_t& _) {
  for (auto& function : _.get_functions()) {
    // Check all referenced blocks are defined within a function
    if (function.get_undefined_block_count() != 0) {
      string undef_blocks("{");
      for (auto undefined_block : function.get_undefined_blocks()) {
        undef_blocks += _.getIdName(undefined_block) + " ";
      }
      return _.diag(SPV_ERROR_INVALID_CFG)
             << "Block(s) " << undef_blocks << "\b}"
             << " are referenced but not defined in function "
             << _.getIdName(function.get_id());
    }

    // Updates each blocks immediate dominators
    vector<const BasicBlock*> postorder;
    vector<const BasicBlock*> postdom_postorder;
    vector<pair<uint32_t, uint32_t>> back_edges;
    if (auto* first_block = function.get_first_block()) {
      /// calculate dominators
      DepthFirstTraversal(*first_block, successor, [](cbb_ptr) {},
                          [&](cbb_ptr b) { postorder.push_back(b); },
                          [&](cbb_ptr from, cbb_ptr to) {
                            back_edges.emplace_back(from->get_id(),
                                                    to->get_id());
                          });
      auto edges = libspirv::CalculateDominators(postorder, predecessor);
      libspirv::UpdateImmediateDominators(
          edges, [](bb_ptr block, bb_ptr dominator) {
            block->SetImmediateDominator(dominator);
          });

      /// calculate post dominators
      auto exit_block = function.get_pseudo_exit_block();
      DepthFirstTraversal(*exit_block, predecessor, [](cbb_ptr) {},
                          [&](cbb_ptr b) { postdom_postorder.push_back(b); },
                          [&](cbb_ptr, cbb_ptr) {});
      auto postdom_edges =
          libspirv::CalculateDominators(postdom_postorder, successor);
      libspirv::UpdateImmediateDominators(
          postdom_edges, [](bb_ptr block, bb_ptr dominator) {
            block->SetImmediatePostDominator(dominator);
          });
    }
    UpdateContinueConstructExitBlocks(function, back_edges);

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

    /// Structured control flow checks are only required for shader capabilities
    if (_.hasCapability(SpvCapabilityShader)) {
      spvCheckReturn(StructuredControlFlowChecks(_, function, back_edges));
    }
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
