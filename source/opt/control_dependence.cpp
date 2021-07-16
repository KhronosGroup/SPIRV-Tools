// Copyright (c) 2021 Google LLC.
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

#include "source/opt/control_dependence.h"

#include <cassert>
#include <tuple>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/cfg.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "spirv/unified1/spirv.h"

// Computes the control dependence graph (CDG). The algorithm in Cytron 1991,
// "Efficiently Computing Static Single Assignment Form and the Control
// Dependence Graph." It relies on the fact that the control dependees (blocks
// on which a block is control dependent on) are exactly the post-dominance
// frontier for that block. The explanation and proofs are given in Section 6 of
// that paper.
// Link: https://www.cs.utexas.edu/~pingali/CS380C/2010/papers/ssaCytron.pdf
//
// The algorithm in Section 4.2 of the same paper is used to construct the
// dominance frontier. It uses the post-dominance tree, which is available in
// the IR context).

namespace spvtools {
namespace opt {
namespace {
// Returns a list of (case value, label id) pairs associated with the given
// OpSwitch instruction.
std::vector<std::pair<uint32_t, uint32_t>> GetSwitchCases(
    const Instruction& inst) {
  assert(inst.opcode() == SpvOpSwitch);
  uint32_t num_labels = (inst.NumInOperands() - 2) / 2;
  std::vector<std::pair<uint32_t, uint32_t>> ret;
  ret.reserve(num_labels);
  for (uint32_t i = 0; i < num_labels; ++i) {
    ret.push_back(std::make_pair(inst.GetSingleWordInOperand(2 + 2 * i),
                                 inst.GetSingleWordInOperand(2 + 2 * i + 1)));
  }
  return ret;
}

// Returns the control dependence corresponding to the CFG edge between |source|
// and |target| (label IDs). Fails if there is no direct edge.
ControlDependence ClassifyControlDependence(const CFG& cfg, uint32_t source,
                                            uint32_t target) {
  ControlDependence dep;
  dep.source_bb_id = source;
  dep.target_bb_id = target;
  if (source == ControlDependenceAnalysis::kPseudoEntryBlock) {
    dep.dependence_type = ControlDependence::DependenceType::kEntry;
    return dep;
  }
  BasicBlock* bb = cfg.block(source);
  const Instruction& branch = *bb->rbegin();
  switch (branch.opcode()) {
    case SpvOpBranchConditional: {
      uint32_t label_true = branch.GetSingleWordInOperand(1);
      uint32_t label_false = branch.GetSingleWordInOperand(2);
      dep.dependence_type =
          ControlDependence::DependenceType::kConditionalBranch;
      dep.dependent_value_id = branch.GetSingleWordInOperand(0);
      assert(label_true != label_false &&
             "true and false labels are the same; control dependence "
             "impossible");
      if (target == label_true) {
        dep.condition_value = true;
      } else if (target == label_false) {
        dep.condition_value = false;
      } else {
        assert(false && "impossible control dependence; non-existent edge");
      }
      break;
    }
    case SpvOpSwitch: {
      dep.dependence_type = ControlDependence::DependenceType::kSwitchCase;
      dep.dependent_value_id = branch.GetSingleWordInOperand(0);
      for (const auto& switch_case : GetSwitchCases(branch)) {
        uint32_t case_value = switch_case.first;
        uint32_t label = switch_case.second;
        if (target == label) {
          dep.switch_case_values.push_back(case_value);
        }
      }
      dep.is_switch_default = target == branch.GetSingleWordInOperand(1);
      assert((dep.is_switch_default || !dep.switch_case_values.empty()) &&
             "impossible control dependence; non-existent edge");
      break;
    }
    default:
      assert(false &&
             "invalid control dependence; opcode of last instruction is not "
             "conditional branch");
  }
  return dep;
}
}  // namespace

constexpr uint32_t ControlDependenceAnalysis::kPseudoEntryBlock;

bool ControlDependence::operator<(const ControlDependence& other) const {
  return std::tie(source_bb_id, target_bb_id, dependence_type,
                  dependent_value_id, switch_case_values, is_switch_default,
                  condition_value) <
         std::tie(other.source_bb_id, other.target_bb_id, other.dependence_type,
                  other.dependent_value_id, other.switch_case_values,
                  other.is_switch_default, other.condition_value);
}

bool ControlDependence::operator==(const ControlDependence& other) const {
  return std::tie(source_bb_id, target_bb_id, dependence_type,
                  dependent_value_id, switch_case_values, is_switch_default,
                  condition_value) ==
         std::tie(other.source_bb_id, other.target_bb_id, other.dependence_type,
                  other.dependent_value_id, other.switch_case_values,
                  other.is_switch_default, other.condition_value);
}

std::ostream& operator<<(std::ostream& os, const ControlDependence& dep) {
  os << dep.source_bb_id << "->" << dep.target_bb_id;
  switch (dep.dependence_type) {
    case ControlDependence::DependenceType::kConditionalBranch:
      os << " if %" << dep.dependent_value_id << " is "
         << (dep.condition_value ? "true" : "false");
      break;
    case ControlDependence::DependenceType::kSwitchCase: {
      os << " switch %" << dep.dependent_value_id << " case ";
      bool first = true;
      for (uint32_t case_value : dep.switch_case_values) {
        if (first) {
          first = false;
        } else {
          os << ", ";
        }
        os << case_value;
      }
      if (dep.is_switch_default) {
        if (!first) {
          first = false;
          os << ", ";
        }
        os << "default";
      }
    } break;
    case ControlDependence::DependenceType::kEntry:
      os << " entry";
      break;
    default:
      os << " (unknown)";
  }
  return os;
}

void ControlDependenceAnalysis::ComputePostDominanceFrontiers(
    const CFG& cfg, const PostDominatorAnalysis& pdom) {
  // Compute post-dominance frontiers (reverse graph).
  // The dominance frontier for a block X is equal to (Equation 4)
  //   DF_local(X) U { B in DF_up(Z) | X = ipdom(Z) }
  //   (ipdom(Z) is the immediate post-dominator of Z.)
  // where
  //   DF_local(X) = { Y | X -> Y in CFG, X does not strictly post-dominate Y }
  //     represents the contribution of X's predecessors to the DF, and
  //   DF_up(Z) = { Y | Y in DF(Z), ipdom(Z) does not strictly post-dominate Y }
  //     (note: ipdom(Z) = X.)
  //     represents the contribution of a block to its immediate post-
  //     dominator's DF.
  // This is computed in one pass through a post-order traversal of the
  // post-dominator tree.

  // Assert that there is a block other than the pseudo exit in the pdom tree,
  // as we need one to get the function entry point (as the pseudo exit is not
  // actually part of the function.)
  assert(!cfg.IsPseudoExitBlock(pdom.GetDomTree().post_begin()->bb_));
  Function* function = pdom.GetDomTree().post_begin()->bb_->GetParent();
  uint32_t function_entry = function->entry()->id();
  reverse_nodes_[kPseudoEntryBlock];  // Ensure GetDependees(0) does not crash.
  for (auto it = pdom.GetDomTree().post_cbegin();
       it != pdom.GetDomTree().post_cend(); ++it) {
    const uint32_t label = it->id();
    ControlDependenceList& edges = reverse_nodes_[label];
    size_t new_size = edges.size();
    new_size += cfg.preds(label).size();
    for (DominatorTreeNode* child : *it) {
      const ControlDependenceList& child_edges = reverse_nodes_[child->id()];
      new_size += child_edges.size();
    }
    edges.reserve(new_size);
    for (uint32_t pred : cfg.preds(label)) {
      if (!pdom.StrictlyDominates(label, pred)) {
        edges.push_back(ClassifyControlDependence(cfg, pred, label));
      }
    }
    if (label == function_entry) {
      // Add edge from pseudo-entry to entry.
      // In CDG construction, an edge is added from entry to exit, so only the
      // exit node can post-dominate entry.
      edges.push_back(ClassifyControlDependence(cfg, kPseudoEntryBlock, label));
    }
    for (DominatorTreeNode* child : *it) {
      // Note: iterate dependences by value, as we need a copy.
      for (ControlDependence dep : reverse_nodes_[child->id()]) {
        // Special-case pseudo-entry, as above.
        if (dep.source_bb_id == kPseudoEntryBlock ||
            !pdom.StrictlyDominates(label, dep.source_bb_id)) {
          dep.target_bb_id = label;
          edges.push_back(dep);
        }
      }
    }
  }
}

void ControlDependenceAnalysis::InitializeGraph(
    const CFG& cfg, const PostDominatorAnalysis& pdom) {
  ComputePostDominanceFrontiers(cfg, pdom);
  // Compute the forward graph from the reverse graph.
  for (const auto& entry : reverse_nodes_) {
    // Ensure an entry is created for each node.
    forward_nodes_[entry.first];
    for (const ControlDependence& dep : entry.second) {
      forward_nodes_[dep.source_bb_id].push_back(dep);
    }
  }
}

}  // namespace opt
}  // namespace spvtools
