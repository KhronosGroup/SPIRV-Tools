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

#ifndef SOURCE_OPT_CONTROL_DEPENDENCE_H_
#define SOURCE_OPT_CONTROL_DEPENDENCE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <ostream>
#include <vector>

#include "source/opt/cfg.h"
#include "source/opt/dominator_analysis.h"

namespace spvtools {
namespace opt {

struct ControlDependence {
  enum class DependenceType {
    kConditionalBranch,
    kSwitchCase,
    kEntry,
  };
  // The label of the source of this dependence, i.e. the dependee.
  uint32_t source_bb_id = 0;
  // The label of the target of this dependence, i.e. the dependent.
  uint32_t target_bb_id = 0;
  // The type of dependence: either a conditional branch, switch-case (any
  // combination of numbered labels and default), or entry (the only condition
  // for this block is that the function is entered.)
  DependenceType dependence_type = DependenceType::kEntry;
  // The id for the value which this dependence is on.
  // For conditional branches, this is the branch condition, and
  // for switch cases, this is the value on which the switch is performed.
  uint32_t dependent_value_id = 0;
  // For switch cases, the values of the cases for this dependence.
  std::vector<uint32_t> switch_case_values;
  // For switch cases, true if this dependence happens when the default branch
  // is taken.
  bool is_switch_default = false;
  // For conditional branches, the value of the condition required for the
  // dependence to happen.
  bool condition_value = false;

  bool operator==(const ControlDependence& other) const;

  // Comparison operators, ordered lexicographically. Total ordering.
  bool operator<(const ControlDependence& other) const;
  bool operator>(const ControlDependence& other) const { return other < *this; }
  bool operator<=(const ControlDependence& other) const {
    return !(*this > other);
  }
  bool operator>=(const ControlDependence& other) const {
    return !(*this < other);
  }
};

// Prints |dep| to |os| in a human-readable way. Examples:
//   %1 -> %2 if %3 is true
//   %4 -> %5 switch %6 case 1, default
//   %0 -> %7 entry
std::ostream& operator<<(std::ostream& os, const ControlDependence& dep);

// Represents the control dependence graph. A basic block is control dependent
// on another if the result of that block (e.g. the condition of a conditional
// branch) influences whether it is executed or not. More formally, a block A is
// control dependent on B iff:
// 1. there exists a path from A to the exit node that does *not* go through B
//    (i.e., A does not postdominate B), and
// 2. there exists a path B -> b_1 -> ... -> b_n -> A such that A post-dominates
//    all nodes b_i.
class ControlDependenceAnalysis {
 public:
  // Map basic block labels to control dependencies/dependents.
  // Not guaranteed to be in any particular order.
  using ControlDependenceList = std::vector<ControlDependence>;
  using ControlDependenceListMap = std::map<uint32_t, ControlDependenceList>;

  // 0, the label number for the pseudo entry block.
  // All control dependences on the pseudo entry block are of type kEntry, and
  // vice versa.
  static constexpr uint32_t kPseudoEntryBlock = 0;

  // Build the control dependence graph for the given control flow graph |cfg|
  // and corresponding post-dominator analysis |pdom|.
  void InitializeGraph(const CFG& cfg, const PostDominatorAnalysis& pdom);

  // Get the list of the nodes that depend on a block.
  // Return value is not guaranteed to be in any particular order.
  const ControlDependenceList& GetDependents(uint32_t block) const {
    return forward_nodes_.at(block);
  }

  // Get the list of the nodes on which a block depends on.
  // Return value is not guaranteed to be in any particular order.
  const ControlDependenceList& GetDependees(uint32_t block) const {
    return reverse_nodes_.at(block);
  }

  // Runs the function |f| on each block label in the CDG. If any iteration
  // returns false, immediately stops iteration and returns false. Otherwise
  // returns true. Nodes are iterated in order of label, including the
  // pseudo-entry block.
  bool WhileEachBlockLabel(std::function<bool(uint32_t)> f) const {
    for (const auto& entry : forward_nodes_) {
      if (!f(entry.first)) {
        return false;
      }
    }
    return true;
  }

  // Runs the function |f| on each block label in the CDG. Nodes are iterated in
  // order of label, including the pseudo-entry block.
  void ForEachBlockLabel(std::function<void(uint32_t)> f) const {
    WhileEachBlockLabel([&f](uint32_t label) {
      f(label);
      return true;
    });
  }

  // Is block |a| (directly) dependent on block |b|?
  bool IsDependent(uint32_t a, uint32_t b) const {
    if (forward_nodes_.find(a) == forward_nodes_.end()) return false;
    // BBs tend to have more dependents than dependees, so search dependees.
    const ControlDependenceList& a_dependees = GetDependees(a);
    return std::find_if(a_dependees.begin(), a_dependees.end(),
                        [b](const ControlDependence& dep) {
                          return dep.source_bb_id == b;
                        }) != a_dependees.end();
  }

 private:
  ControlDependenceListMap forward_nodes_;
  ControlDependenceListMap reverse_nodes_;

  void ComputePostDominanceFrontiers(const CFG& cfg,
                                     const PostDominatorAnalysis& pdom);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONTROL_DEPENDENCE_H_
