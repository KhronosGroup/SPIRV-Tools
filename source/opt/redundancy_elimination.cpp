// Copyright (c) 2017 Google Inc.
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

#include "source/opt/redundancy_elimination.h"

#include <utility>

#include "source/opt/value_number_table.h"

namespace spvtools {
namespace opt {

Pass::Status RedundancyEliminationPass::Process() {
  bool modified = false;
  ValueNumberTable vnTable(context());

  for (auto& func : *get_module()) {
    if (func.IsDeclaration()) {
      continue;
    }

    // Build the dominator tree for this function. It is how the code is
    // traversed.
    DominatorTree& dom_tree =
        context()->GetDominatorAnalysis(&func)->GetDomTree();

    if (EliminateRedundanciesFrom(dom_tree.GetRoot(), vnTable)) {
      modified = true;
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool RedundancyEliminationPass::EliminateRedundanciesFrom(
    DominatorTreeNode* bb, const ValueNumberTable& vnTable) {
  struct State {
    DominatorTreeNode* node;
    std::map<uint32_t, uint32_t> value_to_id_map;
  };
  std::vector<State> todo;
  todo.push_back({bb, std::map<uint32_t, uint32_t>()});
  bool modified = false;
  for (size_t next_node = 0; next_node < todo.size(); next_node++) {
    modified |= EliminateRedundanciesInBB(todo[next_node].node->bb_, vnTable,
                                          &todo[next_node].value_to_id_map);
    const auto& children = todo[next_node].node->children_;
    for (size_t child_index = 0; child_index < children.size(); ++child_index) {
      if (child_index + 1 == children.size()) {
        todo.push_back({children[child_index],
                        std::move(todo[next_node].value_to_id_map)});
      } else {
        todo.push_back({children[child_index], todo[next_node].value_to_id_map});
      }
    }
    todo[next_node].value_to_id_map.clear();
  }
  return modified;
}
}  // namespace opt
}  // namespace spvtools
