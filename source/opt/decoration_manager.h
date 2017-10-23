// Copyright (c) 2017 Pierre Moreau
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

#ifndef LIBSPIRV_OPT_DECORATION_MANAGER_H_
#define LIBSPIRV_OPT_DECORATION_MANAGER_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "instruction.h"
#include "module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// A class for analyzing and managing decorations in an ir::Module.
class DecorationManager {
 public:
  // Constructs a decoration manager from the given |module|
  DecorationManager(ir::Module* module) { AnalyzeDecorations(module); }
  // Removes all decorations from |id|, which should not be a group ID, except
  // for linkage decorations if |keep_linkage| is set.
  void RemoveDecorationsFrom(uint32_t id, bool keep_linkage);
  // Returns a vector of all decorations affecting |id|. If a group is applied
  // to |id|, the decorations of that group are returned rather than the group
  // decoration instruction. If |include_linkage| is not set, linkage
  // decorations won't be returned.
  std::vector<ir::Instruction*> GetDecorationsFor(uint32_t id,
                                                  bool include_linkage);
  std::vector<const ir::Instruction*> GetDecorationsFor(
      uint32_t id, bool include_linkage) const;
  // Returns whether two IDs have the same decorations. Two SpvOpGroupDecorate
  // instructions that apply the same decorations but to different IDs, still
  // count as being the same.
  bool HaveTheSameDecorations(uint32_t id1, uint32_t id2) const;
  // Returns whether two decorations are the same. SpvOpDecorateId is currently
  // not handled and will return false no matter what.
  bool AreDecorationsTheSame(const ir::Instruction* inst1,
                             const ir::Instruction* inst2) const;

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|.
  void ForEachDecoration(uint32_t id, uint32_t decoration,
                         std::function<void(const ir::Instruction& f)>) const;

 private:
  using IdToDecorationInstsMap =
      std::unordered_map<uint32_t, std::vector<ir::Instruction*>>;
  // Analyzes the defs and uses in the given |module| and populates data
  // structures in this class. Does nothing if |module| is nullptr.
  void AnalyzeDecorations(ir::Module* module);

  template <typename T>
  std::vector<T> InternalGetDecorationsFor(uint32_t id, bool include_linkage);

  // Mapping from ids to the instructions applying a decoration to them. In
  // other words, for each id you get all decoration instructions referencing
  // that id, be it directly (SpvOpDecorate, SpvOpMemberDecorate and
  // SpvOpDecorateId), or indirectly (SpvOpGroupDecorate,
  // SpvOpMemberGroupDecorate).
  IdToDecorationInstsMap id_to_decoration_insts_;
  // Mapping from group ids to all the decoration instructions they apply.
  IdToDecorationInstsMap group_to_decoration_insts_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DECORATION_MANAGER_H_
