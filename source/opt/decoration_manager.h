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
#include <unordered_set>
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
  explicit DecorationManager(ir::Module* module) : module_(module) {
    AnalyzeDecorations();
  }
  DecorationManager() = delete;

  // Removes all decorations from |id| (either directly or indirectly) for
  // which |pred| returns true.
  // If |id| is a group ID, OpGroupDecorate and OpGroupMemberDecorate will be
  // removed if they have no targets left, and OpDecorationGroup will be
  // removed if the group is not applied to anyone and contains no decorations.
  void RemoveDecorationsFrom(uint32_t id,
                             std::function<bool(const ir::Instruction&)> pred =
                                 [](const ir::Instruction&) { return true; });

  // Removes all decorations from the result id of |inst|.
  //
  // NOTE: This is only meant to be called from ir_context, as only metadata
  // will be removed, and no actual instruction.
  void RemoveDecoration(ir::Instruction* inst);

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
  // Returns whether the two decorations instructions are the same and are
  // applying the same decorations; unless |ignore_target| is false, the targets
  // to which they are applied to does not matter, except for the member part.
  //
  // This is only valid for OpDecorate, OpMemberDecorate and OpDecorateId; it
  // will return false for other opcodes.
  bool AreDecorationsTheSame(const ir::Instruction* inst1,
                             const ir::Instruction* inst2,
                             bool ignore_target) const;

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|. Processed are all decorations which target |id| either
  // directly or indirectly by Decoration Groups.
  void ForEachDecoration(uint32_t id, uint32_t decoration,
                         std::function<void(const ir::Instruction&)> f);

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|. Processes all decoration which target |id| either directly or
  // indirectly through decoration groups. If |f| returns false, iteration is
  // terminated and this function returns false.
  bool WhileEachDecoration(uint32_t id, uint32_t decoration,
                           std::function<bool(const ir::Instruction&)> f);

  // Clone all decorations from one id |from|.
  // The cloned decorations are assigned to the given id |to| and are
  // added to the module. The purpose is to decorate cloned instructions.
  // This function does not check if the id |to| is already decorated.
  void CloneDecorations(uint32_t from, uint32_t to);

  // Informs the decoration manager of a new decoration that it needs to track.
  void AddDecoration(ir::Instruction* inst);

 private:
  // Analyzes the defs and uses in the given |module| and populates data
  // structures in this class. Does nothing if |module| is nullptr.
  void AnalyzeDecorations();

  template <typename T>
  std::vector<T> InternalGetDecorationsFor(uint32_t id, bool include_linkage);

  // Tracks decoration information of an ID.
  struct TargetData {
    std::vector<ir::Instruction*> direct_decorations;    // All decorate
                                                         // instructions applied
                                                         // to the tracked ID.
    std::vector<ir::Instruction*> indirect_decorations;  // All instructions
                                                         // applying a group to
                                                         // the tracked ID.
    std::vector<ir::Instruction*> decorate_insts;  // All decorate instructions
                                                   // applying the decorations
                                                   // of the tracked ID to
                                                   // targets.
                                                   // It is empty if the
                                                   // tracked ID is not a
                                                   // group.
  };

  // Mapping from ids to the instructions applying a decoration to those ids.
  // In other words, for each id you get all decoration instructions
  // referencing that id, be it directly (SpvOpDecorate, SpvOpMemberDecorate
  // and SpvOpDecorateId), or indirectly (SpvOpGroupDecorate,
  // SpvOpMemberGroupDecorate).
  std::unordered_map<uint32_t, TargetData> id_to_decoration_insts_;
  // The enclosing module.
  ir::Module* module_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DECORATION_MANAGER_H_
