// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_DEF_USE_MANAGER_H_
#define LIBSPIRV_OPT_DEF_USE_MANAGER_H_

#include <list>
#include <set>
#include <unordered_map>
#include <vector>

#include "instruction.h"
#include "module.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {
namespace analysis {

// Class for representing a use of id. Note that:
// * Result type id is a use.
// * Ids referenced in OpSectionMerge & OpLoopMerge are considered as use.
// * Ids referenced in OpPhi's in operands are considered as use.
struct Use {
  ir::Instruction* inst;   // Instruction using the id.
  uint32_t operand_index;  // logical operand index of the id use. This can be
                           // the index of result type id.
};

inline bool operator==(const Use& lhs, const Use& rhs) {
  return lhs.inst == rhs.inst && lhs.operand_index == rhs.operand_index;
}

inline bool operator!=(const Use& lhs, const Use& rhs) {
  return !(lhs == rhs);
}

inline bool operator<(const Use& lhs, const Use& rhs) {
  if (lhs.inst < rhs.inst)
    return true;
  if (lhs.inst > rhs.inst)
    return false;
  return lhs.operand_index < rhs.operand_index;
}

using UseList = std::list<Use>;
using UseEntry = std::pair<ir::Instruction*, ir::Instruction*>;

// A class for analyzing and managing defs and uses in an ir::Module.
class DefUseManager {
 public:
  using IdToDefMap = std::unordered_map<uint32_t, ir::Instruction*>;
  using IdToUsesMap = std::unordered_map<uint32_t, UseList>;
  using IdToUsersMap = std::set<UseEntry>;

  // Constructs a def-use manager from the given |module|. All internal messages
  // will be communicated to the outside via the given message |consumer|. This
  // instance only keeps a reference to the |consumer|, so the |consumer| should
  // outlive this instance.
  DefUseManager(ir::Module* module) { AnalyzeDefUse(module); }

  DefUseManager(const DefUseManager&) = delete;
  DefUseManager(DefUseManager&&) = delete;
  DefUseManager& operator=(const DefUseManager&) = delete;
  DefUseManager& operator=(DefUseManager&&) = delete;

  // Analyzes the defs in the given |inst|.
  void AnalyzeInstDef(ir::Instruction* inst);

  // Analyzes the uses in the given |inst|.
  void AnalyzeInstUse(ir::Instruction* inst);

  // Analyzes the defs and uses in the given |inst|.
  void AnalyzeInstDefUse(ir::Instruction* inst);

  // Returns the def instruction for the given |id|. If there is no instruction
  // defining |id|, returns nullptr.
  ir::Instruction* GetDef(uint32_t id);
  const ir::Instruction* GetDef(uint32_t id) const;
  // Returns the use instructions for the given |id|. If there is no uses of
  // |id|, returns nullptr.
  UseList* GetUses(uint32_t id);
  const UseList* GetUses(uint32_t id) const;

  void ForEachUser(Insruction* def,
                   const std::function<void(ir::Instruction*)>& f);
  void ForEachUser(const Insruction* def,
                   const std::function<void(const ir::Instruction*)>& f) const;
  void ForEachUser(uint32_t id,
                   const std::function<void(ir::Instruction*)>& f);
  void ForEachUser(uint32_t id,
                   const std::function<void(const ir::Instruction*)>& f) const;

  // Returns the annotation instrunctions which are a direct use of the given
  // |id|. This means when the decorations are applied through decoration
  // group(s), this function will just return the OpGroupDecorate
  // instrcution(s) which refer to the given id as an operand. The OpDecorate
  // instructions which decorate the decoration group will not be returned.
  std::vector<ir::Instruction*> GetAnnotations(uint32_t id) const;

  // Returns the map from ids to their def instructions.
  const IdToDefMap& id_to_defs() const { return id_to_def_; }
  // Returns the map from ids to their uses in instructions.
  const IdToUsesMap& id_to_uses() const { return id_to_uses_; }

  // Clear the internal def-use record of the given instruction |inst|. This
  // method will update the use information of the operand ids of |inst|. The
  // record: |inst| uses an |id|, will be removed from the use records of |id|.
  // If |inst| defines an result id, the use record of this result id will also
  // be removed. Does nothing if |inst| was not analyzed before.
  void ClearInst(ir::Instruction* inst);

  // Erases the records that a given instruction uses its operand ids.
  void EraseUseRecordsOfOperandIds(const ir::Instruction* inst);

  friend  bool operator==(const DefUseManager&, const DefUseManager&);
  friend  bool operator!=(const DefUseManager& lhs, const DefUseManager& rhs) {
    return !(lhs == rhs);
  }

 private:
  using InstToUsedIdsMap =
      std::unordered_map<const ir::Instruction*, std::vector<uint32_t>>;

  // Analyzes the defs and uses in the given |module| and populates data
  // structures in this class. Does nothing if |module| is nullptr.
  void AnalyzeDefUse(ir::Module* module);

  IdToDefMap id_to_def_;    // Mapping from ids to their definitions
  IdToUsesMap id_to_uses_;  // Mapping from ids to their uses
  IdToUsersMap id_to_users_; // Mapping from ids to their users
  // Mapping from instructions to the ids used in the instruction.
  InstToUsedIdsMap inst_to_used_ids_;

};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEF_USE_MANAGER_H_
