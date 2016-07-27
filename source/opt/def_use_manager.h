// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_DEF_USE_MANAGER_H_
#define LIBSPIRV_OPT_DEF_USE_MANAGER_H_

#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "instruction.h"
#include "module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// Class for representing a use of id. Note that result:
// * Type id is a use.
// * Ids referenced in OpSectionMerge & OpLoopMerge are considered as use.
// * Ids referenced in OpPhi's in operands are considered as use.
struct Use {
  ir::Instruction* inst;   // Instruction using the id.
  uint32_t operand_index;  // logical operand index of the id use. This can be
                           // the index of result type id.
};

using UseList = std::list<Use>;

// A class for analyzing and managing defs and uses in an ir::Module.
class DefUseManager {
 public:
  using IdToDefMap = std::unordered_map<uint32_t, ir::Instruction*>;
  using IdToUsesMap = std::unordered_map<uint32_t, UseList>;

  // Analyzes the defs and uses in the given |module| and populates data
  // structures in this class.
  // TODO(antiagainst): This method should not modify the given module. Create
  // const overload for ForEachInst().
  void AnalyzeDefUse(ir::Module* module);

  // Returns the def instruction for the given |id|. If there is no instruction
  // defining |id|, returns nullptr.
  ir::Instruction* GetDef(uint32_t id);
  // Returns the use instructions for the given |id|. If there is no uses of
  // |id|, returns nullptr.
  UseList* GetUses(uint32_t id);

  // Returns the map from ids to their def instructions.
  const IdToDefMap& id_to_defs() const { return id_to_def_; }
  // Returns the map from ids to their uses in instructions.
  const IdToUsesMap& id_to_uses() const { return id_to_uses_; }

  // Turns the instruction defining the given |id| into a Nop. Returns true on
  // success, false if the given |id| is not defined at all. This method also
  // erases both the uses of |id| and the |id|-generating instruction's use
  // information kept in this manager, but not the operands in the original
  // instructions.
  bool KillDef(uint32_t id);
  // Replaces all uses of |before| id with |after| id. Returns true if any
  // replacement happens. This method does not kill the definition of the
  // |before| id. If |after| is the same as |before|, does nothing and returns
  // false.
  bool ReplaceAllUsesWith(uint32_t before, uint32_t after);

 private:
  using ResultIdToUsedIdsMap =
      std::unordered_map<uint32_t, std::vector<uint32_t>>;

  // Analyzes the defs and uses in the given |inst|.
  void AnalyzeInstDefUse(ir::Instruction* inst);

  IdToDefMap id_to_def_;    // Mapping from ids to their definitions
  IdToUsesMap id_to_uses_;  // Mapping from ids to their uses
  // Mapping from result ids to the ids used in the instructions generating the
  // result ids.
  ResultIdToUsedIdsMap result_id_to_used_ids_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEF_USE_MANAGER_H_
