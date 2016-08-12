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
#include <unordered_set>
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

  // Reset this manager and analyzes the defs and uses in the given |module|
  // and populates data structures in this class.
  // TODO(antiagainst): This method should not modify the given module. Create
  // const overload for ForEachInst().
  void AnalyzeDefUse(ir::Module* module);

  // Analyzes the defs and uses in the given |inst|. If |inst| has been
  // analyzed by this manager before, the existing records will be overwritten
  // by the latest analysis result.
  void AnalyzeInstDefUse(ir::Instruction* inst);

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
  // Turns the given instruction |inst| to a Nop, also erases both the use
  // records of the result id of |inst| (if any) and the corresponding use
  // records of: |inst| using |inst|'s operand ids.
  void KillInst(ir::Instruction* inst);
  // Replaces all uses of |before| id with |after| id. Returns true if any
  // replacement happens. This method does not kill the definition of the
  // |before| id. If |after| is the same as |before|, does nothing and returns
  // false.
  bool ReplaceAllUsesWith(uint32_t before, uint32_t after);

 private:
  // Clears the internal def-use records of a defined id if the given |def_id|
  // is recorded by this manager. Does nothing if |def_id| has not been
  // recorded yet. This method will erase both the uses of |def_id| and the
  // |def_id|-generating instruction's use information kept in this manager,
  // but not the operands in the original instructions.
  void ClearDef(uint32_t def_id);

  // Clears the internal def-use records of the given instruction |inst| if it
  // has been analyzed by this manager. The use information of its operand ids
  // will be updated: "the record: |inst| uses |operand id| will be removed".
  // If the instruction is defining a result id, the uses of the result id will
  // also be removed. Note that if |inst| has not been analyzed before, this
  // function does nothing even though |inst| may define an existing result id.
  void ClearInst(ir::Instruction* inst);

  // Returns true if the operand's id should be recorded, otherwise returns
  // false;
  bool ShouldRecord(const ir::Operand& operand) const;

  // Erases the record that: instruction |user| uses id |used_id|.
  void EraseInstUseIdRecord(const ir::Instruction& user, uint32_t used_id);

  // Resets the internal records
  void Reset() {
    analyzed_insts_.clear();
    id_to_def_.clear();
    id_to_uses_.clear();
  }

  std::unordered_set<ir::Instruction*>
      analyzed_insts_;      // Analyzed instructions
  IdToDefMap id_to_def_;    // Mapping from ids to their definitions
  IdToUsesMap id_to_uses_;  // Mapping from ids to their uses
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEF_USE_MANAGER_H_
