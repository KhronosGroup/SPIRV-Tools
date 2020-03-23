// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_OPT_DEBUG_INLINED_AT_MANAGER_H_
#define SOURCE_OPT_DEBUG_INLINED_AT_MANAGER_H_

#include <unordered_map>

#include "source/opt/instruction.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// A class for analyzing, managing, and creating OpenCL.100.DebugInfo extension
// instructions.
class DebugInfoManager {
 public:
  // Constructs a def-use manager from the given |module|. All internal messages
  // will be communicated to the outside via the given message |consumer|. This
  // instance only keeps a reference to the |consumer|, so the |consumer| should
  // outlive this instance.
  DebugInfoManager(Module* module) { AnalyzeDefUse(module); }

  DebugInfoManager(const DebugInfoManager&) = delete;
  DebugInfoManager(DebugInfoManager&&) = delete;
  DebugInfoManager& operator=(const DebugInfoManager&) = delete;
  DebugInfoManager& operator=(DebugInfoManager&&) = delete;

  // Analyzes the defs in the given |inst|.
  void AnalyzeInstDef(Instruction* inst);

  // Analyzes the uses in the given |inst|.
  //
  // All operands of |inst| must be analyzed as defs.
  void AnalyzeInstUse(Instruction* inst);

  // Analyzes the defs and uses in the given |inst|.
  void AnalyzeInstDefUse(Instruction* inst);

  // Returns the def instruction for the given |id|. If there is no instruction
  // defining |id|, returns nullptr.
  Instruction* GetDef(uint32_t id);
  const Instruction* GetDef(uint32_t id) const;

  // Runs the given function |f| on each unique user instruction of |def| (or
  // |id|).
  //
  // If one instruction uses |def| in multiple operands, that instruction will
  // only be visited once.
  //
  // |def| (or |id|) must be registered as a definition.
  void ForEachUser(const Instruction* def,
                   const std::function<void(Instruction*)>& f) const;
  void ForEachUser(uint32_t id,
                   const std::function<void(Instruction*)>& f) const;

  // Runs the given function |f| on each unique user instruction of |def| (or
  // |id|). If |f| returns false, iteration is terminated and this function
  // returns false.
  //
  // If one instruction uses |def| in multiple operands, that instruction will
  // be only be visited once.
  //
  // |def| (or |id|) must be registered as a definition.
  bool WhileEachUser(const Instruction* def,
                     const std::function<bool(Instruction*)>& f) const;
  bool WhileEachUser(uint32_t id,
                     const std::function<bool(Instruction*)>& f) const;

  // Runs the given function |f| on each unique use of |def| (or
  // |id|).
  //
  // If one instruction uses |def| in multiple operands, each operand will be
  // visited separately.
  //
  // |def| (or |id|) must be registered as a definition.
  void ForEachUse(
      const Instruction* def,
      const std::function<void(Instruction*, uint32_t operand_index)>& f) const;
  void ForEachUse(
      uint32_t id,
      const std::function<void(Instruction*, uint32_t operand_index)>& f) const;

  // Runs the given function |f| on each unique use of |def| (or
  // |id|). If |f| returns false, iteration is terminated and this function
  // returns false.
  //
  // If one instruction uses |def| in multiple operands, each operand will be
  // visited separately.
  //
  // |def| (or |id|) must be registered as a definition.
  bool WhileEachUse(
      const Instruction* def,
      const std::function<bool(Instruction*, uint32_t operand_index)>& f) const;
  bool WhileEachUse(
      uint32_t id,
      const std::function<bool(Instruction*, uint32_t operand_index)>& f) const;

  // Returns the number of users of |def| (or |id|).
  uint32_t NumUsers(const Instruction* def) const;
  uint32_t NumUsers(uint32_t id) const;

  // Returns the number of uses of |def| (or |id|).
  uint32_t NumUses(const Instruction* def) const;
  uint32_t NumUses(uint32_t id) const;

  // Returns the annotation instrunctions which are a direct use of the given
  // |id|. This means when the decorations are applied through decoration
  // group(s), this function will just return the OpGroupDecorate
  // instrcution(s) which refer to the given id as an operand. The OpDecorate
  // instructions which decorate the decoration group will not be returned.
  std::vector<Instruction*> GetAnnotations(uint32_t id) const;

  // Returns the map from ids to their def instructions.
  const IdToDefMap& id_to_defs() const { return id_to_def_; }
  // Returns the map from instructions to their users.
  const IdToUsersMap& id_to_users() const { return id_to_users_; }

  // Clear the internal def-use record of the given instruction |inst|. This
  // method will update the use information of the operand ids of |inst|. The
  // record: |inst| uses an |id|, will be removed from the use records of |id|.
  // If |inst| defines an result id, the use record of this result id will also
  // be removed. Does nothing if |inst| was not analyzed before.
  void ClearInst(Instruction* inst);

  // Erases the records that a given instruction uses its operand ids.
  void EraseUseRecordsOfOperandIds(const Instruction* inst);

  friend bool operator==(const DebugInfoManager&, const DebugInfoManager&);
  friend bool operator!=(const DebugInfoManager& lhs, const DebugInfoManager& rhs) {
    return !(lhs == rhs);
  }

  // If |inst| has not already been analysed, then analyses its defintion and
  // uses.
  void UpdateDefUse(Instruction* inst);

 private:
  // Map from DebugInlinedAt's result id to its Instruction.
  std::unordered_map<uint32_t, Instruction*> id2inlined_at_;

  // Map from result id of lexical scope debug instruction (DebugFunction,
  // DebugTypeComposite, DebugLexicalBlock, DebugCompilationUnit) to its
  // Instruction.
  std::unordered_map<uint32_t, Instruction*> id2lexical_scope_;

  // Map from result id of OpFunctionParameter to its first DebugDeclare
  // Instruction.
  std::unordered_map<uint32_t, Instruction*> param_id2debugdecl_;

  // Map from function's id to DebugFunction id whose operand is the function.
  std::unordered_map<uint32_t, uint32_t> func_id2dbg_func_id_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DEBUG_INLINED_AT_MANAGER_H_
