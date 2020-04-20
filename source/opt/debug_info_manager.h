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

#ifndef SOURCE_OPT_DEBUG_INFO_MANAGER_H_
#define SOURCE_OPT_DEBUG_INFO_MANAGER_H_

#include <unordered_map>

#include "source/opt/instruction.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// A class for analyzing, managing, and creating OpenCL.DebugInfo.100 extension
// instructions.
class DebugInfoManager {
 public:
  DebugInfoManager(IRContext* context);
  // Constructs a def-use manager from the given |module|. All internal messages
  // will be communicated to the outside via the given message |consumer|. This
  // instance only keeps a reference to the |consumer|, so the |consumer| should
  // outlive this instance.

  DebugInfoManager(const DebugInfoManager&) = delete;
  DebugInfoManager(DebugInfoManager&&) = delete;
  DebugInfoManager& operator=(const DebugInfoManager&) = delete;
  DebugInfoManager& operator=(DebugInfoManager&&) = delete;

  friend bool operator==(const DebugInfoManager&, const DebugInfoManager&);
  friend bool operator!=(const DebugInfoManager& lhs,
                         const DebugInfoManager& rhs) {
    return !(lhs == rhs);
  }

  // Clones DebugDeclare or DebugValue for a local variable whose result id is
  // |orig_var_id|. Set Variable operand of the new DebugDeclare as
  // |new_var_id|. Return the new DebugDeclare or DebugValue. If this is the
  // first DebugDeclare or DebugValue, keep it in |local_var_id_to_dbgdecl_|.
  Instruction* CloneDebugDeclare(uint32_t orig_var_id, uint32_t new_var_id);

  // Returns id of new DebugInlinedAt. Its Line operand is the line number
  // of |line| if |line| is not nullptr. Otherwise, its Line operand
  // is the line number of lexical scope of |scope|. Its Scope and Inlined
  // operands are Scope and Inlined of |scope|. Note that this function puts
  // the new DebugInlinedAt into the tail of debug instructions.
  uint32_t CreateDebugInlinedAt(const Instruction* line,
                                const DebugScope& scope);

  // If |debug_info_none_inst_| is not a nullptr, returns it. Otherwise,
  // creates a new DebugInfoNone instruction and returns it. In addition,
  // insert the new DebugInfoNone instruction before the head of debug
  // instructions.
  Instruction* GetDebugInfoNone();

  // Returns DebugInlinedAt whose id is |dbg_inlined_at_id|. If it does not
  // exist or it is not a DebugInlinedAt instruction, return nullptr.
  Instruction* GetDebugInlinedAt(uint32_t dbg_inlined_at_id);

  // Returns DebugFunction whose Function operand is |fn_id|. If it does not
  // exist, return nullptr.
  Instruction* GetDebugFunction(uint32_t fn_id) {
    auto dbg_fn_it = fn_id_to_dbg_fn_.find(fn_id);
    return dbg_fn_it == fn_id_to_dbg_fn_.end() ? nullptr : dbg_fn_it->second;
  }

  // Clones DebugInlinedAt whose id is |dbg_inlined_at_id|. If
  // |dbg_inlined_at_id| is not an id of DebugInlinedAt, returns nullptr.
  // Note that this function does not insert the new DebugInlinedAt into
  // debug instruction list of the module.
  Instruction* CloneDebugInlinedAt(uint32_t dbg_inlined_at_id);

 private:
  IRContext* context() { return context_; }

  // Analyzes OpenCL.DebugInfo.100 instructions in the given |module| and
  // populates data structures in this class.
  void AnalyzeDebugInsts(Module& module);

  // Returns the DebugDeclare instruction that corresponds to the variable
  // with id |var_id|. Returns |nullptr| if one does not exists.
  Instruction* GetDbgDeclareForVar(uint32_t var_id);

  // Registers the DebugDeclare instruction |inst| into
  // |local_var_id_to_dbgdecl_| using the variable operand of |inst| as a key.
  // If |local_var_id_to_dbgdecl_| already has a key that corresponds to the
  // variable operand of |inst|, it just returns without doing anything.
  void RegisterDbgDeclareForVar(Instruction* inst);

  // Returns the debug instruction whose id is |id|. Returns |nullptr| if one
  // does not exists.
  Instruction* GetDbgInst(uint32_t id);

  // Registers the debug instruction |inst| into |id_to_dbg_inst_| using id of
  // |inst| as a key.
  void RegisterDbgInst(Instruction* inst);

  // Registers the DebugFunction |inst| into |fn_id_to_dbg_fn_| using the
  // function operand of |inst| as a key.
  void RegisterDbgFunction(Instruction* inst);

  IRContext* context_;

  // Mapping from ids of OpenCL.DebugInfo.100 extension instructions
  // to their Instruction instances.
  std::unordered_map<uint32_t, Instruction*> id_to_dbg_inst_;

  // Mapping from ids of local variable OpVariable or OpFunctionParameter
  // to its first DebugDeclare or DebugValue instructions.
  std::unordered_map<uint32_t, Instruction*> local_var_id_to_dbgdecl_;

  // Mapping from function's ids to DebugFunction instructions whose
  // operand is the function.
  std::unordered_map<uint32_t, Instruction*> fn_id_to_dbg_fn_;

  // DebugInfoNone instruction. We need only a single DebugInfoNone.
  // To reuse the existing one, we keep it using this member variable.
  Instruction* debug_info_none_inst_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DEBUG_INFO_MANAGER_H_
