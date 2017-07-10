// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_
#define LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class AggressiveDCEPass : public Pass {

  using cbb_ptr = const ir::BasicBlock*;

 public:
   using GetBlocksFunction =
     std::function<std::vector<ir::BasicBlock*>*(const ir::BasicBlock*)>;

  AggressiveDCEPass();
  const char* name() const override { return "aggressive-dce"; }
  Status Process(ir::Module*) override;

 private:
  // Returns true if |opcode| is a non-ptr access chain op
  bool IsNonPtrAccessChain(const SpvOp opcode) const;

  // Given a load or store |ip|, return the pointer instruction.
  // Also return the base variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Add all store instruction which use |ptrId|, directly or indirectly,
  // to the live instruction worklist.
  void AddStores(uint32_t ptrId);

  // Return true if variable with |varId| is function scope
  bool IsLocalVar(uint32_t varId);

  // Initialize combinator data structures
  void InitCombinatorSets();

  // Return true if core operator |op| has no side-effects. Currently returns
  // true only for shader capability operations.
  // TODO(greg-lunarg): Add kernel and other operators
  bool IsCombinator(uint32_t op) const;

  // Return true if OpExtInst |inst| has no side-effects. Currently returns
  // true only for std.GLSL.450 extensions
  // TODO(greg-lunarg): Add support for other extensions
  bool IsCombinatorExt(ir::Instruction* inst) const;

  // Return true if all extensions in this module are supported by this pass.
  // Currently, no extensions are supported. glsl_std_450 extended instructions
  // are allowed.
  bool AllExtensionsSupported();

  // Kill debug or annotation |inst| if target operand is dead.
  void KillInstIfTargetDead(ir::Instruction* inst);

  // For function |func|, mark all Stores to non-function-scope variables
  // and block terminating instructions as live. Recursively mark the values
  // they use. When complete, delete any non-live instructions. Return true
  // if the function has been modified.
  // 
  // Note: This function does not delete useless control structures. All
  // existing control structures will remain. This can leave not-insignificant
  // sequences of ultimately useless code.
  // TODO(): Remove useless control constructs.
  bool AggressiveDCE(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Module this pass is processing
  ir::Module* module_;

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Live Instruction Worklist.  An instruction is added to this list
  // if it might have a side effect, either directly or indirectly.
  // If we don't know, then add it to this list.  Instructions are
  // removed from this list as the algorithm traces side effects,
  // building up the live instructions set |live_insts_|.
  std::queue<ir::Instruction*> worklist_;

  // Live Instructions
  std::unordered_set<const ir::Instruction*> live_insts_;

  // Live Local Variables
  std::unordered_set<uint32_t> live_local_vars_;

  // Dead instructions. Use for debug cleanup.
  std::unordered_set<const ir::Instruction*> dead_insts_;

  // Opcodes of shader capability core executable instructions
  // without side-effect. This is a whitelist of operators
  // that can safely be left unmarked as live at the beginning of
  // aggressive DCE.
  std::unordered_set<uint32_t> combinator_ops_shader_;

  // Opcodes of GLSL_std_450 extension executable instructions
  // without side-effect. This is a whitelist of operators
  // that can safely be left unmarked as live at the beginning of
  // aggressive DCE.
  std::unordered_set<uint32_t> combinator_ops_glsl_std_450_;

  // Set id for glsl_std_450 extension instructions
  uint32_t glsl_std_450_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_AGGRESSIVE_DCE_PASS_H_

