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

#ifndef LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_
#define LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "mem_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class DeadBranchElimPass : public MemPass {
  using cbb_ptr = const ir::BasicBlock*;

 public:
  DeadBranchElimPass();
  const char* name() const override { return "eliminate-dead-branches"; }
  Status Process(ir::IRContext* context) override;

  ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse;
  }

 private:
  // If |condId| is boolean constant, return conditional value in |condVal| and
  // return true, otherwise return false.
  bool GetConstCondition(uint32_t condId, bool* condVal);

  // If |valId| is a 32-bit integer constant, return value via |value| and
  // return true, otherwise return false.
  bool GetConstInteger(uint32_t valId, uint32_t* value);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, ir::BasicBlock* bp);

  // For function |func|, look for BranchConditionals with constant condition
  // and convert to a Branch to the indicated label. Delete resulting dead
  // blocks. Note some such branches and blocks may be left to avoid creating
  // invalid control flow.
  // TODO(greg-lunarg): Remove remaining constant conditional branches and dead
  // blocks.
  bool EliminateDeadBranches(ir::Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(ir::IRContext* c);
  Pass::Status ProcessImpl();

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_DEAD_BRANCH_ELIM_PASS_H_
