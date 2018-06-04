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

#ifndef LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_
#define LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_

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
class LocalSingleStoreElimPass : public Pass {
  using cbb_ptr = const ir::BasicBlock*;

 public:
  LocalSingleStoreElimPass();
  const char* name() const override { return "eliminate-local-single-store"; }
  Status Process(ir::IRContext* irContext) override;

  ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse |
           ir::IRContext::kAnalysisInstrToBlockMapping;
  }

 private:
  // Do "single-store" optimization of function variables defined only
  // with a single non-access-chain store in |func|. Replace all their
  // non-access-chain loads with the value that is stored and eliminate
  // any resulting dead code.
  bool LocalSingleStoreElim(ir::Function* func);

  // Initialize extensions whitelist
  void InitExtensionWhiteList();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(ir::IRContext* irContext);
  Pass::Status ProcessImpl();

  // If there is a single store to |var_inst|, and it covers the entire
  // variable, then replace all of the loads of the entire variable that are
  // dominated by the store by the value that was stored.  Returns true if the
  // module was changed.
  bool ProcessVariable(ir::Instruction* var_inst);

  // Collects all of the uses of |var_inst| into |uses|.  This looks through
  // OpObjectCopy's that copy the address of the variable, and collects those
  // uses as well.
  void FindUses(const ir::Instruction* var_inst,
                std::vector<ir::Instruction*>* uses) const;

  // Returns a store to |var_inst| if
  //   - it is a store to the entire variable,
  //   - and there are no other instructions that may modify |var_inst|.
  ir::Instruction* FindSingleStoreAndCheckUses(
      ir::Instruction* var_inst,
      const std::vector<ir::Instruction*>& users) const;

  // Returns true if the address that results from |inst| may be used as a base
  // address in a store instruction or may be used to compute the base address
  // of a store instruction.
  bool FeedsAStore(ir::Instruction* inst) const;

  // Replaces all of the loads in |uses| by the value stored in |store_inst|.
  // The load instructions are then killed.
  bool RewriteLoads(ir::Instruction* store_inst,
                    const std::vector<ir::Instruction*>& uses);

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOCAL_SINGLE_STORE_ELIM_PASS_H_
