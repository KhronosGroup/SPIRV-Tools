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

#ifndef SOURCE_OPT_LOCAL_SSA_ELIM_PASS_H_
#define SOURCE_OPT_LOCAL_SSA_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"
#include "source/opt/pass_token.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalMultiStoreElimPass : public MemPass {
  using cbb_ptr = const opt::BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<opt::BasicBlock*>*(const opt::BasicBlock*)>;

  LocalMultiStoreElimPass();

  Status Process(opt::IRContext* c) override;

  opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisDefUse |
           opt::IRContext::kAnalysisInstrToBlockMapping;
  }

 private:
  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(opt::IRContext* c);
  Pass::Status ProcessImpl();

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

class LocalMultiStoreElimPassToken : public PassToken {
 public:
  LocalMultiStoreElimPassToken() = default;
  ~LocalMultiStoreElimPassToken() override = default;

  const char* name() const override { return "eliminate-local-multi-store"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<LocalMultiStoreElimPass>();
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOCAL_SSA_ELIM_PASS_H_
