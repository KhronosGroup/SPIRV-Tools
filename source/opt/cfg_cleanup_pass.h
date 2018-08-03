// Copyright (c) 2017 Google Inc.
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

#ifndef LIBSPIRV_OPT_CFG_CLEANUP_PASS_H_
#define LIBSPIRV_OPT_CFG_CLEANUP_PASS_H_

#include "source/opt/function.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

class CFGCleanupPass : public MemPass {
 public:
  CFGCleanupPass() = default;

  const char* name() const override { return "cfg-cleanup"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse;
  }
};

}  // namespace opt
}  // namespace spvtools

#endif
