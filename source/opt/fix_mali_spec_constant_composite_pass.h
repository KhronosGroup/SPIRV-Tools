// Copyright (c) 2023 Lee Gao
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_OPT_FIX_MALI_SPEC_CONSTANT_COMPOSITE_PASS_H_
#define SOURCE_OPT_FIX_MALI_SPEC_CONSTANT_COMPOSITE_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class FixMaliSpecConstantCompositePass : public MemPass {
 public:
  const char* name() const override { return "fix-mali-spec-constant-composite"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisNone;
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FIX_MALI_SPEC_CONSTANT_COMPOSITE_PASS_H_