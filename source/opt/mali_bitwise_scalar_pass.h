// Copyright (c) 2026 Lee Gao
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

#ifndef SOURCE_OPT_MALI_BITWISE_SCALAR_PASS_H_
#define SOURCE_OPT_MALI_BITWISE_SCALAR_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// From MaliBridge (Arm erratum 3843237, r18p0-r53p0):
// Mali miscompiles an integer vector bitwise op (or shift) whose operand
// is a lane swizzle or a vector built lane by lane. This pass rewrites
// such ops into one scalar op per lane plus an OpCompositeConstruct of the results.
class MaliBitwiseScalarPass : public Pass {
  public:
    explicit MaliBitwiseScalarPass(bool rewrite_all = false)
        : rewrite_all_(rewrite_all) {}

    const char *name() const override { return "mali-bitwise-scalar"; }
    Status Process() override;

    IRContext::Analysis GetPreservedAnalyses() override {
        return IRContext::kAnalysisNone;
    }

  private:
    bool IsPatternOperand(uint32_t id);

    // Rewrite one integer vector bitwise op
    bool Scalarize(Instruction *inst);

    bool rewrite_all_;
};

} // namespace opt
} // namespace spvtools

#endif // SOURCE_OPT_MALI_BITWISE_SCALAR_PASS_H_
