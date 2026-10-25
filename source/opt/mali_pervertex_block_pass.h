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

#ifndef SOURCE_OPT_MALI_PERVERTEX_BLOCK_PASS_H_
#define SOURCE_OPT_MALI_PERVERTEX_BLOCK_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// MaliBridge (r44p1, fixed r47p0): ddk crashes inside its compiler on
// geometry/tessellation pipelines whose per-vertex built-ins (Position,
// PointSize, ClipDistance, CullDistance) are loose variables instead of members
// of a gl_PerVertex block. glslang always emits the block; Zink emits loose
// variables.
//
// This pass folds a stage's loose per-vertex built-ins of each interface
// (input, output) into one Block struct and redirects their uses to its
// members. Anything unexpected leaves the module untouched.
class MaliPerVertexBlockPass : public Pass {
  public:
    MaliPerVertexBlockPass() = default;
    const char *name() const override { return "mali-pervertex-block"; }
    Status Process() override;

    IRContext::Analysis GetPreservedAnalyses() override {
        return IRContext::kAnalysisNone;
    }
};

} // namespace opt
} // namespace spvtools

#endif // SOURCE_OPT_MALI_PERVERTEX_BLOCK_PASS_H_
