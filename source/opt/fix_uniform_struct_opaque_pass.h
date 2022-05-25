// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

#ifndef SOURCE_OPT_FIX_UNIFORM_STRUCT_OPAQUE_PASS_H
#define SOURCE_OPT_FIX_UNIFORM_STRUCT_OPAQUE_PASS_H

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Fix the structures used as uniform that contains opaque sampler types.
//
// Glslang, GLSL front-end, with Vulkan relaxed rules enabled,
// gathers uniforms declared in the global scope into a GL default uniform block.
// If these uniform have struct type with opaque members (samplers), glslang produces
// invalid SPIR-V.
// 
// This pass fixes these samplers access by extracting them into proper variables and
// fix storage classes of uniform access.
// Set descriptor set and automatically affect a binding on this variables.
class FixUniformStructOpaquePass : public Pass {
 public:

  FixUniformStructOpaquePass(uint32_t samplers_descriptor_set) : samplers_descriptor_set_(samplers_descriptor_set) {}

  const char* name() const override { return "uniform_struct_opaque_fixup"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisInstrToBlockMapping |                                    //////////// TODO, determine precisely
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisScalarEvolution |
           IRContext::kAnalysisRegisterPressure |
           IRContext::kAnalysisValueNumberTable |
           IRContext::kAnalysisStructuredCFG |
           IRContext::kAnalysisBuiltinVarId |
           IRContext::kAnalysisIdToFuncMapping | IRContext::kAnalysisTypes |
           IRContext::kAnalysisDefUse | IRContext::kAnalysisConstants;
  }

 private:
  const uint32_t samplers_descriptor_set_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FIX_UNIFORM_STRUCT_OPAQUE_PASS_H
