// Copyright (c) 2021 Google LLC
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

#ifndef SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_
#define SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SpreadVolatileSemantics : public Pass {
 public:
  SpreadVolatileSemantics() {}

  const char* name() const override { return "spread-volatile-semantics"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisDecorations;
  }

 private:
  // Returns whether |var| is a target builtin variable for the volatile
  // semantics based on the Vulkan spec
  // VUID-StandaloneSpirv-VulkanMemoryModel-04678 or
  // VUID-StandaloneSpirv-VulkanMemoryModel-04679.
  bool IsTargetForVolatileSemantics(Instruction* var);

  // Sets Memory Operands of OpLoad instructions that load |var| as
  // Volatile.
  void SetVolatileForLoads(Instruction* var);

  // Adds OpDecorate Volatile for |var| if it does not exist.
  void DecorateVarWithVolatile(Instruction* var);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_
