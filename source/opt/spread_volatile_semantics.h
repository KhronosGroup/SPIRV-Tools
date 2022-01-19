// Copyright (c) 2022 Google LLC
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
  // Returns whether |var_id| is the result id of a target builtin variable for
  // the volatile semantics for |execution_model| based on the Vulkan spec
  // VUID-StandaloneSpirv-VulkanMemoryModel-04678 or
  // VUID-StandaloneSpirv-VulkanMemoryModel-04679.
  bool IsTargetForVolatileSemantics(uint32_t var_id,
                                    SpvExecutionModel execution_model);

  // Collects interface variables that needs the volatile semantics. If an
  // interface variable is used by two entry points and it is a target for
  // one entry point but not for another one, reports an error and returns
  // false. Otherwise, returns true.
  bool CollectTargetsForVolatileSemantics();

  // Sets Memory Operands of OpLoad instructions that load |var| as
  // Volatile.
  void SetVolatileForLoads(Instruction* var);

  // Adds OpDecorate Volatile for |var| if it does not exist.
  void DecorateVarWithVolatile(Instruction* var);

  // Returns whether we have to spread the volatile semantics for the
  // variable with the result id |var_id| or not.
  bool ShouldSpreadVolatileSemanticsForVariable(uint32_t var_id) {
    return var_ids_for_volatile_semantics_.find(var_id) !=
           var_ids_for_volatile_semantics_.end();
  }

  // Specifies that we have to spread the volatile semantics for the
  // variable with the result id |var_id|.
  void MarkVolatileSemanticsForVariable(uint32_t var_id) {
    var_ids_for_volatile_semantics_.insert(var_id);
  }

  // Result ids of variables to spread the volatile semantics.
  std::unordered_set<uint32_t> var_ids_for_volatile_semantics_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SPREAD_VOLATILE_SEMANTICS_H_
