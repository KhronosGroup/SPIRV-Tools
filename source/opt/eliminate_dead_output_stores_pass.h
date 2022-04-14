// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#ifndef SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_
#define SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_

#include <unordered_set>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class EliminateDeadOutputStoresPass : public Pass {
 public:
  explicit EliminateDeadOutputStoresPass(
      std::unordered_set<uint32_t>* live_inputs, bool analyze)
      : live_inputs_(live_inputs), analyze_(analyze) {}

  const char* name() const override { return "find-live-input-components"; }
  Status Process() override;

  // Return the mask of preserved Analyses.
  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Initialize analysis
  void InitializeAnalysis();

  // Initialize elimination
  void InitializeElimination();

  // Do dead output store analysis
  Status DoDeadOutputStoreAnalysis();

  // Do dead output store analysis
  Status DoDeadOutputStoreElimination();

  // Mark all locations live
  void MarkAllLocsLive();

  // Mark all live locations resulting from |user| of |var| at |loc|.
  void MarkRefLive(const Instruction* user, Instruction* var);

  // Kill all dead stores resulting from |user| of |var| at |loc|.
  void KillAllDeadStoresOfRef(Instruction* user, Instruction* var);

  // Kill all dead stores resulting from |ac| of |var| at |loc|.
  void AnalyzeAccessChain(const Instruction* ac,
                          const analysis::Type** curr_type, uint32_t* offset,
                          bool* no_loc, bool input = true);

  // Mark |num| locations starting at location |start|.
  void MarkLocsLive(uint32_t start, uint32_t num);

  // Return true if any of |num| locations starting at location |start| are
  // live.
  bool AnyLocsAreLive(uint32_t start, uint32_t num);

  // Return size of |type_id| in units of locations
  uint32_t GetLocSize(const analysis::Type* type) const;

  // Return offset of |index| into aggregate type |agg_type| in units of
  // input locations
  uint32_t GetLocOffset(uint32_t index, const analysis::Type* agg_type) const;

  // Return type of component of aggregate type |agg_type| at |index|
  const analysis::Type* GetComponentType(uint32_t index,
                                         const analysis::Type* agg_type) const;

  std::unordered_set<uint32_t>* live_inputs_;
  std::vector<Instruction*> kill_list_;
  bool analyze_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_
