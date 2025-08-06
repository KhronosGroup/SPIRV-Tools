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

#ifndef SOURCE_OPT_REMOVE_CLIP_CULL_DIST_PASS_H_
#define SOURCE_OPT_REMOVE_CLIP_CULL_DIST_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LowerClipCullDistancePass : public MemPass {
 public:
  const char* name() const override { return "lower-clip-cull-dist"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisNone;
  }

 private:
  struct BuiltinVariableInfo {
    Instruction* position_var = nullptr;
    uint32_t position_member_index = 0; // The member index if Position is in a struct.
    Instruction* clip_dist_var = nullptr;
    Instruction* cull_dist_var = nullptr;
  };

  enum PassStatus {
    NO_CHANGES,
    CLEANUP_ONLY,
    EMULATED,
  };

  // Main processing function for a single entry point.
  PassStatus ProcessEntryPoint(Instruction* entry_point,
                         std::unordered_set<Instruction*>* dead_builtins);

  // Phase 1: Identification.
  PassStatus FindBuiltinVariables(Instruction* entry_point, BuiltinVariableInfo* info);
  void FindRelevantStores(Instruction* builtin_var,
                          std::vector<Instruction*>* stores);

  // Phase 2: Transformation.
  void InjectClippingCode(Instruction* store_inst,
                          const BuiltinVariableInfo& builtins,
                          spv::ExecutionModel exec_model);
  void InjectScalarCheck(Instruction* store_inst,
                         const BuiltinVariableInfo& builtins,
                         spv::ExecutionModel exec_model);
  void InjectVectorCheck(Instruction* store_inst,
                         const BuiltinVariableInfo& builtins,
                         spv::ExecutionModel exec_model);
  Instruction* FindPositionPointerForStore(Instruction* store_inst,
                                           const BuiltinVariableInfo& builtins,
                                           spv::ExecutionModel exec_model);

  void Cleanup(const std::vector<Instruction*>& dead_vars);

  uint32_t GetConstFloatId(float value);
  uint32_t GetConstUintId(uint32_t value);
  uint32_t GetPointerToFloatTypeId();
  uint32_t GetBoolTypeId();

  std::unordered_map<float, uint32_t> float_const_ids_;
  std::unordered_map<uint32_t, uint32_t> uint_const_ids_;
  uint32_t ptr_to_float_type_id_ = 0;
  uint32_t bool_type_id_ = 0;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REMOVE_CLIP_CULL_DIST_PASS_H_