// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_OPT_CODE_SINK_H_
#define SOURCE_OPT_CODE_SINK_H_

#include <unordered_map>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class CodeSinkingPass : public Pass {
 public:
  const char* name() const override { return "code-sink"; }
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
  // Sinks the instructions in |bb| as much as possible.  Returns true if
  // something changes.
  bool SinkInstructionsInBB(BasicBlock* bb);

  // Tries the sink |inst| as much as possible.  Returns true if the instruction
  // is moved.
  bool SinkInstruction(Instruction* inst);

  // Returns the basic block in which to move |inst| to move is as close as
  // possible to the uses of |inst| without increasing the number of time |inst|
  // will be executed.  Return |nullptr| if there is not need to move |inst|.
  BasicBlock* FindNewBasicBlockFor(Instruction* inst);

  // Return true if |inst| reference memory and it is possible that the data in
  // the memory changes at some point.
  bool ReferencesMutableMemory(Instruction* inst);

  // Returns true if the module contains that instruction that is a memory sync.
  bool HasUniformMemorySync();

  // Returns true if there is it possible that there is a store ti the variable
  // |var_inst|.
  bool HasPossibleStore(Instruction* var_inst);

  // Returns true if one of the basic blocks in |set| exists on a path from the
  // basic block |start| to |end|.
  bool IntersectsPath(uint32_t start, uint32_t end,
                      const std::unordered_set<uint32_t>& set);

  // Returns true if |mem_semantics_id| is a memory semantics is that will
  // implies any memory ordering constraints.
  bool IsSyncOnUniform(uint32_t mem_semantics_id) const;

  // Stores whether or not a check has for uniform storage had taken place.
  bool checked_for_uniform_sync_;

  // Cache of whether or not the module has a memory sync on uniform storage.
  // only valid if |check_for_uniform_sync_| is true.
  bool has_uniform_sync_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CODE_SINK_H_
