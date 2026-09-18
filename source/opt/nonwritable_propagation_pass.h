// Copyright (c) 2026 The Khronos Group Inc.
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

#ifndef LIBSPIRV_OPT_NONWRITABLE_PROPAGATION_PASS_H_
#define LIBSPIRV_OPT_NONWRITABLE_PROPAGATION_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Propagates NonWritable from a buffer's members to the buffer variable.
//
// Some producers decorate every member of a buffer's struct NonWritable
// instead of decorating the variable itself.  That is legal, but consumers
// that read the decoration off the variable -- for example the mapping APIs
// in VK_EXT_descriptor_heap -- do not see it, so the buffer looks writable.
//
// For each buffer variable (OpVariable or OpUntypedVariableKHR) whose
// struct has every member decorated NonWritable, this pass decorates the
// variable NonWritable and then removes the now-redundant member
// decorations.  Every direct member must carry the decoration, whatever its
// type: a struct-typed member without it is writable as far as SPIR-V is
// concerned, so the variable is left alone.
class NonWritablePropagationPass : public Pass {
 public:
  const char* name() const override { return "nonwritable-propagation"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    // Decorations are the only thing added or removed, and the removal goes
    // through the context, so every analysis except the decoration cache
    // survives untouched.
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisDominatorAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Returns the OpTypeStruct backing |var|'s buffer, or nullptr if |var| is
  // not a variable that may carry NonWritable itself.
  Instruction* GetBufferStructType(const Instruction& var) const;

  // Returns true if every member of |struct_type| is decorated NonWritable.
  bool AllMembersNonWritable(const Instruction& struct_type) const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_NONWRITABLE_PROPAGATION_PASS_H_
