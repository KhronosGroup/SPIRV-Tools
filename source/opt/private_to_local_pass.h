// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_
#define SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"

namespace spvtools {
namespace opt {

// This pass implements total redundancy elimination.  This is the same as
// local redundancy elimination except it looks across basic block boundaries.
// An instruction, inst, is totally redundant if there is another instruction
// that dominates inst, and also computes the same value.
class PrivateToLocalPass : public Pass {
 public:
  Status Process(opt::IRContext*) override;
  opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisDefUse |
           opt::IRContext::kAnalysisInstrToBlockMapping |
           opt::IRContext::kAnalysisDecorations |
           opt::IRContext::kAnalysisCombinators | opt::IRContext::kAnalysisCFG |
           opt::IRContext::kAnalysisDominatorAnalysis |
           opt::IRContext::kAnalysisNameMap;
  }

 private:
  // Moves |variable| from the private storage class to the function storage
  // class of |function|.
  void MoveVariable(opt::Instruction* variable, opt::Function* function);

  // |inst| is an instruction declaring a varible.  If that variable is
  // referenced in a single function and all of uses are valid as defined by
  // |IsValidUse|, then that function is returned.  Otherwise, the return
  // value is |nullptr|.
  opt::Function* FindLocalFunction(const opt::Instruction& inst) const;

  // Returns true is |inst| is a valid use of a pointer.  In this case, a
  // valid use is one where the transformation is able to rewrite the type to
  // match a change in storage class of the original variable.
  bool IsValidUse(const opt::Instruction* inst) const;

  // Given the result id of a pointer type, |old_type_id|, this function
  // returns the id of a the same pointer type except the storage class has
  // been changed to function.  If the type does not already exist, it will be
  // created.
  uint32_t GetNewType(uint32_t old_type_id);

  // Updates |inst|, and any instruction dependent on |inst|, to reflect the
  // change of the base pointer now pointing to the function storage class.
  void UpdateUse(opt::Instruction* inst);
  void UpdateUses(uint32_t id);
};

class PrivateToLocalPassToken : public PassToken {
 public:
  PrivateToLocalPassToken() = default;
  ~PrivateToLocalPassToken() override = default;

  const char* name() const override { return "private-to-local"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<PrivateToLocalPass>();
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_
