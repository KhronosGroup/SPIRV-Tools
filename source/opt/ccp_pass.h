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

#ifndef LIBSPIRV_OPT_CCP_PASS_H_
#define LIBSPIRV_OPT_CCP_PASS_H_

#include "constants.h"
#include "function.h"
#include "ir_context.h"
#include "mem_pass.h"
#include "module.h"
#include "propagator.h"

namespace spvtools {
namespace opt {

class CCPPass : public MemPass {
 public:
  CCPPass() = default;
  const char* name() const override { return "ccp"; }
  Status Process(ir::IRContext* c) override;
  virtual ir::IRContext::Analysis GetPreservedAnalyses() override {
    return ir::IRContext::kAnalysisDefUse |
           ir::IRContext::kAnalysisInstrToBlockMapping |
           ir::IRContext::kAnalysisDecorations |
           ir::IRContext::kAnalysisCombinators | ir::IRContext::kAnalysisCFG |
           ir::IRContext::kAnalysisDominatorAnalysis |
           ir::IRContext::kAnalysisNameMap;
  }

 private:
  // Initializes the pass.
  void Initialize(ir::IRContext* c);

  // Runs constant propagation on the given function |fp|. Returns true if any
  // constants were propagated and the IR modified.
  bool PropagateConstants(ir::Function* fp);

  // Visits a single instruction |instr|.  If the instruction is a conditional
  // branch that always jumps to the same basic block, it sets the destination
  // block in |dest_bb|.
  SSAPropagator::PropStatus VisitInstruction(ir::Instruction* instr,
                                             ir::BasicBlock** dest_bb);

  // Visits an OpPhi instruction |phi|. This applies the meet operator for the
  // CCP lattice. Essentially, if all the operands in |phi| have the same
  // constant value C, the result for |phi| gets assigned the value C.
  SSAPropagator::PropStatus VisitPhi(ir::Instruction* phi);

  // Visits an SSA assignment instruction |instr|.  If the RHS of |instr| folds
  // into a constant value C, then the LHS of |instr| is assigned the value C in
  // |values_|.
  SSAPropagator::PropStatus VisitAssignment(ir::Instruction* instr);

  // Visits a branch instruction |instr|. If the branch is conditional
  // (OpBranchConditional or OpSwitch), and the value of its selector is known,
  // |dest_bb| will be set to the corresponding destination block. Unconditional
  // branches always set |dest_bb| to the single destination block.
  SSAPropagator::PropStatus VisitBranch(ir::Instruction* instr,
                                        ir::BasicBlock** dest_bb) const;

  // Replaces all operands used in |fp| with the corresponding constant values
  // in |values_|.  Returns true if any operands were replaced, and false
  // otherwise.
  bool ReplaceValues();

  // Marks |instr| as varying by registering a varying value for its result
  // into the |values_| table. Returns SSAPropagator::kVarying.
  SSAPropagator::PropStatus MarkInstructionVarying(ir::Instruction* instr);

  // Returns true if |id| is the special SSA id that corresponds to a varying
  // value.
  bool IsVaryingValue(uint32_t id) const;

  // Constant manager for the parent IR context.  Used to record new constants
  // generated during propagation.
  analysis::ConstantManager* const_mgr_;

  // Constant value table.  Each entry <id, const_decl_id> in this map
  // represents the compile-time constant value for |id| as declared by
  // |const_decl_id|. Each |const_decl_id| in this table is an OpConstant
  // declaration for the current module.
  //
  // Additionally, this table keeps track of SSA IDs with varying values. If an
  // SSA ID is found to have a varying value, it will have an entry in this
  // table that maps to the special SSA id kVaryingSSAId.  These values are
  // never replaced in the IR, they are used by CCP during propagation.
  std::unordered_map<uint32_t, uint32_t> values_;

  // Propagator engine used.
  std::unique_ptr<SSAPropagator> propagator_;
};

}  // namespace opt
}  // namespace spvtools

#endif
