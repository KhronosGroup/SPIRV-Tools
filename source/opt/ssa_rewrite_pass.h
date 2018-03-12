// Copyright (c) 2018 Google LLC.
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

#ifndef LIBSPIRV_OPT_SSA_REWRITE_PASS_H_
#define LIBSPIRV_OPT_SSA_REWRITE_PASS_H_

#include "basic_block.h"
#include "ir_context.h"
#include "mem_pass.h"

#include <unordered_map>

namespace spvtools {
namespace opt {

// Utility class for passes that need to rewrite a function into SSA.  This
// converts load/store operations on function-local variables into SSA IDs,
// which allows them to be the target of optimizing transformations.
//
// Store and load operations to these variables are converted into
// operations on SSA IDs.  Phi instructions are added when needed.  See the
// SSA construction paper for algorithmic details
// (https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6)
class SSARewriter {
 public:
  SSARewriter(MemPass* pass)
      : pass_(pass), first_phi_id_(pass_->get_module()->IdBound()) {}

  // Rewrites SSA-target variables in function |fp| into SSA.  This is the
  // entry point for the SSA rewrite algorithm.  SSA-target variables are
  // locally defined variables that meet the criteria set by IsSSATargetVar.
  //
  // It returns true if function |fp| was modified.  Otherwise, it returns
  // false.
  bool RewriteFunctionIntoSSA(ir::Function* fp);

 private:
  class PhiCandidate {
   public:
    explicit PhiCandidate(uint32_t var, uint32_t result, ir::BasicBlock* block)
        : var_id_(var),
          result_id_(result),
          bb_(block),
          phi_args_(),
          is_trivial_(false),
          is_complete_(false),
          users_() {}

    PhiCandidate(const PhiCandidate&) = delete;
    PhiCandidate(PhiCandidate&&) = default;
    PhiCandidate& operator=(const PhiCandidate&) = delete;

    uint32_t var_id() const { return var_id_; }
    uint32_t result_id() const { return result_id_; }
    ir::BasicBlock* bb() const { return bb_; }
    std::vector<uint32_t>& phi_args() { return phi_args_; }
    const std::vector<uint32_t>& phi_args() const { return phi_args_; }
    bool is_trivial() const { return is_trivial_; }
    bool is_complete() const { return is_complete_; }
    std::vector<uint32_t>& users() { return users_; }
    const std::vector<uint32_t>& users() const { return users_; }

    // Marks this phi candidate as trivial.
    void MarkTrivial() { is_trivial_ = true; }

    // Marks this phi candidate as incomplete.
    void MarkIncomplete() { is_complete_ = false; }

    // Marks this phi candidate as complete.
    void MarkComplete() { is_complete_ = true; }

    // Pretty prints this Phi candidate into a string and returns it. |cfg| is
    // needed to lookup basic block predecessors.
    std::string PrettyPrint(const ir::CFG* cfg) const;

    // Registers |operand_id| as a user of this Phi candidate.
    void AddUser(uint32_t operand_id) { users_.push_back(operand_id); }

   private:
    // Variable ID that this Phi is merging.
    uint32_t var_id_;

    // SSA ID generated by this Phi (i.e., this is the result ID of the eventual
    // Phi instruction).
    uint32_t result_id_;

    // Basic block to hold this Phi.
    ir::BasicBlock* bb_;

    // Vector of operands for every predecessor block of |bb|.  This vector is
    // organized so that the Ith slot contains the argument coming from the Ith
    // predecessor of |bb|.
    std::vector<uint32_t> phi_args_;

    // True, if this Phi candidate has been found to be trivial.
    bool is_trivial_;

    // False, if this Phi candidate has no arguments or at least one argument is
    // %0.
    bool is_complete_;

    // List of all users for this Phi instruction. Each element is the result ID
    // of the load instruction replaced by this Phi, or the result ID of a Phi
    // candidate that has this Phi in its list of operands.
    std::vector<uint32_t> users_;
  };

  // Type used to keep track of store operations in each basic block.
  typedef std::unordered_map<ir::BasicBlock*,
                             std::unordered_map<uint32_t, uint32_t>>
      BlockDefsMap;

  // Generates all the SSA rewriting decisions for basic block |bb|.  This
  // populates the Phi candidate table (|phi_candidate_|) and the load
  // replacement table (|load_replacement_).
  void GenerateSSAReplacements(ir::BasicBlock* bb);

  // Seals block |bb|.  Sealing a basic block means |bb| and all its
  // predecessors of |bb| have been scanned for loads/stores.
  void SealBlock(ir::BasicBlock* bb);

  // Returns true if |bb| has been sealed.
  bool IsBlockSealed(ir::BasicBlock* bb) {
    return sealed_blocks_.find(bb) != sealed_blocks_.end();
  }

  // Returns the Phi candidate with result ID |id| if it exists in the table
  // |phi_candidates_|. If no such Phi candidate exists, it returns nullptr.
  PhiCandidate* GetPhiCandidate(uint32_t id) {
    auto it = phi_candidates_.find(id);
    return (it != phi_candidates_.end()) ? &it->second : nullptr;
  }

  // Replaces all the users of Phi candidate |phi_cand| to be users of
  // |repl_id|.
  void ReplacePhiUsersWith(const PhiCandidate& phi_cand, uint32_t repl_id);

  // Returns the value ID that should replace the load ID in the given
  // replacement pair |repl|.  The replacement is a pair (|load_id|, |val_id|).
  // If |val_id| is itself replaced by another value in the table, this function
  // will look the replacement for |val_id| until it finds one that is not
  // itself replaced.  For instance, given:
  //
  //            %34 = OpLoad %float %f1
  //            OpStore %t %34
  //            %36 = OpLoad %float %t
  //
  // Assume that %f1 is reached by a Phi candidate %42, the load
  // replacement table will have the following entries:
  //
  //            %34 -> %42
  //            %36 -> %34
  //
  // So, when looking for the replacement for %36, we should not use
  // %34. Rather, we should use %42.  To do this, the chain of
  // replacements must be followed until we reach an element that has
  // no replacement.
  uint32_t GetReplacement(std::pair<uint32_t, uint32_t> repl);

  // Applies all the SSA replacement decisions.  This replaces loads/stores to
  // SSA target variables with their corresponding SSA IDs, and inserts Phi
  // instructions for them.
  bool ApplyReplacements();

  // Registers a definition for variable |var_id| in basic block |bb| with
  // value |val_id|.
  void WriteVariable(uint32_t var_id, ir::BasicBlock* bb, uint32_t val_id) {
    defs_at_block_[bb][var_id] = val_id;
  }

  // Processes the store operation |inst| in basic block |bb|. This extracts the
  // variable ID being stored into, determines whether the variable is an
  // SSA-target variable, and, if it is, it stores its value in the
  // |defs_at_block_| map.
  void ProcessStore(ir::Instruction* inst, ir::BasicBlock* bb);

  // Processes the load operation |inst| in basic block |bb|. This extracts the
  // variable ID being stored into, determines whether the variable is an
  // SSA-target variable, and, if it is, it reads its reaching definition by
  // calling |GetReachingDef|.
  void ProcessLoad(ir::Instruction* inst, ir::BasicBlock* bb);

  // Reads the current definition for variable |var_id| in basic block |bb|.
  // If |var_id| is not defined in block |bb| it walks up the predecessors of
  // |bb|, creating new Phi instructions along the way, if needed.
  //
  // It returns the value for |var_id| from the RHS of the current reaching
  // definition for |var_id|.
  uint32_t GetReachingDef(uint32_t var_id, ir::BasicBlock* bb);

  // Adds arguments to |phi_candidate| by getting the reaching definition of
  // |phi_candidate|'s variable on each of the predecessors of its basic block.
  // After populating the argument list, it determines whether all its arguments
  // are the same.  If so, it returns the ID of the argument that this Phi
  // copies.
  uint32_t AddPhiOperands(PhiCandidate* phi_candidate);

  // Creates a Phi candidate instruction for variable |var_id| in basic block
  // |bb|.
  //
  // Since the rewriting algorithm may remove Phi candidates when it finds them
  // to be trivial, we avoid the expense of creating actual Phi instructions by
  // keeping a pool of Phi candidates (|phi_candidates_|) during rewriting.
  //
  // Once the candidate Phi is created, it returns its ID.
  PhiCandidate& CreatePhiCandidate(uint32_t var_id, ir::BasicBlock* bb);

  // Attempts to remove a trivial Phi candidate |phi_cand|. Trivial Phis are
  // those that only reference themselves and one other value |val| any number
  // of times. This will try to remove any other Phis that become trivial after
  // |phi_cand| is removed.
  //
  // If |phi_cand| is trivial, it returns the SSA ID for the value that should
  // replace it.  Otherwise, it returns the SSA ID for |phi_cand|.
  uint32_t TryRemoveTrivialPhi(PhiCandidate* phi_cand);

  // Finalizes |phi_candidate| by replacing every argument that is still %0 with
  // its reaching definition.
  void FinalizePhiCandidate(PhiCandidate* phi_candidate);

  // Finalizes processing of Phi candidates.  Once the whole function has been
  // scanned for loads and stores, the CFG will still have some incomplete and
  // trivial Phis.  This will add missing arguments and remove trivial Phi
  // candidates.
  void FinalizePhiCandidates();

  // Prints the table of Phi candidates to std::cerr.
  void PrintPhiCandidates() const;

  // Prints the load replacement table to std::cerr.
  void PrintReplacementTable() const;

  // Map holding the value of every SSA-target variable at every basic block
  // where the variable is stored. defs_at_block_[block][var_id] = val_id
  // means that there is a store instruction for variable |var_id| at basic
  // block |block| with value |val_id|.
  BlockDefsMap defs_at_block_;

  // Map, indexed by Phi ID, holding all the Phi candidates created during SSA
  // rewriting.  |phi_candidates_[id]| returns the Phi candidate whose result is
  // |id|.
  std::unordered_map<uint32_t, PhiCandidate> phi_candidates_;

  // Queue of incomplete Phi candidates. These are Phi candidates created at
  // unsealed blocks. They need to be completed before they are instantiated in
  // ApplyReplacements.
  std::queue<PhiCandidate*> incomplete_phis_;

  // List of completed Phi candidates.  These are the only candidates that will
  // become real Phi instructions.
  std::vector<PhiCandidate*> phis_to_generate_;

  // SSA replacement table.  This maps variable IDs, resulting from a load
  // operation, to the value IDs that will replace them after SSA rewriting.
  // After all the rewriting decisions are made, a final scan through the IR is
  // done to replace all uses of the original load ID with the value ID.
  std::unordered_map<uint32_t, uint32_t> load_replacement_;

  // Set of blocks that have been sealed already.
  std::unordered_set<ir::BasicBlock*> sealed_blocks_;

  // Memory pass requesting the SSA rewriter.
  MemPass* pass_;

  // ID of the first Phi created by the SSA rewriter.  During rewriting, any ID
  // bigger than this corresponds to a Phi candidate.
  uint32_t first_phi_id_;
};

class SSARewritePass : public MemPass {
 public:
  SSARewritePass() = default;
  const char* name() const override { return "ssa-rewrite"; }
  Status Process(ir::IRContext* c) override;

 private:
  // Initializes the pass.
  void Initialize(ir::IRContext* c);
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSA_REWRITE_PASS_H_
