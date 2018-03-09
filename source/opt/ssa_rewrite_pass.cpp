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

// This file implements the SSA rewriting algorithm proposed in
//
//      Simple and Efficient Construction of Static Single Assignment Form.
//      Braun M., Buchwald S., Hack S., Leißa R., Mallon C., Zwinkau A. (2013)
//      In: Jhala R., De Bosschere K. (eds)
//      Compiler Construction. CC 2013.
//      Lecture Notes in Computer Science, vol 7791.
//      Springer, Berlin, Heidelberg
//
//      https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6
//
// In contrast to common eager algorithms based on dominance and dominance
// frontier information, this algorithm works backwards from load operations.
//
// When a target variable is loaded, it queries the variable's reaching
// definition.  If the reaching definition is unknown at the current location,
// it searches backwards in the CFG, inserting Phi instructions at join points
// in the CFG along the way until it finds the desired store instruction.
//
// The algorithm avoids repeated lookups using memoization.
//
// For reducible CFGs, which are a superset of the structured CFGs in SPIRV,
// this algorithm is proven to produce minimal SSA.  That is, it inserts the
// minimal number of Phi instructions required to ensure the SSA property, but
// some Phi instructions may be dead
// (https://en.wikipedia.org/wiki/Static_single_assignment_form).

#include "ssa_rewrite_pass.h"
#include "cfg.h"
#include "make_unique.h"
#include "mem_pass.h"
#include "opcode.h"

#include <sstream>

// Debug logging (0: Off, 1-N: Verbosity level).  Replace this with the
// implementation done for
// https://github.com/KhronosGroup/SPIRV-Tools/issues/1351
// #define SSA_REWRITE_DEBUGGING_LEVEL 3

#ifdef SSA_REWRITE_DEBUGGING_LEVEL
#include <ostream>
#else
#define SSA_REWRITE_DEBUGGING_LEVEL 0
#endif

namespace spvtools {
namespace opt {

namespace {
const uint32_t kStoreValIdInIdx = 1;
const uint32_t kVariableInitIdInIdx = 1;
}  // namespace

std::string SSARewriter::PhiCandidate::PrettyPrint(const ir::CFG* cfg) const {
  std::ostringstream str;
  bool is_incomplete = false;
  str << "%" << result_id_ << " = Phi[%" << var_id_ << ", BB %" << bb_->id()
      << "](";
  if (phi_args_.size() > 0) {
    uint32_t arg_ix = 0;
    for (uint32_t pred_label : cfg->preds(bb_->id())) {
      uint32_t arg_id = phi_args_[arg_ix++];
      str << "[%" << arg_id << ", bb(%" << pred_label << ")] ";
      if (arg_id == 0) is_incomplete = true;
    }
  } else {
    is_incomplete = true;
  }
  str << ")" << ((is_trivial_) ? " [TRIVIAL PHI]" : "")
      << ((is_incomplete) ? "  [INCOMPLETE]" : "");

  return str.str();
}

SSARewriter::PhiCandidate& SSARewriter::CreatePhiCandidate(
    uint32_t var_id, ir::BasicBlock* bb,
    std::unique_ptr<std::vector<uint32_t>> phi_args) {
  uint32_t phi_result_id = pass_->context()->TakeNextId();
  auto result = phi_candidates_.emplace(
      phi_result_id, PhiCandidate(var_id, phi_result_id, bb));

  // We should never try to create more than one Phi candidate for the same
  // |var_id| on |bb|.
  assert(result.second == true);

  PhiCandidate& phi_candidate = result.first->second;
  if (phi_args) {
    phi_candidate.SetPhiArgs(*phi_args);
    for (auto arg : phi_candidate.phi_args()) {
      PhiCandidate* defining_phi = GetPhiCandidate(arg);
      if (defining_phi) {
        defining_phi->AddUser(phi_candidate.result_id());
      }
    }
  }

  return phi_candidate;
}

std::unique_ptr<std::vector<uint32_t>> SSARewriter::GetPhiOperands(
    uint32_t var_id, ir::BasicBlock* bb, uint32_t* same_id) {
  uint32_t prev_arg = 0;
  bool first_arg = true, all_same = true;
  std::unique_ptr<std::vector<uint32_t>> operands =
      MakeUnique<std::vector<uint32_t>>();

  *same_id = 0;
  for (uint32_t pred : pass_->cfg()->preds(bb->id())) {
    // Only try to get the reaching definition for this edge if the
    // corresponding block is already sealed. Otherwise, we will be creating
    // unnecessary empty Phis.
    ir::BasicBlock* pred_bb = pass_->cfg()->block(pred);
    uint32_t arg =
        (IsBlockSealed(pred_bb)) ? GetReachingDef(var_id, pred_bb) : 0;
    operands->push_back(arg);
    if (!first_arg && prev_arg != arg) {
      all_same = false;
    }
    prev_arg = arg;
    first_arg = false;
  }

  if (all_same) {
    *same_id = prev_arg;
  }

  return std::move(operands);
}

uint32_t SSARewriter::MaybeCreatePhiCandidate(uint32_t var_id,
                                              ir::BasicBlock* bb) {
  uint32_t same_id;
  std::unique_ptr<std::vector<uint32_t>> phi_args =
      GetPhiOperands(var_id, bb, &same_id);
  if (same_id > 0) {
    // If all the operands are the same ID, we do not need to create a
    // new Phi candidate.  Instead, return the reaching definition
    // for |var_id| that reaches all the operands.
    return same_id;
  }

  PhiCandidate& phi_cand = CreatePhiCandidate(var_id, bb, std::move(phi_args));
  return phi_cand.result_id();
}

uint32_t SSARewriter::GetReachingDef(uint32_t var_id, ir::BasicBlock* bb) {
  // If |var_id| has a definition in |bb|, return it.
  const auto& bb_it = defs_at_block_.find(bb);
  if (bb_it != defs_at_block_.end()) {
    const auto& current_defs = bb_it->second;
    const auto& var_it = current_defs.find(var_id);
    if (var_it != current_defs.end()) {
      return var_it->second;
    }
  }

  // Otherwise, look up the value for |var_id| in |bb|'s predecessors.
  uint32_t val_id = 0;
  size_t num_preds = pass_->cfg()->preds(bb->id()).size();
  if (!IsBlockSealed(bb)) {
    // If |bb| is not yet sealed (i.e., it still has not been processed), create
    // an empty Phi instruction for |var_id|.  This will act as a proxy for when
    // we determine the real reaching definition for |var_id| after the whole
    // CFG has been processed.
    auto& phi_cand = CreatePhiCandidate(var_id, bb, nullptr);
    val_id = phi_cand.result_id();
  } else if (num_preds == 1) {
    // If |bb| has exactly one predecessor, we look for |var_id|'s definition
    // there.
    ir::BasicBlock* pred_bb =
        pass_->cfg()->block(pass_->cfg()->preds(bb->id())[0]);
    val_id = GetReachingDef(var_id, pred_bb);
  } else if (num_preds > 1) {
    // If there is more than one predecessor, this is a join block which may
    // require a Phi instruction.  This will act as |var_id|'s current
    // definition to break potential cycles.
    val_id = MaybeCreatePhiCandidate(var_id, bb);
  }

  // If we could not find a store for this variable in the path from the root
  // of the CFG, the variable is not defined, so we use undef.
  if (val_id == 0) {
    val_id = pass_->GetUndefVal(var_id);
  }

  WriteVariable(var_id, bb, val_id);

  return val_id;
}

void SSARewriter::SealBlock(ir::BasicBlock* bb) {
  auto result = sealed_blocks_.insert(bb);
  (void)result;
  assert(result.second == true &&
         "Tried to seal the same basic block more than once.");
}

void SSARewriter::ProcessStore(ir::Instruction* inst, ir::BasicBlock* bb) {
  auto opcode = inst->opcode();
  assert((opcode == SpvOpStore || opcode == SpvOpVariable) &&
         "Expecting a store or a variable definition instruction.");

  uint32_t var_id = 0;
  uint32_t val_id = 0;
  if (opcode == SpvOpStore) {
    (void)pass_->GetPtr(inst, &var_id);
    val_id = inst->GetSingleWordInOperand(kStoreValIdInIdx);
  } else if (inst->NumInOperands() >= 2) {
    var_id = inst->result_id();
    val_id = inst->GetSingleWordInOperand(kVariableInitIdInIdx);
  }
  if (pass_->IsSSATargetVar(var_id)) {
    WriteVariable(var_id, bb, val_id);

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
    std::cerr << "\tFound store '%" << var_id << " = %" << val_id << "': "
              << inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << "\n";
#endif
  }
}

void SSARewriter::ProcessLoad(ir::Instruction* inst, ir::BasicBlock* bb) {
  uint32_t var_id = 0;
  (void)pass_->GetPtr(inst, &var_id);
  if (pass_->IsSSATargetVar(var_id)) {
    // Get the immediate reaching definition for |var_id|.
    uint32_t val_id = GetReachingDef(var_id, bb);

    // Schedule a replacement for the result of this load instruction with
    // |val_id|. After all the rewriting decisions are made, every use of
    // this load will be replaced with |val_id|.
    const uint32_t load_id = inst->result_id();
    assert(load_replacement_.find(load_id) == load_replacement_.end());
    load_replacement_[load_id] = val_id;
    PhiCandidate* defining_phi = GetPhiCandidate(val_id);
    if (defining_phi) {
      defining_phi->AddUser(load_id);
    }

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
    std::cerr << "\tFound load: "
              << inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << " (replacement for %" << load_id << " is %" << val_id << ")\n";
#endif
  }
}

void SSARewriter::PrintPhiCandidates() const {
  std::cerr << "\nPhi candidates:\n";
  for (const auto& phi_it : phi_candidates_) {
    std::cerr << "\tBB %" << phi_it.second.bb()->id() << ": "
              << phi_it.second.PrettyPrint(pass_->cfg()) << "\n";
  }
  std::cerr << "\n";
}

void SSARewriter::PrintReplacementTable() const {
  std::cerr << "\nLoad replacement table\n";
  for (const auto& it : load_replacement_) {
    std::cerr << "\t%" << it.first << " -> %" << it.second << "\n";
  }
  std::cerr << "\n";
}

void SSARewriter::GenerateSSAReplacements(ir::BasicBlock* bb) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "Generating SSA replacements for block: " << bb->id() << "\n";
  std::cerr << bb->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << "\n";
#endif

  // Seal |bb|. This means that all the stores in it have been scanned and it's
  // ready to feed them into its successors. Note that we seal the block before
  // we scan its instructions, this way loads reading from stores within |bb| do
  // not create unnecessary trivial Phi candidates.
  SealBlock(bb);

  for (auto& inst : *bb) {
    auto opcode = inst.opcode();
    if (opcode == SpvOpStore || opcode == SpvOpVariable) {
      ProcessStore(&inst, bb);
    } else if (inst.opcode() == SpvOpLoad) {
      ProcessLoad(&inst, bb);
    }
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  PrintPhiCandidates();
  PrintReplacementTable();
  std::cerr << "\n\n";
#endif
}

uint32_t SSARewriter::GetReplacement(std::pair<uint32_t, uint32_t> repl) {
  uint32_t val_id = repl.second;
  auto it = load_replacement_.find(val_id);
  while (it != load_replacement_.end()) {
    val_id = it->second;
    it = load_replacement_.find(val_id);
  }
  return val_id;
}

bool SSARewriter::ApplyReplacements() {
  bool modified = false;

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "\n\nApplying replacement decisions to IR\n\n";

  PrintPhiCandidates();
  PrintReplacementTable();
  std::cerr << "\n\n";
#endif

  // Add Phi instructions from all Phi candidates. Note that we traverse the map
  // of candidates in numeric ID order to emit instructions in a stable
  // sequence.
  std::vector<ir::Instruction*> generated_phis;
  for (uint32_t id = first_phi_id_; id < pass_->get_module()->IdBound(); id++) {
    const auto& it = phi_candidates_.find(id);
    if (it == phi_candidates_.end()) continue;
    const PhiCandidate& phi_candidate = it->second;

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
    std::cerr << "Phi candidate: " << phi_candidate.PrettyPrint(pass_->cfg())
              << "\n";
#endif

    if (phi_candidate.is_trivial()) {
      continue;
    }

    assert(!phi_candidate.IsIncomplete() &&
           "Tried to instantiate a Phi instruction from an incomplete Phi "
           "candidate");

    // Build the vector of operands for the new OpPhi instruction.
    uint32_t type_id = pass_->GetPointeeTypeId(
        pass_->get_def_use_mgr()->GetDef(phi_candidate.var_id()));
    std::vector<ir::Operand> phi_operands;
    uint32_t arg_ix = 0;
    for (uint32_t pred_label : pass_->cfg()->preds(phi_candidate.bb()->id())) {
      uint32_t op_val_id = phi_candidate.phi_args()[arg_ix++];
      phi_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {op_val_id}});
      phi_operands.push_back(
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {pred_label}});
    }

    // Generate a new OpPhi instruction and insert it in its basic
    // block.
    std::unique_ptr<ir::Instruction> phi_inst(
        new ir::Instruction(pass_->context(), SpvOpPhi, type_id,
                            phi_candidate.result_id(), phi_operands));
    generated_phis.push_back(phi_inst.get());
    pass_->get_def_use_mgr()->AnalyzeInstDef(&*phi_inst);
    pass_->context()->set_instr_block(&*phi_inst, phi_candidate.bb());
    auto insert_it = phi_candidate.bb()->begin();
    insert_it.InsertBefore(std::move(phi_inst));
    modified = true;
  }

  // Scan uses for all inserted Phi instructions. Do this separately from the
  // registration of the Phi instruction itself to avoid trying to analyze uses
  // of Phi instructions that have not been registered yet.
  for (ir::Instruction* phi_inst : generated_phis) {
    pass_->get_def_use_mgr()->AnalyzeInstUse(&*phi_inst);
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "\n\nReplacing the result of load instructions with the "
               "corresponding SSA id\n\n";
#endif

  // Apply replacements from the load replacement table.
  for (auto& repl : load_replacement_) {
    uint32_t load_id = repl.first;
    uint32_t val_id = GetReplacement(repl);
    ir::Instruction* load_inst =
        pass_->context()->get_def_use_mgr()->GetDef(load_id);

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
    std::cerr << "\t"
              << load_inst->PrettyPrint(
                     SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << "  (%" << load_id << " -> %" << val_id << ")\n";
#endif

    // Remove the load instruction and replace all the uses of this load's
    // result with |val_id|.  Kill any names or decorates using the load's
    // result before replacing to prevent incorrect replacement in those
    // instructions.
    pass_->context()->KillNamesAndDecorates(load_id);
    pass_->context()->ReplaceAllUsesWith(load_id, val_id);
    pass_->context()->KillInst(load_inst);
    modified = true;
  }

  return modified;
}

void SSARewriter::ReplacePhiUsersWith(const PhiCandidate& phi_to_remove,
                                      uint32_t repl_id) {
  for (uint32_t user_id : phi_to_remove.users()) {
    PhiCandidate* user_phi = GetPhiCandidate(user_id);
    if (user_phi) {
      // If the user is a Phi candidate, replace all arguments using
      // |phi_to_remove| with |repl_id|.
      for (uint32_t& arg : user_phi->phi_args()) {
        if (arg == phi_to_remove.result_id()) {
          arg = repl_id;
        }
      }
    } else {
      // For regular loads, traverse the |load_replacement_| table looking for
      // instances of |phi_to_remove|.
      for (auto& it : load_replacement_) {
        if (it.second == phi_to_remove.result_id()) {
          it.second = repl_id;
        }
      }
    }
  }
}

uint32_t SSARewriter::TryRemoveTrivialPhi(PhiCandidate* phi_cand) {
  uint32_t same_id = 0;
  for (uint32_t op : phi_cand->phi_args()) {
    if (op == same_id || op == phi_cand->result_id()) {
      // This is a self-reference operand or a reference to the same value ID.
      continue;
    }
    if (same_id != 0) {
      // This Phi candidate merges at least two values.  Therefore, it is not
      // trivial.
      return phi_cand->result_id();
    }
    same_id = op;
  }

  // The previous logic has determined that this Phi candidate |phi_cand| is
  // trivial.  It is essentially the copy operation phi_cand->phi_result =
  // Phi(same, same, same, ...).  Since it is not necessary, we can re-route
  // all the users of |phi_cand->phi_result| to all its users, and remove
  // |phi_cand|.
  //
  // Mark the Phi candidate as trivial, so it won't be generated.
  phi_cand->MarkTrivial();

  if (same_id == 0) {
    // If this Phi is in the start block or unreachable, its result is
    // undefined.
    same_id = pass_->GetUndefVal(phi_cand->var_id());
  }

  // Since |phi_cand| always produces |same_id|, replace all the users of
  // |phi_cand| with |same_id|.
  ReplacePhiUsersWith(*phi_cand, same_id);

  return same_id;
}

void SSARewriter::CompletePhiCandidate(PhiCandidate* phi_cand) {
  assert(phi_cand->IsIncomplete());

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "Incomplete Phi candidate at BB % " << phi_cand->bb()->id()
            << " : " << phi_cand->PrettyPrint(pass_->cfg()) << " -> ";
#endif

  if (phi_cand->phi_args().size() == 0) {
    uint32_t same_id;
    phi_cand->SetPhiArgs(
        *GetPhiOperands(phi_cand->var_id(), phi_cand->bb(), &same_id));
  } else {
    uint32_t arg_ix = 0;
    for (uint32_t pred : pass_->cfg()->preds(phi_cand->bb()->id())) {
      uint32_t* curr_arg_id = &phi_cand->phi_args()[arg_ix++];
      if (*curr_arg_id == 0) {
        // If a predecessor block remains unsealed, it means that it is not
        // reachable from the entry basic block.  Any definition coming through
        // this block is irrelevant, so we use Undef as the operand.
        ir::BasicBlock* pred_bb = pass_->cfg()->block(pred);
        *curr_arg_id = (IsBlockSealed(pred_bb))
                           ? GetReachingDef(phi_cand->var_id(), pred_bb)
                           : pass_->GetUndefVal(phi_cand->var_id());
      }
    }
  }

  // If the Phi became trivial, remove it.
  TryRemoveTrivialPhi(phi_cand);

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << phi_cand->PrettyPrint(pass_->cfg()) << "\n";
#endif
}

void SSARewriter::FinalizePhiCandidates() {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "Finalizing Phi candidates:\n\n";
  PrintPhiCandidates();
  std::cerr << "\n";
#endif

  // Note that completing Phi candidates may generate new Phi candidates.  New
  // Phi candidates at this stage are guaranteed to be complete, because the
  // whole CFG has been scanned already.  To avoid invalidating the iterator, we
  // first collect all the incomplete Phi candidates and then complete them.
  std::vector<PhiCandidate*> phis_to_complete;
  for (auto& it : phi_candidates_) {
    PhiCandidate* phi_cand = &it.second;
    if (phi_cand->IsIncomplete()) phis_to_complete.push_back(phi_cand);
  }

  // Now, complete the collected candidates.
  for (PhiCandidate* phi_cand : phis_to_complete) {
    CompletePhiCandidate(phi_cand);
  }
}

bool SSARewriter::RewriteFunctionIntoSSA(ir::Function* fp) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "Function before SSA rewrite:\n"
            << fp->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << "\n\n\n";
#endif

  // Generate all the SSA replacements and Phi candidates. This will
  // generate incomplete and trivial Phis.
  pass_->cfg()->ForEachBlockInReversePostOrder(
      fp->entry().get(),
      [this](ir::BasicBlock* bb) { GenerateSSAReplacements(bb); });

  // Remove trivial Phis and add arguments to incomplete Phis.
  FinalizePhiCandidates();

  // Finally, apply all the replacements in the IR.
  bool modified = ApplyReplacements();

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "\n\n\nFunction after SSA rewrite:\n"
            << fp->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << "\n";
#endif

  return modified;
}

void SSARewritePass::Initialize(ir::IRContext* c) { InitializeProcessing(c); }

Pass::Status SSARewritePass::Process(ir::IRContext* c) {
  Initialize(c);

  bool modified = false;
  for (auto& fn : *get_module()) {
    modified |= SSARewriter(this).RewriteFunctionIntoSSA(&fn);
  }
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
