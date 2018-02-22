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
  str << "%" << result_id << " = Phi[%" << var_id << "](";
  if (phi_args.size() > 0) {
    uint32_t arg_ix = 0;
    for (uint32_t pred_label : cfg->preds(bb->id())) {
      uint32_t arg_id = phi_args[arg_ix++];
      str << "[%" << arg_id << ", bb(%" << pred_label << ")] ";
      if (arg_id == 0) is_incomplete = true;
    }
  } else {
    is_incomplete = true;
  }
  str << ")" << ((is_trivial) ? " [TRIVIAL PHI]" : "")
      << ((is_incomplete) ? "  [INCOMPLETE]" : "");

  return str.str();
}

SSARewriter::PhiCandidate& SSARewriter::CreatePhiCandidate(uint32_t var_id,
                                                           ir::BasicBlock* bb) {
  auto result = phi_candidates_[bb].emplace(std::make_pair(
      var_id, PhiCandidate(var_id, pass_->context()->TakeNextId(), bb)));

  // We should never try to create more than one Phi candidate for the same
  // |var_id| on |bb|.
  assert(result.second == true);

  PhiCandidate& phi_candidate = result.first->second;

  return phi_candidate;
}

uint32_t SSARewriter::TryRemoveTrivialPhi(PhiCandidate* phi_cand) {
  uint32_t same = 0;
  for (uint32_t op : phi_cand->phi_args) {
    if (op == same || op == phi_cand->result_id) {
      // This is a self-reference operand or a reference to the same value ID.
      continue;
    }
    if (same != 0) {
      // This Phi candidate merges at least two values.  Therefore, it is not
      // trivial.
      return phi_cand->result_id;
    }
    same = op;
  }

  // The previous logic has determined that this Phi candidate |phi_cand| is
  // trivial.  It is essentially the copy operation phi_cand->phi_result =
  // Phi(same, same, same, ...).  Since it is not necessary, we can re-route all
  // the users of |phi_cand->phi_result| to all its users, and remove
  // |phi_cand|.
  if (same == 0) {
    // If this Phi is in the start block or unreachable, its result is
    // undefined.
    same = pass_->GetUndefVal(phi_cand->var_id);
  }

  // Replace all users of |phi_cand->result_id| with |same|.
  for (auto& it : load_replacement_) {
    if (it.second == phi_cand->result_id) {
      it.second = same;
    }
  }

  // Mark the Phi candidate as trivial, so it won't be generated.
  phi_cand->is_trivial = true;

  return same;
}

uint32_t SSARewriter::AddPhiOperands(SSARewriter::PhiCandidate* phi_cand) {
  assert(phi_cand->phi_args.size() == 0);

  for (const auto& pred : pass_->cfg()->preds(phi_cand->bb->id())) {
    ir::BasicBlock* pred_bb = pass_->cfg()->block(pred);
    uint32_t arg_id = 0;

    // Only try to get the reaching definition for this edge if the
    // corresponding block is already sealed. Otherwise, we will be creating
    // unnecessary trivial Phis.  When this happens, this Phi canddiate will
    // remain incomplete and will be completed after the whole CFG has been
    // processed.
    if (IsBlockSealed(pred_bb)) {
      arg_id = ReadVariable(phi_cand->var_id, pred_bb);
    }
    phi_cand->phi_args.push_back(arg_id);
  }

  return TryRemoveTrivialPhi(phi_cand);
}

uint32_t SSARewriter::ReadVariable(uint32_t var_id, ir::BasicBlock* bb) {
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
  return ReadVariableRecursive(var_id, bb);
}

uint32_t SSARewriter::ReadVariableRecursive(uint32_t var_id,
                                            ir::BasicBlock* bb) {
  uint32_t val_id = 0;

  // If |bb| is not yet sealed (i.e., it still has not been processed), create
  // an empty Phi instruction for |var_id|.  This will act as a proxy for when
  // we determine the real reaching definition for |var_id| after the whole CFG
  // has been processed.
  if (!IsBlockSealed(bb)) {
    auto& phi_cand = CreatePhiCandidate(var_id, bb);
    val_id = phi_cand.result_id;
  } else if (pass_->cfg()->preds(bb->id()).size() == 1) {
    // If |bb| has exactly one predecessor, we look for |var_id|'s definition
    // there.
    ir::BasicBlock* pred =
        pass_->cfg()->block(pass_->cfg()->preds(bb->id())[0]);
    val_id = ReadVariable(var_id, pred);
  } else {
    // Otherwise, create a Phi instruction to act as |var_id|'s current
    // definition to break potential cycles.
    PhiCandidate& phi_cand = CreatePhiCandidate(var_id, bb);
    WriteVariable(var_id, bb, phi_cand.result_id);
    val_id = AddPhiOperands(&phi_cand);
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
    std::cout << "\tFound store '%" << var_id << " = %" << val_id << "': "
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
    uint32_t val_id = ReadVariable(var_id, bb);

    // Schedule a replacement for the result of this load instruction with
    // |val_id|. After all the rewriting decisions are made, every use of
    // this load will be replaced with |val_id|.
    const uint32_t load_id = inst->result_id();
    assert(load_replacement_.find(load_id) == load_replacement_.end());
    load_replacement_[load_id] = val_id;

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
    std::cout << "\tFound load: "
              << inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << " (replacement for %" << load_id << " is %" << val_id << ")\n";
#endif
  }
}

void SSARewriter::GenerateSSAReplacements(ir::BasicBlock* bb) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cout << "Generating SSA replacements for block: " << bb->id() << "\n";
  std::cout << bb->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
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
  std::cout << "\n\nPhi candidates:\n";
  for (const auto& phi_it : phi_candidates_) {
    for (const auto& it : phi_it.second) {
      std::cout << "\tBB %" << phi_it.first->id() << ": "
                << it.second.PrettyPrint(pass_->cfg()) << "\n";
    }
  }

  std::cout << "\nLoad replacement table\n";
  for (const auto& it : load_replacement_) {
    std::cout << "\t%" << it.first << " -> %" << it.second << "\n";
  }

  std::cout << "\n\n";
#endif
}

bool SSARewriter::ApplyReplacements() {
  bool modified = false;

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cout << "\n\nApplying replacement decisions to IR\n\n";
  std::cout << "Converting non-trivial Phi candidates into Phi instructions\n";
#endif

  // Add Phi instructions from all Phi candidates.
  for (auto& it : phi_candidates_) {
    ir::BasicBlock* bb = it.first;

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
    std::cout << "\nNon-trivial Phi candidates at block %" << bb->id() << "\n";
#endif
    const auto& phis_at_block = it.second;
    auto insert_it = bb->begin();
    for (auto& phi_it : phis_at_block) {
      const PhiCandidate& phi_candidate = phi_it.second;

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
      std::cout << "Phi candidate: " << phi_candidate.PrettyPrint(pass_->cfg())
                << "\n";
#endif
      if (phi_candidate.is_trivial) {
        continue;
      }

      uint32_t type_id = pass_->GetPointeeTypeId(
          pass_->get_def_use_mgr()->GetDef(phi_candidate.var_id));
      std::vector<ir::Operand> phi_operands;
      uint32_t ix = 0;
      for (uint32_t pred_label : pass_->cfg()->preds(phi_candidate.bb->id())) {
        uint32_t op_val_id = phi_candidate.phi_args[ix++];
        phi_operands.push_back(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {op_val_id}});
        phi_operands.push_back(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {pred_label}});
      }
      std::unique_ptr<ir::Instruction> phi_inst(
          new ir::Instruction(pass_->context(), SpvOpPhi, type_id,
                              phi_candidate.result_id, phi_operands));
      pass_->get_def_use_mgr()->AnalyzeInstDef(&*phi_inst);
      pass_->context()->set_instr_block(&*phi_inst, bb);
      insert_it = insert_it.InsertBefore(std::move(phi_inst));
      ++insert_it;
      modified = true;
    }
  }

  // Scan uses for all inserted Phi instructions. Do this separately from the
  // registration of the Phi instruction itself to avoid trying to analyze uses
  // of Phi instructions that have not been registered yet.
  for (auto& it : phi_candidates_) {
    ir::BasicBlock* bb = it.first;
    bb->ForEachPhiInst([this](ir::Instruction* phi_inst) {
      pass_->get_def_use_mgr()->AnalyzeInstUse(&*phi_inst);
    });
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cout << "\n\nReplacing the result of load instructions with the "
               "corresponding SSA id\n\n";
#endif

  // Replace load operations.
  for (auto& it : load_replacement_) {
    uint32_t load_id = it.first;
    uint32_t val_id = it.second;
    ir::Instruction* load_inst =
        pass_->context()->get_def_use_mgr()->GetDef(load_id);

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
    std::cout << "\t"
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

void SSARewriter::CompletePhiCandidate(PhiCandidate* phi_cand) {
  assert(phi_cand->IsIncomplete());

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cout << "Incomplete Phi candidate at BB %" << phi_cand->bb->id() << ": "
            << phi_cand->PrettyPrint(pass_->cfg()) << " -> ";
#endif

  if (phi_cand->phi_args.size() == 0) {
    AddPhiOperands(phi_cand);
  } else {
    uint32_t ix = 0;
    for (const auto& pred : pass_->cfg()->preds(phi_cand->bb->id())) {
      ir::BasicBlock* pred_bb = pass_->cfg()->block(pred);
      assert(IsBlockSealed(pred_bb) &&
             "All blocks should have been sealed by now.");
      phi_cand->phi_args[ix++] = ReadVariable(phi_cand->var_id, pred_bb);
    }

    // If the Phi became trivial, remove it.
    TryRemoveTrivialPhi(phi_cand);
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cout << phi_cand->PrettyPrint(pass_->cfg()) << "\n";
#endif
}

void SSARewriter::FinalizePhiCandidates() {
  for (auto& it : phi_candidates_) {
    auto& phis_at_block = it.second;
    for (auto& phi_it : phis_at_block) {
      PhiCandidate& phi_cand = phi_it.second;
      if (phi_cand.IsIncomplete()) {
        CompletePhiCandidate(&phi_cand);
      }
    }
  }
}

bool SSARewriter::RewriteFunctionIntoSSA(ir::Function* fp) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cout << "Function before SSA rewrite:\n"
            << fp->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << "\n\n\n";
#endif

  // First, generate all the SSA replacements and Phi candidates. This will
  // generate incomplete and trivial Phis.
  pass_->cfg()->ForEachBlockInReversePostOrder(
      fp->entry().get(),
      [this](ir::BasicBlock* bb) { GenerateSSAReplacements(bb); });

  // Second, remove trivial Phis and add arguments to incomplete Phis.
  FinalizePhiCandidates();

  // Finally, apply all the replacements in the IR.
  bool modified = ApplyReplacements();

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cout << "\n\n\nFunction after SSA rewrite:\n"
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
