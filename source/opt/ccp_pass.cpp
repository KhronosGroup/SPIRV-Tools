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

// This file implements conditional constant propagation as described in
//
//      Constant propagation with conditional branches,
//      Wegman and Zadeck, ACM TOPLAS 13(2):181-210.
#include "ccp_pass.h"
#include "fold.h"
#include "function.h"
#include "module.h"
#include "propagator.h"

#include <algorithm>

namespace spvtools {
namespace opt {

SSAPropagator::PropStatus CCPPass::VisitPhi(ir::Instruction* phi) {
  uint32_t meet_val_id = 0;

  // Implement the lattice meet operation. The result of this Phi instruction is
  // interesting only if the meet operation over arguments coming through
  // executable edges yields the same constant value.
  for (uint32_t i = 2; i < phi->NumOperands(); i += 2) {
    if (!propagator_->IsPhiArgExecutable(phi, i)) {
      // Ignore arguments coming through non-executable edges.
      continue;
    }
    uint32_t phi_arg_id = phi->GetSingleWordOperand(i);
    auto it = values_.find(phi_arg_id);
    if (it != values_.end()) {
      // We found an argument with a constant value.  Apply the meet operation
      // with the previous arguments.
      if (meet_val_id == 0) {
        // This is the first argument we find.  Initialize the result to its
        // constant value id.
        meet_val_id = it->second;
      } else if (it->second == meet_val_id) {
        // The argument is the same constant value already computed. Continue
        // looking.
        continue;
      } else {
        // We found another constant value, but it is different from the
        // previous computed meet value.  This Phi will never be constant.
        return SSAPropagator::kVarying;
      }
    } else {
      // If any argument is not a constant, the Phi produces nothing
      // interesting for now. The propagator will callback again, if needed.
      return SSAPropagator::kNotInteresting;
    }
  }

  // If there are no incoming executable edges, the meet ID will still be 0. In
  // that case, return not interesting to evaluate the Phi node again.
  if (meet_val_id == 0) {
    return SSAPropagator::kNotInteresting;
  }

  // All the operands have the same constant value represented by |meet_val_id|.
  // Set the Phi's result to that value and declare it interesting.
  values_[phi->result_id()] = meet_val_id;
  return SSAPropagator::kInteresting;
}

SSAPropagator::PropStatus CCPPass::VisitAssignment(ir::Instruction* instr) {
  assert(instr->result_id() != 0 &&
         "Expecting an instruction that produces a result");

  // If this is a copy operation, and the RHS is a known constant, assign its
  // value to the LHS.
  if (instr->opcode() == SpvOpCopyObject) {
    uint32_t rhs_id = instr->GetSingleWordInOperand(0);
    auto it = values_.find(rhs_id);
    if (it != values_.end()) {
      values_[instr->result_id()] = it->second;
      return SSAPropagator::kInteresting;
    }
    return SSAPropagator::kNotInteresting;
  }

  // Instructions with a RHS that cannot produce a constant are always varying.
  if (!instr->IsFoldable()) {
    return SSAPropagator::kVarying;
  }

  // Otherwise, see if the RHS of the assignment folds into a constant value.
  std::vector<uint32_t> cst_val_ids;
  bool missing_constants = false;
  instr->ForEachInId([this, &cst_val_ids, &missing_constants](uint32_t* op_id) {
    auto it = values_.find(*op_id);
    if (it == values_.end()) {
      missing_constants = true;
      return;
    }
    cst_val_ids.push_back(it->second);
  });

  // If we did not find a constant value for every operand in the instruction,
  // do not bother folding it.  Indicate that this instruction does not produce
  // an interesting value for now.
  if (missing_constants) {
    return SSAPropagator::kNotInteresting;
  }

  auto constants = const_mgr_->GetConstantsFromIds(cst_val_ids);
  assert(constants.size() != 0 && "Found undeclared constants");

  // If any of the constants are not supported by the folder, we will not be
  // able to produce a constant out of this instruction.  Consider it varying
  // in that case.
  if (!std::all_of(constants.begin(), constants.end(),
                   [](const analysis::Constant* cst) {
                     return IsFoldableConstant(cst);
                   })) {
    return SSAPropagator::kVarying;
  }

  // Otherwise, fold the instruction with all the operands to produce a new
  // constant.
  uint32_t result_val = FoldScalars(instr->opcode(), constants);
  const analysis::Constant* result_const =
      const_mgr_->GetConstant(const_mgr_->GetType(instr), {result_val});
  ir::Instruction* const_decl =
      const_mgr_->GetDefiningInstruction(result_const);
  values_[instr->result_id()] = const_decl->result_id();
  return SSAPropagator::kInteresting;
}

SSAPropagator::PropStatus CCPPass::VisitBranch(ir::Instruction* instr,
                                               ir::BasicBlock** dest_bb) const {
  assert(instr->IsBranch() && "Expected a branch instruction.");

  *dest_bb = nullptr;
  uint32_t dest_label = 0;
  if (instr->opcode() == SpvOpBranch) {
    // An unconditional jump always goes to its unique destination.
    dest_label = instr->GetSingleWordInOperand(0);
  } else if (instr->opcode() == SpvOpBranchConditional) {
    // For a conditional branch, determine whether the predicate selector has a
    // known value in |values_|.  If it does, set the destination block
    // according to the selector's boolean value.
    uint32_t pred_id = instr->GetSingleWordOperand(0);
    auto it = values_.find(pred_id);
    if (it == values_.end()) {
      // The predicate has an unknown value, either branch could be taken.
      return SSAPropagator::kVarying;
    }

    // Get the constant value for the predicate selector from the value table.
    // Use it to decide which branch will be taken.
    uint32_t pred_val_id = it->second;
    const analysis::Constant* c = const_mgr_->FindDeclaredConstant(pred_val_id);
    assert(c && "Expected to find a constant declaration for a known value.");
    const analysis::BoolConstant* val = c->AsBoolConstant();
    dest_label = val->value() ? instr->GetSingleWordOperand(1)
                              : instr->GetSingleWordOperand(2);
  } else {
    // For an OpSwitch, extract the value taken by the switch selector and check
    // which of the target literals it matches.  The branch associated with that
    // literal is the taken branch.
    assert(instr->opcode() == SpvOpSwitch);
    if (instr->GetOperand(0).words.size() != 1) {
      // If the selector is wider than 32-bits, return varying. TODO(dnovillo):
      // Add support for wider constants.
      return SSAPropagator::kVarying;
    }
    uint32_t select_id = instr->GetSingleWordOperand(0);
    auto it = values_.find(select_id);
    if (it == values_.end()) {
      // The selector has an unknown value, any of the branches could be taken.
      return SSAPropagator::kVarying;
    }

    // Get the constant value for the selector from the value table. Use it to
    // decide which branch will be taken.
    uint32_t select_val_id = it->second;
    const analysis::Constant* c =
        const_mgr_->FindDeclaredConstant(select_val_id);
    assert(c && "Expected to find a constant declaration for a known value.");
    const analysis::IntConstant* val = c->AsIntConstant();

    // Start assuming that the selector will take the default value;
    dest_label = instr->GetSingleWordOperand(1);
    for (uint32_t i = 2; i < instr->NumOperands(); i += 2) {
      if (val->words()[0] == instr->GetSingleWordOperand(i)) {
        dest_label = instr->GetSingleWordOperand(i + 1);
        break;
      }
    }
  }

  assert(dest_label && "Destination label should be set at this point.");
  *dest_bb = context()->cfg()->block(dest_label);
  return SSAPropagator::kInteresting;
}

SSAPropagator::PropStatus CCPPass::VisitInstruction(ir::Instruction* instr,
                                                    ir::BasicBlock** dest_bb) {
  *dest_bb = nullptr;
  if (instr->opcode() == SpvOpPhi) {
    return VisitPhi(instr);
  } else if (instr->IsBranch()) {
    return VisitBranch(instr, dest_bb);
  } else if (instr->result_id()) {
    return VisitAssignment(instr);
  }
  return SSAPropagator::kVarying;
}

bool CCPPass::ReplaceValues() {
  bool retval = false;
  for (const auto& it : values_) {
    uint32_t id = it.first;
    uint32_t cst_id = it.second;
    if (id != cst_id) {
      retval |= context()->ReplaceAllUsesWith(id, cst_id);
    }
  }
  return retval;
}

bool CCPPass::PropagateConstants(ir::Function* fp) {
  const auto visit_fn = [this](ir::Instruction* instr,
                               ir::BasicBlock** dest_bb) {
    return VisitInstruction(instr, dest_bb);
  };

  propagator_ =
      std::unique_ptr<SSAPropagator>(new SSAPropagator(context(), visit_fn));
  if (propagator_->Run(fp)) {
    return ReplaceValues();
  }

  return false;
}

void CCPPass::Initialize(ir::IRContext* c) {
  InitializeProcessing(c);

  const_mgr_ = context()->get_constant_mgr();

  // Populate the constant table with values from constant declarations in the
  // module.  The values of each OpConstant declaration is the identity
  // assignment (i.e., each constant is its own value).
  for (const auto& inst : c->module()->GetConstants()) {
    values_[inst->result_id()] = inst->result_id();
    if (!const_mgr_->MapInst(inst)) {
      assert(false &&
             "Could not map a new constant value to its defining instruction");
    }
  }
}

Pass::Status CCPPass::Process(ir::IRContext* c) {
  Initialize(c);

  // Process all entry point functions.
  ProcessFunction pfn = [this](ir::Function* fp) {
    return PropagateConstants(fp);
  };
  bool modified = ProcessReachableCallTree(pfn, context());
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
