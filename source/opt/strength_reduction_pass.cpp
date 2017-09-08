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

#include "strength_reduction_pass.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

#include "def_use_manager.h"
#include "log.h"
#include "reflect.h"

namespace {
// Utility function to count the number of trailing zeros in constVal.
uint32_t GetShiftAmount(uint32_t constVal) {
  // Faster if we use the hardware count trailing zeros instruction.
  // If not available, we could create a table.
  uint32_t shiftAmount = 0;
  while (constVal != 1) {
    ++shiftAmount;
    constVal = (constVal >> 1);
  }
  return shiftAmount;
}

// Quick check if |val| is a power of 2 or not.
// The idea is that the & will clear out the least
// significant 1 bit.  If it is a power of 2, then
// there is exactly 1 bit set, and the value becomes 0.
bool IsPowerOf2(uint32_t val) {
  if (val == 0) return false;
  return ((val - 1) & val) == 0;
}

}  // namespace

namespace spvtools {
namespace opt {

Pass::Status StrengthReductionPass::Process(ir::Module* module) {
  // Initialize the member variables on a per module basis.
  bool modified = false;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  int32_type_id_ = 0;
  uint32_type_id_ = 0;
  next_id_ = module->IdBound();
  module_ = module;
  must_create_type_ = false;

  FindIntTypes();
  modified = ScanFunctions();
  // Have to reset the id bound.
  module->SetIdBound(next_id_);
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool StrengthReductionPass::ReplaceMultiplyByPowerOf2(
    ir::BasicBlock::iterator& inst) {
  assert(inst->opcode() == SpvOp::SpvOpIMul &&
         "Only works for multiplication of integers.");
  bool modified = false;

  // Currently only works on 32-bit integers.
  if (inst->type_id() != int32_type_id_ && inst->type_id() != uint32_type_id_) {
    return modified;
  }

  // Check the operands for a constant that is a power of 2.
  for (int i = 0; i < 2; i++) {
    uint32_t opId = inst->GetSingleWordInOperand(i);
    ir::Instruction* opInst = def_use_mgr_->GetDef(opId);
    if (opInst->opcode() == SpvOp::SpvOpConstant) {
      // We found a constant operand.
      uint32_t constVal = opInst->GetSingleWordOperand(2);

      if (IsPowerOf2(constVal)) {
        uint32_t shiftAmount = GetShiftAmount(constVal);
        modified = true;

        if (uint32_type_id_ == 0) {
          uint32_type_id_ = CreateUint32Type();
        }

        // Construct the constant.  Currently not worried
        // about duplicate constants.
        uint32_t shiftConstResultId = next_id_++;
        ir::Operand shiftConstant(
            spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
            {shiftAmount});
        std::unique_ptr<ir::Instruction> newConstant(
            new ir::Instruction(SpvOp::SpvOpConstant, uint32_type_id_,
                                shiftConstResultId, {shiftConstant}));
        module_->AddGlobalValue(std::move(newConstant));

        // Create the new instruction.
        uint32_t newResultId = next_id_++;
        std::vector<ir::Operand> newOperands;
        newOperands.push_back(inst->GetInOperand(1 - i));
        ir::Operand shiftOperand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                                 {shiftConstResultId});
        newOperands.push_back(shiftOperand);
        std::unique_ptr<ir::Instruction> newInstruction(
            new ir::Instruction(SpvOp::SpvOpShiftLeftLogical, inst->type_id(),
                                newResultId, newOperands));

        // Insert the new instruction and update the data structures.
        def_use_mgr_->AnalyzeInstDefUse(&*newInstruction);
        inst = inst.InsertBefore(std::move(newInstruction));
        ++inst;
        def_use_mgr_->ReplaceAllUsesWith(inst->result_id(), newResultId);

        // Remove the old instruction.
        def_use_mgr_->KillInst(&*inst);

        // We do not want to replace the instruction twice if both operands
        // are constants that are a power of 2.  So we break here.
        break;
      }
    }
  }

  return modified;
}

void StrengthReductionPass::FindIntTypes() {
  for (auto typeIter = module_->types_values_begin();
       typeIter != module_->types_values_end(); ++typeIter) {
    switch (typeIter->opcode()) {
      case SpvOp::SpvOpTypeInt:
        if (typeIter->GetSingleWordOperand(1) == 32) {
          if (typeIter->GetSingleWordOperand(2) == 1) {
            int32_type_id_ = typeIter->result_id();
          } else {
            uint32_type_id_ = typeIter->result_id();
          }
        }
        break;
      default:
        break;
    }
  }
}

bool StrengthReductionPass::ScanFunctions() {
  // I did not use |ForEachInst| in the module because the function that acts on
  // the instruction gets a pointer to the instruction.  We cannot use that to
  // insert a new instruction.  I want an iterator.
  bool modified = false;
  for (auto& func : *module_) {
    for (auto& bb : func) {
      for (auto inst = bb.begin(); inst != bb.end(); ++inst) {
        switch (inst->opcode()) {
          case SpvOp::SpvOpIMul:
            if (ReplaceMultiplyByPowerOf2(inst)) modified = true;
            break;
          default:
            break;
        }
      }
    }
  }
  return modified;
}

uint32_t StrengthReductionPass::CreateUint32Type() {
  uint32_t type_id = next_id_++;
  ir::Operand widthOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                           {32});
  ir::Operand signOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {0});
  std::unique_ptr<ir::Instruction> newType(new ir::Instruction(
      SpvOp::SpvOpTypeInt, type_id, 0, {widthOperand, signOperand}));
  module_->AddType(std::move(newType));
  return type_id;
}

}  // namespace opt
}  // namespace spvtools
