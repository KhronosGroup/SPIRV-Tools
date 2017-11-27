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
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "def_use_manager.h"
#include "ir_context.h"
#include "log.h"
#include "reflect.h"

namespace {
// Count the number of trailing zeros in the binary representation of
// |constVal|.
uint32_t CountTrailingZeros(uint32_t constVal) {
  // Faster if we use the hardware count trailing zeros instruction.
  // If not available, we could create a table.
  uint32_t shiftAmount = 0;
  while ((constVal & 1) == 0) {
    ++shiftAmount;
    constVal = (constVal >> 1);
  }
  return shiftAmount;
}

// Return true if |val| is a power of 2.
bool IsPowerOf2(uint32_t val) {
  // The idea is that the & will clear out the least
  // significant 1 bit.  If it is a power of 2, then
  // there is exactly 1 bit set, and the value becomes 0.
  if (val == 0) return false;
  return ((val - 1) & val) == 0;
}

}  // namespace

namespace spvtools {
namespace opt {

Pass::Status StrengthReductionPass::Process(ir::IRContext* c) {
  InitializeProcessing(c);

  // Initialize the member variables on a per module basis.
  bool modified = false;
  int32_type_id_ = 0;
  uint32_type_id_ = 0;
  std::memset(constant_ids_, 0, sizeof(constant_ids_));

  FindIntTypesAndConstants();
  modified = ScanFunctions();
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool StrengthReductionPass::ReplaceMultiplyByPowerOf2(
    ir::BasicBlock::iterator* instPtr) {
  ir::BasicBlock::iterator& inst = *instPtr;
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
    ir::Instruction* opInst = get_def_use_mgr()->GetDef(opId);
    if (opInst->opcode() == SpvOp::SpvOpConstant) {
      // We found a constant operand.
      uint32_t constVal = opInst->GetSingleWordOperand(2);

      if (IsPowerOf2(constVal)) {
        modified = true;
        uint32_t shiftAmount = CountTrailingZeros(constVal);
        uint32_t shiftConstResultId = GetConstantId(shiftAmount);

        // Create the new instruction.
        uint32_t newResultId = TakeNextId();
        std::vector<ir::Operand> newOperands;
        newOperands.push_back(inst->GetInOperand(1 - i));
        ir::Operand shiftOperand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                                 {shiftConstResultId});
        newOperands.push_back(shiftOperand);
        std::unique_ptr<ir::Instruction> newInstruction(
            new ir::Instruction(context(), SpvOp::SpvOpShiftLeftLogical,
                                inst->type_id(), newResultId, newOperands));

        // Insert the new instruction and update the data structures.
        inst = inst.InsertBefore(std::move(newInstruction));
        get_def_use_mgr()->AnalyzeInstDefUse(&*inst);
        ++inst;
        context()->ReplaceAllUsesWith(inst->result_id(), newResultId);

        // Remove the old instruction.
        context()->KillInst(&*inst);

        // We do not want to replace the instruction twice if both operands
        // are constants that are a power of 2.  So we break here.
        break;
      }
    }
  }

  return modified;
}

void StrengthReductionPass::FindIntTypesAndConstants() {
  for (auto iter = get_module()->types_values_begin();
       iter != get_module()->types_values_end(); ++iter) {
    switch (iter->opcode()) {
      case SpvOp::SpvOpTypeInt:
        if (iter->GetSingleWordOperand(1) == 32) {
          if (iter->GetSingleWordOperand(2) == 1) {
            int32_type_id_ = iter->result_id();
          } else {
            uint32_type_id_ = iter->result_id();
          }
        }
        break;
      case SpvOp::SpvOpConstant:
        if (iter->type_id() == uint32_type_id_) {
          uint32_t value = iter->GetSingleWordOperand(2);
          if (value <= 32) constant_ids_[value] = iter->result_id();
        }
        break;
      default:
        break;
    }
  }
}

uint32_t StrengthReductionPass::GetConstantId(uint32_t val) {
  assert(val <= 32 &&
         "This function does not handle constants larger than 32.");

  if (constant_ids_[val] == 0) {
    if (uint32_type_id_ == 0) {
      uint32_type_id_ = CreateUint32Type();
    }

    // Construct the constant.
    uint32_t resultId = TakeNextId();
    ir::Operand constant(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                         {val});
    std::unique_ptr<ir::Instruction> newConstant(
        new ir::Instruction(context(), SpvOp::SpvOpConstant, uint32_type_id_,
                            resultId, {constant}));
    get_module()->AddGlobalValue(std::move(newConstant));

    // Notify the DefUseManager about this constant.
    auto constantIter = --get_module()->types_values_end();
    get_def_use_mgr()->AnalyzeInstDef(&*constantIter);

    // Store the result id for next time.
    constant_ids_[val] = resultId;
  }

  return constant_ids_[val];
}

bool StrengthReductionPass::ScanFunctions() {
  // I did not use |ForEachInst| in the module because the function that acts on
  // the instruction gets a pointer to the instruction.  We cannot use that to
  // insert a new instruction.  I want an iterator.
  bool modified = false;
  for (auto& func : *get_module()) {
    for (auto& bb : func) {
      for (auto inst = bb.begin(); inst != bb.end(); ++inst) {
        switch (inst->opcode()) {
          case SpvOp::SpvOpIMul:
            if (ReplaceMultiplyByPowerOf2(&inst)) modified = true;
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
  uint32_t type_id = TakeNextId();
  ir::Operand widthOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                           {32});
  ir::Operand signOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {0});
  std::unique_ptr<ir::Instruction> newType(new ir::Instruction(
      context(), SpvOp::SpvOpTypeInt, type_id, 0, {widthOperand, signOperand}));
  context()->AddType(std::move(newType));
  return type_id;
}

}  // namespace opt
}  // namespace spvtools
