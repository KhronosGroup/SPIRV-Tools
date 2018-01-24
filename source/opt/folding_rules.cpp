// Copyright (c) 2018 Google LLC
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

#include "folding_rules.h"

namespace spvtools {
namespace opt {

namespace {
const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;

FoldingRule IntMultiplyBy1() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpIMul && "Wrong opcode.  Should be OpIMul.");
    for (uint32_t i = 0; i < 2; i++) {
      if (constants[i] == nullptr) {
        continue;
      }
      const analysis::IntConstant* int_constant = constants[i]->AsIntConstant();
      if (int_constant->GetU32BitValue() == 1) {
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1 - i)}}});
        return true;
      }
    }
    return false;
  };
}

FoldingRule InsertFeedingExtract() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    ir::Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeInsert) {
      return false;
    }

    // Find the first position where the list of insert and extract indicies
    // differ, if at all.
    uint32_t i;
    for (i = 1; i < inst->NumInOperands(); ++i) {
      if (i + 1 >= cinst->NumInOperands()) {
        break;
      }

      if (inst->GetSingleWordInOperand(i) !=
          cinst->GetSingleWordInOperand(i + 1)) {
        break;
      }
    }

    // We are extracting the element that was inserted.
    if (i == inst->NumInOperands() && i + 1 == cinst->NumInOperands()) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID,
            {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}}});
      return true;
    }

    // Extracting the value that was inserted along with values for the base
    // composite.  Cannot do anything.
    if (i + 1 == cinst->NumInOperands()) {
      return false;
    }

    // Extracting an element of the value that was inserted.  Extract from
    // that value directly.
    if (i == inst->NumInOperands()) {
      std::vector<ir::Operand> operands;
      operands.push_back(
          {SPV_OPERAND_TYPE_ID,
           {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}});
      for (i = i + 1; i < cinst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {cinst->GetSingleWordInOperand(i)}});
      }
      inst->SetInOperands(std::move(operands));
      return true;
    }

    // Extracting a value that is disjoint from the element being inserted.
    // Rewrite the extract to use the composite input to the insert.
    std::vector<ir::Operand> operands;
    operands.push_back(
        {SPV_OPERAND_TYPE_ID,
         {cinst->GetSingleWordInOperand(kInsertCompositeIdInIdx)}});
    for (i = 1; i < inst->NumInOperands(); ++i) {
      operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {inst->GetSingleWordInOperand(i)}});
    }
    inst->SetInOperands(std::move(operands));
    return true;
  };
}
}  // namespace

spvtools::opt::FoldingRules::FoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules[SpvOpIMul].push_back(IntMultiplyBy1());

  rules[SpvOpCompositeExtract].push_back(InsertFeedingExtract());
}
}  // namespace opt
}  // namespace spvtools
