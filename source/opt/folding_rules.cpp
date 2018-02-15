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

FoldingRule IntMultipleBy1() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpIMul && "Wrong opcode.  Should be OpIMul.");
    for (uint32_t i = 0; i < 2; i++) {
      if (constants[i] == nullptr) {
        continue;
      }
      const analysis::IntConstant* int_constant = constants[i]->AsIntConstant();
      if (int_constant && int_constant->GetU32BitValue() == 1) {
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1 - i)}}});
        return true;
      }
    }
    return false;
  };
}

FoldingRule CompositeConstructFeedingExtract() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    // If the input to an OpCompositeExtract is an OpCompositeConstruct,
    // then we can simply use the appropriate element in the construction.
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    analysis::TypeManager* type_mgr = inst->context()->get_type_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    ir::Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeConstruct) {
      return false;
    }

    std::vector<ir::Operand> operands;
    analysis::Type* composite_type = type_mgr->GetType(cinst->type_id());
    if (composite_type->AsVector() == nullptr) {
      // Get the element being extracted from the OpCompositeConstruct
      // Since it is not a vector, it is simple to extract the single element.
      uint32_t element_index = inst->GetSingleWordInOperand(1);
      uint32_t element_id = cinst->GetSingleWordInOperand(element_index);
      operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});

      // Add the remaining indices for extraction.
      for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {inst->GetSingleWordInOperand(i)}});
      }

    } else {
      // With vectors we have to handle the case where it is concatenating
      // vectors.
      assert(inst->NumInOperands() == 2 &&
             "Expecting a vector of scalar values.");

      uint32_t element_index = inst->GetSingleWordInOperand(1);
      for (uint32_t construct_index = 0;
           construct_index < cinst->NumInOperands(); ++construct_index) {
        uint32_t element_id = cinst->GetSingleWordInOperand(construct_index);
        ir::Instruction* element_def = def_use_mgr->GetDef(element_id);
        analysis::Vector* element_type =
            type_mgr->GetType(element_def->type_id())->AsVector();
        if (element_type) {
          uint32_t vector_size = element_type->element_count();
          if (vector_size < element_index) {
            // The element we want comes after this vector.
            element_index -= vector_size;
          } else {
            // We want an element of this vector.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            operands.push_back(
                {SPV_OPERAND_TYPE_LITERAL_INTEGER, {element_index}});
            break;
          }
        } else {
          if (element_index == 0) {
            // This is a scalar, and we this is the element we are extracting.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            break;
          } else {
            // Skip over this scalar value.
            --element_index;
          }
        }
      }
    }

    // If there were no extra indices, then we have the final object.  No need
    // to extract even more.
    if (operands.size() == 1) {
      inst->SetOpcode(SpvOpCopyObject);
    }

    inst->SetInOperands(std::move(operands));
    return true;
  };
}

FoldingRule CompositeExtractFeedingConstruct() {
  // If the OpCompositeConstruct is simply putting back together elements that
  // where extracted from the same souce, we can simlpy reuse the source.
  //
  // This is a common code pattern because of the way that scalar replacement
  // works.
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeConstruct &&
           "Wrong opcode.  Should be OpCompositeConstruct.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    uint32_t original_id = 0;

    // Check each element to make sure they are:
    // - extractions
    // - extracting the same position they are inserting
    // - all extract from the same id.
    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      uint32_t element_id = inst->GetSingleWordInOperand(i);
      ir::Instruction* element_inst = def_use_mgr->GetDef(element_id);

      if (element_inst->opcode() != SpvOpCompositeExtract) {
        return false;
      }

      if (element_inst->NumInOperands() != 2) {
        return false;
      }

      if (element_inst->GetSingleWordInOperand(1) != i) {
        return false;
      }

      if (i == 0) {
        original_id =
            element_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
      } else if (original_id != element_inst->GetSingleWordInOperand(
                                    kExtractCompositeIdInIdx)) {
        return false;
      }
    }

    // The last check it to see that the object being extracted from is the
    // correct type.
    ir::Instruction* original_inst = def_use_mgr->GetDef(original_id);
    if (original_inst->type_id() != inst->type_id()) {
      return false;
    }

    // Simplify by using the original object.
    inst->SetOpcode(SpvOpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {original_id}}});
    return true;
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

FoldingRule RedundantPhi() {
  // An OpPhi instruction where all values are the same or the result of the phi
  // itself, can be replaced by the value itself.
  return
      [](ir::Instruction* inst, const std::vector<const analysis::Constant*>&) {
        assert(inst->opcode() == SpvOpPhi && "Wrong opcode.  Should be OpPhi.");

        ir::IRContext* context = inst->context();
        analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

        uint32_t incoming_value = 0;

        for (uint32_t i = 0; i < inst->NumInOperands(); i += 2) {
          uint32_t op_id = inst->GetSingleWordInOperand(i);
          if (op_id == inst->result_id()) {
            continue;
          }

          ir::Instruction* op_inst = def_use_mgr->GetDef(op_id);
          if (op_inst->opcode() == SpvOpUndef) {
            // TODO: We should be able to still use op_id if we know that
            // the definition of op_id dominates |inst|.
            return false;
          }

          if (incoming_value == 0) {
            incoming_value = op_id;
          } else if (op_id != incoming_value) {
            // Found two possible value.  Can't simplify.
            return false;
          }
        }

        if (incoming_value == 0) {
          // Code looks invalid.  Don't do anything.
          return false;
        }

        // We have a single incoming value.  Simplify using that value.
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {incoming_value}}});
        return true;
      };
}

FoldingRule RedundantSelect() {
  // An OpSelect instruction where both values are the same or the condition is
  // constant can be replaced by one of the values
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpSelect &&
           "Wrong opcode.  Should be OpSelect.");
    assert(inst->NumInOperands() == 3);
    assert(constants.size() == 3);

    const analysis::BoolConstant* bc =
        constants[0] ? constants[0]->AsBoolConstant() : nullptr;
    uint32_t true_id = inst->GetSingleWordInOperand(1);
    uint32_t false_id = inst->GetSingleWordInOperand(2);

    if (bc) {
      // Select condition is constant, result is known
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {bc->value() ? true_id : false_id}}});
      return true;
    } else if (true_id == false_id) {
      // Both results are the same, condition doesn't matter
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {true_id}}});
      return true;
    } else {
      return false;
    }
  };
}
}  // namespace

spvtools::opt::FoldingRules::FoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[SpvOpCompositeConstruct].push_back(CompositeExtractFeedingConstruct());

  rules_[SpvOpCompositeExtract].push_back(InsertFeedingExtract());
  rules_[SpvOpCompositeExtract].push_back(CompositeConstructFeedingExtract());

  rules_[SpvOpIMul].push_back(IntMultipleBy1());

  rules_[SpvOpPhi].push_back(RedundantPhi());

  rules_[SpvOpSelect].push_back(RedundantSelect());
}
}  // namespace opt
}  // namespace spvtools
