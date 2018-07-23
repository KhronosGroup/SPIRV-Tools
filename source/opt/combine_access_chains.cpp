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

#include "combine_access_chains.h"

#include "constants.h"
#include "ir_builder.h"
#include "ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status CombineAccessChains::Process() {
  bool modified = false;

  for (auto& function : *get_module()) {
    modified |= ProcessFunction(function);
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool CombineAccessChains::ProcessFunction(Function& function) {
  bool modified = false;

  cfg()->ForEachBlockInReversePostOrder(
      function.entry().get(), [&modified, this](BasicBlock* block) {
        block->ForEachInst([&modified, this](Instruction* inst) {
          switch (inst->opcode()) {
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
            case SpvOpPtrAccessChain:
              modified |= CombinePtrAccessChain(inst);
              break;
            default:
              break;
          }
        });
      });

  return modified;
}

uint32_t CombineAccessChains::GetConstantValue(
    const analysis::Constant* constant_inst) {
  if (constant_inst->type()->AsInteger()->width() <= 32) {
    if (constant_inst->type()->AsInteger()->IsSigned()) {
      return static_cast<uint32_t>(constant_inst->GetS32());
    } else {
      return constant_inst->GetU32();
    }
  } else if (constant_inst->type()->AsInteger()->width() <= 64) {
    if (constant_inst->type()->AsInteger()->IsSigned()) {
      return static_cast<uint32_t>(constant_inst->GetS64());
    } else {
      return static_cast<uint32_t>(constant_inst->GetU64());
    }
  } else {
    assert(false);
    return 0u;
  }
}

uint32_t CombineAccessChains::GetArrayStride(const Instruction* inst) {
  uint32_t array_stride = 0;
  context()->get_decoration_mgr()->WhileEachDecoration(
      inst->type_id(), SpvDecorationArrayStride,
      [&array_stride](const Instruction& decoration) {
        if (decoration.opcode() == SpvOpDecorate ||
            decoration.opcode() == SpvOpDecorateId) {
          array_stride = decoration.GetSingleWordInOperand(1);
        } else {
          array_stride = decoration.GetSingleWordInOperand(2);
        }
        return false;
      });
  return array_stride;
}

const analysis::Type* CombineAccessChains::GetIndexedType(
    Instruction* inst, std::vector<Operand>* new_operands) {
  // Walk the types till we find the second to last type in the chain.
  // Also copy the operands to construct the new operands.
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();

  Instruction* base_ptr = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
  const analysis::Type* type = type_mgr->GetType(base_ptr->type_id());
  assert(type->AsPointer());
  type = type->AsPointer()->pointee_type();
  std::vector<uint32_t> element_indices;
  new_operands->push_back({SPV_OPERAND_TYPE_ID, {base_ptr->result_id()}});
  uint32_t starting_index = 1;
  if (inst->opcode() == SpvOpPtrAccessChain) {
    // Skip the first index of OpPtrAccessChain as it does not affect type
    // resolution.
    starting_index = 2;
    new_operands->push_back(inst->GetInOperand(1));
  }
  for (uint32_t i = starting_index; i < inst->NumInOperands() - 1; ++i) {
    Instruction* index_inst =
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(i));
    new_operands->push_back(inst->GetInOperand(i));
    const analysis::Constant* index_constant =
        context()->get_constant_mgr()->GetConstantFromInst(index_inst);
    if (index_constant) {
      uint32_t index_value = GetConstantValue(index_constant);
      element_indices.push_back(index_value);
    } else {
      // This index must not matter to resolve the type in valid SPIR-V.
      element_indices.push_back(0);
    }
  }
  type = type_mgr->GetMemberType(type, element_indices);
  return type;
}

bool CombineAccessChains::CombineIndices(const analysis::Type* type,
                                         Instruction* ptr_input,
                                         Instruction* inst,
                                         std::vector<Operand>* new_operands) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::ConstantManager* constant_mgr = context()->get_constant_mgr();

  Instruction* last_index_inst = def_use_mgr->GetDef(
      ptr_input->GetSingleWordInOperand(ptr_input->NumInOperands() - 1));
  const analysis::Constant* last_index_constant =
      constant_mgr->GetConstantFromInst(last_index_inst);

  Instruction* element_inst =
      def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));
  const analysis::Constant* element_constant =
      constant_mgr->GetConstantFromInst(element_inst);

  uint32_t new_value_id = 0;
  if (ptr_input->opcode() == SpvOpPtrAccessChain &&
      ptr_input->NumInOperands() == 2) {
    // This a PtrAccessChain feeding a PtrAccessChain where the first
    // PtrAccessChain only does the element resolution. Previously, we appended
    // the element operand into |new_operands|, so pop it now.
    new_operands->pop_back();
  }

  // Combine the last index of the AccessChain (|ptr_inst|) with the element
  // operand of the PtrAccessChain (|inst|).
  if (last_index_constant && element_constant) {
    // Combine the constants.
    uint32_t new_value = GetConstantValue(last_index_constant) +
                         GetConstantValue(element_constant);
    const analysis::Constant* new_value_constant =
        constant_mgr->GetConstant(last_index_constant->type(), {new_value});
    Instruction* new_value_inst =
        constant_mgr->GetDefiningInstruction(new_value_constant);
    new_value_id = new_value_inst->result_id();
  } else if (!type->AsStruct()) {
    // TODO(alan-baker): handle this unlikely case.
    if (last_index_inst->type_id() != element_inst->type_id()) return false;

    // Generate an addition of the two indices.
    InstructionBuilder builder(
        context(), inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* addition = builder.AddIAdd(last_index_inst->type_id(),
                                            last_index_inst->result_id(),
                                            element_inst->result_id());
    new_value_id = addition->result_id();
  } else {
    // Indexing into structs must be constant, so bail out here.
    return false;
  }
  new_operands->push_back({SPV_OPERAND_TYPE_ID, {new_value_id}});
  return true;
}

bool CombineAccessChains::CombinePtrAccessChain(Instruction* inst) {
  assert((inst->opcode() == SpvOpPtrAccessChain ||
          inst->opcode() == SpvOpAccessChain ||
          inst->opcode() == SpvOpInBoundsAccessChain) &&
         "Wrong opcode. Expected an access chain.");

  Instruction* ptr_input =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  if (ptr_input->opcode() != SpvOpAccessChain &&
      ptr_input->opcode() != SpvOpInBoundsAccessChain &&
      ptr_input->opcode() != SpvOpPtrAccessChain) {
    return false;
  }

  // %gep1 = OpAccessChain %ptr_type1 %base <indices1>
  // %gep2 = OpPtrAccessChain %ptr_type2 %gep1 %element <indices2>
  // We know %element is a constant. We want to combine to combine it with
  // the last index in %gep1 (if its in bounds) and then tack on the rest
  // of indices.

  // TODO(alan-baker): support this properly.
  uint32_t array_stride = GetArrayStride(ptr_input);
  if (array_stride != 0) return false;

  std::vector<Operand> new_operands;
  const analysis::Type* type = GetIndexedType(ptr_input, &new_operands);

  if (inst->opcode() == SpvOpPtrAccessChain) {
    if (!CombineIndices(type, ptr_input, inst, &new_operands)) return false;
  } else if (ptr_input->opcode() != SpvOpPtrAccessChain ||
             ptr_input->NumInOperands() > 2) {
    // OpAccessChain and OpInBoundsAccessChain both just append the indices.
    new_operands.push_back(
        ptr_input->GetInOperand(ptr_input->NumInOperands() - 1));
  }

  // Copy the remaining index operands.
  uint32_t starting_index = (inst->opcode() == SpvOpPtrAccessChain) ? 2 : 1;
  for (uint32_t i = starting_index; i < inst->NumInOperands(); ++i) {
    new_operands.push_back(inst->GetInOperand(i));
  }

  // Update the instruction. The opcode changes to be the same as
  // |ptr_input|'s opcode. The operands are the combined operands constructed
  // above.
  inst->SetOpcode(ptr_input->opcode());
  inst->SetInOperands(std::move(new_operands));
  context()->AnalyzeUses(inst);
  return true;
}

}  // namespace opt
}  // namespace spvtools
