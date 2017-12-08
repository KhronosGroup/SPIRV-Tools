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

#include "constants.h"
#include "ir_context.h"

#include <unordered_map>
#include <vector>

namespace spvtools {
namespace opt {
namespace analysis {

analysis::Type* ConstantManager::GetType(const ir::Instruction* inst) const {
  return context()->get_type_mgr()->GetType(inst->type_id());
}

uint32_t ConstantManager::FindRecordedConstant(
    const analysis::Constant* c) const {
  auto iter = const_val_to_id_.find(c);
  if (iter == const_val_to_id_.end()) {
    return 0;
  } else {
    return iter->second;
  }
}

std::vector<const analysis::Constant*> ConstantManager::GetConstantsFromIds(
    const std::vector<uint32_t>& ids) const {
  std::vector<const analysis::Constant*> constants;
  for (uint32_t id : ids) {
    if (analysis::Constant* c = FindRecordedConstant(id)) {
      constants.push_back(c);
    } else {
      return {};
    }
  }
  return constants;
}

ir::Instruction* ConstantManager::BuildInstructionAndAddToModule(
    std::unique_ptr<analysis::Constant> c, ir::Module::inst_iterator* pos) {
  analysis::Constant* new_const = c.get();
  uint32_t new_id = context()->TakeNextId();
  const_val_to_id_[new_const] = new_id;
  id_to_const_val_[new_id] = std::move(c);
  auto new_inst = CreateInstruction(new_id, new_const);
  if (!new_inst) return nullptr;
  auto* new_inst_ptr = new_inst.get();
  *pos = pos->InsertBefore(std::move(new_inst));
  ++(*pos);
  context()->get_def_use_mgr()->AnalyzeInstDefUse(new_inst_ptr);
  return new_inst_ptr;
}

analysis::Constant* ConstantManager::FindRecordedConstant(uint32_t id) const {
  auto iter = id_to_const_val_.find(id);
  if (iter == id_to_const_val_.end()) {
    return nullptr;
  } else {
    return iter->second.get();
  }
}

std::unique_ptr<analysis::Constant> ConstantManager::CreateConstant(
    const analysis::Type* type,
    const std::vector<uint32_t>& literal_words_or_ids) const {
  std::unique_ptr<analysis::Constant> new_const;
  if (literal_words_or_ids.size() == 0) {
    // Constant declared with OpConstantNull
    return MakeUnique<analysis::NullConstant>(type);
  } else if (auto* bt = type->AsBool()) {
    assert(literal_words_or_ids.size() == 1 &&
           "Bool constant should be declared with one operand");
    return MakeUnique<analysis::BoolConstant>(bt, literal_words_or_ids.front());
  } else if (auto* it = type->AsInteger()) {
    return MakeUnique<analysis::IntConstant>(it, literal_words_or_ids);
  } else if (auto* ft = type->AsFloat()) {
    return MakeUnique<analysis::FloatConstant>(ft, literal_words_or_ids);
  } else if (auto* vt = type->AsVector()) {
    auto components = GetConstantsFromIds(literal_words_or_ids);
    if (components.empty()) return nullptr;
    // All components of VectorConstant must be of type Bool, Integer or Float.
    if (!std::all_of(components.begin(), components.end(),
                     [](const analysis::Constant* c) {
                       if (c->type()->AsBool() || c->type()->AsInteger() ||
                           c->type()->AsFloat()) {
                         return true;
                       } else {
                         return false;
                       }
                     }))
      return nullptr;
    // All components of VectorConstant must be in the same type.
    const auto* component_type = components.front()->type();
    if (!std::all_of(components.begin(), components.end(),
                     [&component_type](const analysis::Constant* c) {
                       if (c->type() == component_type) return true;
                       return false;
                     }))
      return nullptr;
    return MakeUnique<analysis::VectorConstant>(vt, components);
  } else if (auto* st = type->AsStruct()) {
    auto components = GetConstantsFromIds(literal_words_or_ids);
    if (components.empty()) return nullptr;
    return MakeUnique<analysis::StructConstant>(st, components);
  } else if (auto* at = type->AsArray()) {
    auto components = GetConstantsFromIds(literal_words_or_ids);
    if (components.empty()) return nullptr;
    return MakeUnique<analysis::ArrayConstant>(at, components);
  } else {
    return nullptr;
  }
}

std::unique_ptr<analysis::Constant> ConstantManager::CreateConstantFromInst(
    ir::Instruction* inst) const {
  std::vector<uint32_t> literal_words_or_ids;
  std::unique_ptr<analysis::Constant> new_const;

  // Collect the constant defining literals or component ids.
  for (uint32_t i = 0; i < inst->NumInOperands(); i++) {
    literal_words_or_ids.insert(literal_words_or_ids.end(),
                                inst->GetInOperand(i).words.begin(),
                                inst->GetInOperand(i).words.end());
  }

  switch (inst->opcode()) {
    // OpConstant{True|Flase} have the value embedded in the opcode. So they
    // are not handled by the for-loop above. Here we add the value explicitly.
    case SpvOp::SpvOpConstantTrue:
      literal_words_or_ids.push_back(true);
      break;
    case SpvOp::SpvOpConstantFalse:
      literal_words_or_ids.push_back(false);
      break;
    case SpvOp::SpvOpConstantNull:
    case SpvOp::SpvOpConstant:
    case SpvOp::SpvOpConstantComposite:
    case SpvOp::SpvOpSpecConstantComposite:
      break;
    default:
      return nullptr;
  }
  return CreateConstant(GetType(inst), literal_words_or_ids);
}

std::unique_ptr<ir::Instruction> ConstantManager::CreateInstruction(
    uint32_t id, analysis::Constant* c) const {
  if (c->AsNullConstant()) {
    return MakeUnique<ir::Instruction>(
        context(), SpvOp::SpvOpConstantNull,
        context()->get_type_mgr()->GetId(c->type()), id,
        std::initializer_list<ir::Operand>{});
  } else if (analysis::BoolConstant* bc = c->AsBoolConstant()) {
    return MakeUnique<ir::Instruction>(
        context(),
        bc->value() ? SpvOp::SpvOpConstantTrue : SpvOp::SpvOpConstantFalse,
        context()->get_type_mgr()->GetId(c->type()), id,
        std::initializer_list<ir::Operand>{});
  } else if (analysis::IntConstant* ic = c->AsIntConstant()) {
    return MakeUnique<ir::Instruction>(
        context(), SpvOp::SpvOpConstant,
        context()->get_type_mgr()->GetId(c->type()), id,
        std::initializer_list<ir::Operand>{ir::Operand(
            spv_operand_type_t::SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
            ic->words())});
  } else if (analysis::FloatConstant* fc = c->AsFloatConstant()) {
    return MakeUnique<ir::Instruction>(
        context(), SpvOp::SpvOpConstant,
        context()->get_type_mgr()->GetId(c->type()), id,
        std::initializer_list<ir::Operand>{ir::Operand(
            spv_operand_type_t::SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
            fc->words())});
  } else if (analysis::CompositeConstant* cc = c->AsCompositeConstant()) {
    return CreateCompositeInstruction(id, cc);
  } else {
    return nullptr;
  }
}

std::unique_ptr<ir::Instruction> ConstantManager::CreateCompositeInstruction(
    uint32_t result_id, analysis::CompositeConstant* cc) const {
  std::vector<ir::Operand> operands;
  for (const analysis::Constant* component_const : cc->GetComponents()) {
    uint32_t id = FindRecordedConstant(component_const);
    if (id == 0) {
      // Cannot get the id of the component constant, while all components
      // should have been added to the module prior to the composite constant.
      // Cannot create OpConstantComposite instruction in this case.
      return nullptr;
    }
    operands.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                          std::initializer_list<uint32_t>{id});
  }
  return MakeUnique<ir::Instruction>(
      context(), SpvOp::SpvOpConstantComposite,
      context()->get_type_mgr()->GetId(cc->type()), result_id,
      std::move(operands));
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
