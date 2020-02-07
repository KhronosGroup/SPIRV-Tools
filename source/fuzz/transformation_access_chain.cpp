// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_access_chain.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAccessChain::TransformationAccessChain(
    const spvtools::fuzz::protobufs::TransformationAccessChain& message)
    : message_(message) {}

TransformationAccessChain::TransformationAccessChain(uint32_t fresh_id, uint32_t pointer_id, const std::vector<uint32_t>& index, const protobufs::InstructionDescriptor& instruction_to_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  for (auto an_index : index) {
    message_.add_index(an_index);
  }
  *message_.mutable_instruction_to_insert_before() = instruction_to_insert_before;
}

bool TransformationAccessChain::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  // The result id must be fresh
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }
  // The pointer id must exist and have a type.
  auto pointer = context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }
  // The type must indeed be a pointer
  auto pointer_type = context->get_def_use_mgr()->GetDef(pointer->type_id());
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return false;
  }

  // TODO Check availability
  // TODO Check suitability of insertion point

  // TODO comment this method
  uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);
  for (uint32_t index = 0; index < static_cast<uint32_t>(message_.index_size()); index++) {
    std::pair<bool, uint32_t> maybe_index_value = GetIndexValue(context, index);
    if (!maybe_index_value.first) {
      return false;
    }
    subobject_type_id = fuzzerutil::WalkOneCompositeTypeIndex(context, subobject_type_id, maybe_index_value.second);
    if (!subobject_type_id) {
      return false;
    }
  }
  auto result_type_and_ptr = context->get_type_mgr()->GetTypeAndPointerType(subobject_type_id, static_cast<SpvStorageClass>(pointer_type->GetSingleWordInOperand(0)));
  return context->get_type_mgr()->GetId(result_type_and_ptr.second.get()) != 0;
}

void TransformationAccessChain::Apply(
    opt::IRContext* context,
    spvtools::fuzz::FactManager* fact_manager) const {
  opt::Instruction::OperandList operands;
  operands.push_back({ SPV_OPERAND_TYPE_ID, { message_.pointer_id()} });
  auto pointer_type = context->get_def_use_mgr()->GetDef(context->get_def_use_mgr()->GetDef(message_.pointer_id())->type_id());
  uint32_t subobject_type = pointer_type->GetSingleWordInOperand(1);
  for (auto index : message_.index()) {
    uint32_t index_value = GetIndexValue(context, index).second;
    operands.push_back({ SPV_OPERAND_TYPE_ID, { index_value }});
    subobject_type = fuzzerutil::WalkOneCompositeTypeIndex(context, subobject_type, index_value);
  }
  uint32_t result_type = context->get_type_mgr()->GetId(
          context->get_type_mgr()->GetTypeAndPointerType(subobject_type, static_cast<SpvStorageClass>(pointer_type->GetSingleWordInOperand(0))).second.get());

  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());
  FindInstruction(message_.instruction_to_insert_before(), context)
          ->InsertBefore(MakeUnique<opt::Instruction>(
                  context, SpvOpAccessChain, result_type, message_.fresh_id(),
                  operands));
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  if (fact_manager->PointeeValueIsIrrelevant(message_.pointer_id())) {
    fact_manager->AddFactValueOfPointeeIsIrrelevant(message_.fresh_id());
  }
}

protobufs::Transformation TransformationAccessChain::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_access_chain() = message_;
  return result;
}

std::pair<bool, uint32_t> TransformationAccessChain::GetIndexValue(opt::IRContext* context, uint32_t index) const {
  auto index_instruction = context->get_def_use_mgr()->GetDef(message_.index(index));
  if (!index_instruction || !spvOpcodeIsConstant(index_instruction->opcode())) {
    return {false, 0};
  }
  auto index_type = context->get_def_use_mgr()->GetDef(index_instruction->type_id());
  if (index_type->opcode() != SpvOpTypeInt || index_type->GetSingleWordInOperand(0) != 32) {
    return {false, 0};
  }
  return {true, index_instruction->GetSingleWordInOperand(0)};

}

}  // namespace fuzz
}  // namespace spvtools
