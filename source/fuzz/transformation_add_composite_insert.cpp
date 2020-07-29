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

#include "transformation_add_composite_insert.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationAddCompositeInsert::TransformationAddCompositeInsert(
    const spvtools::fuzz::protobufs::TransformationAddCompositeInsert& message)
    : message_(message) {}

TransformationAddCompositeInsert::TransformationAddCompositeInsert(
    uint32_t fresh_id, uint32_t available_constant_id,
    uint32_t composite_value_id, uint32_t index_to_replace,
    const protobufs::InstructionDescriptor&
        instruction_descriptor_insert_before) {
  message_.set_fresh_id(fresh_id);
  message_.set_available_constant_id(available_constant_id);
  message_.set_composite_value_id(composite_value_id);
  message_.set_index_to_replace(index_to_replace);
  *message_.mutable_instruction_descriptor_insert_before() =
      instruction_descriptor_insert_before;
}
bool TransformationAddCompositeInsert::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // |message_.composite_value_id| must refer to an existing composite value.
  auto composite_value_type =
      ir_context->get_type_mgr()->GetType(message_.composite_value_id());
  if (!fuzzerutil::IsCompositeType(composite_value_type)) {
    return false;
  }
  // |message_.index_to_replace| must refer to a valid index.
  auto composite_value =
      ir_context->get_def_use_mgr()->GetDef(message_.composite_value_id());
  auto num_operands = composite_value->NumInOperands();
  if (message_.index_to_replace() >= num_operands) {
    return false;
  }

  // The type id of |message_.available_constant| and the type id of the
  // component of |message_.composite_value| at index
  // |message_.index_to_replace| must be the same.
  auto constant_to_be_replaced_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(composite_value->GetSingleWordInOperand(
              message_.index_to_replace()))
          ->type_id();
  auto available_constant_type_id =
      ir_context->get_def_use_mgr()
          ->GetDef(message_.available_constant_id())
          ->type_id();

  if (constant_to_be_replaced_type_id != available_constant_type_id) {
    return false;
  }

  //|message_.instruction_descriptor_insert_before| must refer to a valid
  // instruction.
  auto instruction_insert_before = FindInstruction(
      message_.instruction_descriptor_insert_before(), ir_context);

  return instruction_insert_before != nullptr;
}

void TransformationAddCompositeInsert::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.fresh_id()) &&
         "|message_.fresh_id| must be fresh");

  auto composite_value =
      ir_context->get_def_use_mgr()->GetDef(message_.composite_value_id());

  // Insert the requested OpCompositeInsert instruction.
  FindInstruction(message_.instruction_descriptor_insert_before(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCompositeInsert, composite_value->type_id(),
          message_.fresh_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {message_.available_constant_id()}},
               {SPV_OPERAND_TYPE_ID, {message_.composite_value_id()}},
               {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                {message_.index_to_replace()}}})));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationAddCompositeInsert::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_composite_insert() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools