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

#include "transformation_composite_insert.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationCompositeInsert::TransformationCompositeInsert(
    const spvtools::fuzz::protobufs::TransformationCompositeInsert& message)
    : message_(message) {}

TransformationCompositeInsert::TransformationCompositeInsert(
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    uint32_t fresh_id, uint32_t composite_id, uint32_t object_id,
    std::vector<uint32_t>&& index) {
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  message_.set_fresh_id(fresh_id);
  message_.set_composite_id(composite_id);
  message_.set_object_id(object_id);
  for (auto an_index : index) {
    message_.add_index(an_index);
  }
}

bool TransformationCompositeInsert::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.fresh_id| must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }

  // |message_.composite_id| must refer to an existing composite value.
  auto composite =
      ir_context->get_def_use_mgr()->GetDef(message_.composite_id());
  auto composite_type =
      ir_context->get_type_mgr()->GetType(composite->type_id());
  if (!fuzzerutil::IsCompositeType(composite_type)) {
    return false;
  }

  // |message_.index| must refer to a valid index.
  if (fuzzerutil::WalkCompositeTypeIndices(ir_context, composite->type_id(),
                                           message_.index()) == 0) {
    return false;
  }

  // The type id of the object and the type id of the component of the composite
  // at index |message_.index| must be the same.
  auto component_to_be_replaced_type_id = fuzzerutil::WalkCompositeTypeIndices(
      ir_context, composite->type_id(), message_.index());
  if (component_to_be_replaced_type_id == 0) {
    return false;
  }
  auto object_type_id =
      ir_context->get_def_use_mgr()->GetDef(message_.object_id())->type_id();

  if (component_to_be_replaced_type_id != object_type_id) {
    return false;
  }

  // |message_.instruction_to_insert_before| must refer to a valid
  // instruction.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);

  return instruction_to_insert_before != nullptr;
}

void TransformationCompositeInsert::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // |message_.struct_fresh_id| must be fresh.
  assert(fuzzerutil::IsFreshId(ir_context, message_.fresh_id()) &&
         "|message_.fresh_id| must be fresh");

  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.object_id()}});
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.composite_id()}});
  for (auto an_index : message_.index()) {
    in_operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {an_index}});
  }
  auto composite =
      ir_context->get_def_use_mgr()->GetDef(message_.composite_id());

  FindInstruction(message_.instruction_to_insert_before(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpCompositeInsert, composite->type_id(),
          message_.fresh_id(), in_operands));

  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationCompositeInsert::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_composite_insert() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools