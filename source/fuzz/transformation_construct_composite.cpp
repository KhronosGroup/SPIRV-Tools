// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_construct_composite.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationConstructComposite::TransformationConstructComposite(
    const protobufs::TransformationConstructComposite& message)
    : message_(message) {}

TransformationConstructComposite::TransformationConstructComposite(
    uint32_t composite_type_id, std::vector<uint32_t> component,
    uint32_t base_instruction_id, uint32_t offset, uint32_t fresh_id) {
  message_.set_composite_type_id(composite_type_id);
  for (auto a_component : component) {
    message_.add_component(a_component);
  }
  message_.set_base_instruction_id(base_instruction_id);
  message_.set_offset(offset);
  message_.set_fresh_id(fresh_id);
}

bool TransformationConstructComposite::IsApplicable(
    opt::IRContext* context, const FactManager& /*fact_manager*/) const {
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    // We require the id for the composite constructor to be unused.
    return false;
  }

  auto base_instruction =
      context->get_def_use_mgr()->GetDef(message_.base_instruction_id());
  if (!base_instruction) {
    // The given id to insert after is not defined.
    return false;
  }

  auto destination_block = context->get_instr_block(base_instruction);
  if (!destination_block) {
    // The given id to insert after is not in a block.
    return false;
  }

  auto insert_before = fuzzerutil::GetIteratorForBaseInstructionAndOffset(
      destination_block, base_instruction, message_.offset());

  if (insert_before == destination_block->end()) {
    // The offset was inappropriate.
    return false;
  }

  auto composite_type =
      context->get_type_mgr()->GetType(message_.composite_type_id());

  if (composite_type == nullptr) {
    // The composite type was not found
    return false;
  }

  if (composite_type->AsVector()) {
    uint32_t base_element_count = 0;
    auto element_type = composite_type->AsVector()->element_type();
    for (auto& component_id : message_.component()) {
      auto inst = context->get_def_use_mgr()->GetDef(component_id);
      if (inst == nullptr || !inst->type_id()) {
        // The component does not correspond to an instruction with a result
        // type.
        return false;
      }
      auto component_type = context->get_type_mgr()->GetType(inst->type_id());
      assert(component_type);
      if (component_type == element_type) {
        base_element_count++;
      } else if (component_type->AsVector() &&
                 component_type->AsVector()->element_type() == element_type) {
        base_element_count += component_type->AsVector()->element_count();
      } else {
        // The component was not appropriate; e.g. no type corresponding to the
        // given id was found, or the type that was found was not compatible
        // with the vector being constructed.
        return false;
      }
    }
    if (base_element_count != composite_type->AsVector()->element_count()) {
      // The number of components provided (when vector components are flattened
      // out) does not match the length of the vector being constructed.
      return false;
    }
  } else if (composite_type->AsMatrix()) {
    assert(false && "Not handled yet.");
  } else if (composite_type->AsArray()) {
    assert(false && "Not handled yet.");
  } else if (composite_type->AsStruct()) {
    assert(false && "Not handled yet.");
  } else {
    // The type is not a composite
    return false;
  }

  // Now check whether every component being used to initialize the composite is
  // available at the desired program point.
  for (auto& component : message_.component()) {
    auto component_inst = context->get_def_use_mgr()->GetDef(component);
    if (!context->get_instr_block(component)) {
      // The component does not have a block; that means it is in global scope,
      // which is OK. (Whether the component actually corresponds to an
      // instruction is checked above when determining whether types are
      // suitable.)
      continue;
    }
    // Check whether the component is available.
    if (insert_before->HasResultId() &&
        insert_before->result_id() == component) {
      // This constitutes trying to use an id right before it is defined.  The
      // special case is needed due to an instruction always dominating itself.
      return false;
    }
    if (!context
             ->GetDominatorAnalysis(
                 context->get_instr_block(&*insert_before)->GetParent())
             ->Dominates(component_inst, &*insert_before)) {
      // The instruction defining the component must dominate the instruction we
      // wish to insert the composite before.
      return false;
    }
  }

  return true;
}

void TransformationConstructComposite::Apply(
    opt::IRContext* context, FactManager* /*fact_manager*/) const {
  auto base_instruction =
      context->get_def_use_mgr()->GetDef(message_.base_instruction_id());
  auto destination_block = context->get_instr_block(base_instruction);
  auto insert_before = fuzzerutil::GetIteratorForBaseInstructionAndOffset(
      destination_block, base_instruction, message_.offset());

  opt::Instruction::OperandList in_operands;
  for (auto& component_id : message_.component()) {
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {component_id}});
  }

  insert_before.InsertBefore(MakeUnique<opt::Instruction>(
      context, SpvOpCompositeConstruct, message_.composite_type_id(),
      message_.fresh_id(), in_operands));
  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationConstructComposite::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_construct_composite() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
