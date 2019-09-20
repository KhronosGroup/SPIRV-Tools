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

  assert(0);

  return true;
}

void TransformationConstructComposite::Apply(
    opt::IRContext* /*context*/, FactManager* /*fact_manager*/) const {
  assert(false);
}

protobufs::Transformation TransformationConstructComposite::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_construct_composite() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
