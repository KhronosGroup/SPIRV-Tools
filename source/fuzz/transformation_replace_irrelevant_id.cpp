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

#include "source/fuzz/transformation_replace_irrelevant_id.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceIrrelevantId::TransformationReplaceIrrelevantId(
    const protobufs::TransformationReplaceIrrelevantId& message)
    : message_(message) {}

TransformationReplaceIrrelevantId::TransformationReplaceIrrelevantId(
    protobufs::IdUseDescriptor id_use_descriptor, uint32_t replacement_id) {
  *message_.mutable_id_use_descriptor() = id_use_descriptor;
  message_.set_replacement_id(replacement_id);
}

bool TransformationReplaceIrrelevantId::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  auto id_of_interest = message_.id_use_descriptor().id_of_interest();

  // The id must be irrelevant.
  if (!transformation_context.GetFactManager()->IdIsIrrelevant(
          id_of_interest)) {
    return false;
  }

  // Find the instruction containing the id use.
  auto use_instruction =
      FindInstructionContainingUse(message_.id_use_descriptor(), ir_context);
  if (!use_instruction) {
    return false;
  }

  // The type of the id of interest and of the replacement id must be the same.
  uint32_t type_id_of_interest =
      ir_context->get_def_use_mgr()->GetDef(id_of_interest)->type_id();
  uint32_t type_replacement_id = ir_context->get_def_use_mgr()
                                     ->GetDef(message_.replacement_id())
                                     ->type_id();
  if (type_id_of_interest != type_replacement_id) {
    return false;
  }

  // The id use must be replaceable with any other id of the same type.
  return fuzzerutil::IdUseCanBeReplaced(
      ir_context, use_instruction,
      message_.id_use_descriptor().in_operand_index());
}

void TransformationReplaceIrrelevantId::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationReplaceIrrelevantId::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_irrelevant_id() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
