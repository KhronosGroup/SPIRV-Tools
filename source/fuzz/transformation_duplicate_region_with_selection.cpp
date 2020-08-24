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

#include "source/fuzz/transformation_duplicate_region_with_selection.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

/*
 * TransformationAddParameter::TransformationAddParameter(
    uint32_t function_id, uint32_t parameter_fresh_id,
    uint32_t parameter_type_id, std::map<uint32_t, uint32_t> call_parameter_ids,
    uint32_t function_type_fresh_id) {
  message_.set_function_id(function_id);
  message_.set_parameter_fresh_id(parameter_fresh_id);
  message_.set_parameter_type_id(parameter_type_id);
  *message_.mutable_call_parameter_ids() =
      fuzzerutil::MapToRepeatedUInt32Pair(call_parameter_ids);
  message_.set_function_type_fresh_id(function_type_fresh_id);
}

 */

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        const spvtools::fuzz::protobufs::
            TransformationDuplicateRegionWithSelection& message)
    : message_(message) {}

TransformationDuplicateRegionWithSelection::
    TransformationDuplicateRegionWithSelection(
        uint32_t condition_fresh_id, uint32_t merge_label_fresh_id,
        uint32_t then_label_fresh_id, uint32_t else_label_fresh_id,
        uint32_t entry_block_id, uint32_t exit_block_id,
        std::map<uint32_t, uint32_t> original_label_to_duplicate_label,
        std::map<uint32_t, uint32_t> original_id_to_duplicate_id,
        std::map<uint32_t, uint32_t> original_id_to_phi_id) {
  message_.set_condition_fresh_id(condition_fresh_id);
  message_.set_merge_label_fresh_id(merge_label_fresh_id);
  message_.set_then_label_fresh_id(then_label_fresh_id);
  message_.set_else_label_fresh_id(else_label_fresh_id);
  message_.set_entry_block_id(entry_block_id);
  message_.set_exit_block_id(exit_block_id);
  *message_.mutable_original_label_to_duplicate_label() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_label_to_duplicate_label);
  *message_.mutable_original_id_to_duplicate_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_duplicate_id);
  *message_.mutable_original_id_to_phi_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(original_id_to_phi_id);
}

bool TransformationDuplicateRegionWithSelection::IsApplicable(
    opt::IRContext* /*ir_context*/,
    const TransformationContext& /*transformation_context*/) const {
  return false;
}

void TransformationDuplicateRegionWithSelection::Apply(
    opt::IRContext* /*ir_context*/, TransformationContext* /*unused*/) const {}

protobufs::Transformation
TransformationDuplicateRegionWithSelection::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_duplicate_region_with_selection() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
