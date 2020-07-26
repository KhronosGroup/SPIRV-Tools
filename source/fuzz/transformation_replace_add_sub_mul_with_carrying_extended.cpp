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

#include "source/fuzz/transformation_replace_add_sub_mul_with_carrying_extended.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(
        const spvtools::fuzz::protobufs::
            TransformationReplaceAddSubMulWithCarryingExtended& message)
    : message_(message) {}

TransformationReplaceAddSubMulWithCarryingExtended::
    TransformationReplaceAddSubMulWithCarryingExtended(uint32_t fresh_id,
                                                       uint32_t result_id) {
  message_.set_fresh_id(fresh_id);
  message_.set_result_id(result_id);
}

bool TransformationReplaceAddSubMulWithCarryingExtended::IsApplicable(
    opt::IRContext*, const TransformationContext&) const {
  return false;
}

void TransformationReplaceAddSubMulWithCarryingExtended::Apply(
    opt::IRContext*, TransformationContext* /*unused*/) const {}

protobufs::Transformation
TransformationReplaceAddSubMulWithCarryingExtended::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_add_sub_mul_with_carrying_extended() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools