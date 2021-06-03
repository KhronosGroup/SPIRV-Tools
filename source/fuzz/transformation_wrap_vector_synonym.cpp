// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/transformation_wrap_vector_synonym.h"

#include "source/opt/function.h"
#include "source/opt/module.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationWrapVectorSynonym:TransformationWrapVectorSynonym(
    protobufs::TransformationWrapVectorSynonym message)
    : message_(std::move(message)) {}

TransformationWrapVectorSynonym::TransformationWrapVectorSynonym(uint32_t vector_size,
                                                                  uint32_t instruction_id) {

}

bool TransformationWrapVectorSynonym::IsApplicable(

}

void TransformationWrapVectorSynonym::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {

}

protobufs::Transformation TransformationWrapVectorSynonym::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_two_functions() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationWrapVectorSynonym::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
