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

#include "source/fuzz/transformation_replace_id_with_synonym.h"

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceIdWithSynonym::TransformationReplaceIdWithSynonym(
    const spvtools::fuzz::protobufs::TransformationReplaceIdWithSynonym&
        message)
    : message_(message) {}

TransformationReplaceIdWithSynonym::TransformationReplaceIdWithSynonym(
    const protobufs::IdUseDescriptor id_use_descriptor,
    const protobufs::DataDescriptor data_descriptor,
    uint32_t fresh_id_for_temporary) {
  assert(fresh_id_for_temporary == 0 && data_descriptor.index().size() == 0 &&
         "At present we do not support making an id synonymous with an index "
         "into a composite.");
  *message_.mutable_id_use_descriptor() = std::move(id_use_descriptor);
  *message_.mutable_data_descriptor() = std::move(data_descriptor);
  message_.set_fresh_id_for_temporary(fresh_id_for_temporary);
}

bool TransformationReplaceIdWithSynonym::IsApplicable(
    spvtools::opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) const {
  assert(0);
  (void)(context);
  (void)(fact_manager);
  return false;
}

void TransformationReplaceIdWithSynonym::Apply(
    spvtools::opt::IRContext* context,
    spvtools::fuzz::FactManager* fact_manager) const {
  (void)(context);
  (void)(fact_manager);
  // context->InvalidateAnalysesExceptFor(opt::IRContext::Analysis::kAnalysisNone);
  assert(0);
}

protobufs::Transformation TransformationReplaceIdWithSynonym::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_replace_id_with_synonym() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
