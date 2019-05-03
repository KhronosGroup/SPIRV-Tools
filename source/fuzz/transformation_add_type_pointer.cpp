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

#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
namespace transformation {

using opt::IRContext;

bool IsApplicable(const protobufs::TransformationAddTypePointer& message,
                  IRContext* context,
                  const spvtools::fuzz::FactManager& /*unused*/) {
  (void)(message);
  (void)(context);
  assert(0);
}

void Apply(const protobufs::TransformationAddTypePointer& message,
           IRContext* context, spvtools::fuzz::FactManager* /*unused*/) {
  (void)(context);
  assert(0);
  fuzzerutil::UpdateModuleIdBound(context, message.fresh_id());
  // We have added an instruction to the module, so need to be careful about the
  // validity of existing analyses.
  context->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
}

protobufs::TransformationAddTypePointer MakeTransformationAddTypePointer(
    uint32_t fresh_id, SpvStorageClass storage_class, uint32_t base_type_id) {
  protobufs::TransformationAddTypePointer result;
  result.set_fresh_id(fresh_id);
  result.set_storage_class(storage_class);
  result.set_base_type_id(base_type_id);
  return result;
}

}  // namespace transformation
}  // namespace fuzz
}  // namespace spvtools
