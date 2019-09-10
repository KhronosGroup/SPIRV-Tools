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

#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"

namespace spvtools {
namespace fuzz {

FuzzerPassApplyIdSynonyms::FuzzerPassApplyIdSynonyms(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassApplyIdSynonyms::~FuzzerPassApplyIdSynonyms() = default;

void FuzzerPassApplyIdSynonyms::Apply() { assert(0); }

}  // namespace fuzz
}  // namespace spvtools
