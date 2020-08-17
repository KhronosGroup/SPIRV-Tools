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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {
class FuzzerPassAddOpPhiSynonyms : public FuzzerPass {
 public:
  FuzzerPassAddOpPhiSynonyms(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddOpPhiSynonyms() override;

  void Apply() override;

  // Computes the equivalence classes for the non-pointer and non-irrelevant ids
  // in the module, where two ids are considered equivalent iff they have been
  // declared synonymous and they have the same type.
  std::vector<std::set<uint32_t>> GetIdEquivalenceClasses();
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_OPPHI_SYNONYMS_
