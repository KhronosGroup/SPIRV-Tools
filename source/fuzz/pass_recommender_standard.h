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

#ifndef SOURCE_FUZZ_PASS_RECOMMENDER_STANDARD_
#define SOURCE_FUZZ_PASS_RECOMMENDER_STANDARD_

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/pass_instances.h"
#include "source/fuzz/pass_recommender.h"

namespace spvtools {
namespace fuzz {

// TODO COMMENT.
class PassRecommenderStandard : public PassRecommender {
 public:
  PassRecommenderStandard(PassInstances* pass_instances, FuzzerContext* fuzzer_context);

  ~PassRecommenderStandard();

  // TODO COMMENT.
  std::vector<FuzzerPass*> GetFuturePassRecommendations(const FuzzerPass& pass) override;

 private:

  std::vector<FuzzerPass*> RandomOrderAndNonNull(
      const std::vector<FuzzerPass*>& passes);

  PassInstances* pass_instances_;

  FuzzerContext* fuzzer_context_;

};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_PASS_RECOMMENDER_STANDARD_
