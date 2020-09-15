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

#ifndef SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_
#define SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_

#include <vector>

#include "source/fuzz/pass_recommender.h"
#include "source/fuzz/repeated_pass_manager.h"

namespace spvtools {
namespace fuzz {

// TODO COMMENT.
class RepeatedPassManagerLoopedWithRecommendations : public RepeatedPassManager {
 public:
  RepeatedPassManagerLoopedWithRecommendations(FuzzerContext* fuzzer_context, PassInstances* pass_instances, PassRecommender* pass_recommender);

  ~RepeatedPassManagerLoopedWithRecommendations();

  // TODO COMMENT.
  FuzzerPass* ChoosePass() override;

 private:
  // TODO comment
  std::vector<FuzzerPass*> pass_loop_;

  uint32_t next_pass_index_;

};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_MANAGER_LOOPED_WITH_RECOMMENDATIONS_
