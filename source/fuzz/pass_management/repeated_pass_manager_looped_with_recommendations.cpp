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

#include "source/fuzz/pass_management/repeated_pass_manager_looped_with_recommendations.h"

namespace spvtools {
namespace fuzz {

RepeatedPassManagerLoopedWithRecommendations::
    RepeatedPassManagerLoopedWithRecommendations(
        FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
        RepeatedPassRecommender* pass_recommender)
    : RepeatedPassManager(fuzzer_context, pass_instances), next_pass_index_(0) {
  auto& passes = GetPassInstances()->GetPasses();
  do {
    FuzzerPass* current_pass =
        passes[GetFuzzerContext()->RandomIndex(passes)].get();
    pass_loop_.push_back(current_pass);
    for (auto future_pass :
         pass_recommender->GetFuturePassRecommendations(*current_pass)) {
      pass_loop_.push_back(future_pass);
    }
  } while (fuzzer_context->ChoosePercentage(
      fuzzer_context->GetChanceOfAddingAnotherPassToPassLoop()));
}

RepeatedPassManagerLoopedWithRecommendations::
    ~RepeatedPassManagerLoopedWithRecommendations() = default;

FuzzerPass* RepeatedPassManagerLoopedWithRecommendations::ChoosePass() {
  auto result = pass_loop_[next_pass_index_];
  next_pass_index_ =
      (next_pass_index_ + 1) % static_cast<uint32_t>(pass_loop_.size());
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
