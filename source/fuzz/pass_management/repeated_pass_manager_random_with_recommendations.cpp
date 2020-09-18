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

#include "source/fuzz/pass_management/repeated_pass_manager_random_with_recommendations.h"

namespace spvtools {
namespace fuzz {

RepeatedPassManagerRandomWithRecommendations::
    RepeatedPassManagerRandomWithRecommendations(
        FuzzerContext* fuzzer_context, RepeatedPassInstances* pass_instances,
        RepeatedPassRecommender* pass_recommender)
    : RepeatedPassManager(fuzzer_context, pass_instances),
      pass_recommender_(pass_recommender) {}

RepeatedPassManagerRandomWithRecommendations::
    ~RepeatedPassManagerRandomWithRecommendations() = default;

FuzzerPass* RepeatedPassManagerRandomWithRecommendations::ChoosePass() {
  FuzzerPass* result;
  if (recommended_passes_.empty() || GetFuzzerContext()->ChooseEven()) {
    auto& passes = GetPassInstances()->GetPasses();
    result = passes[GetFuzzerContext()->RandomIndex(passes)].get();
  } else {
    result = recommended_passes_.front();
    recommended_passes_.pop_front();
  }
  for (auto future_pass :
       pass_recommender_->GetFuturePassRecommendations(*result)) {
    recommended_passes_.push_back(future_pass);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
