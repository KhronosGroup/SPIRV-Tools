// Copyright (c) 2018 Google LLC
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

#include <cassert>
#include <sstream>

#include "source/spirv_reducer_options.h"

#include "reducer.h"
#include "reduction_pass.h"

namespace spvtools {
namespace reduce {

struct Reducer::Impl {
  explicit Impl(spv_target_env env)
      : target_env(env) {}

  bool ReachedStepLimit(uint32_t current_step,
                        spv_const_reducer_options options);

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
  InterestingnessFunction interestingness_function;
  std::vector<std::unique_ptr<ReductionPass>> passes;
};

Reducer::Reducer(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Reducer::~Reducer() = default;

void Reducer::SetMessageConsumer(MessageConsumer c) {
  for (auto& pass : impl_->passes) {
    pass->SetMessageConsumer(c);
  }
  impl_->consumer = std::move(c);
}

void Reducer::SetInterestingnessFunction(
        Reducer::InterestingnessFunction interestingness_function) {
  impl_->interestingness_function = std::move(interestingness_function);
}

void Reducer::Run(std::vector<uint32_t>&& binary_in,
                  std::vector<uint32_t>* binary_out,
                  spv_const_reducer_options options) const {
  std::vector<uint32_t> current_binary = binary_in;

  // Initial state should be interesting.
  assert(impl_->interestingness_function(current_binary));

  // Keeps track of how many reduction attempts have been tried.  Reduction
  // bails out if this reaches a given limit.
  uint32_t reductions_applied = 0;

  // Determines whether, on completing one round of reduction passes, it is
  // worthwhile trying a further round.
  bool another_round_worthwhile = true;

  // Apply round after round of reduction passes until we hit the reduction
  // step limit, or deem that another round is not going to be worthwhile.
  while(!impl_->ReachedStepLimit(reductions_applied, options)
      && another_round_worthwhile) {

    // At the start of a round of reduction passes, assume another round will
    // not be worthwhile unless we find evidence to the contrary.
    another_round_worthwhile = false;

    // Iterate through the available passes
    for (auto &pass : impl_->passes) {
      // Keep applying this pass at its current granularity until it stops
      // working or we hit the reduction step limit.
      impl_->consumer(SPV_MSG_INFO, nullptr, {},
              ("Trying pass " + pass->GetName() + ".").c_str());
      do {
        std::stringstream stringstream;
        stringstream << "Reduction step " << reductions_applied;
        impl_->consumer(SPV_MSG_INFO, nullptr, {},
                (stringstream.str().c_str()));
        auto maybe_result = pass->ApplyReduction(current_binary);
        reductions_applied++;
        if (maybe_result.empty()) {
          // This pass did not have any impact, so move on to the next pass.
          // If this pass hasn't reached its minimum granularity then it's
          // worth eventually doing another round of reductions, in order to
          // try this pass at a finer granularity.
          impl_->consumer(SPV_MSG_INFO, nullptr, {},
                  ("Pass " + pass->GetName()
                  + " did not make a reduction step.").c_str());
          another_round_worthwhile |= !pass->ReachedMinimumGranularity();
          break;
        }
        impl_->consumer(SPV_MSG_INFO, nullptr, {},
                ("Pass " + pass->GetName()
                + " made a reduction step.").c_str());
        if (impl_->interestingness_function(maybe_result)) {
          // Success!  The binary produced by this reduction step is
          // interesting, so make it the binary of interest henceforth, and
          // note that it's worth doing another round of reduction passes.
          impl_->consumer(SPV_MSG_INFO, nullptr, {},
                  "Reduction step succeeded.");
          current_binary = std::move(maybe_result);
          another_round_worthwhile = true;
        }
        // Bail out if the reduction step limit has been reached.
      } while (!impl_->ReachedStepLimit(reductions_applied, options));
    }
  }

  // Report whether reduction completed, or bailed out early due to reaching
  // the step limit.
  if (impl_->ReachedStepLimit(reductions_applied, options)) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Reached reduction step limit; stopping.");
  } else {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "No more to reduce; stopping.");
  }

  *binary_out = std::move(current_binary);
}

void Reducer::AddReductionPass(
    std::unique_ptr<ReductionPass>&& reduction_pass) {
  impl_->passes.push_back(std::move(reduction_pass));
}

bool Reducer::Impl::ReachedStepLimit(uint32_t current_step,
        spv_const_reducer_options options) {
  return current_step >= options->step_limit;
}

}  // namespace reduce
}  // namespace spvtools