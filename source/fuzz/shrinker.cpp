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

#include "source/fuzz/shrinker.h"

#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/replayer.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

struct Shrinker::Impl {
  explicit Impl(spv_target_env env) : target_env(env) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
};

Shrinker::Shrinker(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Shrinker::~Shrinker() = default;

void Shrinker::SetMessageConsumer(MessageConsumer c) {
  impl_->consumer = std::move(c);
}

Shrinker::ShrinkerResultStatus Shrinker::Run(
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const protobufs::TransformationSequence& transformation_sequence_in,
    const Shrinker::InterestingnessFunction& interestingness_function,
    spv_const_fuzzer_options options, std::vector<uint32_t>* binary_out,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  spvtools::SpirvTools tools(impl_->target_env);
  if (!tools.IsValid()) {
    impl_->consumer(SPV_MSG_ERROR, nullptr, {},
                    "Failed to create SPIRV-Tools interface; stopping.");
    return Shrinker::ShrinkerResultStatus::kFailedToCreateSpirvToolsInterface;
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size())) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial binary is invalid; stopping.");
    return Shrinker::ShrinkerResultStatus::kInitialBinaryInvalid;
  }

  // Make a PRNG, either from a given seed or from a random device.
  PseudoRandomGenerator random_generator(
      options->has_random_seed ? options->random_seed
                               : static_cast<uint32_t>(std::random_device()()));

  Replayer replayer(impl_->target_env);

  std::vector<uint32_t> current_best_binary;
  protobufs::TransformationSequence current_best_transformations;
  std::vector<uint32_t> next_binary;
  protobufs::TransformationSequence next_transformation_sequence;

  auto replayer_result =
      replayer.Run(binary_in, initial_facts, transformation_sequence_in,
                   &current_best_binary, &current_best_transformations);
  if (replayer_result != Replayer::ReplayerResultStatus::kComplete) {
    return ShrinkerResultStatus::kReplayFailed;
  }

  if (!interestingness_function(current_best_binary, 0)) {
    return ShrinkerResultStatus::kInitialBinaryNotInteresting;
  }

  for (uint32_t i = 1;
       i < 100 && !current_best_transformations.transformation().empty(); i++) {
    std::cout << "Attempt " << i << std::endl;
    auto skipit = random_generator.RandomUint32(static_cast<uint32_t>(
        current_best_transformations.transformation().size()));
    protobufs::TransformationSequence candidate_sequence;
    candidate_sequence.clear_transformation();
    std::cout << "Remaining: "
              << current_best_transformations.transformation().size()
              << std::endl;
    for (uint32_t j = 0;
         j < static_cast<uint32_t>(
                 current_best_transformations.transformation().size());
         j++) {
      if (j == skipit) {
        continue;
      }
      protobufs::Transformation transformation =
          current_best_transformations.transformation()[j];
      *candidate_sequence.mutable_transformation()->Add() = transformation;
    }
    next_transformation_sequence.clear_transformation();
    next_binary.clear();
    replayer_result = replayer.Run(binary_in, initial_facts, candidate_sequence,
                                   &next_binary, &next_transformation_sequence);
    if (replayer_result != Replayer::ReplayerResultStatus::kComplete) {
      return ShrinkerResultStatus::kReplayFailed;
    }
    if (interestingness_function(next_binary, i)) {
      std::cout << "Interesting!" << std::endl;
      current_best_binary = next_binary;
      current_best_transformations.CopyFrom(next_transformation_sequence);
    }
  }
  *binary_out = current_best_binary;
  *transformation_sequence_out = current_best_transformations;
  return Shrinker::ShrinkerResultStatus::kComplete;
}

}  // namespace fuzz
}  // namespace spvtools
