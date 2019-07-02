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

#include <sstream>

#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/replayer.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

namespace {

// A helper to get the size of a protobuf transformation sequence in a less
// verbose manner.
uint32_t NumRemainingTransformations(
    const protobufs::TransformationSequence& transformation_sequence) {
  return static_cast<uint32_t>(transformation_sequence.transformation_size());
}

// A helper to return a transformation sequence identical to |transformations|,
// except that a chunk of size |granularity| starting from
// (|transformations.size| - |index + 1| x |granularity|) is removed.
protobufs::TransformationSequence RemoveChunk(
    const protobufs::TransformationSequence& transformations, uint32_t index,
    uint32_t granularity) {
  uint32_t lower = static_cast<uint32_t>(std::max(
      0, static_cast<int>(NumRemainingTransformations(transformations)) -
             static_cast<int>((index + 1) * granularity)));
  uint32_t upper =
      NumRemainingTransformations(transformations) - index * granularity;
  assert(lower < upper);
  assert(upper <= NumRemainingTransformations(transformations));
  protobufs::TransformationSequence result;
  for (uint32_t j = 0; j < NumRemainingTransformations(transformations); j++) {
    if (j >= lower && j < upper) {
      continue;
    }
    protobufs::Transformation transformation =
        transformations.transformation()[j];
    *result.mutable_transformation()->Add() = transformation;
  }
  return result;
}

}  // namespace

struct Shrinker::Impl {
  explicit Impl(spv_target_env env, uint32_t limit)
      : target_env(env), step_limit(limit) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
  const uint32_t step_limit;        // Step limit for reductions.
};

Shrinker::Shrinker(spv_target_env env, uint32_t step_limit)
    : impl_(MakeUnique<Impl>(env, step_limit)) {}

Shrinker::~Shrinker() = default;

void Shrinker::SetMessageConsumer(MessageConsumer c) {
  impl_->consumer = std::move(c);
}

Shrinker::ShrinkerResultStatus Shrinker::Run(
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const protobufs::TransformationSequence& transformation_sequence_in,
    const Shrinker::InterestingnessFunction& interestingness_function,
    std::vector<uint32_t>* binary_out,
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

  std::vector<uint32_t> current_best_binary;
  protobufs::TransformationSequence current_best_transformations;

  // Run a replay of the initial transformation sequence to (a) check that it
  // succeeds, (b) get the binary that results from running these
  // transformations, and (c) get the subsequence of the initial transformations
  // that actually apply (in principle this could be a strict subsequence).
  if (Replayer(impl_->target_env)
          .Run(binary_in, initial_facts, transformation_sequence_in,
               &current_best_binary, &current_best_transformations) !=
      Replayer::ReplayerResultStatus::kComplete) {
    return ShrinkerResultStatus::kReplayFailed;
  }

  // Check that the binary produced by applying the initial transformations is
  // indeed interesting.
  if (!interestingness_function(current_best_binary, 0)) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial binary is not interesting; stopping.");
    return ShrinkerResultStatus::kInitialBinaryNotInteresting;
  }

  uint32_t attempt = 0;  // Keeps track of the number of shrink attempts that
                         // have been tried, whether successful or not.

  uint32_t granularity =
      std::max(1u, NumRemainingTransformations(current_best_transformations) /
                       2);  // The number of contiguous transformations that the
                            // shrinker will try to remove in one go; starts
                            // high and decreases during the shrinking process.

  // Keep shrinking until we:
  // - reach the step limit,
  // - run out of transformations to remove, or
  // - cannot make the granularity of shrinking any finer.
  while (attempt < impl_->step_limit &&
         !current_best_transformations.transformation().empty() &&
         granularity > 0) {
    bool progress_this_round =
        false;  // Used to decide whether to make the granularity with which we
                // remove transformations finer.  If we managed to remove at
                // least one chunk of transformations at a particular
                // granularity, we set this flag so that we do not yet decrease
                // granularity.

    // We go through the transformations in reverse, in chunks of size
    // |granularity|. This is encoded here via an index, |index|, recording how
    // many chunks of size |granularity| have been tried. The loop exits early
    // if we reach the shrinking step limit.
    for (uint32_t index = 0;
         attempt < impl_->step_limit &&
         index * granularity <
             NumRemainingTransformations(current_best_transformations);) {
      // Remove a chunk of transformations according to the current index and
      // granularity.
      auto transformations_with_chunk_removed =
          RemoveChunk(current_best_transformations, index, granularity);

      // Replay the smaller sequence of transformations to get a next binary and
      // transformation sequence. Note that the transformations arising from
      // replay might be even smaller than the transformations with the chunk
      // removed, because removing those transformaitons might make further
      // transformations inapplicable.
      std::vector<uint32_t> next_binary;
      protobufs::TransformationSequence next_transformation_sequence;
      if (Replayer(impl_->target_env)
              .Run(binary_in, initial_facts, transformations_with_chunk_removed,
                   &next_binary, &next_transformation_sequence) !=
          Replayer::ReplayerResultStatus::kComplete) {
        // Replay should not fail; if it does, we need to abort shrinking.
        return ShrinkerResultStatus::kReplayFailed;
      }

      if (interestingness_function(next_binary, attempt)) {
        // If the binary arising from the smaller transformation sequence is
        // interesting, this becomes our current best binary and transformation
        // sequence.
        current_best_binary = next_binary;
        current_best_transformations = next_transformation_sequence;
        progress_this_round = true;
      } else {
        // Otherwise, increase index so that we try another chunk.  Note that we
        // only increase index when the binary was not interesting because
        // otherwise removal of the chunk means that the current index refers to
        // a fresh chunk to be considered.
        index++;
      }
      // Either way, this was a shrink attempt, so increment our count of shrink
      // attempts.
      attempt++;
    }
    if (!progress_this_round) {
      // If we didn't manage to remove any chunks at this granularity, try a
      // smaller granularity.
      granularity = granularity / 2;
    }
  }

  // The output from the shrinker is the best binary we saw, and the
  // transformations that led to it.
  *binary_out = current_best_binary;
  *transformation_sequence_out = current_best_transformations;

  // Indicate whether shrinking completed or was truncated due to reaching the
  // step limit.
  assert(attempt <= impl_->step_limit);
  if (attempt == impl_->step_limit) {
    std::stringstream strstream;
    strstream << "Shrinking did not complete; step limit " << impl_->step_limit
              << " was reached.";
    impl_->consumer(SPV_MSG_WARNING, nullptr, {}, strstream.str().c_str());
    return Shrinker::ShrinkerResultStatus::kStepLimitReached;
  }
  return Shrinker::ShrinkerResultStatus::kComplete;
}

}  // namespace fuzz
}  // namespace spvtools
