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

#ifndef SOURCE_FUZZ_FUZZER_H_
#define SOURCE_FUZZ_FUZZER_H_

#include <memory>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/pass_management/repeated_pass_instances.h"
#include "source/fuzz/pass_management/repeated_pass_recommender.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Transforms a SPIR-V module into a semantically equivalent SPIR-V module by
// running a number of randomized fuzzer passes.
class Fuzzer {
 public:
  // Possible statuses that can result from running the fuzzer.
  enum class FuzzerResultStatus {
    kComplete,
    kFailedToCreateSpirvToolsInterface,
    kFuzzerPassLedToInvalidModule,
    kInitialBinaryInvalid,
  };

  // Each field of this enum corresponds to an available repeated pass
  // strategy, and is used to decide which kind of RepeatedPassManager object
  // to create.
  enum class RepeatedPassStrategy {
    kSimple,
    kRandomWithRecommendations,
    kLoopedWithRecommendations
  };

  // Constructs a fuzzer from the given target environment |target_env|.  |seed|
  // is a seed for pseudo-random number generation.  If |enable_all_passes| is
  // true then all fuzzer passes will be enabled, otherwise a random subset of
  // fuzzer passes will be enabled.  |validate_after_each_fuzzer_pass| controls
  // whether the validator will be invoked after every fuzzer pass is applied,
  // and |validator_options| provides the options that should be used during
  // validation if so.
  Fuzzer(spv_target_env target_env, uint32_t seed, bool enable_all_passes,
         RepeatedPassStrategy repeated_pass_strategy,
         bool validate_after_each_fuzzer_pass,
         spv_validator_options validator_options);

  // Disables copy/move constructor/assignment operations.
  Fuzzer(const Fuzzer&) = delete;
  Fuzzer(Fuzzer&&) = delete;
  Fuzzer& operator=(const Fuzzer&) = delete;
  Fuzzer& operator=(Fuzzer&&) = delete;

  ~Fuzzer();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Transforms |binary_in| to |binary_out| by running a number of randomized
  // fuzzer passes.  Initial facts about the input binary and the context in
  // which it will execute are provided via |initial_facts|.  A source of donor
  // modules to be used by transformations is provided via |donor_suppliers|.
  // The transformation sequence that was applied is returned via
  // |transformation_sequence_out|.
  FuzzerResultStatus Run(
      const std::vector<uint32_t>& binary_in,
      const protobufs::FactSequence& initial_facts,
      const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers,
      std::vector<uint32_t>* binary_out,
      protobufs::TransformationSequence* transformation_sequence_out);

 private:
  // A convenience method to add a repeated fuzzer pass to |pass_instances| with
  // probability 0.5, or with probability 1 if |enable_all_passes_| is true.
  //
  // All fuzzer passes take |ir_context|, |transformation_context|,
  // |fuzzer_context| and |transformation_sequence_out| as parameters.  Extra
  // arguments can be provided via |extra_args|.
  template <typename FuzzerPassT, typename... Args>
  void MaybeAddRepeatedPass(
      RepeatedPassInstances* pass_instances, opt::IRContext* ir_context,
      TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformation_sequence_out,
      Args&&... extra_args) const;

  // A convenience method to add a final fuzzer pass to |passes| with
  // probability 0.5, or with probability 1 if |enable_all_passes_| is true.
  //
  // All fuzzer passes take |ir_context|, |transformation_context|,
  // |fuzzer_context| and |transformation_sequence_out| as parameters.  Extra
  // arguments can be provided via |extra_args|.
  template <typename FuzzerPassT, typename... Args>
  void MaybeAddFinalPass(
      std::vector<std::unique_ptr<FuzzerPass>>* passes,
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformation_sequence_out,
      Args&&... extra_args) const;

  // Decides whether to apply more repeated passes. The probability decreases as
  // the number of transformations that have been applied increases.
  bool ShouldContinueFuzzing(
      const protobufs::TransformationSequence& transformation_sequence_out,
      FuzzerContext* fuzzer_context);

  // Applies |pass|, which must be a pass constructed with |ir_context|, and
  // then returns true if and only if |ir_context| is valid.  |tools| is used to
  // check validity.
  bool ApplyPassAndCheckValidity(FuzzerPass* pass,
                                 const opt::IRContext& ir_context,
                                 const spvtools::SpirvTools& tools) const;

  // Target environment.
  const spv_target_env target_env_;

  // Message consumer.
  MessageConsumer consumer_;

  // Seed for random number generator.
  const uint32_t seed_;

  // Determines whether all passes should be enabled, vs. having passes be
  // probabilistically enabled.
  bool enable_all_passes_;

  // Controls which type of RepeatedPassManager object to create.
  RepeatedPassStrategy repeated_pass_strategy_;

  // Determines whether the validator should be invoked after every fuzzer pass.
  bool validate_after_each_fuzzer_pass_;

  // Options to control validation.
  spv_validator_options validator_options_;

  // The number of repeated fuzzer passes that have been applied is kept track
  // of, in order to enforce a hard limit on the number of times such passes
  // can be applied.
  uint32_t num_repeated_passes_applied_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_H_
