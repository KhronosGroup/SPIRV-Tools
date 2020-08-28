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

#include "source/fuzz/fuzzer_pass.h"
#include "source/fuzz/fuzzer_util.h"
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

  // Constructs a fuzzer from the given target environment |target_env|.  |seed|
  // is a seed for pseudo-random number generation.
  // |validate_after_each_fuzzer_pass| controls whether the validator will be
  // invoked after every fuzzer pass is applied.
  Fuzzer(spv_target_env target_env, uint32_t seed,
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
      protobufs::TransformationSequence* transformation_sequence_out) const;

 private:
  // A *pass recommendation* is a fuzzer pass together with an integer *age*
  // indicating how long ago this fuzzer pass was recommended.  Each time some
  // fuzzer pass is applied, the age of all recommended passes increases by
  // one.  A recommended pass gets removed either because it has been acted on
  // (i.e., the pass has been applied), or because the recommendation gets too
  // old (i.e., the age component reaches some limit).
  using RecommendedPasses = std::vector<std::pair<FuzzerPass*, uint32_t>>;

  // This type is used to record a single instance of every fuzzer pass that
  // is enabled and that can be applied repeatedly.
  struct PassInstances;

  // This method should be invoked right after |completed_pass| has finished.
  // The age of all passes in |recommended_passes| is incremented, with passes
  // that get too old being removed.
  //
  // Zero or more new recommendations from |pass_instances| are made, depending
  // on which pass |completed_pass| was.  For example, if |completed_pass| is a
  // pass that donates new functions into the module, a fuzzer pass that creates
  // function call instructions might be recommended.
  //
  // New recommendations are either added to |recommended_passes|, if not
  // already present, or lead to the age of existing recommendations being
  // decreased.
  static void UpdateRecommendedPasses(FuzzerPass* completed_pass,
                                      const PassInstances& pass_instances,
                                      RecommendedPasses* recommended_passes);

  // Helper method used by UpdateRecommendedPasses.  A no-op if |pass| is
  // |nullptr|.
  //
  // If |pass| is non-null then a recommendation of |pass| with age 0 is added
  // to |recommended_passes| if there is no existing recommendation |pass|;
  // otherwise the existing recommendation has its age reset.
  static void RecommendPass(FuzzerPass* pass,
                            RecommendedPasses* recommended_passes);

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

  // Determines whether the validator should be invoked after every fuzzer pass.
  bool validate_after_each_fuzzer_pass_;

  // Options to control validation.
  spv_validator_options validator_options_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_H_
