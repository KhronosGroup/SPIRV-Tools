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

#include "source/fuzz/fuzzer.h"

#include <cassert>
#include <sstream>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass_add_dead_breaks.h"
#include "source/fuzz/fuzzer_pass_add_useful_constructs.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_split_blocks.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/opt/build_module.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

struct Fuzzer::Impl {
  explicit Impl(spv_target_env env) : target_env(env) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
};

Fuzzer::Fuzzer(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Fuzzer::~Fuzzer() = default;

void Fuzzer::SetMessageConsumer(MessageConsumer c) {
  impl_->consumer = std::move(c);
}

Fuzzer::FuzzerResultStatus Fuzzer::Run(std::vector<uint32_t>&& binary_in,
                                       std::vector<uint32_t>* binary_out,
                                       spv_const_fuzzer_options options) const {
  spvtools::SpirvTools tools(impl_->target_env);
  assert(tools.IsValid() && "Failed to create SPIRV-Tools interface");

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size())) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial binary is invalid; stopping.");
    return Fuzzer::FuzzerResultStatus::kInitialBinaryInvalid;
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context = BuildModule(
      impl_->target_env, impl_->consumer, binary_in.data(), binary_in.size());
  assert(ir_context);

  // Make a PRNG, either from a given seed or from a random device.
  PseudoRandomGenerator random_generator(
      options->has_random_seed ? options->random_seed
                               : (uint32_t)std::random_device()());

  // The fuzzer will introduce new ids into the module.  The module's id bound
  // gives the smallest id that can be used for this purpose.  We add an offset
  // to this so that there is a sizeable gap between the ids used in the
  // original module and the ids used for fuzzing, as a readability aid.
  //
  // TODO(2541) consider the case where the maximum id bound is reached.
  auto minimum_fresh_id = ir_context->module()->id_bound() + 100;
  FuzzerContext fuzzer_context(&random_generator, minimum_fresh_id);

  // This keeps a record of all the transformations that are applied.
  // Currently we do not do anything further with it, but in due course it will
  // be serialized along with the generated binary, for use in test case
  // reduction.
  std::vector<std::unique_ptr<Transformation>> transformations_applied;

  // Add some essential ingredients to the module if they are not already
  // present, such as boolean constants.
  FuzzerPassAddUsefulConstructs().Apply(ir_context.get(), &fuzzer_context,
                                        &transformations_applied);

  // Apply some semantics-preserving passes.
  FuzzerPassSplitBlocks().Apply(ir_context.get(), &fuzzer_context,
                                &transformations_applied);
  FuzzerPassAddDeadBreaks().Apply(ir_context.get(), &fuzzer_context,
                                  &transformations_applied);

  // Finally, give the blocks in the module a good shake-up.
  FuzzerPassPermuteBlocks().Apply(ir_context.get(), &fuzzer_context,
                                  &transformations_applied);

  ir_context->module()->ToBinary(binary_out, false);

  return Fuzzer::FuzzerResultStatus::kComplete;
}

}  // namespace fuzz
}  // namespace spvtools
