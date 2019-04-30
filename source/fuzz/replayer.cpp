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

#include "source/fuzz/replayer.h"
#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation_add_boolean_constant.h"
#include "source/fuzz/transformation_add_dead_break.h"
#include "source/fuzz/transformation_move_block_down.h"
#include "source/fuzz/transformation_split_block.h"
#include "source/opt/build_module.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

struct Replayer::Impl {
  explicit Impl(spv_target_env env) : target_env(env) {}

  const spv_target_env target_env;  // Target environment.
  MessageConsumer consumer;         // Message consumer.
};

Replayer::Replayer(spv_target_env env) : impl_(MakeUnique<Impl>(env)) {}

Replayer::~Replayer() = default;

void Replayer::SetMessageConsumer(MessageConsumer c) {
  impl_->consumer = std::move(c);
}

Replayer::ReplayerResultStatus Replayer::Run(
    const std::vector<uint32_t>& binary_in,
    const protobufs::TransformationSequence& transformation_sequence_in,
    std::vector<uint32_t>* binary_out,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  spvtools::SpirvTools tools(impl_->target_env);
  assert(tools.IsValid() && "Failed to create SPIRV-Tools interface");

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size())) {
    impl_->consumer(SPV_MSG_INFO, nullptr, {},
                    "Initial binary is invalid; stopping.");
    return Replayer::ReplayerResultStatus::kInitialBinaryInvalid;
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context = BuildModule(
      impl_->target_env, impl_->consumer, binary_in.data(), binary_in.size());
  assert(ir_context);

  // An empty fact manager.
  // TODO: settle on a way to provide initial facts.
  FactManager fact_manager;

  // Consider the transformation proto messages in turn.
  for (auto& transformation_message :
       transformation_sequence_in.transformations()) {
    // Check whether the transformation can be applied.
    if (transformation::IsApplicable(transformation_message, ir_context.get(),
                                     fact_manager)) {
      // The transformation is applicable, so apply it, and copy it to the
      // sequence of transformations that were applied.
      transformation::Apply(transformation_message, ir_context.get(),
                            &fact_manager);
      *transformation_sequence_out->add_transformations() =
          transformation_message;
    }
  }

  // Write out the module as a binary.
  ir_context->module()->ToBinary(binary_out, false);
  return Replayer::ReplayerResultStatus::kComplete;
}

}  // namespace fuzz
}  // namespace spvtools
