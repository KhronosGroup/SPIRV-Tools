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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_CONSTANT_WITH_UNIFORM_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_CONSTANT_WITH_UNIFORM_H_

#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

// TODO.
class TransformationReplaceConstantWithUniform : public Transformation {
 public:
  // TODO.
  TransformationReplaceConstantWithUniform(
      module_navigation::IdUseDescriptor id_use_descriptor,
      FactManager::UniformBufferElementDescriptor uniform_descriptor,
      uint32_t fresh_id_for_access_chain, uint32_t fresh_id_for_load)
      : id_use_descriptor_(id_use_descriptor),
        uniform_descriptor_(uniform_descriptor),
        fresh_id_for_access_chain_(fresh_id_for_access_chain),
        fresh_id_for_load_(fresh_id_for_load) {
    assert(fresh_id_for_access_chain_ != fresh_id_for_load_ &&
           "Fresh ids for access chain and load result cannot be the same.");
  }

  // Constructs a transformation from a protobuf message.
  explicit TransformationReplaceConstantWithUniform(
      const protobufs::TransformationReplaceConstantWithUniform& message);

  ~TransformationReplaceConstantWithUniform() override = default;

  // TODO
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) override;

  // TODO
  void Apply(opt::IRContext* context, FactManager* fact_manager) override;

  protobufs::Transformation ToMessage() override;

 private:
  // A descriptor for the id we would like to replace
  const module_navigation::IdUseDescriptor id_use_descriptor_;

  // Uniform descriptor to identify which uniform value to choose.
  const FactManager::UniformBufferElementDescriptor uniform_descriptor_;

  const uint32_t fresh_id_for_access_chain_;

  const uint32_t fresh_id_for_load_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_CONSTANT_WITH_UNIFORM_H_
