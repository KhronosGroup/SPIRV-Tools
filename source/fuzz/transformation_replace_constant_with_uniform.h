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

namespace transformation {

bool IsApplicable(
    const protobufs::TransformationReplaceConstantWithUniform& message,
    opt::IRContext* context, const FactManager& fact_manager);
void Apply(const protobufs::TransformationReplaceConstantWithUniform& message,
           opt::IRContext* context, FactManager* fact_manager);

protobufs::TransformationReplaceConstantWithUniform
MakeTransformationReplaceConstantWithUniform(
    protobufs::IdUseDescriptor id_use,
    protobufs::UniformBufferElementDescriptor uniform_descriptor,
    uint32_t fresh_id_for_access_chain, uint32_t fresh_id_for_load);

}  // namespace transformation

// TODO.
class TransformationReplaceConstantWithUniform : public Transformation {
 public:
  // Constructs a transformation from a protobuf message.
  explicit TransformationReplaceConstantWithUniform(
      const protobufs::TransformationReplaceConstantWithUniform& message)
      : message_(message) {}

  ~TransformationReplaceConstantWithUniform() override = default;

  // TODO
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) override {
    return transformation::IsApplicable(message_, context, fact_manager);
  }

  // TODO
  void Apply(opt::IRContext* context, FactManager* fact_manager) override {
    return transformation::Apply(message_, context, fact_manager);
  }

  protobufs::Transformation ToMessage() override;

 private:
  const protobufs::TransformationReplaceConstantWithUniform message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_CONSTANT_WITH_UNIFORM_H_
