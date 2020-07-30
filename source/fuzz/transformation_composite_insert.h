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

#ifndef SPIRV_TOOLS_TRANSFORMATION_COMPOSITE_INSERT_H
#define SPIRV_TOOLS_TRANSFORMATION_COMPOSITE_INSERT_H

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationCompositeInsert : public Transformation {
 public:
  explicit TransformationCompositeInsert(
      const protobufs::TransformationCompositeInsert& message);

  TransformationCompositeInsert(
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      uint32_t fresh_id, uint32_t composite_id, std::vector<uint32_t>&& index);

  // - |message_.fresh_id| must be fresh.
  // - |message_.composite_id| must refer to an existing composite value.
  // - |message_.index| must refer to a valid index in the composite.
  // - The type id of the object and the type id of the component of the
  // composite
  //   at index |message_.index| must be the same.
  // - |message_.instruction_to_insert_before| must refer to a valid
  //   instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a instruction OpCompositeInsert which creates a new composite from an
  // existing composite, with an element inserted.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationCompositeInsert message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_COMPOSITE_INSERT_H
