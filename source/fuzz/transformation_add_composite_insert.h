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

#ifndef SPIRV_TOOLS_TRANSFORMATION_ADD_COMPOSITE_INSERT_H
#define SPIRV_TOOLS_TRANSFORMATION_ADD_COMPOSITE_INSERT_H

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddCompositeInsert : public Transformation {
 public:
  explicit TransformationAddCompositeInsert(
      const protobufs::TransformationAddCompositeInsert& message);

  TransformationAddCompositeInsert(uint32_t fresh_id,
                                   uint32_t available_constant_id,
                                   uint32_t composite_value_id,
                                   uint32_t index_to_replace,
                                   const protobufs::InstructionDescriptor&
                                       instruction_descriptor_insert_before);

  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddCompositeInsert message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_ADD_COMPOSITE_INSERT_H
