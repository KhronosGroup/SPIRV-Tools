// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddCopyMemory : public Transformation {
 public:
  explicit TransformationAddCopyMemory(
      const protobufs::TransformationAddCopyMemory& message);

  TransformationAddCopyMemory(
      const protobufs::InstructionDescriptor& instruction_descriptor,
      uint32_t target_id, uint32_t source_id);

  // - |instruction_descriptor| must point to a valid instruction in the module.
  // - it should be possible to insert OpCopyMemory before |instruction_descriptor|
  // (i.e. the module remains valid after the insertion).
  // - |target_id| and |source_id| must be result ids for some valid instructions
  // in the module.
  // - types of |target_id| and |source_id| must be OpTypePointer and have the same
  // pointee type.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // An 'OpCopyMemory %target_id %source_id' instruction is inserted before the
  // |instruction_descriptor|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddCopyMemory message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_COPY_MEMORY_H_
