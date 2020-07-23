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

#ifndef SPIRV_TOOLS_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H
#define SPIRV_TOOLS_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceLoadStoreWithCopyMemory : public Transformation {
 public:
  explicit TransformationReplaceLoadStoreWithCopyMemory(
      const protobufs::TransformationReplaceLoadStoreWithCopyMemory& message);

  TransformationReplaceLoadStoreWithCopyMemory(
      const protobufs::InstructionDescriptor& load_descriptor,
      const protobufs::InstructionDescriptor& store_descriptor);

  // - |message_.load_descriptor| must refer to an OpLoad instruction.
  // - |message_.store_descriptor| must refer to an OpStore instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Takes a pair of instruction descriptors to OpLoad and OpStore that have the
  // same intermediate value and replaces the OpStore with an equivalent
  // OpCopyMemory.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceLoadStoreWithCopyMemory message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SPIRV_TOOLS_TRANSFORMATION_REPLACE_LOAD_STORE_WITH_COPY_MEMORY_H
