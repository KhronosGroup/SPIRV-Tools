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

#include "transformation_replace_load_store_with_copy_memory.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        const spvtools::fuzz::protobufs::
            TransformationReplaceLoadStoreWithCopyMemory& message)
    : message_(message) {}

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        const protobufs::InstructionDescriptor& load_descriptor,
        const protobufs::InstructionDescriptor& store_descriptor) {
  *message_.mutable_load_descriptor() = load_descriptor;
  *message_.mutable_store_descriptor() = store_descriptor;
}
bool TransformationReplaceLoadStoreWithCopyMemory::IsApplicable(
    opt::IRContext*, const TransformationContext& /*unused*/) const {
  return false;
}

void TransformationReplaceLoadStoreWithCopyMemory::Apply(
    opt::IRContext*, TransformationContext* /*unused*/) const {}

}  // namespace fuzz
}  // namespace spvtools