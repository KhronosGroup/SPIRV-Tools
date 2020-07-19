// Copyright (c) 2020 Google
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

#include "source/fuzz/transformation_replace_copy_object_with_store_load.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationReplaceCopyObjectWithStoreLoad::
    TransformationReplaceCopyObjectWithStoreLoad(
        const spvtools::fuzz::protobufs::
            TransformationReplaceCopyObjectWithStoreLoad& message)
    : message_(message) {}

TransformationReplaceCopyObjectWithStoreLoad::
    TransformationReplaceCopyObjectWithStoreLoad(
        uint32_t copy_object_result_id, uint32_t fresh_variable_id,
        uint32_t variable_storage_class, uint32_t variable_initializer_id) {
  message_.set_copy_object_result_id(copy_object_result_id);
  message_.set_fresh_variable_id(fresh_variable_id);
  message_.set_variable_storage_class(variable_storage_class);
  message_.set_variable_storage_class(variable_storage_class);
  message_.set_variable_initializer_id(variable_initializer_id);
}

bool TransformationReplaceCopyObjectWithStoreLoad::IsApplicable(
    opt::IRContext* /*unused*/, const TransformationContext& /*unused*/) const {
  return false;
}

void TransformationReplaceCopyObjectWithStoreLoad::Apply(
    opt::IRContext*, TransformationContext*) const {}

protobufs::Transformation
TransformationReplaceCopyObjectWithStoreLoad::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_copy_object_with_store_load() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
