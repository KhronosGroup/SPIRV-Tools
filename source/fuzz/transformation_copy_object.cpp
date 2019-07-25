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

#include "source/fuzz/transformation_copy_object.h"

namespace spvtools {
namespace fuzz {

TransformationCopyObject::TransformationCopyObject(
    const protobufs::TransformationCopyObject& message)
    : message_(message) {}

TransformationCopyObject::TransformationCopyObject(uint32_t fresh_id,
                                                   uint32_t object,
                                                   uint32_t insert_after_id,
                                                   uint32_t offset) {
  message_.set_fresh_id(fresh_id);
  message_.set_object(object);
  message_.set_insert_after_id(insert_after_id);
  message_.set_offset(offset);
}

bool TransformationCopyObject::IsApplicable(
    opt::IRContext* context, const FactManager& fact_manager) const {
  (void)(context);
  (void)(fact_manager);
  assert(0 && "Not yet implemented.");
  return false;
}

void TransformationCopyObject::Apply(opt::IRContext* context,
                                     FactManager* fact_manager) const {
  (void)(context);
  (void)(fact_manager);
  assert(0 && "Not yet implemented.");
}

protobufs::Transformation TransformationCopyObject::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_copy_object() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
