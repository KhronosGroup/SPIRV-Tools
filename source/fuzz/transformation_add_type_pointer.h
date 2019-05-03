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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_POINTER_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_POINTER_H_

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {
namespace transformation {

// TODO
bool IsApplicable(const protobufs::TransformationAddTypePointer& message,
                  opt::IRContext* context, const FactManager& fact_manager);

// TODO
void Apply(const protobufs::TransformationAddTypePointer& message,
           opt::IRContext* context, FactManager* fact_manager);

// Helper factory to create a transformation message.
protobufs::TransformationAddTypePointer MakeTransformationAddTypePointer(
    uint32_t fresh_id, SpvStorageClass storage_class, uint32_t base_type_id);

}  // namespace transformation
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_POINTER_H_
