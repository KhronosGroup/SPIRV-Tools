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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_VECTOR_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_VECTOR_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddTypeVector : public Transformation {
 public:
  explicit TransformationAddTypeVector(
      const protobufs::TransformationAddTypeVector& message);

  TransformationAddTypeVector(uint32_t fresh_id, uint32_t base_type_id,
                              uint32_t size);

  // - |message_.fresh_id| must be a fresh id
  // - |message_.component_type_id| must be the id of a scalar type
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // Adds an OpTypeVector instruction to the module, with component type
  // |message_.component_type_id| and |message_.component_count| components,
  // with result if |message_.fresh_id|.
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddTypeVector message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_TYPE_VECTOR_H_
