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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {
class TransformationAddLoopToCreateIntConstantSynonym : public Transformation {
 public:
  explicit TransformationAddLoopToCreateIntConstantSynonym(
      const protobufs::TransformationAddLoopToCreateIntConstantSynonym&
          message);

  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddLoopToCreateIntConstantSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools
#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_LOOP_TO_CREATE_INT_CONSTANT_SYNONYM_H_
