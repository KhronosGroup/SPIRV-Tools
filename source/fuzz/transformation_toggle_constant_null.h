// Copyright (c) 2020 Stefano Milizia
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_TOGGLE_NULL_CONSTANT_H_
#define SOURCE_FUZZ_TRANSFORMATION_TOGGLE_NULL_CONSTANT_H_

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationToggleConstantNull : public Transformation {
 public:
  explicit TransformationToggleConstantNull(
      const protobufs::TransformationToggleNullConstant& message);

  TransformationToggleConstantNull(uint32_t constant_id);

  // - |message_.instruction_descriptor| must identify an existing scalar
  //   constant (one of OpConstant, OpConstantFalse, OpConstantNull).
  // - The constant or null constant must be of type integer, boolean or
  //   floating-point number.
  // - If the instruction is OpConstant, it must have literal value zero.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationToggleNullConstant message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_TOGGLE_NULL_CONSTANT_H_
