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

class TransformationToggleNullConstant : public Transformation {
 public:
  explicit TransformationToggleNullConstant(
      const protobufs::TransformationToggleNullConstant& message);

  TransformationToggleNullConstant(
      protobufs::InstructionDescriptor& instruction_descriptor);

  // - |message_.instruction_descriptor| must identify an existing scalar
  //   constant (one of OpConstant, OpConstantFalse, OpConstantNull).
  // - The constant or null constant must be of type integer, boolean or
  //   floating-point number.
  // - If the instruction is OpConstant, it must have literal value zero.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

 private:
  protobufs::TransformationToggleNullConstant message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_TOGGLE_NULL_CONSTANT_H_
