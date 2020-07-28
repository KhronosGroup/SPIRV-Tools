// Copyright (c) 2020 André Perez Maselco
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_VECTOR_DYNAMIC_H_
#define SOURCE_FUZZ_TRANSFORMATION_VECTOR_DYNAMIC_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationVectorDynamic : public Transformation {
 public:
  explicit TransformationVectorDynamic(
      const protobufs::TransformationVectorDynamic& message);

  TransformationVectorDynamic(uint32_t instruction_result_id);

  // - |message_.instruction_result_id| must be an instruction result id of an
  // OpCompositeExtract or OpCompositeInsert instruction.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Replaces the OpCompositeExtract and OpCompositeInsert instructions with the
  // OpVectorExtractDynamic and OpVectorInsertDynamic instructions.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Checks |instruction| is defined, is an OpCompositeExtract/Insert
  // instruction and the composite operand is a vector.
  static bool IsVectorOperation(opt::IRContext* ir_context,
                                opt::Instruction* instruction);

  // Returns the result id of an OpConstant instruction of integral type.
  // First it looks for an unsigned version of the constant, if it doesn't
  // exist, it looks for a signed version. Returns 0 if no such instruction is
  // present in the module.
  static uint32_t MaybeGetConstantForIndex(
      opt::IRContext* ir_context, const opt::Instruction& instruction,
      const TransformationContext& transformation_context);

 private:
  protobufs::TransformationVectorDynamic message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_VECTOR_DYNAMIC_H_
