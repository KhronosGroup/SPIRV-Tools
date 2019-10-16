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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetMemoryOperandsMask : public Transformation {
 public:
  explicit TransformationSetMemoryOperandsMask(
      const protobufs::TransformationSetMemoryOperandsMask& message);

  TransformationSetMemoryOperandsMask(
      const protobufs::InstructionDescriptor& memory_access_instruction,
      uint32_t memory_operands_mask, uint32_t memory_operands_mask_index);

  // TODO comment
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // TODO comment
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

  // TODO comment
  static bool IsMemoryAccess(const opt::Instruction& instruction);

  // Does the version of SPIR-V being used support multiple memory operand
  // masks on relevant memory access instructions?
  static bool MultipleMemoryOperandMasksAreSupported(opt::IRContext* context);

 private:
  // TODO comment
  uint32_t GetOriginalMaskInOperandIndex(
      const opt::Instruction& instruction) const;

  // TODO comment
  bool NewMaskIsValid(const opt::Instruction& instruction,
                      uint32_t original_mask_in_operand_index) const;

  protobufs::TransformationSetMemoryOperandsMask message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_MEMORY_OPERANDS_MASK_H_
