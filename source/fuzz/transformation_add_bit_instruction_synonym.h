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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddBitInstructionSynonym : public Transformation {
 public:
  explicit TransformationAddBitInstructionSynonym(
      const protobufs::TransformationAddBitInstructionSynonym& message);

  TransformationAddBitInstructionSynonym(
      const uint32_t instruction_result_id,
      const std::vector<uint32_t>& fresh_ids);

  // - |message_.instruction_result_id| must be a bit instruction.
  // - |message_.fresh_ids| must be fresh ids needed to apply the
  // transformation.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a bit instruction synonym.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the number of fresh ids required to apply the transformation.
  static uint32_t GetRequiredFreshIdCount(opt::IRContext* ir_context,
                                          opt::Instruction* bit_instruction);

 private:
  protobufs::TransformationAddBitInstructionSynonym message_;

  // Adds an OpBitwise* synonym.
  void AddBitwiseSynonym(opt::IRContext* ir_context,
                         TransformationContext* transformation_context,
                         opt::Instruction* bitwise_instruction) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_
