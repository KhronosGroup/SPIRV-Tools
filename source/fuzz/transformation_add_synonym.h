// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddSynonym : public Transformation {
 public:
  explicit TransformationAddSynonym(
      protobufs::TransformationAddSynonym message);

  TransformationAddSynonym(
      uint32_t result_id, const protobufs::InstructionDescriptor& insert_before,
      const protobufs::Instruction& synonymous_instruction);

  // - |result_id| must be a valid result id of some instruction in the module.
  // - |insert_before| must be a valid instruction descriptor and we must be
  //   able to insert |synonymous_instruction| before |insert_before|.
  // - |synonymous_instruction| must be a valid instruction.
  // - |synonymous_instruction| must have a fresh result id.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Inserts |synonymous_instruction| before |insert_before| and creates a fact
  // that the result id of |synonymous_instruction| and the |result_id| are
  // synonymous.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationAddSynonym message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_SYNONYM_H_
