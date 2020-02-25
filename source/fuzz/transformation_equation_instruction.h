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

#ifndef SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_

#include <vector>

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationEquationInstruction : public Transformation {
 public:
  explicit TransformationEquationInstruction(
      const protobufs::TransformationEquationInstruction& message);

  TransformationEquationInstruction(
      uint32_t fresh_id, SpvOp opcode,
      const std::vector<uint32_t>& in_operand_id,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // TODO comment
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // TODO comment
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // TODO comment
  uint32_t MaybeGetResultType(opt::IRContext* context) const;

  protobufs::TransformationEquationInstruction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_EQUATION_INSTRUCTION_H_
