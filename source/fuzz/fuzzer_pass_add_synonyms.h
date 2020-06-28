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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_SYNONYMS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_SYNONYMS_H_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Sprinkles instructions through the module that produce ids, synonymous to
// some other instructions.
class FuzzerPassAddSynonyms : public FuzzerPass {
 public:
  FuzzerPassAddSynonyms(opt::IRContext* ir_context,
                        TransformationContext* transformation_context,
                        FuzzerContext* fuzzer_context,
                        protobufs::TransformationSequence* transformations);

  ~FuzzerPassAddSynonyms() override;

  void Apply() override;

 private:
  // Applies a transformation to create a multiplication by 1 synonym. |inst|
  // must have a scalar type. |opcode| is an opcode used for multiplication
  // (e.g. OpFMul, OpIMul etc.)
  void CreateScalarMultiplicationSynonym(
      const opt::Instruction* inst,
      const protobufs::InstructionDescriptor& instruction_descriptor,
      SpvOp opcode);

  // Applies a transformation to create an addition of 0 synonym. |inst| must
  // have a scalar type. |opcode| is an opcode used for addition (e.g. OpFAdd,
  // OpIAdd etc.)
  void CreateScalarAdditionSynonym(
      const opt::Instruction* inst,
      const protobufs::InstructionDescriptor& instruction_descriptor,
      SpvOp opcode);

  // Applies a transformation to create a multiplication by 1 synonym. |inst|
  // must be a vector. Depending on the type of the vector's components, either
  // OpFMul, OpIMul or OpLogicalAnd will be applied.
  void CreateVectorMultiplicationSynonym(
      const opt::Instruction* inst,
      const protobufs::InstructionDescriptor& instruction_descriptor);

  // Applies a transformation to create an addition of 0 synonym. |inst| must
  // be a vector. Depending on the type of the vector's components, either
  // OpFAdd, OpIAdd or OpLogicalOr will be applied.
  void CreateVectorAdditionSynonym(
      const opt::Instruction* inst,
      const protobufs::InstructionDescriptor& instruction_descriptor);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_SYNONYMS_H_
