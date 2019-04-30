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

#ifndef SOURCE_FUZZ_ID_USE_LOCATOR_H_
#define SOURCE_FUZZ_ID_USE_LOCATOR_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace module_navigation {

// Describes a use of an id as an input operand to an instruction in some block
// of a function.
class IdUseDescriptor {
 public:
  IdUseDescriptor(uint32_t id_of_interest, uint32_t target_instruction_opcode,
                  uint32_t in_operand_index,
                  uint32_t base_instruction_result_id,
                  uint32_t num_opcodes_to_ignore)
      : id_of_interest_(id_of_interest),
        target_instruction_opcode_(target_instruction_opcode),
        in_operand_index_(in_operand_index),
        base_instruction_result_id_(base_instruction_result_id),
        num_opcodes_to_ignore_(num_opcodes_to_ignore) {}

  ~IdUseDescriptor() = default;

  opt::Instruction* FindInstruction(opt::IRContext* context) const;

  uint32_t GetIdOfInterest() const;

  uint32_t GetInOperandIndex() const;

 private:
  // An id that we would like to be able to find a use of.
  const uint32_t id_of_interest_;

  // The opcode for the instruction that uses the id.
  const uint32_t target_instruction_opcode_;

  // The input operand index at which the use is expected.
  const uint32_t in_operand_index_;

  // The id of an instruction after which the instruction that contains the use
  // is believed to occur; it might be the using instruction itself.
  const uint32_t base_instruction_result_id_;

  // The number of matching opcodes to skip over when searching for the using
  // instruction from the base instruction.
  const uint32_t num_opcodes_to_ignore_;
};

}  // namespace module_navigation
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_ID_USE_LOCATOR_H_
