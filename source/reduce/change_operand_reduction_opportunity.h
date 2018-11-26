// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_REDUCE_CHANGE_OPERAND_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_CHANGE_OPERAND_REDUCTION_OPPORTUNITY_H_

#include "reduction_opportunity.h"
#include "source/opt/instruction.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace reduce {

using namespace opt;

// Captures the opportunity to change an id operand of an instruction to some
// other id.
class ChangeOperandReductionOpportunity : public ReductionOpportunity {
 public:
  // Constructs the opportunity to replace operand |operand_index| of |inst|
  // with |new_id|.
  ChangeOperandReductionOpportunity(Instruction* inst, uint32_t operand_index,
                                    uint32_t new_id)
      : inst_(inst),
        operand_index_(operand_index),
        original_id_(inst->GetOperand(operand_index).words[0]),
        original_type_(inst->GetOperand(operand_index).type),
        new_id_(new_id) {}

  // Determines whether the opportunity can be applied; it may have been viable
  // when discovered but later disabled by the application of some other
  // reduction opportunity.
  bool PreconditionHolds() override;

 protected:
  // Apply the change of operand.
  void Apply() override;

 private:
  Instruction* const inst_;
  const uint32_t operand_index_;
  const uint32_t original_id_;
  const spv_operand_type_t original_type_;
  const uint32_t new_id_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CHANGE_OPERAND_REDUCTION_OPPORTUNITY_H_
