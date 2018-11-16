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

namespace spvtools {
namespace reduce {

using namespace opt;

class ChangeOperandReductionOpportunity : public ReductionOpportunity {
 public:
  ChangeOperandReductionOpportunity(Instruction* inst, uint32_t operand_index,
                                    uint32_t original_id, uint32_t new_id)
      : inst_(inst),
        operand_index_(operand_index),
        original_id_(original_id),
        new_id_(new_id) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  Instruction* inst_;
  uint32_t operand_index_;
  uint32_t original_id_;
  uint32_t new_id_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_CHANGE_OPERAND_REDUCTION_OPPORTUNITY_H_
