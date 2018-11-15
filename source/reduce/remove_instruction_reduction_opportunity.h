// Copyright (c) 2018 Google Inc.
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

#ifndef SOURCE_REDUCE_REMOVE_INSTRUCTION_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REMOVE_INSTRUCTION_REDUCTION_OPPORTUNITY_H_

#include "reduction_opportunity.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace reduce {

using namespace opt;

class RemoveInstructionReductionOpportunity : public ReductionOpportunity {
 public:
  explicit RemoveInstructionReductionOpportunity(Instruction* inst)
      : inst_(inst) {}

  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  Instruction* inst_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REMOVE_INSTRUCTION_REDUCTION_OPPORTUNITY_H_
