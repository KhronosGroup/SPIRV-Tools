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

#ifndef SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_

#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace reduce {

// Abstract class capturing an opportunity to apply a reducing transformation.
class ReductionOpportunity {
 public:
  ReductionOpportunity() = default;
  virtual ~ReductionOpportunity() = default;

  // Determines whether the opportunity can be applied; it may have been viable
  // when discovered but later disabled by the application of some other
  // reduction opportunity.
  virtual bool PreconditionHolds() = 0;

  // A no-op if PreconditoinHolds() returns false; otherwise applies the
  // opportunity.
  void TryToApply();

 protected:
  // Apply the reduction opportunity.
  virtual void Apply() = 0;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_
