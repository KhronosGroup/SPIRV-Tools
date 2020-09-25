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

#ifndef SOURCE_OPT_PROPAGATE_DEBUGVALUE_H_
#define SOURCE_OPT_PROPAGATE_DEBUGVALUE_H_

#include "source/opt/mem_pass.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class PropagateDebugvalue : public MemPass {
 public:
  PropagateDebugvalue() = default;

  const char* name() const override { return "propagate-debugvalue"; }

  Status Process() override;

 private:
  // Adds DebugValue instructions from the immediate dominator basic block
  // for each basic block.
  Status PropagateDebugvalueForFunction(Function* fp);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PROPAGATE_DEBUGVALUE_H_
