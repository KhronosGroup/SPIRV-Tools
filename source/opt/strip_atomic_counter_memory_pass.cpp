// Copyright (c) 2019 Google Inc.
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

#include "source/opt/strip_atomic_counter_memory_pass.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status StripAtomicCounterMemoryPass::Process() {
  fprintf(stderr, "Ran StripAtomicCounterMemoryPass::Process()\n");
  return Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
