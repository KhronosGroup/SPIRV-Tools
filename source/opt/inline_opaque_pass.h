// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_INLINE_OPAQUE_PASS_H_
#define LIBSPIRV_OPT_INLINE_OPAQUE_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <vector>
#include <unordered_map>

#include "def_use_manager.h"
#include "module.h"
#include "inline_pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlineOpaquePass : public InlinePass {

 public:
  InlineOpaquePass();
  Status Process(ir::Module*) override;

  const char* name() const override { return "inline-exhaustive"; }

 private:
  // Return true if |typeId| is or contains opaque type
  bool IsOpaqueType(uint32_t typeId);

  // Return true if function call |callInst| has opaque argument or return type
  bool HasOpaqueArgsOrReturn(const ir::Instruction* callInst);

  // Inline all function calls in |func| that have opaque params or return
  // type. Inline similarly all code that is inlined into func. Return true
  // if func is modified.
  bool InlineOpaque(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INLINE_OPAQUE_PASS_H_
