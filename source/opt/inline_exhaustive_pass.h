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

#ifndef SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_
#define SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "source/opt/def_use_manager.h"
#include "source/opt/inline_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InlineExhaustivePass : public InlinePass {
 public:
  InlineExhaustivePass();
  Status Process() override;

  const char* name() const override { return "inline-entry-points-exhaustive"; }

 protected:
  // Substitute a variable for an access chain function argument. See FindAndReplaceAccessChains.
  Pass::Status PassAccessChainByVariable(Function* func, BasicBlock::iterator call_inst_itr);

  // Make a load instruction;
  std::unique_ptr<Instruction> MakeLoad(uint32_t result_type_id, uint32_t result_id,
    uint32_t pointer_id, Instruction const* debug_line_inst, DebugScope const& debug_scope,
    BasicBlock* basic_block);

  std::unique_ptr<Instruction> MakeStore(uint32_t pointer_id, uint32_t object_id,
    Instruction const* debug_line_inst, DebugScope const& debug_scope, BasicBlock* basic_block);

 private:
  // Exhaustively inline all function calls in func as well as in
  // all code that is inlined into func. Returns the status.
  Status InlineExhaustive(Function* func);

  // Substitute variables for access chain function arguments. For each function argument
  // that is an access chain, create a new variable, copy the access chain pointee into the
  // variable before the function call, substitute the variable in the function call,
  // and copy the variable back into the access chain pointee after the function call.
  //
  // This is a workaround for NonSemantic.Shader.DebugInfo.100. DebugDeclare expects the
  // operand variable to be a result id of an OpVariable or OpFunctionParameter. However,
  // function arguments may contain OpAccessChains during legalization which causes problems
  // for DebugDeclare.
  Pass::Status FindAndReplaceAccessChains(Function* func);

  void Initialize();
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INLINE_EXHAUSTIVE_PASS_H_
