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

#include "eliminate_dead_functions_util.h"

namespace spvtools {
namespace opt {

namespace eliminatedeadfunctionsutil {

Module::iterator EliminateFunction(IRContext* context,
                                   Module::iterator* func_iter) {
  bool first_func = *func_iter == context->module()->begin();
  bool seen_func_end = false;
  (*func_iter)
      ->ForEachInst(
          [context, first_func, func_iter, &seen_func_end](Instruction* inst) {
            if (inst->opcode() == SpvOpFunctionEnd) {
              seen_func_end = true;
            }
            // Move non-semantic instructions to the previous function or
            // global values if this is the first function.
            if (seen_func_end && inst->opcode() == SpvOpExtInst) {
              std::unique_ptr<Instruction> clone(inst->Clone(context));
              if (first_func) {
                context->AddGlobalValue(std::move(clone));
              } else {
                auto prev_func_iter = *func_iter;
                --prev_func_iter;
                prev_func_iter->AddNonSemanticInstruction(std::move(clone));
              }
              inst->ToNop();
            } else {
              context->KillNonSemanticInfo(inst);
              context->KillInst(inst);
            }
          },
          true, true);
  return func_iter->Erase();
}

}  // namespace eliminatedeadfunctionsutil
}  // namespace opt
}  // namespace spvtools
