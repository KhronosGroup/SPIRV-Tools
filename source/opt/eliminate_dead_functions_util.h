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

#ifndef SOURCE_OPT_ELIMINATE_DEAD_FUNCTIONS_UTIL_H_
#define SOURCE_OPT_ELIMINATE_DEAD_FUNCTIONS_UTIL_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

// Provides functionality for eliminating functions that are not needed, for use
// by various analyses and passes.
namespace eliminatedeadfunctionsutil {

// Remove all of the instruction in the function body
void EliminateFunctionInstructions(IRContext* context, Function* func);

}  // namespace eliminatedeadfunctionsutil
}  // namespace opt
}  // namespace spvtools

#endif  //  SOURCE_OPT_ELIMINATE_DEAD_FUNCTIONS_UTIL_H_
