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

#ifndef SOURCE_REDUCE_REDUCTION_UTIL_H_
#define SOURCE_REDUCE_REDUCTION_UTIL_H_

#include "spirv-tools/libspirv.hpp"

#include "reduction_opportunity.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

// Checks whether the global value list has an OpUndef of the given type,
// adding one if not, and returns the id of such an OpUndef.
uint32_t FindOrCreateGlobalUndef(opt::IRContext* context, uint32_t type_id);

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_UTIL_H_
