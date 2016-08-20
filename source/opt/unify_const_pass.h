// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_UNIFY_CONSTANT_PASS_H_
#define LIBSPIRV_OPT_UNIFY_CONSTANT_PASS_H_

#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// The optimization pass to de-duplicate the constants. Constants with exactly
// same values and identical form will be unified and only one constant will be
// kept for each unique pair of type and value.
// There are several cases not handled by this pass:
//  1) Constants defined by OpConstantNull instructions (null constants) and
//  constants defined by OpConstantFalse, OpConstant or OpConstantComposite
//  with value(s) 0 (zero-valued normal constants) are not considered
//  equivalent. So null constants won't be used to replace zero-valued normal
//  constants, and other constants won't replace the null constants either.
//  2) Whenever there are decorations to the constant's result id or its type
//  id, the constants won't be handled, which means, it won't be used to
//  replace any other constants, neither can other constants replace it.
//  3) NaN in float point format with different bit patterns are not unified.
class UnifyConstantPass : public Pass {
 public:
  explicit UnifyConstantPass(const MessageConsumer& c) : Pass(c) {}

  const char* name() const override { return "unify-const"; }
  bool Process(ir::Module*) override;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_UNIFY_CONSTANT_PASS_H_
