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

#ifndef LIBSPIRV_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_
#define LIBSPIRV_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_

#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// The transformation pass that specializes the value of spec constants to
// their default values. This pass only processes the spec constants that have
// Spec ID decorations (defined by OpSpecConstant, OpSpecConstantTrue and
// OpSpecConstantFalse instructions) and replaces them with their front-end
// version counterparts (OpConstant, OpConstantTrue and OpConstantFalse). The
// corresponding Spec ID annotation instructions will also be removed. This
// pass does not fold the newly added front-end constants and does not process
// other spec constants defined by OpSpecConstantComposite or OpSpecConstantOp.
class FreezeSpecConstantValuePass : public Pass {
 public:
  explicit FreezeSpecConstantValuePass(const MessageConsumer& c) : Pass(c) {}

  const char* name() const override { return "freeze-spec-const"; }
  bool Process(ir::Module*) override;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_
