// Copyright (c) 2017 Google Inc.
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

#include "validate.h"

#include <string>

#include "diagnostic.h"
#include "opcode.h"
#include "val/validation_state.h"

using libspirv::DiagnosticStream;
using libspirv::ValidationState_t;

namespace libspirv {

// Validates that decorations have been applied properly.
spv_result_t ValidateDecorations(ValidationState_t& vstate) {
  // According the SPIR-V Spec 2.16.1, it is illegal to initialize an imported
  // variable. This means that a module-scope OpVariable with initialization
  // value cannot be marked with the Import Linkage Type (import type id = 1).
  for (auto global_var_id : vstate.global_vars()) {
    // Initializer <id> is an optional argument for OpVariable. If initializer
    // <id> is present, the instruction will have 5 words.
    auto variable_instr = vstate.FindDef(global_var_id);
    if (variable_instr->words().size() == 5u) {
      for (const auto& decoration : vstate.id_decorations(global_var_id)) {
        // the Linkage Type is the last parameter of the decoration.
        if (SpvDecorationLinkageAttributes == decoration.dec_type() &&
            decoration.params().size() >= 2u &&
            decoration.params().back() == 1) {
          return vstate.diag(SPV_ERROR_INVALID_ID)
                 << "A module-scope OpVariable with initialization value "
                    "cannot be marked with the Import Linkage Type.";
        }
      }
    }
  }

  // TODO: Add more decoration validation code here.

  return SPV_SUCCESS;
}

}  // namespace libspirv

