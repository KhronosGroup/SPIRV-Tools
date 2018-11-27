// Copyright (c) 2018 Google Inc.
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

// Validates correctness of ExtInstImport SPIR-V instructions.

#include "source/val/validate.h"

#include <string>
#include <vector>

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ExtInstImportPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  if (opcode != SpvOpExtInstImport) return SPV_SUCCESS;

  if (spvIsWebGPUEnv(_.context()->target_env)) {
    const auto name_id = 1;
    const std::string name(reinterpret_cast<const char*>(
        inst->words().data() + inst->operands()[name_id].offset));
    if (name != "GLSL.std.450") {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "For WebGPU, the only valid parameter to OpExtInstImport is "
                "\"GLSL.std.450\".";
    }
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
