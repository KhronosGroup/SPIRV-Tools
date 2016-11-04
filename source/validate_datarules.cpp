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

// Ensures Data Rules are followed according to the specifications.

#include "validate.h"

#include <cassert>
#include <sstream>
#include <string>

#include "diagnostic.h"
#include "opcode.h"
#include "operand.h"
#include "val/ValidationState.h"

using libspirv::CapabilitySet;
using libspirv::DiagnosticStream;
using libspirv::ValidationState_t;

namespace {

// Validates that the number of components in the vector type is legal.
// Vector types can only be parameterized as having 2, 3, or 4 components.
// If the Vector16 capability is added, 8 and 16 components are also allowed.
spv_result_t ValidateNumVecComponents(ValidationState_t& _,
                                      const spv_parsed_instruction_t* inst) {
  if (inst->opcode == SpvOpTypeVector) {
    // operand 2 specifies the number of components in the vector.
    const uint32_t num_components = inst->words[inst->operands[2].offset];
    if (num_components == 2 || num_components == 3 || num_components == 4) {
      return SPV_SUCCESS;
    }
    if (num_components == 8 || num_components == 16) {
      if (_.HasCapability(SpvCapabilityVector16)) {
        return SPV_SUCCESS;
      } else {
        return _.diag(SPV_ERROR_INVALID_DATA)
               << "Having " << num_components << " components for "
               << spvOpcodeString(static_cast<SpvOp>(inst->opcode))
               << " requires the Vector16 capability";
      }
    }
    return _.diag(SPV_ERROR_INVALID_DATA)
           << "Illegal number of components (" << num_components << ") for "
           << spvOpcodeString(static_cast<SpvOp>(inst->opcode));
  }

  return SPV_SUCCESS;
}

}  // anonymous namespace

namespace libspirv {

// Validates that Data Rules are followed according to the specifications.
// (Data Rules subsection of 2.16.1 Universal Validation Rules)
spv_result_t DataRulesPass(ValidationState_t& _,
                           const spv_parsed_instruction_t* inst) {
  if (auto error = ValidateNumVecComponents(_, inst)) return error;

  // TODO(ehsan): add more data rules validation here.

  return SPV_SUCCESS;
}

}  // namespace libspirv
