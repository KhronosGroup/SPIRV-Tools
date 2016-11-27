// Copyright (c) 2015-2016 The Khronos Group Inc.
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

// Performs validation on instructions that appear inside of a SPIR-V block.

#include "validate.h"

#include <cassert>

#include <sstream>
#include <string>

#include "diagnostic.h"
#include "enum_set.h"
#include "opcode.h"
#include "operand.h"
#include "spirv_definition.h"
#include "val/function.h"
#include "val/validation_state.h"

using libspirv::AssemblyGrammar;
using libspirv::CapabilitySet;
using libspirv::DiagnosticStream;
using libspirv::ValidationState_t;

namespace {

std::string ToString(const CapabilitySet& capabilities,
                     const AssemblyGrammar& grammar) {
  std::stringstream ss;
  capabilities.ForEach([&grammar, &ss](SpvCapability cap) {
    spv_operand_desc desc;
    if (SPV_SUCCESS ==
        grammar.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
      ss << desc->name << " ";
    else
      ss << cap << " ";
  });
  return ss.str();
}

// Reports a missing-capability error to _'s diagnostic stream and returns
// SPV_ERROR_INVALID_CAPABILITY.
spv_result_t CapabilityError(ValidationState_t& _, int which_operand,
                             SpvOp opcode,
                             const std::string& required_capabilities) {
  return _.diag(SPV_ERROR_INVALID_CAPABILITY)
         << "Operand " << which_operand << " of " << spvOpcodeString(opcode)
         << " requires one of these capabilities: " << required_capabilities;
}

// Returns an operand's required capabilities.
CapabilitySet RequiredCapabilities(const AssemblyGrammar& grammar,
                                   spv_operand_type_t type, uint32_t operand) {
  // Mere mention of PointSize, ClipDistance, or CullDistance in a Builtin
  // decoration does not require the associated capability.  The use of such
  // a variable value should trigger the capability requirement, but that's
  // not implemented yet.  This rule is independent of target environment.
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/365
  if (type == SPV_OPERAND_TYPE_BUILT_IN) {
    switch (operand) {
      case SpvBuiltInPointSize:
      case SpvBuiltInClipDistance:
      case SpvBuiltInCullDistance:
        return CapabilitySet();
      default:
        break;
    }
  }

  spv_operand_desc operand_desc;
  if (SPV_SUCCESS == grammar.lookupOperand(type, operand, &operand_desc)) {
    return operand_desc->capabilities;
  }

  return CapabilitySet();
}

}  // namespace

namespace libspirv {

spv_result_t CapCheck(ValidationState_t& _,
                      const spv_parsed_instruction_t* inst) {
  spv_opcode_desc opcode_desc;
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  if (SPV_SUCCESS == _.grammar().lookupOpcode(opcode, &opcode_desc) &&
      !_.HasAnyOf(opcode_desc->capabilities))
    return _.diag(SPV_ERROR_INVALID_CAPABILITY)
           << "Opcode " << spvOpcodeString(opcode)
           << " requires one of these capabilities: "
           << ToString(opcode_desc->capabilities, _.grammar());
  for (int i = 0; i < inst->num_operands; ++i) {
    const auto& operand = inst->operands[i];
    const auto word = inst->words[operand.offset];
    if (spvOperandIsConcreteMask(operand.type)) {
      // Check for required capabilities for each bit position of the mask.
      for (uint32_t mask_bit = 0x80000000; mask_bit; mask_bit >>= 1) {
        if (word & mask_bit) {
          const auto caps =
              RequiredCapabilities(_.grammar(), operand.type, mask_bit);
          if (!_.HasAnyOf(caps)) {
            return CapabilityError(_, i + 1, opcode,
                                   ToString(caps, _.grammar()));
          }
        }
      }
    } else if (spvIsIdType(operand.type)) {
      // TODO(dneto): Check the value referenced by this Id, if we can compute
      // it.  For now, just punt, to fix issue 248:
      // https://github.com/KhronosGroup/SPIRV-Tools/issues/248
    } else {
      // Check the operand word as a whole.
      const auto caps = RequiredCapabilities(_.grammar(), operand.type, word);
      if (!_.HasAnyOf(caps)) {
        return CapabilityError(_, i + 1, opcode, ToString(caps, _.grammar()));
      }
    }
  }
  return SPV_SUCCESS;
}

// Checks that the Resuld <id> is within the valid bound.
spv_result_t LimitCheckIdBound(ValidationState_t& _,
                               const spv_parsed_instruction_t* inst) {
  if (inst->result_id >= _.getIdBound()) {
    return _.diag(SPV_ERROR_INVALID_BINARY)
           << "Result <id> '" << inst->result_id
           << "' must be less than the ID bound '" << _.getIdBound() << "'.";
  }
  return SPV_SUCCESS;
}

spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  if (opcode == SpvOpCapability)
    _.RegisterCapability(
        static_cast<SpvCapability>(inst->words[inst->operands[0].offset]));
  if (opcode == SpvOpMemoryModel) {
    _.set_addressing_model(
        static_cast<SpvAddressingModel>(inst->words[inst->operands[0].offset]));
    _.set_memory_model(
        static_cast<SpvMemoryModel>(inst->words[inst->operands[1].offset]));
  }
  if (opcode == SpvOpVariable) {
    const auto storage_class =
        static_cast<SpvStorageClass>(inst->words[inst->operands[2].offset]);
    if (storage_class == SpvStorageClassGeneric)
      return _.diag(SPV_ERROR_INVALID_BINARY)
             << "OpVariable storage class cannot be Generic";
    if (_.current_layout_section() == kLayoutFunctionDefinitions) {
      if (storage_class != SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT)
               << "Variables must have a function[7] storage class inside"
                  " of a function";
      }
      if (_.current_function().IsFirstBlock(
              _.current_function().current_block()->id()) == false) {
        return _.diag(SPV_ERROR_INVALID_CFG) << "Variables can only be defined "
                                                "in the first block of a "
                                                "function";
      }
    } else {
      if (storage_class == SpvStorageClassFunction) {
        return _.diag(SPV_ERROR_INVALID_LAYOUT)
               << "Variables can not have a function[7] storage class "
                  "outside of a function";
      }
    }
  }

  if (auto error = CapCheck(_, inst)) return error;
  if (auto error = LimitCheckIdBound(_, inst)) return error;

  // All instruction checks have passed.
  return SPV_SUCCESS;
}
}  // namespace libspirv
