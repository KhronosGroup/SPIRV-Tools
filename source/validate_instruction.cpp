// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Performs validation on instructions that appear inside of a SPIR-V block.

#include <cassert>
#include <sstream>
#include <string>

#include "opcode.h"
#include "spirv_definition.h"
#include "validate_passes.h"

using libspirv::ValidationState_t;

namespace {

std::string ToString(spv_capability_mask_t mask,
                     const libspirv::AssemblyGrammar& grammar) {
  std::stringstream ss;
  for (int cap = SpvCapabilityMatrix;
       cap <= SpvCapabilityStorageImageWriteWithoutFormat; ++cap) {
    if (spvIsInBitfield(SPV_CAPABILITY_AS_MASK(cap), mask)) {
      spv_operand_desc desc;
      if (SPV_SUCCESS ==
          grammar.lookupOperand(SPV_OPERAND_TYPE_CAPABILITY, cap, &desc))
        ss << desc->name << " ";
      else
        ss << cap << " ";
    }
  }
  return ss.str();
}

}  // namespace anonymous

namespace libspirv {

spv_result_t CapCheck(ValidationState_t& _,
                      const spv_parsed_instruction_t* inst) {
  spv_opcode_desc opcode_desc;
  if (SPV_SUCCESS == _.grammar().lookupOpcode(inst->opcode, &opcode_desc) &&
      !_.HasAnyOf(opcode_desc->capabilities))
    return _.diag(SPV_ERROR_INVALID_CAPABILITY)
           << "Opcode " << spvOpcodeString(inst->opcode)
           << " requires one of these capabilities: "
           << ToString(opcode_desc->capabilities, _.grammar());
  for (int i = 0; i < inst->num_operands; ++i) {
    spv_operand_desc operand_desc;
    if (SPV_SUCCESS ==
            _.grammar().lookupOperand(inst->operands[i].type,
                                      inst->words[inst->operands[i].offset],
                                      &operand_desc) &&
        !_.HasAnyOf(operand_desc->capabilities))
      return _.diag(SPV_ERROR_INVALID_CAPABILITY)
             << "Operand " << i + 1 << " of " << spvOpcodeString(inst->opcode)
             << " requires one of these capabilities: "
             << ToString(operand_desc->capabilities, _.grammar());
  }
  return SPV_SUCCESS;
}

// clang-format off
spv_result_t InstructionPass(ValidationState_t& _,
                             const spv_parsed_instruction_t* inst) {
  if (_.is_enabled(SPV_VALIDATE_INSTRUCTION_BIT)) {
    if (inst->opcode == SpvOpCapability)
        _.registerCapability(
            static_cast<SpvCapability>(inst->words[inst->operands[0].offset]));
    if (inst->opcode == SpvOpVariable) {
        const auto storage_class =
            static_cast<SpvStorageClass>(inst->words[inst->operands[2].offset]);
        if (storage_class == SpvStorageClassGeneric)
          return _.diag(SPV_ERROR_INVALID_BINARY)
              << "OpVariable storage class cannot be Generic";
        if (_.getLayoutSection() == kLayoutFunctionDefinitions) {
          if (storage_class != SpvStorageClassFunction) {
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "Variables must have a function[7] storage class inside"
                      " of a function";
          }
        } else {
          if (storage_class == SpvStorageClassFunction) {
            return _.diag(SPV_ERROR_INVALID_LAYOUT)
                   << "Variables can not have a function[7] storage class "
                      "outside of a function";
          }
        }
    }
    return CapCheck(_, inst);
  }
  return SPV_SUCCESS;
}
// clang-format on
}  // namespace libspirv
