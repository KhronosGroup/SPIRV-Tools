// Copyright (c) 2018 Google LLC.
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

#include <algorithm>
#include <vector>

#include "source/diagnostic.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_env.h"
#include "source/val/function.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if \c inst is an input or output variable.
bool is_interface_variable(const Instruction* inst, bool is_spv_1_4) {
  if (is_spv_1_4) {
    // Starting in SPIR-V 1.4, all global variables are interface variables.
    return inst->opcode() == SpvOpVariable &&
           inst->word(3u) != SpvStorageClassFunction;
  } else {
    return inst->opcode() == SpvOpVariable &&
           (inst->word(3u) == SpvStorageClassInput ||
            inst->word(3u) == SpvStorageClassOutput);
  }
}

// Checks that \c var is listed as an interface in all the entry points that use
// it.
spv_result_t check_interface_variable(ValidationState_t& _,
                                      const Instruction* var) {
  std::vector<const Function*> functions;
  std::vector<const Instruction*> uses;
  for (auto use : var->uses()) {
    uses.push_back(use.first);
  }
  for (uint32_t i = 0; i < uses.size(); ++i) {
    const auto user = uses[i];
    if (const Function* func = user->function()) {
      functions.push_back(func);
    } else {
      // In the rare case that the variable is used by another instruction in
      // the global scope, continue searching for an instruction used in a
      // function.
      for (auto use : user->uses()) {
        uses.push_back(use.first);
      }
    }
  }

  std::sort(functions.begin(), functions.end(),
            [](const Function* lhs, const Function* rhs) {
              return lhs->id() < rhs->id();
            });
  functions.erase(std::unique(functions.begin(), functions.end()),
                  functions.end());

  std::vector<uint32_t> entry_points;
  for (const auto func : functions) {
    for (auto id : _.FunctionEntryPoints(func->id())) {
      entry_points.push_back(id);
    }
  }

  std::sort(entry_points.begin(), entry_points.end());
  entry_points.erase(std::unique(entry_points.begin(), entry_points.end()),
                     entry_points.end());

  for (auto id : entry_points) {
    for (const auto& desc : _.entry_point_descriptions(id)) {
      bool found = false;
      for (auto interface : desc.interfaces) {
        if (var->id() == interface) {
          found = true;
          break;
        }
      }
      if (!found) {
        return _.diag(SPV_ERROR_INVALID_ID, var)
               << "Interface variable id <" << var->id()
               << "> is used by entry point '" << desc.name << "' id <" << id
               << ">, but is not listed as an interface";
      }
    }
  }

  return SPV_SUCCESS;
}

// This function assumes a base location has been determined already. As such
// any location decoration is invalid.
// TODO: if this code turns out to be slow, there is an opportunity to cache
// the for a given type id.
spv_result_t NumConsumedLocations(ValidationState_t& _, const Instruction* type,
                                  uint32_t* num_locations) {
  *num_locations = 0;
  switch (type->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      *num_locations = 1;
      break;
    case SpvOpTypeVector:
      if ((_.ContainsSizedIntOrFloatType(type->id(), SpvOpTypeInt, 64) ||
           _.ContainsSizedIntOrFloatType(type->id(), SpvOpTypeFloat, 64)) &&
          (type->GetOperandAs<uint32_t>(2) > 2)) {
        *num_locations = 2;
      } else {
        *num_locations = 1;
      }
      break;
    case SpvOpTypeMatrix:
      NumConsumedLocations(_, _.FindDef(type->GetOperandAs<uint32_t>(1)),
                           num_locations);
      *num_locations *= type->GetOperandAs<uint32_t>(2);
      break;
    case SpvOpTypeArray: {
      NumConsumedLocations(_, _.FindDef(type->GetOperandAs<uint32_t>(1)),
                           num_locations);
      bool is_int = false;
      bool is_const = false;
      uint32_t value = 0;
      std::tie(is_int, is_const, value) =
          _.EvalInt32IfConst(type->GetOperandAs<uint32_t>(2));
      if (is_int && is_const) *num_locations *= value;
      break;
    }
    case SpvOpTypeStruct: {
      if (_.HasDecoration(type->id(), SpvDecorationLocation)) {
        return _.diag(SPV_ERROR_INVALID_DATA, type)
               << "Members cannot be assigned a location";
      }
      for (uint32_t i = 1; i < type->operands().size(); ++i) {
        uint32_t member_locations = 0;
        if (auto error = NumConsumedLocations(
                _, _.FindDef(type->GetOperandAs<uint32_t>(i)),
                &member_locations)) {
          return error;
        }
        *num_locations += member_locations;
      }
      break;
    }
    default:
      break;
  }

  return SPV_SUCCESS;
}

uint32_t NumConsumedComponents(ValidationState_t& _, const Instruction* type) {
  uint32_t num_components = 0;
  switch (type->opcode()) {
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      if (type->GetOperandAs<uint32_t>(1) == 64) {
        num_components = 2;
      } else {
        num_components = 1;
      }
      break;
    case SpvOpTypeVector:
      num_components =
          NumConsumedComponents(_, _.FindDef(type->GetOperandAs<uint32_t>(1)));
      num_components *= type->GetOperandAs<uint32_t>(2);
      break;
    default:
      // This is an error that is validated elsewhere.
      break;
  }

  return num_components;
}

spv_result_t GetLocationsForVariable(ValidationState_t& _,
                                     const Instruction* entry_point,
                                     const Instruction* variable,
                                     std::vector<bool>* locations) {
  auto ptr_type_id = variable->GetOperandAs<uint32_t>(0);
  auto ptr_type = _.FindDef(ptr_type_id);
  auto type_id = ptr_type->GetOperandAs<uint32_t>(2);
  auto type = _.FindDef(type_id);
  // Unpack optional arrayness.
  if (type->opcode() == SpvOpTypeArray ||
      type->opcode() == SpvOpTypeRuntimeArray) {
    type_id = type->GetOperandAs<uint32_t>(1);
    type = _.FindDef(type_id);
  }
  bool has_location = false;
  uint32_t location = 0;
  bool has_component = false;
  uint32_t component = 0;
  bool is_block = _.HasDecoration(type_id, SpvDecorationBlock);
  for (auto& dec : _.id_decorations(variable->id())) {
    if (dec.dec_type() == SpvDecorationLocation) {
      if (has_location && dec.params()[0] != location) {
        return _.diag(SPV_ERROR_INVALID_DATA, variable)
               << "Variable has conflicting location decorations";
      }
      has_location = true;
      location = dec.params()[0];
    } else if (dec.dec_type() == SpvDecorationComponent) {
      if (has_component && dec.params()[0] != component) {
        return _.diag(SPV_ERROR_INVALID_DATA, variable)
               << "Variable has conflicting component decorations";
      }
      has_component = true;
      component = dec.params()[0];
    } else if (dec.dec_type() == SpvDecorationBuiltIn) {
      // Don't check built-ins.
      return SPV_SUCCESS;
    }
  }

  if (type->opcode() == SpvOpTypeStruct) {
    // Don't check built-ins.
    if (_.HasDecoration(type_id, SpvDecorationBuiltIn))
      return SPV_SUCCESS;
  }

  // Only block-decorated structs don't need a location on the variable.
  if (!has_location && !is_block) {
    return _.diag(SPV_ERROR_INVALID_DATA, variable)
           << "Variable must be decorated with a location";
  }

  const std::string storage_class =
      (variable->GetOperandAs<SpvStorageClass>(2) == SpvStorageClassInput)
          ? "input"
          : "output";
  if (has_location) {
    uint32_t num_locations = 0;
    if (auto error = NumConsumedLocations(_, type, &num_locations))
      return error;

    uint32_t num_components = NumConsumedComponents(_, type);
    uint32_t start = location * 4;
    uint32_t end = (location + num_locations) * 4;
    if (num_components != 0) {
      start += component;
      end = location * 4 + component + num_components;
    }
    if (end > locations->size()) {
      locations->resize(end, false);
    }
    for (uint32_t i = start; i < end; ++i) {
      if (locations->at(i)) {
        return _.diag(SPV_ERROR_INVALID_DATA, entry_point)
               << "Entry-point has conflicting " << storage_class
               << " location assignment at location " << i / 4 << ", component " << i % 4;
      }
      (*locations)[i] = true;
    }
  } else {
    // For Block-decorated structs with no location assigned to the variable,
    // each member of the block must be assigned a location. Also record any
    // member component assignments.
    std::unordered_map<uint32_t, uint32_t> member_locations;
    std::unordered_map<uint32_t, uint32_t> member_components;
    for (auto& dec : _.id_decorations(type_id)) {
      if (dec.dec_type() == SpvDecorationLocation) {
        auto where = member_locations.find(dec.struct_member_index());
        if (where == member_locations.end()) {
          member_locations[dec.struct_member_index()] = dec.params()[0];
        } else if (where->second != dec.params()[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA, type)
                 << "Member index " << dec.struct_member_index()
                 << " has conflicting location assignments";
        }
      } else if (dec.dec_type() == SpvDecorationComponent) {
        auto where = member_components.find(dec.struct_member_index());
        if (where == member_components.end()) {
          member_components[dec.struct_member_index()] = dec.params()[0];
        } else if (where->second != dec.params()[0]) {
          return _.diag(SPV_ERROR_INVALID_DATA, type)
                 << "Member index " << dec.struct_member_index()
                 << " has conflicting location assignments";
        }
      }
    }

    for (uint32_t i = 1; i < type->operands().size(); ++i) {
      auto where = member_locations.find(i - 1);
      if (where == member_locations.end()) {
        return _.diag(SPV_ERROR_INVALID_DATA, type)
               << "Member index " << i - 1
               << " is missing a location assignment";
      }
      location = where->second;
      auto member = _.FindDef(type->GetOperandAs<uint32_t>(i));
      uint32_t num_locations = 0;
      if (auto error = NumConsumedLocations(_, member, &num_locations))
        return error;

      uint32_t num_components = NumConsumedComponents(_, member);
      component = 0;
      if (member_components.count(i - 1)) {
        component = member_components[i - 1];
      }

      uint32_t start = location * 4;
      uint32_t end = (location + num_locations) * 4;
      if (num_components != 0) {
        start += component;
        end = location * 4 + component + num_components;
      }
      if (end > locations->size()) {
        locations->resize(end, false);
      }
      for (uint32_t l = start; l < end; ++l) {
        if (locations->at(l)) {
          return _.diag(SPV_ERROR_INVALID_DATA, entry_point)
                 << "Entry-point has conflicting " << storage_class
                 << " location assignment at location " << l / 4 << ", component " << l % 4;
        }
        (*locations)[l] = true;
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateLocations(ValidationState_t& _, const Instruction* entry_point) {
  std::vector<bool> input_locations(64, false);
  std::vector<bool> output_locations(64, false);
  for (uint32_t i = 3; i < entry_point->operands().size(); ++i) {
    auto interface_id = entry_point->GetOperandAs<uint32_t>(i);
    auto interface_var = _.FindDef(interface_id);
    auto storage_class = interface_var->GetOperandAs<SpvStorageClass>(2);
    if (storage_class != SpvStorageClassInput &&
        storage_class != SpvStorageClassOutput) {
      continue;
    }

    auto locations = (storage_class == SpvStorageClassInput)
                         ? &input_locations
                         : &output_locations;
    if (auto error =
            GetLocationsForVariable(_, entry_point, interface_var, locations))
      return error;
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ValidateInterfaces(ValidationState_t& _) {
  bool is_spv_1_4 = _.version() >= SPV_SPIRV_VERSION_WORD(1, 4);
  for (auto& inst : _.ordered_instructions()) {
    if (is_interface_variable(&inst, is_spv_1_4)) {
      if (auto error = check_interface_variable(_, &inst)) {
        return error;
      }
    }
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    for (auto& inst : _.ordered_instructions()) {
      if (inst.opcode() == SpvOpEntryPoint) {
        if (auto error = ValidateLocations(_, &inst)) return error;
      }
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
