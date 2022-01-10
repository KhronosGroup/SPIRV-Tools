// Copyright (c) 2021 Google LLC
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

#include "source/opt/spread_volatile_semantics.h"

#include "source/opt/decoration_manager.h"
#include "source/opt/ir_builder.h"
#include "source/spirv_constant.h"

namespace spvtools {
namespace opt {
namespace {

const uint32_t kOpDecorateInOperandBuiltinDecoration = 2u;
const uint32_t kOpLoadInOperandMemoryOperands = 1u;

bool HasBuiltinDecoration(analysis::DecorationManager* decoration_manager,
                          Instruction* var, uint32_t built_in) {
  return decoration_manager->FindDecoration(
      var->result_id(), SpvDecorationBuiltIn,
      [built_in](const Instruction& inst) {
        return built_in == inst.GetSingleWordInOperand(
                               kOpDecorateInOperandBuiltinDecoration);
      });
}

bool IsBuiltInForRayTracingVolatileSemantics(uint32_t built_in) {
  switch (built_in) {
    case SpvBuiltInSMIDNV:
    case SpvBuiltInWarpIDNV:
    case SpvBuiltInSubgroupSize:
    case SpvBuiltInSubgroupLocalInvocationId:
    case SpvBuiltInSubgroupEqMask:
    case SpvBuiltInSubgroupGeMask:
    case SpvBuiltInSubgroupGtMask:
    case SpvBuiltInSubgroupLeMask:
    case SpvBuiltInSubgroupLtMask:
      return true;
    default:
      return false;
  }
}

bool HasBuiltinForRayTracingVolatileSemantics(
    analysis::DecorationManager* decoration_manager, Instruction* var) {
  return decoration_manager->FindDecoration(
      var->result_id(), SpvDecorationBuiltIn, [](const Instruction& inst) {
        uint32_t built_in =
            inst.GetSingleWordInOperand(kOpDecorateInOperandBuiltinDecoration);
        return IsBuiltInForRayTracingVolatileSemantics(built_in);
      });
}

bool HasVolatileDecoration(analysis::DecorationManager* decoration_manager,
                           uint32_t var_id) {
  return decoration_manager->HasDecoration(var_id, SpvDecorationVolatile);
}

}  // namespace

Pass::Status SpreadVolatileSemantics::Process() {
  Status status = Status::SuccessWithoutChange;
  bool is_vk_memory_model_enabled = context()->get_feature_mgr()->HasCapability(
      SpvCapabilityVulkanMemoryModel);
  for (Instruction& var : context()->types_values()) {
    if (!IsTargetForVolatileSemantics(&var)) {
      continue;
    }

    if (is_vk_memory_model_enabled) {
      SetVolatileForLoads(&var);
    } else {
      DecorateVarWithVolatile(&var);
    }
    status = Status::SuccessWithChange;
  }
  return status;
}

void SpreadVolatileSemantics::DecorateVarWithVolatile(Instruction* var) {
  analysis::DecorationManager* decoration_manager =
      context()->get_decoration_mgr();
  uint32_t var_id = var->result_id();
  if (HasVolatileDecoration(decoration_manager, var_id)) {
    return;
  }
  get_decoration_mgr()->AddDecoration(
      SpvOpDecorate,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {var_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {SpvDecorationVolatile}}});
}

void SpreadVolatileSemantics::SetVolatileForLoads(Instruction* var) {
  auto* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->ForEachUser(var, [](Instruction* user) {
    if (user->opcode() != SpvOpLoad) {
      return;
    }

    if (user->NumInOperands() <= kOpLoadInOperandMemoryOperands) {
      user->AddOperand(
          {SPV_OPERAND_TYPE_MEMORY_ACCESS, {SpvMemoryAccessVolatileMask}});
      return;
    }

    uint32_t memory_operands =
        user->GetSingleWordInOperand(kOpLoadInOperandMemoryOperands);
    memory_operands |= SpvMemoryAccessVolatileMask;
    user->SetInOperand(kOpLoadInOperandMemoryOperands, {memory_operands});
  });
}

bool SpreadVolatileSemantics::IsTargetForVolatileSemantics(Instruction* var) {
  analysis::DecorationManager* decoration_manager =
      context()->get_decoration_mgr();
  auto execution_model = context()->GetExecutionModel();
  if (execution_model == SpvExecutionModelFragment) {
    return get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 6) &&
           HasBuiltinDecoration(decoration_manager, var,
                                SpvBuiltInHelperInvocation);
  }

  if (execution_model == SpvExecutionModelIntersectionKHR ||
      execution_model == SpvExecutionModelIntersectionNV) {
    if (HasBuiltinDecoration(decoration_manager, var, SpvBuiltInRayTmaxKHR)) {
      return true;
    }
  }

  switch (execution_model) {
    case SpvExecutionModelRayGenerationKHR:
    case SpvExecutionModelClosestHitKHR:
    case SpvExecutionModelMissKHR:
    case SpvExecutionModelCallableKHR:
    case SpvExecutionModelIntersectionKHR:
      return HasBuiltinForRayTracingVolatileSemantics(decoration_manager, var);
    default:
      return false;
  }
}

}  // namespace opt
}  // namespace spvtools
