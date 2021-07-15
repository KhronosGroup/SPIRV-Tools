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

#include "source/opt/freeze_spec_constant_value_pass.h"

#include "source/opt/ir_context.h"
#include "source/opt/set_spec_constant_default_value_pass.h"
#include "source/util/parse_number.h"

namespace spvtools {
namespace opt {

Pass::Status FreezeSpecConstantValuePass::Process() {
  // The operand index of decoration target in an OpDecorate instruction.
  const uint32_t kTargetIdOperandIndex = 0;
  // The operand index of the decoration literal in an OpDecorate instruction.
  const uint32_t kDecorationOperandIndex = 1;
  // The operand index of Spec id literal value in an OpDecorate SpecId
  // instruction.
  const uint32_t kSpecIdLiteralOperandIndex = 2;
  // The number of operands in an OpDecorate SpecId instruction.
  const uint32_t kOpDecorateSpecIdNumOperands = 3;

  auto ctx = context();
  auto def_use_mgr = get_def_use_mgr();
  bool modified = false;

  // Scan through all the annotation instructions to find 'OpDecorate SpecId'
  // instructions. This iteration style is modeled on
  // |InstructionList::ForEachInst| so that calling |KillInst| doesn't
  // invalidate the iterator.
  auto next = ctx->annotation_begin();
  for (auto i = next; i != ctx->annotation_end(); i = next) {
    ++next;

    // Only process 'OpDecorate SpecId' instructions
    Instruction& inst = *i;
    if (inst.opcode() != SpvOp::SpvOpDecorate) continue;
    if (inst.NumOperands() != kOpDecorateSpecIdNumOperands) continue;
    if (inst.GetSingleWordInOperand(kDecorationOperandIndex) !=
        uint32_t(SpvDecoration::SpvDecorationSpecId)) {
      continue;
    }

    // 'inst' is an OpDecorate SpecId instruction.
    uint32_t spec_id = inst.GetSingleWordOperand(kSpecIdLiteralOperandIndex);
    uint32_t target_id = inst.GetSingleWordOperand(kTargetIdOperandIndex);

    if (!ShouldFreezeSpecId(spec_id)) {
      continue;
    }

    // Find the spec constant defining instruction. Note that the
    // target_id might be a decoration group id.
    Instruction* spec_inst = nullptr;
    if (Instruction* target_inst = def_use_mgr->GetDef(target_id)) {
      if (target_inst->opcode() == SpvOp::SpvOpDecorationGroup) {
        spec_inst =
            SetSpecConstantDefaultValuePass::GetSpecIdTargetFromDecorationGroup(
                *target_inst, def_use_mgr);
      } else {
        spec_inst = target_inst;
      }
    }
    if (!spec_inst) continue;

    // Freeze the OpSpecConstant{|True|False} instruction and remove the
    // OpDecorate SpecId instruction
    switch (spec_inst->opcode()) {
      case SpvOp::SpvOpSpecConstant:
        spec_inst->SetOpcode(SpvOp::SpvOpConstant);
        ctx->KillInst(&inst);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantTrue:
        spec_inst->SetOpcode(SpvOp::SpvOpConstantTrue);
        ctx->KillInst(&inst);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantFalse:
        spec_inst->SetOpcode(SpvOp::SpvOpConstantFalse);
        ctx->KillInst(&inst);
        modified = true;
        break;
      default:
        break;
    }
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FreezeSpecConstantValuePass::ShouldFreezeSpecId(uint32_t spec_id) const {
  // If spec_ids_ is empty, freeze all spec constants.
  if (spec_ids_.empty()) {
    return true;
  }

  // Otherwise freeze only the ids present in spec_ids_.
  return (spec_ids_.find(spec_id) != spec_ids_.end());
}

std::unique_ptr<FreezeSpecConstantValuePass::SpecIdSet>
FreezeSpecConstantValuePass::ParseSpecIdsString(const char* str) {
  if (!str) return nullptr;

  auto spec_ids = MakeUnique<SpecIdSet>();

  // The parsing loop, break when points to the end.
  while (*str) {
    // Find the next spec id.
    while (std::isspace(*str)) str++;  // Skip leading spaces.
    const char* id_begin = str;
    while (!std::isspace(*str) && *str) str++;
    const char* id_end = str;
    std::string spec_id_str(id_begin, id_end - id_begin);
    uint32_t spec_id = 0;
    if (!utils::ParseNumber(spec_id_str.c_str(), &spec_id)) {
      // The spec id is not a valid uint32 number.
      return nullptr;
    }
    // Add to set.
    spec_ids->insert(spec_id);

    // Skip trailing spaces.
    while (std::isspace(*str)) str++;
  }

  return spec_ids;
}

}  // namespace opt
}  // namespace spvtools
