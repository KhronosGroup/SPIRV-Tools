// Copyright (c) 2016 Google Inc.
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

#include "passes.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "reflect.h"

namespace spvtools {
namespace opt {

bool StripDebugInfoPass::Process(ir::Module* module) {
  bool modified = !module->debugs().empty();
  module->debugs().clear();

  module->ForEachInst([&modified](ir::Instruction* inst) {
    modified |= !inst->dbg_line_insts().empty();
    inst->dbg_line_insts().clear();
  });

  return modified;
}

bool FreezeSpecConstantValuePass::Process(ir::Module* module) {
  bool modified = false;
  module->ForEachInst([&modified](ir::Instruction* inst) {
    switch (inst->opcode()) {
      case SpvOp::SpvOpSpecConstant:
        inst->SetOpcode(SpvOp::SpvOpConstant);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantTrue:
        inst->SetOpcode(SpvOp::SpvOpConstantTrue);
        modified = true;
        break;
      case SpvOp::SpvOpSpecConstantFalse:
        inst->SetOpcode(SpvOp::SpvOpConstantFalse);
        modified = true;
        break;
      case SpvOp::SpvOpDecorate:
        if (inst->GetSingleWordInOperand(1) ==
            SpvDecoration::SpvDecorationSpecId) {
          inst->ToNop();
          modified = true;
        }
        break;
      default:
        break;
    }
  });
  return modified;
}

bool EliminateDeadConstantPass::Process(ir::Module* module) {
  std::unordered_set<ir::Instruction*> working_list;
  // Traverse all the instructions to get the initial set of dead constants as
  // working list and count number of real uses for constants. Uses in
  // annotation instructions do not count.
  std::unordered_map<ir::Instruction*, size_t> use_counts;
  std::vector<ir::Instruction*> constants = module->GetConstants();
  for (auto* c : constants) {
    uint32_t const_id = c->result_id();
    size_t count = 0;
    if (const analysis::UseList* uses =
            module->GetDefUseInfo().GetUses(const_id)) {
      count =
          std::count_if(uses->begin(), uses->end(), [](const analysis::Use& u) {
            return !(ir::IsAnnotationInst(u.inst->opcode()) ||
                     ir::IsDebugInst(u.inst->opcode()));
          });
    }
    use_counts[c] = count;
    if (!count) {
      working_list.insert(c);
    }
  }

  // Start from the constants with 0 uses, back trace through the def-use chain
  // to find all dead constants.
  std::unordered_set<ir::Instruction*> dead_consts;
  while (!working_list.empty()) {
    ir::Instruction* inst = *working_list.begin();
    // Back propagate if the instruction contains IDs in its operands.
    switch (inst->opcode()) {
      case SpvOp::SpvOpConstantComposite:
      case SpvOp::SpvOpSpecConstantComposite:
      case SpvOp::SpvOpSpecConstantOp:
        for (uint32_t i = 0; i < inst->NumInOperands(); i++) {
          // SpecConstantOp instruction contains 'opcode' as its operand. Need
          // to exclude such operands when decreasing uses.
          if (inst->GetInOperand(i).type != SPV_OPERAND_TYPE_ID) {
            continue;
          }
          uint32_t operand_id = inst->GetSingleWordInOperand(i);
          ir::Instruction* def_inst =
              module->GetDefUseInfo().GetDef(operand_id);
          // If the use_count does not have any count for the def_inst,
          // def_inst must not be a constant, and should be ignored here.
          if (!use_counts.count(def_inst)) {
            continue;
          }
          // The number of uses should never be less then 0, so it can not be
          // less than 1 before it decreases.
          assert(use_counts[def_inst] > 0);
          --use_counts[def_inst];
          if (!use_counts[def_inst]) {
            working_list.insert(def_inst);
          }
        }
        break;
      default:
        break;
    }
    dead_consts.insert(inst);
    working_list.erase(inst);
  }

  // Find all annotation and debug instructions that are referencing dead
  // constants.
  std::unordered_set<ir::Instruction*> dead_others;
  for (auto* dc : dead_consts) {
    if (const analysis::UseList* uses =
            module->GetDefUseInfo().GetUses(dc->result_id())) {
      for (const auto& u : *uses) {
        if (ir::IsAnnotationInst(u.inst->opcode()) ||
            ir::IsDebugInst(u.inst->opcode())) {
          dead_others.insert(u.inst);
        }
      }
    }
  }

  // Turn all dead instructions and uses of them to nop
  for (auto* dc : dead_consts) {
    // module->KillDef(dc->result_id());
    module->GetDefUseInfo().KillInst(dc);
  }
  for (auto* da : dead_others) {
    da->ToNop();
  }
  return !dead_consts.empty();
}

}  // namespace opt
}  // namespace spvtools
