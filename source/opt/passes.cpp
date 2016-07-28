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

}  // namespace opt
}  // namespace spvtools
