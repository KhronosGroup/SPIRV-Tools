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

#include <cassert>

#include "reflect.h"
#include "spv_builder.h"

namespace spvtools {
namespace opt {
namespace ir {

void SpvBuilder::AddInstruction(const spv_parsed_instruction_t* inst) {
  const auto opcode = static_cast<SpvOp>(inst->opcode);
  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == SpvOpFunction) {
    assert(function_ == nullptr);
    assert(block_ == nullptr);
    function_.reset(new Function(module_, Inst(*inst)));
  } else if (opcode == SpvOpFunctionEnd) {
    assert(function_ != nullptr);
    assert(block_ == nullptr);
    module_->AddFunction(std::move(*function_.release()));
    function_ = nullptr;
  } else if (opcode == SpvOpLabel) {
    assert(function_ != nullptr);
    assert(block_ == nullptr);
    block_.reset(new BasicBlock(function_.get(), Inst(*inst)));
  } else if (IsTerminatorInst(opcode)) {
    assert(function_ != nullptr);
    assert(block_ != nullptr);
    block_->AddInstruction(Inst(*inst));
    function_->AddBasicBlock(std::move(*block_.release()));
    block_ = nullptr;
  } else {
    if (function_ == nullptr) {  // Outside function definition
      assert(block_ == nullptr);
      if (opcode == SpvOpCapability) {
        module_->AddCapability(Inst(*inst));
      } else if (opcode == SpvOpExtension) {
        module_->AddExtension(Inst(*inst));
      } else if (opcode == SpvOpExtInstImport) {
        module_->AddExtInstSet(Inst(*inst));
      } else if (opcode == SpvOpMemoryModel) {
        module_->SetMemoryModel(Inst(*inst));
      } else if (opcode == SpvOpEntryPoint) {
        module_->AddEntryPoint(Inst(*inst));
      } else if (opcode == SpvOpExecutionMode) {
        module_->AddExecutionMode(Inst(*inst));
      } else if (IsDebugInst(opcode)) {
        module_->AddDebugInst(Inst(*inst));
      } else if (IsAnnotationInst(opcode)) {
        module_->AddAnnotationInst(Inst(*inst));
      } else if (IsTypeInst(opcode)) {
        module_->AddType(Inst(*inst));
      } else if (IsConstantInst(opcode)) {
        module_->AddConstant(Inst(*inst));
      } else if (opcode == SpvOpVariable) {
        module_->AddVariable(Inst(*inst));
      } else {
        assert(0 && "unhandled inst type outside function defintion");
      }
    } else {
      if (block_ == nullptr) {  // Inside function but outside blocks
        assert(opcode == SpvOpFunctionParameter);
        function_->AddParameter(Inst(*inst));
      } else {
        block_->AddInstruction(Inst(*inst));
      }
    }
  }
}

}  // namespace ir
}  // namespace opt
}  // namespace spvtools
