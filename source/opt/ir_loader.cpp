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

#include "ir_loader.h"

#include <cassert>

#include "reflect.h"

namespace spvtools {
namespace ir {

void IrLoader::AddInstruction(const spv_parsed_instruction_t* inst) {
  const auto opcode = static_cast<SpvOp>(inst->opcode);
  if (IsDebugLineInst(opcode)) {
    dbg_line_info_.push_back(Instruction(*inst));
    return;
  }

  Instruction spv_inst(*inst, std::move(dbg_line_info_));
  dbg_line_info_.clear();
  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == SpvOpFunction) {
    assert(function_ == nullptr);
    assert(block_ == nullptr);
    function_.reset(new Function(std::move(spv_inst)));
  } else if (opcode == SpvOpFunctionEnd) {
    assert(function_ != nullptr);
    assert(block_ == nullptr);
    module_->AddFunction(std::move(*function_.release()));
    function_ = nullptr;
  } else if (opcode == SpvOpLabel) {
    assert(function_ != nullptr);
    assert(block_ == nullptr);
    block_.reset(new BasicBlock(std::move(spv_inst)));
  } else if (IsTerminatorInst(opcode)) {
    assert(function_ != nullptr);
    assert(block_ != nullptr);
    block_->AddInstruction(std::move(spv_inst));
    function_->AddBasicBlock(std::move(*block_.release()));
    block_ = nullptr;
  } else {
    if (function_ == nullptr) {  // Outside function definition
      assert(block_ == nullptr);
      if (opcode == SpvOpCapability) {
        module_->AddCapability(std::move(spv_inst));
      } else if (opcode == SpvOpExtension) {
        module_->AddExtension(std::move(spv_inst));
      } else if (opcode == SpvOpExtInstImport) {
        module_->AddExtInstImport(std::move(spv_inst));
      } else if (opcode == SpvOpMemoryModel) {
        module_->SetMemoryModel(std::move(spv_inst));
      } else if (opcode == SpvOpEntryPoint) {
        module_->AddEntryPoint(std::move(spv_inst));
      } else if (opcode == SpvOpExecutionMode) {
        module_->AddExecutionMode(std::move(spv_inst));
      } else if (IsDebugInst(opcode)) {
        module_->AddDebugInst(std::move(spv_inst));
      } else if (IsAnnotationInst(opcode)) {
        module_->AddAnnotationInst(std::move(spv_inst));
      } else if (IsTypeInst(opcode)) {
        module_->AddType(std::move(spv_inst));
      } else if (IsConstantInst(opcode)) {
        module_->AddConstant(std::move(spv_inst));
      } else if (opcode == SpvOpVariable) {
        module_->AddGlobalVariable(std::move(spv_inst));
      } else {
        assert(0 && "unhandled inst type outside function defintion");
      }
    } else {
      if (block_ == nullptr) {  // Inside function but outside blocks
        assert(opcode == SpvOpFunctionParameter);
        function_->AddParameter(std::move(spv_inst));
      } else {
        block_->AddInstruction(std::move(spv_inst));
      }
    }
  }
}

// Resolves internal references among the module, functions, basic blocks, etc.
// This function should be called after adding all instructions.
//
// This concluding call is needed because the whole in memory representation is
// designed around rvalues and move semantics, which subject to pointer
// invalidation during module construction internally.
void IrLoader::EndModule() {
  for (auto& function : module_->functions()) {
    for (auto& bb : function.basic_blocks()) {
      bb.SetParent(&function);
    }
    function.SetParent(module_);
  }
}

}  // namespace ir
}  // namespace spvtools
