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

#include "ir_loader.h"

#include "log.h"
#include "reflect.h"

namespace spvtools {
namespace ir {

void IrLoader::AddInstruction(const spv_parsed_instruction_t* inst) {
  const auto opcode = static_cast<SpvOp>(inst->opcode);
  if (IsDebugLineInst(opcode)) {
    dbg_line_info_.push_back(Instruction(*inst));
    return;
  }

  std::unique_ptr<Instruction> spv_inst(
      new Instruction(*inst, std::move(dbg_line_info_)));
  dbg_line_info_.clear();
  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == SpvOpFunction) {
    SPIRV_ASSERT(consumer_, function_ == nullptr);
    SPIRV_ASSERT(consumer_, block_ == nullptr);
    function_.reset(new Function(std::move(spv_inst)));
  } else if (opcode == SpvOpFunctionEnd) {
    SPIRV_ASSERT(consumer_, function_ != nullptr);
    SPIRV_ASSERT(consumer_, block_ == nullptr);
    function_->SetFunctionEnd(std::move(spv_inst));
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  } else if (opcode == SpvOpLabel) {
    SPIRV_ASSERT(consumer_, function_ != nullptr);
    SPIRV_ASSERT(consumer_, block_ == nullptr);
    block_.reset(new BasicBlock(std::move(spv_inst)));
  } else if (IsTerminatorInst(opcode)) {
    SPIRV_ASSERT(consumer_, function_ != nullptr);
    SPIRV_ASSERT(consumer_, block_ != nullptr);
    block_->AddInstruction(std::move(spv_inst));
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
  } else {
    if (function_ == nullptr) {  // Outside function definition
      SPIRV_ASSERT(consumer_, block_ == nullptr);
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
      } else if (IsConstantInst(opcode) || opcode == SpvOpVariable ||
                 opcode == SpvOpUndef) {
        module_->AddGlobalValue(std::move(spv_inst));
      } else {
        SPIRV_UNIMPLEMENTED(consumer_,
                            "unhandled inst type outside function defintion");
      }
    } else {
      if (block_ == nullptr) {  // Inside function but outside blocks
        SPIRV_ASSERT(consumer_, opcode == SpvOpFunctionParameter);
        function_->AddParameter(std::move(spv_inst));
      } else {
        block_->AddInstruction(std::move(spv_inst));
      }
    }
  }
}

// Resolves internal references among the module, functions, basic blocks, etc.
// This function should be called after adding all instructions.
void IrLoader::EndModule() {
  if (block_ && function_) {
    // We're in the middle of a basic block, but the terminator is missing.
    // Register the block anyway.  This lets us write tests with less
    // boilerplate.
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
  }
  if (function_) {
    // We're in the middle of a function, but the OpFunctionEnd is missing.
    // Register the function anyway.  This lets us write tests with less
    // boilerplate.
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  }
  for (auto& function : *module_) {
    for (auto& bb : function) bb.SetParent(&function);
    function.SetParent(module_);
  }
}

}  // namespace ir
}  // namespace spvtools
