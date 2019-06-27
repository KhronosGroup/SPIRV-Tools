// Copyright (c) 2019 Google LLC
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

#include "source/opt/isolate_function_calls_pass.h"

#include <vector>
#include <cstdio>

#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {

bool IsolateFunctionCallsPass::IsolateCalls(Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != SpvOp::SpvOpFunctionCall) continue;

      modified = true;

      // Grab the block index in the function
      auto blockIndex = bi - func->begin();

      uint32_t postCallBlockId = context()->TakeNextId();
      uint32_t callBlockId = context()->TakeNextId();
      auto nextInst = ii; ++nextInst;

      BasicBlock *block = &*bi;
      /*BasicBlock *postCallBlock = */ block->SplitBasicBlock(context(), postCallBlockId, nextInst);
      BasicBlock *callBlock = block->SplitBasicBlock(context(), callBlockId, ii);

      // Add branch from call block to post-call block
      std::unique_ptr<Instruction> postCallBranch(
              new Instruction(context(), SpvOpBranch, 0, 0,
                  {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {postCallBlockId}}}));
      callBlock->AddInstruction(std::move(postCallBranch));

      // Add branch from pre-call block to call block
      std::unique_ptr<Instruction> callBranch(
              new Instruction(context(), SpvOpBranch, 0, 0,
                  {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {callBlockId}}}));
      block->AddInstruction(std::move(callBranch));

      // Insertion may have invalidated bi, reconstruct it.
      bi = func->begin() + blockIndex;
      bi++; // skip over the call block
      break;
    }
  }
  return modified;
}

Pass::Status IsolateFunctionCallsPass::Process() {
  // Process all entry point functions.
  bool anyModified = false;
  ProcessFunction pfn = [&anyModified, this](Function* fp) { anyModified |= IsolateCalls(fp); return false; };
  context()->ProcessEntryPointCallTree(pfn);
  return anyModified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

IsolateFunctionCallsPass::IsolateFunctionCallsPass() = default;

}  // namespace opt
}  // namespace spvtools
