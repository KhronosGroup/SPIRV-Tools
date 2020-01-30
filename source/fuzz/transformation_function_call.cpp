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

#include "source/fuzz/transformation_function_call.h"

#include "source/fuzz/call_graph.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationFunctionCall::TransformationFunctionCall(
    const spvtools::fuzz::protobufs::TransformationFunctionCall& message)
    : message_(message) {}

TransformationFunctionCall::TransformationFunctionCall(/* TODO */) {
  assert(false && "Not implemented yet");
}

bool TransformationFunctionCall::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) const {
  // The result id must be fresh
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }

  // The function must exist
  auto callee_inst = context->get_def_use_mgr()->GetDef(message_.callee_id());
  if (!callee_inst || callee_inst->opcode() != SpvOpFunction) {
    return false;
  }

  auto callee_type_inst = context->get_def_use_mgr()->GetDef(callee_inst->GetSingleWordInOperand(1));
  assert (callee_type_inst->opcode() == SpvOpTypeFunction && "Bad function type.");

  // The number of expected function arguments must match the number of given arguments.  The number of expected arguments is one less than the function type's number
  // of input operands, as one operand is for the return type.
  if (callee_type_inst->NumInOperands() - 1 != static_cast<uint32_t>(message_.argument_id().size())) {
    return false;
  }

  // The instruction descriptor must refer to a position where it is valid to
  // insert the call
  auto insert_before = FindInstruction(message_.instruction_to_insert_before(), context);
  if (insert_before) {
    return false;
  }
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpFunctionCall,
                                                    insert_before)) {
    return false;
  }

  auto block = context->get_instr_block(insert_before);

  // If the block is not dead, the function must be livesafe
  bool block_is_dead = fact_manager.BlockIsDead(block->id());
  if (block_is_dead && !fact_manager.FunctionIsLivesafe(message_.callee_id())) {
    return false;
  }

  // The ids must all match and have the right types and satisfy rules on
  // pointers.  If the block is not dead, pointers must be arbitrary.
  for (uint32_t arg_index = 0; arg_index < static_cast<uint32_t>(message_.argument_id().size()); arg_index++) {
    opt::Instruction* arg_inst = context->get_def_use_mgr()->GetDef(message_.argument_id(arg_index));
    opt::Instruction* arg_type_inst = context->get_def_use_mgr()->GetDef(arg_inst->type_id());
    if (arg_type_inst->result_id() != callee_type_inst->GetSingleWordInOperand(arg_index + 1)) {
      return false;
    }
    if (arg_type_inst->opcode() == SpvOpTypePointer) {
      switch (arg_inst->opcode()) {
        case SpvOpFunctionParameter:
        case SpvOpVariable:
          // These are OK
          break;
        default:
          // Other pointer ids cannot be passed as parameters
          return false;
      }
      if (!block_is_dead && fact_manager.VariableValueIsArbitrary(arg_inst->result_id())) {
        return false;
      }
    }
  }

  // Introducing the call must not lead to recursion.
  if (message_.callee_id() == block->GetParent()->result_id()) {
    // This would be direct recursion.
    return false;
  }
  CallGraph call_graph(context);
  if (call_graph.GetIndirectCallees(message_.callee_id()).count(block->GetParent()->result_id() != 0)) {
    // This would be indirect recursion.
    return false;
  }
  return true;
}

void TransformationFunctionCall::Apply(
    opt::IRContext* /*context*/,
    spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation TransformationFunctionCall::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_function_call() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
