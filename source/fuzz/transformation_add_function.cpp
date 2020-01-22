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

#include "source/fuzz/transformation_add_function.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_message.h"

namespace spvtools {
namespace fuzz {

TransformationAddFunction::TransformationAddFunction(
    const spvtools::fuzz::protobufs::TransformationAddFunction& message)
    : message_(message) {}

TransformationAddFunction::TransformationAddFunction(
    const std::vector<protobufs::Instruction>& instructions) {
  for (auto& instruction : instructions) {
    *message_.add_instruction() = instruction;
  }
  message_.set_is_livesafe(false);
}

TransformationAddFunction::TransformationAddFunction(
    const std::vector<protobufs::Instruction>& instructions,
    uint32_t loop_limiter_variable_id, uint32_t loop_limit_constant_id,
    const std::vector<protobufs::LoopLimiterInfo>& loop_limiters,
    uint32_t kill_unreachable_return_value_id,
    const std::vector<protobufs::AccessChainClampingInfo>&
        access_chain_clampers) {
  for (auto& instruction : instructions) {
    *message_.add_instruction() = instruction;
  }
  message_.set_is_livesafe(true);
  message_.set_loop_limiter_variable_id(loop_limiter_variable_id);
  message_.set_loop_limit_constant_id(loop_limit_constant_id);
  for (auto& loop_limiter : loop_limiters) {
    *message_.add_loop_limiter_info() = loop_limiter;
  }
  message_.set_kill_unreachable_return_value_id(
      kill_unreachable_return_value_id);
  for (auto& access_clamper : access_chain_clampers) {
    *message_.add_access_chain_clamping_info() = access_clamper;
  }
}

bool TransformationAddFunction::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) const {
  // This transformation may use a lot of ids, all of which need to be fresh
  // and distinct.  This set tracks them.
  std::set<uint32_t> ids_used_by_this_transformation;

  // Ensure that all result ids in the new function are fresh and distinct.
  for (auto& instruction : message_.instruction()) {
    if (instruction.result_id()) {
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              instruction.result_id(), context,
              &ids_used_by_this_transformation)) {
        return false;
      }
    }
  }

  if (message_.is_livesafe()) {
    // Ensure that all ids provided for making the function livesafe are fresh
    // and distinct.
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            message_.loop_limiter_variable_id(), context,
            &ids_used_by_this_transformation)) {
      return false;
    }
    for (auto& loop_limiter_info : message_.loop_limiter_info()) {
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              loop_limiter_info.new_block_id(), context,
              &ids_used_by_this_transformation)) {
        return false;
      }
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              loop_limiter_info.load_id(), context,
              &ids_used_by_this_transformation)) {
        return false;
      }
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              loop_limiter_info.increment_id(), context,
              &ids_used_by_this_transformation)) {
        return false;
      }
      if (!CheckIdIsFreshAndNotUsedByThisTransformation(
              loop_limiter_info.compare_id(), context,
              &ids_used_by_this_transformation)) {
        return false;
      }
    }
    for (auto& access_chain_clamping_info :
         message_.access_chain_clamping_info()) {
      for (auto& pair : access_chain_clamping_info.compare_and_select_ids()) {
        if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                pair.first(), context, &ids_used_by_this_transformation)) {
          return false;
        }
        if (!CheckIdIsFreshAndNotUsedByThisTransformation(
                pair.second(), context, &ids_used_by_this_transformation)) {
          return false;
        }
      }
    }
  }

  // Because checking all the conditions for a function to be valid is a big
  // job that the SPIR-V validator can already do, a "try it and see" approach
  // is taken here.

  // We first clone the current module, so that we can try adding the new
  // function without risking wrecking |context|.
  auto cloned_module = fuzzerutil::CloneIRContext(context);

  // We try to add a function to the cloned module, which may fail if
  // |message_.instruction| is not sufficiently well-formed.
  if (!TryToAddFunction(cloned_module.get())) {
    return false;
  }

  if (message_.is_livesafe()) {
    // We make the cloned module livesafe.
    if (!TryToMakeFunctionLivesafe(cloned_module.get(), fact_manager)) {
      return false;
    }
  }

  // Having managed to add the new function to the cloned module, and
  // potentially also made it livesafe, we ascertain whether the cloned module
  // is still valid.  If it is, the transformation is applicable.
  return fuzzerutil::IsValid(cloned_module.get());
}

void TransformationAddFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* fact_manager) const {
  // Add the function to the module.  As the transformation is applicable, this
  // should succeed.
  bool success = TryToAddFunction(context);
  assert(success && "The function should be successfully added.");
  (void)(success);  // Keep release builds happy (otherwise they may complain
                    // that |success| is not used).

  if (message_.is_livesafe()) {
    // Make the function livesafe, which also should succeed.
    success = TryToMakeFunctionLivesafe(context, *fact_manager);
    assert(success && "It should be possible to make the function livesafe.");
    (void)(success);  // Keep release builds happy.

    // Inform the fact manager that the function is livesafe.
    assert(message_.instruction(0).opcode() == SpvOpFunction &&
           "The first instruction of an 'add function' transformation must be "
           "OpFunction.");
    fact_manager->AddFactFunctionIsLivesafe(
        message_.instruction(0).result_id());
  } else {
    // Inform the fact manager that all blocks in the function are dead.
    for (auto& inst : message_.instruction()) {
      if (inst.opcode() == SpvOpLabel) {
        fact_manager->AddFactBlockIsDead(inst.result_id());
      }
    }
  }
  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationAddFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_function() = message_;
  return result;
}

bool TransformationAddFunction::TryToAddFunction(
    opt::IRContext* context) const {
  // This function returns false if |message_.instruction| was not well-formed
  // enough to actually create a function and add it to |context|.

  // A function must have at least some instructions.
  if (message_.instruction().empty()) {
    return false;
  }

  // A function must start with OpFunction.
  auto function_begin = message_.instruction(0);
  if (function_begin.opcode() != SpvOpFunction) {
    return false;
  }

  // Make a function, headed by the OpFunction instruction.
  std::unique_ptr<opt::Function> new_function = MakeUnique<opt::Function>(
      InstructionFromMessage(context, function_begin));

  // Keeps track of which instruction protobuf message we are currently
  // considering.
  uint32_t instruction_index = 1;
  const auto num_instructions =
      static_cast<uint32_t>(message_.instruction().size());

  // Iterate through all function parameter instructions, adding parameters to
  // the new function.
  while (instruction_index < num_instructions &&
         message_.instruction(instruction_index).opcode() ==
             SpvOpFunctionParameter) {
    new_function->AddParameter(InstructionFromMessage(
        context, message_.instruction(instruction_index)));
    instruction_index++;
  }

  // After the parameters, there needs to be a label.
  if (instruction_index == num_instructions ||
      message_.instruction(instruction_index).opcode() != SpvOpLabel) {
    return false;
  }

  // Iterate through the instructions block by block until the end of the
  // function is reached.
  while (instruction_index < num_instructions &&
         message_.instruction(instruction_index).opcode() != SpvOpFunctionEnd) {
    // Invariant: we should always be at a label instruction at this point.
    assert(message_.instruction(instruction_index).opcode() == SpvOpLabel);

    // Make a basic block using the label instruction, with the new function
    // as its parent.
    std::unique_ptr<opt::BasicBlock> block =
        MakeUnique<opt::BasicBlock>(InstructionFromMessage(
            context, message_.instruction(instruction_index)));
    block->SetParent(new_function.get());

    // Consider successive instructions until we hit another label or the end
    // of the function, adding each such instruction to the block.
    instruction_index++;
    while (instruction_index < num_instructions &&
           message_.instruction(instruction_index).opcode() !=
               SpvOpFunctionEnd &&
           message_.instruction(instruction_index).opcode() != SpvOpLabel) {
      block->AddInstruction(InstructionFromMessage(
          context, message_.instruction(instruction_index)));
      instruction_index++;
    }
    // Add the block to the new function.
    new_function->AddBasicBlock(std::move(block));
  }
  // Having considered all the blocks, we should be at the last instruction and
  // it needs to be OpFunctionEnd.
  if (instruction_index != num_instructions - 1 ||
      message_.instruction(instruction_index).opcode() != SpvOpFunctionEnd) {
    return false;
  }
  // Set the function's final instruction, add the function to the module and
  // report success.
  new_function->SetFunctionEnd(
      InstructionFromMessage(context, message_.instruction(instruction_index)));
  context->AddFunction(std::move(new_function));

  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  return true;
}

bool TransformationAddFunction::TryToMakeFunctionLivesafe(
    opt::IRContext* context, const FactManager& fact_manager) const {
  assert(message_.is_livesafe() && "Precondition: is_livesafe must hold.");

  // Get a pointer to the added function.
  opt::Function* added_function = nullptr;
  for (auto& function : *context->module()) {
    if (function.result_id() == message_.instruction(0).result_id()) {
      added_function = &function;
      break;
    }
  }
  assert(added_function && "The added function should have been found.");

  if (!TryToAddLoopLimiters(context, added_function)) {
    // Adding loop limiters did not work; bail out.
    return false;
  }

  // Consider all the instructions in the function, and:
  // - attempt to replace OpKill and OpUnreachable with return instructions
  // - attempt to clamp access chains to be within bounds
  // - check that OpFunctionCall instructions are only to livesafe functions
  for (auto& block : *added_function) {
    for (auto& inst : block) {
      switch (inst.opcode()) {
        case SpvOpKill:
        case SpvOpUnreachable:
          if (!TryToTurnKillOrUnreachableIntoReturn(context, added_function,
                                                    &inst)) {
            return false;
          }
          break;
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain:
          if (!TryToClampAccessChainIndices(context, &inst)) {
            return false;
          }
          break;
        case SpvOpFunctionCall:
          // A livesafe function my only call other livesafe functions.
          if (!fact_manager.FunctionIsLivesafe(
                  inst.GetSingleWordInOperand(0))) {
            return false;
          }
        default:
          break;
      }
    }
  }
  return true;
}

bool TransformationAddFunction::TryToAddLoopLimiters(
    opt::IRContext* context, opt::Function* added_function) const {
  // Collect up all the loop headers so that we can subsequently add loop
  // limiting logic.
  std::vector<opt::BasicBlock*> loop_headers;
  for (auto& block : *added_function) {
    if (block.IsLoopHeader()) {
      loop_headers.push_back(&block);
    }
  }

  if (loop_headers.empty()) {
    // There are no loops, so no need to add any loop limiters.
    return true;
  }

  // Check that the module contains appropriate ingredients for declaring and
  // manipulating a loop limiter.

  auto loop_limit_constant_id_instr =
      context->get_def_use_mgr()->GetDef(message_.loop_limit_constant_id());
  if (!loop_limit_constant_id_instr ||
      loop_limit_constant_id_instr->opcode() != SpvOpConstant) {
    // The loop limit constant id instruction must exist and have an
    // appropriate opcode.
    return false;
  }

  auto loop_limit_type = context->get_def_use_mgr()->GetDef(
      loop_limit_constant_id_instr->type_id());
  if (loop_limit_type->opcode() != SpvOpTypeInt ||
      loop_limit_type->GetSingleWordInOperand(0) != 32) {
    // The type of the loop limit constant must be 32-bit integer.  It
    // doesn't actually matter whether the integer is signed or not.
    return false;
  }

  // Find the id of the "unsigned int" type.
  opt::analysis::Integer unsigned_int_type(32, false);
  uint32_t unsigned_int_type_id =
      context->get_type_mgr()->GetId(&unsigned_int_type);
  if (!unsigned_int_type_id) {
    // Unsigned int is not available; we need this type in order to add loop
    // limiters.
    return false;
  }
  auto registered_unsigned_int_type =
      context->get_type_mgr()->GetRegisteredType(&unsigned_int_type);

  // Look for 0 of type unsigned int.
  opt::analysis::IntConstant zero(registered_unsigned_int_type->AsInteger(),
                                  {0});
  auto registered_zero = context->get_constant_mgr()->FindConstant(&zero);
  if (!registered_zero) {
    // We need 0 in order to be able to initialize loop limiters.
    return false;
  }
  uint32_t zero_id = context->get_constant_mgr()
                         ->GetDefiningInstruction(registered_zero)
                         ->result_id();

  // Look for 1 of type unsigned int.
  opt::analysis::IntConstant one(registered_unsigned_int_type->AsInteger(),
                                 {1});
  auto registered_one = context->get_constant_mgr()->FindConstant(&one);
  if (!registered_one) {
    // We need 1 in order to be able to increment loop limiters.
    return false;
  }
  uint32_t one_id = context->get_constant_mgr()
                        ->GetDefiningInstruction(registered_one)
                        ->result_id();

  // Look for pointer-to-unsigned int type.
  opt::analysis::Pointer pointer_to_unsigned_int_type(
      registered_unsigned_int_type, SpvStorageClassFunction);
  uint32_t pointer_to_unsigned_int_type_id =
      context->get_type_mgr()->GetId(&pointer_to_unsigned_int_type);
  if (!pointer_to_unsigned_int_type_id) {
    // We need pointer-to-unsigned int in order to declare the loop limiter
    // variable.
    return false;
  }

  // Look for bool type.
  opt::analysis::Bool bool_type;
  uint32_t bool_type_id = context->get_type_mgr()->GetId(&bool_type);
  if (!bool_type_id) {
    // We need bool in order to compare the loop limiter's value with the loop
    // limit constant.
    return false;
  }

  // Declare the loop limiter variable at the start of the function's entry
  // block, via an instruction of the form:
  //   %loop_limiter_var = SpvOpVariable %ptr_to_uint Function %zero
  added_function->begin()->begin()->InsertBefore(MakeUnique<opt::Instruction>(
      context, SpvOpVariable, pointer_to_unsigned_int_type_id,
      message_.loop_limiter_variable_id(),
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}},
           {SPV_OPERAND_TYPE_ID, {zero_id}}})));
  // Update the module's id bound since we have added the loop limiter
  // variable id.
  fuzzerutil::UpdateModuleIdBound(context, message_.loop_limiter_variable_id());

  // Consider each loop in turn.
  for (auto block : loop_headers) {
    // Go through the sequence of loop limiter infos and find the one
    // corresponding to this loop.
    bool found = false;
    protobufs::LoopLimiterInfo loop_limiter_info;
    for (auto& info : message_.loop_limiter_info()) {
      if (info.loop_header_id() == block->id()) {
        loop_limiter_info = info;
        found = true;
        break;
      }
    }
    if (!found) {
      // We don't have loop limiter info for this loop header.
      return false;
    }

    // Suppose the loop header has the form:
    //
    // %l = OpLabel
    //      ... non-merge instructions ...
    //      OpLoopMerge %loop_merge %loop_continue Control
    //      terminator
    //
    // We will turn this into:
    //
    //  %l = OpLabel
    //       ... non-merge instructions ...
    // %t1 = OpLoad %uint32 %loop_limiter
    // %t2 = OpIAdd %uint32 %t1 %one
    //       OpStore %loop_limiter %t2
    // %t3 = OpUGreaterThanEqual %bool %t1 %loop_limit
    //       OpLoopMerge %loop_merge %loop_continue Control
    //       OpBranchConditional %t3 %loop_merge %new_block_id
    //
    // %new_block_id = OpLabel
    //       terminator

    // Find the merge instruction for the loop header.
    opt::Instruction* merge_inst_it = nullptr;
    opt::BasicBlock::iterator inst_it = block->begin();
    while (!inst_it->IsBlockTerminator()) {
      if (inst_it->opcode() == SpvOpLoopMerge) {
        merge_inst_it = &*inst_it;
      }
      ++inst_it;
    }
    assert(merge_inst_it && "A loop header has to have a merge instruction.");

    // Split the basic block right before |inst_it|, which is guaranteed to be
    // at the block's terminator.
    assert(inst_it->IsBlockTerminator() &&
           "We should have reached the block's terminator instruction.");
    block->SplitBasicBlock(context, loop_limiter_info.new_block_id(), inst_it);

    std::vector<std::unique_ptr<opt::Instruction>> new_instructions;

    // Add a load from the loop limiter variable, of the form:
    //   %t1 = OpLoad %uint32 %loop_limiter
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpLoad, unsigned_int_type_id, loop_limiter_info.load_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.loop_limiter_variable_id()}}})));

    // Increment the loaded value:
    //   %t2 = OpIAdd %uint32 %t1 %one
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpIAdd, unsigned_int_type_id,
        loop_limiter_info.increment_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {loop_limiter_info.load_id()}},
             {SPV_OPERAND_TYPE_ID, {one_id}}})));

    // Store the incremented value back to the loop limiter variable:
    //   OpStore %loop_limiter %t2
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpStore, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.loop_limiter_variable_id()}},
             {SPV_OPERAND_TYPE_ID, {loop_limiter_info.increment_id()}}})));

    // Compare the loaded value with the loop limit:
    //   %t3 = OpUGreaterThanEqual %bool %t1 %loop_limit
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpUGreaterThanEqual, bool_type_id,
        loop_limiter_info.compare_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {loop_limiter_info.load_id()}},
             {SPV_OPERAND_TYPE_ID, {message_.loop_limit_constant_id()}}})));

    // Add the new instructions before the merge block.
    merge_inst_it->InsertBefore(std::move(new_instructions));

    // Instead of the block's original terminator, add a conditional
    // branch to the loop's merge block (if the loop limit was reached),
    // or to a new block otherwise:
    //   OpBranchConditional %t3 %loop_merge %new_block_id
    uint32_t merge_block_id = merge_inst_it->GetSingleWordInOperand(0);
    block->AddInstruction(MakeUnique<opt::Instruction>(
        context, SpvOpBranchConditional, 0, 0,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {loop_limiter_info.compare_id()}},
             {SPV_OPERAND_TYPE_ID, {merge_block_id}},
             {SPV_OPERAND_TYPE_ID, {loop_limiter_info.new_block_id()}}})));

    // Update the module's id bound with respect to the various ids that
    // have been used for loop limiter manipulation.
    fuzzerutil::UpdateModuleIdBound(context, loop_limiter_info.load_id());
    fuzzerutil::UpdateModuleIdBound(context, loop_limiter_info.increment_id());
    fuzzerutil::UpdateModuleIdBound(context, loop_limiter_info.compare_id());
    fuzzerutil::UpdateModuleIdBound(context, loop_limiter_info.new_block_id());
  }
  return true;
}

bool TransformationAddFunction::TryToTurnKillOrUnreachableIntoReturn(
    opt::IRContext* context, opt::Function* added_function,
    opt::Instruction* kill_or_unreachable_inst) const {
  assert((kill_or_unreachable_inst->opcode() == SpvOpKill ||
          kill_or_unreachable_inst->opcode() == SpvOpUnreachable) &&
         "Precondition: instruction must be OpKill or OpUnreachable.");

  // Get the function's return type.
  auto function_return_type_inst =
      context->get_def_use_mgr()->GetDef(added_function->type_id());

  if (function_return_type_inst->opcode() == SpvOpTypeVoid) {
    // The function has void return type, so change this instruction to
    // OpReturn.
    kill_or_unreachable_inst->SetOpcode(SpvOpReturn);
  } else {
    // The function has non-void return type, so change this instruction
    // to OpReturnValue, using the value id provided with the
    // transformation.

    // We first check that the id, %id, provided with the transformation
    // specifically to turn OpKill and OpUnreachable instructions into
    // OpReturnValue %id has the same type as the function's return type.
    if (context->get_def_use_mgr()
            ->GetDef(message_.kill_unreachable_return_value_id())
            ->type_id() != function_return_type_inst->result_id()) {
      return false;
    }
    kill_or_unreachable_inst->SetOpcode(SpvOpReturnValue);
    kill_or_unreachable_inst->SetInOperands(
        {{SPV_OPERAND_TYPE_ID, {message_.kill_unreachable_return_value_id()}}});
  }
  return true;
}

bool TransformationAddFunction::TryToClampAccessChainIndices(
    opt::IRContext* context, opt::Instruction* access_chain_inst) const {
  assert((access_chain_inst->opcode() == SpvOpAccessChain ||
          access_chain_inst->opcode() == SpvOpInBoundsAccessChain) &&
         "Precondition: instruction must be OpAccessChain or "
         "OpInBoundsAccessChain.");

  // Find the AccessChainClampingInfo associated with this access chain.
  const protobufs::AccessChainClampingInfo* access_chain_clamping_info =
      nullptr;
  for (auto& clamping_info : message_.access_chain_clamping_info()) {
    if (clamping_info.access_chain_id() == access_chain_inst->result_id()) {
      access_chain_clamping_info = &clamping_info;
      break;
    }
  }
  if (!access_chain_clamping_info) {
    // No access chain clamping information was found; the function cannot be
    // made livesafe.
    return false;
  }

  // Check that there is a (compare_id, select_id) pair for every
  // index associated with the instruction.
  if (static_cast<uint32_t>(
          access_chain_clamping_info->compare_and_select_ids().size()) !=
      access_chain_inst->NumInOperands() - 1) {
    return false;
  }

  // Walk the access chain, clamping each index to be within bounds if it is
  // not a constant.
  auto base_object = context->get_def_use_mgr()->GetDef(
      access_chain_inst->GetSingleWordInOperand(0));
  assert(base_object && "The base object must exist.");
  auto pointer_type =
      context->get_def_use_mgr()->GetDef(base_object->type_id());
  assert(pointer_type && pointer_type->opcode() == SpvOpTypePointer &&
         "The base object must have pointer type.");
  auto should_be_composite_type = context->get_def_use_mgr()->GetDef(
      pointer_type->GetSingleWordInOperand(1));

  // Consider each index input operand in turn (operand 0 is the base object).
  for (uint32_t index = 1; index < access_chain_inst->NumInOperands();
       index++) {
    // Get the bound for the composite being indexed into; e.g. the number of
    // columns of matrix or the size of an array.
    uint32_t bound =
        GetBoundForCompositeIndex(context, *should_be_composite_type);

    // Get the instruction associated with the index and figure out its integer
    // type.
    const uint32_t index_id = access_chain_inst->GetSingleWordInOperand(index);
    auto index_inst = context->get_def_use_mgr()->GetDef(index_id);
    auto index_type_inst =
        context->get_def_use_mgr()->GetDef(index_inst->type_id());
    assert(index_type_inst->opcode() == SpvOpTypeInt);
    assert(index_type_inst->GetSingleWordInOperand(0) == 32);
    opt::analysis::Integer* index_int_type =
        context->get_type_mgr()
            ->GetType(index_type_inst->result_id())
            ->AsInteger();

    if (index_inst->opcode() != SpvOpConstant) {
      // The index is non-constant so we need to clamp it.
      assert(should_be_composite_type->opcode() != SpvOpTypeStruct &&
             "Access chain indices into structures are required to be "
             "constants.");
      opt::analysis::IntConstant bound_minus_one(index_int_type, {bound - 1});
      if (!context->get_constant_mgr()->FindConstant(&bound_minus_one)) {
        // We do not have an integer constant whose value is |bound| -1.
        return false;
      }

      opt::analysis::Bool bool_type;
      uint32_t bool_type_id = context->get_type_mgr()->GetId(&bool_type);
      if (!bool_type_id) {
        // Bool type is not declared; we cannot do a comparison.
        return false;
      }

      uint32_t bound_minus_one_id =
          context->get_constant_mgr()
              ->GetDefiningInstruction(&bound_minus_one)
              ->result_id();

      uint32_t compare_id =
          access_chain_clamping_info->compare_and_select_ids(index - 1).first();
      uint32_t select_id =
          access_chain_clamping_info->compare_and_select_ids(index - 1)
              .second();
      std::vector<std::unique_ptr<opt::Instruction>> new_instructions;

      // Compare the index with the bound via an instruction of the form:
      //   %t1 = OpULessThanEqual %bool %index %bound_minus_one
      new_instructions.push_back(MakeUnique<opt::Instruction>(
          context, SpvOpULessThanEqual, bool_type_id, compare_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {index_inst->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));

      // Select the index if in-bounds, otherwise one less than the bound:
      //   %t2 = OpSelect %int_type %t1 %index %bound_minus_one
      new_instructions.push_back(MakeUnique<opt::Instruction>(
          context, SpvOpSelect, index_type_inst->result_id(), select_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {compare_id}},
               {SPV_OPERAND_TYPE_ID, {index_inst->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));

      // Add the new instructions before the access chain
      access_chain_inst->InsertBefore(std::move(new_instructions));

      // Replace %index with %t2.
      access_chain_inst->SetInOperand(index, {select_id});
      fuzzerutil::UpdateModuleIdBound(context, compare_id);
      fuzzerutil::UpdateModuleIdBound(context, select_id);
    } else {
      // Assert that the index is smaller (unsigned) than this value.
      // Return false if it is not (to keep compilers happy).
      if (index_inst->GetSingleWordInOperand(0) >= bound) {
        assert(false &&
               "The function has a statically out-of-bounds access; "
               "this should not occur.");
        return false;
      }
    }
    should_be_composite_type =
        FollowCompositeIndex(context, *should_be_composite_type, index_id);
  }
  return true;
}

uint32_t TransformationAddFunction::GetBoundForCompositeIndex(
    opt::IRContext* context, const opt::Instruction& composite_type_inst) {
  switch (composite_type_inst.opcode()) {
    case SpvOpTypeArray:
      return fuzzerutil::GetArraySize(composite_type_inst, context);
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
      return composite_type_inst.GetSingleWordInOperand(1);
    case SpvOpTypeStruct: {
      return fuzzerutil::GetNumberOfStructMembers(composite_type_inst);
    }
    default:
      assert(false && "Unknown composite type.");
      return 0;
  }
}

opt::Instruction* TransformationAddFunction::FollowCompositeIndex(
    opt::IRContext* context, const opt::Instruction& composite_type_inst,
    uint32_t index_id) {
  uint32_t sub_object_type_id;
  switch (composite_type_inst.opcode()) {
    case SpvOpTypeArray:
      sub_object_type_id = composite_type_inst.GetSingleWordInOperand(0);
      break;
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
      sub_object_type_id = composite_type_inst.GetSingleWordInOperand(0);
      break;
    case SpvOpTypeStruct: {
      auto index_inst = context->get_def_use_mgr()->GetDef(index_id);
      assert(index_inst->opcode() == SpvOpConstant);
      assert(
          context->get_def_use_mgr()->GetDef(index_inst->type_id())->opcode() ==
          SpvOpTypeInt);
      assert(context->get_def_use_mgr()
                 ->GetDef(index_inst->type_id())
                 ->GetSingleWordInOperand(0) == 32);
      uint32_t index_value = index_inst->GetSingleWordInOperand(0);
      sub_object_type_id =
          composite_type_inst.GetSingleWordInOperand(index_value);
      break;
    }
    default:
      assert(false && "Unknown composite type.");
      sub_object_type_id = 0;
      break;
  }
  assert(sub_object_type_id && "No sub-object found.");
  return context->get_def_use_mgr()->GetDef(sub_object_type_id);
}

}  // namespace fuzz
}  // namespace spvtools
