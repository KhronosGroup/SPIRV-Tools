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
    const spvtools::fuzz::FactManager& /*unused*/) const {
  std::set<uint32_t> ids_used_by_this_transformation;
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

  if (message_.is_livesafe()) {
    std::vector<uint32_t> loop_header_ids = GetLoopHeaderIds();

    if (!loop_header_ids.empty()) {
      auto loop_limiter_type_and_constant_ids =
          GetLoopLimiterTypeAndConstantIds(context);
      if (!loop_limiter_type_and_constant_ids.unsigned_int_type) {
        // Unsigned 32-bit int type must exist.
        return false;
      }
      if (!loop_limiter_type_and_constant_ids.pointer_to_unsigned_int_type) {
        // Pointer to unsigned 32-bit int type must exist.
        return false;
      }
      if (!loop_limiter_type_and_constant_ids.bool_type) {
        // Bool type must exist.
        return false;
      }
      if (!loop_limiter_type_and_constant_ids.zero) {
        // The value 0, of unsigned 32-bit in type, must exist.
        return false;
      }
      if (!loop_limiter_type_and_constant_ids.one) {
        // The value 1, of unsigned 32-bit in type, must exist.
        return false;
      }
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

      // Collect all the loop header ids for which loop limiter information has
      // been supplied.
      std::set<uint32_t> loop_headers_for_which_limiters_are_supplied;
      for (auto& loop_limiter : message_.loop_limiter_info()) {
        loop_headers_for_which_limiters_are_supplied.insert(
            loop_limiter.loop_header_id());
      }
      for (auto loop_header_id : loop_header_ids) {
        if (loop_headers_for_which_limiters_are_supplied.count(
                loop_header_id) == 0) {
          // We don't have loop limiter info for this loop header.
          return false;
        }
      }
    }

    // TODO requires commenting.
    if (FunctionContainsKillOrUnreachable()) {
      auto function_return_type_inst = context->get_def_use_mgr()->GetDef(
          message_.instruction(0).result_type_id());
      if (!function_return_type_inst) {
        return false;
      }
      if (function_return_type_inst->opcode() != SpvOpTypeVoid) {
        if (context->get_def_use_mgr()
                ->GetDef(message_.kill_unreachable_return_value_id())
                ->type_id() != function_return_type_inst->result_id()) {
          return false;
        }
      }
    }

    // TODO comment
    for (auto& instruction : message_.instruction()) {
      switch (instruction.opcode()) {
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain: {
          const protobufs::AccessChainClampingInfo* access_chain_clamping_info =
              nullptr;
          for (auto& clamping_info : message_.access_chain_clamping_info()) {
            if (clamping_info.access_chain_id() == instruction.result_id()) {
              access_chain_clamping_info = &clamping_info;
            }
          }
          if (!access_chain_clamping_info) {
            return false;
          }
          if (access_chain_clamping_info->compare_and_select_ids().size() !=
              instruction.input_operand().size() - 1) {
            return false;
          }
        } break;
        default:
          break;
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
  // TODO revise comment since we do not return here
  // Having managed to add the new function to the cloned module, we ascertain
  // whether the cloned module is still valid.  If it is, the transformation is
  // applicable.
  if (!fuzzerutil::IsValid(cloned_module.get())) {
    return false;
  }

  // TODO comment
  if (message_.is_livesafe()) {
    if (!TryToMakeFunctionLivesafe(cloned_module.get())) {
      return false;
    }
  }

  return true;
}

void TransformationAddFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* fact_manager) const {
  auto success = TryToAddFunction(context);
  assert(success && "The function should be successfully added.");
  (void)(success);  // Keep release builds happy (otherwise they may complain
                    // that |success| is not used).

  if (message_.is_livesafe()) {
    TryToMakeFunctionLivesafe(context);
    assert(message_.instruction(0).opcode() == SpvOpFunction &&
           "The first instruction of an 'add function' transformation must be "
           "OpFunction.");
    // Inform the fact manager that the function is livesafe.
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

  // If |message_.loop_limiter| holds, the function is made livesafe.
  // TODO: Right now only loop limiters are added.  Bounds clamping and
  //  removal of OpKill and OpUnreachable will come next.  Further care will be
  //  needed to handle constructs that can only be invoked under uniform control
  //  flow, if violating such rules has 'catch fire' semantics.

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

TransformationAddFunction::LoopLimiterTypeAndConstantIds
TransformationAddFunction::GetLoopLimiterTypeAndConstantIds(
    opt::IRContext* context) const {
  // Set all fields to 0 in case they turn out to be unavailable.
  LoopLimiterTypeAndConstantIds result = {0, 0, 0, 0, 0};

  // Find the id of the "unsigned int" type.
  opt::analysis::Integer unsigned_int_type(32, false);
  result.unsigned_int_type = context->get_type_mgr()->GetId(&unsigned_int_type);
  if (!result.unsigned_int_type) {
    // Unsigned int is not available, and so no constant of this type, nor a
    // pointer to it, can be either.
    return result;
  }
  auto registered_unsigned_int_type =
      context->get_type_mgr()->GetRegisteredType(&unsigned_int_type);

  // Look for 0, adding its id if found.
  opt::analysis::IntConstant zero(registered_unsigned_int_type->AsInteger(),
                                  {0});
  auto registered_zero = context->get_constant_mgr()->FindConstant(&zero);
  if (registered_zero) {
    result.zero = context->get_constant_mgr()
                      ->GetDefiningInstruction(registered_zero)
                      ->result_id();
  }

  // Look for 1, adding its id if found.
  opt::analysis::IntConstant one(registered_unsigned_int_type->AsInteger(),
                                 {1});
  auto registered_one = context->get_constant_mgr()->FindConstant(&one);
  if (registered_one) {
    result.one = context->get_constant_mgr()
                     ->GetDefiningInstruction(registered_one)
                     ->result_id();
  }

  // Look for pointer-to-unsigned int type, adding its id if found.
  opt::analysis::Pointer pointer_to_unsigned_int_type(
      registered_unsigned_int_type, SpvStorageClassFunction);
  result.pointer_to_unsigned_int_type =
      context->get_type_mgr()->GetId(&pointer_to_unsigned_int_type);

  // Look for bool type, adding its id if found.
  opt::analysis::Bool bool_type;
  result.bool_type = context->get_type_mgr()->GetId(&bool_type);

  return result;
}

std::vector<uint32_t> TransformationAddFunction::GetLoopHeaderIds() const {
  std::vector<uint32_t> result;
  uint32_t last_label = 0;
  // Check whether every loop header in the function has associated loop
  // limiter information.
  for (auto& instruction : message_.instruction()) {
    switch (instruction.opcode()) {
      case SpvOpLabel:
        // Track the latest block that was encountered.
        last_label = instruction.result_id();
        break;
      case SpvOpLoopMerge:
        // When a loop merge is found, the latest block must be a loop
        // header.
        result.push_back(last_label);
        break;
      default:
        break;
    }
  }
  return result;
}

bool TransformationAddFunction::FunctionContainsKillOrUnreachable() const {
  for (auto& instruction : message_.instruction()) {
    switch (instruction.opcode()) {
      case SpvOpKill:
      case SpvOpUnreachable:
        return true;
      default:
        break;
    }
  }
  return false;
}

bool TransformationAddFunction::TryToMakeFunctionLivesafe(
    opt::IRContext* context) const {
  opt::Function* added_function = nullptr;
  for (auto& function : *context->module()) {
    if (function.result_id() == message_.instruction(0).result_id()) {
      added_function = &function;
      break;
    }
  }
  assert(added_function && "The added function should have been found.");

  AddLoopLimiters(context, added_function);

  // TODO comment
  for (auto& block : *added_function) {
    for (auto& inst : block) {
      switch (inst.opcode()) {
        case SpvOpKill:
        case SpvOpUnreachable:
          TurnKillOrUnreachableIntoReturn(context, added_function, &inst);
          break;
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain:
          if (!TryToClampAccessChainIndices(context, &inst)) {
            return false;
          }
          break;
        default:
          break;
      }
    }
  }
  return true;
}

void TransformationAddFunction::AddLoopLimiters(
    opt::IRContext* context, opt::Function* added_function) const {
  auto loop_limiter_types_and_ids = GetLoopLimiterTypeAndConstantIds(context);

  if (!GetLoopHeaderIds().empty()) {
    for (auto inst_it = added_function->begin()->begin();; ++inst_it) {
      if (inst_it->opcode() == SpvOpVariable) {
        continue;
      }
      // The current instruction is the first instruction in the function's
      // entry block that is not OpVariable.  This is the right place to add the
      // loop limiter variable.

      // Add an instruction of the form:
      // %loop_limiter_var = SpvOpVariable %ptr_to_uint Function %zero
      inst_it->InsertBefore(MakeUnique<opt::Instruction>(
          context, SpvOpVariable,
          loop_limiter_types_and_ids.pointer_to_unsigned_int_type,
          message_.loop_limiter_variable_id(),
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassFunction}},
               {SPV_OPERAND_TYPE_ID, {loop_limiter_types_and_ids.zero}}})));
      // Update the module's id bound since we have added the loop limiter
      // variable id.
      fuzzerutil::UpdateModuleIdBound(context,
                                      message_.loop_limiter_variable_id());

      break;
    }
  }

  // Collect up all the loop headers so that we can subsequently add loop
  // limiting logic.  As that logic involves splitting blocks, we cannot
  // safely do it on-the-fly.
  std::vector<opt::BasicBlock*> loop_headers;
  for (auto& block : *added_function) {
    if (block.IsLoopHeader()) {
      loop_headers.push_back(&block);
    }
  }

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
    assert(found && "There should be a loop limiter info for every loop.");

    opt::Instruction* merge_inst_it = nullptr;
    opt::BasicBlock::iterator inst_it = block->begin();
    while (!inst_it->IsBlockTerminator()) {
      if (inst_it->opcode() == SpvOpLoopMerge) {
        merge_inst_it = &*inst_it;
      }
      ++inst_it;
    }
    assert(merge_inst_it && "A loop header has to have a merge instruction.");
    block->SplitBasicBlock(context, loop_limiter_info.new_block_id(), inst_it);

    std::vector<std::unique_ptr<opt::Instruction>> new_instructions;
    // Add a load from the loop limiter variable, of the form:
    //   %t1 = OpLoad %uint32 %loop_limiter
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpLoad, loop_limiter_types_and_ids.unsigned_int_type,
        loop_limiter_info.load_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {message_.loop_limiter_variable_id()}}})));
    // Increment the loaded value:
    //   %t2 = OpIAdd %uint32 %t1 %one
    new_instructions.push_back(MakeUnique<opt::Instruction>(
        context, SpvOpIAdd, loop_limiter_types_and_ids.unsigned_int_type,
        loop_limiter_info.increment_id(),
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {loop_limiter_info.load_id()}},
             {SPV_OPERAND_TYPE_ID, {loop_limiter_types_and_ids.one}}})));
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
        context, SpvOpUGreaterThanEqual, loop_limiter_types_and_ids.bool_type,
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
}

void TransformationAddFunction::TurnKillOrUnreachableIntoReturn(
    opt::IRContext* context, opt::Function* added_function,
    opt::Instruction* kill_or_unreachable_inst) const {
  assert((kill_or_unreachable_inst->opcode() == SpvOpKill ||
          kill_or_unreachable_inst->opcode() == SpvOpUnreachable) &&
         "Precondition: instruction must be OpKill or OpUnreachable.");
  if (context->get_def_use_mgr()->GetDef(added_function->type_id())->opcode() ==
      SpvOpTypeVoid) {
    // The function has void return type, so change this instruction to
    // OpReturn.
    kill_or_unreachable_inst->SetOpcode(SpvOpReturn);
  } else {
    // The function has non-void return type, so change this insruction
    // to OpReturnValue, using the value id provided with the
    // transformation.
    kill_or_unreachable_inst->SetOpcode(SpvOpReturnValue);
    kill_or_unreachable_inst->SetInOperands(
        {{SPV_OPERAND_TYPE_ID, {message_.kill_unreachable_return_value_id()}}});
  }
}

bool TransformationAddFunction::TryToClampAccessChainIndices(
    opt::IRContext* context, opt::Instruction* access_chain_inst) const {
  assert((access_chain_inst->opcode() == SpvOpAccessChain ||
          access_chain_inst->opcode() == SpvOpInBoundsAccessChain) &&
         "Precondition: instruction must be OpAccessChain or "
         "OpInBoundsAccessChain.");
  const protobufs::AccessChainClampingInfo* access_chain_clamping_info =
      nullptr;
  for (auto& clamping_info : message_.access_chain_clamping_info()) {
    if (clamping_info.access_chain_id() == access_chain_inst->result_id()) {
      access_chain_clamping_info = &clamping_info;
      break;
    }
  }
  assert(access_chain_clamping_info &&
         "An access chain clamping info object should have been found "
         "for this access chain.");

  auto base_object = context->get_def_use_mgr()->GetDef(
      access_chain_inst->GetSingleWordInOperand(0));
  assert(base_object && "The base object must exist.");
  auto pointer_type =
      context->get_def_use_mgr()->GetDef(base_object->type_id());
  assert(pointer_type && pointer_type->opcode() == SpvOpTypePointer &&
         "The base object must have pointer type.");
  auto should_be_composite_type = context->get_def_use_mgr()->GetDef(
      pointer_type->GetSingleWordInOperand(1));
  for (uint32_t index = 1; index < access_chain_inst->NumInOperands();
       index++) {
    uint32_t bound =
        GetBoundForCompositeIndex(context, *should_be_composite_type);
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

      // TODO comment using spirv-assembly
      new_instructions.push_back(MakeUnique<opt::Instruction>(
          context, SpvOpULessThanEqual, bool_type_id, compare_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {index_inst->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));
      // TODO comment using spirv-assembly
      new_instructions.push_back(MakeUnique<opt::Instruction>(
          context, SpvOpSelect, index_type_inst->result_id(), select_id,
          opt::Instruction::OperandList(
              {{SPV_OPERAND_TYPE_ID, {compare_id}},
               {SPV_OPERAND_TYPE_ID, {index_inst->result_id()}},
               {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));
      access_chain_inst->InsertBefore(std::move(new_instructions));
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
