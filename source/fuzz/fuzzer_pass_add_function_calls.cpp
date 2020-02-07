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

#include "source/fuzz/fuzzer_pass_add_function_calls.h"

#include "source/fuzz/call_graph.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_function_call.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddFunctionCalls::FuzzerPassAddFunctionCalls(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassAddFunctionCalls::~FuzzerPassAddFunctionCalls() = default;

void FuzzerPassAddFunctionCalls::Apply() {
  MaybeAddTransformationBeforeEachInstruction(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        // Check whether it is legitimate to insert a function call before the
        // instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpFunctionCall,
                                                          inst_it)) {
          return;
        }

        // Randomly decide whether to try inserting a function call here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfCallingFunction())) {
          return;
        }
        CallGraph call_graph(GetIRContext());
        std::vector<opt::Function*> candidate_functions;
        for (auto& other_function : *GetIRContext()->module()) {
          candidate_functions.push_back(&other_function);
        }
        while (!candidate_functions.empty()) {
          opt::Function* candidate_function =
              GetFuzzerContext()->RemoveAtRandomIndex(&candidate_functions);
          if (TransformationFunctionCall::FunctionIsEntryPoint(
                  GetIRContext(), candidate_function->result_id())) {
            // This function is an entry point, which cannot also be the target
            // of OpFunctionCall
            continue;
          }

          if (candidate_function == function) {
            // Calling this function would lead to direct recursion
            continue;
          }
          if (!GetFactManager()->BlockIsDead(block->id()) &&
              !GetFactManager()->FunctionIsLivesafe(
                  candidate_function->result_id())) {
            // Unless in a dead block, only livesafe functions can be invoked
            continue;
          }
          if (call_graph.GetIndirectCallees(candidate_function->result_id())
                  .count(function->result_id())) {
            // Calling this function could lead to indirect recursion
            continue;
          }
          // Find all instructions in scope that could be used for a call; weed
          // out unsuitable pointer arguments immediately.
          std::vector<opt::Instruction*> potentially_suitable_instructions =
              FindAvailableInstructions(
                  function, block, inst_it,
                  [this, block](opt::IRContext* context,
                                opt::Instruction* inst) -> bool {
                    if (!inst->HasResultId()) {
                      // TODO revise We can only make a synonym of an
                      // instruction
                      //  that generates an id.
                      return false;
                    }
                    if (!inst->type_id()) {
                      // TODO revise We can only make a synonym of an
                      // instruction that has a type.
                      return false;
                    }
                    if (context->get_def_use_mgr()
                            ->GetDef(inst->type_id())
                            ->opcode() == SpvOpTypePointer) {
                      switch (inst->opcode()) {
                        case SpvOpFunctionParameter:
                        case SpvOpVariable:
                          break;
                        default:
                          return false;
                      }
                      if (!GetFactManager()->BlockIsDead(block->id()) &&
                          !GetFactManager()->PointeeValueIsIrrelevant(
                              inst->result_id())) {
                        return false;
                      }
                    }
                    return true;
                  });

          std::map<uint32_t, std::vector<opt::Instruction*>>
              type_to_available_instructions;
          for (auto inst : potentially_suitable_instructions) {
            if (type_to_available_instructions.count(inst->type_id()) == 0) {
              type_to_available_instructions.insert({inst->type_id(), {}});
            }
            type_to_available_instructions.at(inst->type_id()).push_back(inst);
          }

          opt::Instruction* function_type =
              GetIRContext()->get_def_use_mgr()->GetDef(
                  candidate_function->DefInst().GetSingleWordInOperand(1));
          assert(function_type->opcode() == SpvOpTypeFunction &&
                 "The function type does not have the expected opcode.");
          std::vector<uint32_t> arg_ids;
          for (uint32_t arg_index = 1;
               arg_index < function_type->NumInOperands(); arg_index++) {
            auto arg_type_id =
                GetIRContext()
                    ->get_def_use_mgr()
                    ->GetDef(function_type->GetSingleWordInOperand(arg_index))
                    ->result_id();
            if (type_to_available_instructions.count(arg_type_id)) {
              std::vector<opt::Instruction*> candidate_arguments =
                  type_to_available_instructions.at(arg_type_id);
              arg_ids.push_back(
                  candidate_arguments[GetFuzzerContext()->RandomIndex(
                                          candidate_arguments)]
                      ->result_id());
            } else {
              // We don't have a suitable id in scope to pass, so we must make
              // something up.
              auto type_instruction =
                  GetIRContext()->get_def_use_mgr()->GetDef(arg_type_id);

              if (type_instruction->opcode() == SpvOpTypePointer) {
                // In the case of a pointer, we make a new variable, at function
                // or global scope depending on the storage class of the
                // pointer.

                // Get a fresh id for the new variable.
                uint32_t fresh_variable_id = GetFuzzerContext()->GetFreshId();

                // The id of this variable is what we pass as the parameter to
                // the call.
                arg_ids.push_back(fresh_variable_id);

                // Now bring the variable into existence.
                if (type_instruction->GetSingleWordInOperand(0) ==
                    SpvStorageClassFunction) {
                  // Add a new zero-initialized local variable to the current
                  // function, noting that its pointee value is irrelevant.
                  ApplyTransformation(TransformationAddLocalVariable(
                      fresh_variable_id, arg_type_id, function->result_id(),
                      FindOrCreateZeroConstant(
                          type_instruction->GetSingleWordInOperand(1)),
                      true));
                } else {
                  assert(type_instruction->GetSingleWordInOperand(0) ==
                             SpvStorageClassPrivate &&
                         "Only Function and Private storage classes are "
                         "supported at present.");
                  // Add a new zero-initialized global variable to the module,
                  // noting that its pointee value is irrelevant.
                  ApplyTransformation(TransformationAddGlobalVariable(
                      fresh_variable_id, arg_type_id,
                      FindOrCreateZeroConstant(
                          type_instruction->GetSingleWordInOperand(1)),
                      true));
                }
              } else {
                // TODO need issue: we should have a way of noting that this
                //  zero value is arbitrary and could be fuzzed by other
                //  passes.
                arg_ids.push_back(FindOrCreateZeroConstant(arg_type_id));
              }
            }
          }
          ApplyTransformation(TransformationFunctionCall(
              GetFuzzerContext()->GetFreshId(), candidate_function->result_id(),
              arg_ids, instruction_descriptor));
        }
      });
}

}  // namespace fuzz
}  // namespace spvtools
