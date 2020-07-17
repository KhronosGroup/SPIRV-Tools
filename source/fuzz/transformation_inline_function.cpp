// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_inline_function.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationInlineFunction::TransformationInlineFunction(
    const spvtools::fuzz::protobufs::TransformationInlineFunction& message)
    : message_(message) {}

TransformationInlineFunction::TransformationInlineFunction(
    const std::map<uint32_t, uint32_t>& result_id_map,
    uint32_t function_call_id) {
  *message_.mutable_result_id_map() =
      google::protobuf::Map<google::protobuf::uint32, google::protobuf::uint32>(
          result_id_map.begin(), result_id_map.end());
  message_.set_function_call_id(function_call_id);
}

bool TransformationInlineFunction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The values in the |message_.result_id_map| must be all fresh and all
  // distinct.
  std::set<uint32_t> ids_used_by_this_transformation;
  for (auto& pair : message_.result_id_map()) {
    if (!CheckIdIsFreshAndNotUsedByThisTransformation(
            pair.second, ir_context, &ids_used_by_this_transformation)) {
      return false;
    }
  }

  // |function_call_instruction| must be defined and must be an OpFunctionCall
  // instruction.
  auto function_call_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.function_call_id());
  if (function_call_instruction == nullptr ||
      function_call_instruction->opcode() != SpvOpFunctionCall) {
    return false;
  }

  // The called function must not have an early return.
  auto called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));
  if (called_function->HasEarlyReturn()) {
    return false;
  }

  // |message_.result_id_map| must have an entry for every result id in the
  // called function.
  for (auto& block : *called_function) {
    // Since the entry block label will not be inlined, only the remaining
    // labels must have a corresponding value in the map.
    if (&block != &*called_function->entry() &&
        !message_.result_id_map().count(block.GetLabel()->result_id())) {
      return false;
    }
    for (auto& instruction : block) {
      if (instruction.HasResultId() &&
          instruction.result_id() != called_function->GetReturnValueId() &&
          !message_.result_id_map().count(instruction.result_id())) {
        return false;
      }
    }
  }

  return true;
}

void TransformationInlineFunction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto function_call_instruction =
      ir_context->get_def_use_mgr()->GetDef(message_.function_call_id());
  auto caller_function =
      ir_context->get_instr_block(function_call_instruction)->GetParent();
  auto called_function = fuzzerutil::FindFunction(
      ir_context, function_call_instruction->GetSingleWordInOperand(0));

  for (auto& block : *called_function) {
    // The called function entry block label will not be inlined.
    if (&block != &*called_function->entry()) {
      auto cloned_label_instruction = block.GetLabelInst()->Clone(ir_context);
      uint32_t fresh_id =
          message_.result_id_map().at(cloned_label_instruction->result_id());
      cloned_label_instruction->SetResultId(fresh_id);
      function_call_instruction->InsertBefore(
          std::unique_ptr<opt::Instruction>(cloned_label_instruction));
      fuzzerutil::UpdateModuleIdBound(ir_context, fresh_id);
    }

    for (auto& instruction : block) {
      // The function return instruction will not be inlined.
      if (&instruction == &*called_function->tail()->tail()) {
        continue;
      }

      // Replaces the operand ids with their mapped result ids.
      auto cloned_instruction = instruction.Clone(ir_context);
      cloned_instruction->ForEachInId(
          [this, called_function, function_call_instruction](uint32_t* id) {
            // If the id is mapped, then set it to its mapped value.
            if (message_.result_id_map().count(*id)) {
              *id = message_.result_id_map().at(*id);
              return;
            }

            uint32_t parameter_index = 0;
            called_function->ForEachParam(
                [id, function_call_instruction,
                 &parameter_index](opt::Instruction* parameter_instruction) {
                  // If the id is a function parameter, then set it to the
                  // parameter value passed in the function call instruction.
                  if (*id == parameter_instruction->result_id()) {
                    *id = function_call_instruction->GetSingleWordInOperand(
                        parameter_index + 1);
                  }
                  parameter_index++;
                });
          });

      // If |cloned_instruction| has a result id, then set it to its mapped
      // value.
      if (cloned_instruction->HasResultId() &&
          message_.result_id_map().count(cloned_instruction->result_id())) {
        uint32_t result_id =
            message_.result_id_map().at(cloned_instruction->result_id());
        cloned_instruction->SetResultId(result_id);
        fuzzerutil::UpdateModuleIdBound(ir_context, result_id);
      }

      // If the called function return instruction is OpReturnValue and the
      // |cloned_instruction| result id is the returned value, then sets the
      // |cloned_instruction| result id to the function call result id.
      if (cloned_instruction->result_id() ==
          called_function->GetReturnValueId()) {
        cloned_instruction->SetResultId(function_call_instruction->result_id());
      }

      if (cloned_instruction->opcode() == SpvOpVariable) {
        // All OpVariable instructions in a function must be in the first block
        // in the function.
        caller_function->begin()->begin()->InsertBefore(
            std::unique_ptr<opt::Instruction>(cloned_instruction));
      } else {
        function_call_instruction->InsertBefore(
            std::unique_ptr<opt::Instruction>(cloned_instruction));
      }
    }
  }

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationInlineFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_inline_function() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
