// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/fuzzer_pass_add_equation_instructions.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_equation_instruction.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddEquationInstructions::FuzzerPassAddEquationInstructions(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassAddEquationInstructions::~FuzzerPassAddEquationInstructions() =
    default;

void FuzzerPassAddEquationInstructions::Apply() {
  MaybeAddTransformationBeforeEachInstruction(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingEquationInstruction())) {
          return;
        }

        // Check that it is OK to add an equation instruction before the given
        // instruction in principle - e.g. check that this does not lead to
        // inserting before an OpVariable or OpPhi instruction.  We use OpIAdd
        // as an example opcode for this check, to be representative of *some*
        // opcode that defines an equation, even though we may choose a
        // different opcode below.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpIAdd, inst_it)) {
          return;
        }

        // Get all available instructions with result ids and types that are not
        // OpUndef.
        std::vector<opt::Instruction*> available_instructions =
            FindAvailableInstructions(
                function, block, inst_it,
                [](opt::IRContext*, opt::Instruction* instruction) -> bool {
                  return instruction->result_id() && instruction->type_id() &&
                         instruction->opcode() != SpvOpUndef;
                });

        // Try the opcodes for which we know how to make ids at random until
        // something works.
        std::vector<SpvOp> candidate_opcodes = {SpvOpIAdd, SpvOpISub,
                                                SpvOpLogicalNot, SpvOpSNegate};
        do {
          auto opcode =
              GetFuzzerContext()->RemoveAtRandomIndex(&candidate_opcodes);
          switch (opcode) {
            case SpvOpIAdd:
            case SpvOpISub: {
              auto integer_instructions =
                  GetIntegerInstructions(available_instructions);
              if (!integer_instructions.empty()) {
                auto lhs = integer_instructions.at(
                    GetFuzzerContext()->RandomIndex(integer_instructions));
                auto lhs_type =
                    GetIRContext()->get_type_mgr()->GetType(lhs->type_id());
                auto candidate_rhs_instructions = RestrictToWidth(
                    integer_instructions,
                    lhs_type->AsVector() ? lhs_type->AsVector()->element_count()
                                         : 1);
                auto rhs = candidate_rhs_instructions.at(
                    GetFuzzerContext()->RandomIndex(
                        candidate_rhs_instructions));
                ApplyTransformation(TransformationEquationInstruction(
                    GetFuzzerContext()->GetFreshId(), opcode,
                    {lhs->result_id(), rhs->result_id()},
                    instruction_descriptor));
                return;
              }
              break;
            }
            case SpvOpLogicalNot: {
              auto boolean_instructions =
                  GetBooleanInstructions(available_instructions);
              if (!boolean_instructions.empty()) {
                ApplyTransformation(TransformationEquationInstruction(
                    GetFuzzerContext()->GetFreshId(), opcode,
                    {boolean_instructions
                         .at(GetFuzzerContext()->RandomIndex(
                             boolean_instructions))
                         ->result_id()},
                    instruction_descriptor));
                return;
              }
              break;
            }
            case SpvOpSNegate: {
              auto integer_instructions =
                  GetIntegerInstructions(available_instructions);
              if (!integer_instructions.empty()) {
                ApplyTransformation(TransformationEquationInstruction(
                    GetFuzzerContext()->GetFreshId(), opcode,
                    {integer_instructions
                         .at(GetFuzzerContext()->RandomIndex(
                             integer_instructions))
                         ->result_id()},
                    instruction_descriptor));
                return;
              }
              break;
            }
            default:
              assert(false && "Unexpected opcode.");
              break;
          }
        } while (!candidate_opcodes.empty());
        // Reaching here means that we did not manage to apply any
        // transformation at this point of the module.
      });
}

std::vector<opt::Instruction*>
FuzzerPassAddEquationInstructions::GetIntegerInstructions(
    const std::vector<opt::Instruction*>& instructions) const {
  std::vector<opt::Instruction*> result;
  for (auto& inst : instructions) {
    auto type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
    if (type->AsInteger() ||
        (type->AsVector() && type->AsVector()->element_type()->AsInteger())) {
      result.push_back(inst);
    }
  }
  return result;
}

std::vector<opt::Instruction*>
FuzzerPassAddEquationInstructions::GetBooleanInstructions(
    const std::vector<opt::Instruction*>& instructions) const {
  std::vector<opt::Instruction*> result;
  for (auto& inst : instructions) {
    auto type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
    if (type->AsBool() ||
        (type->AsVector() && type->AsVector()->element_type()->AsBool())) {
      result.push_back(inst);
    }
  }
  return result;
}

std::vector<opt::Instruction*>
FuzzerPassAddEquationInstructions::RestrictToWidth(
    const std::vector<opt::Instruction*>& instructions, uint32_t width) const {
  std::vector<opt::Instruction*> result;
  for (auto& inst : instructions) {
    auto type = GetIRContext()->get_type_mgr()->GetType(inst->type_id());
    if ((width == 1 && !type->AsVector()) ||
        (width > 1 && type->AsVector()->element_count() == width)) {
      result.push_back(inst);
    }
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
