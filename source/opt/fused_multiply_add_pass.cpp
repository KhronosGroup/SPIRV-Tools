

#include "source/opt/fused_multiply_add_pass.h"

#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>

#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status FusedMultiplyAddPass::Process() {
  Status status = Status::SuccessWithoutChange;
  auto ctx = context();
  auto module = get_module();

  uint32_t instSetId = ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  assert(instSetId != 0);
  
  for (auto& func : *module) {
    for (auto& block : func) {

      auto iter = block.begin();
      while (iter != block.end()) {
        auto* instruction = &(*iter);
        switch(instruction->opcode())
        {
          case SpvOpFMul:
			      get_def_use_mgr()->WhileEachUser( instruction, [instSetId, instruction, ctx](Instruction* use) {
      				if (use->opcode() == SpvOpFAdd && use->type_id() == instruction->type_id()) {
                InstructionBuilder builder(ctx, use, IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
                
                std::vector<Operand> operands;
                operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {instSetId}});
                operands.push_back({SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {GLSLstd450Fma}});
                operands.push_back(instruction->GetOperand(2));
                operands.push_back(instruction->GetOperand(3));

                auto const& AddOperandL = use->GetOperand(2);
                auto const& AddOperandR = use->GetOperand(3);
                if (AddOperandL.words.size() == 1 && AddOperandL.words[0] == instruction->result_id())
                {
                  operands.push_back(AddOperandR);
                }
                else
                {
                  operands.push_back(AddOperandL);
                }

                std::unique_ptr<Instruction> new_inst(
                  new Instruction(builder.GetContext(), SpvOpExtInst, use->type_id(),
                  builder.GetContext()->TakeNextId(), operands));
                uint32_t result_id = new_inst->result_id();
                assert(result_id != 0);
                builder.AddInstruction(std::move(new_inst));

                ctx->ReplaceAllUsesWith(use->result_id(), result_id);
				      }
				      return true;
			      });
            break;
          default:
            break;
        }

        ++iter;
      }
    }
  }
//  for (auto& f : *get_module()) {
//    Status functionStatus = ProcessFunction(&f);
//    if (functionStatus == Status::Failure)
//      return functionStatus;
//    else if (functionStatus == Status::SuccessWithChange)
//      status = functionStatus;
//  }

  return status;
}

}  // namespace opt
}  // namespace spvtools
