

#include "source/opt/fused_multiply_add_pass.h"

#include <algorithm>
#include <queue>
#include <tuple>
#include <utility>

#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status FusedMultiplyAddPass::Process() {
  Status status = Status::SuccessWithoutChange;
  auto ctx = context();
  auto module = get_module();

  uint32_t inst_set_id = ctx->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  assert(inst_set_id != 0);

  for (auto& func : *module) {
    for (auto& block : func) {
      for (auto iter = block.begin(); iter != block.end(); ++iter) {
        auto* instruction = &(*iter);
        switch (instruction->opcode()) {
          case SpvOpFMul:
            if (ProcessSpvOpFMul(ctx, instruction, inst_set_id)) {
              status = Status::SuccessWithChange;
            }
            break;
          default:
            break;
        }
      }
    }
  }

  return status;
}

bool FusedMultiplyAddPass::ProcessSpvOpFMul(IRContext* ctx,
                                            Instruction* instruction,
                                            uint32_t inst_set_id) {
  bool modified = false;
  get_def_use_mgr()->ForEachUser(
    instruction,
    [inst_set_id, instruction, ctx, &modified](Instruction* use) {
      if (use->opcode() == SpvOpFAdd &&
	      use->type_id() == instruction->type_id()) {
        InstructionBuilder builder(
          ctx, use,
          IRContext::kAnalysisDefUse |
          IRContext::kAnalysisInstrToBlockMapping);

        Instruction* new_inst = builder.AddNaryExtendedInstruction(
          use->type_id(), inst_set_id, GLSLstd450Fma, {});
        assert(new_inst != nullptr);
        new_inst->AddOperand(instruction->GetOperand(2));
        new_inst->AddOperand(instruction->GetOperand(3));

        auto const& AddOperandL = use->GetOperand(2);
        auto const& AddOperandR = use->GetOperand(3);
        if (AddOperandL.words.size() == 1 &&
            AddOperandL.words[0] == instruction->result_id()) {
          new_inst->AddOperand(AddOperandR);
        } else {
          new_inst->AddOperand(AddOperandL);
        }

        ctx->ReplaceAllUsesWith(use->result_id(), new_inst->result_id());
        ctx->UpdateDefUse(new_inst);

        modified = true;
      }
    }
  );
  return modified;
}

}  // namespace opt
}  // namespace spvtools
