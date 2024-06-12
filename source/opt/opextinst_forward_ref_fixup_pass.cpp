// Copyright (c) 2024 Google Inc.
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

#include "source/opt/opextinst_forward_ref_fixup_pass.h"

#include <set>
#include <string>

#include "source/extensions.h"
#include "source/opt/ir_context.h"
#include "type_manager.h"

namespace spvtools {
namespace opt {
namespace {

bool Fixup(IRContext* ctx, const std::unordered_set<uint32_t>& all_debug_ids,
           std::unordered_set<uint32_t>& seen_ids,
           bool* hasAtLeastOneForwardReference, Instruction* inst) {
  seen_ids.insert(inst->result_id());

  if (inst->opcode() != spv::Op::OpExtInst &&
      inst->opcode() != spv::Op::OpExtInstWithForwardRefsKHR)
    return false;

  const uint32_t num_in_operands = inst->NumInOperands();
  bool hasForwardReferences = false;
  for (uint32_t i = 0; i < num_in_operands; ++i) {
    const Operand& op = inst->GetInOperand(i);
    if (!spvIsIdType(op.type)) continue;

    if (all_debug_ids.count(op.AsId()) == 0) continue;

    if (seen_ids.count(op.AsId()) == 0) {
      hasForwardReferences = true;
      *hasAtLeastOneForwardReference = true;
      break;
    }
  }

  if (hasForwardReferences &&
      inst->opcode() != spv::Op::OpExtInstWithForwardRefsKHR)
    inst->SetOpcode(spv::Op::OpExtInstWithForwardRefsKHR);
  else if (!hasForwardReferences && inst->opcode() != spv::Op::OpExtInst)
    inst->SetOpcode(spv::Op::OpExtInst);
  else
    return false;

  ctx->AnalyzeUses(inst);
  return true;
}

}  // namespace

Pass::Status OpExtInstWithForwardReferenceFixupPass::Process() {
  std::unordered_set<uint32_t> seen_ids;
  std::unordered_set<uint32_t> all_debug_ids;

  for (const auto& it : get_module()->ext_inst_debuginfo())
    all_debug_ids.insert(it.result_id());

  for (const auto& it : get_module()->ext_inst_imports()) {
    all_debug_ids.insert(it.result_id());
    seen_ids.insert(it.result_id());
  }

  bool changed = false;
  bool hasAtLeastOneForwardReference = false;
  IRContext* ctx = context();
  for (auto& it : get_module()->ext_inst_debuginfo())
    changed |= Fixup(ctx, all_debug_ids, seen_ids,
                     &hasAtLeastOneForwardReference, &it);

  if (hasAtLeastOneForwardReference !=
      ctx->get_feature_mgr()->HasExtension(
          kSPV_KHR_relaxed_extended_instruction)) {
    if (hasAtLeastOneForwardReference)
      ctx->AddExtension("SPV_KHR_relaxed_extended_instruction");
    else
      ctx->RemoveExtension(Extension::kSPV_KHR_relaxed_extended_instruction);
    changed = true;
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
