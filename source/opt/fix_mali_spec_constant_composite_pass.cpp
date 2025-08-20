// Copyright (c) 2025 Lee Gao
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

#include "source/opt/fix_mali_spec_constant_composite_pass.h"

#include "source/opt/ir_builder.h"
#include "source/util/hex_float.h"

#define LOG_(level, fmt, ...) \
  do { \
    char buffer[256]; \
    snprintf(buffer, sizeof(buffer), \
      "SpecConstantCompositePass: " fmt, ##__VA_ARGS__); \
    consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer); \
  } while (0)

#define LOGD(fmt, ...) LOG_(SPV_MSG_DEBUG, fmt, ## __VA_ARGS__)
#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ## __VA_ARGS__)
#define LOGE(fmt, ...) LOG_(SPV_MSG_ERROR, fmt, ## __VA_ARGS__)

namespace spvtools {
namespace opt {

Pass::Status FixMaliSpecConstantCompositePass::Process() {
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    return Status::SuccessWithoutChange;
  }

  bool changed = false;
  for (Instruction* var : context()->GetConstants()) {
    // Look for the following pattern
    //     %bool = OpTypeBool
    // %t0_bound = OpSpecConstantTrue %bool | OpSpecConstantFalse %bool | OpSpecConstant %_
    //   %v4bool = OpTypeVector %bool 4
    //       %55 = OpConstantComposite %v4bool %t0_bound %t0_bound %t0_bound
    //     %t0_bound
    if (var->opcode() == spv::Op::OpConstantComposite) {
      // Get the type operand, looking for `%v4bool = OpTypeVector %bool 4`
      const auto type_id = var->type_id();
      const Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
      if (type_inst->opcode() != spv::Op::OpTypeVector) {
        continue;
      }
      const auto component_type_id = type_inst->GetSingleWordInOperand(0);
      const Instruction* component_type_inst = get_def_use_mgr()->GetDef(component_type_id);
      if (component_type_inst->opcode() != spv::Op::OpTypeBool) {
        continue;
      }
      // Check to see if _any_ of the operands are OpSpecConstantTrue | OpSpecConstantFalse | OpSpecConstant
      bool op_spec_constant = false;
      for (auto i = 0u; i < var->NumInOperands(); i++) {
        const auto operand_id = var->GetSingleWordInOperand(i);
        const Instruction* operand_inst = get_def_use_mgr()->GetDef(operand_id);
        if (IsSpecConstantInst(operand_inst->opcode())) {
          op_spec_constant = true;
          break;
        }
      }
      if (op_spec_constant) {
        LOG("Found OpConstantComposite %%v4bool with all specialized-const operands. "
            "Converting to OpSpecConstantComposite to avoid Mali shader compiler bug");
        LOGD("%s", var->PrettyPrint().c_str());
        LOGD("    .type: %s", component_type_inst->PrettyPrint().c_str());
        LOGD("    .type: %s", type_inst->PrettyPrint().c_str());
        for (auto i = 0u; i < var->NumInOperands(); i++) {
          const auto operand_id = var->GetSingleWordInOperand(i);
          const Instruction* operand_inst = get_def_use_mgr()->GetDef(operand_id);
          LOGD("    .arg%d: %s", i, operand_inst->PrettyPrint().c_str());
        }
        // Change the op-type of var from OpConstantComposite to OpSpecConstantComposite
        var->SetOpcode(spv::Op::OpSpecConstantComposite);
        changed = true;
      }
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
