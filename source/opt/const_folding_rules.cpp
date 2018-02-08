// Copyright (c) 2018 Google LLC
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

#include "const_folding_rules.h"

namespace spvtools {
namespace opt {

namespace {
const uint32_t kExtractCompositeIdInIdx = 0;

ConstantFoldingRule FoldExtractWithConstants() {
  // Folds an OpcompositeExtract where input is a composite constant.
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    const analysis::Constant* c = constants[kExtractCompositeIdInIdx];
    if (c == nullptr) {
      return nullptr;
    }

    for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
      uint32_t element_index = inst->GetSingleWordInOperand(i);
      if (c->AsNullConstant()) {
        // Return Null for the return type.
        ir::IRContext* context = inst->context();
        analysis::ConstantManager* const_mgr = context->get_constant_mgr();
        analysis::TypeManager* type_mgr = context->get_type_mgr();
        const analysis::NullConstant null_const(
            type_mgr->GetType(inst->type_id()));
        const analysis::Constant* real_const =
            const_mgr->FindConstant(&null_const);
        if (real_const == nullptr) {
          ir::Instruction* const_inst =
              const_mgr->GetDefiningInstruction(&null_const);
          real_const = const_mgr->GetConstantFromInst(const_inst);
        }
        return real_const;
      }

      auto cc = c->AsCompositeConstant();
      assert(cc != nullptr);
      auto components = cc->GetComponents();
      c = components[element_index];
    }
    return c;
  };
}

ConstantFoldingRule FoldCompositeWithConstants() {
  // Folds an OpCompositeConstruct where all of the inputs are constants to a
  // constant.  A new constant is created if necessary.
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* new_type = type_mgr->GetType(inst->type_id());

    std::vector<uint32_t> ids;
    for (const analysis::Constant* element_const : constants) {
      if (element_const == nullptr) {
        return nullptr;
      }
      uint32_t element_id = const_mgr->FindDeclaredConstant(element_const);
      if (element_id == 0) {
        return nullptr;
      }
      ids.push_back(element_id);
    }
    return const_mgr->GetConstant(new_type, ids);
  };
}
}  // namespace

spvtools::opt::ConstantFoldingRules::ConstantFoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[SpvOpCompositeConstruct].push_back(FoldCompositeWithConstants());
  rules_[SpvOpCompositeExtract].push_back(FoldExtractWithConstants());
}
}  // namespace opt
}  // namespace spvtools
