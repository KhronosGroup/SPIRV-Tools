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

#include "source/fuzz/transformation_add_global_variable.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddGlobalVariable::TransformationAddGlobalVariable(
    const spvtools::fuzz::protobufs::TransformationAddGlobalVariable& message)
    : message_(message) {}

TransformationAddGlobalVariable::TransformationAddGlobalVariable(
    uint32_t fresh_id, uint32_t type_id, uint32_t initializer_id) {
  message_.set_fresh_id(fresh_id);
  message_.set_type_id(type_id);
  message_.set_initializer_id(initializer_id);
}

bool TransformationAddGlobalVariable::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }
  // The type id must correspond to a type.
  auto type = context->get_type_mgr()->GetType(message_.type_id());
  if (!type) {
    return false;
  }
  // That type must be a pointer type ...
  auto pointer_type = type->AsPointer();
  if (!pointer_type) {
    return false;
  }
  // ... with Private storage class.
  if (pointer_type->storage_class() != SpvStorageClassPrivate) {
    return false;
  }
  if (message_.initializer_id()) {
    auto initilizer_instruction =
        context->get_def_use_mgr()->GetDef(message_.initializer_id());
    // If an initializer id is provided it must correspond to an instruction.
    if (!initilizer_instruction) {
      return false;
    }
    // The instruction must be a constant.
    switch (initilizer_instruction->opcode()) {
      case SpvOpConstant:
      case SpvOpConstantComposite:
        if (pointer_type->pointee_type() !=
            context->get_type_mgr()->GetType(
                initilizer_instruction->type_id())) {
          return false;
        }
        break;
      default:
        // No other instructions can be used as an initializer.
        return false;
    }
  }
  return true;
}

void TransformationAddGlobalVariable::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  opt::Instruction::OperandList input_operands;
  input_operands.push_back(
      {SPV_OPERAND_TYPE_STORAGE_CLASS, {SpvStorageClassPrivate}});
  if (message_.initializer_id()) {
    input_operands.push_back(
        {SPV_OPERAND_TYPE_ID, {message_.initializer_id()}});
  }
  context->module()->AddGlobalValue(
      MakeUnique<opt::Instruction>(context, SpvOpVariable, message_.type_id(),
                                   message_.fresh_id(), input_operands));
  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());
  // We have added an instruction to the module, so need to be careful about the
  // validity of existing analyses.
  context->InvalidateAnalysesExceptFor(opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddGlobalVariable::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_global_variable() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
