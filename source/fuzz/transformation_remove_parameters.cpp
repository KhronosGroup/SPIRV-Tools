// Copyright (c) 2020 Vasyl Teliman
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

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_remove_parameters.h"

namespace spvtools {
namespace fuzz {

TransformationRemoveParameters::TransformationRemoveParameters(
    const protobufs::TransformationRemoveParameters& message)
    : message_(message) {}

TransformationRemoveParameters::TransformationRemoveParameters(
    uint32_t function_id, uint32_t new_type_id,
    const std::vector<uint32_t>& parameter_index,
    const std::vector<uint32_t>& fresh_id,
    const std::vector<uint32_t>& initializer_id) {
  message_.set_function_id(function_id);
  message_.set_new_type_id(new_type_id);

  for (auto index : parameter_index) {
    message_.add_parameter_index(index);
  }

  for (auto id : fresh_id) {
    message_.add_fresh_id(id);
  }

  for (auto id : initializer_id) {
    message_.add_initializer_id(id);
  }
}

bool TransformationRemoveParameters::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Check that function exists and is not an entry point.
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  auto params = fuzzerutil::GetParameters(*function);
  assert(!params.empty() &&
         "The function doesn't have any parameters to remove");

  std::vector<uint32_t> param_index(message_.parameter_index().begin(),
                                    message_.parameter_index().end());

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/pull/3421/):
  //  uncomment when the PR is merged.
  // assert(!fuzzerutil::HasDuplicates(parameter_index) &&
  //        "Duplicated parameter indices.");

  // Check that |message_.parameter_id| has valid size.
  if (param_index.empty() || param_index.size() > params.size()) {
    return false;
  }

  // Check that parameter indices are valid.
  if (!std::all_of(
          param_index.begin(), param_index.end(),
          [&params](uint32_t index) { return index < params.size(); })) {
    return false;
  }

  // Check that new function type is valid.
  const auto* old_type_inst = fuzzerutil::GetFunctionType(ir_context, function);
  assert(old_type_inst && old_type_inst->opcode() == SpvOpTypeFunction &&
         "Function type is invalid");

  const auto* new_type_inst =
      ir_context->get_def_use_mgr()->GetDef(message_.new_type_id());
  if (!new_type_inst || new_type_inst->opcode() != SpvOpTypeFunction) {
    return false;
  }

  // Check that new function type has the same return type.
  if (old_type_inst->GetSingleWordInOperand(0) !=
      new_type_inst->GetSingleWordInOperand(0)) {
    return false;
  }

  // Check that new function type has valid parameters' types.
  std::vector<uint32_t> new_type_ids;
  for (uint32_t i = 0, n = static_cast<uint32_t>(params.size()); i < n; ++i) {
    if (std::find(param_index.begin(), param_index.end(), i) ==
        param_index.end()) {
      new_type_ids.push_back(params[i]->type_id());
    }
  }

  // -1 for the return type.
  if (new_type_ids.size() != new_type_inst->NumInOperands() - 1) {
    return false;
  }

  for (uint32_t i = 0, n = new_type_inst->NumInOperands() - 1; i < n; ++i) {
    // Note that this checks whether parameters are ordered correctly as well.
    if (new_type_inst->GetSingleWordInOperand(i + 1) != new_type_ids[i]) {
      return false;
    }
  }

  // Check that |message_.fresh_id| has valid size.
  if (static_cast<size_t>(message_.fresh_id().size()) != param_index.size()) {
    return false;
  }

  // Check that all ids are fresh.
  if (!std::all_of(message_.fresh_id().begin(), message_.fresh_id().end(),
                   [ir_context](uint32_t id) {
                     return fuzzerutil::IsFreshId(ir_context, id);
                   })) {
    return false;
  }

  // Check that |message_.initializer_id| has valid size.
  if (static_cast<size_t>(message_.initializer_id().size()) !=
      param_index.size()) {
    return false;
  }

  // Check that initializers are valid.
  for (size_t i = 0, n = param_index.size(); i < n; ++i) {
    // Check that OpTypePointer instruction with Private storage class exists in
    // the module.
    if (!fuzzerutil::MaybeGetPointerType(ir_context,
                                         params[param_index[i]]->type_id(),
                                         SpvStorageClassPrivate)) {
      return false;
    }

    const auto* pointee_type =
        ir_context->get_type_mgr()->GetType(params[param_index[i]]->type_id());
    assert(pointee_type && "Parameter type must exist");

    const auto* initializer_inst = ir_context->get_def_use_mgr()->GetDef(
        message_.initializer_id()[static_cast<int>(i)]);

    // Check that initializer has correct type.
    if (!initializer_inst ||
        initializer_inst->type_id() != params[param_index[i]]->type_id()) {
      return false;
    }
  }

  return true;
}

void TransformationRemoveParameters::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Unpack some |message_| components.
  const auto& parameter_index = message_.parameter_index();
  const auto& fresh_id = message_.fresh_id();
  const auto& initializer_id = message_.initializer_id();

  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function && "Function must exist");

  auto params = fuzzerutil::GetParameters(function);
  for (int i = 0, n = parameter_index.size(); i < n; ++i) {
    const auto* param_inst = params[parameter_index[i]];

    // Add global variable to store function's argument.
    fuzzerutil::AddGlobalVariable(
        ir_context, fresh_id[i],
        fuzzerutil::MaybeGetPointerType(ir_context, param_inst->type_id(),
                                        SpvStorageClassPrivate),
        SpvStorageClassPrivate, initializer_id[i]);

    if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
            param_inst->result_id())) {
      transformation_context->GetFactManager()
          ->AddFactValueOfPointeeIsIrrelevant(fresh_id[i]);
      // We don't remove PointeeIsIrrelevant fact for the removed parameter
      // since its result id is still present in the module (even though it does
      // not correspond to the OpFunctionParameter instruction).

      // A removed parameter might have been created previously by the
      // TransformationAddParameter and have a scalar type. Since scalar types
      // don't support PointeeValueIsIrrelevant, the created variable won't be
      // marked as irrelevant either.
    }

    // Insert OpLoad instruction right after OpVariable instructions. There can
    // be no OpPhi instructions since its the first block of the function.
    auto it = function->begin()->begin();
    while (it != function->begin()->end() &&
           !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it)) {
      ++it;
    }

    assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpLoad, it) &&
           "Can't insert OpLoad into the first basic block of the function");

    // Load argument from the global variable into the object with the same
    // result id and type as the removed parameter had.
    it.InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpLoad, param_inst->type_id(), param_inst->result_id(),
        opt::Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {fresh_id[i]}}}));

    // Remove the parameter from the function. This must be the last step since
    // this will invalidate |param_inst|.
    function->RemoveParameter(param_inst->result_id());
  }

  // Update module id bound. |fresh_id| is guaranteed to be non-empty.
  fuzzerutil::UpdateModuleIdBound(
      ir_context, *std::max_element(fresh_id.begin(), fresh_id.end()));

  // Update function's type id.
  function->DefInst().SetInOperand(1, {message_.new_type_id()});

  // Update all OpFunctionCall.
  ir_context->get_def_use_mgr()->ForEachUser(
      function->result_id(),
      [ir_context, &parameter_index, &fresh_id](opt::Instruction* inst) {
        if (inst->opcode() != SpvOpFunctionCall) {
          return;
        }

        auto it = fuzzerutil::GetIteratorForInstruction(
            ir_context->get_instr_block(inst), inst);
        assert(fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpStore, it) &&
               "Can't insert OpStore right before the function call");

        for (int i = 0, n = parameter_index.size(); i < n; ++i) {
          // +1 since the first operand of OpFunctionCall is the id of the
          // function.
          auto operand = inst->GetSingleWordInOperand(parameter_index[i] + 1);
          inst->RemoveInOperand(parameter_index[i] + 1);

          // Load operand data into the global variable.
          it.InsertBefore(MakeUnique<opt::Instruction>(
              ir_context, SpvOpStore, 0, 0,
              opt::Instruction::OperandList{
                  {SPV_OPERAND_TYPE_ID, {fresh_id[i]}},
                  {SPV_OPERAND_TYPE_ID, {operand}}}));
        }
      });

  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationRemoveParameters::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_remove_parameters() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
