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

#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

bool TransformationReplaceConstantWithUniform::IsApplicable(
    spvtools::opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) {
  if (context->get_def_use_mgr()->GetDef(fresh_id_for_access_chain_)) {
    return false;
  }

  if (context->get_def_use_mgr()->GetDef(fresh_id_for_load_)) {
    return false;
  }

  auto declared_constant = context->get_constant_mgr()->FindDeclaredConstant(
      id_use_descriptor_.GetIdOfInterest());

  if (!declared_constant) {
    return false;
  }

  if (!declared_constant->AsScalarConstant()) {
    return false;
  }

  auto constant_associated_with_uniform =
      fact_manager.GetConstantFromUniformDescriptor(uniform_descriptor_);
  if (!constant_associated_with_uniform) {
    return false;
  }

  if (!constant_associated_with_uniform->AsScalarConstant()) {
    return false;
  }

  if (!declared_constant->type()->IsSame(
          constant_associated_with_uniform->type())) {
    return false;
  }

  if (declared_constant->AsScalarConstant()->words() !=
      constant_associated_with_uniform->AsScalarConstant()->words()) {
    return false;
  }

  auto instruction_using_constant = id_use_descriptor_.FindInstruction(context);
  if (!instruction_using_constant) {
    return false;
  }

  opt::analysis::Pointer pointer_to_type_of_constant(declared_constant->type(),
                                                     SpvStorageClassUniform);
  if (!context->get_type_mgr()->GetId(&pointer_to_type_of_constant)) {
    return false;
  }

  opt::analysis::Integer int_type(32, true);
  if (!context->get_type_mgr()->GetId(&int_type)) {
    return false;
  }

  auto registered_int_type =
      context->get_type_mgr()->GetRegisteredType(&int_type)->AsInteger();
  auto int_type_id = context->get_type_mgr()->GetId(&int_type);
  for (auto index : uniform_descriptor_.indices) {
    opt::analysis::IntConstant int_constant(registered_int_type, {index});
    if (!context->get_constant_mgr()->FindDeclaredConstant(&int_constant,
                                                           int_type_id)) {
      return false;
    }
  }

  return true;
}

void TransformationReplaceConstantWithUniform::Apply(
    spvtools::opt::IRContext* context,
    spvtools::fuzz::FactManager* /*unused*/) {
  auto inst = id_use_descriptor_.FindInstruction(context);
  assert(inst && "Precondition requires that the id use can be found.");
  assert(inst->GetSingleWordInOperand(id_use_descriptor_.GetInOperandIndex()) ==
             id_use_descriptor_.GetIdOfInterest() &&
         "Does not appear to be a usage of the desired id.");

  auto constant_inst =
      context->get_def_use_mgr()->GetDef(id_use_descriptor_.GetIdOfInterest());
  auto constant_type_id = constant_inst->type_id();
  auto type_and_pointer_type = context->get_type_mgr()->GetTypeAndPointerType(
      constant_type_id, SpvStorageClassUniform);
  assert(type_and_pointer_type.first != nullptr);
  assert(type_and_pointer_type.second != nullptr);
  auto pointer_to_uniform_constant_type_id =
      context->get_type_mgr()->GetId(type_and_pointer_type.second.get());

  opt::Instruction::OperandList operands_for_access_chain;
  operands_for_access_chain.push_back(
      {SPV_OPERAND_TYPE_ID, {uniform_descriptor_.uniform_variable_id}});

  opt::analysis::Integer int_type(32, true);
  auto registered_int_type =
      context->get_type_mgr()->GetRegisteredType(&int_type)->AsInteger();
  auto int_type_id = context->get_type_mgr()->GetId(&int_type);
  for (auto index : uniform_descriptor_.indices) {
    opt::analysis::IntConstant int_constant(registered_int_type, {index});
    auto constant_id = context->get_constant_mgr()->FindDeclaredConstant(
        &int_constant, int_type_id);
    operands_for_access_chain.push_back({SPV_OPERAND_TYPE_ID, {constant_id}});
  }
  inst->InsertBefore(MakeUnique<opt::Instruction>(
      context, SpvOpAccessChain, pointer_to_uniform_constant_type_id,
      fresh_id_for_access_chain_, operands_for_access_chain));

  opt::Instruction::OperandList operands_for_load = {
      {SPV_OPERAND_TYPE_ID, {fresh_id_for_access_chain_}}};
  inst->InsertBefore(
      MakeUnique<opt::Instruction>(context, SpvOpLoad, constant_type_id,
                                   fresh_id_for_load_, operands_for_load));

  inst->SetInOperand(id_use_descriptor_.GetInOperandIndex(),
                     {fresh_id_for_load_});

  fuzzerutil::UpdateModuleIdBound(context, fresh_id_for_load_);
  fuzzerutil::UpdateModuleIdBound(context, fresh_id_for_access_chain_);
}

protobufs::Transformation
TransformationReplaceConstantWithUniform::ToMessage() {
  assert(false && "TODO");
  return protobufs::Transformation();
}

}  // namespace fuzz
}  // namespace spvtools
