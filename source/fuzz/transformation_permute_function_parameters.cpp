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

#include <vector>
#include <unordered_set>

#include "source/fuzz/transformation_permute_function_parameters.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationPermuteFunctionParameters::TransformationPermuteFunctionParameters(
    const spvtools::fuzz::protobufs::TransformationPermuteFunctionParameters& message)
    : message_(message) {}

TransformationPermuteFunctionParameters::TransformationPermuteFunctionParameters(
    uint32_t function_id,
    uint32_t fresh_type_id,
    const std::vector<uint32_t>& permutation,
    const std::vector<protobufs::InstructionDescriptor>& call_site) {
  message_.set_function_id(function_id);
  message_.set_fresh_type_id(fresh_type_id);

  for (auto index : permutation) {
    message_.add_permutation(index);
  }

  for (const auto& instruction : call_site) {
    *message_.add_call_site() = instruction;
  }
}

bool TransformationPermuteFunctionParameters::IsApplicable(
    opt::IRContext* context, const FactManager& /*unused*/) const {
  // Check that function exists
  auto* function = fuzzerutil::FindFunction(context, message_.function_id());
  if (!function ||
      function->DefInst().opcode() != SpvOpFunction ||
      fuzzerutil::FunctionIsEntryPoint(context, function->result_id())) {
    return false;
  }

  if (!fuzzerutil::IsFreshId(context, message_.fresh_type_id())) {
    return false;
  }

  // Check that permutation has valid indices
  auto* function_type = fuzzerutil::GetFunctionType(context, function);
  assert(function_type && "Function type is null");

  const auto& permutation = message_.permutation();

  uint32_t arg_size = function_type->NumInOperands();

  if (static_cast<uint32_t>(permutation.size()) != arg_size) {
    return false;
  }

  // Return type can't change its position
  // Also, return type is always defined, so |permutation| is never empty
  if (permutation.empty() || permutation[0] != 0) {
    return false;
  }

  std::unordered_set<uint32_t> set;

  for (auto index : permutation) {
    // Don't compare a signed integer with 0
    if (index >= arg_size) {
      return false;
    }

    set.insert(index);
  }

  // Check that permutation doesn't have duplicated values
  if (set.size() != arg_size) {
    return false;
  }

  // Check that call instructions are valid
  for (const auto& descriptor : message_.call_site()) {
    const auto* instruction = FindInstruction(descriptor, context);
    if (!instruction ||
        instruction->opcode() != SpvOpFunctionCall ||
        instruction->GetSingleWordInOperand(0) != message_.function_id()) {
      return false;
    }
  }

  return true;
}

void TransformationPermuteFunctionParameters::Apply(opt::IRContext* context,
                                              FactManager* /*unused*/) const {
  // Retrieve all data from the message
  uint32_t function_id = message_.function_id();
  uint32_t fresh_type_id = message_.fresh_type_id();
  const auto& permutation = message_.permutation();
  const auto& call_site = message_.call_site();

  // Find the function that will be transformed
  auto* function = fuzzerutil::FindFunction(context, function_id);
  assert(function && "Can't find the function");

  // Find the type to transform
  auto* function_type = fuzzerutil::GetFunctionType(context, function);
  assert(function_type && "Function type is null");

  // Create a vector of permuted arguments
  opt::Instruction::OperandList operands;
  std::vector<uint32_t> operand_data;
  for (auto index : permutation) {
    operands.push_back(function_type->GetInOperand(index));
    operand_data.push_back(operands.back().words[0]);
  }

  // Check if there is already a type with permuted arguments
  if (uint32_t new_type = fuzzerutil::FindFunctionType(context, operand_data)) {
    // Set function's type to that type
    function->DefInst().SetInOperand(1, {new_type});
  } else {
    fuzzerutil::UpdateModuleIdBound(context, fresh_type_id);
    // Create a new type with permuted parameters
    auto type = MakeUnique<opt::Instruction>(
        context, SpvOpTypeFunction, /*type_id*/ 0, fresh_type_id, operands);

    context->AddType(std::move(type));
    function->DefInst().SetInOperand(1, {fresh_type_id});
  }


  std::vector<uint32_t> param_id = {0}; // account for return type
  function->ForEachParam([&param_id](const opt::Instruction* param) {
    param_id.push_back(param->result_id());
  });

  // TODO: too many vectors, perhaps better to use in-place algorithm
  // or add getter/setter for function parameters
  std::vector<uint32_t> permuted_param_id;
  for (auto index : permutation) {
    permuted_param_id.push_back(param_id[index]);
  }

  // Set OpFunctionParam instructions to point to new parameters
  // i == 1 because we don't take function return type into account
  size_t i = 1;
  function->ForEachParam([&](opt::Instruction* param) {
    param->SetResultType(operand_data[i]);
    param->SetResultId(permuted_param_id[i]);
    ++i;
  });

  // Fix all OpFunctionCall instructions
  for (const auto& descriptor : call_site) {
    auto* call = FindInstruction(descriptor, context);
    assert(call && "Call instruction is null");

    opt::Instruction::OperandList call_operands;
    for (auto index : permutation) {
      call_operands.push_back(call->GetInOperand(index));
    }

    call->SetInOperands(std::move(call_operands));
  }
}

protobufs::Transformation TransformationPermuteFunctionParameters::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_permute_function_parameters() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
