// Copyright (c) 2021 Mostafa Ashraf
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

#include "source/fuzz/transformation_swap_function_variables.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(
    const spvtools::fuzz::protobufs::TransformationSwapFunctionVariables&
        message)
    : message_(message) {}

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(
    uint32_t variable_1_id, uint32_t variable_2_id) {
  message_.set_variable_id(variable_1_id);
  message_.set_variable_id(variable_2_id);
}

bool TransformationSwapFunctionVariables::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context /*unused*/) const {
  uint32_t variable_1_id = message_.variable_id();
  uint32_t variable_2_id = message_.variable_id();

  // The get_instr_block(..) overloaded method return BasicBlock* or nullptr.
  auto* block_1 = ir_context->get_instr_block(variable_1_id);
  assert(block_1 && "The Block related to the first id is null");
  auto* block_2 = ir_context->get_instr_block(variable_2_id);
  assert(block_2 && "The Block related to the second id is null");

  auto function_id_1 = block_1->GetParent()->result_id();
  auto function_id_2 = block_2->GetParent()->result_id();

  return (function_id_1 == function_id_2) ? true : false;
}

void TransformationSwapFunctionVariables::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context /*unused*/) const {
  uint32_t variable_1_id = message_.variable_id();
  uint32_t variable_2_id = message_.variable_id();

  // The get_instr_block(..) overloaded method return BasicBlock* or nullptr.
  auto* block = ir_context->get_instr_block(variable_1_id);

  auto function = block->GetParent();

  auto first_block = function->entry().get();

  std::swap(*first_block[variable_1_id], *first_block[variable_2_id]);
}

protobufs::Transformation TransformationSwapFunctionVariables::ToMessage()
    const {
  protobufs::Transformation result;
  *result.transformation_swap_function_variables() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapFunctionVariables::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
