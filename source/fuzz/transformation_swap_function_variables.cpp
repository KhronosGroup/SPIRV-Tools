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
    std::pair<uint32_t, uint32_t> pair_id, uint32_t function_id,
    uint32_t fresh_id) {
  message_.set_function_id(funcion_id);
  message_.set_fresh_id(fresh_id);
  protobufs::UInt32Pair pair;
  pair.set_first(pair_id.first);
  pair.set_second(pair_id.second);
  message_.set_pair(pair);
}

bool TransformationSwapFunctionVariables::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Here we check if funciton exists and check for it's entry point
  // FunctionIsEntryPoint -> Returns |true| if one of entry points has function
  // id |function_id|
  const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
  if (!function ||
      fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
  }

  // Retrieve ids and check they exists in the same function
  protobufs::UInt32Pair pair_id = message_.get_pair();
  auto first_block = function->entry().get();
  bool first_id_flag = false, second_id_flag = false;
  for (auto& instruction : *first_block) {
    uint32_t _id_ = instruction.result_id();
    first_id_flag = (_id_ == pair_id.first) ? true : false;
    second_id_flag = (_id_ == pair_id.second) ? true : false;
    if (first_id_flag && second_id_flag) {
      return true;
    }
  }
  return false;
}

void TransformationSwapFunctionVariables::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Check exists of functions
  auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
  assert(function && "function doesn't exists");
  protobufs::UInt32Pair pair_id = message_.get_pair();

  auto first_block = function->entry().get();
  uint32_t first_index = -1, second_index = -1, index = 0;
  for (auto& instruction : *first_block) {
    auto inst_id = instruction.result_id();
    first_index = (inst_id == pair_id.first) ? index : -1;
    second_index = (inst_id == pair_id.second) ? index : -1;
    ++index;
  }

  std::swap(*first_block[first_index], *first_block[second_index]);
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