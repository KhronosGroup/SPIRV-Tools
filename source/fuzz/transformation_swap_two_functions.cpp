// Copyright (c) 2021 Shiyu Liu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/fuzz/transformation_swap_two_functions.h"

#include "source/opt/function.h"
#include "source/opt/module.h"

namespace spvtools {
namespace fuzz {

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(
    protobufs::TransformationSwapTwoFunctions message)
    : message_(std::move(message)) {}

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(uint32_t id1,
                                                               uint32_t id2) {
  assert(id1 != id2 && "Two functions cannot be the same.");
  message_.set_function_id1(id1);
  message_.set_function_id2(id2);
}

bool TransformationSwapTwoFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  assert((ir_context->GetFunction(message_.function_id1()) != nullptr ||
          ir_context->GetFunction(message_.function_id2()) != nullptr) &&
         "Both functions are not in range.");
  assert(ir_context->GetFunction(message_.function_id1()) != nullptr &&
         "Function 1 is not in range.");
  assert(ir_context->GetFunction(message_.function_id2()) != nullptr &&
         "Function 2 is not in range.");

  return true;
}

void TransformationSwapTwoFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Found the two functions in ir_context and swap their position.
  // Offsets mark the relevant distance of the function from module().begin().
  bool func1_found = false;
  bool func2_found = false;

  // Initialize the position (underlying: UptrVectorIterator<Function>)
  opt::Module::iterator func1_it = ir_context->module()->begin();
  opt::Module::iterator func2_it = ir_context->module()->begin();
  for (auto& func : *ir_context->module()) {
    if (func.result_id() == message_.function_id1()) func1_found = true;
    if (func.result_id() == message_.function_id2()) func2_found = true;
    // Once we found the target function, we stop increment iterator and thus
    // after one iteration, func1_it and func2_it should be the iterator with
    // their updated position.
    // If we have not found (ie. found = false), we kept incrementing.
    if (!func1_found) ++func1_it;
    if (!func2_found) ++func2_it;
  }
  // Two function pointers are all set, swap the two functions within the
  // module.
  std::iter_swap(func1_it.Get(), func2_it.Get());
}

protobufs::Transformation TransformationSwapTwoFunctions::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_two_functions() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapTwoFunctions::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
