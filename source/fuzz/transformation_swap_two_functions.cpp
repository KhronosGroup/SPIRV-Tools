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

#include <algorithm>

namespace spvtools {
namespace fuzz {

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(
    protobufs::TransformationSwapTwoFunctions message)
    : message_(std::move(message)) {}

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(uint32_t id1,
                                                               uint32_t id2) {
  assert(id1 != id2 && "The two function ids cannot be the same.");
  message_.set_function_id1(id1);
  message_.set_function_id2(id2);
}

bool TransformationSwapTwoFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto func1_ptr = ir_context->GetFunction(message_.function_id1());
  auto func2_ptr = ir_context->GetFunction(message_.function_id2());
  if (!func1_ptr || !func2_ptr ||
      func1_ptr->result_id() == func2_ptr->result_id())
    return false;
  return true;
}

void TransformationSwapTwoFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Initialize the position (underlying: UptrVectorIterator<Function>)
  opt::Module::iterator func1_it = ir_context->module()->begin();
  opt::Module::iterator func2_it = ir_context->module()->begin();

  for (auto iter = ir_context->module()->begin();
       iter != ir_context->module()->end(); ++iter) {
    if ((*iter).result_id() == message_.function_id1()) func1_it = iter;
    if ((*iter).result_id() == message_.function_id2()) func2_it = iter;
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
