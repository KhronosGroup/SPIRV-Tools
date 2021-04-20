// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/transformation_swap_two_functions.h"

#include "source/opt/function.h"
#include "source/opt/module.h"

namespace spvtools {
namespace fuzz {

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(
    protobufs::TransformationSwapTwoFunctions message)
    : message_(std::move(message)) {}

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(uint32_t id1, uint32_t id2) {
  message_.set_function_id1(id1);
  message_.set_function_id2(id2);
}

bool TransformationSwapTwoFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
      
  assert(message_.function_id1()!=message_.function_id2() && " Two functions cannot be the same.");
  assert(ir_context->GetFunction(message_.function_id1())!=nullptr && "Function 1 is not in range.");
  assert(ir_context->GetFunction(message_.function_id2())!=nullptr && "Function 2 is not in range."); 

  return true;
}

void TransformationSwapTwoFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Found the two functions in ir_context and swap their position. 

  // TODO: Modify function.h, get function pointer and perform swap

  opt::Function* func1_ptr = ir_context->GetFunction(message_.function_id1());
  opt::Function* func2_ptr = ir_context->GetFunction(message_.function_id2());

  assert( func1_ptr!=nullptr  && "ERROR: Function 1 was not found with the given id."); 
  assert( func2_ptr!=nullptr  && "ERROR: Function 2 was not found with the given id.");
  assert( &func1_ptr != &func2_ptr && "ERROR: Two functions cannot be the same.");
  //TODO: testing after modifying function.h

  //std::swap(*func1_ptr, *func2_ptr); 
}

protobufs::Transformation TransformationSwapTwoFunctions::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_two_functions() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapTwoFunctions::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
