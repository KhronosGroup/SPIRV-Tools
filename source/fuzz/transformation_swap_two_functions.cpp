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

#include "source/fuzz/transformation_swap_two_functions.h"

#include "source/opt/function.h"

namespace spvtools {
namespace fuzz {

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(
    protobufs::TransformationSwapTwoFunctions message)
    : message_(std::move(message)) {}

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(uint32_t id1, uint32_t id2) {
  message_.set_function_id1(id1);
  message_.set_function_id2(id2);
}

bool TransformationMoveSwapTwoFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // Go through every function in ir_context and return true only when both ids are found.
  //not applicable since two swapped functions are the same one. 
  if(message_.function_id1()==message_.function_id2()) return false;  

  bool foundFunc1 = false, foundFunc2 = false;
  
  for (auto& function : *ir_context->module()) {//iterate through every functions in module
    if(function->result_id()==message_.function_id1()) foundFunc1 = true;
    if(function->result_id()==message_.function_id2()) foundFunc2 = true;
  }

  // return true only when both functions are found with given ids
  return foundFunc1 && foundFunc2;
}

void TransformationSwapTwoFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Found the two functions in ir_context and swap their position. 
  auto ptr1 = nullptr, iter2 = nullptr; 
  for(auto func_it = *ir_context->module().begin(); func_it!=*ir_context->module().end();++func_it) {
    if(func_it->result_id()==message_.function_id1()) iter1 = func_it; 
    if(func_it->result_id()==message_.function_id2()) iter2 = func_it; 
  } 
  
  
  assert(ptr1!=nullptr && "Whoops, function 1 was not found."); 
  assert(ptr2!=nullptr && "Whoops, function 2 was not found.");
  assert(&ptr1!=&ptr2 && "Whoops, two functions cannot be the same");
  //two function pointers are all set, swap the two functions within the module  
  //TODO 
  std::iter_swap(ptr1, pt2); 
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
