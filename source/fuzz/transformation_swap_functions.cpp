// Copyright (c) 2021 Emiljano Gjiriti
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

#include "source/fuzz/transformation_swap_functions.h"

#include <unordered_set>
#include <utility>
#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationSwapFunctions::TransformationSwapFunctions(
    protobufs::TransformationSwapFunctions message)
    : message_(std::move(message)) {}

TransformationSwapFunctions::TransformationSwapFunctions(uint32_t result_id1,
                                                         uint32_t result_id2) {
  message_.set_result_id1(result_id1);
  message_.set_result_id2(result_id2);
}

bool TransformationSwapFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // A swap transformation is possible if:
  // - Both the result_ids are valid result_ids of some functions in the module
  // and
  // - The result_ids are not equal
  opt::Function* f1 = ir_context->GetFunction(message_.result_id1());
  opt::Function* f2 = ir_context->GetFunction(message_.result_id2());

  if (f1 == nullptr || f2 == nullptr) {
    return false;
  }
  assert(message_.result_id1() != message_.result_id2() &&
         "Cannot swap a function with itself");
  return true;
}

void TransformationSwapFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Function that swaps two functions based on their result_ids
  auto fp1 =
      std::find_if(ir_context->module()->begin(), ir_context->module()->end(),
                   [this](const opt::Function& function) {
                     return function.result_id() == message_.result_id1();
                   });
  auto fp2 =
      std::find_if(ir_context->module()->begin(), ir_context->module()->end(),
                   [this](const opt::Function& function) {
                     return function.result_id() == message_.result_id2();
                   });
  std::iter_swap(fp1.Get(), fp2.Get());
}

std::unordered_set<uint32_t> TransformationSwapFunctions::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

protobufs::Transformation TransformationSwapFunctions::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_functions() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
