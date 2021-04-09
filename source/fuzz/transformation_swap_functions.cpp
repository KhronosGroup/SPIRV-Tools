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

TransformationSwapFunctions::TransformationSwapFunctions(uint32_t func1_id,
                                                         uint32_t func2_id) {
  message_.set_func1_id(func1_id);
  message_.set_func2_id(func2_id);
}

bool TransformationSwapFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  bool found_func1 = false;
  bool found_func2 = false;

  for (auto it = ir_context->module()->cbegin();
       it != ir_context->module()->cend(); ++it) {
    if (it->result_id() == message_.func1_id()) {
      found_func1 = true;
    }

    if (it->result_id() == message_.func2_id()) {
      found_func2 = true;
    }
  }

  if ((message_.func1_id() == message_.func2_id()) ||
      !(found_func1 && found_func2)) {
    return false;
  }
  return true;
}

void TransformationSwapFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto fp1 = ir_context->module()->cbegin();
  auto fp2 = ir_context->module()->cbegin();
  // Find the functions that will be swapped

  for (auto it1 = ir_context->module()->cbegin();
       it1 != ir_context->module()->cend(); ++it1) {
    if (it1->result_id() == message_.func1_id()) {
      fp1 = it1;
    }
    if (it1->result_id() == message_.func2_id()) {
      fp2 = it1;
    }
  }
  std::swap(fp1, fp2);
  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
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
