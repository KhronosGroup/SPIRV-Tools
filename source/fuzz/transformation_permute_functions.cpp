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

#include "source/fuzz/transformation_permute_functions.h"

#include <vector>
#include <utility>
#include <unordered_set>

#include "source/fuzz/fuzzer_util.h"
#include "source/util/generate_swaps.h"

namespace spvtools {
namespace fuzz {

TransformationPermuteFunctions::
    TransformationPermuteFunctions(
        const spvtools::fuzz::protobufs::
            TransformationPermuteFunctions& message)
    : message_(message) {}

TransformationPermuteFunctions::
    TransformationPermuteFunctions(
        const std::vector<uint32_t>& permutations) {

  for (auto func_id : permutations) {
    message_.add_permutation(func_id);
  }
}


bool TransformationPermuteFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {

  std::vector<uint32_t> permutation(message_.permutation().begin(),
                                    message_.permutation().end());

  auto func_size   = (ir_context->module()->end() - ir_context->module()->begin());

  // // |permutation| vector should be equal to the number of arguments
  if (static_cast<uint32_t>(permutation.size()) != func_size) {
    return false;
  }

  // Check that permutation doesn't have duplicated values.
  assert(!fuzzerutil::HasDuplicates(permutation) &&
         "Permutation has duplicates");

   
   return true;
}



void TransformationPermuteFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Find the function that will be transformed
  
  std::vector<uint32_t> original_order;
  using std::pair;
   std::vector<uint32_t> permutation(message_.permutation().begin(),
                                    message_.permutation().end());
  
  for (const auto& function : *ir_context->module()) {
    uint32_t function_id = function.result_id();

    original_order.push_back(function_id);

  }

  std::vector<pair<uint32_t,uint32_t>> perms;

  perms = generate_swaps(original_order,permutation);
  
  for(auto it1 = ir_context->module()->cbegin(); it1!= ir_context->module()->cend(); ++it1){
    for(auto it2 = ir_context->module()->cbegin(); it2!=ir_context->module()->cend();++it2){
      auto find_pos = std::find(perms.begin(),perms.end(),
      std::make_pair(std::min(it1->result_id(),it2->result_id()),std::max(it1->result_id(),it2->result_id())));
      if(find_pos != perms.end()){
        std::swap(it1,it2);
      }
    }
  }
  // Make sure our changes are analyzed
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}


std::unordered_set<uint32_t>
TransformationPermuteFunctions::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}


protobufs::Transformation TransformationPermuteFunctions::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_permute_functions() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
