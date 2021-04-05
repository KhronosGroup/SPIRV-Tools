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

#include "source/fuzz/fuzzer_pass_permute_functions.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_permute_functions.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctions::FuzzerPassPermuteFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}



void FuzzerPassPermuteFunctions::Apply() {
  
  std::vector<uint32_t>permutation;
  std::vector<uint32_t>original_order;
  int entry_point_id = -1;
  size_t ind;

  for (const auto& function : *GetIRContext()->module()) {
    uint32_t function_id = function.result_id();

    permutation.push_back(function_id);
    original_order.push_back(function_id);

    // Skip the function if it is an entry point
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(), function_id)) {
      
      entry_point_id = function_id;
    
    }
  }

  GetFuzzerContext()->Shuffle(&permutation);
  
  // Need to keep the entry point function at the beginning of module 
  if(entry_point_id != -1){
      size_t i;
      for(i=0; i<permutation.size(); i++){
          if(permutation[i] == (uint32_t)entry_point_id){
              ind = i;
          }
      }
      
      std::iter_swap(permutation.begin(),permutation.begin()+ind);

  }

  // generate swaps from a given permutation
  // transformations needed to reach a certain permutation
  // Apply the transformation
  ApplyTransformation(TransformationPermuteFunctions(permutation));
}

}  // namespace fuzz
}  // namespace spvtools
