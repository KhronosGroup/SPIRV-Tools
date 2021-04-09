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

#include "source/fuzz/fuzzer_pass_swap_functions.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/transformation_swap_two_functions.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSwapFunctions::FuzzerPassSwapFunctions(
    opt::IRContext* ir_context, TransformationContext* transformation_context, 
    FuzzerContext* fuzzer_context, 
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context, 
                 transformations) {}

void FuzzerPassSwapFunctions::Apply() {
    // Here we start by doing exhaustive swap testing: 
    // For every function a and function b, where a and b has a valid id in [function_ids] set
    // and a and b are not the same function (ie, have a different id). 
    // For every combination of a and b, we do transformation_swap_two_functions(a.id, b.id) and 
    // make sure everyone of them have the correct result returned. 
    // When the function space is large, we might choose functions arbitrarily from the function space
    // and do random swaps.

    // Collect all functions by their id from the given module.
    std::vector<uint32_t> function_ids; 
    for(auto& function : *GetIRContext()->module()) {
        function_ids.emplace_back(function.result_id());
    }
    
    int id_size = function_ids.size(); 
    // We iterate through every combination of id i & j where i!=j.
    for(int i = 0; i<id_size-1; ++i) {
       for(int j = i+1; j<id_size; ++j) {  
         // Randomly decide whether to ignore function swap.
         if (!GetFuzzerContext()->ChoosePercentage(
                 GetFuzzerContext()->GetChanceOfSwappingFunctions())) {
           continue;
         }         
         // We do a swap between functions and break if such swap cannot be performed.
         TransformationSwapTwoFunctions transformation(function_ids[i], function_ids[j]); 
         if(!MaybeApplyTransformation(transformation)) {
           break; 
         }
       }
    }
}

}  // namespace fuzz
}  // namespace spvtools
