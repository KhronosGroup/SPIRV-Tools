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

#include "source/fuzz/fuzzer_pass_permute_function_variables.h"
#include <numeric>
#include<vector>
#include<algorithm>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

FuzzerPassPermuteFunctionVariables::FuzzerPassPermuteFunctionVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {} // Here we call parent constructor


void FuzzerPassPermuteFunctionVariables::Apply() {
  // here I want to loop over all functions and for each one,
  // Then deterimne number of OpVariable and swap them.
for (auto& Func : *GetIRContext()->module()) {
    uint32_t FunctionId = Func.result_id();

    // Entry point mean something like e.g. main(), so skip it.
    // Need to be consider a "main()" function also, so commented down section.
    // if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(), function_id)) {
    //   continue;
    // }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfPermuteFunctionVariables())) {
      continue;
    }
    // IDs storage section
    // entry() return unique_ptr of "BasicBlock"
    auto FBlock = Func->entry().get();

    // iterate over block
    std::std::vector<Instruction*> VarsPtr;
    std::vector<uint32_t> VarsIDs;
    for(auto BlockItrator = FBlock->begin();BlockItrator!=FBlock->end();BlockItrator++)
    {
      if(BlockItrator->opcode() == SpvOpVariable)
      {

        Instruction* Instuction = &(*BlockItrator);
        VarsPtr.push_back(Instuction)
        VarsIDs.push_back(Instuction->result_id());
      }
      else{
        continue;
      }
    }

    uint32_t VarsSize = VarsIDs.size()

    // permutation section
    std::vector<uint32_t> permutation(VarsSize); // 8 0 -> 7
    // Below function fill from ZEROs -->> Vector len-1
    std::iota(permutation.begin(), permutation.end(),0);
    GetFuzzerContext()->Shuffle(&permutation);

    std::vector<std::pair<uint32_t,uint32_t>> Pair_Ids;
    /* This concept using in DES cipher
    // (a1,a2,...,as)=(as,as−1)∘(as,as−2)∘...∘(as,a2)∘(as,a1)
    // Above formula is cycle, Apply product of transpositions
    // Every Permutation can be written as a product of transpositions and transpositions
    // Is special case of Permutation but between two numbers
    // start
    */
    for(uint32_t _Lindex= VarsSize-2; _Lindex > 0 ;_Lindex--)
      Pair_Ids.push_back(std::make_pair(VarsSize-1,_Lindex))
    // end

    //  Apply Transformation
    for(std::pair<uint32_t,uint32_t> Pair_Id:Pair_Ids){
      ApplyTransformation(TransformationSwapFunctionVariables(
        Pair_Id,
        function_id,
        /* GetFreshId() give me new id not used, to use it current process */
        GetFuzzerContext()->GetFreshId()));
    }

}
}

}  // namespace fuzz
}  // namespace spvtools
