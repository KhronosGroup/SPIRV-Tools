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

#include "source/fuzz/transformation_swap_function_variables.h"

namespace spvtools {
namespace fuzz {

TransformationSwapFunctionVariables::
    TransformationSwapFunctionVariables(
        const spvtools::fuzz::protobufs::
            TransformationSwapFunctionVariables& message)
    : message_(message) {}

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(uint32_t var_id1,
                                    uint32_t var_id2){


}

void TransformationSwapFunctionVariables::Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const {

}

bool TransformationSwapFunctionVariables::IsApplicable(opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const{


}   

protobufs::Transformation TransformationSwapFunctionVariables::ToMessage() const {

}     

std::unordered_set<uint32_t> TransformationSwapFunctionVariables::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}


}  // namespace fuzz
}  // namespace spvtools