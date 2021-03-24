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
#include "source/fuzz/fuzzer_util.h"


namespace spvtools {
namespace fuzz {

TransformationSwapFunctionVariables::
    TransformationSwapFunctionVariables(
        const spvtools::fuzz::protobufs::
            TransformationSwapFunctionVariables& message)
    : message_(message) {}

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(uint32_t var_id1,
                                    uint32_t var_id2,uint32_t function_id){
        this->var_id1 = var_id1;
        this->var_id2 = var_id2;
        this->function_id = function_id

}

bool TransformationSwapFunctionVariables::IsApplicable(opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const{

     // return vector of OpFunctionParameter
    bool _1stFlag = false ,_2ndFlag = false;
    auto functionVars = fuzzerutil::GetParameters(ir_context,this->funcion_id);
    for(opt::Instruction* var : functionVars)
    {
        auto id = var->result_id();
        if(id  == this->var_id1)
            _1stFlag = true;
        else if(id == this->var_id2)
            _2ndFlag = true;
        if(_1stFlag&&_2ndFlag)
            return true;
    }
    // mutable var
    this->_2varsExists  = true
    return false;
}


void TransformationSwapFunctionVariables::Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const {

    if(this->_2varsExists)
    {
        auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
        function->ForEachParam(
            [this->var_id1,this->var_id2](const opt::Instruction* param){
                 if(param->result_id()==this->var_id1)
                 {
                    //  here need to swap 2vars
                 }
            }
        );
    }

    assert("one of the variables doesn't exists");

}


protobufs::Transformation TransformationSwapFunctionVariables::ToMessage() const {

protobufs::Transformation result;
// there is line here !!
  *result.transformation_swap_function_variables() = message_; // !!
  return result;

}

std::unordered_set<uint32_t> TransformationSwapFunctionVariables::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}


}  // namespace fuzz
}  // namespace spvtools
