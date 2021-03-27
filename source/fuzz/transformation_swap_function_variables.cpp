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

#include "source/fuzz/transformation_swap_function_variables.h"
#include "source/fuzz/fuzzer_util.h"


namespace spvtools {
namespace fuzz {

TransformationSwapFunctionVariables::
    TransformationSwapFunctionVariables(
        const spvtools::fuzz::protobufs::
            TransformationSwapFunctionVariables& message)
    : message_(message) {}

TransformationSwapFunctionVariables::TransformationSwapFunctionVariables(
                    std::pair<uint32_t,uint32_t>Pair_Id,
                    uint32_t function_id,
                    uint32_t fresh_id){
    message_.set_function_id(funcion_id);
    message_.set_fresh_id(fresh_id);
    protobufs::UInt32Pair pair;
    pair.set_first(Pair_Id.first);
    pair.set_second(Pair_Id.second);
    message_.set_pair(pair);
}

bool TransformationSwapFunctionVariables::IsApplicable(opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const{
          // Check that function exists
    const auto* function =
      fuzzerutil::FindFunction(ir_context, message_.function_id());
    // Here we check if funciton exists and check for it's entry point 
    //  FunctionIsEntryPoint -> Returns |true| if one of entry points has function id |function_id|
    if (!function || fuzzerutil::FunctionIsEntryPoint(ir_context, function->result_id())) {
    return false;
    }
    protobufs::UInt32Pair IdPair = message_.get_pair();
    auto _1stBlock = function->entry().get();
    bool _1stID = false, _2ndID = false
    for(auto BlockItrator = _1stBlock->begin();BlockItrator!=_1stBlock->end();BlockItrator++)
    {
        Instruction* Instuction = &(*BlockItrator);
        uint32_t _id_ = Instuction->result_id();
        _1stID = (_id_ == IdPair.first)? true:false;
        _2ndID = (_id_ == IdPair.second)? true:false;
        if(_1stID && _2ndID)
            return true;
    }
    return false;
}


void TransformationSwapFunctionVariables::Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const {
    // Check exists of functions
     auto* function = fuzzerutil::FindFunction(ir_context, message_.function_id());
    assert(function && "function doesn't exists");
    protobufs::UInt32Pair IdPair = message_.get_pair();

    auto FBlock = function->entry().get();
    for(auto BlockItrator = FBlock->begin();
    uint32_t _1stIndex=-1,_2ndIndex=-1;
    uint32_t i = 0;
    BlockItrator!=FBlock->end() && BlockItrator->opcode() == SpvOpVariable
    ;BlockItrator++)
    {
      Instruction* Instuction = &(*BlockItrator);
      auto _id_ = Instuction->result_id()
      _1stIndex = (_id_ == IdPair.first)? i:-1;
      _2ndIndex = (_id_ == IdPair.second)? i:-1;
      i++;
    }

    std::swap(FBlock[_1stIndex],FBlock[_2ndIndex]);
}


protobufs::Transformation TransformationSwapFunctionVariables::ToMessage() const {

protobufs::Transformation result;
  *result.transformation_swap_function_variables() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapFunctionVariables::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}


}  // namespace fuzz
}  // namespace spvtools
