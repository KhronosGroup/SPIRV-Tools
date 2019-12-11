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

#include "source/fuzz/transformation_add_type_function.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypeFunction::TransformationAddTypeFunction(
    const spvtools::fuzz::protobufs::TransformationAddTypeFunction& message)
    : message_(message) {}

TransformationAddTypeFunction::TransformationAddTypeFunction(
    uint32_t fresh_id, uint32_t return_type_id,
    const std::vector<uint32_t>& argument_type_ids) {
  message_.set_fresh_id(fresh_id);
  message_.set_return_type_id(return_type_id);
  for (auto id : argument_type_ids) {
    message_.add_argument_type_id(id);
  }
}

bool TransformationAddTypeFunction::IsApplicable(
    opt::IRContext* context,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    return false;
  }
  if (!fuzzerutil::IsNonFunctionTypeId(context, message_.return_type_id())) {
    return false;
  }
  for (auto argument_type_id : message_.argument_type_id()) {
    if (!fuzzerutil::IsNonFunctionTypeId(context, argument_type_id)) {
      return false;
    }
  }
  return true;
}

void TransformationAddTypeFunction::Apply(
    opt::IRContext* context, spvtools::fuzz::FactManager* /*unused*/) const {
  opt::Instruction::OperandList in_operands;
  in_operands.push_back({SPV_OPERAND_TYPE_ID, {message_.return_type_id()}});
  for (auto argument_type_id : message_.argument_type_id()) {
    in_operands.push_back({SPV_OPERAND_TYPE_ID, {argument_type_id}});
  }
  context->module()->AddType(MakeUnique<opt::Instruction>(
      context, SpvOpTypeFunction, 0, message_.fresh_id(), in_operands));
  fuzzerutil::UpdateModuleIdBound(context, message_.fresh_id());
  // We have added an instruction to the module, so need to be careful about the
  // validity of existing analyses.
  context->InvalidateAnalysesExceptFor(opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddTypeFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_function() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
