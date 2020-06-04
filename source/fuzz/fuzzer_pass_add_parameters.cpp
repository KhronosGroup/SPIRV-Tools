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

#include "source/fuzz/fuzzer_pass_add_parameters.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_parameters.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddParameters::FuzzerPassAddParameters(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddParameters::~FuzzerPassAddParameters() = default;

void FuzzerPassAddParameters::Apply() {
  const auto& type_candidates = ComputeTypeCandidates();
  const uint32_t kMaxNumOfParameters = 5;

  for (const auto& function : *GetIRContext()->module()) {
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(),
                                         function.result_id())) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAddingParameters())) {
      continue;
    }

    const auto* type_inst =
        fuzzerutil::GetFunctionType(GetIRContext(), &function);
    assert(type_inst);

    // -1 because we don't take return type into account.
    auto num_old_parameters = type_inst->NumInOperands() - 1;
    auto num_new_parameters =
        GetFuzzerContext()->ChooseBetweenMinAndMax({1, kMaxNumOfParameters});

    std::vector<uint32_t> all_types(num_old_parameters), new_types,
        parameter_ids, constant_ids;

    // Get type ids for old parameters.
    std::iota(all_types.begin(), all_types.end(), 1);
    std::transform(all_types.begin(), all_types.end(), all_types.begin(),
                   [type_inst](uint32_t index) {
                     return type_inst->GetSingleWordInOperand(index);
                   });

    // Get type ids for new parameters...
    std::generate_n(std::back_inserter(new_types), num_new_parameters,
                    [this, &type_candidates] {
                      return type_candidates[GetFuzzerContext()->RandomIndex(
                          type_candidates)]();
                    });

    // ...append them to the old ones.
    all_types.insert(all_types.end(), new_types.begin(), new_types.end());

    // Create constants to initialize new parameters from.
    std::transform(
        new_types.begin(), new_types.end(), std::back_inserter(constant_ids),
        [this](uint32_t type_id) { return FindOrCreateZeroConstant(type_id); });

    // Generate result ids for new parameters.
    std::generate_n(std::back_inserter(parameter_ids), num_new_parameters,
                    [this] { return GetFuzzerContext()->GetFreshId(); });

    auto result_type_id = type_inst->GetSingleWordInOperand(0);
    ApplyTransformation(TransformationAddParameters(
        function.result_id(),
        FindOrCreateFunctionType(result_type_id, all_types),
        std::move(new_types), std::move(parameter_ids),
        std::move(constant_ids)));
  }
}

std::vector<std::function<uint32_t()>>
FuzzerPassAddParameters::ComputeTypeCandidates() {
  using opt::analysis::Bool;
  using opt::analysis::Float;
  using opt::analysis::Integer;

  // These providers will be used if there are no types in the module.
  std::unordered_map<size_t, std::function<uint32_t()>> candidates = {
      {Bool().HashValue(), [this] { return FindOrCreateBoolType(); }},
      {Integer(32, true).HashValue(),
       [this] { return FindOrCreateIntegerType(32, true); }},
      {Integer(32, false).HashValue(),
       [this] { return FindOrCreateIntegerType(32, false); }},
      {Float(32).HashValue(), [this] { return FindOrCreateFloatType(32); }}};

  for (const auto* type_inst : GetIRContext()->module()->GetTypes()) {
    switch (type_inst->opcode()) {
      case SpvOpTypeBool:
      case SpvOpTypeInt:
      case SpvOpTypeFloat:
      case SpvOpTypeArray:
      case SpvOpTypeMatrix:
      case SpvOpTypeVector:
      case SpvOpTypeStruct: {
        auto result_id = type_inst->result_id();
        const auto* type = GetIRContext()->get_type_mgr()->GetType(result_id);
        assert(type);

        candidates[type->HashValue()] = [result_id] { return result_id; };
      } break;
      default:
          // Ignore other types.
          ;
    }
  }

  std::vector<std::function<uint32_t()>> result;
  result.reserve(candidates.size());

  for (auto& item : candidates) {
    result.emplace_back(std::move(item.second));
  }

  assert(!result.empty());
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
